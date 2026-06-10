// Fused local-support B-spline surface evaluation.
//
// Each (u_i, v_j) sample depends on a (p+1)x(p+1) window of the control
// grid. One thread per sample reads that window ONCE and accumulates all
// five basis/derivative pair contractions:
//      out0 = sum  bu  * bv  * G     (S numerator / W)
//      out1 = sum dbu  * bv  * G     (d/du)
//      out2 = sum  bu  * dbv * G     (d/dv)
//      out3 = sum dbuu * bv  * G     (d2/du2)
//      out4 = sum  bu  * dbvv* G     (d2/dv2)
// The rational quotient rule is applied OUTSIDE (PyTorch, elementwise) so
// gradients of the division come from autograd, exactly.
//
// Backward computes exact gradients of these multilinear sums w.r.t. the
// control grid (atomicAdd scatter into the 4x4 window) and w.r.t. the
// compact per-sample basis values (so knot optimization — which produces
// the basis values differentiably in Python — keeps its gradient path).

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define ORD 4  // degree 3 => 4 basis values per sample (compile-time)

template <typename scalar_t>
__global__ void bspline_tp_forward_kernel(
    const scalar_t* __restrict__ grid,    // [H, W, C]
    const scalar_t* __restrict__ bu,      // [Mu, ORD]  basis values
    const scalar_t* __restrict__ dbu,     // [Mu, ORD]  1st derivs
    const scalar_t* __restrict__ d2bu,    // [Mu, ORD]  2nd derivs
    const scalar_t* __restrict__ bv,      // [Mv, ORD]
    const scalar_t* __restrict__ dbv,     // [Mv, ORD]
    const scalar_t* __restrict__ d2bv,    // [Mv, ORD]
    const int64_t* __restrict__ span_u,   // [Mu] first ctrl index of window
    const int64_t* __restrict__ span_v,   // [Mv]
    scalar_t* __restrict__ out,           // [5, Mu, Mv, C]
    const int Mu, const int Mv, const int H, const int W, const int C)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = Mu * Mv;
    if (idx >= total) return;

    const int i = idx / Mv;   // u sample
    const int j = idx % Mv;   // v sample

    const int su = (int)span_u[i];
    const int sv = (int)span_v[j];

    // Registers for the 4 basis values of this sample in each direction
    scalar_t bu_r[ORD], dbu_r[ORD], d2bu_r[ORD];
    scalar_t bv_r[ORD], dbv_r[ORD], d2bv_r[ORD];
    #pragma unroll
    for (int k = 0; k < ORD; ++k) {
        bu_r[k]   = bu[i * ORD + k];
        dbu_r[k]  = dbu[i * ORD + k];
        d2bu_r[k] = d2bu[i * ORD + k];
        bv_r[k]   = bv[j * ORD + k];
        dbv_r[k]  = dbv[j * ORD + k];
        d2bv_r[k] = d2bv[j * ORD + k];
    }

    const int64_t out_stride = (int64_t)Mu * Mv * C;
    const int64_t base = ((int64_t)i * Mv + j) * C;

    for (int c = 0; c < C; ++c) {
        scalar_t s = 0, s_u = 0, s_v = 0, s_uu = 0, s_vv = 0;
        #pragma unroll
        for (int ku = 0; ku < ORD; ++ku) {
            const int gu = su + ku;
            scalar_t row = 0, row_v = 0, row_vv = 0;
            #pragma unroll
            for (int kv = 0; kv < ORD; ++kv) {
                const int gv = sv + kv;
                const scalar_t g = grid[((int64_t)gu * W + gv) * C + c];
                row    += bv_r[kv]   * g;
                row_v  += dbv_r[kv]  * g;
                row_vv += d2bv_r[kv] * g;
            }
            s    += bu_r[ku]   * row;
            s_u  += dbu_r[ku]  * row;
            s_v  += bu_r[ku]   * row_v;
            s_uu += d2bu_r[ku] * row;
            s_vv += bu_r[ku]   * row_vv;
        }
        out[0 * out_stride + base + c] = s;
        out[1 * out_stride + base + c] = s_u;
        out[2 * out_stride + base + c] = s_v;
        out[3 * out_stride + base + c] = s_uu;
        out[4 * out_stride + base + c] = s_vv;
    }
}

template <typename scalar_t>
__global__ void bspline_tp_backward_kernel(
    const scalar_t* __restrict__ grad_out, // [5, Mu, Mv, C]
    const scalar_t* __restrict__ grid,     // [H, W, C]
    const scalar_t* __restrict__ bu,
    const scalar_t* __restrict__ dbu,
    const scalar_t* __restrict__ d2bu,
    const scalar_t* __restrict__ bv,
    const scalar_t* __restrict__ dbv,
    const scalar_t* __restrict__ d2bv,
    const int64_t* __restrict__ span_u,
    const int64_t* __restrict__ span_v,
    scalar_t* __restrict__ d_grid,   // [H, W, C]
    scalar_t* __restrict__ d_bu,     // [Mu, ORD]
    scalar_t* __restrict__ d_dbu,    // [Mu, ORD]
    scalar_t* __restrict__ d_d2bu,   // [Mu, ORD]
    scalar_t* __restrict__ d_bv,     // [Mv, ORD]
    scalar_t* __restrict__ d_dbv,    // [Mv, ORD]
    scalar_t* __restrict__ d_d2bv,   // [Mv, ORD]
    const int Mu, const int Mv, const int H, const int W, const int C)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = Mu * Mv;
    if (idx >= total) return;

    const int i = idx / Mv;
    const int j = idx % Mv;
    const int su = (int)span_u[i];
    const int sv = (int)span_v[j];

    scalar_t bu_r[ORD], dbu_r[ORD], d2bu_r[ORD];
    scalar_t bv_r[ORD], dbv_r[ORD], d2bv_r[ORD];
    #pragma unroll
    for (int k = 0; k < ORD; ++k) {
        bu_r[k]   = bu[i * ORD + k];
        dbu_r[k]  = dbu[i * ORD + k];
        d2bu_r[k] = d2bu[i * ORD + k];
        bv_r[k]   = bv[j * ORD + k];
        dbv_r[k]  = dbv[j * ORD + k];
        d2bv_r[k] = d2bv[j * ORD + k];
    }

    // Per-sample accumulators for basis-value grads (reduced over kv/ku & c)
    scalar_t acc_bu[ORD] = {0}, acc_dbu[ORD] = {0}, acc_d2bu[ORD] = {0};
    scalar_t acc_bv[ORD] = {0}, acc_dbv[ORD] = {0}, acc_d2bv[ORD] = {0};

    const int64_t out_stride = (int64_t)Mu * Mv * C;
    const int64_t base = ((int64_t)i * Mv + j) * C;

    for (int c = 0; c < C; ++c) {
        const scalar_t go0 = grad_out[0 * out_stride + base + c];
        const scalar_t go1 = grad_out[1 * out_stride + base + c];
        const scalar_t go2 = grad_out[2 * out_stride + base + c];
        const scalar_t go3 = grad_out[3 * out_stride + base + c];
        const scalar_t go4 = grad_out[4 * out_stride + base + c];

        #pragma unroll
        for (int ku = 0; ku < ORD; ++ku) {
            const int gu = su + ku;
            // weight of grid element (gu, gv, c) in each of the 5 outputs:
            //   w0 = bu[ku]*bv[kv], w1 = dbu[ku]*bv[kv], w2 = bu[ku]*dbv[kv],
            //   w3 = d2bu[ku]*bv[kv], w4 = bu[ku]*d2bv[kv]
            const scalar_t cu0 = bu_r[ku], cu1 = dbu_r[ku], cu3 = d2bu_r[ku];
            scalar_t row = 0, row_v = 0, row_vv = 0;   // for basis grads
            #pragma unroll
            for (int kv = 0; kv < ORD; ++kv) {
                const int gv = sv + kv;
                const int64_t gidx = ((int64_t)gu * W + gv) * C + c;
                const scalar_t g = grid[gidx];

                const scalar_t cv0 = bv_r[kv], cv2 = dbv_r[kv], cv4 = d2bv_r[kv];

                // d(out)/d(grid)
                const scalar_t dg =
                      go0 * cu0 * cv0
                    + go1 * cu1 * cv0
                    + go2 * cu0 * cv2
                    + go3 * cu3 * cv0
                    + go4 * cu0 * cv4;
                atomicAdd(&d_grid[gidx], dg);

                // accumulate for u-basis grads (sum over kv first)
                row    += cv0 * g;
                row_v  += cv2 * g;
                row_vv += cv4 * g;

                // v-basis grads: d(out)/d(bv[kv]) etc.
                acc_bv[kv]   += (go0 * cu0 + go1 * cu1 + go3 * cu3) * g;
                acc_dbv[kv]  += go2 * cu0 * g;
                acc_d2bv[kv] += go4 * cu0 * g;
            }
            acc_bu[ku]   += go0 * row + go2 * row_v + go4 * row_vv;
            acc_dbu[ku]  += go1 * row;
            acc_d2bu[ku] += go3 * row;
        }
    }

    #pragma unroll
    for (int k = 0; k < ORD; ++k) {
        atomicAdd(&d_bu[i * ORD + k],   acc_bu[k]);
        atomicAdd(&d_dbu[i * ORD + k],  acc_dbu[k]);
        atomicAdd(&d_d2bu[i * ORD + k], acc_d2bu[k]);
        atomicAdd(&d_bv[j * ORD + k],   acc_bv[k]);
        atomicAdd(&d_dbv[j * ORD + k],  acc_dbv[k]);
        atomicAdd(&d_d2bv[j * ORD + k], acc_d2bv[k]);
    }
}

torch::Tensor bspline_tp_forward(
    torch::Tensor grid,
    torch::Tensor bu, torch::Tensor dbu, torch::Tensor d2bu,
    torch::Tensor bv, torch::Tensor dbv, torch::Tensor d2bv,
    torch::Tensor span_u, torch::Tensor span_v)
{
    TORCH_CHECK(grid.is_cuda() && grid.dim() == 3, "grid must be CUDA [H,W,C]");
    TORCH_CHECK(bu.size(1) == ORD, "compiled for degree 3 (4 basis values)");
    const int H = grid.size(0), W = grid.size(1), C = grid.size(2);
    const int Mu = bu.size(0), Mv = bv.size(0);

    auto out = torch::empty({5, Mu, Mv, C}, grid.options());
    const int threads = 256;
    const int blocks = (Mu * Mv + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(grid.scalar_type(), "bspline_tp_forward", ([&] {
        bspline_tp_forward_kernel<scalar_t><<<blocks, threads>>>(
            grid.contiguous().data_ptr<scalar_t>(),
            bu.contiguous().data_ptr<scalar_t>(),
            dbu.contiguous().data_ptr<scalar_t>(),
            d2bu.contiguous().data_ptr<scalar_t>(),
            bv.contiguous().data_ptr<scalar_t>(),
            dbv.contiguous().data_ptr<scalar_t>(),
            d2bv.contiguous().data_ptr<scalar_t>(),
            span_u.contiguous().data_ptr<int64_t>(),
            span_v.contiguous().data_ptr<int64_t>(),
            out.data_ptr<scalar_t>(), Mu, Mv, H, W, C);
    }));
    return out;
}

std::vector<torch::Tensor> bspline_tp_backward(
    torch::Tensor grad_out, torch::Tensor grid,
    torch::Tensor bu, torch::Tensor dbu, torch::Tensor d2bu,
    torch::Tensor bv, torch::Tensor dbv, torch::Tensor d2bv,
    torch::Tensor span_u, torch::Tensor span_v)
{
    const int H = grid.size(0), W = grid.size(1), C = grid.size(2);
    const int Mu = bu.size(0), Mv = bv.size(0);

    auto d_grid = torch::zeros_like(grid);
    auto d_bu   = torch::zeros_like(bu);
    auto d_dbu  = torch::zeros_like(dbu);
    auto d_d2bu = torch::zeros_like(d2bu);
    auto d_bv   = torch::zeros_like(bv);
    auto d_dbv  = torch::zeros_like(dbv);
    auto d_d2bv = torch::zeros_like(d2bv);

    const int threads = 256;
    const int blocks = (Mu * Mv + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(grid.scalar_type(), "bspline_tp_backward", ([&] {
        bspline_tp_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_out.contiguous().data_ptr<scalar_t>(),
            grid.contiguous().data_ptr<scalar_t>(),
            bu.contiguous().data_ptr<scalar_t>(),
            dbu.contiguous().data_ptr<scalar_t>(),
            d2bu.contiguous().data_ptr<scalar_t>(),
            bv.contiguous().data_ptr<scalar_t>(),
            dbv.contiguous().data_ptr<scalar_t>(),
            d2bv.contiguous().data_ptr<scalar_t>(),
            span_u.contiguous().data_ptr<int64_t>(),
            span_v.contiguous().data_ptr<int64_t>(),
            d_grid.data_ptr<scalar_t>(),
            d_bu.data_ptr<scalar_t>(), d_dbu.data_ptr<scalar_t>(),
            d_d2bu.data_ptr<scalar_t>(),
            d_bv.data_ptr<scalar_t>(), d_dbv.data_ptr<scalar_t>(),
            d_d2bv.data_ptr<scalar_t>(),
            Mu, Mv, H, W, C);
    }));
    return {d_grid, d_bu, d_dbu, d_d2bu, d_bv, d_dbv, d_d2bv};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &bspline_tp_forward, "Fused B-spline TP contraction forward");
    m.def("backward", &bspline_tp_backward, "Fused B-spline TP contraction backward");
}
