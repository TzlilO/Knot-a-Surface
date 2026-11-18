"""Cubic tensor B-spline charts in numpy: bases with first derivatives, Boehm knot insertion, sparse design matrices.

The same mathematics as talk/nurbs.js (Cox-de Boor, clamped knots, Boehm's insertion), written for many points at once.
A direction of a chart is either OPEN (clamped knot vector, possibly non-uniform after insertion) or PERIODIC (uniform,
n control points wrapping round; refined by doubling, which is exact because the doubled space contains the old one).
Nets are stored (nv, nu, d): row j (v), column i (u), control point j * nu + i, as in talk/dtu-data.js."""
import numpy as np
from scipy import sparse

DEG = 3


def clamped(n):
    """Uniform clamped cubic knots for n control points (= clampedKnots(n, 3) in talk/nurbs.js, knots() in fit.py)."""
    return np.r_[np.zeros(4), np.arange(1, n - 3) / (n - 3), np.ones(4)]


class Open:
    """A clamped cubic direction with knot vector U (len n + 4)."""
    periodic = False

    def __init__(self, U):
        self.U = np.asarray(U, float); self.n = len(self.U) - 4

    @classmethod
    def uniform(cls, n): return cls(clamped(n))

    def span(self, t):
        return np.clip(np.searchsorted(self.U, t, side='right') - 1, 3, self.n - 1)

    def basis(self, t, deriv=False):
        """Indices (m, 4) of the four non-zero basis functions at t and their values (and first derivatives)."""
        U = self.U; t = np.clip(np.asarray(t, float), 0.0, 1.0); k = self.span(t); m = len(t)
        N = np.zeros((m, 4)); N[:, 0] = 1.0; left = np.zeros((m, 4)); right = np.zeros((m, 4)); low = None
        for j in range(1, 4):
            left[:, j] = t - U[k + 1 - j]; right[:, j] = U[k + j] - t
            if j == 3: low = N[:, :3].copy()                    # degree-2 values of functions k-2, k-1, k
            saved = np.zeros(m)
            for r in range(j):
                den = right[:, r + 1] + left[:, j - r]; tmp = N[:, r] / np.where(den == 0, 1, den)
                N[:, r] = saved + right[:, r + 1] * tmp; saved = left[:, j - r] * tmp
            N[:, j] = saved
        idx = k[:, None] - 3 + np.arange(4)
        if not deriv: return idx, N
        # N'_{i,3} = 3 (N_{i,2} / (U[i+3] - U[i]) - N_{i+1,2} / (U[i+4] - U[i+1])), i = k-3 .. k
        L2 = np.zeros((m, 5)); L2[:, 1:4] = low                 # degree-2 values of functions k-3 .. k+1 (outer two are 0)
        dN = np.zeros((m, 4))
        for a in range(4):
            i = k - 3 + a; d1 = U[i + 3] - U[i]; d2 = U[i + 4] - U[i + 1]
            dN[:, a] = 3 * (L2[:, a] / np.where(d1 == 0, 1, d1) * (d1 != 0) - L2[:, a + 1] / np.where(d2 == 0, 1, d2) * (d2 != 0))
        return idx, N, dN

    def sites(self):          # Greville abscissae: where control point i "sits"
        U = self.U; n = self.n; return (U[1:n + 1] + U[2:n + 2] + U[3:n + 3]) / 3

    def spans(self):          # the non-empty knot intervals
        u = np.unique(self.U); return np.c_[u[:-1], u[1:]]

    def insert(self, t, net, axis):
        """Boehm's insertion of knot t; net's `axis` runs over this direction's control points. Returns (new direction,
        new net); the surface does not change."""
        U, n = self.U, self.n; k = int(self.span(np.array([t]))[0])
        P = np.moveaxis(net, axis, 0); Q = np.empty((n + 1,) + P.shape[1:])
        Q[:k - 2] = P[:k - 2]; Q[k + 1:] = P[k:]
        for i in range(k - 2, k + 1):
            a = (t - U[i]) / (U[i + 3] - U[i]); Q[i] = (1 - a) * P[i - 1] + a * P[i]
        return Open(np.insert(U, k + 1, t)), np.moveaxis(Q, 0, axis)


class Periodic:
    """A uniform cubic direction closed round: n control points, control point j sits at t = j / n."""
    periodic = True

    def __init__(self, n): self.n = int(n)

    def basis(self, t, deriv=False):
        s = np.mod(np.asarray(t, float), 1.0) * self.n; i = np.floor(s).astype(int); f = s - i
        idx = np.mod(i[:, None] + np.arange(-1, 3), self.n)
        N = np.stack([(1 - f) ** 3, 3 * f ** 3 - 6 * f ** 2 + 4, -3 * f ** 3 + 3 * f ** 2 + 3 * f + 1, f ** 3], 1) / 6
        if not deriv: return idx, N
        dN = np.stack([-3 * (1 - f) ** 2, 9 * f ** 2 - 12 * f, -9 * f ** 2 + 6 * f + 3, 3 * f ** 2], 1) / 6 * self.n
        return idx, N, dN

    def sites(self): return np.arange(self.n) / self.n

    def spans(self): return np.c_[np.arange(self.n), np.arange(1, self.n + 1)] / self.n


def evaluate(au, av, net, u, v, deriv=False):
    """Points (m, d) of the chart at parameters u, v (and S_u, S_v)."""
    r = au.basis(u, deriv); s = av.basis(v, deriv)
    iu, Nu = r[0], r[1]; iv, Nv = s[0], s[1]; nu = au.n; flat = net.reshape(-1, net.shape[-1])
    cols = iv[:, :, None] * nu + iu[:, None, :]                       # (m, 4, 4)
    C = flat[cols]                                                    # (m, 4, 4, d)
    S = np.einsum('ma,mb,mabd->md', Nv, Nu, C)
    if not deriv: return S
    Su = np.einsum('ma,mb,mabd->md', Nv, r[2], C); Sv = np.einsum('ma,mb,mabd->md', s[2], Nu, C)
    return S, Su, Sv


def grid(au, av, net, nu_s, nv_s, deriv=False):
    """The chart on a regular grid of nv_s x nu_s parameters (periodic directions without the repeated end)."""
    us = np.arange(nu_s) / nu_s if au.periodic else np.linspace(0, 1, nu_s)
    vs = np.arange(nv_s) / nv_s if av.periodic else np.linspace(0, 1, nv_s)
    U, V = np.meshgrid(us, vs)
    out = evaluate(au, av, net, U.ravel(), V.ravel(), deriv)
    if not deriv: return out.reshape(nv_s, nu_s, -1), U, V
    return tuple(o.reshape(nv_s, nu_s, -1) for o in out), U, V


def design(au, av, u, v):
    """Sparse (m, nv * nu) design matrix: row p holds the 16 basis products of point p."""
    iu, Nu = au.basis(u); iv, Nv = av.basis(v); m = len(u)
    cols = (iv[:, :, None] * au.n + iu[:, None, :]).reshape(m, 16); vals = (Nv[:, :, None] * Nu[:, None, :]).reshape(m, 16)
    return sparse.csr_matrix((vals.ravel(), cols.ravel(), np.arange(0, 16 * m + 1, 16)), shape=(m, au.n * av.n))


def second_differences(au, av):
    """Second differences of the net along u and along v (wrapping round in a periodic direction)."""
    def d2(ax):
        n = ax.n
        if ax.periodic:
            I = sparse.identity(n, format='csr'); return (sparse.eye(n, n, -1) - 2 * I + sparse.eye(n, n, 1) + sparse.eye(n, n, n - 1) + sparse.eye(n, n, -(n - 1))).tocsr()
        return sparse.diags([np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)], [0, 1, 2], shape=(n - 2, n)).tocsr()
    Iu, Iv = sparse.identity(au.n), sparse.identity(av.n)
    return sparse.vstack([sparse.kron(Iv, d2(au)), sparse.kron(d2(av), Iu)]).tocsr()


def double_periodic(ax, net, axis):
    """Refine a periodic direction to 2n control points. The doubled uniform space holds the old surface exactly
    (all spans split at their middles = knot insertion at every span); the new net is that exact representation."""
    n = ax.n; P = np.moveaxis(net, axis, 0); Q = np.empty((2 * n,) + P.shape[1:])
    # uniform cubic subdivision masks (Lane-Riesenfeld): vertex points (1, 6, 1) / 8, edge points (4, 4) / 8
    Q[0::2] = (np.roll(P, 1, 0) + 6 * P + np.roll(P, -1, 0)) / 8; Q[1::2] = (P + np.roll(P, -1, 0)) / 2
    return Periodic(2 * n), np.moveaxis(Q, 0, axis)
