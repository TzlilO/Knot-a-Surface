#!/usr/bin/env python3
"""DTU Chamfer distance of the repository's OFFLINE FITS (the owl, scan 122, and the rabbit, scan 55), in one go.

    python3 -I tools/eval/eval_fits.py --dtu DATA --ref REF -o OUT

DATA: the scans as tools/dtu/fit.py reads them (only DATA/scan<N>/cameras.npz is used here). REF: the reference of
the DTU evaluation (see dtu_chamfer.py). For each scan it writes the charts' lattices as meshes in the DTU world frame
(chart_mesh.py: the frame check must pass), evaluates each with dtu_chamfer.py, and writes OUT/summary.json and
OUT/summary.md. Without the reference cloud of a scan it runs the protocol up to the observation mask
(`--mask-only`) and says which file is missing; nothing else is put in its place.

These charts were fitted offline to depth maps that were themselves rendered from the dataset's structured-light
reference. The numbers say how well a spline chart holds that surface. They are not trained results, not results
of the thesis method, and not comparable with the rows of published tables: summary.md says so above the table.
"""
import argparse, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__)); TALK = os.path.join(HERE, '..', '..', 'talk')
FILES = {122: os.path.join(TALK, 'dtu-data.js'), 55: os.path.join(TALK, 'dtu', 'scan55.js')}
# for orientation only: what the talk prints per scan (DTU in talk/deck.js: Table 1 of the paper; Chamfer, mm)
PRINTED = {55: dict(pgsr=0.28, ours=0.329), 122: dict(pgsr=0.31, ours=0.341)}
HONEST = ('OFFLINE FITS, NOT TRAINED. The charts were fitted by tools/dtu/fit.py to the depth maps of the Hugging Face copy Y1ue/DTU, '
          'which are MVSNet\'s `Depths_raw`: depth rendered from a surface reconstructed from the SAME structured-light reference these numbers are '
          'measured against. They say how well a spline chart of that size holds the reference surface, not how well anything was reconstructed '
          'from photographs, and they are not comparable with the rows of published DTU tables.')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--dtu', required=True, help='folder with scan<N>/cameras.npz (the layout of tools/dtu/fit.py\'s data dir)')
    ap.add_argument('--ref', required=True, help='folder with the evaluation reference (ObsMask, Plane, stl clouds)')
    ap.add_argument('-o', '--out', required=True)
    ap.add_argument('--scans', type=int, nargs='*', default=[122, 55])
    ap.add_argument('--budget', type=int, nargs='*', default=[160000, 30000], help='talk lattices to evaluate (default: 160000 and 30000, the talk\'s defaults on a desktop and on a small screen)')
    ap.add_argument('--refine', type=int, default=2, help='also evaluate the full lattice with K x K vertices per control point (default 2; 1 = skip)')
    ap.add_argument('--calibration', help='optional: Calibration/cal18 of the dataset, for chart_mesh.py\'s check')
    ap.add_argument('--max-minutes', type=float, default=30.0, help='wall-clock budget of EACH evaluation (default 30)')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); py = [sys.executable, '-I']; rows = []; t0 = time.time(); missing = []
    for scan in a.scans:
        cmd = py + [os.path.join(HERE, 'chart_mesh.py'), FILES[scan], '--cameras', os.path.join(a.dtu, 'scan%d' % scan, 'cameras.npz'), '-o', a.out, '--budget'] + [str(b) for b in a.budget] + ['--refine', str(a.refine)]
        if a.calibration: cmd += ['--calibration', a.calibration]
        if subprocess.run(cmd).returncode: sys.exit('chart_mesh.py failed for scan %d: nothing was evaluated' % scan)
        frame = json.load(open(os.path.join(a.out, 'scan%d_frame.json' % scan)))
        for m in frame['meshes']:
            mesh = os.path.join(a.out, m['file']); out = os.path.splitext(mesh)[0] + '.dtu_chamfer.json'
            cmd = py + [os.path.join(HERE, 'dtu_chamfer.py'), mesh, '--scan', str(scan), '--data', a.ref, '--out', out, '--seed', str(a.seed), '--max-minutes', str(a.max_minutes),
                        '--note', 'offline fit (tools/dtu/fit.py), not trained: ' + m['what']]
            rc = subprocess.run(cmd, stdout=subprocess.DEVNULL).returncode; R = json.load(open(out))
            if R['stop']['reason'] == 'missing_reference':
                if R['stop']['detail'] not in missing: missing.append(R['stop']['detail'])
                subprocess.run(cmd + ['--mask-only'], stdout=subprocess.DEVNULL); R = json.load(open(out))
            r = R.get('result') or {}; c = R.get('counts', {})
            rows.append(dict(scan=scan, mesh=m['file'], lattice=m['lattice'], table='all' in m['file'].rsplit('_', 1)[-1], charts=m['charts'], vertices=m['vertices'], triangles=m['triangles'],
                             accuracy_mm=r.get('accuracy_mm'), completeness_mm=r.get('completeness_mm'), overall_mm=r.get('overall_mm'),
                             points_input=c.get('data_in_mask'), points_reference=(r.get('completeness') or {}).get('of'), seconds=R['seconds'], stop=R['stop']['reason']))
            print('scan %d  %-34s %s' % (scan, m['file'], 'acc %.4f  comp %.4f  overall %.4f mm' % (r['accuracy_mm'], r['completeness_mm'], r['overall_mm']) if r else 'no result: ' + R['stop']['reason']), flush=True)
    S = dict(what='DTU Chamfer distance of the repository\'s offline fits', honest=HONEST, printed_for_orientation=dict(source='talk/deck.js (Table 1 of the paper), Chamfer in mm; a different thing from an offline fit', scans=PRINTED),
             missing=missing, rows=rows, seconds=round(time.time() - t0, 1))
    with open(os.path.join(a.out, 'summary.json'), 'w') as f: json.dump(S, f, indent=1)
    f3 = lambda x: '-' if x is None else '%.3f' % x
    md = ['# DTU Chamfer distance of the offline fits', '', HONEST, '']
    if missing: md += ['**No distances were measured for a scan whose reference cloud is missing.** ' + ' '.join(missing), '']
    md += ['| scan | lattice | charts | vertices | accuracy, mm | completeness, mm | overall, mm | input points in the mask | reference points above the plane |', '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        md.append('| %d | %s | %s | %s | %s | %s | %s | %s | %s |' % (r['scan'], r['lattice'], ' + '.join('%s %d x %d' % (c['name'], c['na'], c['nb']) for c in r['charts']), format(r['vertices'], ','),
                                                           f3(r['accuracy_mm']), f3(r['completeness_mm']), f3(r['overall_mm']), '-' if r['points_input'] is None else format(r['points_input'], ','), '-' if r['points_reference'] is None else format(r['points_reference'], ',')))
    md += ['', 'For orientation only (a different thing: methods that reconstruct from photographs), the talk prints for these scans, from Table 1 of the paper: ' +
           '; '.join('scan %d: PGSR %.2f, ours %.3f' % (s, PRINTED[s]['pgsr'], PRINTED[s]['ours']) for s in a.scans if s in PRINTED) + ' (Chamfer, mm).', '']
    with open(os.path.join(a.out, 'summary.md'), 'w') as f: f.write('\n'.join(md))
    print('\n'.join(md)); sys.exit(2 if missing or any(r['stop'] not in ('completed',) for r in rows) else 0)


if __name__ == '__main__':
    main()
