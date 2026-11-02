"""Build the cross-method results table AND the provenance manifest needed to reproduce it.

The benchmark's whole claim is that N methods were compared FAIRLY: same crops, same split, same evaluator,
each method using only its own model/transform/eval-feature. That claim is only checkable if every number
carries its inputs with it. This module does two jobs:

    report  -- emit the comparison table (markdown + csv) plus a JSON manifest recording, per result, the
               checkpoint (with sha256), feature dim, eval feature tag, pooling mode, crop counts, the split
               files (with sha256) and the environment that produced it.
    verify  -- re-derive the facts a reader would have to trust, and fail loudly on any mismatch:
               npz row counts vs the split kwcoco, feature dim vs what the npz claims, pid sets shared
               across roles, chip files present on disk, and one-track-per-video.

Neither reads the models. Everything comes from the .npz caches, the eval JSONs and the split files, so the
checks are independent of the extractors that produced them.

    python -m caliper_eval.report report --eval-root .../eval/combined_brc1_brc2/cross-collection
    python -m caliper_eval.report verify --eval-root .../eval/combined_brc1_brc2/cross-collection \
                                       --splits-dir .../splits/cross-collection
"""
import argparse
import glob
import hashlib
import json
import os
import os.path as osp
import sys

import numpy as np

try:
    from caliper_eval import features_io
except ImportError:  # tolerate being run from the repo without install
    sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
    from caliper_eval import features_io

_SHA_CACHE = {}


def sha256(path, chunk=1 << 22):
    """Content hash of a (possibly multi-GB) file. Cached: the same checkpoint is referenced by every role."""
    if path in _SHA_CACHE:
        return _SHA_CACHE[path]
    if not path or not osp.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(chunk), b''):
            h.update(blk)
    _SHA_CACHE[path] = h.hexdigest()
    return _SHA_CACHE[path]


def _s(v):
    """numpy 0-d array / bytes -> str."""
    v = v.item() if hasattr(v, 'item') else v
    return v.decode() if isinstance(v, bytes) else str(v)


def collect(eval_root):
    """One record per <method>/<source> dir under eval_root, merging eval JSON + npz provenance."""
    out = []
    for d in sorted(glob.glob(osp.join(eval_root, '*', '*'))):
        if not osp.isdir(d):
            continue
        evals = sorted(glob.glob(osp.join(d, 'eval_*.json')))
        npzs = sorted(glob.glob(osp.join(d, '*.npz')))
        if not npzs:
            continue
        rec = dict(method=osp.basename(osp.dirname(d)), source=osp.basename(d), dir=d,
                   roles={}, eval=None, eval_file=None)
        for p in npzs:
            # NEVER touch z['feats'] here: these caches are up to 16 GB each and decompressing them to read
            # a shape would mean reading ~110 GB to print a table. The contract already stores feat_dim as a
            # scalar, and the row count equals len(pid).
            try:
                z = np.load(p, allow_pickle=False)
            except Exception as e:
                # a job is very likely mid-write: np.savez_compressed leaves a truncated zip until it closes.
                # Report it and move on rather than aborting a status query over a live directory.
                rec['roles']['INCOMPLETE:' + osp.basename(p)] = dict(
                    npz=osp.basename(p), incomplete=True, error=type(e).__name__,
                    bytes=os.path.getsize(p))
                continue
            role = _s(z['split_role'])
            tok = _s(z['camera_token'])
            pid = z['pid']
            vid = z['video']
            rec['roles']['{}_{}'.format(role, tok)] = dict(
                npz=osp.basename(p), n_crops=int(pid.shape[0]), feat_dim=int(z['feat_dim']),
                n_videos=int(len(set(vid.tolist()))), n_pids=int(len(set(pid.tolist()))),
                pids=set(int(x) for x in pid.tolist()), bytes=os.path.getsize(p))
            # provenance is identical across roles; capture once
            rec.setdefault('weight', _s(z['weight']))
            rec.setdefault('neck_feat', _s(z['neck_feat']))
            rec.setdefault('pooling', _s(z['pooling']) if 'pooling' in z else '')
            rec.setdefault('crop_root', _s(z['crop_root']))
            rec.setdefault('split', _s(z['split']))
        if evals:
            rec['eval_file'] = evals[0]
            rec['eval'] = json.load(open(evals[0]))
        out.append(rec)
    return out


def headline(rec):
    """(pooling_used, combined_metrics, per_range_metrics) from the eval JSON, preferring video pooling."""
    if not rec['eval']:
        return None, None, {}
    res = rec['eval'].get('results', {})
    for pool in ('video', 'image'):
        if pool in res:
            r = res[pool]
            return pool, r.get('combined'), {k: v for k, v in r.items() if k != 'combined'}
    return None, None, {}


def cmd_report(args):
    recs = collect(args.eval_root)
    if not recs:
        raise SystemExit("no results under " + args.eval_root)

    rows = []
    for rec in recs:
        pool, comb, per = headline(rec)
        rows.append((rec, pool, comb, per))
    rows.sort(key=lambda t: -(t[2]['mAP'] if t[2] else -1))

    # The eval JSON's n_query is qF.shape[0] -- the PRE-filter count. eval_func silently drops queries whose
    # identity has no gallery counterpart and divides by num_valid_q, so on a split like cross-collection
    # (39 of 239 subjects mated) the reported 6789 is ~5.7x the real denominator of 1198. Recover the honest
    # number so nobody misreads the table.
    for rec, pool, comb, per in rows:
        gal = [r for k, r in rec['roles'].items() if k.startswith('gallery')]
        qry = [r for k, r in rec['roles'].items() if k.startswith('query')]
        if gal and qry:
            gp = set().union(*[r['pids'] for r in gal])
            rec['n_query_videos'] = sum(r['n_videos'] for r in qry)
            rec['mated_ids'] = len(gp & set().union(*[r['pids'] for r in qry]))

    ranges = sorted({k for _, _, _, per in rows for k in per})
    hdr = ['method/source', 'dim', 'pool', 'mAP', 'rank1', 'rank5', 'rank10'] + ['r1@' + r for r in ranges]
    lines_md = ['| ' + ' | '.join(hdr) + ' |', '|' + '|'.join(['---'] * len(hdr)) + '|']
    lines_csv = [','.join(hdr)]
    for rec, pool, comb, per in rows:
        dim = next(iter(rec['roles'].values()))['feat_dim'] if rec['roles'] else 0
        name = '{}/{}'.format(rec['method'], rec['source'])
        if comb is None:
            cells = [name, str(dim), pool or '-', 'NOT EVALUATED', '', '', ''] + [''] * len(ranges)
        else:
            cells = [name, str(dim), pool or '-',
                     '%.4f' % comb['mAP'], '%.4f' % comb['rank1'],
                     '%.4f' % comb['rank5'], '%.4f' % comb['rank10']]
            cells += ['%.4f' % per[r]['rank1'] if r in per else '' for r in ranges]
        lines_md.append('| ' + ' | '.join(cells) + ' |')
        lines_csv.append(','.join(cells))

    table_md = '\n'.join(lines_md)
    print(table_md)

    # ---- provenance manifest -------------------------------------------------------------------------
    manifest = dict(eval_root=args.eval_root, split=recs[0].get('split'),
                    crop_root=recs[0].get('crop_root'), results=[])
    for rec, pool, comb, per in rows:
        w = rec.get('weight')
        manifest['results'].append(dict(
            method=rec['method'], source=rec['source'], pooling_used=pool,
            declared_pooling=rec.get('pooling'), neck_feat=rec.get('neck_feat'),
            weight=w, weight_sha256=sha256(w) if args.hash_weights else None,
            weight_bytes=os.path.getsize(w) if w and osp.exists(w) else None,
            # drop the pid sets: they are a working detail and not JSON-serialisable
            roles={k: {kk: vv for kk, vv in r.items() if kk != 'pids'} for k, r in rec['roles'].items()},
            n_query_videos=rec.get('n_query_videos'), mated_ids=rec.get('mated_ids'),
            note=('n_query in the eval JSON is the PRE-filter count; eval_func divides by num_valid_q, '
                  'i.e. only queries whose identity appears in the gallery'),
            eval_file=rec['eval_file'],
            metrics=dict(combined=comb, per_range=per)))
    if args.splits_dir:
        manifest['split_files'] = {osp.basename(p): dict(sha256=sha256(p), bytes=os.path.getsize(p))
                                   for p in sorted(glob.glob(osp.join(args.splits_dir, '*.kwcoco.json')))}
    manifest['environment'] = dict(python=sys.version.split()[0], numpy=np.__version__,
                                   interpreter=sys.executable)

    os.makedirs(args.out_dir or args.eval_root, exist_ok=True)
    base = args.out_dir or args.eval_root
    for fname, payload in (('results_table.md', table_md + '\n'),
                           ('results_table.csv', '\n'.join(lines_csv) + '\n'),
                           ('provenance.json', json.dumps(manifest, indent=2) + '\n')):
        with open(osp.join(base, fname), 'w') as f:
            f.write(payload)
        print('wrote', osp.join(base, fname), file=sys.stderr)


def cmd_verify(args):
    """Re-derive what a reader must trust. Any mismatch is a hard failure."""
    fails, checks = [], 0

    truth = {}
    if args.splits_dir:
        for role, coco, tok in features_io.discover_role_files(args.splits_dir):
            crecs, itok, croot = features_io.load_crops_full(coco, role)
            key = '{}_{}'.format(role, tok or itok)
            pids = {r['pid'] for r in crecs}
            vids = {r['video'] for r in crecs}
            multi = sum(1 for v in vids
                        if len({r['pid'] for r in crecs if r['video'] == v}) > 1) if args.deep else 0
            truth[key] = dict(n_crops=len(crecs), n_videos=len(vids), n_pids=len(pids),
                              pids=pids, crop_root=croot, multi=multi)
            checks += 1
            print('split  %-24s crops=%7d videos=%6d pids=%3d' % (key, len(crecs), len(vids), len(pids)))
            # every crop file must exist (sampled unless --deep)
            sample = crecs if args.deep else crecs[::max(1, len(crecs) // 200)]
            missing = [r['path'] for r in sample if not osp.exists(r['path'])]
            if missing:
                fails.append('%s: %d/%d sampled crop files MISSING, e.g. %s'
                             % (key, len(missing), len(sample), missing[0]))

    for rec in collect(args.eval_root):
        name = '{}/{}'.format(rec['method'], rec['source'])
        dims = {r['feat_dim'] for r in rec['roles'].values()}
        checks += 1
        if len(dims) != 1:
            fails.append('%s: inconsistent feature dim across roles: %s' % (name, dims))
        for key, r in rec['roles'].items():
            if key in truth:
                checks += 1
                if r['n_crops'] != truth[key]['n_crops']:
                    fails.append('%s [%s]: npz has %d rows, split has %d crops'
                                 % (name, key, r['n_crops'], truth[key]['n_crops']))
                if r['n_videos'] != truth[key]['n_videos']:
                    fails.append('%s [%s]: npz has %d videos, split has %d'
                                 % (name, key, r['n_videos'], truth[key]['n_videos']))
        # a video-native cache MUST declare pooling='video' -- its per-crop rows are broadcast copies and
        # image pooling over them is degenerate
        if rec.get('pooling') not in ('', 'video'):
            fails.append('%s: unexpected declared pooling %r' % (name, rec.get('pooling')))
        # gallery and query must share a pid space, else nothing can ever match
        gal = [k for k in rec['roles'] if k.startswith('gallery')]
        qry = [k for k in rec['roles'] if k.startswith('query')]
        if gal and qry and truth:
            gk, qk = gal[0], qry[0]
            if gk in truth and qk in truth:
                checks += 1
                mated = truth[gk]['pids'] & truth[qk]['pids']
                if not mated:
                    fails.append('%s: gallery and query share ZERO pids -- nothing can match' % name)
                else:
                    print('mated  %-24s %d identities in both roles' % (name, len(mated)))
        print('cache  %-24s dim=%-5s pooling=%-7r roles=%d' % (
            name, dims.pop() if len(dims) == 1 else '?', rec.get('pooling'), len(rec['roles'])))

    print('\n%d checks run, %d failures' % (checks, len(fails)))
    for f in fails:
        print('  FAIL:', f)
    return 1 if fails else 0


def cmd_summary(args):
    """Status matrix across every dataset/split: one row per method/source, one column per split.

    Two jobs at once. (a) Shows at a glance what has run and what has not, so 'where are we' does not mean
    trawling directories. (b) Reports each method's RANK within every split -- a method whose rank moves a
    long way between splits is the anomaly worth raising, since a fair benchmark should mostly preserve
    ordering unless a split stresses something specific (e.g. clothes-invariance)."""
    splits, cells = [], {}
    for eroot in sorted(glob.glob(osp.join(args.bench_root, 'eval', '*', '*'))):
        if not osp.isdir(eroot):
            continue
        label = '/'.join(eroot.split(os.sep)[-2:])
        recs = collect(eroot)
        scored = []
        for rec in recs:
            pool, comb, per = headline(rec)
            key = '{}/{}'.format(rec['method'], rec['source'])
            scored.append((key, comb['mAP'] if comb else None, comb['rank1'] if comb else None))
        if not scored:
            continue
        splits.append(label)
        ranked = sorted([s for s in scored if s[1] is not None], key=lambda t: -t[1])
        rank_of = {k: i + 1 for i, (k, _, _) in enumerate(ranked)}
        for key, mAP, r1 in scored:
            cells[(key, label)] = (mAP, r1, rank_of.get(key))

    methods = sorted({k for k, _ in cells})
    w = max([len(m) for m in methods] + [14])
    print('%-*s  %s' % (w, 'method/source', '  '.join('%-22s' % s for s in splits)))
    for m in methods:
        row = []
        for s in splits:
            c = cells.get((m, s))
            row.append('%-22s' % ('-- not run --' if c is None else
                                  ('mAP %.3f R1 %.3f #%d' % (c[0], c[1], c[2]) if c[0] is not None
                                   else 'extracted, no eval')))
        print('%-*s  %s' % (w, m, '  '.join(row)))

    print('\nlegend: #N = rank by mAP within that split. Large rank movement between splits is the thing to '
          'question with the team -- it means a split is stressing something the others do not.')
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)

    s = sub.add_parser('summary', help='status matrix of every method across every split')
    s.add_argument('--bench-root', default='./reid-bench')
    s.set_defaults(func=cmd_summary)

    r = sub.add_parser('report', help='emit results table + provenance manifest')
    r.add_argument('--eval-root', required=True, help='.../eval/<dataset>/<split>')
    r.add_argument('--splits-dir', default='', help='hash the split kwcoco files too')
    r.add_argument('--out-dir', default='', help='default: eval-root')
    r.add_argument('--hash-weights', action='store_true',
                   help='sha256 every checkpoint (slow but this is what makes the table reproducible)')
    r.set_defaults(func=cmd_report)

    v = sub.add_parser('verify', help='re-derive and check the facts behind the table')
    v.add_argument('--eval-root', required=True)
    v.add_argument('--splits-dir', default='')
    v.add_argument('--deep', action='store_true',
                   help='check EVERY crop file exists and scan for multi-identity videos (slow)')
    v.set_defaults(func=cmd_verify)

    args = ap.parse_args()
    sys.exit(args.func(args) or 0)


if __name__ == '__main__':
    main()
