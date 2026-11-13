# encoding: utf-8
"""Score cached ReID features against a gallery. Report mAP and rank accuracy for each query
group and for all queries combined."""
import argparse
import glob
import json
import os.path as osp
import sys
from collections import defaultdict

import numpy as np

# allow running as a plain script (python .../eval_from_features.py) as well as `-m caliper_eval...`
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
from caliper_eval.features_io import load_npz          # noqa: E402
from caliper_eval.eval_reid import eval_func           # noqa: E402

ALL_POOLINGS = ['image', 'video']


def discover_roles(features_dir):
    """Find one gallery archive and the query archives, sorted by range."""
    files = sorted(glob.glob(osp.join(features_dir, '*.npz')))
    gallery = [f for f in files if osp.basename(f).startswith('gallery_')]
    queries = [f for f in files if osp.basename(f).startswith('query_')]
    if len(gallery) != 1:
        raise SystemExit("expected exactly one gallery_*.npz in {}, found {}".format(
            features_dir, [osp.basename(f) for f in gallery]))
    if not queries:
        raise SystemExit("no query_*.npz found in {}".format(features_dir))

    def range_key(f):
        digits = ''.join(c for c in osp.basename(f) if c.isdigit())
        return int(digits) if digits else 1 << 30

    return gallery[0], sorted(queries, key=range_key)


def track_keep_mask(d, min_track_crops):
    if not min_track_crops or min_track_crops <= 0:
        return np.ones(len(d['pid']), dtype=bool)
    key = np.char.add(np.char.add(d['video'].astype(str), '#'), d['dive_track_id'].astype(str))
    uniq, counts = np.unique(key, return_counts=True)
    small = set(uniq[counts < min_track_crops].tolist())
    return np.array([k not in small for k in key])


def subsample_per_video(d, keep, max_per_video):
    """Keep up to max_per_video evenly spaced frames per video before pooling. Zero keeps all
    frames."""
    if not max_per_video or max_per_video <= 0:
        return keep
    frame = d['frame']; video = d['video']
    by_video = defaultdict(list)
    for i in np.nonzero(keep)[0]:
        by_video[video[i]].append(i)
    new_keep = np.zeros(len(keep), dtype=bool)
    for idx in by_video.values():
        idx = sorted(idx, key=lambda i: frame[i])
        n = len(idx)
        if n > max_per_video:
            sel = sorted({round(j * (n - 1) / (max_per_video - 1)) for j in range(max_per_video)})
            idx = [idx[s] for s in sel]
        new_keep[idx] = True
    return new_keep


def group_keys(d, level):
    if level == 'image':
        return np.arange(len(d['pid'])).astype(str)
    if level == 'video':
        return d['video'].astype(str)
    raise ValueError(level)


def pool(d, keep, level, pool_op):
    """Aggregate kept crops into (F, pids, camids) at the chosen level (image=per crop, video=per clip)."""
    keys = group_keys(d, level)
    groups = defaultdict(list)
    for i in np.nonzero(keep)[0]:
        groups[keys[i]].append(i)
    feats = d['feats']; pid = d['pid']; camid = d['camid']
    F, pids, camids = [], [], []
    for k in sorted(groups):
        idx = groups[k]
        fl = feats[idx]
        F.append(fl.mean(0) if pool_op == 'mean' else fl.max(0))
        pids.append(int(pid[idx[0]])); camids.append(int(camid[idx[0]]))
    return np.asarray(F, dtype=np.float32), np.asarray(pids), np.asarray(camids)


def distance(qF, gF, metric):
    if metric == 'cosine':
        qn = qF / np.linalg.norm(qF, axis=1, keepdims=True).clip(min=1e-12)
        gn = gF / np.linalg.norm(gF, axis=1, keepdims=True).clip(min=1e-12)
        return (1.0 - qn @ gn.T).astype(np.float32)
    q2 = (qF ** 2).sum(1, keepdims=True)
    g2 = (gF ** 2).sum(1, keepdims=True)
    return (q2 + g2.T - 2.0 * qF @ gF.T).astype(np.float32)


def evaluate(qF, q_pids, q_camids, gF, g_pids, g_camids, metric, cam_exclusion):
    dm = distance(qF, gF, metric)
    cmc, mAP, dropped = eval_func(dm, q_pids, g_pids, q_camids, g_camids, cam_exclusion=cam_exclusion)
    return dict(n_query=int(qF.shape[0]), n_gallery=int(gF.shape[0]), mAP=float(mAP),
                rank1=float(cmc[0]), rank5=float(cmc[min(4, len(cmc)-1)]),
                rank10=float(cmc[min(9, len(cmc)-1)]),
                dropped_queries=int(dropped),
                cmc=[float(x) for x in cmc])   # full CMC curve (up to max_rank) for plotting


def main():
    ap = argparse.ArgumentParser(
        description="Score saved gallery and query features.",
        epilog="Paper scoring example: caliper-eval --features-dir /path/to/features --pooling video",
    )
    ap.add_argument('--features-dir', required=True, help='Directory containing the gallery and query feature archives.')
    ap.add_argument('--pooling', nargs='+', default=ALL_POOLINGS, choices=ALL_POOLINGS,
                    help='Score each crop (image) or each video (video). The default runs both. Use video for paper scores and for SINet or X-TFCLIP features.')
    ap.add_argument('--pool-op', default='mean', choices=['mean', 'max'],
                    help='Combine crop features by their mean (default) or elementwise maximum.')
    ap.add_argument('--metric', default='cosine', choices=['cosine', 'euclidean'],
                    help='Distance used to rank matches (default: cosine).')
    ap.add_argument('--min-track-crops', type=int, default=0,
                    help='Skip tracks with fewer than this many crops. Zero keeps all tracks.')
    ap.add_argument('--max-per-video', type=int, default=0,
                    help='Maximum number of evenly spaced frames per video before pooling. Zero uses all frames.')
    ap.add_argument('--cam-exclusion', default='none', choices=['none', 'market'],
                    help='Use none for CALIPeR. The Market1501 rule (market) excludes gallery '
                         'entries that share both the query identity and its camera.')
    ap.add_argument('--out', default='',
                    help='Path for the score JSON. By default it is saved beside the features.')
    ap.add_argument('--fig6-generation-dir', help='For Figure 6, give the directory containing the generated real and simulated query sets.')
    args = ap.parse_args()

    if args.fig6_generation_dir and sys.version_info < (3, 10):
        ap.error('--fig6-generation-dir requires Python 3.10 or newer')

    if args.fig6_generation_dir and (args.pooling != ['video'] or args.pool_op != 'mean'
            or args.metric != 'cosine' or args.cam_exclusion != 'none'
            or args.max_per_video != 0 or args.min_track_crops != 0):
        ap.error('Figure 6 requires --pooling video --pool-op mean --metric cosine, no filtering or camera exclusion')

    g_path, q_paths = discover_roles(args.features_dir)
    g = load_npz(g_path)
    queries = [load_npz(p) for p in q_paths]
    fig6 = None
    if args.fig6_generation_dir:
        from turbulence_eval.shared_fig6 import evaluation_provenance
        fig6 = evaluation_provenance(args.fig6_generation_dir, [g_path] + q_paths, [g] + queries)

    # Video-native caches require video pooling.
    required = str(g['pooling']) if 'pooling' in g else ''
    if required:
        bad = [p for p in args.pooling if p != required]
        if bad:
            raise SystemExit("this feature cache (method={}) declares pooling='{}' as the only supported mode; "
                             "requested {}. Use --pooling {}".format(
                                 str(g['method']), required, bad, required))
    prov = dict(method=str(g['method']), source=str(g['source']), split=str(g['split']),
                neck_feat=str(g['neck_feat']), weight=str(g['weight']))
    gkeep = subsample_per_video(g, track_keep_mask(g, args.min_track_crops), args.max_per_video)
    qkeep = [subsample_per_video(q, track_keep_mask(q, args.min_track_crops), args.max_per_video)
             for q in queries]
    print("method={method} source={source} split={split} | gallery={gr} queries={qr} [neck={neck_feat}]".format(
        gr=str(g['camera_token']), qr=[str(q['camera_token']) for q in queries], **prov))

    results = {}
    for level in args.pooling:
        gF, g_pids, g_camids = pool(g, gkeep, level, args.pool_op)
        per_range = {}
        qF_all, qp_all, qc_all = [], [], []
        print("\n== pooling={} (op={}, metric={}, max_per_video={}) ==  gallery units={} ({} subj)".format(
            level, args.pool_op, args.metric, args.max_per_video, gF.shape[0], len(set(g_pids.tolist()))))
        for q, keep in zip(queries, qkeep):
            qF, q_pids, q_camids = pool(q, keep, level, args.pool_op)
            res = evaluate(qF, q_pids, q_camids, gF, g_pids, g_camids, args.metric, args.cam_exclusion)
            per_range[str(q['camera_token'])] = res
            print("  [{:>7}] units={:5d}  mAP={:.1%}  R1={:.1%}  R5={:.1%}  R10={:.1%}".format(
                str(q['camera_token']), res['n_query'], res['mAP'], res['rank1'], res['rank5'], res['rank10']))
            qF_all.append(qF); qp_all.append(q_pids); qc_all.append(q_camids)
        qF = np.concatenate(qF_all); q_pids = np.concatenate(qp_all); q_camids = np.concatenate(qc_all)
        per_range['combined'] = evaluate(qF, q_pids, q_camids, gF, g_pids, g_camids, args.metric, args.cam_exclusion)
        c = per_range['combined']
        print("  [{:>7}] units={:5d}  mAP={:.1%}  R1={:.1%}  R5={:.1%}  R10={:.1%}".format(
            'combined', c['n_query'], c['mAP'], c['rank1'], c['rank5'], c['rank10']))
        results[level] = per_range

    summary = dict(pool_op=args.pool_op, metric=args.metric, min_track_crops=args.min_track_crops,
                   max_per_video=args.max_per_video, cam_exclusion=args.cam_exclusion,
                   results=results, **prov)
    if fig6 is not None:
        summary['fig6'] = fig6
    # the default (cam_exclusion='none') output is eval_<pool_op>_<metric>.json; a 'market' run gets a
    # distinguishing suffix so both can live in the same features dir.
    suffix = '' if args.cam_exclusion == 'none' else '_' + args.cam_exclusion
    out = args.out or osp.join(args.features_dir, 'eval_{}_{}{}.json'.format(args.pool_op, args.metric, suffix))
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    print("\nwrote", out)


if __name__ == '__main__':
    main()
