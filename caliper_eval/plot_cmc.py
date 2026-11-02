# encoding: utf-8
"""Overlay CMC curves across methods and/or query ranges on a single axis.

Accumulates over whatever methods have been evaluated: it scans every
  <eval-root>/<dataset>/<split>/<method>/<source>/eval_<pool_op>_<metric>.json
written by caliper_eval.eval_from_features (which now stores the full per-curve 'cmc'), and plots one line
per (method/source x range) you select. Color encodes method/source; linestyle encodes the query range, so
a method-comparison (one range, many methods) and a range-comparison (one method, many ranges) both read
cleanly on one plot. Re-run anytime as more methods finish.

NOTE: requires the eval JSONs to contain 'cmc' -> re-run eval_from_features once after this feature was
added (older JSONs without 'cmc' are skipped with a warning).

Usage:
  # method comparison at the combined query, video pooling:
  python -m caliper_eval.plot_cmc --pooling video --ranges combined
  # range comparison for one method:
  python -m caliper_eval.plot_cmc --pooling video --methods bot --ranges 500m 1000m
  # everything (methods x ranges) on one axis:
  python -m caliper_eval.plot_cmc --pooling video --ranges 500m 1000m combined
"""
import argparse
import glob
import json
import os.path as osp
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
from caliper_eval import paths  # noqa: E402

LINESTYLES = ['-', '--', ':', '-.']


def discover(eval_root, dataset, split, pool_op, metric):
    """Return [(method, source, summary_dict), ...] for every eval json found (sorted)."""
    pat = osp.join(eval_root, dataset, split, '*', '*', 'eval_{}_{}.json'.format(pool_op, metric))
    found = []
    for p in sorted(glob.glob(pat)):
        parts = p.split(osp.sep)
        method, source = parts[-3], parts[-2]
        with open(p) as f:
            found.append((method, source, json.load(f)))
    return found


def main():
    ap = argparse.ArgumentParser(description="Overlay CMC curves across methods/ranges on one axis")
    ap.add_argument('--split', default='ground_range')
    ap.add_argument('--dataset', default='kitbrc1')
    ap.add_argument('--eval-root', default='', help="default: <REID_BENCH_ROOT>/eval")
    ap.add_argument('--pool-op', default='mean', choices=['mean', 'max'])
    ap.add_argument('--metric', default='cosine', choices=['cosine', 'euclidean'])
    ap.add_argument('--pooling', default='video', choices=['image', 'video'])
    ap.add_argument('--ranges', nargs='+', default=['combined'],
                    help="query ranges to plot (e.g. 500m 1000m combined)")
    ap.add_argument('--methods', nargs='+', default=None, help="filter to these methods (default: all found)")
    ap.add_argument('--max-rank', type=int, default=20)
    ap.add_argument('--title', default='')
    ap.add_argument('--out', default='')
    args = ap.parse_args()

    eval_root = args.eval_root or osp.join(paths.ROOT, 'eval')
    found = discover(eval_root, args.dataset, args.split, args.pool_op, args.metric)
    if args.methods:
        found = [(m, s, d) for (m, s, d) in found if m in args.methods]
    if not found:
        raise SystemExit("no eval_{}_{}.json under {}".format(
            args.pool_op, args.metric, osp.join(eval_root, args.dataset, args.split)))

    # color per method/source, linestyle per range -> overlays stay legible
    msrcs = list(dict.fromkeys("{}/{}".format(m, s) for (m, s, _) in found))
    cmap = plt.get_cmap('tab10')
    color = {k: cmap(i % 10) for i, k in enumerate(msrcs)}
    ls = {r: LINESTYLES[i % len(LINESTYLES)] for i, r in enumerate(args.ranges)}

    plt.figure(figsize=(8, 6))
    plotted = 0
    for (m, s, d) in found:
        msrc = "{}/{}".format(m, s)
        for r in args.ranges:
            res = d.get('results', {}).get(args.pooling, {}).get(r)
            if not res or 'cmc' not in res:
                print("skip {} @ {}: no cmc for pooling={} (re-run eval_from_features to store cmc)".format(
                    msrc, r, args.pooling))
                continue
            cmc = res['cmc'][:args.max_rank]
            xs = list(range(1, len(cmc) + 1))
            label = "{} @ {} (mAP {:.1f}, R1 {:.1f})".format(msrc, r, res['mAP'] * 100, res['rank1'] * 100)
            plt.plot(xs, [c * 100 for c in cmc], color=color[msrc], linestyle=ls[r], marker='o', ms=3, label=label)
            plotted += 1
    if plotted == 0:
        raise SystemExit("nothing to plot (no matching 'cmc' found for pooling={} ranges={})".format(
            args.pooling, args.ranges))

    plt.xlabel('Rank'); plt.ylabel('Matching rate (%)'); plt.ylim(0, 100)
    plt.title(args.title or "CMC - {}/{} | {} | {}".format(args.dataset, args.split, args.pooling, args.metric))
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, loc='lower right')
    out = args.out or osp.join(eval_root, args.dataset, args.split,
                               "cmc_{}_{}_{}.png".format(args.pooling, args.pool_op, args.metric))
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print("wrote", out, "({} curves)".format(plotted))


if __name__ == '__main__':
    main()
