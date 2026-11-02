# encoding: utf-8
"""Extract every training run's val curves to JSON and plot one PNG per run.

WHY THIS EXISTS. `val_metrics.jsonl` is written per run by caliper_eval.train_metrics during training,
one row per (epoch, query-range). That is the only record of how a run's held-out performance moved,
and it is the sole legitimate basis for choosing which checkpoint to report -- paperGR is test data
and must not be used to select. But the files are scattered one-per-run under
reid-bench/train/<method>/<run>/, in a format nobody reads by eye, so the curves were repeatedly being
re-derived by ad-hoc scripts that were never saved. This is that code, saved.

WHAT IT EMITS, into --out-dir (default <repo>/val_curves/):

    val_curves.json          {method: {run: {subset: [{epoch, mAP, R1, n_query, n_gallery, ...}]}}}
    png/<method>_<run>.png   one figure per run; three lines (1000m / 500m / pooled `all`)

The JSON is the durable artifact -- every number quoted in an analysis should be traceable to it, and
PDFs or any other format can be regenerated from it later without re-reading the training tree.

ONLY HOLDOUT RUNS HAVE CURVES. A `full230*` run trains on all 230 identities, so it has no held-out
split, writes no val_metrics.jsonl, and simply does not appear here. That is not a gap to fix: those
runs are evaluated on paperGR instead. Runs are discovered from the directory tree rather than a
hardcoded list, so X-TFCLIP and CAL appear automatically once their matrices land.

READ THE 1000m CURVE, NOT THE POOLED ONE. Over ~30 held-out identities the 500m half saturates for
every method (zero-shot BoT is R1 89.3 at 500m against 10.4 at 1000m), so the pooled `all` curve is
dominated by a range that carries almost no signal about which epoch is best.

Needs matplotlib. It reads JSONL and writes images, so no method repo needs to be importable:

    python -m caliper_eval.make_val_plots [--train-root DIR] [--out-dir DIR]
"""
import argparse
import json
import os
import os.path as osp
import sys

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
from caliper_eval import paths  # noqa: E402

# Subset -> (colour, linestyle, z-order). Colours are the validated categorical slots 1-3 of the
# house palette; 1000m is drawn last and heaviest because it is the curve that decides anything.
SUBSETS = [
    ('all',   '#1baf7a', ':',  1, 'pooled'),
    ('500m',  '#eb6834', '--', 2, '500m'),
    ('1000m', '#2a78d6', '-',  3, '1000m'),
]


def collect(train_root):
    """{method: {run: {subset: [rows sorted by epoch]}}} from every val_metrics.jsonl under the tree."""
    out = {}
    for method in sorted(os.listdir(train_root)):
        mdir = osp.join(train_root, method)
        if not osp.isdir(mdir):
            continue
        for run in sorted(os.listdir(mdir)):
            f = osp.join(mdir, run, 'val_metrics.jsonl')
            if not osp.isfile(f):
                continue          # no holdout -> no val split -> nothing to plot. Expected.
            per_subset = {}
            with open(f) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    per_subset.setdefault(r['subset'], []).append(r)
            for rows in per_subset.values():
                rows.sort(key=lambda r: r['epoch'])
            if per_subset:
                out.setdefault(method, {})[run] = per_subset
    return out


def plot_run(method, run, per_subset, out_png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=160)
    n_q = {}
    for subset, colour, ls, z, label in SUBSETS:
        rows = per_subset.get(subset)
        if not rows:
            continue
        xs = [r['epoch'] for r in rows]
        ys = [100.0 * r['mAP'] for r in rows]
        n_q[label] = rows[0].get('n_query')
        ax.plot(xs, ys, color=colour, linestyle=ls, linewidth=1.8, zorder=z,
                label='%s (n=%s)' % (label, rows[0].get('n_query', '?')))
        # mark the maximum -- the epoch this curve would select
        bi = max(range(len(ys)), key=lambda i: ys[i])
        ax.plot([xs[bi]], [ys[bi]], marker='o', ms=5, color=colour, zorder=z + 10,
                markeredgecolor='white', markeredgewidth=1.0)

    ax.set_xlabel('epoch')
    ax.set_ylabel('val mAP (%)')
    ax.set_title('%s / %s' % (method, run), fontsize=11, loc='left')
    ax.grid(True, linewidth=0.5, alpha=0.35)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc='lower right', ncol=3)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--train-root', default=osp.join(paths.ROOT, 'train'),
                    help='tree of <method>/<run>/val_metrics.jsonl (default: paths.ROOT/train)')
    ap.add_argument('--out-dir', default=osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))),
                                                  'val_curves'))
    ap.add_argument('--no-png', action='store_true', help='write only val_curves.json')
    args = ap.parse_args()

    data = collect(args.train_root)
    if not data:
        raise SystemExit('no val_metrics.jsonl found under %s' % args.train_root)

    os.makedirs(args.out_dir, exist_ok=True)
    jpath = osp.join(args.out_dir, 'val_curves.json')
    with open(jpath, 'w') as f:
        json.dump(data, f, indent=1, sort_keys=True)
    n_runs = sum(len(v) for v in data.values())
    n_rows = sum(len(rows) for m in data.values() for r in m.values() for rows in r.values())
    print('wrote %s  (%d methods, %d runs with val, %d rows)'
          % (jpath, len(data), n_runs, n_rows))

    if args.no_png:
        return
    png_dir = osp.join(args.out_dir, 'png')
    os.makedirs(png_dir, exist_ok=True)
    for method in sorted(data):
        for run in sorted(data[method]):
            out_png = osp.join(png_dir, '%s_%s.png' % (method, run))
            plot_run(method, run, data[method][run], out_png)
            c = data[method][run].get('1000m') or []
            best = max(c, key=lambda r: r['mAP']) if c else None
            print('  %-46s %s' % (osp.relpath(out_png, args.out_dir),
                                  ('1000m peak e%d = %.1f' % (best['epoch'], 100 * best['mAP']))
                                  if best else ''))


if __name__ == '__main__':
    main()
