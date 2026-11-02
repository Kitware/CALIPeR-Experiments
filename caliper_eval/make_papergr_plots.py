# encoding: utf-8
"""Extract the paperGR sweep into JSON and plot each run's score as a function of epoch.

WHY THIS EXISTS. `make_val_plots.py` covers the held-out val curves, but it excludes `full230` by
construction: a run that trains on all 230 identities has no val split, writes no `val_metrics.jsonl`,
and therefore has no epoch curve anywhere. `full230` is also **the reported model**. So the run type
whose epoch behaviour matters most is the one with no curve -- and the only record of how it moves
across epochs is the paperGR sweep (plan Steps 3-4), which leaves one `eval_mean_cosine.json` per
checkpoint scattered across 118 directories. This is the tool that reads them back as curves.

It is the paperGR counterpart of make_val_plots.py and deliberately mirrors its shape, CLI and house
style, so the two sets of figures read as one system.

WHAT IT EMITS, into --out-dir (default <repo>/papergr_curves/):

    papergr_curves.json          {method: {run: {range: [{epoch, mAP, rank1, n_query, ...}]}}}
    SWEEP.md                     the same data as tables; epochs in NUMERIC order, zero-shot row first
    png/<method>_<run>.png       one figure per run; a line per query range + zero-shot baselines

The JSON is the durable artifact -- every number quoted in an analysis should be traceable to it, and
figures in any other format can be regenerated from it without re-reading the benchmark tree.

⚠ THIS TOOL NEVER WRITES eval/RESULTS.md, BY DESIGN. RESULTS.md is the curated zero-shot-vs-reported
comparison and must gain ONE row per method; the sweep has 118 scores. Letting results_table.py pick
them up turns its paperGR table from 11 rows into 128 AND sorts epochs as strings (e100 before e15).
The sweep therefore reports into SWEEP.md here; updating RESULTS.md is plan Step 9, done once Step 6a
has selected the reported epoch.

⚠ PAPERGR IS A MEASUREMENT, NOT A SELECTION MECHANISM (plan Step 6, governing rule). The reported
epoch is chosen from val curves alone; paperGR is test data. A full paperGR curve makes it trivial to
read off the best point and publish it, which would inflate every reported number by an unknown
amount. This tool therefore labels the peak it marks as a measurement and never emits a "best epoch"
recommendation. Quote the val-selected epoch's RANK within the curve instead -- that quantifies
transfer without licensing a re-pick.

⚠ --overlay-val IS OFF BY DEFAULT AND SHOULD STAY THAT WAY FOR `full230`. Val curves come from
`hold30_*` runs (200 train ids) and these paperGR curves from `full230` (230 train ids) -- DIFFERENT
MODELS. Overlaying them implies a checkpoint-for-checkpoint correspondence that does not exist; it is
a shape comparison only (plan Step 6d).

Requires Matplotlib in the active Python environment. It reads JSON and writes images, so
no method repo needs to be importable:

    python -m caliper_eval.make_papergr_plots [--out-dir DIR] [--overlay-val]
"""
import argparse
import glob
import json
import os
import os.path as osp
import re
import sys

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
from caliper_eval import paths  # noqa: E402

# A swept checkpoint's source tag is "<run>_e<epoch>", e.g. caliper-ground_full230_e25.
# The greedy .+ is LOAD-BEARING: it anchors on the LAST _e<digits>, so caliper-ground_full230_ep40_e5
# parses as (caliper-ground_full230_ep40, 5). A non-greedy match, or a split on the first '_e', yields
# run='caliper-ground_full230' / epoch='p40_e5' and silently merges the control run into full230 --
# three of the four run names on disk contain an 'e' token (full230_ep40, full230_ep150,
# full230_s1x30), so this is not a hypothetical.
EPOCH_TAG = re.compile(r'^(?P<run>.+)_e(?P<epoch>\d+)$')

# Finetuned runs carry this prefix; it is stripped for display but is what distinguishes a swept
# checkpoint from a zero-shot source tag (market1501, msmt17, lsvid, agvpreid, ...).
FINETUNED_PREFIX = 'caliper-ground_'

# Each method's zero-shot row, drawn as the baseline to beat. These are exactly the checkpoints the
# finetuning runs were INITIALISED from (plan, "Zero-shot baselines to beat"), so the comparison is
# like-for-like rather than best-available.
ZERO_SHOT_SOURCE = {
    'bot': 'market1501',
    'personvit': 'msmt17',
    'clip-reid': 'msmt17',
    'sinet': 'lsvid',
    'x-tfclip': 'agvpreid',
    'cal': 'last',
}

# Range -> (colour, linestyle, z-order, label). Colours are the validated categorical slots 1-3 of
# the house palette, matching make_val_plots.py so val and paperGR figures read as one system.
# NOTE the key difference from val rows: paperGR JSONs call the pooled set `combined`, val calls it
# `all`. Same concept, different key -- see val_subset() for the mapping used by --overlay-val.
RANGES = [
    ('combined', '#1baf7a', ':',  1, 'pooled'),
    ('500m',     '#eb6834', '--', 2, '500m'),
    ('1000m',    '#2a78d6', '-',  3, '1000m'),
]

VAL_SUBSET_FOR = {'combined': 'all', '500m': '500m', '1000m': '1000m'}


def eval_root(args):
    return args.eval_root or osp.join(paths.ROOT, 'eval')


def split_root(args):
    return osp.join(eval_root(args), args.dataset, args.split)


def collect(args):
    """{method: {run: {range: [rows sorted by epoch]}}} from every swept checkpoint's eval json.

    Methods and runs are DISCOVERED from the tree rather than hardcoded, so x-tfclip and cal appear
    automatically once their sweeps land.
    """
    fname = 'eval_{}_{}.json'.format(args.pool_op, args.metric)
    out = {}
    for p in sorted(glob.glob(osp.join(split_root(args), '*', '*', fname))):
        parts = p.split(osp.sep)
        method, source = parts[-3], parts[-2]
        if not source.startswith(FINETUNED_PREFIX):
            continue                      # a zero-shot row, not a swept checkpoint
        m = EPOCH_TAG.match(source)
        if not m:
            continue                      # finetuned but not epoch-tagged; nothing to place on an axis
        run = m.group('run')[len(FINETUNED_PREFIX):]
        epoch = int(m.group('epoch'))
        with open(p) as f:
            res = json.load(f)['results'].get(args.pooling, {})
        for rng, vals in res.items():
            row = {k: v for k, v in vals.items() if k != 'cmc'}
            row['epoch'] = epoch
            out.setdefault(method, {}).setdefault(run, {}).setdefault(rng, []).append(row)
    for runs in out.values():
        for per_range in runs.values():
            for rows in per_range.values():
                rows.sort(key=lambda r: r['epoch'])
    return out


def coverage(args):
    """(scored, total, [unscored dirs]) over every extracted checkpoint dir.

    Extraction and scoring are separate steps, and a dir holding three .npz files but no eval json is
    the normal mid-pipeline state -- NOT an error. Reporting it is the point: a curve drawn from a
    partially-scored sweep looks exactly like a complete one.
    """
    fname = 'eval_{}_{}.json'.format(args.pool_op, args.metric)
    scored, unscored = 0, []
    for d in sorted(glob.glob(osp.join(split_root(args), '*', FINETUNED_PREFIX + '*'))):
        if not osp.isdir(d) or not glob.glob(osp.join(d, '*.npz')):
            continue
        if osp.isfile(osp.join(d, fname)):
            scored += 1
        else:
            unscored.append(osp.join(osp.basename(osp.dirname(d)), osp.basename(d)))
    return scored, scored + len(unscored), unscored


def zero_shot(args, method):
    """{range: {mAP, rank1, ...}} for the checkpoint this method was finetuned FROM, or {}."""
    src = ZERO_SHOT_SOURCE.get(method)
    if not src:
        return {}
    p = osp.join(split_root(args), method, src,
                 'eval_{}_{}.json'.format(args.pool_op, args.metric))
    if not osp.isfile(p):
        return {}
    with open(p) as f:
        return json.load(f)['results'].get(args.pooling, {})


def load_val(args, method, run):
    """Val curves for the overlay, or {}. Falls back across holdout rules; see the ⚠ in the docstring."""
    p = args.val_curves or osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))),
                                    'val_curves', 'val_curves.json')
    if not osp.isfile(p):
        return {}
    with open(p) as f:
        data = json.load(f)
    runs = data.get(method, {})
    if run in runs:                       # same run has its own val curve (a hold30_* sweep)
        return runs[run]
    return runs.get(args.overlay_val_run, {})


def plot_run(args, method, run, per_range, zs, val):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=160)
    drawn = 0
    for rng, colour, ls, z, label in RANGES:
        rows = per_range.get(rng)
        if not rows:
            continue
        drawn += 1
        xs = [r['epoch'] for r in rows]
        ys = [100.0 * r[args.y] for r in rows]
        ax.plot(xs, ys, color=colour, linestyle=ls, linewidth=1.8, zorder=z,
                label='%s (n=%s)' % (label, rows[0].get('n_query', '?')))
        # Mark the maximum. This is a MEASUREMENT of where the curve peaks, not a recommendation --
        # see the governing rule in the module docstring.
        bi = max(range(len(ys)), key=lambda i: ys[i])
        ax.plot([xs[bi]], [ys[bi]], marker='o', ms=5, color=colour, zorder=z + 10,
                markeredgecolor='white', markeredgewidth=1.0)
        # Zero-shot baseline for this range: the number finetuning has to beat.
        if rng in zs:
            ax.axhline(100.0 * zs[rng][args.y], color=colour, linestyle=(0, (1, 3)),
                       linewidth=1.0, alpha=0.75, zorder=0)
    if not drawn:
        plt.close(fig)
        return False

    if val:
        ax2 = ax.twinx()
        for rng, colour, ls, z, label in RANGES:
            rows = val.get(VAL_SUBSET_FOR.get(rng, rng))
            if not rows:
                continue
            ax2.plot([r['epoch'] for r in rows], [100.0 * r['mAP'] for r in rows],
                     color=colour, linestyle=ls, linewidth=1.0, alpha=0.35, zorder=0)
        ax2.set_ylabel('val mAP (%%) — %s, DIFFERENT MODEL' % args.overlay_val_run, fontsize=7)
        for side in ('top',):
            ax2.spines[side].set_visible(False)

    ax.set_xlabel('epoch')
    ax.set_ylabel('paperGR %s (%%)' % args.y)
    ax.set_title('%s / %s   —  paperGR (dotted = zero-shot)' % (method, run), fontsize=11, loc='left')
    ax.grid(True, linewidth=0.5, alpha=0.35)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc='lower right', ncol=3)
    fig.tight_layout()
    fig.savefig(osp.join(args.out_dir, 'png', '%s_%s.png' % (method, run)))
    plt.close(fig)
    return True


def write_markdown(args, data, zs_by_method, out_md):
    """A human-readable sweep table -- the deliberate alternative to writing into eval/RESULTS.md.

    RESULTS.md is the curated zero-shot-vs-reported comparison and must gain ONE row per method, not
    118 (plan Step 9). The sweep is curve data, so it gets its own document, with epochs as rows in
    NUMERIC order -- results_table.py would sort them as strings and put e100 before e15.
    """
    L = ['# paperGR sweep — score as a function of epoch', '',
         'Generated by `caliper_eval.make_papergr_plots`. One section per run; the zero-shot row is the',
         'checkpoint that run was finetuned FROM, so each column is a like-for-like comparison.', '',
         '⚠ **These are measurements, not selections.** The reported epoch is chosen from val curves',
         'only — paperGR is test data (plan Step 6, governing rule). Quote the val-selected epoch\'s',
         'rank within a curve; do not re-pick the maximum.', '']
    for method in sorted(data):
        zs = zs_by_method.get(method, {})
        for run in sorted(data[method]):
            per_range = data[method][run]
            ranges = [r for r, _, _, _, _ in RANGES if r in per_range]
            if not ranges:
                continue
            by_epoch = {}
            for rng in ranges:
                for row in per_range[rng]:
                    by_epoch.setdefault(row['epoch'], {})[rng] = row
            L += ['', '## %s / %s' % (method, run), '',
                  '| epoch | ' + ' | '.join('%s mAP | %s R1' % (r, r) for r in ranges) + ' |',
                  '|--:' * (1 + 2 * len(ranges)) + '|']
            if zs:
                cells = ' | '.join(
                    ('%.1f | %.1f' % (100 * zs[r]['mAP'], 100 * zs[r]['rank1'])) if r in zs else '— | —'
                    for r in ranges)
                L.append('| *zero-shot* | %s |' % cells)
            for ep in sorted(by_epoch):                 # NUMERIC, not lexicographic
                cells = ' | '.join(
                    ('%.1f | %.1f' % (100 * by_epoch[ep][r]['mAP'], 100 * by_epoch[ep][r]['rank1']))
                    if r in by_epoch[ep] else '— | —' for r in ranges)
                L.append('| %d | %s |' % (ep, cells))
    with open(out_md, 'w') as f:
        f.write('\n'.join(L) + '\n')
    return out_md


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--eval-root', default='', help='default: <REID_BENCH_ROOT>/eval')
    ap.add_argument('--dataset', default='combined_brc1_brc2')
    ap.add_argument('--split', default='paperGR')
    ap.add_argument('--pool-op', default='mean', choices=['mean', 'max'])
    ap.add_argument('--metric', default='cosine', choices=['cosine', 'euclidean'])
    ap.add_argument('--pooling', default='video', choices=['image', 'video'])
    ap.add_argument('--y', default='mAP', choices=['mAP', 'rank1', 'rank5', 'rank10'],
                    help='which metric to put on the y axis (default mAP)')
    ap.add_argument('--runs', nargs='+', default=None,
                    help="only these runs, e.g. full230 (default: every run found)")
    ap.add_argument('--methods', nargs='+', default=None)
    ap.add_argument('--out-dir', default=osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))),
                                                  'papergr_curves'))
    ap.add_argument('--no-png', action='store_true', help='write only papergr_curves.json')
    ap.add_argument('--no-markdown', action='store_true',
                    help='skip SWEEP.md. This tool NEVER writes eval/RESULTS.md -- that is plan Step 9')
    ap.add_argument('--overlay-val', action='store_true',
                    help='overlay val curves on a twin axis. OFF by default: for full230 these are '
                         'different training runs (200 vs 230 ids), a shape comparison only')
    ap.add_argument('--overlay-val-run', default='hold30_thin')
    ap.add_argument('--val-curves', default='', help='default: <repo>/val_curves/val_curves.json')
    args = ap.parse_args()

    scored, total, unscored = coverage(args)
    print('scored %d/%d extracted checkpoint dirs (%d unscored)' % (scored, total, len(unscored)))
    if unscored:
        # Name what is missing rather than drawing a partial curve that looks complete.
        by_run = {}
        for u in unscored:
            method, source = u.split(osp.sep)
            m = EPOCH_TAG.match(source)
            key = '%s/%s' % (method, m.group('run')[len(FINETUNED_PREFIX):] if m else source)
            by_run[key] = by_run.get(key, 0) + 1
        print('  unscored by run: ' + ', '.join('%s x%d' % (k, v) for k, v in sorted(by_run.items())))
        print('  -> run plan Step 5 (caliper_eval.eval_from_features) over those dirs first')

    data = collect(args)
    if args.methods:
        data = {m: v for m, v in data.items() if m in args.methods}
    if args.runs:
        data = {m: {r: c for r, c in v.items() if r in args.runs} for m, v in data.items()}
        data = {m: v for m, v in data.items() if v}
    if not data:
        raise SystemExit('no scored paperGR checkpoints found under %s -- run Step 5 first'
                         % split_root(args))

    os.makedirs(args.out_dir, exist_ok=True)
    jpath = osp.join(args.out_dir, 'papergr_curves.json')
    with open(jpath, 'w') as f:
        json.dump(data, f, indent=1, sort_keys=True)
    n_runs = sum(len(v) for v in data.values())
    n_pts = sum(len(rows) for m in data.values() for r in m.values() for rows in r.values())
    print('wrote %s  (%d methods, %d runs, %d points)' % (jpath, len(data), n_runs, n_pts))

    zs_by_method = {m: zero_shot(args, m) for m in data}
    if not args.no_markdown:
        md = write_markdown(args, data, zs_by_method, osp.join(args.out_dir, 'SWEEP.md'))
        print('wrote %s' % md)

    if args.no_png:
        return
    os.makedirs(osp.join(args.out_dir, 'png'), exist_ok=True)
    for method in sorted(data):
        zs = zs_by_method[method]
        if not zs:
            print('  ! no zero-shot baseline for %s -- plotted without it' % method)
        for run in sorted(data[method]):
            val = load_val(args, method, run) if args.overlay_val else {}
            if not plot_run(args, method, run, data[method][run], zs, val):
                continue
            c = data[method][run].get('1000m') or []
            note = ''
            if c:
                best = max(c, key=lambda r: r[args.y])
                base = zs.get('1000m', {}).get(args.y)
                note = '1000m peak e%d = %.1f (measurement — do NOT select on this)' % (
                    best['epoch'], 100 * best[args.y])
                if base is not None:
                    note += ' vs zero-shot %.1f' % (100 * base)
            print('  %-42s %d pts  %s' % ('png/%s_%s.png' % (method, run), len(c), note))


if __name__ == '__main__':
    main()
