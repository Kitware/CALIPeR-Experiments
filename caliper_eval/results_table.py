# encoding: utf-8
"""Build a Markdown report from saved evaluations and their gallery and query splits."""
import argparse
import collections
import glob
import json
import os
import os.path as osp
import math
import sys
import tempfile
from pathlib import Path

from caliper_eval.validation import published_protocol

from caliper_eval import paths

# Root of the released CALIPeR dataset tree. Override with CALIPER_DATA_ROOT.
TRAIN_TEST_IDS = osp.join(os.environ.get('CALIPER_DATA_ROOT', 'data'), 'train_test_ids.json')

# Group the results by method and pretraining source.
COMBO_ORDER = ['bot/market1501', 'sinet/mars', 'sinet/lsvid',
               'clip-reid/market1501', 'clip-reid/msmt17',
               'personvit/market1501', 'personvit/msmt17',
               'sapiensid/wb12m', 'cal/last', 'x-tfclip/agvpreid']


def combo_key(c):
    if c in COMBO_ORDER:
        return (0, COMBO_ORDER.index(c), '')
    base = c.split('/')[0]                       # keep variants next to their method, e.g. cal/last-*
    near = [i for i, k in enumerate(COMBO_ORDER) if k.split('/')[0] == base]
    return (0, near[-1] + 0.5, c) if near else (1, 0, c)

# split dir -> (display name, gallery -> query, provenance)
# Map the paper protocols to their splits; report broader splits separately.
KNOWN = collections.OrderedDict([
    ('combined_brc1_brc2/paperGR', ('paperGR', 'gallery 380m → query 500m + 1000m',
     'combined_brc1_brc2_test/test_ground-range_combined_paperGR.json')),
    ('combined_brc1_brc2/paperGR_panel_q500', ('paperGR 500m panel',
     'gallery 380m → panel queries at 500m', '')),
    ('combined_brc1_brc2/paperGR_panel_q1000', ('paperGR 1000m panel',
     'gallery 380m → panel queries at 1000m', '')),
    ('combined_brc1_brc2/ga-panel_test', ('ground-aerial panel (Ground-vs-Elevated)',
     'ground gallery → aerial panel queries', '')),
    ('combined_brc1_brc2/ef-wbonly_test', ('whole-body enrollment panel (Enroll-vs-Field)',
     'whole-body enrollment gallery → field panel queries', '')),
    ('combined_brc1_brc2/cc-brc1_test', ('ground clothing BRC1 (Same-collect-CC component)',
     'BRC1 ground cameras, set 1 → set 2', '')),
    ('combined_brc1_brc2/cc-brc2_test', ('ground clothing BRC2 (Same-collect-CC component)',
     'BRC2 ground cameras, set 1 → set 2', '')),
    ('combined_brc1_brc2/ccg_test', ('ground-only longitudinal',
     'BRC1 ground cameras → BRC2 ground cameras', '')),
    ('kitbrc1/ground_range_sim', ('ground_range_sim', 'real 380m gallery → pyBSM-degraded queries',
     'not identity-partitioned; 3 Cn² levels -- feeds the simulated-vs-real comparison, not the main survey')),
    ('combined_brc1_brc2/cross-collection_test', ('cross-collection (broader diagnostic)',
     'BRC1 field → BRC2 field, test ids only',
     'combined_brc1_brc2_test/test_cross-collection_combined.json')),
    ('combined_brc1_brc2/ground-aerial_test', ('ground-aerial (broader diagnostic)',
     'ground cams → aerial cams, test ids only',
     'combined_brc1_brc2_test/test_ground-aerial_combined.json')),
    ('combined_brc1_brc2/enroll-field_test', ('enroll-field (broader diagnostic)',
     'enrollment stills + gait → field video, test ids only',
     'combined_brc1_brc2_test/test_enroll-field_combined.json')),
    ('combined_brc1_brc2/clothing-brc1_test', ('clothing BRC1 (broader diagnostic)',
     'BRC1 set1 → BRC1 set2, test ids only',
     'combined_brc1_brc2_test/test_clothing_combined_brc1.json')),
    ('combined_brc1_brc2/clothing-brc2_test', ('clothing BRC2 (broader diagnostic)',
     'BRC2 set1 → BRC2 set2, test ids only',
     'combined_brc1_brc2_test/test_clothing_combined_brc2.json')),
])

# Placeholder for an unavailable score.
PENDING = '—'

PAPER_COLUMNS = ('500m', '1000m', 'Ground-vs-Elevated', 'Enroll-vs-Field',
                 'Same-collect-CC', 'Longitudinal')


def paper_values(cells, combo):
    """Select paper populations without falling back to a broader or incomplete split."""
    def get(split, token='combined'):
        return cells.get('combined_brc1_brc2/' + split, {}).get(combo, {}).get(token)

    if combo in ('cal/last', 'x-tfclip/agvpreid'):
        ranges = [get('paperGR', '500m'), get('paperGR', '1000m')]
    else:
        ranges = [get('paperGR_panel_q500'), get('paperGR_panel_q1000')]
    clothing = [get('cc-brc1_test'), get('cc-brc2_test')]
    average = None
    if all(component is not None for component in clothing):
        # Equal collection weights, regardless of query counts. Round only after averaging.
        average = {metric: sum(c[metric] for c in clothing) / 2.0 for metric in ('mAP', 'rank1')}
    return ranges + [get('ga-panel_test'), get('ef-wbonly_test'), average, get('ccg_test')]


def paper_summary(cells):
    lines = ['## Zero-shot results by paper protocol\n',
             'Results from the shared evaluator, mAP / Rank-1 (%).',
             'Splits containing training identities are excluded. Missing results appear as —.',
             'Longitudinal and Same-collect-CC use ground cameras only.',
             'Same-collect-CC averages BRC1 and BRC2 metrics equally; both are required.\n',
             '| method / source | ' + ' | '.join(PAPER_COLUMNS) + ' |',
             '|---' * (1 + len(PAPER_COLUMNS)) + '|']
    for combo in COMBO_ORDER:
        values = paper_values(cells, combo)
        formatted = [PENDING if v is None else '{:.1f} / {:.1f}'.format(v['mAP'] * 100, v['rank1'] * 100)
                     for v in values]
        lines.append('| `{}` | {} |'.format(combo, ' | '.join(formatted)))
    return lines + ['']


def split_meta(split_dir, train_ids):
    """(gallery clips, gallery ids, query clips, query ids, n training ids) from the kwcoco files."""
    d = osp.join(paths.splits_root(), split_dir.split('/')[-1])
    if not osp.isdir(d):
        return None
    subs, g, q, gi, qi = set(), 0, 0, 0, 0
    for f in sorted(glob.glob(osp.join(d, '*.kwcoco.json'))):
        c = json.load(open(f))
        nm = {s['id']: s['name'] for s in c['subjects']}
        subs |= set(nm.values())
        rid = {nm[t['subject_id']] for t in c['tracks']}
        if osp.basename(f).startswith('gallery'):
            g += len(c['videos']); gi = len(rid)
        else:
            q += len(c['videos']); qi = max(qi, len(rid))
    return dict(g=g, gi=gi, q=q, qi=qi, ntr=len(subs & train_ids))


def range_key(k):
    digits = ''.join(ch for ch in k if ch.isdigit())
    return int(digits) if digits else 1 << 30


def query_tokens(split_dir):
    """Read query labels from the split filenames for the report columns."""
    d = osp.join(paths.splits_root(), split_dir.split('/')[-1])
    toks = [osp.basename(f)[len('query_'):-len('.kwcoco.json')]
            for f in sorted(glob.glob(osp.join(d, 'query_*.kwcoco.json')))]
    return sorted(toks, key=range_key)


def load_train_ids(filename):
    """Read the required partition without treating invalid input as an empty training set."""
    with open(filename, encoding='utf-8') as f:
        partition = json.load(f)
    train = partition.get('train') if isinstance(partition, dict) else None
    if (not isinstance(train, list) or not train or
            any(not isinstance(subject, str) or not subject.strip() for subject in train)):
        raise ValueError("expected a JSON object with a nonempty 'train' array of nonempty subject-name strings")
    if any(subject != subject.strip() for subject in train):
        raise ValueError("training subject names must not contain leading or trailing whitespace; "
                         "use the exact canonical names from the evaluation splits")
    return set(train)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', default='', help='default: <REID_BENCH_ROOT>/eval/RESULTS.md')
    ap.add_argument('--train-test-ids', default=TRAIN_TEST_IDS, metavar='PATH',
                    help='identity partition JSON; default: <CALIPER_DATA_ROOT>/train_test_ids.json '
                         '(CALIPER_DATA_ROOT defaults to ./data)')
    args = ap.parse_args()
    try:
        train_ids = load_train_ids(args.train_test_ids)
    except (OSError, ValueError) as exc:
        ap.error('--train-test-ids {!r}: {}'.format(args.train_test_ids, exc))
    E = osp.join(paths.ROOT, 'eval')
    out = args.out or osp.join(E, 'RESULTS.md')

    cells = collections.defaultdict(dict)
    selected, excluded = {}, collections.defaultdict(list)
    for f in sorted(glob.glob(osp.join(E, '*', '*', '*', '*', 'eval_*.json'))):
        p = Path(f).parts
        split, combo = '/'.join(p[-5:-3]), '/'.join(p[-3:-1])
        if split not in KNOWN:
            continue
        try:
            with open(f, encoding='utf-8') as stream:
                result = json.load(stream)
            if not published_protocol(result):
                excluded[(split, combo)].append(f)
                continue
            key = (split, combo)
            if key in selected:
                raise ValueError('multiple eligible results: {} and {}; keep one evaluation per method/source'.format(selected[key], f))
            if result.get('method') != p[-3] or result.get('source') != p[-2] or result.get('split') != p[-4]:
                raise ValueError('result metadata disagrees with its benchmark directory: ' + f)
            values = result['results']['video']
            expected = set(query_tokens(split))
            if not expected or set(values) != expected | {'combined'}:
                raise ValueError('query columns disagree with current split files: ' + f)
            for metrics in values.values():
                for name in ('mAP', 'rank1'):
                    value = metrics[name]
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                        raise ValueError('invalid {} in {}'.format(name, f))
            selected[key] = f
            cells[split][combo] = values
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            ap.error(str(exc))
    for key, filenames in excluded.items():
        if key not in selected:
            ap.error('no compatible video evaluation for {}/{}; rescore with --pooling video --pool-op mean --metric cosine --cam-exclusion none --min-track-crops 0 --max-per-video 0'.format(*key))
        print('excluded {} incompatible evaluation variant(s) for {}/{}'.format(len(filenames), *key), file=sys.stderr)
    # Include known splits with manifests, even before scores are saved.
    metas = {s: split_meta(s, train_ids) for s in KNOWN}
    order = [s for s in KNOWN if metas.get(s)]

    L = ["# CALIPeR ReID results\n",
         "Video pooling, cosine, all frames (`--max-per-video 0`). Regenerate with",
         "`python -m caliper_eval.results_table`.\n",
         "## Training identities in evaluation splits\n",
         "A split is only valid after finetuning if it contains **no training identities**.",
         "The table counts training identities in each gallery and query set.\n",
         "| split | gallery → query | gallery clips / ids | query clips / ids | training ids "
         "| valid post-training | Scored checkpoints |",
         "|---|---|---|---|--:|---|---|"]
    for s in order:
        name, desc, src = KNOWN.get(s, (s, '', ''))
        m = metas[s]
        ok = '**yes**' if m['ntr'] == 0 else '**NO** ({} ids)'.format(m['ntr'])
        L.append("| `{}` | {} | {:,} / {} | {:,} / {} | {} | {} | {} |".format(
            name, desc, m['g'], m['gi'], m['q'], m['qi'], m['ntr'], ok,
            len(cells[s])))

    safe_cells = {s: cells[s] for s in order if metas[s]['ntr'] == 0}
    L += ['', *paper_summary(safe_cells)]
    combos = sorted({c for s in order for c in cells[s]}, key=combo_key)
    L += ["", "## Which method ran on which split\n",
          "| method / source | " + " | ".join(KNOWN.get(s, (s,))[0] for s in order) + " |",
          "|---" * (1 + len(order)) + "|"]
    for c in combos:
        L.append("| `{}` | ".format(c) + " | ".join(
            "✓" if c in cells[s] else PENDING for s in order) + " |")
    L.append("")

    for s in order:
        m, name = metas.get(s), KNOWN.get(s, (s, '', ''))[0]
        desc, src = KNOWN.get(s, (s, '', ''))[1:]
        L += ["", "---", "", "## {}\n".format(name), "{}.{}\n".format(desc, " Source: `{}`".format(src) if src else ""),
              "Gallery **{:,} clips / {} identities** · query **{:,} clips / {} identities** · "
              "training identities present: **{}**{}\n".format(
                  m['g'], m['gi'], m['q'], m['qi'], m['ntr'],
                  " (overlaps the training set)" if m['ntr'] else " (no overlap)")]

        if not cells[s]:
            toks = query_tokens(s)
            L += ["No saved scores for this split. If you already have features, run `caliper-eval --features-dir /path/to/features --pooling video`.\n",
                  "| method / source | " + " | ".join("{} mAP | {} R1".format(t, t) for t in toks) +
                  " | comb mAP | comb R1 |", "|---" * (3 + 2 * len(toks)) + "|"]
            for c in COMBO_ORDER:
                L.append("| `{}` | ".format(c) + " | ".join([PENDING] * (2 * len(toks) + 2)) + " |")
            continue
        rows = sorted(((combo_key(c), c, r) for c, r in cells[s].items()))
        hdr = sorted([k for k in rows[0][2] if k != 'combined'], key=range_key)
        ncol = 3 + 2 * len(hdr)          # method + (mAP,R1) per range + (comb mAP, comb R1)
        L += ["| method / source | " + " | ".join("{} mAP | {} R1".format(h, h) for h in hdr) +
              " | comb mAP | comb R1 |", "|---" * ncol + "|"]
        for _, c, r in rows:
            cellstr = " | ".join("{:.1f} | {:.1f}".format(r[h]['mAP'] * 100, r[h]['rank1'] * 100)
                                 for h in hdr if h in r)
            L.append("| {} | {} | {:.1f} | {:.1f} |".format(
                c, cellstr, r['combined']['mAP'] * 100, r['combined']['rank1'] * 100))

    L += ["", "---", "", "## Notes", "",
          "- Use the exact identity-partitioned paper manifests. Changing a gallery changes its "
          "distractors and can change the metrics even when the checkpoint is identical. The "
          "identity check above does not verify manifest versions or camera membership.",
          "- **`ground_range_sim` is not identity-partitioned** (100 subjects, 28 training "
          "identities) and isn't part of the main zero-shot survey — it exists to compare simulated "
          "vs. real query CMC curves (same real 380m gallery, pyBSM-degraded 1000m queries at 3 "
          "turbulence strengths vs. real 1000m queries).",
          "- **Longitudinal uses ground cameras only**. Queries without gallery matches are "
          "excluded from mAP/CMC. Broader splits are reported separately.",
          "- **Same-collect-CC uses ground cameras only**. Average the separately scored BRC1 and "
          "BRC2 metrics with equal weights, before rounding.",
          "- **`enroll-field`'s gallery is 232 ids but 0 of them are training ids** — it is the 170 test "
          "subjects plus the 62 canonical ids that carry no field media and so fall in neither "
          "partition, kept as enrollment distractors.",
          "- **Ground-range manifests:** `test_ground-range_combined_paperGR.json` exists in both "
          "`combined_brc1_brc2_test/` (170 ids, 0 training) and `combined_brc1_brc2/` (365 gallery ids, "
          "195 training). Only the `_test` one is used here.", ""]
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=osp.dirname(osp.abspath(out)), delete=False) as stream:
            temporary = stream.name
            stream.write("\n".join(L))
        os.replace(temporary, out)
    finally:
        if temporary and osp.exists(temporary):
            os.unlink(temporary)
    print("wrote {} ({} splits, {} method/checkpoint pairs)".format(out, len(order), len(combos)))


if __name__ == '__main__':
    main()
