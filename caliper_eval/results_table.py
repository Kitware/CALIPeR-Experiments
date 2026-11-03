# encoding: utf-8
"""Build a Markdown report from saved evaluations and their gallery and query splits."""
import argparse
import collections
import glob
import json
import os
import os.path as osp

from caliper_eval import paths

# Root of the released CALIPeR dataset tree. Override with CALIPER_DATA_ROOT.
TRAIN_TEST_IDS = osp.join(os.environ.get('CALIPER_DATA_ROOT', 'data'), 'CALIPER_PyBSM', 'train_test_ids.json')

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

# split dir -> (display name, query -> gallery, provenance)
# Scoped to exactly what feeds the published results (RESULTS.md): the zero-shot survey and
# fine-tuning headline table (paperGR), the four "other modalities" columns (ground-aerial_test,
# enroll-field_test, the two clothing splits, cross-collection_test), and the simulated-vs-real
# comparison (ground_range_sim). All but ground_range_sim use the identity-partitioned ("_test")
# splits, so none of these rows mix training identities into the gallery as distractors.
KNOWN = collections.OrderedDict([
    ('combined_brc1_brc2/paperGR', ('paperGR', 'gallery 380m → query 500m + 1000m',
     'combined_brc1_brc2_test/test_ground-range_combined_paperGR.json')),
    ('kitbrc1/ground_range_sim', ('ground_range_sim', 'real 380m gallery → pyBSM-degraded queries',
     'not identity-partitioned; 3 Cn² levels -- feeds the simulated-vs-real comparison, not the main survey')),
    ('combined_brc1_brc2/cross-collection_test', ('cross-collection (Longitudinal)',
     'BRC1 field → BRC2 field, test ids only',
     'combined_brc1_brc2_test/test_cross-collection_combined.json')),
    ('combined_brc1_brc2/ground-aerial_test', ('ground-aerial (Ground-vs-Elevated)',
     'ground cams → aerial cams, test ids only',
     'combined_brc1_brc2_test/test_ground-aerial_combined.json')),
    ('combined_brc1_brc2/enroll-field_test', ('enroll-field (Enroll-vs-Field)',
     'field video → enrollment stills + gait, test ids only',
     'combined_brc1_brc2_test/test_enroll-field_combined.json')),
    ('combined_brc1_brc2/clothing-brc1_test', ('clothing BRC1 (Same-collect-CC)',
     'BRC1 set1 → BRC1 set2, test ids only',
     'combined_brc1_brc2_test/test_clothing_combined_brc1.json')),
    ('combined_brc1_brc2/clothing-brc2_test', ('clothing BRC2 (Same-collect-CC)',
     'BRC2 set1 → BRC2 set2, test ids only',
     'combined_brc1_brc2_test/test_clothing_combined_brc2.json')),
])

# Placeholder for an unavailable score.
PENDING = '—'


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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', default='', help='default: <REID_BENCH_ROOT>/eval/RESULTS.md')
    args = ap.parse_args()
    E = osp.join(paths.ROOT, 'eval')
    out = args.out or osp.join(E, 'RESULTS.md')
    train_ids = set(json.load(open(TRAIN_TEST_IDS))['train'])

    cells = collections.defaultdict(dict)       # split -> combo -> results
    for f in glob.glob(osp.join(E, '*', '*', '*', '*', 'eval_*.json')):
        p = f.split(osp.sep)
        cells['{}/{}'.format(p[-5], p[-4])]['{}/{}'.format(p[-3], p[-2])] = json.load(open(f))['results']['video']
    # Include known splits with manifests, even before features are extracted.
    metas = {s: split_meta(s, train_ids) for s in KNOWN}
    order = [s for s in KNOWN if metas.get(s)]

    L = ["# CALIPeR ReID results\n",
         "Video pooling, cosine, all frames (`--max-per-video 0`). Regenerate with",
         "`python -m caliper_eval.results_table`.\n",
         "## Training identities in evaluation splits\n",
         "A split is only valid after finetuning if it contains **no training identities**. All",
         "splits except `ground_range_sim` exclude training identities.\n",
         "| split | query → gallery | gallery clips / ids | query clips / ids | training ids "
         "| valid post-training | features |",
         "|---|---|---|---|--:|---|---|"]
    for s in order:
        name, desc, src = KNOWN.get(s, (s, '', ''))
        m = metas[s]
        ok = '**yes**' if m['ntr'] == 0 else '**NO** ({} ids)'.format(m['ntr'])
        L.append("| `{}` | {} | {:,} / {} | {:,} / {} | {} | {} | {} |".format(
            name, desc, m['g'], m['gi'], m['q'], m['qi'], m['ntr'], ok,
            '{} combos'.format(len(cells[s])) if cells[s] else '**{}**'.format(PENDING)))

    combos = sorted({c for s in order for c in cells[s]}, key=combo_key)
    L += ["", "## Which method ran on which split\n",
          "| method / source | " + " | ".join(KNOWN.get(s, (s,))[0] for s in order) + " |",
          "|---" * (1 + len(order)) + "|"]
    for c in combos:
        # `—` means this combo has no features on that split (PENDING renders the same way for a
        # split with no features at all).
        L.append("| `{}` | ".format(c) + " | ".join(
            "✓" if c in cells[s] else ("—" if cells[s] else PENDING) for s in order) + " |")
    L.append("")

    for s in order:
        m, name = metas.get(s), KNOWN.get(s, (s, '', ''))[0]
        desc, src = KNOWN.get(s, (s, '', ''))[1:]
        L += ["", "---", "", "## {}\n".format(name), "{}. Source: `{}`\n".format(desc, src),
              "Gallery **{:,} clips / {} identities** · query **{:,} clips / {} identities** · "
              "training identities present: **{}**{}\n".format(
                  m['g'], m['gi'], m['q'], m['qi'], m['ntr'],
                  " (overlaps the training set)" if m['ntr'] else " (no overlap)")]

        if not cells[s]:
            # Split built, no features extracted. Emit the empty table (which methods, which query
            # ranges) so the split's shape is visible in the doc itself.
            toks = query_tokens(s)
            L += ["**Features not extracted** for this split.\n",
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
          "- **Every split here except `ground_range_sim` uses the identity-partitioned (`_test`) "
          "variant** — training identities are excluded from the gallery, so distractor counts are "
          "consistent across rows and comparable to the fine-tuned checkpoints' own eval. Dropping "
          "training identities from a gallery also drops them as distractors, which would otherwise "
          "make an unpartitioned split score higher for reasons that have nothing to do with the "
          "model -- so don't compare these numbers against an unpartitioned run of the same split.",
          "- **`ground_range_sim` is not identity-partitioned** (100 subjects, 28 training "
          "identities) and isn't part of the main zero-shot survey — it exists to compare simulated "
          "vs. real query CMC curves (same real 380m gallery, pyBSM-degraded 1000m queries at 3 "
          "turbulence strengths vs. real 1000m queries).",
          "- **`cross-collection` is open-set** (39 mated of 96 query ids); a query with no gallery "
          "match is excluded from the mean automatically (counted as `dropped_queries` in the "
          "eval JSON). ~±15 pp Rank-1 resolution; "
          "qualitative only.",
          "- **`enroll-field`'s gallery is 232 ids but 0 of them are training ids** — it is the 170 test "
          "subjects plus the 62 canonical ids that carry no field media and so fall in neither "
          "partition, kept as enrollment distractors.",
          "- **Ground-range manifests:** `test_ground-range_combined_paperGR.json` exists in both "
          "`combined_brc1_brc2_test/` (170 ids, 0 training) and `combined_brc1_brc2/` (365 gallery ids, "
          "195 training). Only the `_test` one is used here.", ""]
    open(out, 'w').write("\n".join(L))
    print("wrote {} ({} splits, {} combos)".format(out, len(order), len(combos)))


if __name__ == '__main__':
    main()
