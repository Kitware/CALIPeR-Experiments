# encoding: utf-8
"""Build a Markdown report from saved evaluations and their gallery and query splits."""
import argparse
import collections
import glob
import json
import os.path as osp

from caliper_eval import paths

TRAIN_TEST_IDS = './data/train_test_ids.json'

# Reporting order, matching paperGR/results_table.md. Grouped by method with the smaller pretraining
# corpus first, so the msmt17>market1501 and lsvid>mars pairs read as adjacent comparisons. Anything
# not listed (e.g. the finetuned `caliper-ground` runs) is appended alphabetically.
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
KNOWN = collections.OrderedDict([
    ('combined_brc1_brc2/paperGR', ('paperGR', 'gallery 380m → query 500m + 1000m',
     'combined_brc1_brc2_test/test_ground-range_combined_paperGR.json')),
    ('kitbrc1/ground_range', ('ground_range (LEGACY)', 'gallery 380m → query 500m + 1000m',
     'built pre-partition from benchmark-data-new/KITBRC1')),
    ('kitbrc1/ground_range_sim', ('ground_range_sim (LEGACY)', 'real 380m gallery → pyBSM-degraded queries',
     'built pre-partition; 3 Cn² levels')),
    ('combined_brc1_brc2/cross-collection', ('cross-collection', 'BRC1 field → BRC2 field',
     'built from the UNPARTITIONED combined_brc1_brc2/')),
    ('combined_brc1_brc2/cross-collection_test', ('cross-collection (partitioned)',
     'BRC1 field → BRC2 field, test ids only',
     'combined_brc1_brc2_test/test_cross-collection_combined.json')),
    ('combined_brc1_brc2/ground-aerial', ('ground-aerial', 'ground cams → aerial cams',
     'built from the UNPARTITIONED combined_brc1_brc2/')),
    ('combined_brc1_brc2/ground-aerial_test', ('ground-aerial (partitioned)',
     'ground cams → aerial cams, test ids only',
     'combined_brc1_brc2_test/test_ground-aerial_combined.json')),
    ('combined_brc1_brc2/enroll-field_test', ('enroll-field (partitioned)',
     'field video → enrollment stills + gait, test ids only',
     'combined_brc1_brc2_test/test_enroll-field_combined.json')),
    ('combined_brc1_brc2/clothing-brc1_test', ('clothing BRC1 (partitioned)',
     'BRC1 set1 → BRC1 set2, test ids only',
     'combined_brc1_brc2_test/test_clothing_combined_brc1.json')),
    ('combined_brc1_brc2/clothing-brc2_test', ('clothing BRC2 (partitioned)',
     'BRC2 set1 → BRC2 set2, test ids only',
     'combined_brc1_brc2_test/test_clothing_combined_brc2.json')),
    ('combined_brc1_brc2/clothing-brc1_test_fieldonly', ('clothing BRC1 (field only)',
     'BRC1 set1 -> set2, field video only, test ids only',
     'derived: manifests/clothing-brc1_fieldonly.json (controlled media removed)')),
    ('combined_brc1_brc2/clothing-brc2_test_fieldonly', ('clothing BRC2 (field only)',
     'BRC2 set1 -> set2, field video only, test ids only',
     'derived: manifests/clothing-brc2_fieldonly.json (controlled media removed)')),
    ('combined_brc1_brc2/enroll-field_test_wbonly', ('enroll-field (wb only)',
     'field video -> controlled whole-body + gait, test ids only',
     'derived: manifests/enroll-field_wbonly.json (controlled/face removed)')),
    ('combined_brc1_brc2/q1000', ('q1000 (probe)', 'all other ranges → query 1000m',
     'combined_brc1_brc2_test/test_ground-range_combined_q1000.json')),
])

# Placeholder for an unavailable score.
PENDING = 'NOW'


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
    ap.add_argument('--out', default='')
    args = ap.parse_args()
    E = osp.join(paths.ROOT, 'eval')
    out = args.out or osp.join(E, 'RESULTS.md')
    train_ids = set(json.load(open(TRAIN_TEST_IDS))['train'])

    cells = collections.defaultdict(dict)       # split -> combo -> results
    for f in glob.glob(osp.join(E, '*', '*', '*', '*', 'eval_*.json')):
        p = f.split(osp.sep)
        cells['{}/{}'.format(p[-5], p[-4])]['{}/{}'.format(p[-3], p[-2])] = json.load(open(f))['results']['video']
    # A KNOWN split is listed as soon as its kwcoco exists, even with no eval cells yet: its identity
    # counts and post-training validity are already decided by the split, and listing it is what keeps
    # a planned-but-unextracted split visible instead of silently absent.
    candidates = list(KNOWN) + sorted(s for s in cells if s not in KNOWN)
    metas = {s: split_meta(s, train_ids) for s in candidates}
    order = [s for s in candidates if metas.get(s)]

    L = ["# CALIPeR ReID results\n",
         "Video pooling, cosine, all frames (`--max-per-video 0`). Regenerate with",
         "`python -m caliper_eval.results_table`.\n",
         "## Training identities in evaluation splits\n",
         "A split is only valid after finetuning if it contains **no training identities**. Most of",
         "these predate the 232/170 identity partition.\n",
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

    combos = sorted({c for v in cells.values() for c in v}, key=combo_key)
    L += ["", "## Which method ran on which split\n",
          "| method / source | " + " | ".join(KNOWN.get(s, (s,))[0] for s in order) + " |",
          "|---" * (1 + len(order)) + "|"]
    for c in combos:
        # `—` means this combo did not run on an otherwise-extracted split; PENDING means nothing has
        # run on that split at all, so the blank is a scheduling state rather than a decision.
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
            # Split built, nothing extracted. Emit the table the extraction will fill, so the shape of
            # the pending work (which methods, which query ranges) is visible in the doc itself.
            toks = query_tokens(s)
            L += ["⏳ **Features not extracted yet** — every metric below reads `{}` until "
                  "`{}` has been run for this split.\n".format(PENDING, 'extract_caliper_features.py'),
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
          "- **`paperGR` is the only ground-range split built on the identity partition.** "
          "`ground_range`/`ground_range_sim` are legacy (100 subjects, 28 of them training identities, "
          "and an old confidence gate on the 1000m stratum). Not comparable cell-for-cell.",
          "- **`ground-aerial` and `cross-collection` contain training identities** — both were built "
          "from the unpartitioned `combined_brc1_brc2/`, not the `_test` variant, despite the latter "
          "existing. `ground-aerial` carries all 232; `cross-collection` carries 89, all of them "
          "gallery distractors (its query side was built `--mated-only`, so it holds 39 test ids).",
          "- **The `(partitioned)` splits are the fixed re-cuts of those two, plus `enroll-field`.** "
          "Same protocols, rebuilt from `combined_brc1_brc2_test/` so every identity on both sides is "
          "in the 170-id test partition. They are ADDITIONS, not replacements: the unpartitioned rows "
          "stay so the existing zero-shot numbers remain readable.",
          "- ⚠ **Partitioned and unpartitioned rows are NOT comparable cell-for-cell.** Dropping the "
          "training identities also drops them as gallery distractors — `cross-collection` goes 202 → "
          "113 gallery ids and `ground-aerial` 400 → 170 — so the partitioned splits are EASIER and "
          "will score higher for reasons that have nothing to do with the model.",
          "- **`cross-collection` is open-set** (39 mated of 96 query ids) and `eval_reid.py` silently "
          "drops unmated probes. ~±15 pp Rank-1 resolution; qualitative only. The partitioned re-cut "
          "keeps the same 39 mated ids, so it is open-set on the same terms.",
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
