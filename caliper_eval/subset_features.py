# encoding: utf-8
"""Build a feature cache for one split out of caches already extracted for OTHER splits -- no GPU.

Several splits in this benchmark are re-cuts of the same media. `test_cross-collection_combined`
spans essentially every field video in the corpus, so a cache extracted for it already contains the
features `test_ground-aerial_combined` needs -- the two splits differ only in how those videos are
assigned to gallery/query roles, and in which identities they keep. Features are per-crop and
independent of split membership, so re-extracting would recompute identical vectors.

This module copies the rows across instead. It is deliberately NOT a role-to-role subsetter: source
rows from every role of every source dir are pooled and re-assigned by crop path, so a target
gallery may be drawn from a source query and vice versa.

Two things are taken from the TARGET split, never inherited from the source cache:

  * `pid` -- the source's identity numbering belongs to the source split's subject registry.
  * `camid` -- `eval_reid.eval_func` excludes gallery entries sharing a query's (pid, camid). The
    camera id is a per-split registry index, so inheriting it from a differently-built split would
    silently corrupt that exclusion.

Refuses to write if any target crop is absent from the pooled source, which is what distinguishes
"these splits are in a subset relation" from "this cache is quietly incomplete".

Usage:
  python -m caliper_eval.subset_features --dst-split ground-aerial_test \
      --src-dir eval/combined_brc1_brc2/cross-collection/cal/last-fullquery \
      --dst-source last --dataset combined_brc1_brc2
"""
import argparse
import glob
import json
import os
import os.path as osp

import numpy as np

from caliper_eval import paths

SCALARS = ['crop_root', 'neck_feat', 'weight', 'method', 'source', 'pooling', 'feat_dim']


def target_index(splits_dir):
    """[(role_stem, token, crop_rels, pids, camids, pid_to_subject)] for one split dir, in kwcoco order."""
    out = []
    for f in sorted(glob.glob(osp.join(splits_dir, '*.kwcoco.json'))):
        with open(f) as fh:
            d = json.load(fh)
        stem = osp.basename(f)[:-len('.kwcoco.json')]
        token = d['videos'][0]['camera_token'] if stem == 'gallery' else stem[len('query_'):]
        root = d.get('info', {}).get('crop_root', '')
        subj = {s['id']: s['name'] for s in d['subjects']}
        cam = {v['id']: int(v['camera_id']) for v in d['videos']}
        track = {t['id']: t for t in d['tracks']}
        ann = {a['image_id']: a for a in d['annotations']}
        rels, pids, cams = [], [], []
        for im in d['images']:
            a = ann.get(im['id'])
            if a is None:
                continue
            p = im['file_name']
            rels.append(osp.relpath(p, root) if osp.isabs(p) and root else p)
            pids.append(int(track[a['track_id']]['subject_id']))
            cams.append(cam[im['video_id']])
        out.append((stem, token, rels, np.asarray(pids, np.int64), np.asarray(cams, np.int64),
                    {int(k): v for k, v in subj.items()}))
    if not out:
        raise SystemExit("no kwcoco files in {}".format(splits_dir))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--dst-split', required=True)
    ap.add_argument('--src-dir', action='append', required=True,
                    help='feature dir to draw rows from (repeatable), relative to ROOT or absolute')
    ap.add_argument('--dataset', default='combined_brc1_brc2')
    ap.add_argument('--dst-source', default='', help='source label for the new cache (default: keep)')
    ap.add_argument('--dry-run', action='store_true', help='report coverage, write nothing')
    args = ap.parse_args()

    roles = target_index(paths.splits_dir(args.dst_split))
    want = {}                                   # crop rel -> (role idx, position)
    for ri, (_, _, rels, _, _, _) in enumerate(roles):
        for pos, r in enumerate(rels):
            want[r] = (ri, pos)
    print("target {}: {} roles, {:,} crops".format(args.dst_split, len(roles), len(want)))

    src_dirs = [d if osp.isabs(d) else osp.join(paths.ROOT, d) for d in args.src_dir]
    meta, dim = None, None
    filled = [None] * len(roles)
    seen = 0

    for sd in src_dirs:
        for f in sorted(glob.glob(osp.join(sd, '*.npz'))):
            z = np.load(f, allow_pickle=False)
            if meta is None:
                meta = {k: z[k] for k in SCALARS if k in z.files}
                dim = int(z['feat_dim'])
                for ri, (_, _, rels, _, _, _) in enumerate(roles):
                    filled[ri] = (np.zeros((len(rels), dim), np.float32),
                                  np.zeros(len(rels), bool))
            elif int(z['feat_dim']) != dim:
                raise SystemExit("{}: feat_dim {} != {}".format(f, int(z['feat_dim']), dim))
            rel = z['crop_path']
            feats = z['feats']
            hit = 0
            for i, r in enumerate(rel):
                t = want.get(str(r))
                if t is None:
                    continue
                ri, pos = t
                filled[ri][0][pos] = feats[i]
                filled[ri][1][pos] = True
                hit += 1
            seen += hit
            print("   {:<58} {:>9,} rows, {:>9,} used".format(
                osp.join(osp.basename(osp.dirname(osp.dirname(sd))), osp.basename(f)), len(rel), hit))
            del feats, rel, z

    missing = sum(int((~ok).sum()) for _, ok in filled)
    print("pooled {:,} matching rows; {:,} target crops unmatched".format(seen, missing))
    if missing:
        for (stem, _, rels, _, _, _), (_, ok) in zip(roles, filled):
            n = int((~ok).sum())
            if n:
                ex = [rels[i] for i in np.nonzero(~ok)[0][:2]]
                print("   {}: {:,} missing, e.g. {}".format(stem, n, ex))
        raise SystemExit("target is NOT covered by the source caches -- extract this split instead")

    if args.dry_run:
        print("dry run: nothing written")
        return

    out_dir = osp.join(paths.ROOT, 'eval', args.dataset, args.dst_split,
                       str(meta['method']), args.dst_source or str(meta['source']))
    os.makedirs(out_dir, exist_ok=True)
    for (stem, token, rels, pids, cams, subj), (F, _) in zip(roles, filled):
        name = ('gallery_{}.npz' if stem == 'gallery' else 'query_{}.npz').format(token)
        out = dict(meta)
        out.update(feats=F, pid=pids, camid=cams,
                   crop_path=np.asarray(rels),
                   video=np.asarray([osp.dirname(r) for r in rels]),
                   dive_track_id=np.full(len(rels), -1, np.int64),
                   frame=np.asarray([int(osp.splitext(osp.basename(r))[0]) for r in rels], np.int64),
                   split_role=np.asarray(stem if stem == 'gallery' else 'query'),
                   camera_token=np.asarray(token),
                   split=np.asarray(args.dst_split),
                   source=np.asarray(args.dst_source or str(meta['source'])),
                   pid_to_subject=np.asarray(json.dumps(
                       {int(p): subj[int(p)] for p in set(pids.tolist())})))
        p = osp.join(out_dir, name)
        np.savez_compressed(p, **out)
        print("wrote {}  {:,} x {}".format(p, len(rels), dim))


if __name__ == '__main__':
    main()
