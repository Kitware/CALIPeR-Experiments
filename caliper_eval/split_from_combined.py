"""Build crop-level gallery and query splits from a video split and its chip index."""
import argparse
import json
import os
import os.path as osp
from collections import OrderedDict

# Root of the released CALIPeR dataset tree (splits/, reid-chips/, ...). Override with CALIPER_DATA_ROOT.
_DATA_ROOT = os.environ.get('CALIPER_DATA_ROOT', 'data')
DEFAULT_CHIP_INDEX = osp.join(_DATA_ROOT, 'reid-chips', 'chip_index.jsonl')
DEFAULT_CROP_ROOT = osp.join(_DATA_ROOT, 'reid-chips', 'v4')


def load_chip_index(path=DEFAULT_CHIP_INDEX):
    """Index usable chip records by source video path."""
    idx = {}
    n_total = n_bad = 0
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            n_total += 1
            if r.get('status') != 'ok' or not r.get('chips'):
                n_bad += 1
                continue
            idx[r['relpath']] = r
    print("chip index: {} records, {} usable, {} dropped (no confident track)".format(
        n_total, len(idx), n_bad))
    return idx


def camera_of(rec):
    """Return the physical camera identifier and the range or role label."""
    phys = rec['relpath'].split('/')[-2]
    return phys, (rec.get('tok') or phys)


def build_role(recs, subject_ids, camera_ids, crop_root, forced_token, role_camid=None):
    """Build one crop-level split. Gallery and query roles share the subject and camera ID
    mappings."""
    out = dict(info=dict(crop_root=crop_root), subjects=[], videos=[], tracks=[], images=[], annotations=[])
    vid_id = trk_id = img_id = ann_id = 0
    for rec in recs:
        canon = rec['canon']
        if canon not in subject_ids:
            subject_ids[canon] = len(subject_ids) + 1
        phys, tok = camera_of(rec)
        if phys not in camera_ids:
            camera_ids[phys] = len(camera_ids) + 1

        vid_id += 1
        trk_id += 1
        # Use the role label for splits spanning multiple cameras.
        out['videos'].append(dict(id=vid_id, name=rec['relpath'],
                                  camera_id=role_camid if role_camid is not None else camera_ids[phys],
                                  camera_token=forced_token or tok))
        out['tracks'].append(dict(id=trk_id, subject_id=subject_ids[canon],
                                  dive_track_id=int(rec['track_id']) if rec.get('track_id') is not None else -1))

        stem = rec['relpath'].rsplit('.', 1)[0]
        for frame in rec['frames']:
            img_id += 1
            ann_id += 1
            out['images'].append(dict(id=img_id,
                                      file_name=osp.join(crop_root, stem, '{:06d}.jpg'.format(int(frame))),
                                      video_id=vid_id, frame_index=int(frame)))
            out['annotations'].append(dict(id=ann_id, image_id=img_id, track_id=trk_id))
    return out


def finalize_subjects(role_docs, subject_ids):
    """Write the SAME subjects registry into every role doc of the split."""
    subs = [dict(id=i, name=n) for n, i in sorted(subject_ids.items(), key=lambda kv: kv[1])]
    for doc in role_docs:
        doc['subjects'] = subs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--split-json', required=True, help='video-level split manifest')
    ap.add_argument('--out-dir', required=True, help='directory to emit gallery/query kwcoco into')
    ap.add_argument('--chip-index', default=DEFAULT_CHIP_INDEX,
                    help="Crop index (default: $CALIPER_DATA_ROOT/reid-chips/chip_index.jsonl).")
    ap.add_argument('--crop-root', default=DEFAULT_CROP_ROOT,
                    help="Root directory for the indexed crops.")
    ap.add_argument('--gallery-token', default='gallery', help="Label for the gallery in saved features.")
    ap.add_argument('--query-token', default='query', help="Label for the query file, query_<label>.kwcoco.json.")
    ap.add_argument('--query-token-per-tok', action='store_true',
                    help="Write one query file per distance. Use this for the 500m and 1000m paperGR queries.")
    ap.add_argument('--camid-by-role', action='store_true',
                    help="Assign camera ID 1 to the gallery and 2 to queries. Use this when a protocol "
                         "treats each role as a camera. For paper clothing scores, keep the physical "
                         "IDs and score with --cam-exclusion none.")
    ap.add_argument('--mated-only', action='store_true',
                    help="Skip queries whose subject is absent from the gallery. Keep all gallery subjects. "
                         "Leave this off for open-set evaluation, which needs unmatched queries.")
    ap.add_argument('--max-videos-per-role', type=int, default=0,
                    help="Limit videos in each role for a quick trial. Zero uses all videos.")

    args = ap.parse_args()

    idx = load_chip_index(args.chip_index)
    split = json.load(open(args.split_json))
    os.makedirs(args.out_dir, exist_ok=True)

    subject_ids, camera_ids = OrderedDict(), OrderedDict()
    docs, written = [], []
    # smoke mode: restrict to subjects present in BOTH roles, so the mini split still has matches to score
    keep_subjects = None
    if args.max_videos_per_role:
        keep_subjects = sorted(set(split['gallery']) & set(split['query']))

    mated = set(split['gallery']) & set(split['query'])
    for role, token in (('gallery', args.gallery_token), ('query', args.query_token)):
        if keep_subjects is None:
            if args.mated_only and role == 'query':
                listed = [p for subj, ps in split[role].items() if subj in mated for p in ps]
                print("mated-only: query restricted to {} of {} subjects".format(
                    len(mated), len(split['query'])))
            else:
                listed = [p for subj, ps in split[role].items() for p in ps]
        else:
            listed = []
            for subj in keep_subjects:                      # whole subjects, in order, until the cap
                if len(listed) >= args.max_videos_per_role:
                    break
                listed.extend(split[role].get(subj, []))
            listed = listed[:args.max_videos_per_role]
        recs = [idx[p] for p in listed if p in idx]
        missing = len(listed) - len(recs)
        # sort for determinism: identical inputs must produce byte-identical kwcoco
        recs.sort(key=lambda r: r['relpath'])

        if role == 'query' and args.query_token_per_tok:
            # one file per range bucket, all sharing this invocation's subject/camera registries
            by_tok = OrderedDict()
            for rec in recs:                       # recs already sorted -> buckets are deterministic
                by_tok.setdefault(camera_of(rec)[1], []).append(rec)
            n_listed_each = {t: len(v) for t, v in by_tok.items()}
            for tk in sorted(by_tok):
                doc = build_role(by_tok[tk], subject_ids, camera_ids, args.crop_root, tk,
                                 role_camid=2 if args.camid_by_role else None)
                docs.append(doc)
                written.append((osp.join(args.out_dir, 'query_{}.kwcoco.json'.format(tk)), doc,
                                'query/{}'.format(tk), n_listed_each[tk], 0))
            if missing:
                print("query: {} of {} listed videos unusable/absent (dropped before bucketing)".format(
                    missing, len(listed)))
            continue

        doc = build_role(recs, subject_ids, camera_ids, args.crop_root, token,
                         role_camid=(1 if role == 'gallery' else 2) if args.camid_by_role else None)
        docs.append(doc)
        fname = 'gallery.kwcoco.json' if role == 'gallery' else 'query_{}.kwcoco.json'.format(token)
        written.append((osp.join(args.out_dir, fname), doc, role, len(listed), missing))

    finalize_subjects(docs, subject_ids)

    for path, doc, role, n_listed, missing in written:
        with open(path, 'w') as f:
            json.dump(doc, f)
        print("[{}] {} videos listed, {} unusable/absent -> {} videos, {} crops, {} subjects\n    {}".format(
            role, n_listed, missing, len(doc['videos']), len(doc['images']),
            len({t['subject_id'] for t in doc['tracks']}), path))
    print("subjects registry: {} ids shared across both roles; {} distinct physical cameras".format(
        len(subject_ids), len(camera_ids)))


if __name__ == '__main__':
    main()
