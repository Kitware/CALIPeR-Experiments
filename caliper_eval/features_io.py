# encoding: utf-8
"""Read crop splits and store features, subject IDs, camera IDs, and frame locations in NumPy
archives."""
import glob
import json
import os.path as osp

import numpy as np

# the standard kwcoco split files under <splits_dir>/<split>/ (discover_role_files() also picks up any
# other query_*.kwcoco.json present)
SPLIT_FILES = {
    'gallery': 'gallery.kwcoco.json',
    'query_500m': 'query_500m.kwcoco.json',
    'query_1000m': 'query_1000m.kwcoco.json',
}


def feature_filename(role, camera_token):
    """Return the archive name for a gallery or query camera token."""
    base = 'gallery' if role == 'gallery' else 'query'
    return "{}_{}.npz".format(base, camera_token)


def load_crops_full(coco_path, split_role):
    """Return crop records with absolute paths, the camera token, and the crop root."""
    with open(coco_path) as f:
        d = json.load(f)
    crop_root = d.get('info', {}).get('crop_root', '')
    subj_name = {s['id']: s['name'] for s in d['subjects']}
    cam_of_video = {v['id']: v['camera_id'] for v in d['videos']}
    tok_of_video = {v['id']: v['camera_token'] for v in d['videos']}
    stem_of_video = {v['id']: v['name'] for v in d['videos']}
    track = {t['id']: t for t in d['tracks']}
    ann_of_image = {a['image_id']: a for a in d['annotations']}
    recs = []
    for im in d['images']:
        a = ann_of_image.get(im['id'])
        if a is None:
            continue
        t = track[a['track_id']]
        pid = t['subject_id']
        dt = t.get('dive_track_id')
        recs.append(dict(path=im['file_name'], subject=subj_name[pid], pid=int(pid),
                         camid=int(cam_of_video[im['video_id']]), video=stem_of_video[im['video_id']],
                         dive_track_id=int(dt) if dt is not None else -1,
                         frame=int(im.get('frame_index', 0))))
    tokens = {tok_of_video[v['id']] for v in d['videos']}
    camera_token = tokens.pop() if len(tokens) == 1 else '|'.join(sorted(tokens))
    return recs, camera_token, crop_root


def discover_role_files(splits_dir):
    """List (role, path, token) for gallery and query splits. Query filenames supply their
    tokens. The gallery uses its own camera metadata."""
    items = []
    g = osp.join(splits_dir, 'gallery.kwcoco.json')
    if osp.exists(g):
        items.append(('gallery', g, None))
    for f in sorted(glob.glob(osp.join(splits_dir, 'query_*.kwcoco.json'))):
        token = osp.basename(f)[len('query_'):-len('.kwcoco.json')]
        items.append(('query', f, token))
    if not items:
        raise FileNotFoundError("no gallery.kwcoco.json or query_*.kwcoco.json in {}".format(splits_dir))
    return items


def write_features(out_path, feats, recs, *, crop_root, split_role, camera_token,
                   neck_feat, weight, split, method, source, pooling=''):
    """Save the feature matrix and its matching crop records, in row order, for one gallery or
    query role."""
    feats = np.asarray(feats, dtype=np.float32)
    assert feats.ndim == 2 and len(feats) == len(recs), "feats must be (len(recs), D)"
    rel = [osp.relpath(r['path'], crop_root) if crop_root else r['path'] for r in recs]
    pid_to_subject = {int(r['pid']): r['subject'] for r in recs}
    np.savez_compressed(
        out_path,
        feats=feats,
        pid=np.asarray([r['pid'] for r in recs], dtype=np.int64),
        camid=np.asarray([r['camid'] for r in recs], dtype=np.int64),
        video=np.asarray([r['video'] for r in recs]),
        dive_track_id=np.asarray([r['dive_track_id'] for r in recs], dtype=np.int64),
        frame=np.asarray([r['frame'] for r in recs], dtype=np.int64),
        crop_path=np.asarray(rel),
        crop_root=np.asarray(crop_root),
        split_role=np.asarray(split_role),
        camera_token=np.asarray(camera_token),
        neck_feat=np.asarray(neck_feat),
        feat_dim=np.asarray(feats.shape[1]),
        weight=np.asarray(weight),
        split=np.asarray(split),
        method=np.asarray(method),
        source=np.asarray(source),
        pooling=np.asarray(pooling),
        pid_to_subject=np.asarray(json.dumps(pid_to_subject)),
    )
    return out_path


def load_npz(path):
    """Load one role's .npz into a plain dict of arrays."""
    if not osp.exists(path):
        raise FileNotFoundError("missing feature cache: {}".format(path))
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
