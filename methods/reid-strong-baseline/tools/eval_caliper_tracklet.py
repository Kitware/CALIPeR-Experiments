# encoding: utf-8
"""Evaluate BoT retrieval using mean-pooled crop features for each video and subject."""
import argparse
import json
import os
import os.path as osp
import sys
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append('.')
from config import cfg                                  # noqa: E402
from modeling import build_model                        # noqa: E402
from data.transforms import build_transforms            # noqa: E402
from data.datasets.dataset_loader import ImageDataset   # noqa: E402
from data.datasets.eval_reid import eval_func           # noqa: E402

try:
    from caliper_eval import paths                        # noqa: E402
except ImportError:
    _REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..', '..'))
    sys.path.insert(0, os.environ.get('CALIPER_EVAL_HOME', _REPO_ROOT))
    from caliper_eval import paths                        # noqa: E402

QUERY_CAMERAS = {'500m': 'query_500m.kwcoco.json', '1000m': 'query_1000m.kwcoco.json'}
GALLERY_FILE = 'gallery.kwcoco.json'


def load_crops(coco_path):
    """Return [(crop_path, pid, camid, video_id), ...] from one crop-level kwcoco split file."""
    with open(coco_path) as f:
        d = json.load(f)
    cam_of_video = {v['id']: v['camera_id'] for v in d['videos']}
    subj_of_track = {t['id']: t['subject_id'] for t in d['tracks']}
    ann_of_image = {a['image_id']: a for a in d['annotations']}
    recs = []
    for im in d['images']:
        a = ann_of_image.get(im['id'])
        if a is None:
            continue
        tid = a['track_id']
        recs.append((im['file_name'], subj_of_track[tid], cam_of_video[im['video_id']], im['video_id'], tid))
    return [(r[0], r[1], r[2], r[3]) for r in recs]


def extract_features(model, crops, transforms, device, batch_size, num_workers):
    """Return {crop_path: feature_tensor(cpu)} for the unique crop paths in `crops`."""
    uniq = list({c[0]: (c[0], c[1], c[2]) for c in crops}.values())   # dedupe by path
    ds = ImageDataset(uniq, transforms)

    def collate(b):
        imgs, pids, camids, paths = zip(*b)
        return torch.stack(imgs, 0), list(paths)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate)
    feats = {}
    model.eval()
    with torch.no_grad():
        for i, (imgs, paths) in enumerate(loader):
            f = model(imgs.to(device)).cpu()
            for p, fi in zip(paths, f):
                feats[p] = fi
            if (i + 1) % 50 == 0:
                print('  extracted {}/{} batches'.format(i + 1, len(loader)), flush=True)
    return feats


def pool_appearances(crops, feats, feat_norm='yes'):
    """Average crop features for each video and subject, with optional L2 normalization."""
    groups = defaultdict(list)
    meta = {}
    for path, pid, camid, vid in crops:
        if path in feats:
            groups[(vid, pid)].append(feats[path])
            meta[(vid, pid)] = (pid, camid)
    keys = sorted(groups)
    F = torch.stack([torch.stack(groups[k]).mean(0) for k in keys])
    if feat_norm == 'yes':
        F = torch.nn.functional.normalize(F, dim=1, p=2)
    pids = np.asarray([meta[k][0] for k in keys])
    camids = np.asarray([meta[k][1] for k in keys])
    return F, pids, camids


def eval_pair(qF, q_pids, q_camids, gF, g_pids, g_camids):
    m, n = qF.shape[0], gF.shape[0]
    distmat = (qF.pow(2).sum(1, keepdim=True) + gF.pow(2).sum(1, keepdim=True).t() - 2 * qF @ gF.t())
    cmc, mAP = eval_func(distmat.numpy(), q_pids, g_pids, q_camids, g_camids)
    return cmc, mAP


def main():
    parser = argparse.ArgumentParser(description="BRC1 video-averaging ReID")
    parser.add_argument("--config_file", default="", type=str)
    parser.add_argument("--split", default="ground_range", type=str)
    parser.add_argument("--splits-dir", default=paths.splits_root(), type=str,
                        help="dir holding <split>/{gallery,query_*}.kwcoco.json")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    device = 'cuda' if cfg.MODEL.DEVICE == 'cuda' else 'cpu'
    if device == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    transforms = build_transforms(cfg, is_train=False)

    split_dir = osp.join(args.splits_dir, args.split)
    gallery = load_crops(osp.join(split_dir, GALLERY_FILE))
    queries = {name: load_crops(osp.join(split_dir, fn))
               for name, fn in QUERY_CAMERAS.items()}
    print("loaded crops: gallery {} | {}".format(
        len(gallery), " | ".join("{} {}".format(n, len(q)) for n, q in queries.items())))

    num_classes = len({pid for _, pid, _, _ in gallery}) or 100
    model = build_model(cfg, num_classes)
    model.load_param(cfg.TEST.WEIGHT)
    model = model.to(device)

    all_crops = gallery + [c for q in queries.values() for c in q]
    print("extracting features for {} unique crops...".format(len({c[0] for c in all_crops})), flush=True)
    feats = extract_features(model, all_crops, transforms, device,
                             cfg.TEST.IMS_PER_BATCH, cfg.DATALOADER.NUM_WORKERS)

    gF, g_pids, g_camids = pool_appearances(gallery, feats, cfg.TEST.FEAT_NORM)
    print("gallery appearances: {} ({} subjects)".format(gF.shape[0], len(set(g_pids.tolist()))))

    results = {}
    qF_all, qp_all, qc_all = [], [], []
    for name, q in queries.items():
        qF, q_pids, q_camids = pool_appearances(q, feats, cfg.TEST.FEAT_NORM)
        cmc, mAP = eval_pair(qF, q_pids, q_camids, gF, g_pids, g_camids)
        results[name] = dict(n_query=int(qF.shape[0]), mAP=float(mAP),
                             rank1=float(cmc[0]), rank5=float(cmc[4]), rank10=float(cmc[9]))
        print("[{:>5}] appearances={:4d}  mAP={:.1%}  R1={:.1%}  R5={:.1%}  R10={:.1%}".format(
            name, qF.shape[0], mAP, cmc[0], cmc[4], cmc[9]))
        qF_all.append(qF); qp_all.append(q_pids); qc_all.append(q_camids)

    qF = torch.cat(qF_all); q_pids = np.concatenate(qp_all); q_camids = np.concatenate(qc_all)
    cmc, mAP = eval_pair(qF, q_pids, q_camids, gF, g_pids, g_camids)
    results['combined'] = dict(n_query=int(qF.shape[0]), mAP=float(mAP),
                               rank1=float(cmc[0]), rank5=float(cmc[4]), rank10=float(cmc[9]))
    print("[combined] appearances={:4d}  mAP={:.1%}  R1={:.1%}  R5={:.1%}  R10={:.1%}".format(
        qF.shape[0], mAP, cmc[0], cmc[4], cmc[9]))

    out_dir = cfg.OUTPUT_DIR or './logs/kitbrc1_market_tracklet'
    os.makedirs(out_dir, exist_ok=True)
    summary = dict(split=args.split,
                   neck_feat=cfg.TEST.NECK_FEAT, feat_norm=cfg.TEST.FEAT_NORM,
                   weight=cfg.TEST.WEIGHT, results=results)
    with open(osp.join(out_dir, 'modeB_results.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print("wrote", osp.join(out_dir, 'modeB_results.json'))


if __name__ == '__main__':
    main()
