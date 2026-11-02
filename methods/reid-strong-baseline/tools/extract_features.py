# encoding: utf-8
"""Extract BoT crop features for the CALIPeR gallery and query sets."""
import argparse
import os
import os.path as osp
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append('.')
from config import cfg                                  # noqa: E402
from modeling import build_model                        # noqa: E402
from data.transforms import build_transforms            # noqa: E402
from data.datasets.dataset_loader import ImageDataset   # noqa: E402
from caliper_eval.train_entry import load_backbone         # noqa: E402


class _StdoutLogger(object):
    """load_backbone() wants a logger; this script prints. Adapt rather than configure logging."""

    def info(self, msg):
        print(msg, flush=True)


try:
    from caliper_eval import paths, features_io
except ImportError:
    sys.path.insert(0, osp.expanduser(os.environ.get('CALIPER_EVAL_HOME', osp.abspath(osp.join(osp.dirname(__file__), '../../../')))))
    from caliper_eval import paths, features_io


def extract(model, crop_paths, transforms, device, batch_size, num_workers):
    """Extract features for unique crop paths and return them in crop-record order."""
    uniq = sorted(set(crop_paths))
    ds = ImageDataset([(p, 0, 0) for p in uniq], transforms)

    def collate(b):
        imgs, _pids, _camids, ps = zip(*b)
        return torch.stack(imgs, 0), list(ps)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate)
    feats = {}
    model.eval()
    with torch.no_grad():
        for i, (imgs, ps) in enumerate(loader):
            f = model(imgs.to(device)).cpu().numpy().astype(np.float32)
            for p, fi in zip(ps, f):
                feats[p] = fi
            if (i + 1) % 50 == 0:
                print('  extracted {}/{} batches'.format(i + 1, len(loader)), flush=True)
    return feats


def main():
    parser = argparse.ArgumentParser(description="BoT extract-once feature cache (shared .npz format)")
    parser.add_argument("--config_file", default="", type=str)
    parser.add_argument("--split", default="ground_range", type=str)
    parser.add_argument("--method", default="bot", type=str, help="method name (output namespace + provenance)")
    parser.add_argument("--source", default="market1501", type=str, help="pretraining source domain")
    parser.add_argument("--splits-dir", default="", type=str, help="default: paths.splits_dir(split)")
    parser.add_argument("--out-dir", default="", type=str,
                        help="default: paths.features_dir(method, source, split)")
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

    splits_dir = args.splits_dir or paths.splits_dir(args.split)
    out_dir = args.out_dir or paths.features_dir(args.split, args.method, args.source)
    os.makedirs(out_dir, exist_ok=True)

    # num_classes only sizes the classifier head, which is skipped on load either way.
    model = build_model(cfg, 100)

    # Check each backbone tensor before extracting features.
    load_backbone(model, cfg.TEST.WEIGHT, _StdoutLogger(), skip=('classifier',))
    model = model.to(device)

    for role, coco, fname_token in features_io.discover_role_files(splits_dir):
        recs, internal_token, crop_root = features_io.load_crops_full(coco, role)
        camera_token = fname_token or internal_token   # queries: token from filename; gallery: from kwcoco
        crop_paths = [r['path'] for r in recs]
        print("[{}] {} crops ({} unique) range={}; extracting...".format(
            role, len(recs), len(set(crop_paths)), camera_token), flush=True)
        feat_of_path = extract(model, crop_paths, transforms, device,
                               cfg.TEST.IMS_PER_BATCH, cfg.DATALOADER.NUM_WORKERS)
        feats = np.stack([feat_of_path[p] for p in crop_paths]).astype(np.float32)
        out = osp.join(out_dir, features_io.feature_filename(role, camera_token))
        features_io.write_features(out, feats, recs, crop_root=crop_root, split_role=role,
                                   camera_token=camera_token, neck_feat=cfg.TEST.NECK_FEAT,
                                   weight=cfg.TEST.WEIGHT, split=args.split,
                                   method=args.method, source=args.source)
        print("  wrote {}  (feats {}, {:.1f} MB)".format(out, feats.shape, os.path.getsize(out) / 1e6))


if __name__ == '__main__':
    main()
