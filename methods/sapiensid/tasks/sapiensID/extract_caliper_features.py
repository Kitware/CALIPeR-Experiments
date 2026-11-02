# encoding: utf-8
"""Extract SapiensID crop features for the CALIPeR gallery and query sets."""
import argparse
import os
import os.path as osp
import sys

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset

np.bool = np.bool_      # compat shims some sub-deps (mxnet/older) still expect
np.object = np.object_
ImageFile.LOAD_TRUNCATED_IMAGES = True

# repo layout: this file is in tasks/sapiensID/ ; `src` lives here, `general_utils` at the repo root
TASK = osp.dirname(osp.abspath(__file__))
REPO = osp.dirname(osp.dirname(TASK))
for _p in (TASK, REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)
try:                                   # load .env (HF_TOKEN for the HF face-detector fallback)
    from dotenv import load_dotenv
    load_dotenv(osp.join(REPO, '.env'))
except Exception:
    pass

from src.models import get_model                 # noqa: E402
from src.aligners import get_aligner             # noqa: E402
from src.pipelines import pipeline_from_name     # noqa: E402
from general_utils.config_utils import load_config  # noqa: E402

try:
    from caliper_eval import paths, features_io
except ImportError:
    sys.path.insert(0, osp.expanduser(os.environ.get('CALIPER_EVAL_HOME', osp.abspath(osp.join(osp.dirname(__file__), '../../../../')))))
    from caliper_eval import paths, features_io


class CropDataset(Dataset):
    def __init__(self, crop_paths, transform):
        self.paths = crop_paths
        self.t = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.t(Image.open(self.paths[i]).convert('RGB')), self.paths[i]


def extract(pipeline, crop_paths, transform, device, batch_size, num_workers):
    uniq = sorted(set(crop_paths))
    ds = CropDataset(uniq, transform)

    def collate(b):
        imgs, ps = zip(*b)
        return torch.stack(imgs, 0), list(ps)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate)
    feats = {}
    with torch.no_grad():
        for i, (imgs, ps) in enumerate(loader):
            f = pipeline(imgs.to(device)).cpu().numpy().astype(np.float32)
            for p, fi in zip(ps, f):
                feats[p] = fi
            if (i + 1) % 20 == 0:
                print('  extracted {}/{} batches'.format(i + 1, len(loader)), flush=True)
    return feats


def main():
    ap = argparse.ArgumentParser(description="SapiensID feature extractor (shared .npz format)")
    ap.add_argument("--ckpt-dir", required=True, help="dir with model.yaml + model.pth (e.g. checkpoints/sapiensid_wb12m)")
    ap.add_argument("--source", required=True, help="checkpoint/source label (e.g. wb12m, wb4m)")
    ap.add_argument("--aligner", default="yolo_dfa.yaml", help="aligner config under src/aligners/configs/")
    ap.add_argument("--method", default="sapiensid")
    ap.add_argument("--split", default="ground_range")
    ap.add_argument("--splits-dir", default="", help="default: paths.splits_dir(split)")
    ap.add_argument("--out-dir", default="", help="default: paths.features_dir(split, method, source)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()

    # build model + aligner + pipeline exactly like tasks/sapiensID/eval.py (single-GPU, no Fabric)
    model_config = load_config(osp.join(args.ckpt_dir, 'model.yaml'))
    model = get_model(model_config)
    model.load_state_dict_from_path(osp.join(args.ckpt_dir, 'model.pth'))

    aligner_config = load_config(osp.join(TASK, 'src', 'aligners', 'configs', args.aligner))
    aligner_config.rgb_mean = model_config.rgb_mean      # resolve the ${models.rgb_mean} interpolation
    aligner_config.rgb_std = model_config.rgb_std
    aligner = get_aligner(aligner_config)

    pipeline = pipeline_from_name('infer_aligner_model_pipeline', model, aligner)
    transform = pipeline.make_test_transform()

    model = model.to(args.device).eval()
    if aligner is not None:
        aligner = aligner.to(args.device)
    pipeline.eval()

    splits_dir = args.splits_dir or paths.splits_dir(args.split)
    out_dir = args.out_dir or paths.features_dir(args.split, args.method, args.source)
    os.makedirs(out_dir, exist_ok=True)

    for role, coco, fname_token in features_io.discover_role_files(splits_dir):
        recs, internal_token, crop_root = features_io.load_crops_full(coco, role)
        camera_token = fname_token or internal_token   # queries: token from filename; gallery: from kwcoco
        crop_paths = [r['path'] for r in recs]
        print("[{}] {} crops ({} unique) range={}; extracting...".format(
            role, len(recs), len(set(crop_paths)), camera_token), flush=True)
        feat_of_path = extract(pipeline, crop_paths, transform, args.device, args.batch_size, args.num_workers)
        feats = np.stack([feat_of_path[p] for p in crop_paths]).astype(np.float32)
        out = osp.join(out_dir, features_io.feature_filename(role, camera_token))
        features_io.write_features(out, feats, recs, crop_root=crop_root, split_role=role,
                                   camera_token=camera_token, neck_feat='final_feat',
                                   weight=osp.join(args.ckpt_dir, 'model.pth'),
                                   split=args.split, method=args.method, source=args.source)
        print("  wrote {}  (feats {}, {:.1f} MB)".format(out, feats.shape, os.path.getsize(out) / 1e6))


if __name__ == "__main__":
    main()
