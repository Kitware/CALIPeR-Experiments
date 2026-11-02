# encoding: utf-8
"""Extract CAL crop features for the CALIPeR gallery and query sets."""
import argparse
import os
import os.path as osp
import sys

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

# GEFF repo root (this file lives at the repo root). CAL's internal imports are absolute
# (`from ReIDModules.CAL...`), so the ROOT -- not ReIDModules/CAL -- must be on sys.path.
REPO = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, REPO)
from ReIDModules.CAL.configs.default_img import _C as cal_cfg   # noqa: E402
from ReIDModules.CAL.models.img_resnet import ResNet50          # noqa: E402

# shared eval package
try:
    from caliper_eval import paths, features_io
except ImportError:
    sys.path.insert(0, osp.expanduser(os.environ.get('CALIPER_EVAL_HOME', osp.abspath(osp.join(osp.dirname(__file__), '../../')))))
    from caliper_eval import paths, features_io


def parse_args():
    ap = argparse.ArgumentParser(description="CAL feature extractor (shared .npz format)")
    ap.add_argument("--weight", default=osp.join(REPO, "weights/CAL/last-checkpoint.pth.tar"),
                    help="CAL checkpoint .pth.tar (dict with 'model_state_dict')")
    ap.add_argument("--source", required=True,
                    help="training dataset label: last | ltcc | prcc | vc-clothes")
    ap.add_argument("--method", default="cal")
    ap.add_argument("--split", default="ground_range_sim")
    ap.add_argument("--splits-dir", default="", help="default: paths.splits_dir(split)")
    ap.add_argument("--out-dir", default="", help="default: paths.features_dir(split, method, source, dataset)")
    ap.add_argument("--dataset", default="kitbrc1",
                    help="dataset tree under reid-bench/eval/ (e.g. combined_brc1_brc2)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-flip", action="store_true", help='Disable horizontal-flip augmentation at evaluation.')
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()
    return args


class CropDataset(Dataset):
    """One item == one crop image. CAL is an image model, so this is a plain per-crop dataset."""

    def __init__(self, paths_list, transform):
        self.paths = paths_list
        self.t = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        return self.t(Image.open(p).convert('RGB')), p


def build_test_transform(cfg):
    """Build the preprocessing transform used by the released checkpoint."""
    return T.Compose([
        T.Resize((cfg.DATA.HEIGHT, cfg.DATA.WIDTH)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_backbone(weight, cfg, device):
    """Load pretrained backbone weights and report coverage, skipping the classifier when
    transferring between datasets."""
    model = ResNet50(cfg)
    ckpt = torch.load(weight, map_location='cpu', weights_only=False)
    sd_ck = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    sd_ck = {k.replace('module.', ''): v for k, v in sd_ck.items()}
    sd = model.state_dict()
    loaded, skipped = set(), []
    for k, v in sd_ck.items():
        if k in sd and hasattr(v, 'shape') and sd[k].shape == v.shape:
            sd[k].copy_(v)
            loaded.add(k)
        else:
            skipped.append("{}{}".format(k, tuple(v.shape) if hasattr(v, 'shape') else '?'))
    unfilled = [k for k in sd if k not in loaded]
    print("load: matched {}/{} backbone tensors ({:.0%}); model params unfilled: {}".format(
        len(loaded), len(sd_ck), len(loaded) / max(len(sd_ck), 1), len(unfilled)))
    if skipped or unfilled:
        print("  WARNING skipped e.g. {} ; unfilled e.g. {}".format(skipped[:6], unfilled[:6]))
    if loaded != set(sd.keys()):
        raise SystemExit("aborting: backbone/checkpoint mismatch ({} of {} model params filled) -- wrong "
                         "checkpoint or config?".format(len(loaded), len(sd)))
    ep, r1 = ckpt.get('epoch'), ckpt.get('rank1')
    print("  checkpoint provenance: epoch={} rank1_on_its_own_testset={}".format(ep, r1))
    return model.to(device).eval()


@torch.no_grad()
def extract(model, crop_paths, transform, args):
    """Extract features for unique crop paths and return them in crop-record order."""
    uniq = sorted(set(crop_paths))
    loader = DataLoader(CropDataset(uniq, transform), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True,
                        collate_fn=lambda b: (torch.stack([x[0] for x in b], 0), [x[1] for x in b]))
    feats = {}
    for i, (imgs, ps) in enumerate(loader):
        imgs = imgs.to(args.device, non_blocking=True)
        f = model(imgs)
        if not args.no_flip:
            f = torch.stack([f, model(torch.flip(imgs, [3]))], dim=0).mean(dim=0)
        f = f.float().cpu().numpy().astype(np.float32)
        for p, fi in zip(ps, f):
            feats[p] = fi
        if (i + 1) % 50 == 0:
            print('  extracted {}/{} batches'.format(i + 1, len(loader)), flush=True)
    return feats


def main():
    args = parse_args()
    cal_cfg.defrost()
    cal_cfg.freeze()
    print("building CAL ResNet50: {}x{}  feature_dim={}  pooling={}".format(
        cal_cfg.DATA.HEIGHT, cal_cfg.DATA.WIDTH, cal_cfg.MODEL.FEATURE_DIM, cal_cfg.MODEL.POOLING.NAME))
    model = load_backbone(args.weight, cal_cfg, args.device)
    transform = build_test_transform(cal_cfg)

    splits_dir = args.splits_dir or paths.splits_dir(args.split)
    out_dir = args.out_dir or paths.features_dir(args.split, args.method, args.source, dataset=args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    neck_tag = "after_bn_{}{}".format(cal_cfg.MODEL.FEATURE_DIM, '' if args.no_flip else '_hflip')

    for role, coco, fname_token in features_io.discover_role_files(splits_dir):
        recs, internal_token, crop_root = features_io.load_crops_full(coco, role)
        camera_token = fname_token or internal_token   # queries: token from filename; gallery: from kwcoco
        crop_paths = [r['path'] for r in recs]
        print("[{}] {} crops ({} unique) range={}; extracting...".format(
            role, len(recs), len(set(crop_paths)), camera_token), flush=True)

        feat_of_path = extract(model, crop_paths, transform, args)
        feats = np.stack([feat_of_path[p] for p in crop_paths]).astype(np.float32)
        out = osp.join(out_dir, features_io.feature_filename(role, camera_token))
        # pooling='' -- per-crop features, supports image and video pooling
        features_io.write_features(out, feats, recs, crop_root=crop_root, split_role=role,
                                   camera_token=camera_token, neck_feat=neck_tag, weight=args.weight,
                                   split=args.split, method=args.method, source=args.source)
        print("  wrote {}  (feats {}, {:.1f} MB)".format(out, feats.shape, os.path.getsize(out) / 1e6))


if __name__ == "__main__":
    main()
