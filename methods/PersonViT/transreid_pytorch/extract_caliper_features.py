# encoding: utf-8
"""Extract PersonViT crop features for the CALIPeR gallery and query sets."""
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

# PersonViT repo (this file lives in transreid_pytorch/)
REPO = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, REPO)
from config import cfg                       # noqa: E402
from model.make_model import make_model      # noqa: E402

# shared eval package
try:
    from caliper_eval import paths, features_io
except ImportError:
    # fall back to the monorepo root (the directory containing caliper_eval/)
    _REPO_ROOT = osp.abspath(osp.join(REPO, '..', '..', '..'))
    sys.path.insert(0, os.environ.get('CALIPER_EVAL_HOME', _REPO_ROOT))
    from caliper_eval import paths, features_io


def read_image(path):
    return Image.open(path).convert('RGB')


class CropDataset(Dataset):
    def __init__(self, crop_paths, transform):
        self.paths = crop_paths
        self.t = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.t(read_image(self.paths[i])), self.paths[i]


def load_param_report(model, path):
    """Load checkpoint tensors by name and shape and report missing or mismatched weights."""
    sd = model.state_dict()
    ckpt = torch.load(path, map_location='cpu')
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    loaded, skipped = set(), []
    for k, v in ckpt.items():
        if k in sd and hasattr(v, 'shape') and sd[k].shape == v.shape:
            sd[k].copy_(v); loaded.add(k)
        else:
            skipped.append(k)
    ckpt_cov = len(loaded) / max(len(ckpt), 1)
    # classifier head is intentionally unfilled (we build num_class=1); everything else should load
    unfilled = [k for k in sd if k not in loaded and not k.startswith(('classifier', 'classifier_'))]
    print("load: matched {}/{} ckpt tensors ({:.0%}); model params unfilled (excl. classifier): {}".format(
        len(loaded), len(ckpt), ckpt_cov, len(unfilled)))
    if ckpt_cov < 0.9 or unfilled:
        print("  WARNING: likely CONFIG<->checkpoint mismatch. unfilled e.g.: {} ; skipped ckpt e.g.: {}".format(
            unfilled[:6], skipped[:6]))
    return ckpt_cov, len(unfilled)


def extract(model, crop_paths, transform, device, batch_size, num_workers):
    uniq = sorted(set(crop_paths))
    ds = CropDataset(uniq, transform)

    def collate(b):
        imgs, ps = zip(*b)
        return torch.stack(imgs, 0), list(ps)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate)
    feats = {}
    model.eval()
    with torch.no_grad():
        for i, (imgs, ps) in enumerate(loader):
            f = model(imgs.to(device), cam_label=None, view_label=None).cpu().numpy().astype(np.float32)
            for p, fi in zip(ps, f):
                feats[p] = fi
            if (i + 1) % 50 == 0:
                print('  extracted {}/{} batches'.format(i + 1, len(loader)), flush=True)
    return feats


def main():
    ap = argparse.ArgumentParser(description="PersonViT feature extractor (shared .npz format)")
    ap.add_argument("--config-file", required=True, help="PersonViT config matching the checkpoint")
    ap.add_argument("--weight", required=True, help="PersonViT ReID checkpoint (HF lakeAGI/PersonViTReID)")
    ap.add_argument("--source", required=True, help="pretraining source domain (e.g. market1501, msmt17)")
    ap.add_argument("--method", default="personvit")
    ap.add_argument("--split", default="paperGR")
    ap.add_argument("--dataset", default="combined_brc1_brc2",
                    help="eval dataset the split belongs to (kitbrc1 for ground_range_sim)")
    ap.add_argument("--splits-dir", default="", help="Crop split directory (default: $REID_BENCH_ROOT/splits/<split>).")
    ap.add_argument("--out-dir", default="",
                    help="Where to save features (default: $REID_BENCH_ROOT/eval/<dataset>/<split>/<method>/<source>).")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("opts", default=[], nargs=argparse.REMAINDER)
    args = ap.parse_args()

    cfg.merge_from_file(args.config_file)
    # skip loading the self-supervised backbone at build; the full ReID checkpoint is loaded below
    cfg.merge_from_list(['MODEL.PRETRAIN_CHOICE', 'none'])
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    transform = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
    ])

    model = make_model(cfg, num_class=1, camera_num=1, view_num=1)  # SIE off -> camera/view unused
    cov, n_unfilled = load_param_report(model, args.weight)   # both-direction coverage; catches CONFIG mismatch
    if cov < 0.5 or n_unfilled > 5:
        raise SystemExit("aborting: load mismatch (ckpt coverage {:.0%}, {} model params unfilled) -- wrong "
                         "--config-file for this checkpoint? (e.g. vit_small.yml vs vit_base.yml)".format(cov, n_unfilled))
    model = model.to(args.device)

    splits_dir = args.splits_dir or paths.splits_dir(args.split)
    out_dir = args.out_dir or paths.features_dir(args.split, args.method, args.source, dataset=args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    for role, coco, fname_token in features_io.discover_role_files(splits_dir):
        recs, internal_token, crop_root = features_io.load_crops_full(coco, role)
        camera_token = fname_token or internal_token   # queries: token from filename; gallery: from kwcoco
        crop_paths = [r['path'] for r in recs]
        print("[{}] {} crops ({} unique) range={}; extracting...".format(
            role, len(recs), len(set(crop_paths)), camera_token), flush=True)
        feat_of_path = extract(model, crop_paths, transform, args.device, args.batch_size, args.num_workers)
        feats = np.stack([feat_of_path[p] for p in crop_paths]).astype(np.float32)
        out = osp.join(out_dir, features_io.feature_filename(role, camera_token))
        features_io.write_features(out, feats, recs, crop_root=crop_root, split_role=role,
                                   camera_token=camera_token, neck_feat=cfg.TEST.NECK_FEAT, weight=args.weight,
                                   split=args.split, method=args.method, source=args.source)
        print("  wrote {}  (feats {}, {:.1f} MB)".format(out, feats.shape, os.path.getsize(out) / 1e6))


if __name__ == "__main__":
    main()
