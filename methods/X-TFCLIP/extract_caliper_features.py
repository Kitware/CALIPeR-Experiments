# encoding: utf-8
"""Extract X-TFCLIP tracklet features for the CALIPeR gallery and query sets."""
import argparse
import math
import os
import os.path as osp
import sys
from collections import defaultdict

import numpy as np
import torch
import torchvision.transforms.functional as VF
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Resolve X-TFCLIP’s config, model, and utility modules from its method directory.
REPO = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, REPO)
from config import cfg                                 # noqa: E402
from model.make_model_clipreid import make_model       # noqa: E402
import utils.seqtransforms as SeqT                     # noqa: E402

# shared eval package
try:
    from caliper_eval import paths, features_io
except ImportError:
    _REPO_ROOT = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)), '..', '..'))   # monorepo root
    sys.path.insert(0, osp.expanduser(os.environ.get('CALIPER_EVAL_HOME', _REPO_ROOT)))
    from caliper_eval import paths, features_io


def parse_args():
    ap = argparse.ArgumentParser(description="X-TFCLIP (video ReID) feature extractor (shared .npz format)")
    # NOTE: eval.py's default config (..._288x144_v2.yml) does not exist in this repo; this one does.
    ap.add_argument("--config-file", default=osp.join(REPO, "configs/vit_clipreid_288x144.yml"))
    ap.add_argument("--weight", default=osp.join(REPO, "weights/checkpoint_ep.pth.tar"))
    ap.add_argument("--source", required=True, help="pretraining source dataset label (e.g. agvpreid)")
    ap.add_argument("--method", default="x-tfclip")
    ap.add_argument("--split", default="paperGR")
    ap.add_argument("--splits-dir", default="", help="Crop split directory (default: $REID_BENCH_ROOT/splits/<split>).")
    ap.add_argument("--out-dir", default="", help="Where to save features (default: $REID_BENCH_ROOT/eval/<dataset>/<split>/<method>/<source>).")
    ap.add_argument("--dataset", default="combined_brc1_brc2",
                    help="dataset tree under eval/ (combined_brc1_brc2; kitbrc1 for ground_range_sim)")
    ap.add_argument("--device", default="cuda", help="must be cuda: make_model hardcodes clip_model.to('cuda')")
    ap.add_argument("--max-clips", type=int, default=0,
                    help='Maximum clips to average per tracklet. Zero uses all columns of the '
                         'sampling table and one uses the first column.')
    ap.add_argument("--pad", default="repeat", choices=["repeat", "wrap"],
                    help='Pad the final chunk by repeating the last frame (repeat) or cycling '
                         'through its frames (wrap).')
    ap.add_argument("--cam-id", type=int, default=0,
                    help='Camera index used for the SIE embedding on every clip.')
    ap.add_argument("--no-flip", action="store_true", help='Disable horizontal-flip augmentation at evaluation.')
    ap.add_argument("--batch-clips", type=int, default=16, help="clips per GPU batch (model is batch-invariant)")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("opts", default=[], nargs=argparse.REMAINDER)
    args = ap.parse_args()

    # vit_clipreid_288x144.yml merges cleanly (no None-section workaround needed, unlike CLIP-ReID's yml)
    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()
    return args


def build_test_transform(cfg):
    """Build the preprocessing transform used by the released checkpoint."""
    return SeqT.Compose([
        SeqT.RectScale(cfg.INPUT.SIZE_TRAIN[0], cfg.INPUT.SIZE_TRAIN[1]),
        SeqT.ToTensor(),
        SeqT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def rrs_chunks(num, seq_len, pad='repeat'):
    """Arrange ordered tracklet frames into rows for restricted random sampling, using the
    selected padding mode."""
    S = seq_len
    if num < S:
        inter_val = 1
        strip = list(range(num)) + [num - 1] * (S - num)
    else:
        inter_val = int(math.ceil(num / float(S)))
        strip = list(range(num)) + [num - 1] * (inter_val * S - num)
    table = np.asarray(strip, dtype=np.int64).reshape(S, inter_val)   # row s == their chunk s
    n_pad = len(strip) - num
    if pad == 'wrap' and n_pad > 0 and num >= S:
        real = inter_val - n_pad                       # real entries in the final chunk (>= 1)
        for k in range(real, inter_val):
            table[S - 1, k] = table[S - 1, k % real]
    return table


def clip_offsets(inter_val, max_clips):
    """Select columns of the sampling table. A nonpositive max_clips selects all columns."""
    if max_clips <= 0 or max_clips >= inter_val:
        return list(range(inter_val))
    if max_clips == 1:
        return [0]
    return sorted({int(round(j * (inter_val - 1) / (max_clips - 1))) for j in range(max_clips)})


class ClipDataset(Dataset):
    """Load each sampled clip with its tracklet index."""

    def __init__(self, tracklets, transform, seq_len, max_clips, pad):
        self.tracklets = tracklets       # list of (video_key, [abs frame paths, in frame order])
        self.t = transform
        self.items = []                  # (tracklet_idx, np.int64[seq_len] frame positions)
        for ti, (_, fps) in enumerate(tracklets):
            table = rrs_chunks(len(fps), seq_len, pad)          # (seq_len, inter_val)
            for j in clip_offsets(table.shape[1], max_clips):
                self.items.append((ti, table[:, j].copy()))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ti, idx = self.items[i]
        fps = self.tracklets[ti][1]
        imgs = [Image.open(fps[k]).convert('RGB') for k in idx]
        seq = self.t([imgs])[0]                                 # SeqT works on seqs[modality][frame]
        return ti, torch.stack(seq, dim=0)                      # (T, 3, H, W)


def collate_clips(batch):
    tis, clips = zip(*batch)
    return torch.as_tensor(tis, dtype=torch.long), torch.stack(clips, 0)   # (B,), (B,T,3,H,W)


def load_param_report(model, path):
    """Load checkpoint tensors by name and shape and report missing or mismatched weights."""
    sd = model.state_dict()
    ckpt = read_state_dict(path)
    loaded, skipped = set(), []
    for k, v in ckpt.items():
        if k in sd and hasattr(v, 'shape') and sd[k].shape == v.shape:
            sd[k].copy_(v)
            loaded.add(k)
        else:
            shp = tuple(v.shape) if hasattr(v, 'shape') else '?'
            skipped.append("{}{}".format(k, shp))
    unfilled = [k for k in sd if k not in loaded]
    print("load: matched {}/{} ckpt tensors ({:.0%}); model params unfilled: {}".format(
        len(loaded), len(ckpt), len(loaded) / max(len(ckpt), 1), len(unfilled)))
    if skipped or unfilled:
        print("  WARNING skipped ckpt e.g. {} ; unfilled e.g. {}".format(skipped[:6], unfilled[:6]))
    return len(loaded), len(ckpt), len(unfilled)


def read_state_dict(path):
    """Read checkpoint tensors, accepting a nested state dictionary and DataParallel prefixes."""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    return {k.replace('module.', ''): v for k, v in ckpt.items()}


def build_model(cfg, weight, device):
    """Construct the video model using the checkpoint’s class and camera dimensions."""
    ckpt = read_state_dict(weight)
    num_class = int(ckpt['classifier2.weight'].shape[0])     # 689 for the released checkpoint
    camera_num = int(ckpt['cv_embed'].shape[0])              # 6   for the released checkpoint
    seq_len_ck = int(ckpt['video_embed'].shape[0])           # 8   for the released checkpoint
    del ckpt
    if seq_len_ck != cfg.INPUT.SEQ_LEN:
        raise SystemExit("checkpoint video_embed has {} rows but cfg.INPUT.SEQ_LEN={} -- wrong config"
                         .format(seq_len_ck, cfg.INPUT.SEQ_LEN))
    if not str(device).startswith('cuda'):
        raise SystemExit("--device must be cuda: make_model hardcodes clip_model.to('cuda') "
                         "(model/make_model_clipreid.py:178). Select the GPU with CUDA_VISIBLE_DEVICES.")
    print("building X-TFCLIP {}: num_class={} camera_num={} seq_len={} size={} neck_feat={}".format(
        cfg.MODEL.NAME, num_class, camera_num, cfg.INPUT.SEQ_LEN,
        tuple(cfg.INPUT.SIZE_TRAIN), cfg.TEST.NECK_FEAT))
    model = make_model(cfg, num_class=num_class, camera_num=camera_num, view_num=1)
    n_load, n_ck, n_unfilled = load_param_report(model, weight)
    if n_load != n_ck or n_unfilled:
        raise SystemExit("aborting: checkpoint/model mismatch ({}/{} ckpt tensors loaded, {} model params "
                         "unfilled) -- wrong --config-file for this checkpoint?".format(n_load, n_ck, n_unfilled))
    # make_model leaves everything except image_encoder on CPU (only clip_model is moved) -> move the rest.
    return model.to(device).eval(), camera_num


@torch.no_grad()
def extract_tracklet_feats(model, tracklets, transform, args, n_tracklets):
    """Average the clip features for each tracklet."""
    ds = ClipDataset(tracklets, transform, cfg.INPUT.SEQ_LEN, args.max_clips, args.pad)
    loader = DataLoader(ds, batch_size=args.batch_clips, shuffle=False, num_workers=args.num_workers,
                        pin_memory=True, collate_fn=collate_clips)
    print("  {} clips over {} tracklets ({:.1f} clips/tracklet); cam_id={} flip_tta={}".format(
        len(ds), n_tracklets, len(ds) / float(n_tracklets), args.cam_id, not args.no_flip), flush=True)

    acc, cnt = None, np.zeros(n_tracklets, dtype=np.float64)
    for b, (tis, clips) in enumerate(loader):
        clips = clips.to(args.device, non_blocking=True)                       # (B, T, 3, H, W)
        cam = torch.full((clips.shape[0],), args.cam_id, dtype=torch.long, device=args.device)
        f = model(clips, cam_label=cam, view_label=None)                       # (B, 2048), raw
        if not args.no_flip:
            f2 = model(VF.hflip(clips), cam_label=cam, view_label=None)
            f = torch.stack([f, f2], dim=0).mean(dim=0)
        f = f.float().cpu().numpy()
        if acc is None:
            acc = np.zeros((n_tracklets, f.shape[1]), dtype=np.float64)
        for t, fi in zip(tis.tolist(), f):
            acc[t] += fi
            cnt[t] += 1.0
        if (b + 1) % 50 == 0:
            print('  {}/{} clip batches'.format(b + 1, len(loader)), flush=True)
    assert acc is not None and cnt.min() > 0, "some tracklet produced no clip"
    return (acc / cnt[:, None]).astype(np.float32)                             # (n_tracklets, 2048)


def assert_one_track_per_video(by_video, recs, role):
    """Check that each video has a single tracklet before assigning its pooled feature to the
    crop rows."""
    bad = []
    for vid, idxs in by_video.items():
        pids = {recs[i]['pid'] for i in idxs}
        trks = {recs[i]['dive_track_id'] for i in idxs}
        if len(pids) > 1 or len(trks) > 1:
            bad.append((vid, sorted(pids), sorted(trks)))
    if bad:
        detail = "\n".join("    {}  pids={}  dive_track_ids={}".format(v, p, t) for v, p, t in bad[:10])
        raise SystemExit(
            "Split [{}]: {} video group(s) contain more than one identity or DIVE track.\n"
            "Each video must contain one subject and one tracklet. Check the split entries below.\n"
            "{}{}".format(role, len(bad), detail,
                          "\n    ... and {} more".format(len(bad) - 10) if len(bad) > 10 else ""))
    print("  Checked {} single-track videos".format(len(by_video)))


def main():
    args = parse_args()
    model, camera_num = build_model(cfg, args.weight, args.device)
    if not (0 <= args.cam_id < camera_num):
        raise SystemExit("--cam-id {} out of range: cv_embed has {} rows".format(args.cam_id, camera_num))
    transform = build_test_transform(cfg)

    splits_dir = args.splits_dir or paths.splits_dir(args.split)
    out_dir = args.out_dir or paths.features_dir(args.split, args.method, args.source, dataset=args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    # provenance stamp: eval feature + how many RRS clips were averaged + whether flip TTA was used
    neck_tag = "{}_cat2048_rrs{}{}".format(
        cfg.TEST.NECK_FEAT, 'all' if args.max_clips <= 0 else args.max_clips,
        '' if args.no_flip else '_hflip')

    for role, coco, fname_token in features_io.discover_role_files(splits_dir):
        recs, internal_token, crop_root = features_io.load_crops_full(coco, role)
        camera_token = fname_token or internal_token   # queries: token from filename; gallery: from kwcoco

        # group crop indices by video stem (== the eval's video-pooling key); order each tracklet by frame
        by_video = defaultdict(list)
        for i, r in enumerate(recs):
            by_video[r['video']].append(i)
        assert_one_track_per_video(by_video, recs, role)
        tracklets = []
        for vid, idxs in by_video.items():
            idxs = sorted(idxs, key=lambda i: recs[i]['frame'])
            tracklets.append((vid, [recs[i]['path'] for i in idxs]))
        print("[{}] {} crops in {} tracklets range={}; extracting...".format(
            role, len(recs), len(tracklets), camera_token), flush=True)

        F = extract_tracklet_feats(model, tracklets, transform, args, len(tracklets))
        feat_of_video = {tracklets[i][0]: F[i] for i in range(len(tracklets))}

        # broadcast each tracklet feature to all its crops (row-aligned to recs for the per-crop contract);
        # Use video pooling because the crop rows share one tracklet feature.
        feats = np.stack([feat_of_video[r['video']] for r in recs]).astype(np.float32)
        out = osp.join(out_dir, features_io.feature_filename(role, camera_token))
        features_io.write_features(out, feats, recs, crop_root=crop_root, split_role=role,
                                   camera_token=camera_token, neck_feat=neck_tag, weight=args.weight,
                                   split=args.split, method=args.method, source=args.source, pooling='video')
        print("  wrote {}  (feats {}, {:.1f} MB)".format(out, feats.shape, os.path.getsize(out) / 1e6))


if __name__ == "__main__":
    main()
