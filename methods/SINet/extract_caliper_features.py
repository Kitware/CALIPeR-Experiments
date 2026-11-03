# encoding: utf-8
"""Extract SINet tracklet features for the CALIPeR gallery and query sets."""
import argparse
import os
import os.path as osp
import sys

import numpy as np
import torch
import torchvision.models as tvm
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset

# SINet repo (this file lives at the repo root)
REPO = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, REPO)

import transforms.spatial_transforms as ST    # noqa: E402
from models import init_model                  # noqa: E402
from utils.video_loader import image_loader    # noqa: E402

# shared eval package
try:
    from caliper_eval import paths, features_io
except ImportError:
    _REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
    sys.path.insert(0, os.environ.get('CALIPER_EVAL_HOME', _REPO_ROOT))
    from caliper_eval import paths, features_io


def build_test_transform(height=256, width=128):
    """Build the preprocessing transform used by the released checkpoint."""
    return ST.Compose([
        ST.Scale((height, width), interpolation=3),      # 3 == PIL.Image.BICUBIC
        ST.ToTensor(),
        ST.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def loop_pad(paths_list, size):
    """Extend the frame indices to a multiple of the clip length while retaining temporal order."""
    p = list(paths_list)
    exp = ((len(p) + size - 1) // size) * size
    while len(p) != exp:
        lack = exp - len(p)
        p.extend(p[-lack:] if len(p) > lack else p)
    assert len(p) % size == 0
    return p


class TrackletDataset(Dataset):
    """Load the ordered clips and tracklet key for each video."""

    def __init__(self, tracklets, transform, seq_len):
        self.tracklets = tracklets          # list of (video_key, [abs frame paths in frame order])
        self.t = transform
        self.seq_len = seq_len

    def __len__(self):
        return len(self.tracklets)

    def __getitem__(self, i):
        key, frame_paths = self.tracklets[i]
        fp = loop_pad(frame_paths, self.seq_len)
        self.t.randomize_parameters()                       # test transform: no-op, but keep the API contract
        clip = [self.t(image_loader(p)) for p in fp]        # each (C, H, W)
        clip = torch.stack(clip, 1)                          # (C, T, H, W)
        C, T, H, W = clip.size()
        # 'restricted' sampler: (C, T) -> (C, seq_len, n_clips) -> (n_clips, C, seq_len, ...)
        clip = clip.reshape(C, self.seq_len, T // self.seq_len, H, W).permute(2, 0, 1, 3, 4).contiguous()
        return clip, key


def extract_tracklet_feat(model, clips, max_clip_per_batch, device):
    """extract_feat_all_frames: per-clip forward -> (n_clips, C, t) -> mean over t -> mean over clips -> (C,)."""
    clips = clips.to(device)
    feats = []
    n = clips.size(0)
    for i in range((n - 1) // max_clip_per_batch + 1):
        chunk = clips[i * max_clip_per_batch:(i + 1) * max_clip_per_batch]
        f = model(chunk)            # eval-mode: (b, C, t)
        f = f.mean(-1)              # (b, C)
        feats.append(f)
    f = torch.cat(feats, 0).mean(0)     # (C,)
    return f.detach().cpu().numpy().astype(np.float32)


def load_model(arch, weight, seq_len, device):
    ck = torch.load(weight, map_location='cpu', weights_only=False)
    sd = ck['state_dict'] if isinstance(ck, dict) and 'state_dict' in ck else ck
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    num_classes = sd['classifier.weight'].shape[0]      # derive head size so the strict load matches

    # The checkpoint supplies the backbone weights, so skip the ImageNet download.
    _orig = tvm.resnet50
    tvm.resnet50 = lambda *a, **k: _orig(weights=None)
    try:
        model = init_model(name=arch, num_classes=num_classes,
                           losses=['xent', 'htri', 'infonce'], seq_len=seq_len)
    finally:
        tvm.resnet50 = _orig

    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.startswith('projection')]   # projection head unused at inference
    if missing or unexpected:
        print("WARNING load mismatch: {} missing, {} unexpected (e.g. missing {}, unexpected {})".format(
            len(missing), len(unexpected), missing[:4], unexpected[:4]))
    else:
        print("load: all {} tensors matched (num_classes={})".format(len(sd), num_classes))
    return model.to(device).eval(), num_classes


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
    ap = argparse.ArgumentParser(description="SINet (video ReID) feature extractor (shared .npz format)")
    ap.add_argument("--weight", required=True, help="SINet checkpoint .pth.tar (e.g. weights/sinet_mars.pth.tar)")
    ap.add_argument("--source", required=True, help="pretraining video dataset label (mars | lsvid)")
    ap.add_argument("--arch", default="sinet", choices=["sinet", "sbnet", "idnet", "sanet", "resnet50"])
    ap.add_argument("--method", default="sinet")
    ap.add_argument("--split", default="paperGR")
    ap.add_argument("--dataset", default="combined_brc1_brc2",
                    help="dataset namespace of the output dir (ground_range_sim uses kitbrc1)")
    ap.add_argument("--splits-dir", default="", help="Crop split directory (default: $REID_BENCH_ROOT/splits/<split>).")
    ap.add_argument("--out-dir", default="", help="Where to save features (default: $REID_BENCH_ROOT/eval/<dataset>/<split>/<method>/<source>).")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seq-len", type=int, default=4, help="frames per clip (must match training: SiNet=4)")
    ap.add_argument("--max-clip-per-batch", type=int, default=45)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()

    model, _ = load_model(args.arch, args.weight, args.seq_len, args.device)
    transform = build_test_transform(args.height, args.width)

    splits_dir = args.splits_dir or paths.splits_dir(args.split)
    out_dir = args.out_dir or paths.features_dir(args.split, args.method, args.source, dataset=args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    for role, coco, fname_token in features_io.discover_role_files(splits_dir):
        recs, internal_token, crop_root = features_io.load_crops_full(coco, role)
        camera_token = fname_token or internal_token   # queries: token from filename; gallery: from kwcoco

        # group crop indices by video stem (== eval video-pooling key); order each tracklet by frame
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

        ds = TrackletDataset(tracklets, transform, args.seq_len)
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers,
                            collate_fn=lambda b: b[0])      # one tracklet per item: (clip, key)
        feat_of_video = {}
        with torch.no_grad():
            for n, (clips, key) in enumerate(loader):
                feat_of_video[key] = extract_tracklet_feat(model, clips, args.max_clip_per_batch, args.device)
                if (n + 1) % 50 == 0:
                    print('  {}/{} tracklets'.format(n + 1, len(tracklets)), flush=True)

        # broadcast each tracklet feature to all its crops (row-aligned to recs for the per-crop contract)
        feats = np.stack([feat_of_video[r['video']] for r in recs]).astype(np.float32)
        out = osp.join(out_dir, features_io.feature_filename(role, camera_token))
        features_io.write_features(out, feats, recs, crop_root=crop_root, split_role=role,
                                   camera_token=camera_token, neck_feat='tracklet_mean_bn', weight=args.weight,
                                   split=args.split, method=args.method, source=args.source, pooling='video')
        print("  wrote {}  (feats {}, {:.1f} MB)".format(out, feats.shape, os.path.getsize(out) / 1e6))


if __name__ == "__main__":
    main()
