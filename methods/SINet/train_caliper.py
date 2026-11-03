# encoding: utf-8
"""Fine-tune SINet on CALIPeR ground-field imagery."""
import os
import os.path as osp
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

sys.path.append('.')

# Import main with argv stubbed so ITS module-level parse_args() takes defaults instead of ours.
# Nothing else runs at import: main() is guarded by __main__.
_argv, sys.argv = sys.argv, [sys.argv[0]]
import main as sinet                                            # noqa: E402
sys.argv = _argv

import transforms.spatial_transforms as ST                      # noqa: E402
import transforms.temporal_transforms as TT                     # noqa: E402
from caliper_ground import CaliperGround                            # noqa: E402
from models import init_model                                   # noqa: E402
from utils.losses import TripletLoss, InfoNce                   # noqa: E402
from utils.samplers import RandomIdentitySampler                # noqa: E402
from utils.utils import Logger, save_checkpoint, print_time     # noqa: E402
from utils.eval_metrics import evaluate                       # noqa: E402
from utils.video_loader import VideoDataset, VideoDatasetInfer  # noqa: E402

from caliper_eval.train_corpus import HOLDOUT_RULES               # noqa: E402
from caliper_eval.train_entry import (load_backbone, require_single_gpu,   # noqa: E402
                                    add_sim_args, log_composition, parse_severities)
from caliper_eval.train_sampler import MixingTrackletSampler        # noqa: E402
from caliper_eval.train_metadata import write_run_json            # noqa: E402
from caliper_eval.train_metrics import CAMID_NAMES                # noqa: E402


def parse():
    """Reuse SINet's own parser so every upstream flag keeps working, then add ours."""
    ap = sinet.parser
    ap.add_argument("--holdout", type=int, default=0,
                    help='Number of identities reserved for validation. Zero trains on every identity.')
    ap.add_argument("--holdout-rule", default="random", choices=sorted(HOLDOUT_RULES),
                    help='Select validation identities at random or by the fewest 1000m samples. '
                         'thin-relaxed requires 400m and 1000m coverage. The other rules also '
                         'require 500m.')
    ap.add_argument("--chip-root", default="", help="override the chip root (e.g. a RAM-disk copy)")
    ap.add_argument("--ckpt_step", type=int, default=25,
                    help='Save a checkpoint every N epochs, independently of validation.')
    ap.add_argument("--init-weight", default="",
                    help='Pretrained ReID checkpoint used to initialize the backbone.')
    # Add synthetic sampling options to the SINet parser.
    add_sim_args(ap)
    args = ap.parse_args()
    # train()/test() read main.args as a global -- rebind it so they see our values
    sinet.args = args
    return args


def test_by_range(model, queryloader, galleryloader, epoch, save_dir):
    """Score held-out query tracklets separately for each camera range."""
    import json
    # Use evaluation mode to extract the tracklet embedding.
    model.eval()
    ff = sinet.extract_feat_sampled_frames
    qf, q_pids, q_camids = sinet._feats_of_loader(model, queryloader, ff, use_gpu=True)
    gf, g_pids, g_camids = sinet._feats_of_loader(model, galleryloader, ff, use_gpu=True)
    m, n = qf.size(0), gf.size(0)
    d = (torch.pow(qf, 2).sum(1, keepdim=True).expand(m, n) +
         torch.pow(gf, 2).sum(1, keepdim=True).expand(n, m).t())
    d.addmm_(qf, gf.t(), beta=1, alpha=-2)
    # kept as a TORCH TENSOR: SINet's evaluate() calls torch.argsort on it (eval_metrics.py:31),
    # unlike the other repos' eval_func which take numpy. d.cpu().numpy() here raises TypeError.
    d = d.cpu()

    subsets = [('all', np.ones(m, dtype=bool))]
    for c in sorted(set(q_camids.tolist())):
        subsets.append((CAMID_NAMES.get(c, 'cam%s' % c), q_camids == c))
    pooled_rank1 = -np.inf
    for name, mask in subsets:
        if not mask.any():
            continue
        tmask = torch.from_numpy(mask)
        cmc, mAP = evaluate(d[tmask], q_pids[mask], g_pids, q_camids[mask], g_camids)
        if name == 'all':
            pooled_rank1 = cmc[0]
        with open(osp.join(save_dir, 'val_metrics.jsonl'), 'a') as f:
            f.write(json.dumps({'epoch': int(epoch), 'subset': name, 'mAP': float(mAP),
                                'R1': float(cmc[0]), 'R5': float(cmc[4]) if len(cmc) > 4 else None,
                                'R10': float(cmc[9]) if len(cmc) > 9 else None,
                                'n_query': int(mask.sum()), 'n_gallery': int(n)}) + chr(10))
        print_time("  val[%6s] epoch %3d  mAP %.1f%%  R1 %.1f%%  (%d queries)"
                   % (name, epoch, 100 * mAP, 100 * cmc[0], int(mask.sum())))
    return pooled_rank1


def main():
    args = parse()

    # Use the GPU selected in the caller’s environment.
    require_single_gpu("SINet wraps the model in nn.DataParallel unconditionally (main.py:218).")

    # With no holdout the val split is empty and test() dies on it. This is the argparse analogue of
    # caliper_eval.train_entry.disable_eval_if_no_holdout -- SINet is not yacs, so it is inlined.
    if not args.holdout and args.eval_step > 0 and args.eval_step <= args.max_epoch:
        print_time("no --holdout, so the val split is empty: forcing --eval_step {} -> {} so "
                   "evaluation never fires (checkpointing is unaffected; see --ckpt_step)."
                   .format(args.eval_step, args.max_epoch + 1))
        args.eval_step = args.max_epoch + 1

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    sys.stdout = Logger(osp.join(args.save_dir, 'log_train.log'))
    print_time("============ SINet / CALIPeR ground-field ============")
    print("\n".join("\t\t%s: %s" % (k, v) for k, v in sorted(vars(args).items())))

    # Apply the same resize and augmentation to real and synthetic crops.
    dataset = CaliperGround(holdout=args.holdout, holdout_rule=args.holdout_rule, seed=args.seed,
                          chip_root=args.chip_root or None,
                          sim_mode=args.sim_mode, sim_p=args.sim_p,
                          sim_chip_root=args.sim_chip_root or None,
                          sim_severities=parse_severities(args.sim_severity))
    print_time("holdout={} rule={} -> {} val ids of {} eligible; {} train ids"
               .format(args.holdout, args.holdout_rule, len(dataset.corpus.val_ids),
                       len(dataset.corpus.eligible), len(dataset.corpus.train_ids)))
    print_time("Training record: " + write_run_json(args.save_dir, cfg=None, args=args,
                                                 corpus=dataset.corpus, repo_dir='.'))

    # transforms copied verbatim from main.py:135-148 so a CALIPeR run differs from a stock one in the
    # corpus and nothing else
    st_train = ST.Compose([ST.Scale((args.height, args.width), interpolation=3),
                           ST.RandomHorizontalFlip(), ST.ToTensor(),
                           ST.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                           ST.RandomErasing()])
    st_test = ST.Compose([ST.Scale((args.height, args.width), interpolation=3), ST.ToTensor(),
                          ST.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    tt_train = TT.TemporalRestrictedCrop(size=args.seq_len)
    tt_test = TT.TemporalRestrictedBeginCrop(size=args.seq_len)

    if dataset.variant_groups:
        # Choose one member of each variant group before drawing tracklets.
        train_sampler = MixingTrackletSampler(
            dataset.train, dataset.train_tracklets, num_instances=args.num_instances,
            seed=args.seed, variant_groups=dataset.variant_groups)
    else:
        train_sampler = RandomIdentitySampler(dataset.train, num_instances=args.num_instances)
    trainloader = DataLoader(
        VideoDataset(dataset.train, spatial_transform=st_train, temporal_transform=tt_train),
        sampler=train_sampler,
        batch_size=args.train_batch, num_workers=args.workers, pin_memory=True, drop_last=True)
    queryloader = DataLoader(
        VideoDataset(dataset.query, spatial_transform=st_test, temporal_transform=tt_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers, pin_memory=True)
    galleryloader = DataLoader(
        VideoDataset(dataset.gallery, spatial_transform=st_test, temporal_transform=tt_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers, pin_memory=True)
    print_time("epoch = {} tracklets ({} iterations at batch {})"
               .format(len(trainloader.sampler), len(trainloader), args.train_batch))
    # SINet counts tracklets per epoch.
    log_composition(_L(), dataset.corpus, trainloader.sampler,
                    sim_mode=args.sim_mode, sim_p=args.sim_p, batch_size=args.train_batch,
                    holdout=args.holdout, unit="tracklets",
                    expected_epoch={None: 2416, 'add': 2892, 'dup': 2892,
                                    'mix': 2416, 'swapall': 2416})

    model = init_model(name=args.arch, num_classes=dataset.num_train_pids,
                       losses=args.losses, seq_len=args.seq_len)
    init = args.init_weight or (args.pretrain_model_path if args.pretrain else "")
    if init:
        # Load the pretrained backbone.
        load_backbone(model, init, _L(), skip=("classifier",))
    else:
        print_time("no --init-weight: training from scratch")

    criterions = {'xent': nn.CrossEntropyLoss(),
                  'htri': TripletLoss(margin=args.margin, distance=args.distance),
                  'infonce': InfoNce(num_instance=args.num_instances)}
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.stepsize, gamma=args.gamma)
    model = nn.DataParallel(model).cuda()

    print_time("==> Start training ({} epochs)".format(args.max_epoch))
    best_rank1, best_epoch, t0 = -np.inf, 0, time.time()
    for epoch in range(args.start_epoch, args.max_epoch):
        sinet.train(epoch, model, criterions, optimizer, trainloader, use_gpu=True)
        scheduler.step()
        last = (epoch + 1) == args.max_epoch
        # DECOUPLED, unlike upstream. main.py saves only inside the eval block, so a no-holdout run
        # (eval disabled) would retain a single checkpoint at max_epoch.
        do_eval = bool(len(dataset.query)) and args.eval_step > 0 and \
            ((epoch + 1) % args.eval_step == 0 or last)
        do_ckpt = (epoch + 1) % args.ckpt_step == 0 or last

        rank1 = -np.inf
        if do_eval:
            with torch.no_grad():
                rank1 = test_by_range(model, queryloader, galleryloader, epoch + 1, args.save_dir)
            if rank1 > best_rank1:
                best_rank1, best_epoch = rank1, epoch + 1
        if do_ckpt:
            save_checkpoint({'state_dict': model.module.state_dict(), 'rank1': rank1,
                             'epoch': epoch}, rank1 >= best_rank1 and do_eval,
                            osp.join(args.save_dir, 'checkpoint_ep%d.pth.tar' % (epoch + 1)))
    print_time("==> Best Rank-1 {:.1%} at epoch {}".format(best_rank1, best_epoch))
    print_time("Finished in {}".format(str(__import__('datetime').timedelta(
        seconds=round(time.time() - t0)))))


class _L(object):
    """load_backbone() wants a logger; SINet prints. Adapt rather than depend on logging config."""
    def info(self, m):
        print_time(m)


if __name__ == '__main__':
    main()
