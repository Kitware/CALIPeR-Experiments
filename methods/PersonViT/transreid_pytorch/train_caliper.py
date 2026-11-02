# encoding: utf-8
"""Fine-tune PersonViT on CALIPeR ground-field imagery."""
import argparse
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append('.')
import processor.processor as processor_mod                 # noqa: E402  (patched below)
from config import cfg                                      # noqa: E402
from datasets.bases import ImageDataset                     # noqa: E402
from datasets.caliper_ground import CaliperGround               # noqa: E402
from datasets.make_dataloader import train_collate_fn, val_collate_fn   # noqa: E402
from loss import make_loss                                  # noqa: E402
from model import make_model                                # noqa: E402
from processor import do_train                              # noqa: E402
from solver import make_optimizer                           # noqa: E402
from solver.scheduler_factory import create_scheduler       # noqa: E402
from solver.lr_scheduler import WarmupMultiStepLR           # noqa: E402
from utils.logger import setup_logger                       # noqa: E402
from utils.metrics import R1_mAP_eval, eval_func            # noqa: E402

from caliper_eval.train_corpus import HOLDOUT_RULES           # noqa: E402
from caliper_eval.train_metrics import make_by_range_metric   # noqa: E402
from caliper_eval.train_metadata import write_run_json        # noqa: E402
from caliper_eval.train_sampler import (ResamplingIdentitySampler,   # noqa: E402
                                      MixingIdentitySampler)
from caliper_eval.train_entry import (add_sim_args, log_composition,   # noqa: E402
                                     parse_severities)

import torchvision.transforms as T                          # noqa: E402
from timm.data.random_erasing import RandomErasing          # noqa: E402


def require_single_gpu():
    """Check that exactly one GPU is visible to the trainer."""
    n = torch.cuda.device_count()
    if n != 1:
        raise SystemExit(
            "Training requires one GPU. Found {} visible devices.\n"
            "  Fix: sbatch --gres=gpu:1, or CUDA_VISIBLE_DEVICES=<n> for an interactive run."
            .format(n))


def load_backbone(model, path, logger):
    """Load pretrained backbone weights and report coverage, skipping the classifier when
    transferring between datasets."""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    if 'model' in ckpt and isinstance(ckpt['model'], dict):
        ckpt = ckpt['model']
    own = model.state_dict()
    loaded, skipped_cls, mismatch, missing = 0, 0, [], []
    for k, v in ckpt.items():
        kk = k.replace('module.', '')
        if 'classifier' in kk:
            skipped_cls += 1
            continue
        if kk not in own:
            missing.append(kk)
        elif own[kk].shape != v.shape:
            mismatch.append((kk, tuple(own[kk].shape), tuple(v.shape)))
        else:
            own[kk].copy_(v)
            loaded += 1
    ck = {k.replace('module.', '') for k in ckpt}
    unfilled = [k for k in own if 'classifier' not in k and k not in ck]
    logger.info("init from {}".format(path))
    logger.info("  loaded {} tensors | classifier keys skipped {} (head is re-initialised) | "
                "in ckpt but not in model {} | shape mismatch {} | in model but not in ckpt {}"
                .format(loaded, skipped_cls, len(missing), len(mismatch), len(unfilled)))
    for k, a, b in mismatch[:5]:
        logger.info("    shape mismatch {}: model {} vs ckpt {}".format(k, a, b))
    for k in unfilled[:5]:
        logger.info("    Missing weights: {}".format(k))
    for k in missing[:5]:
        logger.info("    Only in checkpoint: {}".format(k))
    if mismatch or unfilled:
        raise SystemExit("Incomplete backbone weights. Check the tensor names and shapes listed above.")
    return loaded


def build_loaders(cfg, args):
    """Build training and validation loaders for the CALIPeR corpus."""
    train_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
        T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
        T.Pad(cfg.INPUT.PADDING),
        T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
    ])
    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
    ])

    # Apply the same resize and augmentation to real and synthetic crops.
    ds = CaliperGround(holdout=args.holdout, holdout_rule=args.holdout_rule, seed=args.seed,
                     chip_root=args.chip_root or None,
                     sim_mode=args.sim_mode, sim_p=args.sim_p,
                     sim_chip_root=args.sim_chip_root or None,
                     sim_severities=parse_severities(args.sim_severity))
    if ds.variant_groups:
        # exactly one rendering of each 400m tracklet per epoch, redrawn every epoch
        sampler = MixingIdentitySampler(
            ds.train, ds.train_tracklets,
            batch_size=cfg.SOLVER.IMS_PER_BATCH,
            num_instances=cfg.DATALOADER.NUM_INSTANCE,
            cap=args.frames_per_tracklet, seed=args.seed,
            variant_groups=ds.variant_groups)
    else:
        sampler = ResamplingIdentitySampler(
            ds.train, ds.train_tracklets,
            batch_size=cfg.SOLVER.IMS_PER_BATCH,
            num_instances=cfg.DATALOADER.NUM_INSTANCE,
            cap=args.frames_per_tracklet, seed=args.seed)
    train_loader = DataLoader(ImageDataset(ds.train, train_transforms),
                              batch_size=cfg.SOLVER.IMS_PER_BATCH, sampler=sampler,
                              num_workers=cfg.DATALOADER.NUM_WORKERS, collate_fn=train_collate_fn,
                              pin_memory=True)

    # With an empty holdout, schedule validation after the last training epoch.
    val_loader = DataLoader(ImageDataset(ds.query + ds.gallery, val_transforms),
                            batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False,
                            num_workers=cfg.DATALOADER.NUM_WORKERS, collate_fn=val_collate_fn)
    return train_loader, val_loader, len(ds.query), ds, sampler


def main():
    ap = argparse.ArgumentParser(description="Finetune PersonViT on the CALIPeR ground-field corpus")
    ap.add_argument("--config_file", default="configs/caliper_ground_vit.yml")
    ap.add_argument("--init-weight", default="",
                    help="ReID checkpoint for the backbone; classifier keys are skipped.")
    ap.add_argument("--holdout", type=int, default=0,
                    help='Number of identities reserved for validation. Zero trains on every identity.')
    ap.add_argument("--holdout-rule", default="random", choices=sorted(HOLDOUT_RULES),
                    help='Select validation identities at random or by the fewest 1000m samples. '
                         'thin-relaxed requires 400m and 1000m coverage. The other rules also '
                         'require 500m.')
    ap.add_argument("--frames-per-tracklet", type=int, default=32, help="per-epoch cap; 0 = all")
    ap.add_argument("--chip-root", default="", help="override the chip root (e.g. a /dev/shm stage)")
    ap.add_argument("--seed", type=int, default=1234)
    # This entry point predates caliper_eval.train_entry.add_caliper_args and keeps its own parser, so
    # the synthetic flags are attached here rather than inherited.
    add_sim_args(ap)
    ap.add_argument("--stop-after", type=int, default=0,
                    help='Stop after this epoch while keeping the full learning-rate schedule.')
    ap.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = ap.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    if args.init_weight:
        # Use the supplied backbone checkpoint when initializing the transformer.
        cfg.MODEL.PRETRAIN_CHOICE = 'none'
    cfg.freeze()

    seed = cfg.SOLVER.SEED if hasattr(cfg.SOLVER, 'SEED') else args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    logger = setup_logger("transreid", cfg.OUTPUT_DIR, if_train=True)
    logger.info("Running with config:\n{}".format(cfg))
    logger.info("args: {}".format(vars(args)))
    require_single_gpu()          # before anything expensive; see the docstring

    train_loader, val_loader, num_query, ds, sampler = build_loaders(cfg, args)
    logger.info("num_classes={}  images/epoch={}  (corpus is larger; the sampler re-draws each epoch)"
                .format(ds.num_train_pids, len(sampler)))
    # Check the training population before fitting the model.
    log_composition(logger, ds.corpus, sampler, sim_mode=args.sim_mode, sim_p=args.sim_p,
                    batch_size=cfg.SOLVER.IMS_PER_BATCH, holdout=args.holdout, unit="images")
    logger.info("holdout={} rule={} -> {} val ids of {} eligible; {} train ids"
                .format(args.holdout, args.holdout_rule, len(ds.corpus.val_ids),
                        len(ds.corpus.eligible), len(ds.corpus.train_ids)))
    logger.info("Training record: {}".format(
        write_run_json(cfg.OUTPUT_DIR, cfg=cfg, args=args, corpus=ds.corpus, repo_dir='.')))

    # Use the per-range validation metric in the method’s trainer.
    if args.holdout:
        processor_mod.R1_mAP_eval = make_by_range_metric(
            R1_mAP_eval, eval_func, cfg.OUTPUT_DIR,
            eval_period=cfg.SOLVER.EVAL_PERIOD, logger=logger, return_super=True)

    model = make_model(cfg, num_class=ds.num_train_pids,
                       camera_num=ds.num_train_cams, view_num=ds.num_train_vids)
    if args.init_weight:
        load_backbone(model, args.init_weight, logger)
    else:
        logger.info("no --init-weight: backbone is whatever PRETRAIN_CHOICE gave "
                    "({})".format(cfg.MODEL.PRETRAIN_CHOICE))

    loss_func, center_criterion = make_loss(cfg, num_classes=ds.num_train_pids)
    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)
    if cfg.SOLVER.WARMUP_METHOD == 'cosine':
        logger.info('===========using cosine learning rate=======')
        scheduler = create_scheduler(cfg, optimizer)
    else:
        logger.info('===========using normal learning rate=======')
        scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA,
                                      cfg.SOLVER.WARMUP_FACTOR, cfg.SOLVER.WARMUP_EPOCHS,
                                      cfg.SOLVER.WARMUP_METHOD)

    # End the loop early while keeping the original learning-rate schedule.
    if args.stop_after:
        if args.stop_after > cfg.SOLVER.MAX_EPOCHS:
            raise SystemExit("--stop-after {} exceeds SOLVER.MAX_EPOCHS {}; nothing to truncate."
                             .format(args.stop_after, cfg.SOLVER.MAX_EPOCHS))
        logger.info("--stop-after {}: LR schedule stays built for MAX_EPOCHS {} (cosine t_initial is "
                    "already fixed); only the training loop is shortened."
                    .format(args.stop_after, cfg.SOLVER.MAX_EPOCHS))
        cfg.defrost()
        cfg.SOLVER.MAX_EPOCHS = args.stop_after
        cfg.freeze()

    do_train(cfg, model, center_criterion, train_loader, val_loader, optimizer, optimizer_center,
             scheduler, loss_func, num_query, local_rank=0)


if __name__ == '__main__':
    main()
