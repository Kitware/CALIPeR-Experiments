# encoding: utf-8
"""Fine-tune CLIP-ReID on CALIPeR ground-field imagery."""
import argparse
import json
import os
import os.path as osp
import random
import sys

import numpy as np
import torch
import torchvision.transforms as T
from timm.data.random_erasing import RandomErasing
from torch.utils.data import DataLoader

sys.path.append('.')
import processor.processor_clipreid_stage2 as stage2_mod    # noqa: E402  (patched below)
from config import cfg                                      # noqa: E402
from datasets.bases import ImageDataset                     # noqa: E402
from datasets.caliper_ground import CaliperGround               # noqa: E402
from datasets.make_dataloader_clipreid import train_collate_fn, val_collate_fn   # noqa: E402
from loss.make_loss import make_loss                        # noqa: E402
from model.make_model_clipreid import make_model            # noqa: E402
from processor.processor_clipreid_stage1 import do_train_stage1   # noqa: E402
from processor.processor_clipreid_stage2 import do_train_stage2   # noqa: E402
from solver.lr_scheduler import WarmupMultiStepLR           # noqa: E402
from solver.make_optimizer_prompt import make_optimizer_1stage, make_optimizer_2stage  # noqa: E402
from solver.scheduler_factory import create_scheduler       # noqa: E402
from utils.logger import setup_logger                       # noqa: E402
from utils.metrics import R1_mAP_eval, eval_func            # noqa: E402

from caliper_eval.train_entry import (add_caliper_args, disable_eval_if_no_holdout, load_backbone,
                                     require_single_gpu)  # noqa: E402
from caliper_eval.train_metrics import make_by_range_metric   # noqa: E402
from caliper_eval.train_metadata import write_run_json        # noqa: E402
from caliper_eval.train_sampler import (ResamplingIdentitySampler,   # noqa: E402
                                      MixingIdentitySampler)
from caliper_eval.train_entry import (add_sim_args, log_composition,   # noqa: E402
                                     parse_severities)


def train_transform(cfg):
    """Build the training image transform from the method configuration."""
    return T.Compose([
        T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
        T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
        T.Pad(cfg.INPUT.PADDING),
        T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
    ])


def val_transform(cfg):
    """Build the validation image transform from the method configuration."""
    return T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
    ])


def _val_loader(cfg, ds, val_transforms):
    return DataLoader(
        ImageDataset(ds.query + ds.gallery, val_transforms), batch_size=cfg.TEST.IMS_PER_BATCH,
        shuffle=False, num_workers=cfg.DATALOADER.NUM_WORKERS, collate_fn=val_collate_fn)


def load_stage1(model, path, corpus, logger):
    """Load the fitted prompt bank after matching its training identities to the current corpus."""
    meta = osp.join(osp.dirname(osp.abspath(path)), 'run.json')
    if not osp.exists(meta):
        raise SystemExit(
            "Missing run.json beside %s. Use a stage-1 checkpoint with its training identity list." % path)
    src = json.load(open(meta)).get('corpus', {}).get('train_ids')
    here = sorted(corpus.train_ids)
    if src is None:
        raise SystemExit("Cannot load --stage1-init %s: %s records no corpus.train_ids." % (path, meta))
    if list(src) != here:
        extra, gone = set(src) - set(here), set(here) - set(src)
        raise SystemExit(
            "Cannot load --stage1-init %s: identity set differs from the run that produced it.\n"
            "  source %d ids, here %d ids; only in source: %s; only here: %s"
            % (path, len(src), len(here), sorted(extra)[:5], sorted(gone)[:5]))

    n = load_backbone(model, path, logger, skip=('classifier',))
    ctx = model.prompt_learner.cls_ctx.data
    if ctx.shape[0] != len(here):
        raise SystemExit("Cannot load --stage1-init %s: bank is %d-way, corpus is %d-way."
                         % (path, ctx.shape[0], len(here)))
    # Record a summary of the loaded prompts.
    logger.info("Loaded stage-1 prompts from {} ({} tensors). cls_ctx {} "
                "sum={:.6f} absmax={:.6f} over {} verified identities"
                .format(path, n, tuple(ctx.shape), float(ctx.sum()), float(ctx.abs().max()), len(here)))
    return n


def build_loaders(cfg, args):
    """Build training and validation loaders for the CALIPeR corpus."""
    train_transforms = train_transform(cfg)
    val_transforms = val_transform(cfg)

    def train_ds(items):
        return ImageDataset(items, train_transforms)

    ds = CaliperGround(holdout=args.holdout, holdout_rule=args.holdout_rule, seed=args.seed,
                     chip_root=args.chip_root or None,
                     sim_mode=args.sim_mode, sim_p=args.sim_p,
                     sim_chip_root=args.sim_chip_root or None,
                     sim_severities=parse_severities(args.sim_severity))

    if ds.variant_groups:
        sampler = MixingIdentitySampler(
            ds.train, ds.train_tracklets,
            batch_size=cfg.SOLVER.STAGE2.IMS_PER_BATCH,
            num_instances=cfg.DATALOADER.NUM_INSTANCE,
            cap=args.frames_per_tracklet, seed=args.seed,
            variant_groups=ds.variant_groups)
    else:
        sampler = ResamplingIdentitySampler(
            ds.train, ds.train_tracklets,
            batch_size=cfg.SOLVER.STAGE2.IMS_PER_BATCH,
            num_instances=cfg.DATALOADER.NUM_INSTANCE,
            cap=args.frames_per_tracklet, seed=args.seed)
    train_loader_stage2 = DataLoader(
        train_ds(ds.train), batch_size=cfg.SOLVER.STAGE2.IMS_PER_BATCH,
        sampler=sampler, num_workers=cfg.DATALOADER.NUM_WORKERS, collate_fn=train_collate_fn,
        pin_memory=True)

    # Stage 1 caches features once. Use real crops by default or select a fixed set of variants.
    if args.stage1_init:
        # Stage 1 is not run at all: its result -- the fitted prompt bank -- is loaded from a
        # completed run in main(). Nothing to build, and notably no second corpus to parse.
        ds.stage1_note, ds.stage1_n = 'LOADED from %s' % args.stage1_init, 0
        return train_loader_stage2, None, _val_loader(cfg, ds, val_transforms), len(ds.query), ds, sampler

    s1_items, s1_note = ds.train, 'all variants (naive)'
    if args.sim_mode and args.sim_stage1 == 'real':
        # A second, sim-free corpus -- stage 1 becomes byte-identical to the baseline's, so the
        # synthetic content enters at stage 2 only.
        real = CaliperGround(holdout=args.holdout, holdout_rule=args.holdout_rule, seed=args.seed,
                           chip_root=args.chip_root or None, verbose=False)
        s1_items, s1_note = real.train, 'real corpus only'
    elif args.sim_mode and args.sim_stage1 == 'mix':
        # One draw from the same variant groups stage 2 uses, kept UNCAPPED so the only difference
        # from 'real' is which pixels sit in each slot -- not how many frames per tracklet.
        pick = MixingIdentitySampler(
            ds.train, ds.train_tracklets, batch_size=cfg.SOLVER.STAGE2.IMS_PER_BATCH,
            num_instances=cfg.DATALOADER.NUM_INSTANCE, cap=0, seed=args.seed,
            variant_groups=ds.variant_groups)
        keep = pick._select(__import__('random').Random(args.seed))
        grouped = pick._grouped
        s1_items = [it for it, t in zip(ds.train, ds.train_tracklets)
                    if t not in grouped or t in keep]
        s1_note = 'one mix draw, uncapped'
    train_loader_stage1 = DataLoader(
        ImageDataset(s1_items, val_transforms), batch_size=cfg.SOLVER.STAGE1.IMS_PER_BATCH,
        shuffle=True, num_workers=cfg.DATALOADER.NUM_WORKERS, collate_fn=train_collate_fn)
    ds.stage1_note, ds.stage1_n = s1_note, len(s1_items)

    return (train_loader_stage2, train_loader_stage1, _val_loader(cfg, ds, val_transforms),
            len(ds.query), ds, sampler)


def main():
    ap = add_caliper_args(
        argparse.ArgumentParser(description="Finetune CLIP-ReID on the CALIPeR ground-field corpus"),
        default_config="configs/caliper_ground_clipreid.yml")
    add_sim_args(ap)
    ap.add_argument("--stage1-init", default="",
                    help='Reuse stage-1 prompts from this checkpoint. The adjacent run.json must '
                         'list the same training identities.')
    ap.add_argument("--sim-stage1", default="real", choices=("real", "mix", "naive"),
                    help='Stage-1 crops: real only (real), one sampled variant per group (mix), or '
                         'all variants (naive).')
    args = ap.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    # Set the stage-2 validation period before freezing the configuration.
    disable_eval_if_no_holdout(cfg, args.holdout, None,
                               key='SOLVER.STAGE2.EVAL_PERIOD',
                               max_key='SOLVER.STAGE2.MAX_EPOCHS')
    cfg.freeze()

    torch.manual_seed(cfg.SOLVER.SEED)
    torch.cuda.manual_seed_all(cfg.SOLVER.SEED)
    np.random.seed(cfg.SOLVER.SEED)
    random.seed(cfg.SOLVER.SEED)
    torch.backends.cudnn.benchmark = True

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    logger = setup_logger("transreid", cfg.OUTPUT_DIR, if_train=True)
    logger.info("Running with config:\n{}".format(cfg))
    logger.info("args: {}".format(vars(args)))
    require_single_gpu("processor_clipreid_stage1.py:28-30 would DataParallel across every card.")

    (train_loader_stage2, train_loader_stage1, val_loader,
     num_query, ds, sampler) = build_loaders(cfg, args)
    logger.info("num_classes={}  stage2 images/epoch={}  stage1 walks {} once [{}]"
                .format(ds.num_train_pids, len(sampler), ds.stage1_n, ds.stage1_note))
    log_composition(logger, ds.corpus, sampler, sim_mode=args.sim_mode, sim_p=args.sim_p,
                    batch_size=cfg.SOLVER.STAGE2.IMS_PER_BATCH, holdout=args.holdout,
                    unit="images")
    logger.info("holdout={} rule={} -> {} val ids of {} eligible; {} train ids"
                .format(args.holdout, args.holdout_rule, len(ds.corpus.val_ids),
                        len(ds.corpus.eligible), len(ds.corpus.train_ids)))
    logger.info("Training record: {}".format(
        write_run_json(cfg.OUTPUT_DIR, cfg=cfg, args=args, corpus=ds.corpus, repo_dir='.')))

    # Per-range validation, stage 2 only -- stage 1 computes no ReID metric. do_train_stage2 resolves
    # R1_mAP_eval by name at call time, so rebinding it on the module leaves processor/ untouched.
    if args.holdout:
        stage2_mod.R1_mAP_eval = make_by_range_metric(
            R1_mAP_eval, eval_func, cfg.OUTPUT_DIR,
            eval_period=cfg.SOLVER.STAGE2.EVAL_PERIOD, logger=logger, return_super=True)

    model = make_model(cfg, num_class=ds.num_train_pids,
                       camera_num=ds.num_train_cams, view_num=ds.num_train_vids)
    if args.init_weight:
        load_backbone(model, args.init_weight, logger, skip=('classifier', 'prompt_learner'))
    else:
        logger.info("no --init-weight: backbone is CLIP's pretrained tower only")
    if args.stage1_init:
        # Load the fitted prompts after initializing the encoder.
        load_stage1(model, args.stage1_init, ds.corpus, logger)
        if args.sim_mode:
            logger.info("--sim-stage1={} IGNORED: stage 1 is loaded, not run".format(args.sim_stage1))

    loss_func, center_criterion = make_loss(cfg, num_classes=ds.num_train_pids)

    if not args.stage1_init:
        optimizer_1stage = make_optimizer_1stage(cfg, model)
        scheduler_1stage = create_scheduler(
            optimizer_1stage, num_epochs=cfg.SOLVER.STAGE1.MAX_EPOCHS,
            lr_min=cfg.SOLVER.STAGE1.LR_MIN, warmup_lr_init=cfg.SOLVER.STAGE1.WARMUP_LR_INIT,
            warmup_t=cfg.SOLVER.STAGE1.WARMUP_EPOCHS, noise_range=None)
        do_train_stage1(cfg, model, train_loader_stage1, optimizer_1stage, scheduler_1stage, 0)

    optimizer_2stage, optimizer_center_2stage = make_optimizer_2stage(cfg, model, center_criterion)
    scheduler_2stage = WarmupMultiStepLR(
        optimizer_2stage, cfg.SOLVER.STAGE2.STEPS, cfg.SOLVER.STAGE2.GAMMA,
        cfg.SOLVER.STAGE2.WARMUP_FACTOR, cfg.SOLVER.STAGE2.WARMUP_ITERS,
        cfg.SOLVER.STAGE2.WARMUP_METHOD)
    do_train_stage2(cfg, model, center_criterion, train_loader_stage2, val_loader,
                    optimizer_2stage, optimizer_center_2stage, scheduler_2stage, loss_func,
                    num_query, 0)


if __name__ == '__main__':
    main()
