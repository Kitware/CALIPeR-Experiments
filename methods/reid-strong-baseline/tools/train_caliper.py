# encoding: utf-8
"""Fine-tune BoT on CALIPeR ground-field imagery."""
import argparse
import os
import sys

import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader

sys.path.append('.')
import engine.trainer                                     # noqa: E402  (patched below)
from config import cfg                                    # noqa: E402
from data.collate_batch import train_collate_fn, val_collate_fn   # noqa: E402
from data.datasets import ImageDataset                    # noqa: E402
from data.datasets.caliper_ground import CaliperGround        # noqa: E402
from data.datasets.eval_reid import eval_func             # noqa: E402
from data.transforms import build_transforms              # noqa: E402
from engine.trainer import do_train                       # noqa: E402
from layers import make_loss                              # noqa: E402
from modeling import build_model                          # noqa: E402
from solver import make_optimizer, WarmupMultiStepLR      # noqa: E402
from utils.logger import setup_logger                     # noqa: E402
from utils.reid_metric import R1_mAP                      # noqa: E402

from caliper_eval.train_corpus import HOLDOUT_RULES, SIM_MODES   # noqa: E402
from caliper_eval.train_metrics import make_by_range_metric  # noqa: E402
from caliper_eval.train_metadata import write_run_json, append_epoch_map   # noqa: E402
from caliper_eval.train_sampler import ResamplingIdentitySampler, MixingIdentitySampler  # noqa: E402


def require_single_gpu():
    """Check that exactly one GPU is visible to the trainer."""
    n = torch.cuda.device_count()
    if n != 1:
        raise SystemExit(
            "Training requires one GPU. Found {} visible devices.\n"
            "  engine/trainer.py wraps the model in nn.DataParallel when device_count() > 1, so this\n"
            "  run would silently spread across all of them.\n"
            "  Fix: sbatch --gres=gpu:1, or CUDA_VISIBLE_DEVICES=<n> for an interactive run."
            .format(n))


def load_backbone(model, path, logger):
    """Load pretrained backbone weights and report coverage, skipping the classifier when
    transferring between datasets."""
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    own = model.state_dict()
    loaded, skipped_cls, shape_mismatch, missing = 0, 0, [], []
    for k, v in ckpt.items():
        if 'classifier' in k:
            skipped_cls += 1
            continue
        if k not in own:
            missing.append(k)
        elif own[k].shape != v.shape:
            shape_mismatch.append((k, tuple(own[k].shape), tuple(v.shape)))
        else:
            own[k].copy_(v)
            loaded += 1
    unfilled = [k for k in own if 'classifier' not in k and k not in ckpt]
    logger.info("init from {}".format(path))
    logger.info("  loaded {} tensors | classifier keys skipped {} (head is re-initialised) | "
                "in ckpt but not in model {} | shape mismatch {} | in model but not in ckpt {}"
                .format(loaded, skipped_cls, len(missing), len(shape_mismatch), len(unfilled)))
    for k, a, b in shape_mismatch[:5]:
        logger.info("    shape mismatch {}: model {} vs ckpt {}".format(k, a, b))
    for k in unfilled[:5]:
        logger.info("    Missing weights: {}".format(k))
    if shape_mismatch or unfilled:
        raise SystemExit("Incomplete backbone weights. Check the tensor names and shapes listed above.")
    return loaded


def build_loaders(cfg, args):
    ds = CaliperGround(holdout=args.holdout, holdout_rule=args.holdout_rule, seed=args.seed,
                     chip_root=args.chip_root or None,
                     sim_mode=args.sim_mode, sim_p=args.sim_p,
                     sim_chip_root=args.sim_chip_root or None,
                     sim_severities=(tuple(x.strip() for x in args.sim_severity.split(','))
                                     if args.sim_severity else None))
    # Apply the same resize and augmentation to real and synthetic crops.
    train_set = ImageDataset(ds.train, build_transforms(cfg, is_train=True))
    if ds.variant_groups:
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
    train_loader = DataLoader(train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, sampler=sampler,
                              num_workers=cfg.DATALOADER.NUM_WORKERS, collate_fn=train_collate_fn,
                              pin_memory=True)

    # With an empty holdout, schedule validation after the last training epoch.
    val_items = ds.query + ds.gallery
    val_loader = DataLoader(ImageDataset(val_items, build_transforms(cfg, is_train=False)),
                            batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False,
                            num_workers=cfg.DATALOADER.NUM_WORKERS, collate_fn=val_collate_fn)
    return train_loader, val_loader, len(ds.query), ds.num_train_pids, sampler, ds


EXPECTED_EPOCH = {None: 75216, 'add': 89468, 'dup': 89468, 'mix': 75216, 'swapall': 75216}
EXPECTED_1000M_SUBJECTS = {None: 104, 'add': 191, 'dup': 104, 'mix': 191, 'swapall': 191}


def log_composition(logger, ds, sampler, args, cfg):
    """Report the training population and check the expected epoch size and long-range identity
    count."""
    c = ds.corpus
    logger.info("sim_mode=%s sim_p=%s  -> %s", args.sim_mode, args.sim_p, c.stats())
    comp = c.range_composition()
    logger.info("range composition (subjects / tracklets / chips / capped-img), ALL variants:")
    for tok in ('10m', '160m', '250m', '400m', '500m', '1000m'):
        if tok in comp:
            logger.info("    %6s  %4d / %5d / %8d / %7d", tok, *comp[tok])

    n1k = comp.get('1000m', (0,))[0]
    logger.info("1000m-labelled subjects: %d", n1k)
    if hasattr(sampler, 'epoch_composition'):
        tot, syn = sampler.epoch_composition(0)
        logger.info("epoch 0 draw: %d capped images, %d synthetic (%.1f%%)",
                    tot, syn, 100.0 * syn / max(tot, 1))
    bs = cfg.SOLVER.IMS_PER_BATCH
    logger.info("len(sampler)=%d -> %d iterations at batch %d", len(sampler), len(sampler) // bs, bs)

    # Only assert on the exact configuration the plan pins down; a holdout run legitimately differs.
    if args.holdout:
        return
    want_n = EXPECTED_EPOCH.get(args.sim_mode)
    want_ids = EXPECTED_1000M_SUBJECTS.get(args.sim_mode)
    bad = []
    if want_n is not None and len(sampler) != want_n:
        bad.append("epoch is %d images, expected %d" % (len(sampler), want_n))
    if want_ids is not None and n1k != want_ids:
        bad.append("1000m-labelled subjects is %d, expected %d" % (n1k, want_ids))
    if bad:
        raise SystemExit(
            "refusing to train: the corpus does not match the plan for sim_mode=%s.\n  %s\n"
            "  Check the training manifest and sampling options." % (args.sim_mode, "\n  ".join(bad)))


def main():
    ap = argparse.ArgumentParser(description="Finetune BoT on the CALIPeR ground-field corpus")
    ap.add_argument("--config_file", default="configs/caliper_ground_softmax_triplet.yml")
    ap.add_argument("--init-weight", default="",
                    help="ReID checkpoint for the backbone; classifier keys are skipped. "
                         "Empty = ImageNet init only.")
    ap.add_argument("--holdout", type=int, default=0,
                    help='Number of identities reserved for validation. Zero trains on every identity.')
    ap.add_argument("--holdout-rule", default="random", choices=sorted(HOLDOUT_RULES),
                    help='Select validation identities at random or by the fewest 1000m samples. '
                         'thin-relaxed requires 400m and 1000m coverage. The other rules also '
                         'require 500m.')
    ap.add_argument("--sim-mode", default=None, choices=sorted(SIM_MODES),
                    help='Training variants: add a synthetic twin, duplicate the real tracklet '
                         '(dup), sample real or synthetic (mix), or replace real with synthetic '
                         '(swapall). Omit for real-only training.')
    ap.add_argument("--sim-p", type=float, default=0.5,
                    help='Probability of choosing a synthetic tracklet in mix mode, redrawn each epoch.')
    ap.add_argument("--sim-severity", default="",
                    help='Comma-separated turbulence strengths, chosen from cn2_1e-13, cn2_4e-13, '
                         'and cn2_1e-12. The default uses all three.')
    ap.add_argument("--sim-chip-root", default="",
                    help='Directory containing synthetic training crops.')
    ap.add_argument("--frames-per-tracklet", type=int, default=32, help="per-epoch cap; 0 = all")
    ap.add_argument("--chip-root", default="", help="override the chip root (e.g. a /dev/shm stage)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = ap.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    if args.init_weight:
        # build_model would otherwise load ImageNet weights that load_backbone immediately overwrites,
        # and 'imagenet' additionally requires MODEL.PRETRAIN_PATH to be a real file.
        cfg.MODEL.PRETRAIN_CHOICE = 'none'
    cfg.freeze()

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    logger = setup_logger("reid_baseline", cfg.OUTPUT_DIR, 0)
    logger.info("Running with config:\n{}".format(cfg))
    logger.info("args: {}".format(vars(args)))
    require_single_gpu()          # before anything expensive; see the docstring
    cudnn.benchmark = True

    train_loader, val_loader, num_query, num_classes, sampler, ds = build_loaders(cfg, args)
    logger.info("num_classes={}  images/epoch={}  (corpus is larger; the sampler re-draws each epoch)"
                .format(num_classes, len(sampler)))
    logger.info("holdout={} rule={} -> {} val ids of {} eligible; {} train ids"
                .format(args.holdout, args.holdout_rule, len(ds.corpus.val_ids),
                        len(ds.corpus.eligible), len(ds.corpus.train_ids)))
    log_composition(logger, ds, sampler, args, cfg)
    logger.info("Training record: {}".format(
        write_run_json(cfg.OUTPUT_DIR, cfg=cfg, args=args, corpus=ds.corpus, repo_dir='.')))

    # Use the per-range validation metric in the method’s trainer.
    if args.holdout:
        engine.trainer.R1_mAP = make_by_range_metric(
            R1_mAP, eval_func, cfg.OUTPUT_DIR,
            eval_period=cfg.SOLVER.EVAL_PERIOD, logger=logger)

    # Record the iteration for each epoch so the corresponding checkpoint can be located.
    _base_ckpt = engine.trainer.ModelCheckpoint

    class _EpochMappingCheckpoint(_base_ckpt):
        def __call__(self, engine_, *a, **kw):
            _base_ckpt.__call__(self, engine_, *a, **kw)
            append_epoch_map(cfg.OUTPUT_DIR, engine_.state.epoch, engine_.state.iteration)

    engine.trainer.ModelCheckpoint = _EpochMappingCheckpoint

    model = build_model(cfg, num_classes)
    if args.init_weight:
        load_backbone(model, args.init_weight, logger)
    else:
        logger.info("no --init-weight: backbone is whatever MODEL.PRETRAIN_CHOICE gave "
                    "({})".format(cfg.MODEL.PRETRAIN_CHOICE))

    optimizer = make_optimizer(cfg, model)
    loss_func = make_loss(cfg, num_classes)
    scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA,
                                  cfg.SOLVER.WARMUP_FACTOR, cfg.SOLVER.WARMUP_ITERS,
                                  cfg.SOLVER.WARMUP_METHOD)

    do_train(cfg, model, train_loader, val_loader, optimizer, scheduler, loss_func, num_query,
             start_epoch=0)


if __name__ == '__main__':
    main()
