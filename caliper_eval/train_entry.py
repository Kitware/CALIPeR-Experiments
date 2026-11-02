# encoding: utf-8
"""Configure fine-tuning, load pretrained weights, and check the training corpus."""
import argparse

import torch


def require_single_gpu(why=""):
    """Check that exactly one GPU is visible to the trainer."""
    n = torch.cuda.device_count()
    if n != 1:
        raise SystemExit(
            "Training requires one GPU. Found {} visible devices.{}\n"
            "  Fix: set CUDA_VISIBLE_DEVICES=<n> before starting training."
            .format(n, ("\n  " + why) if why else ""))


def load_backbone(model, path, logger, skip=('classifier',), strict=True):
    """Load pretrained backbone weights and report coverage, skipping the classifier when
    transferring between datasets."""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    if isinstance(ckpt, dict) and 'model' in ckpt and isinstance(ckpt['model'], dict):
        ckpt = ckpt['model']

    def skipped(k):
        return any(s in k for s in skip)

    own = model.state_dict()
    loaded, n_skipped, mismatch, missing = 0, 0, [], []
    for k, v in ckpt.items():
        kk = k.replace('module.', '')
        if skipped(kk):
            n_skipped += 1
        elif kk not in own:
            missing.append(kk)
        elif own[kk].shape != v.shape:
            mismatch.append((kk, tuple(own[kk].shape), tuple(v.shape)))
        else:
            own[kk].copy_(v)
            loaded += 1
    ck = {k.replace('module.', '') for k in ckpt}
    unfilled = [k for k in own if not skipped(k) and k not in ck]

    logger.info("init from {}".format(path))
    logger.info("  loaded {} tensors | skipped {} ({}) | in ckpt but not in model {} | "
                "shape mismatch {} | in model but not in ckpt {}"
                .format(loaded, n_skipped, ", ".join(skip), len(missing), len(mismatch),
                        len(unfilled)))
    for k, a, b in mismatch[:5]:
        logger.info("    shape mismatch {}: model {} vs ckpt {}".format(k, a, b))
    for k in unfilled[:5]:
        logger.info("    Missing weights: {}".format(k))
    for k in missing[:5]:
        logger.info("    Only in checkpoint: {}".format(k))
    if strict and (mismatch or unfilled):
        raise SystemExit("Incomplete backbone weights. Check the tensor names and shapes listed above.")
    return loaded


def disable_eval_if_no_holdout(cfg, holdout, logger, key='SOLVER.EVAL_PERIOD',
                               max_key='SOLVER.MAX_EPOCHS'):
    """Move validation past the final epoch when the holdout is empty. Return whether the
    configuration changed."""
    def get(node, path):
        for part in path.split('.'):
            node = getattr(node, part)
        return node

    def setv(root, path, value):
        parts = path.split('.')
        node = root
        for part in parts[:-1]:
            node = getattr(node, part)
        setattr(node, parts[-1], value)

    if holdout:
        return False
    max_epochs = get(cfg, max_key)
    if get(cfg, key) > max_epochs:
        return False
    was = get(cfg, key)
    setv(cfg, key, max_epochs + 1)
    msg = ("No validation identities. Setting {} from {} to {}."
           .format(key, was, max_epochs + 1))
    (logger.info if logger else print)(msg)
    return True


def add_sim_args(ap):
    """Add training options for synthetic crops, sampling probability, and turbulence strength."""
    from .train_corpus import SIM_MODES
    ap.add_argument("--sim-mode", default=None, choices=sorted(SIM_MODES),
                    help='Training variants: add a synthetic twin, duplicate the real tracklet '
                         '(dup), sample real or synthetic (mix), or replace real with synthetic '
                         '(swapall). Omit for real-only training.')
    ap.add_argument("--sim-p", type=float, default=0.5,
                    help='Probability of choosing a synthetic tracklet in mix mode, redrawn each epoch.')
    ap.add_argument("--sim-chip-root", default="",
                    help='Directory containing synthetic training crops.')
    ap.add_argument("--sim-severity", default="",
                    help='Comma-separated turbulence strengths, chosen from cn2_1e-13, cn2_4e-13, '
                         'and cn2_1e-12. The default uses all three.')
    return ap


def parse_severities(arg):
    """Parse a comma-separated list of strengths. An empty value selects all strengths."""
    if not arg:
        return None
    return tuple(x.strip() for x in arg.split(',') if x.strip())


# Expected full-corpus counts for image training. Video trainers supply tracklet counts.
EXPECTED_EPOCH_IMAGES = {None: 75216, 'add': 89468, 'dup': 89468, 'mix': 75216, 'swapall': 75216}
EXPECTED_1000M_SUBJECTS = {None: 104, 'add': 191, 'dup': 104, 'mix': 191, 'swapall': 191}


def log_composition(logger, corpus, sampler, sim_mode=None, sim_p=None, batch_size=None,
                    holdout=0, expected_epoch=None, expected_1000m=None, unit="images"):
    """Report the training population and check the expected epoch size and long-range identity
    count."""
    if expected_epoch is None:
        expected_epoch = EXPECTED_EPOCH_IMAGES
    if expected_1000m is None:
        expected_1000m = EXPECTED_1000M_SUBJECTS
    say = (logger.info if logger is not None else print)
    n = len(sampler) if hasattr(sampler, '__len__') else int(sampler)

    say("sim_mode={} sim_p={}  -> {}".format(sim_mode, sim_p, corpus.stats()))
    comp = corpus.range_composition()
    say("range composition (subjects / tracklets / chips / capped-img), ALL variants:")
    for tok in ('10m', '160m', '250m', '400m', '500m', '1000m'):
        if tok in comp:
            say("    {:>6s}  {:4d} / {:5d} / {:8d} / {:7d}".format(tok, *comp[tok]))

    n1k = comp.get('1000m', (0,))[0]
    say("1000m-labelled subjects: {}".format(n1k))
    if hasattr(sampler, 'epoch_composition'):
        tot, syn = sampler.epoch_composition(0)
        say("epoch 0 draw: {} capped {}, {} synthetic ({:.1f}%)"
            .format(tot, unit, syn, 100.0 * syn / max(tot, 1)))
    if batch_size:
        say("len(sampler)={} {} -> {} iterations at batch {}"
            .format(n, unit, n // batch_size, batch_size))
    else:
        say("len(sampler)={} {}".format(n, unit))

    # Only assert on the exact configuration the plan pins down; a holdout run legitimately differs.
    if holdout:
        say("--holdout={} set, so the corpus legitimately differs from the plan's numbers: "
            "composition asserts skipped.".format(holdout))
        return
    want_n = expected_epoch.get(sim_mode)
    want_ids = expected_1000m.get(sim_mode)
    bad = []
    if want_n is not None and n != want_n:
        bad.append("epoch is {} {}, expected {}".format(n, unit, want_n))
    if want_ids is not None and n1k != want_ids:
        bad.append("1000m-labelled subjects is {}, expected {}".format(n1k, want_ids))
    if bad:
        raise SystemExit(
            "refusing to train: the corpus does not match the plan for sim_mode={}.\n  {}\n"
            "  Check the training manifest and sampling options.".format(sim_mode, "\n  ".join(bad)))
    say("composition OK for sim_mode={}: {} {}/epoch, {} 1000m-labelled subjects"
        .format(sim_mode, n, unit, n1k))


def add_caliper_args(ap, default_config="", default_init=""):
    """Add the shared CALIPeR training options to the argument parser."""
    from .train_corpus import HOLDOUT_RULES
    ap.add_argument("--config_file", default=default_config)
    ap.add_argument("--init-weight", default=default_init,
                    help="pretrained ReID checkpoint; class-dependent keys are skipped")
    ap.add_argument("--holdout", type=int, default=0,
                    help='Number of identities reserved for validation. Zero trains on every identity.')
    ap.add_argument("--holdout-rule", default="random", choices=sorted(HOLDOUT_RULES),
                    help='Select validation identities at random or by the fewest 1000m samples. '
                         'thin-relaxed requires 400m and 1000m coverage. The other rules also '
                         'require 500m.')
    ap.add_argument("--frames-per-tracklet", type=int, default=32, help="per-epoch cap; 0 = all")
    ap.add_argument("--chip-root", default="", help="override the chip root (e.g. a /dev/shm stage)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    return ap
