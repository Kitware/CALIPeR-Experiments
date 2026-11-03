# encoding: utf-8
"""Locate crop splits, feature caches, and results under REID_BENCH_ROOT."""
import os
import os.path as osp

DEFAULT_ROOT = 'reid-bench'
ROOT = os.environ.get('REID_BENCH_ROOT', DEFAULT_ROOT)


def splits_root():
    return osp.join(ROOT, 'splits')


def splits_dir(split):
    return osp.join(ROOT, 'splits', split)


def annot_path():
    return osp.join(ROOT, 'splits', 'annot_brc1.kwcoco.json')


def features_dir(split, method, source, dataset='combined_brc1_brc2'):
    return osp.join(ROOT, 'eval', dataset, split, method, source)
