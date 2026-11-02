# Adapted from HaoLuo/reid-strong-baseline (2019), MIT.
# See methods/README.md for the copyright and permission notice.
# encoding: utf-8
"""Compute cumulative match characteristics and mean average precision for ranked gallery
matches."""
import numpy as np


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50, cam_exclusion='none'):
    """Compute the CMC curve and mAP for queries with gallery matches. Also return the number of
    skipped queries."""
    if cam_exclusion not in ('market', 'none'):
        raise ValueError("cam_exclusion must be 'market' or 'none', got {!r}".format(cam_exclusion))
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    num_dropped_q = 0  # queries with zero gallery matches after exclusion
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        order = indices[q_idx]
        if cam_exclusion == 'market':
            # remove gallery samples that have the same pid and camid with query
            remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
            keep = np.invert(remove)
        else:
            keep = np.ones(len(order), dtype=bool)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            num_dropped_q += 1
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP, num_dropped_q
