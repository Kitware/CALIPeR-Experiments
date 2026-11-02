"""Tracklet retrieval with first-correct CMC, mean AP and mean INP."""

import numpy as np


def score_retrieval(query_features, gallery_features, queries, gallery, *, max_rank=20):
    """Score matched query tracklets using average tied ranks for CMC and AP."""
    q, g = np.asarray(query_features), np.asarray(gallery_features)
    if (q.ndim != 2 or g.ndim != 2 or q.shape[1] != g.shape[1]
            or len(q) != len(queries) or len(g) != len(gallery) or not len(g)
            or not np.isfinite(q).all() or not np.isfinite(g).all()):
        raise ValueError("Invalid query/gallery feature shapes or values")
    gallery_ids = np.array([t["canonical_id"] for t in gallery])
    per_probe, unmated = [], 0
    # Score bounded blocks rather than allocating the entire Q x G matrix.
    for start in range(0, len(q), 256):
        scores = q[start:start + 256] @ g.T
        for tracklet, similarities in zip(queries[start:start + 256], scores):
            correct = gallery_ids == tracklet["canonical_id"]
            if not correct.any():
                unmated += 1
                continue
            order = np.argsort(-similarities, kind="stable")
            ordered_scores = similarities[order]
            ordered_correct = correct[order]
            starts_block = np.r_[True, ordered_scores[1:] != ordered_scores[:-1]]
            starts = np.flatnonzero(starts_block)
            ends = np.r_[starts[1:], len(order)]
            blocks = np.cumsum(starts_block) - 1
            positions = ((starts + 1 + ends) / 2.0)[blocks][ordered_correct]
            # INP uses the end of the final positive's tie block, keeping it <=1.
            last_worst = ends[blocks][ordered_correct][-1]
            per_probe.append(dict(id=tracklet["canonical_id"], query_id=tracklet["key"],
                                  ap=float(min(1, np.mean(np.arange(1, len(positions) + 1) / positions))),
                                  inp=float(min(1, len(positions) / last_worst)), rank=float(positions[0])))
    if not per_probe:
        raise ValueError("No mated queries remain for retrieval scoring")
    return dict(mAP=float(np.mean([p["ap"] for p in per_probe])),
                mINP=float(np.mean([p["inp"] for p in per_probe])),
                cmc={f"rank{k}": sum(p["rank"] <= k for p in per_probe) / len(per_probe)
                     for k in sorted({1, 5, 10, 20, max_rank})},
                n_probes_scored=len(per_probe), n_probes_unmated=unmated, n_gallery=len(gallery),
                unmated_policy="exclude", per_probe=per_probe)
