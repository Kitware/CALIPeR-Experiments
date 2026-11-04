"""Match evaluation features to the real and simulated query sets used for Figure 6."""

from collections import Counter
import json
import math
from pathlib import Path

from .reid_data import digest_json, prepare_conditions, sha256_file
from .reid_protocol import SCOPES, SEVERITIES, folder_name

PROTOCOL = "caliper-features-fig6-v1"


def evaluation_provenance(generation_dir, feature_paths, caches):
    """Validate all five cached populations before the shared evaluator scores them."""
    import numpy as np

    manifest = prepare_conditions(generation_dir)
    collections = {t["collection"] for rows in [manifest["gallery"], *manifest["queries"].values()] for t in rows}
    scope = {frozenset(("brc1", "brc2")): "combined", frozenset(("brc1",)): "brc1_only",
             frozenset(("brc2",)): "brc2_only"}[frozenset(collections)]
    tokens = {"real-500m": "500m", "real-1000m": "1000m",
              **{k: "500m_sim_1000m_" + v["name"] for k, v in manifest["variants"].items()}}
    by_token = {str(c["camera_token"]): (p, c) for p, c in zip(feature_paths[1:], caches[1:])}
    if len(by_token) != len(caches) - 1 or set(by_token) != set(tokens.values()):
        raise ValueError("Figure 6 needs exactly the two real and three generated query caches")
    roles = [("gallery", feature_paths[0], caches[0], manifest["gallery"])]
    roles += [(k, *by_token[token], manifest["queries"][k]) for k, token in tokens.items()]
    # Numeric identity labels must agree across roles: the shared scorer uses these labels.
    identities, fingerprints = {}, {}
    metadata = {key: str(caches[0][key]) for key in ("method", "source", "split", "weight", "neck_feat")}
    for condition, path, cache, tracklets in roles:
        if any(str(cache[k]) != v for k, v in metadata.items()):
            raise ValueError(f"Mixed model/checkpoint/preprocessing metadata: {path}")
        if str(cache.get("pooling", "")) not in ("", "video"):
            raise ValueError(f"Figure 6 requires video-compatible features: {path}")
        if any(len(cache[k]) != len(cache["feats"]) for k in ("video", "pid", "frame", "crop_path", "camid")):
            raise ValueError(f"Feature-cache rows have different lengths: {path}")
        subjects = json.loads(str(cache["pid_to_subject"]))
        for pid, subject in subjects.items():
            if identities.setdefault(pid, subject) != subject:
                raise ValueError("Identity labels differ between feature caches")
        expected = Counter((t["relpath"], t["canonical_id"], f["frame"], str(Path(f["path"]).resolve()))
                           for t in tracklets for f in t["frames"])
        actual = Counter((str(v), subjects[str(int(pid))], int(frame),
                          str((Path(str(cache["crop_root"])) / str(crop)).resolve()))
                         for v, pid, frame, crop in zip(cache["video"], cache["pid"],
                                                       cache["frame"], cache["crop_path"]))
        if (actual != expected or len(set(cache["video"])) != len(tracklets)
                or len(cache["feats"]) != sum(expected.values())):
            raise ValueError(f"Feature cache does not match generated {condition} frames/tracklets: {path}")
        if not np.isfinite(cache["feats"]).all() or np.any(np.linalg.norm(cache["feats"], axis=1) == 0):
            raise ValueError(f"Nonfinite or zero features: {path}")
        fingerprints[condition] = dict(path=str(Path(path).resolve()), sha256=sha256_file(path))
    # Multiple numeric labels for one subject would silently lose genuine matches.
    if len(set(identities.values())) != len(identities):
        raise ValueError("A subject has multiple numeric identity labels")
    weight = Path(metadata["weight"]).expanduser().resolve()
    return dict(protocol=PROTOCOL, scope=scope, tokens=tokens,
                evaluation_manifest_sha256=digest_json(manifest),
                gallery_sha256=digest_json(manifest["gallery"]),
                model_weights_sha256=sha256_file(weight), feature_files=fingerprints,
                counts=manifest["summary"]["combined"],
                pairing="Generation source frames and feature-cache rows checked before shared scoring.")


def load_evaluations(paths, scopes, max_rank):
    """Read shared evaluation JSONs without loading models, caches or image data."""
    if len(scopes) != 1 or scopes[0] not in SCOPES:
        raise ValueError("Shared results require one scope; generate/evaluate each collection separately")
    cells, inputs = {}, set()
    for model, filename in paths.items():
        folder_name(model)
        path = Path(filename)
        doc = json.loads(path.read_text())
        if (doc["pool_op"] != "mean" or doc["metric"] != "cosine" or doc["cam_exclusion"] != "none"
                or doc["max_per_video"] != 0 or doc["min_track_crops"] != 0):
            raise ValueError(f"Figure 6 requires mean/cosine, no camera exclusion, all frames: {path}")
        prov = doc.get("fig6")
        if not prov or prov["protocol"] != PROTOCOL:
            raise ValueError(f"Rescore with --fig6-generation-dir to record Figure 6 provenance: {path}")
        if prov["scope"] != scopes[0]:
            raise ValueError(f"Result collection scope differs from --scopes: {path}")
        if model != doc["method"] + "-" + doc["source"]:
            raise ValueError(f"Model name must match saved method-source: {path}")
        if not prov["model_weights_sha256"]:
            raise ValueError(f"Missing checkpoint fingerprint: {path}")
        inputs.add((prov["evaluation_manifest_sha256"], prov["gallery_sha256"]))
        if set(prov["tokens"]) != {"real-500m", "real-1000m", *SEVERITIES}:
            raise ValueError(f"Missing Figure 6 conditions: {path}")
        for condition, token in prov["tokens"].items():
            result = doc["results"]["video"][token]
            curve = result["cmc"]
            counts = prov["counts"]["conditions"][condition]
            if (len(curve) < min(max(20, max_rank), result["n_gallery"])
                    or not curve or any(not math.isfinite(v) or not 0 <= v <= 1 for v in curve)
                    or any(a > b for a, b in zip(curve, curve[1:]))
                    or not math.isfinite(result["mAP"]) or not 0 <= result["mAP"] <= 1
                    or result["n_gallery"] != prov["counts"]["gallery_tracklets"]
                    or result["n_query"] != counts["tracklets"]
                    or result["dropped_queries"] != counts["unmated"]):
                raise ValueError(f"Invalid CMC/counts for {condition}: {path}")
            # CMC remains constant beyond the final gallery candidate.
            curve = curve + [curve[-1]] * max(0, max(20, max_rank) - len(curve))
            for k in (1, 5, 10):
                if not math.isfinite(result[f"rank{k}"]) or abs(result[f"rank{k}"] - curve[k - 1]) > 1e-7:
                    raise ValueError(f"Rank summary disagrees with CMC: {path}")
            ret = dict(mAP=result["mAP"], cmc={f"rank{k}": curve[k-1] for k in (1, 5, 10, 20)},
                       n_probes_scored=counts["scored"], n_gallery=result["n_gallery"])
            cells[(scopes[0], model, condition)] = dict(
                path=path, retrieval=ret, curve=curve,
                provenance=dict(protocol_version=PROTOCOL, **{k: v for k, v in prov.items() if k != "protocol"}))
    if len(inputs) != 1 or any(not all(pair) for pair in inputs):
        raise ValueError("Models were evaluated on different generation/gallery inputs")
    return cells, list(paths)
