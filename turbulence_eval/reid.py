"""Prepare, evaluate and report Figure 6 using configured external model adapters."""

import argparse
import gc
import json
from pathlib import Path
import os
import tempfile

from .reid_config import load_config
from .reid_data import digest_json, prepare, prepare_conditions, sha256_file
from .reid_protocol import PROTOCOL, STEMS, in_scope


def evaluate(config):
    from .reid_metrics import score_retrieval
    from .reid_models import adapter_imports, load_model, resolve_device, tracklet_feature
    import numpy as np

    output = Path(config["paths"]["output_dir"])
    manifest_path = output / "prepared" / "conditions.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["protocol"] != PROTOCOL or manifest["generation_dir"] != config["paths"]["generation_dir"]:
        raise ValueError("Prepared manifest protocol or generation_dir disagrees with configuration")
    if any(scope not in manifest["scopes"] for scope in config["scopes"]):
        raise ValueError("Requested evaluation scope was not prepared")
    # Verify even cache hits against the prepared image content before scoring.
    seen = set()
    for tracklet in manifest["gallery"] + [t for rows in manifest["queries"].values() for t in rows]:
        for f in tracklet["frames"]:
            if f["path"] not in seen:
                if sha256_file(f["path"]) != f["sha256"]:
                    raise ValueError(f"Image changed after preparation: {f['path']}")
                seen.add(f["path"])
    manifest_sha = sha256_file(manifest_path)
    device = resolve_device(config["device"])
    written = []
    for name, settings in config["models"].items():
        print(f"Loading {name} on {device}", flush=True)
        encoder, signature, identity = load_model(name, settings, device=device, batch_size=config["batch_size"])
        memory = {}
        try:
            def features(tracklets):
                vectors = []
                for t in tracklets:
                    key = digest_json(t["frames"])
                    if key not in memory:
                        memory[key] = tracklet_feature(t, encoder, settings, signature,
                                                       config["paths"]["cache_dir"], config["batch_size"])
                    vectors.append(memory[key])
                return np.stack(vectors)

            for scope in config["scopes"]:
                gallery = [t for t in manifest["gallery"] if in_scope(t, scope)]
                gallery_features = None
                gallery_signature = digest_json(gallery)
                for condition, stem in STEMS.items():
                    path = output / "results" / scope / f"{stem}__{name}" / "results.json"
                    run_signature = digest_json(dict(manifest=manifest_sha, model=signature, scope=scope,
                                                      condition=condition, scoring="cosine-average-ties-v1",
                                                      max_rank=config["max_rank"]))
                    if path.exists():
                        saved = json.loads(path.read_text())
                        if saved["provenance"].get("evaluation_signature") != run_signature:
                            raise ValueError(f"Existing result has different inputs/settings; use a new output_dir: {path}")
                        print(f"Reusing {scope}/{condition}/{name}", flush=True)
                        written.append(str(path))
                        continue
                    queries = [t for t in manifest["queries"][condition] if in_scope(t, scope)]
                    if gallery_features is None:
                        gallery_features = features(gallery)
                    ret = score_retrieval(features(queries), gallery_features, queries, gallery,
                                          max_rank=config["max_rank"])
                    info = dict(label=settings["label"], training_dataset=settings["corpus"],
                                color=settings["color"], input_kind=settings["input_kind"],
                                factory=settings["factory"], fingerprint=signature, identity=identity)
                    result = dict(retrieval=ret, provenance=dict(
                        protocol_version=PROTOCOL, evaluation_manifest_sha256=manifest_sha,
                        evaluation_signature=run_signature, gallery_sha256=gallery_signature,
                        model=name, model_weights_sha256=identity["files"][settings["checkpoint"]], model_info=info,
                        split=stem + ".json", condition=condition, scope=scope, gallery_unit="tracklet",
                        unmated_policy="exclude", tie_policy="average-rank; INP worst-block-end",
                        reference_embedder=False, device=device,
                        generation_sha256=manifest["generation_sha256"]))
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
                        tmp = Path(stream.name)
                        try:
                            json.dump(result, stream, indent=2, allow_nan=False)
                            stream.write("\n")
                        except BaseException:
                            tmp.unlink(missing_ok=True)
                            raise
                    os.replace(tmp, path)
                    written.append(str(path))
                    print(f"{scope}/{condition}/{name}: mAP={ret['mAP']:.4f}, R1={ret['cmc']['rank1']:.4f}", flush=True)
        finally:
            close = getattr(encoder, "close", None)
            if callable(close):
                with adapter_imports(settings.get("python_path")):
                    close()
            del encoder
            memory.clear()
            gc.collect()
    return written


def report(config):
    from .export_fig6 import export_results
    root = Path(config["paths"]["output_dir"])
    return export_results(root / "results", root / "report", scopes=config["scopes"],
                          models=list(config["models"]), max_rank=config["max_rank"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preview", "prepare", "evaluate", "report", "run"))
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.stage == "preview":
        manifest = prepare_conditions(config["paths"]["generation_dir"], config["scopes"], hash_images=False)
        print(json.dumps(dict(summary=manifest["summary"], models=config["models"], paths=config["paths"]), indent=2))
    elif args.stage == "prepare":
        print(json.dumps(prepare(config)["summary"], indent=2))
    elif args.stage == "evaluate":
        evaluate(config)
    elif args.stage == "report":
        report(config)
    else:
        manifest = Path(config["paths"]["output_dir"]) / "prepared" / "conditions.json"
        if not manifest.exists():
            prepare(config)
        evaluate(config)
        report(config)


if __name__ == "__main__":
    main()
