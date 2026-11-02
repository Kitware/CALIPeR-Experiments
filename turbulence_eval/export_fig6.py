"""Export Figure 6 from standalone or original v4-sim1 saved Re-ID results."""

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path

from .reid_data import sha256_file, write_json
from .reid_protocol import DEFAULT_MODELS, PROTOCOL, SCOPES, SEVERITIES, STEMS, folder_name


def cmc_curve(ranks, max_rank):
    return [sum(r <= k for r in ranks) / len(ranks) for k in range(1, max_rank + 1)]


def load_cells(results_dir, scopes, models=None):
    root = Path(results_dir)
    if not scopes or len(set(scopes)) != len(scopes) or set(scopes) - set(SCOPES):
        raise ValueError("Provide unique Figure 6 scopes: combined, brc1_only, brc2_only")
    if models is None:
        prefix = STEMS["real-500m"] + "__"
        found = {p.parent.name[len(prefix):] for p in (root / scopes[0]).glob(prefix + "*/results.json")}
        models = [m for m in DEFAULT_MODELS if m in found] + sorted(found - set(DEFAULT_MODELS))
    if not models or len(set(models)) != len(models):
        raise ValueError("No models found or duplicate model names")
    cells = {}
    for scope in scopes:
        for model in models:
            folder_name(model)
            for condition, stem in STEMS.items():
                path = root / scope / f"{stem}__{model}" / "results.json"
                result = json.loads(path.read_text())
                ret, prov = result["retrieval"], result["provenance"]
                if (prov.get("model") != model or Path(prov.get("split", "")).name != stem + ".json"
                        or prov.get("reference_embedder")):
                    raise ValueError(f"Model/split attribution or reference-embedder error: {path}")
                if prov.get("gallery_unit") != "tracklet" or ret.get("unmated_policy") != "exclude":
                    raise ValueError(f"Figure 6 requires tracklet retrieval and unmated=exclude: {path}")
                probes = ret["per_probe"]
                ranks = [p["rank"] for p in probes]
                if (not ranks or len(ranks) != ret["n_probes_scored"] or ret["n_gallery"] < 1
                        or any(r is None or not math.isfinite(r) or r < 1 for r in ranks)):
                    raise ValueError(f"Invalid per-query ranks/counts: {path}")
                if not math.isfinite(ret["mAP"]) or not 0 <= ret["mAP"] <= 1:
                    raise ValueError(f"Invalid mAP: {path}")
                for k in (1, 5, 10, 20):
                    value = ret["cmc"][f"rank{k}"]
                    actual = sum(r <= k for r in ranks) / len(ranks)
                    if not math.isfinite(value) or abs(actual - value) > 1e-9:
                        raise ValueError(f"CMC@{k} disagrees with per-query ranks: {path}")
                cells[(scope, model, condition)] = dict(path=path, retrieval=ret, provenance=prov, ranks=ranks)
    protocols = {c["provenance"].get("protocol_version") for c in cells.values()}
    if len(protocols) != 1 or next(iter(protocols)) not in (PROTOCOL, "v4-sim1"):
        raise ValueError(f"Unsupported or mixed evaluation protocols: {protocols}")
    protocol = next(iter(protocols))
    fingerprint_key = "evaluation_manifest_sha256" if protocol == PROTOCOL else "chip_index_sha256"
    hashes = {c["provenance"].get(fingerprint_key) for c in cells.values()}
    if len(hashes) != 1 or not next(iter(hashes)):
        raise ValueError(f"Missing or mixed {fingerprint_key}")
    for model in models:
        model_hashes = {c["provenance"].get("model_weights_sha256")
                        for (s, m, k), c in cells.items() if m == model}
        if len(model_hashes) != 1 or not next(iter(model_hashes)):
            raise ValueError(f"Missing or inconsistent checkpoint fingerprint for {model}")
        if protocol == PROTOCOL:
            signatures = {c["provenance"]["model_info"]["fingerprint"]
                          for (s, m, k), c in cells.items() if m == model}
            if len(signatures) != 1:
                raise ValueError(f"Model adapter/preprocessing differs between conditions for {model}")
    for scope in scopes:
        group = [c for (s, m, k), c in cells.items() if s == scope]
        if len({c["retrieval"]["n_gallery"] for c in group}) != 1:
            raise ValueError(f"Different gallery counts in {scope}")
        if protocol == PROTOCOL:
            gallery_hashes = {c["provenance"].get("gallery_sha256") for c in group}
            if len(gallery_hashes) != 1 or not next(iter(gallery_hashes)):
                raise ValueError(f"Different gallery content in {scope}")
        for model in models:
            anchor = cells[(scope, model, "real-500m")]["retrieval"]["per_probe"]
            for condition in SEVERITIES:
                probes = cells[(scope, model, condition)]["retrieval"]["per_probe"]
                if Counter(p["id"] for p in probes) != Counter(p["id"] for p in anchor):
                    raise ValueError(f"Query identity counts differ: {scope}/{model}/{condition}")
                if protocol == PROTOCOL:
                    if (any("query_id" not in p for p in probes + anchor)
                            or Counter((p["query_id"], p["id"]) for p in probes)
                            != Counter((p["query_id"], p["id"]) for p in anchor)):
                        raise ValueError(f"Query tracklet pairing differs: {scope}/{model}/{condition}")
    return cells, models


def _csv(path, rows):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _metadata(cells, models):
    colors = ["#0072B2", "#009E73", "#E69F00", "#CC79A7", "#D55E00", "#56B4E9", "#222222"]
    result = {}
    for index, model in enumerate(models):
        cell = next(c for (s, m, k), c in cells.items() if m == model)
        info = cell["provenance"].get("model_info", {})
        label, corpus, color = DEFAULT_MODELS.get(model, (model, "", colors[index % len(colors)]))
        result[model] = dict(label=info.get("label", label), corpus=info.get("training_dataset", corpus),
                             color=info.get("color") or color)
    return result


def _figures(cells, models, scope, output, meta, max_rank):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    suffix = "" if scope == "combined" else "_" + scope
    ks = list(range(1, max_rank + 1))
    styles = {"weak": ":", "medium": "-", "strong": "--"}
    written = []

    def save(fig, stem):
        fig.tight_layout()
        for extension in ("png", "pdf", "svg"):
            path = output / (stem + suffix + "." + extension)
            fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=.02)
            written.append(str(path))
        plt.close(fig)

    def curve(model, condition):
        return cmc_curve(cells[(scope, model, condition)]["ranks"], max_rank)

    fig, ax = plt.subplots(figsize=(5, 2.8))
    for model in models:
        for condition, style in styles.items():
            diff = [a - b for a, b in zip(curve(model, condition), curve(model, "real-1000m"))]
            ax.plot(ks, diff, color=meta[model]["color"], linestyle=style,
                    linewidth=1.6 if condition == "medium" else 1.2)
    ax.axhline(0, color="0.15", linewidth=1.4)
    ax.set(xlabel="Rank", ylabel="CMC Probability Difference\n(simulated − real 1000m)", xlim=(1, max_rank))
    ax.set_xticks([1] + list(range(5, max_rank + 1, 5)))
    ax.grid(alpha=.25)
    handles = [Line2D([], [], color=meta[m]["color"], label=meta[m]["label"]) for m in models]
    handles += [Line2D([], [], color=".25", linestyle=styles[k], label=f"{k} (Cn² {v:.0e})")
                for k, v in SEVERITIES.items()]
    ax.legend(handles=handles, loc="upper right", fontsize=6, title="Model / severity", title_fontsize=6, ncol=2)
    save(fig, "sim_severity_diff_1000m")
    for baseline, tag in (("real-1000m", "1000m"), ("real-500m", "500m")):
        fig, axes = plt.subplots(1, 3, figsize=(7, 3), sharey=True)
        for ax, condition in zip(axes, SEVERITIES):
            for model in models:
                ax.plot(ks, curve(model, condition), color=meta[model]["color"], label=meta[model]["label"])
                ax.plot(ks, curve(model, baseline), color=meta[model]["color"], linestyle="--")
            ax.set(title=f"{condition} (Cn² {SEVERITIES[condition]:.0e})", xlabel="Rank", xlim=(1, max_rank), ylim=(0, 1))
            ax.grid(alpha=.25)
        axes[0].set_ylabel("Cumulative Match Probability")
        axes[0].legend(fontsize=6, loc="lower right")
        fig.text(.5, .01, f"Solid: simulated 1000m; dashed: {baseline}", ha="center", fontsize=7)
        save(fig, "sim_severity_" + tag)
    return written


def export_results(results_dir, output_dir, *, scopes=("combined",), models=None, max_rank=20, figures=True):
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Report output directory already exists: {output}")
    if max_rank < 2:
        raise ValueError("Figure export requires max_rank >= 2")
    cells, models = load_cells(results_dir, scopes, models)
    meta = _metadata(cells, models)
    output.mkdir(parents=True)
    written = []
    for scope in scopes:
        rows = []
        for model in models:
            for condition in STEMS:
                ret = cells[(scope, model, condition)]["retrieval"]
                rows.append(dict(model=model, condition=condition, mAP=100 * ret["mAP"],
                                 **{f"R{k}": 100 * ret["cmc"][f"rank{k}"] for k in (1, 5, 10, 20)},
                                 n_probes=ret["n_probes_scored"], n_gallery=ret["n_gallery"]))
        path = output / f"sim_severity_{scope}.csv"
        _csv(path, rows)
        written.append(str(path))
        table = ["| Model | Condition | mAP (%) | R1 (%) | R5 (%) | R10 (%) | Queries |",
                 "|---|---|---:|---:|---:|---:|---:|"]
        table += [f"| {r['model']} | {r['condition']} | {r['mAP']:.1f} | {r['R1']:.1f} | {r['R5']:.1f} | {r['R10']:.1f} | {r['n_probes']} |" for r in rows]
        table += ["", "Fixed real 400m gallery. Simulated queries are 500m chips degraded to 1000m.",
                  "Real 1000m queries are a different population. CMC uses the first correct match per query tracklet.",
                  "Difference figures show CMC(simulated) − CMC(real 1000m); positive values mean simulated queries are easier.", ""]
        path = output / f"sim_severity_{scope}.md"
        path.write_text("\n".join(table))
        written.append(str(path))
        if figures:
            written += _figures(cells, models, scope, output, meta, max_rank)
    provenance = dict(results_dir=str(Path(results_dir).resolve()), scopes=list(scopes), models=models,
                      protocol=next(iter(cells.values()))["provenance"]["protocol_version"], max_rank=max_rank,
                      pairing="Exact source frames checked at preparation; query tracklet IDs checked at export."
                      if next(iter(cells.values()))["provenance"]["protocol_version"] == PROTOCOL else
                      "Legacy exporter contract: identity-count pairing; source-frame pairing belongs to the upstream importer.",
                      cells={"/".join(key): dict(path=str(c["path"].resolve()), sha256=sha256_file(c["path"]))
                             for key, c in cells.items()})
    path = output / "sim_severity_provenance.json"
    write_json(path, provenance)
    written.append(str(path))
    for path in written:
        print(f"wrote {path}", flush=True)
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path, help="New directory; existing reports are refused")
    parser.add_argument("--scopes", default="combined", help="Comma-separated combined,brc1_only,brc2_only")
    parser.add_argument("--models", help="Comma-separated model names; default: discover complete cells")
    parser.add_argument("--max-rank", type=int, default=20)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    export_results(args.results_dir, args.output_dir, scopes=args.scopes.split(","),
                   models=args.models.split(",") if args.models else None,
                   max_rank=args.max_rank, figures=not args.no_figures)


if __name__ == "__main__":
    main()
