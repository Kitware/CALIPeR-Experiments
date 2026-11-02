"""Table 3 metrics over indexed real chips and saved synthetic-chip manifests."""

from collections import defaultdict
import json
from pathlib import Path

from .inputs import InputFrame, Selection, sources_for_variant


def _read_jsonl(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def frame_key(frame):
    return frame.collection, frame.relpath, frame.frame


def load_chip_comparison(data_dir):
    """Validate complete generation before creating a report directory."""
    generation = json.loads((data_dir / "generation.json").read_text())
    settings = generation["settings"]
    if generation.get("kind") != "synthetic_chips" or settings["input"].get("purpose") != "quality":
        raise ValueError("Chip quality analysis requires input.purpose = quality")
    sources = [InputFrame(**row) for row in json.loads((data_dir / "selection.json").read_text())["sources"]]
    real = [InputFrame(**row) for row in _read_jsonl(data_dir / "real_chips.jsonl")]
    selection = Selection(sources, {}, {}, real)
    variants = {v["name"]: v for v in settings["variants"]}
    source_index = {frame_key(frame): frame for frame in sources}
    if len(source_index) != len(sources):
        raise ValueError("Duplicate selected source chips")
    outputs = {name: {} for name in variants}
    for row in _read_jsonl(data_dir / "chips.jsonl"):
        name = row["name"]
        if name not in variants:
            raise ValueError(f"Unknown output variant: {name}")
        frame = InputFrame(**row["source"])
        key = frame_key(frame)
        if source_index.get(key) != frame or any(row.get(k) != value for k, value in variants[name].items()):
            raise ValueError(f"Synthetic-chip provenance disagrees for {name}/{key}")
        if key in outputs[name]:
            raise ValueError(f"Duplicate synthetic chip: {name}/{key}")
        path = Path(row["chip_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        outputs[name][key] = path
    by_target = defaultdict(dict)
    for name, variant in variants.items():
        target, simulator = variant["target_camera"], variant["simulator"]
        if simulator in by_target[target]:
            raise ValueError(f"Select one variant per simulator/target for Table 3: {target}/{simulator}")
        expected = {frame_key(r) for r in sources_for_variant(selection, variant)}
        if not expected or set(outputs[name]) != expected:
            raise ValueError(f"Incomplete synthetic-chip manifest for {name}")
        by_target[target][simulator] = name
    real_by_subject = defaultdict(lambda: defaultdict(list))
    for frame in real:
        if not Path(frame.path).is_file():
            raise FileNotFoundError(frame.path)
        real_by_subject[frame.canonical_id][frame.tok].append(frame.path)
    subject_sources = defaultdict(list)
    for frame in sources:
        subject_sources[frame.canonical_id].append(frame)
    groups = {}
    from .cameras import range_token
    for target, simulators in by_target.items():
        groups[target] = {}
        ordered = [name for name in ("pybsm", "daatsim", "p2s") if name in simulators]
        for subject, source_frames in sorted(subject_sources.items()):
            pools = real_by_subject[subject]
            if not pools[settings["source_token"]] or not pools[range_token(target)]:
                continue
            groups[target][subject] = dict(
                real_source=pools[settings["source_token"]], real_target=pools[range_token(target)],
                **{sim: [outputs[simulators[sim]][frame_key(frame)] for frame in source_frames] for sim in ordered},
            )
        if not groups[target]:
            raise ValueError(f"No common subject cohort for {target}")
    return generation, groups, by_target


def create_chip_table3(data_dir, *, output_dir=None, **overrides):
    from .quality import analyze_quality
    from .table3 import _report_path, _write_table3_report
    generation, groups, variants = load_chip_comparison(data_dir)
    settings = generation["settings"]
    output = Path(output_dir).resolve() if output_dir is not None else _report_path(settings, data_dir)
    if output == data_dir:
        raise ValueError("The report output_dir must differ from generated data")
    if output.exists():
        raise FileExistsError(output)
    analysis = settings["analysis"]
    analysis.update({key: value for key, value in overrides.items() if value is not None})
    if analysis["num_images"] < 1:
        raise ValueError("Analysis num_images must be positive")
    output.mkdir(parents=True, exist_ok=False)
    results = {}
    for target, subjects in groups.items():
        results[target] = analyze_quality(
            data_dir, data_dir, settings["source_camera"], target, output / target / "quality",
            **analysis, image_groups=subjects,
            simulations={sim: (data_dir, variants[target][sim]) for sim in next(iter(subjects.values()))
                         if sim not in ("real_source", "real_target")},
        )
    cohorts = {target: list(subjects) for target, subjects in groups.items()}
    _write_table3_report(data_dir, output, generation, results, cohorts)
    return results
