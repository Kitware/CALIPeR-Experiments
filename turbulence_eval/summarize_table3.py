"""Summarize Table 3 differences across subjects using the median and scaled median absolute
deviation."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist, median


# Normal-consistent MAD scale: approximately 1.4826, without rounding the factor.
MAD_NORMAL_SCALE = 1 / NormalDist().inv_cdf(.75)


def _read_csv(path):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        if not columns or len(set(columns)) != len(columns):
            raise ValueError(f"Missing or duplicate CSV columns: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    return columns, rows


def _subject_rows(path, expected):
    columns, rows = _read_csv(path)
    if "subject" not in columns:
        raise ValueError(f"Missing subject column: {path}")
    subjects = [row["subject"] for row in rows]
    if len(set(subjects)) != len(subjects):
        raise ValueError(f"Duplicate subjects: {path}")
    if set(subjects) != set(expected):
        raise ValueError(f"Subjects do not match the original cohort: {path}")
    values = {}
    for column in columns:
        if column == "subject":
            continue
        data = [float(row[column]) for row in rows]
        if not all(math.isfinite(value) for value in data):
            raise ValueError(f"Non-finite values in {path}: {column}")
        values[column] = data
    return values


def _statistics(values):
    center = median(values)
    mad = median(abs(value - center) for value in values)
    return dict(median=center, robust_sd=MAD_NORMAL_SCALE * mad)


def summarize_table3(report_dir, output_dir=None, *, overwrite=False):
    """Read an existing report and write a new median report directory."""
    source = Path(report_dir).resolve()
    output = Path(output_dir).resolve() if output_dir is not None else source / "median"
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output}")
        previous_path = output / "run.json"
        if not previous_path.is_file():
            raise ValueError(f"Output has no median-report metadata: {output}")
        previous = json.loads(previous_path.read_text())
        if previous.get("aggregation") != "median" or previous.get("source_report") != str(source):
            raise ValueError(f"Output is not a median report from this source: {output}")
    run_path = source / "run.json"
    original = json.loads(run_path.read_text())
    source_camera = original["settings"]["source_camera"]
    table_path = source / "table3.csv"
    _, table = _read_csv(table_path)
    metrics = [row["metric"] for row in table]
    if len(set(metrics)) != len(metrics):
        raise ValueError(f"Duplicate metric rows: {table_path}")
    summaries, inputs = {}, [run_path, table_path]
    for target, cohort in original["cohorts"].items():
        if not cohort or len(set(cohort)) != len(cohort):
            raise ValueError(f"Missing or duplicate subjects in cohort {target}")
        folder = source / target / "quality"
        suffix = f"{source_camera}_to_{target}"
        values = {}
        for filename in (f"agreement_per_subject_{suffix}.csv", f"psd_distance_{suffix}.csv"):
            path = folder / filename
            current = _subject_rows(path, cohort)
            if set(values) & set(current):
                raise ValueError(f"Duplicate metric columns across per-subject CSVs: {target}")
            values.update(current)
            inputs.append(path)
        summaries[target] = {}
        for simulator in original["summary"][target]:
            summaries[target][simulator] = {}
            for metric in metrics:
                column = f"{simulator}|{metric}"
                if column not in values:
                    raise ValueError(f"Missing per-subject column for {target}: {column}")
                summaries[target][simulator][metric] = _statistics(values[column])

    comparisons = [(target, name) for target in summaries for name in summaries[target]]
    columns = [f"{target} / {name}" for target, name in comparisons]
    rows = []
    for metric in metrics:
        row = dict(metric=metric)
        for column, (target, name) in zip(columns, comparisons):
            for statistic, value in summaries[target][name][metric].items():
                row[column + " " + statistic] = value
        rows.append(row)
    markdown = ["| metric | " + " | ".join(columns) + " |",
                "|---|" + "---:|" * len(columns)]
    for metric in metrics:
        cells = []
        for target, name in comparisons:
            stats = summaries[target][name][metric]
            cells.append(f"{stats['median']:.4g} ± {stats['robust_sd']:.4g}")
        markdown.append("| " + metric + " | " + " | ".join(cells) + " |")
    markdown.extend([
        "", "Entries are median ± robust standard deviation across subjects.",
        "Scalar entries: median_subjects(|mean_frames_sim − mean_frames_target|).",
        "Within-subject frame means and PSD distances are reused from the original analysis.",
        "PSD L2 = RMSE; KL = KL(simulated || target), in nats. Lower is closer agreement.",
        "Robust SD = 1.4826 × median_subjects(|x − median_subjects(x)|), using the unrounded normal-consistency factor.",
        "The ± term describes subject-to-subject spread; subjects have equal weight.",
        "Cohort sizes: " + ", ".join(f"{target}: {len(original['cohorts'][target])}" for target in summaries) + ".",
        "Camera labels identify the calibration models used for the original analysis.", "",
    ])
    metadata = dict(
        aggregation="median", unit="subject", spread="robust_sd",
        spread_method="normal_consistent_mad", mad_scale=MAD_NORMAL_SCALE,
        spread_formula="mad_scale * median(abs(x - median(x)))", source_report=str(source),
        settings=original["settings"], cohorts=original["cohorts"],
        summary={target: {name: {metric: stats["median"] for metric, stats in values.items()}
                          for name, values in simulators.items()}
                 for target, simulators in summaries.items()},
        summary_robust_sd={target: {name: {metric: stats["robust_sd"] for metric, stats in values.items()}
                                    for name, values in simulators.items()}
                           for target, simulators in summaries.items()},
        input_sha256={str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in inputs},
    )
    output.mkdir(parents=True, exist_ok=overwrite)
    with (output / "table3.csv").open("w" if overwrite else "x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "table3.md").write_text("\n".join(markdown))
    (output / "run.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True, help="Existing Table 3 report directory.")
    parser.add_argument("--output-dir", type=Path, help="New directory; defaults to <report-dir>/median.")
    parser.add_argument("--overwrite", action="store_true", help="Update an existing median report from the same source.")
    args = parser.parse_args()
    summarize_table3(args.report_dir, args.output_dir, overwrite=args.overwrite)
    output = args.output_dir or args.report_dir / "median"
    print(f"Median Table 3 saved to {output.resolve() / 'table3.md'}")


if __name__ == "__main__":
    main()
