"""Generate synthetic person crops or full frames from a configuration file."""

import argparse
from pathlib import Path

from configparser import ConfigParser


def generate_synthetic_data(config_file, *, output_data_path=None, output_dir=None, preview=False):
    """Select real frames and save the configured simulator outputs."""
    if output_data_path is not None and output_dir is not None:
        raise ValueError("Use output_data_path or its output_dir alias, not both")
    override = output_data_path if output_data_path is not None else output_dir
    config = ConfigParser(interpolation=None)
    with Path(config_file).open() as stream:
        config.read_file(stream)
    legacy = (config.has_option("paths", "stable_frames_dir")
              and not config.has_section("input")
              and not any(section.startswith("variant:") for section in config.sections()))
    if legacy:
        if preview:
            from .table3 import _select, load_config
            cohorts, excluded, selection = _select(load_config(config_file))
            result = dict(cohorts=cohorts, excluded=excluded,
                          selected_frames=sum(len(files) for groups in selection.values() for files in groups.values()))
            import json
            print(json.dumps(result, indent=2))
            return result
        from .table3 import generate_synthetic_data as legacy_generate
        return legacy_generate(config_file, output_data_path=override)
    from .pipeline import run_generation
    return run_generation(config_file, output_data_path=override, preview=preview)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Generation INI (chips by default), or legacy Table 3 INI.")
    parser.add_argument("--output-data-path", "--output-dir", dest="output_data_path", type=Path,
                        help="Override INI output_data_path; must be new. --output-dir is a compatibility alias.")
    parser.add_argument("--preview", action="store_true", help="Validate and summarize input selection without generating files.")
    args = parser.parse_args()
    generate_synthetic_data(args.config, output_data_path=args.output_data_path, preview=args.preview)


if __name__ == "__main__":
    main()
