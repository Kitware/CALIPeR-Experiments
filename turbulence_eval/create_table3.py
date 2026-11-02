"""Compare saved simulator outputs with real targets and write Table 3."""

import argparse
from pathlib import Path

from .table3 import create_table3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Read output_data_path and output_dir from an INI file.")
    parser.add_argument("--data-dir", type=Path, help="Completed generation directory; overrides INI output_data_path.")
    parser.add_argument("--output-dir", type=Path, help="New report directory; overrides INI or saved output_dir.")
    parser.add_argument("--num-images", type=int, default=None,
                        help="Override the saved analysis frame budget per subject per group (default: 5).")
    parser.add_argument("--resize-512", action=argparse.BooleanOptionalAction, default=None,
                        help="Override saved resizing; evaluate every image at 512x512.")
    parser.add_argument("--grayscale", action=argparse.BooleanOptionalAction, default=None,
                        help="Override saved grayscale conversion.")
    parser.add_argument("--trim-border", type=int, default=None,
                        help='Exclude this many native pixels from each edge of all real/synthetic '
                             'images before resizing.')
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel frame measurements.")
    parser.add_argument("--skip-brisque", action="store_const", const=False, default=None,
                        help="Omit BRISQUE for this report, producing five metric rows.", dest="include_brisque")
    args = parser.parse_args()
    if args.data_dir is None and args.config is None:
        parser.error("provide --config or --data-dir")
    create_table3(args.data_dir, config_file=args.config, output_dir=args.output_dir, include_brisque=args.include_brisque,
                  resize_512=args.resize_512, grayscale=args.grayscale, num_images=args.num_images,
                  trim_border=args.trim_border, workers=args.workers)


if __name__ == "__main__":
    main()
