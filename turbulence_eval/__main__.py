"""Generate simulated imagery from a configuration file or camera settings."""

import argparse
import sys

from .cameras import CALIPER_CAMERA_CONFIGS


def main():
    # Config-driven chip generation is the primary workflow. Preserve the direct
    # image-directory CLI for existing callers.
    directory_flags = ("--input-dir", "--stable-frames-dir")
    if not any(arg == flag or arg.startswith(flag + "=") for arg in sys.argv[1:] for flag in directory_flags):
        from .generate_synthetic_data import main as generate_main
        return generate_main()
    from .adapter import SimulationOptions
    from .generate import generate_images
    parser = argparse.ArgumentParser(description="Simulate CALIPeR ground-camera imagery with pyBSM.")
    parser.epilog = "For indexed chips (default config format): python -m turbulence_eval --config turbulence_eval/configs/chips_test.ini [--preview]"
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input-dir", help="Flat directory of source images.")
    inputs.add_argument("--stable-frames-dir", help="Root containing <subject>/<camera>/ images.")
    parser.add_argument("--source-camera", required=True, choices=CALIPER_CAMERA_CONFIGS)
    parser.add_argument("--target-camera", required=True, choices=CALIPER_CAMERA_CONFIGS)
    parser.add_argument("--output-dir", required=True, help="New directory; existing directories are refused.")
    parser.add_argument("--max-images", type=int, default=-1, help="Images per subject; -1 means all.")
    parser.add_argument("--seed", type=int, default=42, help="Frame-sampling seed (default: 42).")
    parser.add_argument("--noise-seed", type=int, default=None, help="Optional reproducible sensor-noise seed.")
    parser.add_argument("--cn2", type=float, default=None, help="Override Cn2 for source AND target cameras.")
    parser.add_argument("--no-bridge", action="store_true")
    parser.add_argument("--no-deconv", action="store_true")
    parser.add_argument("--no-resample", action="store_true")
    parser.add_argument("--no-distortion", action="store_true")
    args = parser.parse_args()
    options = SimulationOptions(
        cn2=args.cn2, bridge=not args.no_bridge, deconvolve=not args.no_deconv,
        resample=not args.no_resample, distortion=not args.no_distortion,
        noise_seed=args.noise_seed,
    )
    files = generate_images(
        args.input_dir or args.stable_frames_dir, args.source_camera, args.target_camera,
        args.output_dir, subject_layout=args.stable_frames_dir is not None,
        max_images=args.max_images, seed=args.seed, options=options,
    )
    print(f"Wrote {len(files)} images to {args.output_dir}")


if __name__ == "__main__":
    main()
