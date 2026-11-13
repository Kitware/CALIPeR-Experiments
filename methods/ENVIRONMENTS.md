# CALIPeR method environments

Use **Python 3.10** for the method entrypoints. The NumPy-only shared evaluator
still supports Python 3.7+. Install one method's requirements in its own virtual
environment, from the repository root. For example, for BoT:

```bash
python3.10 -m venv .venv-bot
source .venv-bot/bin/activate
python -m pip install -r methods/requirements/bot.txt -e .
python -m pip check
```

The shared requirements pin Torch 2.12.0, torchvision 0.27.0, and NumPy 1.26.4.
Install matching CPU or CUDA Torch wheels for your hardware. Each method file adds
the packages its CALIPeR extractor or trainer needs.

| Method | Requirements from repository root | Working directory | Check the installation |
|---|---|---|---|
| BoT | `methods/requirements/bot.txt` | `methods/reid-strong-baseline` | `python tools/extract_features.py --help` and `python tools/train_caliper.py --help` |
| SINet | `methods/requirements/sinet.txt` | `methods/SINet` | `python extract_caliper_features.py --help` and `python train_caliper.py --help` |
| CLIP-ReID | `methods/requirements/clip-reid.txt` | `methods/CLIP-ReID` | `python extract_caliper_features.py --help` and `python train_caliper.py --help` |
| PersonViT | `methods/requirements/personvit.txt` | `methods/PersonViT/transreid_pytorch` | `python extract_caliper_features.py --help` and `python train_caliper.py --help` |
| CAL | `methods/requirements/cal.txt` | `methods/GEFF` | `python extract_caliper_features.py --help` |
| X-TFCLIP | `methods/requirements/x-tfclip.txt` | `methods/X-TFCLIP` | `python extract_caliper_features.py --help` |
| SapiensID | `methods/requirements/sapiensid.txt` | `methods/sapiensid/tasks/sapiensID` | `python extract_caliper_features.py --help` |

SapiensID can use its Python fallback for the optional `rpe_ops` extension. For checkpoint
inference or training, supply the weights and data listed in the [benchmark README](../README.md).
Use the versions here for CALIPeR, including pytorch-ignite 0.5.2 for BoT.

To build the WebBody dataset, install the [WebBody requirements](sapiensid/tasks/WebBody/requirements.txt),
which include Fabric, MXNet, and parquet support. With `--delete`, WebBody checks that every
output shard is complete and that the inputs still match before removing source files.

The simulation stack has its own [requirements](../turbulence_eval/requirements.txt).
Third-party licenses and attribution are documented in
[the methods README](README.md).
