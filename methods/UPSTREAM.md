# Upstream sources

Each directory here is a copy of an upstream repository at the commit listed below, with CALIPeR
integration added on top. The upstream authors own the original code; see each repository for
authorship, citation and license terms.

For CALIPeR setup and evaluation, use the [benchmark README](../README.md) and
[training guide](../TRAINING.md). Method READMEs retain the upstream workflows below a CALIPeR
navigation note. The GEFF and X-TFCLIP top-level directories currently have no license file.
Some nested components include their own license notices.

Use the [path conventions](PORTABILITY.md) to set the working directory and locate datasets,
checkpoints, and outputs for an upstream workflow.

| directory | upstream | base commit | date |
|---|---|---|---|
| `reid-strong-baseline` (BoT) | https://github.com/michuanhaohao/reid-strong-baseline | `7fda2273cc656b4db65f33ef38962873c44658a4` | 2020-04-23 |
| `SINet` | https://github.com/baist/SINet | `252ea9876d88178f22b83e919b050df515bfaebd` | 2022-05-22 |
| `CLIP-ReID` | https://github.com/Syliz517/CLIP-ReID | `58f33db1c23956d2d322781203d36a7889e055c2` | 2023-11-21 |
| `PersonViT` | https://github.com/hustvl/PersonViT | `4e00536c64802418a94e170fbf7424f27cd96f46` | 2025-06-11 |
| `GEFF` (CAL) | https://github.com/bar371/GEFF | `45a765c25f7cacf8aabb4930812190d365720760` | 2024-04-28 |
| `X-TFCLIP` | https://github.com/BiDAlab/X-TFCLIP | `f6c32e68de2ddb98d530eb1a550db0f62a4bd007` | 2026-05-01 |
| `sapiensid` | https://github.com/mk-minchul/sapiensid | `bf55adc5e1028eff5653c15e753b65eeb1336105` | 2025-11-17 |

## Changes relative to upstream

Files marked *new* were added for CALIPeR; all other listed files are upstream files we modified.

**reid-strong-baseline**: `configs/caliper_ground_softmax_triplet.yml`, `data/datasets/caliper_ground.py`,
`data/datasets/kitbrc1.py`, `tools/train_caliper.py`, `tools/extract_features.py`,
`tools/eval_caliper_tracklet.py`, `extract_features.sh`, `Test-kitbrc1-ground_range-{image,video}.sh`
(*new*); `engine/trainer.py` (pytorch-ignite ≥ 0.3 checkpoint API; keeps every periodic checkpoint),
`config/defaults.py` (reads `CALIPER_DATA_ROOT`), `data/datasets/__init__.py` (registers `kitbrc1`),
`README.md`, `.gitignore`.

**SINet**: `caliper_ground.py`, `train_caliper.py`, `extract_caliper_features.py`, `extract_features.sh`,
`.gitignore` (*new*); `transforms/spatial_transforms.py` (`collections.abc` import for Python ≥ 3.10),
`README.md` (CALIPeR navigation).

**CLIP-ReID**: `configs/caliper_ground_clipreid.yml`, `datasets/caliper_ground.py`, `train_caliper.py`,
`extract_caliper_features.py`, `extract_features.sh` (*new*); `loss/triplet_loss.py` (removed an unused
import), `.gitignore`, `README.md` (CALIPeR navigation).

**PersonViT**: `transreid_pytorch/configs/caliper_ground_vit.yml`, `transreid_pytorch/datasets/caliper_ground.py`,
`transreid_pytorch/train_caliper.py`, `transreid_pytorch/extract_caliper_features.py`,
`transreid_pytorch/extract_features.sh`, `.gitignore` (*new*); `README.md` (CALIPeR navigation).

**GEFF**: `extract_caliper_features.py`, `extract_features.sh` (*new*); `.gitignore`,
`README.md` (CALIPeR navigation).

**X-TFCLIP**: `extract_caliper_features.py`, `extract_features.sh`, `.gitignore` (*new*);
`model/clip/bpe_simple_vocab_16e6.txt.gz` (*new*: the standard CLIP BPE vocabulary, needed by the
text encoder); `README.md` (CALIPeR navigation and link formatting).

**sapiensid**: `tasks/sapiensID/extract_caliper_features.py`, `tasks/sapiensID/extract_features.sh` (*new*);
`tasks/sapiensID/src/aligners/keypoint_predictor/__init__.py` (one-line indexing fix), `.gitignore`,
`README.md` (CALIPeR navigation).
