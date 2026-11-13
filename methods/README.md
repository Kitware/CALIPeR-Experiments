# Methods and attribution

Repository-authored code and documentation use the root Apache-2.0 license.
Third-party code, model weights and datasets retain their upstream terms and
attribution. Use the [evaluation guide](../README.md) to run a method and the [training guide](../TRAINING.md)
to fine-tune a checkpoint. Please cite each method’s paper when reporting results.

## Find your method

| Method | CALIPeR guide | Paper or citation |
|---|---|---|
| BoT | [reid-strong-baseline](reid-strong-baseline/README.md) | [Bag of Tricks and a Strong Baseline](https://openaccess.thecvf.com/content_CVPRW_2019/papers/TRMTMCT/Luo_Bag_of_Tricks_and_a_Strong_Baseline_for_Deep_Person_CVPRW_2019_paper.pdf) |
| SINet | [SINet](SINet/README.md) | [Citation](SINet/README.md#citation) |
| CLIP-ReID | [CLIP-ReID](CLIP-ReID/README.md) | [Citation](CLIP-ReID/README.md#citation) |
| PersonViT | [PersonViT](PersonViT/README.md) | [PersonViT paper](https://arxiv.org/abs/2408.05398) |
| CAL | [GEFF](GEFF/README.md) | [Citation](GEFF/README.md#citation) |
| X-TFCLIP | [X-TFCLIP](X-TFCLIP/README.md) | [Challenge paper](X-TFCLIP/README.md#paper) |
| SapiensID | [SapiensID](sapiensid/README.md) | [SapiensID paper](https://arxiv.org/abs/2504.04708) |

## Component licenses and notices


| Component | Source and notice | License and attribution notes |
|---|---|---|
| `turbulence_eval/_vendor/daatsim.py`, `daatsim_config.py` | [DAATSim](https://github.com/Riponcs/DAATSim); [attribution and adaptation notice](../turbulence_eval/_vendor/DAATSIM_NOTICE.txt); [license](../turbulence_eval/_vendor/DAATSIM_LICENSE.txt) | CC BY-NC 4.0, including its noncommercial terms. Imported source revision was not recorded. |
| `turbulence_eval/_vendor/p2s.py` and P2S assets | [TurbulenceSimulatorPython](https://github.com/Riponcs/TurbulenceSimulatorPython); [retained license](../turbulence_eval/_vendor/P2S_LICENSE) | CC BY-NC-SA 4.0, including its noncommercial and ShareAlike terms. See the source header for the original Mao, Chimitt and Chan attribution. Imported source revision and separate weight provenance were not recorded. |
| `methods/sapiensid` | [SapiensID](https://github.com/mk-minchul/sapiensid); [license](sapiensid/LICENSE) | CC BY-NC 4.0; upstream notices retained. |
| `methods/sapiensid/tasks/WebBody/c_detect/yolov8n-pose.pt` and the Ultralytics dependency | [Ultralytics source](https://github.com/ultralytics/ultralytics); [AGPL-3.0 terms](https://www.ultralytics.com/legal/agpl-3-0-software-license) | Bundled checkpoint metadata identifies AGPL-3.0. Refer to the upstream terms for use and redistribution of the model and dependency. The exact checkpoint release was not recorded. |
| `methods/GEFF` | [Author repository](https://github.com/bar371/GEFF); [GEFF / ReFace paper](https://arxiv.org/abs/2211.13807); [upstream citation](GEFF/README.md#citation) | Author-released research implementation accompanying the paper. Please cite Arkushin, Cohen, Peleg and Fried when using this work. The copied snapshot has no standalone top-level license file; nested component notices are retained. |
| `methods/X-TFCLIP` | [Author repository](https://github.com/BiDAlab/X-TFCLIP); [AG-VPReID 2025 challenge paper](https://arxiv.org/abs/2506.22843); [upstream citation](X-TFCLIP/README.md#paper) | Author-released research implementation. The upstream README identifies MIT through its license badge and requests citation of the challenge paper. The copied snapshot does not include a standalone top-level license file. |
| Other vendored method code | [Upstream inventory](UPSTREAM.md) and licenses within each directory | Original notices, citations and upstream hashes are retained. Nested components retain their own terms. |

Use the citation entries in each method's README when reporting results or building
on its implementation. Preserve author credits and applicable license notices when
redistributing code. The pinned source revisions and CALIPeR adaptations are listed
in [methods/UPSTREAM.md](UPSTREAM.md).

## Shared retrieval metric

The Market1501 evaluator in `caliper_eval/eval_reid.py` is adapted from HaoLuo’s
[reid-strong-baseline](https://github.com/michuanhaohao/reid-strong-baseline).

MIT License

Copyright (c) [2019] [HaoLuo]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
