# CALIPeR ReID — published results

Method names below link to the upstream repository each is built on; our forks, with the CALIPeR
integration, are under [`methods/`](methods/). BoT, SINet, CLIP-ReID and PersonViT include fine-tuning
code ([TRAINING.md](TRAINING.md)). CAL, X-TFCLIP and SapiensID include feature extraction only. The
SapiensID fork also carries one small upstream bugfix (an aligner indexing error).

## Headline: pre-trained vs. fine-tuned vs. fine-tuned + synthetic

Values are mAP / Rank-1 (%). Bold marks the best value in each column. The range columns
compare each released checkpoint (Pre-tr.) with fine-tuning on real crops (FT) and on real
and synthetic crops (FT+syn). The last four columns give zero-shot results. SapiensID was
evaluated zero-shot only.

| Method | Train Data | 500m Pre-tr. | 500m FT | 500m FT+syn | 1000m Pre-tr. | 1000m FT | 1000m FT+syn | Ground-vs-Elevated | Enroll-vs-Field | Same-collect-CC | Longitudinal |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| [BoT](https://github.com/michuanhaohao/reid-strong-baseline) | Market1501 | 44.7 / 89.8 | 50.3 / 95.7 | 51.1 / 96.0 | 8.9 / 11.0 | 30.8 / 53.4 | 37.5 / 69.1 | 3.2 / 13.4 | 3.2 / 8.1 | 4.5 / 8.0 | 2.4 / 5.8 |
| [SINet](https://github.com/baist/SINet) | LS-VID | 51.1 / 95.9 | **54.7** / 96.4 | **54.6** / 96.3 | 35.7 / 65.0 | 48.1 / 89.5 | **47.4** / **88.1** | 14.2 / 56.2 | 11.4 / 40.3 | 8.0 / 14.3 | 4.7 / 10.2 |
| [CLIP-ReID](https://github.com/Syliz517/CLIP-ReID) | MSMT17 | 51.0 / 95.8 | 54.5 / **96.6** | 54.5 / **96.9** | 40.9 / 77.0 | 48.9 / **91.2** | 46.7 / 87.6 | 23.8 / 72.0 | 29.3 / 67.9 | 11.4 / 12.7 | 8.0 / 17.5 |
| [PersonViT](https://github.com/hustvl/PersonViT) | MSMT17 | 52.3 / 96.1 | 51.3 / 96.2 | 52.1 / 96.4 | **49.7** / **93.5** | 44.6 / 87.1 | 43.9 / 84.7 | **34.6** / **85.9** | **34.6** / **81.6** | 13.9 / 14.7 | **10.3** / 18.3 |
| [SapiensID](https://github.com/mk-minchul/sapiensid) | WebBody-12M | **53.9** / **96.4** | N/A | N/A | 22.1 / 35.8 | N/A | N/A | 17.6 / 59.0 | 14.1 / 53.8 | **15.4** / **33.5** | 8.5 / **21.9** |

FT and FT+syn are single training runs (one seed each), read out at an epoch chosen on a held-out
validation split, not on the test split. See [TRAINING.md](TRAINING.md) for the settings and the
epoch-selection procedure.

## Full zero-shot survey (all methods, all released checkpoints)

This table includes every released checkpoint evaluated on CALIPeR. Values are mAP / Rank-1 (%).

| Method | Train Data | Ground-Range 500m | Ground-Range 1000m | Ground-vs-Elevated | Enroll-vs-Field | Same-collect-CC | Longitudinal |
|---|---|--:|--:|--:|--:|--:|--:|
| [BoT](https://github.com/michuanhaohao/reid-strong-baseline) | Market1501 | 44.7 / 89.8 | 8.9 / 11.0 | 3.2 / 13.4 | 3.2 / 8.1 | 4.5 / 8.0 | 2.4 / 5.8 |
| [SINet](https://github.com/baist/SINet) | MARS | 42.1 / 84.6 | 7.0 / 8.4 | 2.7 / 9.7 | 2.7 / 8.9 | 4.6 / 9.0 | 2.2 / 5.2 |
| [SINet](https://github.com/baist/SINet) | LS-VID | 51.1 / 95.9 | 35.7 / 65.0 | 14.2 / 56.2 | 11.4 / 40.3 | 8.0 / 14.3 | 4.7 / 10.2 |
| [CLIP-ReID](https://github.com/Syliz517/CLIP-ReID) | Market1501 | 49.6 / 95.9 | 24.9 / 42.8 | 17.9 / 60.8 | 21.1 / 50.0 | 10.9 / 15.0 | 6.5 / 15.3 |
| [CLIP-ReID](https://github.com/Syliz517/CLIP-ReID) | MSMT17 | 51.0 / 95.8 | 40.9 / 77.0 | 23.8 / 72.0 | 29.3 / 67.9 | 11.4 / 12.7 | 8.0 / 17.5 |
| [PersonViT](https://github.com/hustvl/PersonViT) | Market1501 | 51.3 / 96.2 | 44.8 / 83.6 | 27.9 / 74.6 | 28.2 / 70.4 | 12.4 / 13.0 | 10.1 / 17.9 |
| [PersonViT](https://github.com/hustvl/PersonViT) | MSMT17 | 52.3 / 96.1 | **49.7** / **93.5** | **34.6** / **85.9** | **34.6** / **81.6** | 13.9 / 14.7 | **10.3** / 18.3 |
| [SapiensID](https://github.com/mk-minchul/sapiensid) | WebBody-12M | **53.9** / 96.4 | 22.1 / 35.8 | 17.6 / 59.0 | 14.1 / 53.8 | **15.4** / **33.5** | 8.5 / **21.9** |
| [CAL](https://github.com/bar371/GEFF) | LaST | 47.8 / 94.2 | 14.4 / 21.6 | 5.0 / 25.9 | 7.8 / 17.8 | 6.1 / 13.5 | 4.2 / 9.1 |
| [X-TFCLIP](https://github.com/BiDAlab/X-TFCLIP) | AG-VPReID | 51.0 / 95.7 | 39.7 / 69.8 | 24.7 / 73.6 | 23.8 / 56.8 | 13.2 / 17.2 | 8.8 / 16.2 |
