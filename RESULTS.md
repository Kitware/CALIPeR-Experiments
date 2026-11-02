# CALIPeR ReID — published results

## Headline: pre-trained vs. fine-tuned vs. fine-tuned + synthetic

Values are mAP / Rank-1 (%). Bold marks the best value in each column. The range columns
compare each released checkpoint (Pre-tr.) with fine-tuning on real crops (FT) and on real
and synthetic crops (FT+syn). The last four columns give zero-shot results. SapiensID was
evaluated zero-shot only.

| Method | Train Data | 500m Pre-tr. | 500m FT | 500m FT+syn | 1000m Pre-tr. | 1000m FT | 1000m FT+syn | Ground-vs-Elevated | Enroll-vs-Field | Same-collect-CC | Longitudinal |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| BoT [27] | Market1501 | 44.7 / 89.8 | 50.3 / 95.7 | 51.1 / 96.0 | 8.9 / 11.0 | 30.8 / 53.4 | 37.5 / 69.1 | 3.2 / 13.4 | 3.2 / 8.1 | 4.5 / 8.0 | 2.4 / 5.8 |
| SINet [1] | LS-VID | 51.1 / 95.9 | **54.7** / 96.4 | **54.6** / 96.3 | 35.7 / 65.0 | 48.1 / 89.5 | **47.4** / **88.1** | 14.2 / 56.2 | 11.4 / 40.3 | 8.0 / 14.3 | 4.7 / 10.2 |
| CLIP-ReID [23] | MSMT17 | 51.0 / 95.8 | 54.5 / **96.6** | 54.5 / **96.9** | 40.9 / 77.0 | 48.9 / **91.2** | 46.7 / 87.6 | 23.8 / 72.0 | 29.3 / 67.9 | 11.4 / 12.7 | 8.0 / 17.5 |
| PersonViT [15] | MSMT17 | 52.3 / 96.1 | 51.3 / 96.2 | 52.1 / 96.4 | **49.7** / **93.5** | 44.6 / 87.1 | 43.9 / 84.7 | **34.6** / **85.9** | **34.6** / **81.6** | 13.9 / 14.7 | **10.3** / 18.3 |
| SapiensID [19] | WebBody-12M | **53.9** / **96.4** | N/A | N/A | 22.1 / 35.8 | N/A | N/A | 17.6 / 59.0 | 14.1 / 53.8 | **15.4** / **33.5** | 8.5 / **21.9** |

FT and FT+syn use one seed per arm; these runs do not establish a noise floor.

### ⚠ Eval protocol note — Longitudinal column, verification TODO

`eval_reid.py` had a bug (now fixed, see git history) that unconditionally discarded gallery items
sharing `(pid, camid)` with the query — correct for Market1501, but on the Longitudinal
(cross-collection) split a true match can legitimately share a camera id with its query, so this
silently dropped genuine matches. Verified on BoT/Market1501: the old buggy code reproduced this
repo's previously-logged number (mAP 1.0 / R1 3.3) exactly; the fix moves it to 1.2 / 4.7, toward
the published 2.4 / 5.8. The remaining gap is assumed to be noise (low query count, low-single-digit
mAP, so a handful of rank-1 hits swings it by a point or two) — **not yet independently confirmed**.

**Follow-up to do before fully trusting this column:**
- Re-run the other 3 rows that had the same discrepancy on this column (SINet/LS-VID,
  PersonViT/MSMT17, SapiensID/WebBody-12M) with the fix and confirm the same pattern.
- Check whether the other "other modalities" columns (especially Same-collect-CC, which like
  Longitudinal spans separate collections/cameras) have the same camid-exclusion sensitivity, or
  whether it's isolated to Longitudinal.

## Full zero-shot survey (all methods, all released checkpoints)

This table includes every released checkpoint evaluated on CALIPeR. Values are mAP / Rank-1 (%).

| Method | Train Data | Ground-Range 500m | Ground-Range 1000m | Ground-vs-Elevated | Enroll-vs-Field | Same-collect-CC | Longitudinal |
|---|---|--:|--:|--:|--:|--:|--:|
| BoT [12] | Market1501 | 44.7 / 89.8 | 8.9 / 11.0 | 3.2 / 13.4 | 3.2 / 8.1 | 4.5 / 8.0 | 2.4 / 5.8 |
| SINet [2] | MARS | 42.1 / 84.6 | 7.0 / 8.4 | 2.7 / 9.7 | 2.7 / 8.9 | 4.6 / 9.0 | 2.2 / 5.2 |
| SINet [2] | LS-VID | 51.1 / 95.9 | 35.7 / 65.0 | 14.2 / 56.2 | 11.4 / 40.3 | 8.0 / 14.3 | 4.7 / 10.2 |
| CLIP-ReID [11] | Market1501 | 49.6 / 95.9 | 24.9 / 42.8 | 17.9 / 60.8 | 21.1 / 50.0 | 10.9 / 15.0 | 6.5 / 15.3 |
| CLIP-ReID [11] | MSMT17 | 51.0 / 95.8 | 40.9 / 77.0 | 23.8 / 72.0 | 29.3 / 67.9 | 11.4 / 12.7 | 8.0 / 17.5 |
| PersonViT [8] | Market1501 | 51.3 / 96.2 | 44.8 / 83.6 | 27.9 / 74.6 | 28.2 / 70.4 | 12.4 / 13.0 | 10.1 / 17.9 |
| PersonViT [8] | MSMT17 | 52.3 / 96.1 | **49.7** / **93.5** | **34.6** / **85.9** | **34.6** / **81.6** | 13.9 / 14.7 | **10.3** / 18.3 |
| SapiensID [9] | WebBody-12M | **53.9** / 96.4 | 22.1 / 35.8 | 17.6 / 59.0 | 14.1 / 53.8 | **15.4** / **33.5** | 8.5 / **21.9** |
| CAL [6] | LaST | 47.8 / 94.2 | 14.4 / 21.6 | 5.0 / 25.9 | 7.8 / 17.8 | 6.1 / 13.5 | 4.2 / 9.1 |
| X-TFCLIP [14] | AG-VPReID | 51.0 / 95.7 | 39.7 / 69.8 | 24.7 / 73.6 | 23.8 / 56.8 | 13.2 / 17.2 | 8.8 / 16.2 |

Bracketed numbers are bibliography references to each method's original paper — see the
publication for the full citation list.
