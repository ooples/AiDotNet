# Model Family Test Progress

## Completed
| Category | Pass/Total | Notes |
|---|---|---|
| ActivationFunctions | 260/260 | Fixed in other PR |
| LossFunctions | 36/36 | Fixed in other PR |
| NeuralNetworks/Layers | ~860/912 | Fixed in other PR |
| GaussianProcess | 128/128 | 1 fix: DeepGP CI coverage variance floor |

## In Progress
| Category | Files | Status |
|---|---|---|
| Clustering | 27 | MeanShift (3 failures), ConsensusClustering (1 failure). Rest unknown. |
| TimeSeries | 29 | TODO |
| Classification | 47 | TODO |
| Regression | 57 | TODO |
| Diffusion | 244 | BLOCKED on Tensors PRs #64 (Conv2D auto-contiguous) + #65 (GroupNorm pooled array) |

## Clustering — Per-Model Status
- [ ] AffinityPropagation
- [ ] AgglomerativeClustering
- [ ] BIRCH
- [ ] BisectingKMeans
- [ ] CLARANS
- [ ] ConsensusClustering — KNOWN FAIL
- [ ] DBSCAN
- [ ] DivisiveClustering
- [ ] FuzzyCMeans
- [ ] GMM
- [ ] HDBSClustering
- [ ] KMeans
- [ ] KMedoids
- [ ] MeanShift — KNOWN FAIL (3 tests, also very slow ~30 min)
- [ ] MiniBatchKMeans
- [ ] OPTICS
- [ ] SelfOrganizingMap
- [ ] SpectralBiclustering
- [ ] SpectralClustering
- [ ] SpectralCoClustering
- [ ] StreamingKMeans
- [ ] VBGMM
- [ ] WardHierarchical
- [ ] XMeans

## Fixes Applied
1. **DeepGaussianProcess.cs** — CI coverage 0% → PASS. Added training data variance as floor for noise level. Per Damianou & Lawrence 2013.
2. **DiffusionResBlock.cs** — Engine.TensorAdd → BroadcastAdd for time embedding [1,C,1,1] + feature map [1,C,H,W]. Per Ho et al. 2020.
3. **Tensors PR #64** — Conv2DInto auto-contiguous for non-contiguous view inputs (upstream).
4. **Tensors PR #65** — GroupNorm generic path pooled array size mismatch (upstream).
