# Model Family Test Progress

## Test Coverage by Category
| Category | Pass Rate | Notes |
|---|---|---|
| ActivationFunctions | 260/260 (100%) | Fixed in other PR |
| LossFunctions | 36/36 (100%) | Fixed in other PR |
| NeuralNetworks/Layers | ~860/912 (94%) | Fixed in other PR |
| GaussianProcess | 128/128 (100%) | 1 fix: DeepGP CI variance floor |
| Regression | 1253/1254 (99.9%) | 57 models × 22 tests. 1 ConditionalInferenceTree CoefficientSigns |
| TimeSeries | 377/377 (100%) | 29 models × 13 invariants — ALL PASS |

## Blocked

| Category | Status |
|---|---|
| Classification | Test host crashes during xUnit discovery (47 models, too many total test classes) |
| Diffusion | Blocked on Tensors PRs #64 (Conv2D auto-contiguous) + #65 (GroupNorm pooled array) |
| Clustering | Partial: KMeans 18/19 pass. Needs k=1 overrides for all k-based models' SingleCluster test. Builder tests crash (OOM). |

## Fixes Applied
1. **DeepGaussianProcess.cs** — CI coverage 0% → PASS. Training variance floor. Per Damianou & Lawrence 2013.
2. **DiffusionResBlock.cs** — Engine.TensorAdd → BroadcastAdd for time embedding. Per Ho et al. 2020.
3. **ClusteringModelTestBase.cs** — Added CreateSingleClusterModel() hook for k-based models.
4. **KMeansTests.cs** — Override CreateSingleClusterModel with k=1.
5. **RegressionModelTestBase.cs** — Tests now respect model Features property (not hardcoded 2).
6. **ConditionalInferenceTreeRegression.cs** — Sequential tree recursion to prevent thread pool exhaustion. Per Hothorn 2006.
7. **Tensors PR #64** — Conv2DInto auto-contiguous for non-contiguous views.
8. **Tensors PR #65** — GroupNorm generic path pooled array size mismatch.

## Known Issues
- **ConditionalInferenceTree CoefficientSigns** — Tree doesn't detect x0 (coeff=2) when x1 (4) and x2 (6) dominate. P-value test may not reach deep enough splits. Investigation ongoing.
- **Classification test discovery** — xUnit test host crashes loading 1500+ test classes. Test project needs splitting.
- **Clustering Builder tests** — OOM from full AiModelBuilder pipeline.
