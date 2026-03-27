# Model Family Test Progress

## Completed
| Category | Pass/Total | Notes |
|---|---|---|
| ActivationFunctions | 260/260 | Fixed in other PR |
| LossFunctions | 36/36 | Fixed in other PR |
| NeuralNetworks/Layers | ~860/912 | Fixed in other PR |
| GaussianProcess | 128/128 | 1 fix: DeepGP CI coverage variance floor |
| Regression | 1253/1254 | 57 models × 22 tests. 1 failure: ConditionalInferenceTree CoefficientSigns |

## In Progress
| Category | Files | Status |
|---|---|---|
| Clustering | 27 | KMeans 18/19 pass (Builder crashes). SingleCluster needs k=1 overrides. |
| Classification | 47 | TODO |
| TimeSeries | 29 | TODO |
| Diffusion | 244 | BLOCKED on Tensors PRs #64 + #65 |

## Fixes Applied
1. **DeepGaussianProcess.cs** — CI coverage 0% → PASS. Training variance floor. Per Damianou & Lawrence 2013.
2. **DiffusionResBlock.cs** — Engine.TensorAdd → BroadcastAdd for time embedding. Per Ho et al. 2020.
3. **ClusteringModelTestBase.cs** — Added CreateSingleClusterModel() hook for k-based models.
4. **KMeansTests.cs** — Override CreateSingleClusterModel with k=1.
5. **RegressionModelTestBase.cs** — Tests now use model's Features property instead of hardcoded 2.
   - MonotonicResponse, CoefficientSigns, FeaturePermutation: generalized to N features
   - IrrelevantFeature, FeaturePermutation: skip for univariate models (Features < 2)

## Known Issues
- **ConditionalInferenceTreeRegression** — CoefficientSigns fails (x0 effect=0). Tree doesn't split on x0 when x1 has larger effect. Likely p-value threshold issue per Hothorn 2006.
- **Clustering Builder tests** — Test host crashes (OOM from full AiModelBuilder pipeline).
- **Diffusion models** — Blocked on 2 upstream Tensors bugs (PRs #64, #65).

## Upstream Tensors Fixes
- **PR #64** — Conv2DInto auto-contiguous for non-contiguous views
- **PR #65** — GroupNorm pooled array size mismatch in generic path
