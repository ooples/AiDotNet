# IFullModel migration audit

**Question (from the facade-config work):** many model-like components "should eventually inherit
`IFullModel`." How big is that migration, and which components genuinely need it?

**Headline:** far smaller than it looks. `ModelBase<T, TInput, TOutput>` implements `IFullModel`, and
the domain interfaces almost every model derives from **already inherit `IFullModel`**. Most of the
"non-IFullModel" interfaces a naïve scan turns up are false positives — they reach `IFullModel`
transitively; the scan only read their *direct* declaration.

## How this was measured

Scanned every `public interface I*Model / I*Detector / I*Classifier / I*Regression / I*Forecaster`
under `src/Interfaces`, then resolved each one's inheritance chain to see whether it reaches
`IFullModel<T, …>`. A direct `: IRegression<T>` counts as IFullModel because `IRegression : IFullModel`.

## Already `IFullModel` (transitively) — no work

These root interfaces inherit `IFullModel`, so every interface below them is already a full model. This
is the bulk of the "59 hits" a direct-declaration scan reports:

| root interface | derives | covers |
|---|---|---|
| `IRegression<T>` | `IFullModel<T, Matrix<T>, Vector<T>>` | `INonLinearRegression`, `ITreeBasedRegression`, `ILinearRegression`, … |
| `IClassifier<T>` | `IFullModel<T, Matrix<T>, Vector<T>>` | `IProbabilisticClassifier`, `IOnlineClassifier`, `IOrdinalClassifier`, `IDecisionFunctionClassifier`, `ISemiSupervisedClassifier`, `ITreeBasedClassifier`, `ITimeSeriesClassifier`, `IGaussianProcessClassifier` |
| `INeuralNetwork<T>` | `IFullModel<T, Tensor<T>, Tensor<T>>` | `INeuralNetworkModel`, `IDetectionBackbone`, `ILayeredModel`, … |
| `ITimeSeriesModel<T>` | `IFullModel<T, Matrix<T>, Vector<T>>` | time-series family |
| `IDiffusionModel<T>` | `IFullModel<T, Tensor<T>, Tensor<T>>` | `I3DDiffusionModel`, `IAudioDiffusionModel`, `IVideoDiffusionModel`, `ILatentDiffusionModel` |
| `ISegmentationModel<T>` | `IFullModel<T, Tensor<T>, Tensor<T>>` | `ISemanticSegmentation`, `IInstanceSegmentation`, `IPanopticSegmentation`, `IMedicalSegmentation`, `IOpenVocab/Promptable/Referring/VideoSegmentation` |
| `IFinancialModel<T>` | `IFullModel<T, Tensor<T>, Tensor<T>>` | `IForecastingModel`, financial family |
| `IGradientModel<T>` | `IFullModel<T, TInput, TOutput>` | gradient-boosting family |

**Done already this PR** (were declared standalone, now formalized): `IClustering` (pre-existing),
`IGaussianProcess`, `IAnomalyDetector` — both migrated by adding the `IFullModel` base, because
`GaussianProcessBase`/`AnomalyDetectorBase` already derive from `ModelBase`.

## Genuinely NOT `IFullModel` — the real decision surface

These do not reach `IFullModel`. They split into three groups with different right answers.

### Group A — embed/generate/inference components (NOT train-on-(X,y) models)

`IMultimodalEmbedding` (~10 impls: `IBlipModel`, `IBlip2Model`, `ILLaVAModel`, `IFlamingoModel`,
`IGpt4VisionModel`, `IImageBindModel`, `IVideoCLIPModel`, `IUnifiedMultimodalModel`, `IDallE3Model`,
`IAudioVisualCorrespondenceModel`), `IAudioLanguageModel`, `IAudioFoundationModel`, `ILanguageModel`
(~3 impls), `IEmbeddingModel` (already handled — surfaced as a transform component this PR).

**Shape:** `Embed(...)`, `Generate(...)`, `Encode(...)` — no supervised `Train(X, y)`. Forcing
`IFullModel` is a square peg, exactly as it was for `IEmbeddingModel`.
**Recommendation:** treat as **preprocessing/transform or inference components** (the `IEmbeddingModel`
precedent — surface on the result / apply as a transform), not pipeline models.

### Group B — perception/signal detectors

`IKeyDetector`, `IPitchDetector`, `IVoiceActivityDetector`, `IOutlierDetector`.

**Shape:** `Detect(signal) → events/labels`. `IOutlierDetector` overlaps `IAnomalyDetector` and would
migrate the same trivial way **if** its impls derive from `ModelBase`. The audio detectors are
signal-processing components.
**Recommendation:** `IOutlierDetector` → migrate like `IAnomalyDetector` (verify base first). Audio/
signal detectors → transform/inference components (Group A treatment).

### Group C — role/infra contracts (NOT models)

`ITeacherModel` (distillation role), `IClientModel` (federated role), `IOnnxModel` (runtime wrapper),
`IPredictiveModel` / `IGraphInferenceModel` (narrow predict contracts), `IFitDetector`, `IBiasDetector`,
`IAdaptedMetaModel`.

**Shape:** these are *roles a model plays* or *tooling*, not trainable models.
**Recommendation:** **leave as-is.** Making these `IFullModel` would be wrong — they are intentionally
narrower than a full model.

## Suggested plan

1. **No broad migration needed** — the model families are already `IFullModel`. The two obvious
   standalone models (`IGaussianProcess`, `IAnomalyDetector`) are already migrated.
2. **`IOutlierDetector`** — one more trivial migration if its impls derive from `ModelBase`; check and do.
3. **Group A / audio-signal Group B** — decide per-family whether each becomes a surfaced transform
   component (the `IEmbeddingModel` pattern) or stays a domain API. No `IFullModel` for these.
4. **Group C** — no change.

Net: the "bigger issue" is mostly already solved by the `ModelBase : IFullModel` design. What remains is
a handful of embed/generate/detector components that should be wired as transform/inference components,
not forced into the supervised model contract.
