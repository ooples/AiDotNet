# AiModelBuilder facade — configuration audit

**54 of 114 `Configure*` methods are fully inert**: every private field they assign is never read anywhere. The fluent API accepts the value, returns `this`, and drops it. No exception, no warning — the model trains as if the call never happened.

Tracking issue: #1876. Related PR: #1875.

---

## Resolved (this PR)

- **Tier 1** — RESOLVED (loss / LR scheduler / stopping criterion / data splitter / cross-validation; activation → `NotSupportedException`).
- **Knowledge distillation** — integrated with the tape training flow (surrogate-loss adapter); loss-decrease test.
- **Tier 2 metrics** — `ClusterMetric`/`ExternalClusterMetric` wired: every clustering model is auto-evaluated with cluster-validity indices (`AiModelResult.ClusteringEvaluation`), configured metrics extend the default set. `DistanceMetric`/`SimilarityMetric` auto-evaluated (predicted-vs-target under the metric). `Benchmark` and `ScoringRule` **removed** (post-build action / model options is the door).
- **Model-ctor-param methods (11) removed** — `KernelFunction`, `LinkFunction`, `MatrixDecomposition`, `RadialBasisFunction`, `WaveletFunction`, `WindowFunction`, `Layer`, `Interpolation`, `Interpolation2D`, `PDESpecification`, `NoiseScheduler`. Each is a constructor parameter of the consuming model; options is the door.
- **IFullModel migration** — `IGaussianProcess` and `IAnomalyDetector` now derive from `IFullModel<T, Matrix<T>, Vector<T>>` (their bases already implement it via `ModelBase`); their inert Configure methods removed — `ConfigureModel` covers them. See `IFULLMODEL_MIGRATION_AUDIT.md` for the full migration surface (mostly already IFullModel transitively).
- **EmbeddingModel** — surfaced on the result as a preprocessing/transform component (`AiModelResult.EmbeddingModel`), not forced into the model contract.
- **DriftDetection** — wired to a two-lens `DriftMonitor` that attributes drift to concept (error stream) vs covariate (prediction distribution) in one pass; `AiModelResult.DriftReport` + live `DriftMonitor`.
- **ActiveLearning + QueryStrategy** — wired to BADGE-style diversity-aware batch selection (uncertainty × diversity, three-tier representation cascade: gradient embeddings → model representation → input features); `AiModelResult.ActiveLearningSelection`. Added opt-in `ISupportsRepresentation<T, TInput>`.
- **DataTransformer** — was already removed as a strict duplicate of the fully-wired `ConfigurePreprocessing`.
- **DistillationStrategy** — already reconciled onto the KD options in both call orderings during the KD work; now also engages the KD path (failing clearly on a missing teacher) when configured alone, instead of being silently ignored.
- **ContinualLearning** — redesigned to a one-model, strategy-based API (nullable strategy → EWC + experience-replay hybrid default); training routes through a continual learner built around the configured model, and an all-tasks retention report (retained accuracy, forgetting, forward/backward transfer) is surfaced on `AiModelResult.ContinualLearningReport`.

Remaining: Tier 3 (SSLMethod, ExplorationStrategy, Environment), Tier 4/6 domain + robustness/tooling methods, and the `PARTIAL` federated/hyperparameter/pipeline fields.

---

## How this was measured

```
1. parse every `public IAiModelBuilder<...> Configure*(...)` body in src/AiModelBuilder*.cs
2. collect the private fields it assigns
3. count reads across all of src/ (excluding src/Generated) as:
       occurrences - assignments - declaration
4. classify:
     INERT   = every field it assigns has 0 reads
     PARTIAL = some assigned fields dead, others live
     WIRED   = all assigned fields have readers
```

Result: **54 INERT, 3 PARTIAL, 57 WIRED.**

### Two caveats that matter

**1. This is triage, not proof.** Reads via reflection or source generators are not detected (`src/Generated` is excluded). An earlier, sloppier pass reported 66 and produced false positives — `ConfigureFitDetector` and `ConfigureFitnessCalculator` **are** read and were wrongly flagged. Confirm each entry before fixing it.

**2. The read-count is a LOWER BOUND on the work.** A field having a reader does not mean the value reaches training. `ConfigureLossFunction` is the worst case: even after wiring the field, the facade's loss is only used for *reporting* — the gradient comes from the model's own loss. See Tier 1.

---

# Tier 1 — Training-critical (correctness bugs) — **RESOLVED**

These change what the model *learns*. A user configured them and training silently ignored them.

| method | resolution |
|---|---|
| `ConfigureLossFunction` | **wired** — set on the model so `OnModelChanged` carries it to the optimizer |
| `ConfigureDataSplitter` | **wired** — unlocks ~58 splitters incl. purged; nested inner split for validation |
| `ConfigureCrossValidation` | **wired** + new `PurgedWalkForwardCrossValidator` |
| `ConfigureLearningRateScheduler` | **wired** into optimizer options; adaptive rule unified into `AdaptiveFitnessScheduler` |
| `ConfigureStoppingCriterion` | **wired** at `InvokeTrainingEpoch` |
| `ConfigureRegressionMetric` / `ConfigureClassificationMetric` | **wired** → `AiModelResult.ConfiguredMetrics` |
| `ConfigureCurriculumScheduler` | **wired** → `CurriculumLearningOptions.CustomScheduler`, order-independent |
| `ConfigureActivationFunction` | **removed** — activation is a per-layer constructor parameter |
| `ConfigureModelOptions` | **removed** — options are a model constructor parameter |

## What tier 1 proved

**The dead config surface was hiding broken implementations underneath it.** Two of these were not merely unwired — the code they should have reached was itself broken, and nobody could find out:

- **Cross-validation could not have worked for anyone.** `CrossValidatorBase.PerformCrossValidation` built *empty* test data for every fold and handed it to the optimizer; `ValidationHelper` rejects that outright (`"Test matrix cannot be empty"`), so every optimizer that validates its input threw before a single fold trained. In k-fold there is no third partition — the held-out fold *is* the test set.
- **The purged fold geometry was sized by the wrong dimension.** `GetInputSize` returns the *feature* count, so a 300×3 matrix was handed to the splitter as "3 samples". It surfaced only because the error message quoted its own input.

**Test-suite failures blamed on "flakiness" were real defects too.** Five `ResourceMonitor` tests slept a fixed 100–200ms and assumed a thread-pool timer had fired — measuring scheduling latency, not the monitor (which works: 300ms → 7 snapshots). `ClearHistory` asserted empty-after-clear, which passed whether or not anything was ever collected — it would have kept passing if `ClearHistory` were deleted. A perf test timed eager then compiled back-to-back, charging any intervening load entirely to the second.

Rule of thumb this yields: **wiring a dead knob is the cheap part — the expensive part is what you find behind it.**

## 1.1 `ConfigureLossFunction` — dead at three layers

**Layer 1 — the field.** `_configuredLossFunction` assigned at `CoreML.cs:46`, never read.

**Layer 2 — the facade's loss is only for reporting.** `BuildPipeline.cs:314` resolves `_model.DefaultLossFunction`, and it has exactly one consumer:

```csharp
// BuildPipeline.cs:579 — the ONLY use
var loss = lossFunction.CalculateLoss(predictionVector, targetVector);
```

That feeds the epoch-loss metric. The **gradient** comes from the model's own loss (`NeuralNetworkBase.cs:927, :990`). So the one-line fix `_configuredLossFunction ?? _model.DefaultLossFunction` makes the *logged* loss honor the caller while training still optimizes something else — worse than today, because the number would no longer describe what was optimized.

**Layer 3 — the real seam already exists.** Do **not** add a parallel path:

```csharp
// Optimizers/GradientBasedOptimizerBase.cs:462-496  — OnModelChanged
//   adopts newModel.DefaultLossFunction into the optimizer's LossFunction (:490)
//   guarded by GradientOptions.LossFunctionExplicitlySet (:468)
```

**Therefore: set the loss on the MODEL so `DefaultLossFunction` returns it, and the existing auto-sync carries it to the optimizer.**

Per-family injection points:

| family | field | injectable? |
|---|---|---|
| `NeuralNetworkBase` | `protected ILossFunction<T> LossFunction` (`:210`), **not readonly**, ctor `:447`, exposed virtual `DefaultLossFunction` `:11496` | yes — add an `internal` setter |
| `TimeSeriesModelBase` | `_defaultLossFunction = options.LossFunction ?? new MeanSquaredErrorLoss<T>()` (`:304`) | yes via options — **but see split-brain below** |
| `RegressionBase` | `private readonly ILossFunction<T> _defaultLossFunction` (`:86`), set `:159` | needs ctor arg or new setter |
| `ClassifierBase` | `readonly`, set `:159` | needs ctor arg or new setter |
| `MultiLabelClassifierBase` | hardcodes `new BinaryCrossEntropyLoss<T>()` (`:93`) | **not injectable at all** |
| various (`OCRBase.cs:524`, `DiffusionAutoML.cs:798`, …) | `=> new MeanSquaredErrorLoss<T>()` expression-bodied | **cannot honor a configured loss without a signature change** |

### Correction to an earlier claim

`TimeSeriesRegressionOptions.LossFunction` (`:127`) **is** read — at `TimeSeriesModelBase.cs:304`, surfaced via `DefaultLossFunction` (`:2098`) and honored by `ComputeGradients` (`:2100-2102`). The accurate statement is narrower: **no concrete TimeSeries model's `TrainCore` consults it.** Each neural TS model builds its own optimizer inside `TrainCore` and hardcodes a loss. That is a genuine split-brain — the option affects the external gradient path but not actual training.

### Models that MUST reject an injected loss

| model | its loss | why substitution breaks it |
|---|---|---|
| **DeepARModel** | Gaussian NLL over `(mean, scale)`, hand-rolled on the tape (`:396`, called `:277`) | the head emits two tensors; `ILossFunction.CalculateLoss(Vector, Vector)` is a single-prediction contract and cannot express it. Substituting MSE silently kills the σ branch and destroys the probabilistic forecast. The μ/σ coupling (`∂NLL/∂μ = (μ−y)/σ²`) is documented at `:386-395`. |
| **TemporalFusionTransformer** | `QuantileLoss<T>` per level, summed pinball (`:195-197`, `:200-210`) | output is `[B, H*Q]` (horizon × quantiles). A point-wise loss is shape-mismatched against `[B, H]` and would train Q identical heads. The pinball loss *defines* the output semantics. |
| **DLinearModel** | none — MSE gradient is analytic/inlined (`double err = pred - y[i];`, `:148`) | there is no `ILossFunction` seam to inject into without rewriting `TrainCore`. |

Swappable (all plain MSE): NBEATS (`:298`, `:625`), NHiTS (`:265`, `:524`), Informer (`:306`, `:458`), Autoformer (`:315`).

### The contract is wider than training can honor

`ILossFunction<T>` (`src/Interfaces/ILossFunction.cs:26`) has three members: `CalculateLoss` (`:34`), `CalculateDerivative` (`:42`), `CalculateLossAndGradientGpu` (`:54`). It has **no `ComputeTapeLoss`** — that lives on the abstract `LossFunctionBase<T>`, and the tape paths hard-require the downcast:

```
NeuralNetworkBase.cs:6885-6887 →
  "LossFunction must derive from LossFunctionBase<T> for tape-based training"
  (also :3717, :7389, :8400, :9139)
```

So an arbitrary `ILossFunction<T>` that isn't a `LossFunctionBase<T>` **throws deep inside training**. `ConfigureLossFunction(ILossFunction<T>)` therefore advertises a wider contract than it can serve — validate up-front at configure/build time rather than failing mid-fit.

## 1.2 `ConfigureLearningRateScheduler` — the optimizer already supports it

**Optimizers already consume schedulers.** Do **not** add a facade-level scheduler loop:

```csharp
// Models/Options/GradientBasedOptimizerOptions.cs
public ILearningRateScheduler? LearningRateScheduler { get; set; }              // :269
public SchedulerStepMode SchedulerStepMode { get; set; } = StepPerEpoch;        // :284

// Optimizers/GradientBasedOptimizerBase.cs
protected ILearningRateScheduler? _learningRateScheduler;   // :222
public ILearningRateScheduler? LearningRateScheduler => …;  // :307  (GET-ONLY)
ctor :430-437 seeds SetLearningRate(scheduler?.CurrentLearningRate ?? InitialLearningRate)
public double StepScheduler()      // :1976
public virtual void OnEpochEnd()   // :2001
public virtual void OnBatchEnd()   // :2036
```

**So the wiring is: push `_configuredLearningRateScheduler` into the optimizer's *options*** (there is no post-construction setter — `:307` is get-only).

### Three traps

**(a) Re-adding facade LR mutation reintroduces a fixed bug.** `BuildPipeline.cs:297-311` documents it: a facade-owned `learningRate *= 0.99` per epoch used to run alongside the optimizer's own schedule and bypass `Optimizer.Step()`. It was deliberately deleted. The streaming loop now correctly defers (`:564-572`).

**(b) `UseAdaptiveLearningRate` is a live second decay path.**

```csharp
// Optimizers/OptimizerBase.cs:1648-1677  UpdateAdaptiveParameters
if (Options.UseAdaptiveLearningRate) {
    CurrentLearningRate = Multiply(CurrentLearningRate, LearningRateDecay);  // :1664
    ... else Divide(...)                                                     // :1670
}
```
That writes the same `CurrentLearningRate` a scheduler sets via `SetLearningRate` (`:506`) — the adaptive rule stomps the scheduler every step. `AdamOptimizer.cs:132` already treats them as mutually exclusive for the GPU path. **Reject or ignore `UseAdaptiveLearningRate` when a scheduler is configured.**

**(c) `ReduceOnPlateauScheduler.Step()` is a silent no-op.**

```csharp
// LearningRateSchedulers/ReduceOnPlateauScheduler.cs
public double Step(double metric)  // :142  — the real logic
public override double Step()      // :180-184 — _currentStep++; return _currentLearningRate;  NO REDUCTION
```
The metric overload is **not on `ILearningRateScheduler`**. Any wiring must type-test for `ReduceOnPlateauScheduler` and feed it the validation loss — otherwise the default-on plateau scheduler does nothing.

**(d) Only Adam ticks the scheduler.** `OnEpochEnd()`/`OnBatchEnd()` are called only from `AdamOptimizer.cs:294, :367` and `NeuralNetworkBase.cs:9216`. The facade never ticks them, so **non-Adam gradient optimizers get a flat LR**.

Available schedulers (`src/LearningRateSchedulers/`): Constant, CosineAnnealingLR, CosineAnnealingWarmRestarts, CyclicLR, ExponentialLR, LambdaLR, LinearWarmup, MultiStepLR, Noam, OneCycleLR, PolynomialLR, **ReduceOnPlateau**, SequentialLR, StepLR — plus `LearningRateSchedulerFactory` (`CreateForCNN`/`ForTransformer`/`ForFineTuning`/`CreateAdaptive`/…).

## 1.3 `ConfigureStoppingCriterion` — impedance mismatch

`IStoppingCriterion<T>` (`src/ActiveLearning/Interfaces/IStoppingCriterion.cs:21-51`) is scoped to **active learning**, not generic epochs:

```csharp
bool ShouldStop(ActiveLearningContext<T> context);   // :38
T GetProgress(ActiveLearningContext<T> context);     // :45
void Reset();                                        // :50
```

`ActiveLearningContext<T>` (`:61-127`) carries `CurrentIteration`, `LossHistory`, `ValidationAccuracyHistory`, `ElapsedTime`, `MaxTime` — but also labeling-specific fields (`TotalLabeled`, `MaxBudget`, `UnlabeledRemaining`). Per-epoch use requires synthesizing a context.

Epoch-usable implementations: `ConvergenceCriterion` (`:45`), `PerformancePlateauCriterion` (`:42`), `TimeBudgetCriterion` (`:39`). Not epoch-usable: `BudgetExhausted`, `UnlabeledPoolExhausted`, `UncertaintyThreshold`, `QueryQuality`.

**Natural call site:** `InvokeTrainingEpoch` (`BuildPipeline.cs:160`) — it already owns the `out string? stopReason` continue/abort contract and every epoch loop routes through it.

## 1.4 `ConfigureDataSplitter` — and the purged splitters nobody can reach

**Two unrelated things share the name.** This is the crux:

- **`IDataSplitter<T>`** (`Preprocessing/DataPreparation/IDataSplitter.cs:40`) — what `_configuredDataSplitter` holds. Returns `DataSplitResult<T>` from `Split(Matrix<T>, Vector<T>?)`.
- **`DataSplitter`** (`Preprocessing/DataPreparation/DataSplitter.cs:18`) — an unrelated **static class**, which is what the pipeline actually calls:

```csharp
public static (TInput XTrain, TOutput yTrain, TInput XVal, TOutput yVal, TInput XTest, TOutput yTest)
    Split<T, TInput, TOutput>(TInput X, TOutput y,
        double trainRatio = 0.7, double validationRatio = 0.15,
        bool shuffle = true, int randomSeed = 42)          // :33-40
```

Call sites hardcode the ratios: `BuildPipeline.cs:1830-1831` (supervised) and `:1174-1175` (AutoML). Shuffle is decided at `:1822-1829`:

```csharp
bool isTimeSeriesModel = _model is TimeSeries.TimeSeriesModelBase<T> || …TaskFamilyOverride == TimeSeriesForecasting …;
bool shuffleBeforeSplit = !isTimeSeriesModel && _trainingGroups is null;
```

**The library already contains purged/chronological splitters, they ALREADY inherit `DataSplitterBase<T>`, and they are unreachable from the facade only because `_configuredDataSplitter` is never read.**

Every splitter in `Preprocessing/DataPreparation/Splitting/TimeSeries/` already derives from `DataSplitterBase<T>` (and therefore satisfies `IDataSplitter<T>` — the exact type the dead field holds):

| splitter | base | purge/embargo |
|---|---|---|
| `PurgedKFoldSplitter.cs:40` | `DataSplitterBase` | yes |
| `CombinatorialPurgedSplitter.cs:32` | `DataSplitterBase` | yes |
| `AnchoredWalkForwardSplitter.cs:34` | `DataSplitterBase` | — |
| `WalkForwardSplitter.cs:34` | `DataSplitterBase` | — |
| `RollingOriginSplitter.cs:34` | `DataSplitterBase` | — |
| `TimeSeriesSplitter.cs:40` | `DataSplitterBase` | — |
| `SlidingWindowSplitter.cs:35` | `DataSplitterBase` | — |
| `BlockedTimeSeriesSplitter.cs:35` | `DataSplitterBase` | — |

Plus ~50 further `DataSplitterBase<T>` subclasses under `Splitting/{Basic,Stratified,GroupBased,CrossValidation,Bootstrap,Nested,Online,Specialized,DomainSpecific,FederatedLearning,ActiveLearning}`.

**So wiring this ONE field unlocks ~58 splitters at once, purged ones included.** That makes §1.4 the highest leverage item in the audit: the "exceed industry standards" capability is already written and merely unreachable.

### The one straggler

`Finance/Evaluation/PurgedWalkForwardValidator.cs:27` is a **`static class`** — it does not inherit `DataSplitterBase<T>`, so it cannot be passed to `ConfigureDataSplitter`:

```csharp
public static IReadOnlyList<Fold> Split(int nSamples, int labelHorizon, int nSplits,
                                        int embargo = 0, bool expanding = true);   // :62-67
```

It is also the only implementation carrying the two parameters that matter most — `labelHorizon` and `embargo`. **It should be reshaped into a `DataSplitterBase<T>` subclass** (or wrapped by one) so it is wireable like everything else, rather than remaining a parallel static API that the facade can never reach.

**Why this matters:** a plain chronological split still leaks when label windows straddle the boundary — a 20-bar forward return computed at the split edge sees validation data. Purge+embargo fixes that, and **no mainstream ML library ships it**.

**Remaining wiring cost:** an adapter at `:1830` mapping `DataSplitResult<T>` → the 6-tuple the pipeline expects, falling back to the static `DataSplitter` for tensor/generic cases, and reconciling splitters whose `SupportsValidation == false`.

## 1.5 `ConfigureCrossValidation`

```csharp
// Interfaces/ICrossValidator.cs:60-64
CrossValidationResult<T, TInput, TOutput> Validate(
    IFullModel<T, TInput, TOutput> model, TInput X, TOutput y, IOptimizer<T, TInput, TOutput> optimizer);
```

Dead-store chain: `AiModelBuilder.cs:247` → `Workflows.cs:544-549` → `Configuration/AiModelCrossValidation.cs:10-14` (`CrossValidator` also never read). `Validate` is never invoked from the builder.

The doc comment at `Workflows.cs:542` claims *"Use the evaluation methods on AiModelResult to perform cross-validation after building"* — **no such wiring exists**. Either implement it or correct the doc.

Implementations: `KFold`, `StratifiedKFold`, `GroupKFold`, `LeaveOneOut`, `Nested`, `TimeSeriesCrossValidator` (`:29` — rolling-origin, **no purge/embargo**). The only purged logic lives in `PurgedWalkForwardValidator`/`PurgedKFoldSplitter`, neither of which implements `ICrossValidator`.

## 1.6 `ConfigureActivationFunction` — recommend `NotSupportedException`

Unlike the rest of Tier 1, this one should **not** be wired. Evidence:

1. **Per-layer and constructor-only.** `LayerBase.ScalarActivation` is `{ get; private set; }` (`Layers/LayerBase.cs:79`), assigned only in ctors (`:1145`, `:1234`). No setter exists anywhere.
2. **The codebase already rejected a single knob.** `NeuralNetworkRegressionOptions` splits the role in two: `HiddenActivationFunction` (`:226`, ReLU) and `OutputActivationFunction` (`:258`, Identity). One builder-level method cannot say which it means — and applying it to both is actively harmful (ReLU on a regressor's output clamps negatives; sigmoid on hidden layers reintroduces vanishing gradients).
3. **A second axis it cannot express:** `IActivationFunction<T>` (scalar) vs `IVectorActivationFunction<T>` (`DenseLayer.cs:434`). Softmax is vector-only.
4. **Layers are built inside model constructors**, before the builder sees the model. Wiring would mean mutating a field the fused-activation plan cache derives identity from (`LayerBase.cs:1950`, `:2237`).

**Silently ignoring it is the worst option** — users believe it applied. Throw, with a message pointing at per-layer constructors or the role-specific options.

---

# Tier 1b — Concrete classes that cannot be wired because they skip the base class

A `Configure*` method takes an **interface**. Any implementation that is a `static class`, or that does not derive from the contract's base, **cannot be passed to the facade at all** — no amount of wiring on the builder side will reach it. Fixing the dead field is necessary but not sufficient if the capability you want lives in a class the API cannot accept.

Audit of the training contracts:

| contract | expected base | in-repo implementations conforming? |
|---|---|---|
| `ILossFunction<T>` | `LossFunctionBase<T>` | ✅ 40/41 files derive it (the outlier is the base/interface itself) |
| `IDataSplitter<T>` | `DataSplitterBase<T>` | ✅ all ~58 derive it, TimeSeries/purged included |
| `ICrossValidator` | `CrossValidatorBase` | ✅ all derive it — **but none implements purge/embargo** |
| `ILearningRateScheduler` | `LearningRateSchedulerBase` | ✅ all derive it |
| `IStoppingCriterion<T>` | *(no base exists)* | 7 implement the interface directly — acceptable |

## The gaps

**1. `PurgedWalkForwardValidator` is a `static class`** (`Finance/Evaluation/PurgedWalkForwardValidator.cs:27`). It is the **only** implementation carrying `labelHorizon` and `embargo` — the two parameters that make a financial split honest — and being static it can never be handed to `ConfigureDataSplitter`. It should derive from `DataSplitterBase<T>` (or be wrapped by a subclass that does), so it is wireable like every other splitter instead of a parallel static API the facade cannot reach.

**2. There is no purged `ICrossValidator`.** `CrossValidators/TimeSeriesCrossValidator.cs:29` is rolling-origin with **no purge/embargo**. The purge logic exists only in `PurgedKFoldSplitter` / `CombinatorialPurgedSplitter` (which are `IDataSplitter`, not `ICrossValidator`) and in the static validator above. So once `ConfigureCrossValidation` is wired, there will still be **nothing purged to pass it**. A `CrossValidatorBase` subclass wrapping the purged fold logic is needed for the CV path to reach parity with the splitter path.

**3. The loss contract is wider than the tape path.** All 40 in-repo losses derive `LossFunctionBase<T>`, so in-repo usage is safe — but `ConfigureLossFunction` accepts the **interface**, and a user-supplied `ILossFunction<T>` that is not a `LossFunctionBase<T>` throws deep inside training (`NeuralNetworkBase.cs:6887`). Either narrow the parameter type, or validate at configure time and fail fast with a clear message.

## Rule of thumb

**If a capability is worth configuring, its implementations must derive from the contract's base class.** A static helper is unreachable by definition, and a parallel static API guarantees the facade's version drifts from the "real" one — which is exactly how the purged splitters ended up written, correct, and unusable.

---

# Tier 2 — Evaluation & metrics

Do not change what is learned, but silently change or omit what is *reported*.

| method | file:line |
|---|---|
| `ConfigureClassificationMetric` | `Coverage.cs:115` |
| `ConfigureRegressionMetric` | `Coverage.cs:131` |
| `ConfigureClusterMetric` | `Coverage.cs:214` |
| `ConfigureExternalClusterMetric` | `Coverage.cs:231` |
| `ConfigureDistanceMetric` | `DomainML.cs:194` |
| `ConfigureSimilarityMetric` | `Coverage.cs:498` |
| `ConfigureScoringRule` | `DomainML.cs:227` |
| `ConfigureBenchmark` | `Coverage.cs:180` |

---

# Tier 3 — Learning workflows

Whole features that appear configurable and never run.

| method | file:line |
|---|---|
| `ConfigureActiveLearning` | `DomainML.cs:63` |
| `ConfigureContinualLearning` | `DomainML.cs:80` |
| `ConfigureSSLMethod` | `Coverage.cs:282` |
| `ConfigureDistillationStrategy` | `Coverage.cs:333` |
| `ConfigureDriftDetection` | `DomainML.cs:97` |
| `ConfigureQueryStrategy` | `Coverage.cs:466` |
| `ConfigureExplorationStrategy` | `Coverage.cs:265` |
| `ConfigureEnvironment` | `Coverage.cs:400` |
| `ConfigureDataTransformer` | `Coverage.cs:82` |

---

# Tier 4 — Domain models

| method | file:line |
|---|---|
| `ConfigureAnomalyDetector` | `CoreML.cs:124` |
| `ConfigureClustering` | `CoreML.cs:92` |
| `ConfigureGaussianProcess` | `CoreML.cs:254` |
| `ConfigureAudioEffect` | `DomainML.cs:47` |
| `ConfigureAudioEnhancer` | `Coverage.cs:482` |
| `ConfigureVideoModel` | `DomainML.cs:113` |
| `ConfigureDocumentModel` | `DomainML.cs:145` |
| `ConfigureDocumentStore` | `Coverage.cs:164` |
| `ConfigurePointCloudModel` | `DomainML.cs:129` |
| `ConfigureFinancialModel` | `DomainML.cs:161` |
| `ConfigureEmbeddingModel` | `DomainML.cs:210` |
| `ConfigureTimeSeriesDecomposition` | `Coverage.cs:316` |
| `ConfigureInterpolation` | `CoreML.cs:140` |
| `ConfigureInterpolation2D` | `CoreML.cs:156` |
| `ConfigurePDESpecification` | `Coverage.cs:197` |
| `ConfigureNoiseScheduler` | `Coverage.cs:384` |

---

# Tier 5 — Math primitives

| method | file:line |
|---|---|
| `ConfigureKernelFunction` | `CoreML.cs:76` |
| `ConfigureLinkFunction` | `CoreML.cs:221` |
| `ConfigureMatrixDecomposition` | `CoreML.cs:237` |
| `ConfigureRadialBasisFunction` | `DomainML.cs:178` |
| `ConfigureWaveletFunction` | `CoreML.cs:188` |
| `ConfigureWindowFunction` | `CoreML.cs:172` |
| `ConfigureLayer` | `CoreML.cs:108` |

---

# Tier 6 — Robustness, compression, tooling

| method | file:line |
|---|---|
| `ConfigureAdversarialAttack` | `Coverage.cs:416` |
| `ConfigureAdversarialDefense` | `Coverage.cs:432` |
| `ConfigureCertifiedDefense` | `Coverage.cs:449` |
| `ConfigureModelCompressionStrategy` | `Coverage.cs:350` |
| `ConfigureModelExplainer` | `DomainML.cs:244` |
| `ConfigureTool` | `Coverage.cs:367` |

---

# PARTIAL — some fields dead, method still does something

| method | dead | live |
|---|---|---|
| `ConfigureFederatedLearning` | 8 (`_federatedContributionEvaluator`, `_federatedDriftDetector`, `_federatedFairnessConstraint`, `_federatedPrivateSetIntersection`, `_federatedSecureComputationProtocol`, `_federatedTeeProvider`, `_federatedUnlearner`, `_federatedZkProofSystem`) | 6 |
| `ConfigureHyperparameterOptimizer` | 1 (`_hyperparameterTrials`) | 2 |
| `ConfigurePipelineParallelism` | 1 (`_pipelineMicroBatchCount`) | 3 |

---

# Cross-cutting: industry-standard defaults

The facade is meant to ship industry-standard defaults, so early stopping and LR scheduling should be **on by default**, not merely configurable. For reference, the ecosystem's defaults are weaker than assumed:

| library | default |
|---|---|
| sklearn `MLPClassifier` | `early_stopping=False` |
| Keras `EarlyStopping` | opt-in; `restore_best_weights=False` |
| PyTorch / Lightning | you must add the callback yourself |

So **on-by-default with best-weight restore already exceeds all three.** What would exceed them further, and is mostly wiring of code that already exists here:

1. **Purged + embargoed validation splits** (§1.4) — absent from every mainstream library; embargo sized automatically from the declared label horizon.
2. **Restore-best-weights on validation loss by default** — beats Keras's default outright.
3. **Divergence/NaN guard** coordinated with stopping — `TrainingMonitoring/HealthMonitorCallback` already exists and is not wired by default.
4. **ReduceLROnPlateau coordinated with the stopping rule** — reduce first, stop only once reduction stops helping (minding the `Step()` no-op trap in §1.2c).

## Related: the training-callback defect (#1875)

Same bug class, already diagnosed. `ConfigureTrainingCallback` documents *"return `false` to request an early stop"*. For the TimeSeries family that was unenforceable — those models own their epoch loop inside `Train()`, which the facade calls once, so the callback fired **once, after training finished**: `facade observed 1 epoch(s); expected one per training epoch`.

**Neural networks are unaffected** — verified by test. They train through the optimizer path, which already drives the per-epoch seam (`OptimizerBase.SetEpochProgressCallback` → `InvokeTrainingEpoch`), so callbacks see real epochs and a veto stops at exactly the requested epoch. `TrainingContractParityTests` now pins that behavior so the two families cannot silently diverge again.

---

# Suggested guard

A test or analyzer asserting that **every `Configure*` backing field has a reader** would stop this class of bug regressing. It would not catch the deeper Layer-2/Layer-3 problem (a value that is read but never reaches training), so pair it with per-family behavioral tests: configure X, assert training actually used X.
