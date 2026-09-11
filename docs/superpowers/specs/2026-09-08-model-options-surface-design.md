# Model Options as the User-Facing Configuration Surface

**Issue:** #2090
**Date:** 2026-09-08
**Status:** Approved design; implementation is tracked on separate phase branches.
Phase 1 scaffolding is recorded at `9f7fe07b6` and `944a1fd72`, and phase 2 at
`671836a34`. Those commits are not ancestors of this specification branch;
"recorded" below does not mean merged into master or fully validated against the revised contract.

---

## 1. Summary

Model `Options` classes are intended to be the only user-facing configuration
surface for model-specific parameters, reached through the facade as:

```csharp
var model   = new BGE<double>(architecture, new BGEOptions { NumLayers = 24 });
var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>()
                  .ConfigureModel(model);
```

Many models in `src/NeuralNetworks`, `src/Video` and `src/Document` accept, store
and return an `Options` object without consuming its model-specific values.
Their tunable values live instead in defaulted constructor parameters with no
`Options` equivalent. Section 2 records the measured exceptions as well as this
recurring defect; it is not a claim that every model in those directories is inert.

This design makes `Options` load-bearing across those three areas, adds a
compiler-checked ratchet so the two surfaces cannot drift apart again, and adds
paper-fidelity tests for the default values — which measurement shows are
currently placeholders, not paper values.

---

## 2. Measured baseline

The following source-scan figures are historical estimates recorded on 2026-09-08,
not a current count or a complete scope ledger. The authoritative historical
reflection baselines are the `Baseline` constants in
`OptionsSurfaceRatchetTests.cs` at `9f7fe07b6` (1067) and `671836a34` (977).
This docs-only review verifies those immutable sources and the arithmetic; it does
not claim to have rebuilt either historical assembly and repeated its measurement.
Every implementation phase must publish a fresh, complete gap ledger and the
assembly/commit used to produce it, as required by §7.

### 2.1 The Options object is inert

| Measurement | Count |
| --- | --- |
| Model files in `src/NeuralNetworks` with a public constructor | 117 |
| …that declare `private readonly XOptions _options` | 104 |
| …that ever **read** a value off it (`_options.Something`) | **9** |
| …that read via the base `Options.Something` | 8 |
| …that pass `_options` into a `LayerHelper` factory | **0** |

The recurring signature is exactly four mentions of `options` per file: the doc
comment, the field, `GetOptions() => _options`, and
`_options = options ?? new XOptions()`. The object is a sink.

### 2.2 The values live in constructor parameters instead

| Measurement | Count |
| --- | --- |
| Tunable defaulted constructor parameters across those models | **470** |
| …that have a corresponding property on the model's `Options` class | **1** |
| Models carrying ≥1 such parameter | **96** |
| `Options` classes under `src/NeuralNetworks/Options` with zero own properties | 99 / 106 |

The parameters are counted as the union across *all* public constructors of each
model, after excluding the `options` parameter itself, interface- and
delegate-typed collaborators, and artifact-path parameters (`modelPath`,
`modelIdentity`, `seed`). 82 of the 117 models declare more than one public
constructor; an earlier scan of only the first constructor gave 205, which
undercounted by more than half — for `BGE` the first constructor is the
parameterless one. 470 is the corrected figure and the one the ratchet in §7
reproduces.

Base classes do not rescue this. `ModelOptions` declares `Seed`;
`NeuralNetworkOptions` declares `EncoderLayerCount`. On this specification branch,
the latter is copied by several Options copy constructors, but those copies do not
make it affect a model's encoder/decoder boundary. No consuming model path was found
in the audited families. The explicit disposition is to remove this inert property,
not to count it as configuration coverage; see §5.2.

### 2.3 `src/NeuralNetworks` is the outlier, not the norm

Across all 1,622 `*Options.cs` files under `src/`:

| Area | Options classes | declare properties | set defaults in ctor | fully inert |
| --- | ---: | ---: | ---: | ---: |
| VisionLanguage | 186 | 157 | 186 | **0** |
| TextToSpeech | 125 | 70 | 125 | **0** |
| Audio | 105 | 105 | 23 | **0** |
| MetaLearning | 102 | 102 | 0 | **0** |
| SpeechRecognition | 96 | 96 | 96 | **0** |
| **NeuralNetworks** | **107** | **8** | **0** | **99** |
| Video | 108 | 65 | 0 | 43 |
| Document | 30 | 3 | 0 | 27 |

Five of the eight largest areas have **zero** inert Options classes. Of the 200
fully-inert classes in the entire repository, 169 are in the three areas this
spec covers: `NeuralNetworks` (99), `Video` (43) and `Document` (27).

**The reusable Options pattern already exists, but a populated Options class is
not proof that a model consumes it.** The expanded audit in §2.6 demonstrates why
#2090 needs both structural and behavioral checks across the full in-scope hierarchy.

### 2.4 The defaults in `src/NeuralNetworks` are placeholders, not paper values

This contradicts an earlier conclusion of mine and is stated plainly because it
changes the scope of the work.

**Corrected 2026-09-08.** The figures first published here were measured on the
branch `feature/cpu-offload-optimizer-fsdp`, not on master, and were wrong. On
master:

| | count |
| --- | ---: |
| Language models defaulting `modelDimension = 256` | **13 of 17** |
| …carrying genuine paper-scale values | **4 of 17** |

The four that are already right: `Zamba` (3712 × 76, 4096 context), `Zamba2`
(3584 × 81, 4096), `Griffin` and `Hawk` (2048 × 24, 2048). Thirteen models
sharing `256 × 4 × 512` across thirteen different papers is still the tell — no
two of those papers specify the same width — but it is thirteen, not seventeen,
and the correction matters because it changes how much of phase 9 is repair
versus confirmation.

The lesson is recorded rather than quietly fixed: **a measurement is scoped to
the tree it was taken on.** Every figure in this document was re-taken against
master before phase 1.

For contrast, where the value came from a real source it is visibly correct:

- `BGE` — `vocabSize 30522, embeddingDimension 768, numLayers 12, numHeads 12, feedForwardDim 3072` (BERT-base, exactly)
- `RT2Options` — `VisionDim 1024, DecoderDim 4096, NumVisionLayers 24, NumDecoderLayers 32, NumHeads 32` (PaLI-X)

So the repository has two tiers: paper-faithful defaults (`LayerHelper`
factories, `VisionLanguage`, `Audio`, `SpeechRecognition`) and placeholder
defaults (the `NeuralNetworks` sequence models). Paper verification is therefore
**not** a redundant audit of values that are already right — it has real defects
to find, and it is in scope for this work.

### 2.5 `Video` and `Document` have the identical defect

Scanned with the same rules, recursing into subdirectories because models there
are not flat files:

| Area | Types with a public ctor | Models with >=1 tunable param | Tunable params | Covered by Options |
| --- | ---: | ---: | ---: | ---: |
| `src/Video` | 108 | 44 | 134 | **0** |
| `src/Document` | 29 | 29 | 205 | **2** |

`src/Document` is the worst-affected area in the repository by density: **every
one** of its 29 model types carries tunable constructor parameters, averaging
seven each, and only 2 of 205 have an Options property. `TrOCR` and `Donut`
carry 11 apiece.

### 2.6 The reflection baseline found a fourth cluster (measured 2026-09-08)

Phase 1's ratchet recorded **1067**, against the file-based estimate of 806.
The arithmetic difference is 261, but these are not interchangeable populations:
the reflection test already excludes architecture descriptors and compiled-model
hosts, and it scans the full model inheritance tree rather than three directories.
Therefore 261 must not be treated as a proven, exhaustively identified audio cluster.
The original scan asked *"do this area's Options classes declare properties or set
constructor defaults?"*, not *"do this area's models read them?"* The broader audit
found additional affected models, including:

| Model | Tunable ctor params | Options type it is handed |
| --- | ---: | --- |
| `Tacotron2Model` | 20 | `OnnxModelOptions` (generic) |
| `TtsModel` | 17 | `OnnxModelOptions` (generic) |
| `VITSModel` | 16 | `OnnxModelOptions` (generic) |
| `SpeechEmotionRecognizer` | 11 | **none — no options parameter at all** |
| `DCCRN` | 10 | `OnnxModelOptions` (generic) |

`TextToSpeech` has 125 Options classes, every one of which sets its defaults in a
constructor — and its models are configured against a *generic ONNX* options
type rather than any of them. A healthy-looking Options class and a model that
reads it are independent properties, and §2.3 only measured the first.

The additional audio/speech estimate is 192, not 261. The remaining work must be
identified by the per-model ledger rather than silently omitted or classified as
out of scope. Section 8 accounts for the entire 1067 baseline, including the
currently unallocated remainder.

---

## 3. What the prototype overturned

Recorded so the reasoning is auditable, and because three premises that shaped
both the issue's "Agreed design" and my own earlier plan turned out to be wrong.

| Premise | Verdict | Evidence |
| --- | --- | --- |
| "Users cannot configure these models at all" | **False** | The constructor works today; `new BGE<double>(arch, numLayers: 24)` compiles and takes effect. |
| "`ConfigureModel` taking a built model is a facade defect" | **False** | It is the seam. Options reaches the facade *through* the constructor; the model is configured before the builder sees it. |
| "We must author paper-derived defaults from a PDF corpus" | **Partly false** | 496 of 662 `LayerHelper` factories are already parameterised with paper values. The corpus is needed to *verify*, and to *repair* the placeholder values in §2.4 — not to author from scratch. |
| "The ratchet counts documented-but-absent properties" | **Broken** | PR #2088 deletes those doc assignments, so the metric would read zero on merge and enforce nothing. Replaced in §7. |
| "Adding a property to an empty Options class fixes the defect" | **False** | Inert either way unless the constructor reads it. Wiring is the fix; the property is a prerequisite. |

---

## 4. Reference implementation (already in the repo)

`src/VisionLanguage/Robotics/RT2Options.cs` is the pattern to copy:

```csharp
public class RT2Options : VisionLanguageActionOptions
{
    public RT2Options()
    {
        VisionDim = 1024;         // PaLI-X
        DecoderDim = 4096;
        NumVisionLayers = 24;
        NumDecoderLayers = 32;
        NumHeads = 32;
        ImageSize = 224;
        VocabSize = 32000;
    }

    public RT2Options(RT2Options other) { /* copy ctor */ }
}
```

Three properties of this shape make it the right target:

1. **The family base declares the property once.** `VisionDim` is not redeclared
   in every sibling; `VisionLanguageActionOptions` owns it. This is what makes
   the leaf classes small instead of making 285 of them large.
2. **The leaf constructor carries the paper defaults.** The per-model knowledge
   sits in the one place that is per-model.
3. **The model reads it.** `RT2` uses `_options.VisionDim` and passes it into
   `LayerHelper<T>.CreateDefaultRoboticsActionLayers(...)` — the wiring that
   `src/NeuralNetworks` is missing.

The historical RT2 comment about adding `EncoderLayerCount` records a real
custom-layer partitioning problem, not a justification for keeping an inert
Options property. Section 5.2 resolves that topology setting through the
architecture/layout contract. The existing RT2 constructor illustrates where
paper defaults belong; it must adopt §5.5's non-recording initialization path if
explicit-assignment tracking is added.

---

## 5. Design

### 5.1 Scope

The original three-area source estimate was 470 parameters in `NeuralNetworks`,
134 in `Video`, and 205 in `Document`: **809 total, of which 3 had an Options
property, leaving 806 missing before source-scan exclusions**. These numbers are
retained as historical planning data, not as the reflection ratchet's population.

Of the 470 in `NeuralNetworks`:

| Group | Models | Params | Disposition |
| --- | ---: | ---: | --- |
| Architecture types (`NeuralNetworkArchitecture`, `Transformer…`, `DualStream…`, `TripleStream…`, `AudioTextDualStream…`) | 5 | 48 | **Out of scope.** Topology knobs stay on the architecture type per the locked bucket-B decision. |
| Infrastructure hosts (`CompiledModelHost`, `ChainedCompiledModelHost`) | 1 | 1 | **Out of scope.** `modelIdentity` / `shapeMode` are not hyperparameters. |
| **Vision-language / multimodal** | 11 | 107 | In scope → `VisionLanguageModelOptions` |
| **Sequence / language models** | 18 | 98 | In scope → `SequenceModelOptions` |
| **Embedding & retrieval** (BGE, ColBERT, SGPT, SPLADE, SimCSE, Instructor, Matryoshka, FastText, GloVe, Word2Vec, TransformerEmbedding) | 11 | 77 | In scope → `EmbeddingModelOptions` |
| **GAN family** | 10 | 41 | In scope → `GanOptions` |
| **Long tail** (graph nets, classic CNN/RNN, autoencoders, RBM/DBM, spiking, mesh/voxel, …) | 40 | 97 | In scope; sub-clustered in phase 5 |
| **Total in scope, `NeuralNetworks`** | **90** | **420** | |

`src/Video` (44 models, 134 params) and `src/Document` (29 models, 205 params)
are in scope in full — neither contains architecture types or compiled-model
hosts, so nothing is excluded. Their family bases are drawn from the bases the
models already share (`DocumentNeuralNetworkBase<T>` and the `Video`
equivalents), rather than from new groupings:

| Area | Models | Params | Family base |
| --- | ---: | ---: | --- |
| `src/Document` | 29 | 205 | `DocumentNeuralNetworkOptions` under the existing `DocumentNeuralNetworkBase<T>` hierarchy |
| `src/Video` | 44 | 134 | `VideoHyperparameterOptions`, sub-split by task (segmentation, generation, super-resolution, tracking) |

**Historical three-area subset: 163 models, 757 missing parameters.** This is not
the full scope after §2.6: the authoritative initial in-scope gap count is 1067.

Source-estimate arithmetic: 806 missing, less 49 estimated architecture/host
parameters, leaves 757. Within `NeuralNetworks` the five groups sum to
98 + 107 + 77 + 41 + 97 = 420. The reflection test excludes the relevant types
before counting, so those 49 must **not** be subtracted from 1067 or retained as
a completion floor. All 1067 measured gaps require disposition. The final model
count and family ownership come from the complete ledger, not from adding
incompatible source-scan estimates.

The long tail is 40 models averaging 2.4 parameters each. It is the largest
model count and the smallest per-model effort, but it is also where §9.3 (family
grouping inferred from parameter names) is most likely to be wrong, so it is
sequenced after the more cohesive `NeuralNetworks` families.

### 5.2 Family base classes

Seven family base classes declare shared knobs once. The authoritative roster
below is checked against implementation snapshot `944a1fd72`. All seven derive
from `ModelHyperparameterOptions`, which derives from `NeuralNetworkOptions`
and owns the shared training option `MaxGradNorm`; that common infrastructure
base is not an eighth family. Four family bases live under
`src/NeuralNetworks/Options/`, two under `src/Models/Options/`, and one under
`src/Video/Options/`.

- **`DocumentNeuralNetworkOptions` already existed and all 29 Document options
  classes already derive from it.** It was empty. Extending it reaches the whole
  area's available property surface without changing every leaf's parent — no new
  base was needed. Models still have to consume the added values.
- **Video has no base in use.** 96 of its 108 options classes derive straight
  from `NeuralNetworkOptions`. A `VideoModelOptions<T>` exists but takes a type
  parameter it never uses, follows the nullable + `Effective*` pattern, and is
  derived from by exactly one class; `DocumentModelOptions<T>` is its twin and
  nothing derives from it at all. Both look like an earlier attempt at this same
  work that was never wired up. They are left alone here and removed in their
  areas' phases. The new base is named `VideoHyperparameterOptions` to avoid
  colliding with the abandoned one.

| Family base | Location | Declared shared properties at `944a1fd72` |
| --- | --- | --- |
| `SequenceModelOptions` | `src/NeuralNetworks/Options/SequenceModelOptions.cs` | `VocabSize`, `ModelDimension`, `NumLayers`, `NumHeads`, `StateDimension`, `MaxSequenceLength`, `AttentionInterval`, `ExpandFactor`, `FfnMultiplier` |
| `VisionLanguageModelOptions` | `src/NeuralNetworks/Options/VisionLanguageModelOptions.cs` | `EmbeddingDimension`, `MaxSequenceLength`, `ImageSize`, `PatchSize`, `Channels`, `VocabSize`, `NumHeads`, `HiddenDim`, `NumEncoderLayers`, `VisionHiddenDim`, `NumVisionLayers` |
| `GanOptions` | `src/NeuralNetworks/Options/GanOptions.cs` | `LatentSize`, `GeneratorChannels`, `DiscriminatorChannels`, `ImageChannels`, `CriticIterations`, `InitialLearningRate` |
| `EmbeddingModelOptions` | `src/NeuralNetworks/Options/EmbeddingModelOptions.cs` | `VocabSize`, `EmbeddingDimension`, `MaxSequenceLength`, `NumLayers`, `NumHeads`, `FeedForwardDim` |
| `DocumentNeuralNetworkOptions` | `src/Models/Options/DocumentNeuralNetworkOptions.cs` | `ImageSize`, `ImageWidth`, `ImageHeight`, `PatchSize`, `MaxSequenceLength`, `VocabSize`, `HiddenDim`, `NumHeads`, `NumLayers`, `NumEncoderLayers`, `NumDecoderLayers`, `VisionDim`, `VisionLayers`, `BackboneChannels`, `NumClasses` |
| `VideoHyperparameterOptions` | `src/Video/Options/VideoHyperparameterOptions.cs` | `NumFeatures`, `NumLayers`, `NumFrames`, `EmbedDim`, `NumHeads`, `NumClasses`, `ScaleFactor`, `NumIterations` |
| `AudioHyperparameterOptions` | `src/Models/Options/AudioHyperparameterOptions.cs` | `SampleRate`, `NumMels`, `FftSize`, `HopLength`, `HiddenDim`, `NumHeads`, `NumEncoderLayers`, `NumDecoderLayers`, `SpeakingRate`, `Language` |

`AudioHyperparameterOptions` spans `TextToSpeech`, `SpeechRecognition` and
`Audio`, which share signal settings. The source uses `HopLength` and `FftSize`;
any migration from `hopSize` or `frameSize` must use a reviewed, typed parameter-to-property
mapping rather than count a renamed, correctly wired option as a missing property.
`PoolingStrategy` remains on the applicable embedding leaf options unless its
meaning and enum type are genuinely shared; it is not declared by the recorded family base.

Each leaf `XxxOptions` sets its own paper defaults in its parameterless
constructor and adds only genuinely model-specific properties. Defaults must use
the non-explicit initialization path in §5.5; ordinary tracked setters alone do
not satisfy this constructor contract.

**Inert-property disposition, required before family migration is considered complete:**
Remove `NeuralNetworkOptions.EncoderLayerCount` and its copy-constructor assignments
in the shared topology/options cleanup. Do not retain an ignored compatibility
property or add a second topology surface. Encoder/decoder partitioning for supplied
custom layers belongs to the architecture/layout contract and must be consumed and
validated there. The migration includes a surface assertion that the Options property
is absent, plus generated/shared-layout tests for the intended custom-layer partition.
This specification does not itself delete a production API; the required implementation
change is explicit and remains outstanding on the recorded phase branches.

### 5.3 Constructors

The tunable scalar parameters are **removed** from the constructor. What remains
is the architecture, the options object, and the non-configuration
collaborators:

```csharp
// before
public BGE(NeuralNetworkArchitecture<T> architecture, ITokenizer? tokenizer = null,
    IGradientBasedOptimizer<...>? optimizer = null,
    int vocabSize = 30522, int embeddingDimension = 768, int maxSequenceLength = 512,
    int numLayers = 12, int numHeads = 12, int feedForwardDim = 3072,
    PoolingStrategy poolingStrategy = PoolingStrategy.ClsToken,
    ILossFunction<T>? lossFunction = null, double maxGradNorm = 1.0,
    BGEOptions? options = null)

// after
public BGE(NeuralNetworkArchitecture<T> architecture, BGEOptions? options = null,
    ITokenizer? tokenizer = null,
    IGradientBasedOptimizer<...>? optimizer = null,
    ILossFunction<T>? lossFunction = null)
```

The body reads `_options.NumLayers` and passes it to the `LayerHelper` factory,
in place of today's `_numLayers` field.

**Decided 2026-09-08: remove them, with no `[Obsolete]` forwarding overloads.**
A forwarding overload would keep alive exactly the second surface this work
exists to eliminate, and the ratchet in §7 could never reach zero while one
remained. AiDotNet has not shipped v1, so there is no compatibility obligation.

`CreateNewInstance()` implementations that currently re-pass the scalar fields
(e.g. `Zamba2LanguageModel.cs:192`) are rewritten to pass `_options`.

### 5.4 Nullable vs. non-nullable properties — deviation from CLAUDE.md

`CLAUDE.md` mandates nullable properties with an internal `GetEffectiveX()`
default. This design uses **non-nullable properties with the default assigned in
the leaf constructor** (the RT2 pattern), for three reasons:

1. The nullable pattern encodes "null means use your best judgment at runtime" —
   appropriate for `BatchSize` (depends on data size) or `EnableGPU` (depends on
   hardware). A model's `NumLayers` has no runtime-dependent best value; the best
   value is the paper's, and it is known statically.
2. It is already the convention for model options in this repo: 582 Options
   classes set defaults in a constructor, including every one of the 186 in
   `VisionLanguage` and 96 in `SpeechRecognition`.
3. `GetEffectiveX()` for ~160 properties is ~160 methods that all return a
   constant.

Infrastructure config (`TelemetryConfig`, `ProfilingConfig`, `AutoMLOptions`)
keeps the nullable pattern unchanged.

**Decided 2026-09-08: adopt the RT2 pattern for model hyperparameters.**
`CLAUDE.md` has been amended (2026-09-08) to scope its nullable +
`GetEffectiveX()` rule to infrastructure configuration and to add a "Model
Hyperparameter Options" section carrying this pattern, so the two conventions are
written down rather than left as an undocumented split.

### 5.5 Conflict between `Layers` and topology knobs

Per the locked decision: when `architecture.Layers` is populated **and** an
Options topology knob is set to a value that contradicts it, the constructor
throws an `ArgumentException` naming both sides:

> `BGEOptions.NumLayers = 24 conflicts with the 12 layers supplied in
> NeuralNetworkArchitecture.Layers. Supply one or the other.`

Silence is not acceptable here — `RT2.cs:295-315` documents a real bug caused by
guessing when the two disagreed. A constructor-applied default is not an explicit
choice. A user assignment **is explicit even when its value equals the paper default**.
For example, a fresh `BGEOptions` with paper-default depth may accompany a valid
custom layer layout without a conflict, but an object initializer explicitly
requesting that same depth must conflict if the custom layout represents another depth.

The required implementation contract is:

1. Track explicit assignments with generated, strongly typed `OptionPropertyId`
   enum values, such as `EmbeddingNumLayers`, in `HashSet<OptionPropertyId>`.
   Keys identify the declaring semantic property; aliases share one canonical key.
   Property-name strings may appear in diagnostics, never as policy or dispatch keys.
2. Provide a protected, synchronous default-initialization scope for leaf constructors.
   While that scope is active, setters apply paper defaults without recording an
   explicit assignment. Scope disposal restores tracking even when initialization
   throws; it does not clear assignments already recorded by another constructor.
   The scope ends before any caller's object initializer or normal setter runs.
   This retains the RT2-style leaf defaults without misclassifying them as user input.
3. Copy constructors copy values through the same non-recording initialization path
   and clone the explicit-key set exactly, including inherited and leaf-specific keys.
   Do not reconstruct explicitness by comparing values with defaults, invoke ordinary
   setters during a copy, or share the mutable set between source and copy.
   Configuration/checkpoint round trips that preserve Options must preserve this
   metadata too; generated enum identities need an explicit versioned wire contract.
4. Conflict checks use the model's typed logical-layout metadata, not a generic
   `architecture.Layers.Count` comparison: one logical block can contain several layers.
   Only contradictory, explicitly assigned topology keys cause this conflict.
   Training, output and runtime options do not become topology conflicts.

Generated/shared-base regressions must distinguish untouched defaults, explicit
non-default values, explicit values equal to defaults, consistent and inconsistent
custom layouts, copies of each case, and independent mutation of source/copy metadata.
Also verify that an initialization exception cannot leave tracking disabled. The
historical phase-1 auto-properties do not implement these guarantees yet; adding
the roster alone is not completion of this requirement.

---

## 6. Paper fidelity

Per your decision, verification ships with this work rather than after it.

1. A reviewed data file, `docs/model-paper-defaults.tsv`, one row per
   (model, parameter, value, paper title, arXiv URL, section/table reference).
   Populated from cached PDFs, **human-confirmed before merge** — the generated
   file is a proposal, not the source of truth.
2. A `[PaperDefaults]` attribute on each Options class pointing at its paper,
   following the existing `[ResearchPaper]` attribute in
   `src/Attributes/ResearchPaperAttribute.cs` and the `[PaperOptimizer]` shape
   landing in #2098.
3. A test that instantiates each Options class with its parameterless
   constructor and asserts every property matches its row in the TSV. This is
   what catches the §2.4 placeholders and prevents new ones.

The placeholder values identified in §2.4 are corrected to paper values as part
of this work.
That is a behavioural change to defaults — intended, and the reason it must land
before v1 rather than after.

---

## 7. The ratchet

**Metric:** the number of tunable defaulted constructor parameters on an
in-scope model that have no correspondingly-named property on that model's
`Options` type.

**What counts as a model:** any concrete type transitively assignable to
`NeuralNetworkBase<T>`. This matters — most models do not name that base
directly. `BGE` derives from `TransformerEmbeddingNetwork<T>`,
`MambaLanguageModel` from `TokenLanguageModelLayoutBase<T>`, `TrOCR` from
`DocumentNeuralNetworkBase<T>`; a repo-wide search for `: NeuralNetworkBase<`
under `src/NeuralNetworks` finds only 3 files. Reflection walks the base chain
and gets this right; no file-path or naming heuristic does.

- **Initial recorded baseline: 1067 at `9f7fe07b6`; after the recorded phase-2
  migration: 977 at `671836a34`.** These values supersede the three-area source
  estimate of 806. Excluded types are removed before counting: **in-scope target: 0,
  with no nonzero out-of-scope floor**.
- Implemented as a reflection test over `AiDotNet.dll`, so it needs no
  documentation to exist and cannot be defeated by #2088's doc deletions.
- The numerical difference from the file-based proxy is 261, not a verified list
  of additional models or an exemption budget. See §2.6 and the residual accounting in §8.
- The historical implementation stores a `Baseline` constant in
  `tests/AiDotNet.Tests/IntegrationTests/Configuration/OptionsSurfaceRatchetTests.cs`,
  not a separate text file. It rejects increases and currently permits a `Slack` of
  10 before requiring a lower baseline. Every migration must nevertheless publish
  its **exact** new count and lower the constant deliberately. The completion gate
  is exactly zero; historical slack must not allow remaining gaps to disappear.
- Persist the complete gap ledger, not just the largest 15 entries: owning model
  identity, constructor parameter, Options type/property or missing mapping,
  semantic category, and migration phase. Partial assembly/type loading makes the
  measurement invalid and must fail closed, not report an artificially lower count.
  Every nonzero gap has an owner; an unassigned row is not an exclusion.

**Exclusions, stated precisely** (each of these produced a false positive in the
baseline scan and must be excluded by the test, not by the scanner's accident):

- the `options` parameter itself
- parameters whose type is an interface or delegate (optimizer, loss function,
  tokenizer) — collaborators, not configuration
- the five architecture types and two compiled-model hosts of §5.1, already
  excluded by the historical `GetModelTypes()` implementation before gap counting
- `modelIdentity`, and any parameter typed `string?` defaulting to `null` that
  names an artifact path rather than a hyperparameter

A second, property-specific assertion pins the fix rather than just the presence
of a property. For every applicable migrated option, `GetOptions()` must retain the
configured value **and** a controlled test must observe the behavior it governs.
Use generated, typed `OptionBehaviorKind` metadata, not property-name strings or
hand-authored per-model exceptions, to select the shared test contract:

| Semantic kind | Required observable proof |
| --- | --- |
| `OptionBehaviorKind.Topology` | The relevant logical depth, width, head count, branch or parameter shape changes as requested; layer-list length alone is insufficient. |
| `OptionBehaviorKind.Training` | A controlled training step or gradient calculation reflects the setting. For `MaxGradNorm`, a gradient above the threshold is clipped by the expected amount, while the disabled-clipping case follows its documented behavior. |
| `OptionBehaviorKind.Output` | Known activations produce the requested output behavior. For `PoolingStrategy`, compare the expected CLS/mean/etc. output on input for which those results differ. |
| `OptionBehaviorKind.Runtime` | The supported runtime path or resource policy is observably selected and numerical results remain within the declared contract; CPU tests do not remove production GPU support. |

Non-topology tests should also confirm that unrelated topology is unchanged.
Do not add artificial layers to make training/output/runtime settings satisfy a
topology assertion. Setting every value to a convenient number, or merely checking
`GetOptions()`, is not behavioral proof. Inapplicable inherited options must have
an explicit validated disposition, not be silently ignored or exempted by a string whitelist.

These cases belong in the test scaffold generator and shared model/layout test
bases. They use deliberately small supported configurations and deterministic data;
generated model test files are never patched manually. Each phase must enable the
appropriate behavioral contracts before declaring its migrated models complete.

---

## 8. Phasing

Each phase must be independently mergeable and leave the build green. The table
separates recorded baseline constants from forecasts; a forecast is not test
evidence and must be replaced by a measured, non-overlapping ledger when its phase runs.
The original phase-2 source estimate was 98; the recorded implementation reduced
the baseline by 90. That eight-gap difference must not be silently carried forward
as completed work.

| Phase | Content and provenance | Ratchet (before → after) |
| --- | --- | --- |
| 1 | Seven-family scaffolding recorded at `9f7fe07b6` plus `944a1fd72`; revised explicitness/inert-property contracts remain required | Initial recorded baseline **1067** |
| 2 | Sequence/language migration recorded at `671836a34` (17 models, 90-gap reduction) | 1067 → 977 |
| 3 | Vision-language/multimodal; provisional reduction 107 | 977 → 870 |
| 4 | Embedding/retrieval and GAN; provisional reduction 77 + 41 = 118 | 870 → 752 |
| 5 | Neural-network long tail; provisional reduction 97 | 752 → 655 |
| 6 | Document; provisional reduction 203 | 655 → 452 |
| 7 | Video; provisional reduction 134 | 452 → 318 |
| 8a | Text-to-speech, speech recognition and audio; provisional reduction 192 | 318 → 126 |
| 8b | Identify and migrate **all residual in-scope ledger entries**; provisional remainder 126, not an exemption | 126 → 0 |
| 9 | Reviewed paper-defaults data, fidelity tests and correction of placeholder values; preserve the zero-gap gate | 0 → 0 |

The forecast is now arithmetically complete:
`1067 - (90 + 107 + 118 + 97 + 203 + 134 + 192 + 126) = 0`.
The older subset estimate of 757 and audio estimate of 192 would leave 118 from
1067; replacing the sequence estimate of 98 by its recorded reduction of 90 leaves
126. Neither this calculation nor the original 261 difference identifies the actual
residual models. Phase 8b therefore starts with a complete typed gap ledger and
assigns every residual entry to a reviewed family migration; it cannot be waived
or replaced by lowering the threshold to a nonzero floor. If the earlier forecasts
overlap or differ from measurements, recalculate the residual from the actual ledger.

Completion requires a complete successful measurement of **zero in-scope gaps**,
enabled property-specific behavioral cases for all migrated options, and the
paper-default tests. Excluded architecture/host parameters are never part of this
count. A partially loaded assembly, missing behavior cases or remaining unassigned
entries blocks completion even when the numeric threshold happens to pass.

`Document` precedes `Video` because it is the denser defect (every one of its 29
models is affected, averaging seven parameters each) and because its models
already share `DocumentNeuralNetworkBase<T>`, so its family base is read off the
existing hierarchy rather than inferred.

**Phases 5, 8 and 9 require explicit numerical-behavior review**, not just a
signature/source-equivalence check:

- Phase 5 can activate previously ignored long-tail options or repair model defaults.
  Require before/after default and explicit-option cases, seeded forward/training
  regression tests through the shared model-family bases, and release notes for every
  intentional default/output change.
- Phase 8 wires audio/speech signal settings and any residual configuration paths.
  Require signal/output-shape and numerical fixtures for affected sample-rate, FFT,
  hop-length, speaking-rate and other applicable settings, plus small end-to-end
  model-family regressions. Document changed behavior and migration guidance.
- Phase 9 changes paper-default values. Require independently reviewed source rows,
  default-value assertions, explicitly small generated execution fixtures, and
  release notes showing the old/new defaults and resource or numerical implications.

This is a minimum list, not permission to ignore behavior changes in other phases.
If any migration starts consuming a formerly ignored constructor/Options value,
the corresponding before/after behavioral proof and release note are required there too.
Mechanical rewiring and intentional numerical repairs should remain separately reviewable.
Several phases exceed the 100-file PR limit; each will be split by family, not by
arbitrary file count.

---

## 9. Risks and open questions

1. **Default-value changes alter results.** Phase 5 may repair long-tail defaults;
   phase 9 explicitly repairs the sequence-model placeholders in §2.4. Pre-v1
   does not remove the obligation to prove and document those changes. Phase 9 is
   last so paper-value changes remain distinguishable from mechanical migration.
2. **Paper values may be too large for CI.** A faithful Mamba default
   (768 × 24) instantiated in a unit test is much heavier than 256 × 4. Tests
   must construct explicitly-small Options rather than relying on defaults; if
   any test depends on the default being small it must be corrected in its shared
   test base or generator before the applicable behavior-changing phase lands.
   *This is the risk most likely to force a design change* — if it turns out that
   many tests depend on small defaults, the alternative is a documented
   `XxxOptions.Small()` factory for test use, which I would rather add
   deliberately than discover under time pressure.
3. **Family-base grouping is inferred from parameter names**, not from a type
   hierarchy that exists today. If two models share a parameter name with
   different meanings, the shared property is wrong. Phase 2 must verify each
   model's usage before hoisting, not trust the name.
4. **Scope size.** The historical three-area subset alone contains 163 models
   and 757 missing parameters, while the authoritative initial baseline is 1067
   in-scope gaps. The final family/model roster must be taken from the ledger.
   Individual areas can ship independently, but stopping after an area is partial
   delivery, not completion of #2090; the remaining count and owners stay visible.
5. **Unverified:** Not every in-scope model's observable behavior has been
   confirmed to derive from the parameters being moved. If some model ignores
   its own constructor parameter today, moving it to Options preserves a
   pre-existing bug rather than fixing it. The §7 behavioural assertion is
   designed to surface exactly this, and will be run before phase 2 is declared
   complete.

---

## 10. Decisions locked before this spec

Recorded so the spec can be reviewed against what was agreed:

- Paper-derived defaults sourced via cached PDFs into a reviewed, human-confirmed
  data file — not generated straight into code
- Scope covers both the 93 doc-promised types and the wider set, with a ratchet
- Bucket-B topology knobs are owned by `NeuralNetworkArchitecture`;
  `AiModelBuilder` is the single user-facing surface
- `Layers` + a contradicting knob is a validation error naming the conflict
- A separate attribute following #2098's `[PaperOptimizer]` pattern
- An empty Options class can be legitimately correct when the complete constructor
  and behavior audit finds no model-specific knobs; the superseded first-constructor
  scan is not evidence for exempting a model.

## 11. Decisions resolved 2026-09-08

1. **§5.3 — remove the long constructor parameter lists**, with no `[Obsolete]`
   forwarding overloads. Pre-v1, and a surviving overload would keep the ratchet
   off zero permanently.
2. **§5.4 — adopt the RT2 pattern** (non-nullable property on the family base,
   paper default in the leaf constructor) for model hyperparameters. `CLAUDE.md`
   has been amended accordingly.
3. **§9.4 — widen #2090 to cover `Video` and `Document`** rather than filing a
   follow-up issue. One sweep, one ratchet, one consistent result; the cost is
   at least the historical 163-model/757-gap subset, plus the broader in-scope
   ledger now required by §2.6 and §8.

4. **`TextToSpeech`, `SpeechRecognition` and `Audio` — decided 2026-09-08: give
   them their own phase (now phase 8).** They were excluded from §5.1 on the
   strength of a measurement that asked the wrong question (§2.6). Including them
   is necessary but not sufficient for the zero-gap target: the 192 estimate
   does not cover the entire residual population. Phase 8b resolves every remaining
   ledger entry, so no in-scope parameters disappear from the completion criterion.
   `AudioHyperparameterOptions` was added to phase 1 as the seventh family base
   (`944a1fd72`): 22 models, 192 tunable parameters, dominated by signal settings
   rather than network shape — `sampleRate` alone appears in 21 of the 22.

Phase 1 was recorded in two commits: `9f7fe07b6` introduced six family bases and
the 1067 baseline; `944a1fd72` added Audio, making **seven family bases in total**.
The original behavioral assertion was skipped pending phase 2. These historical
facts do not imply the revised initialization, semantic-behavior and zero-gap
completion contracts have already been implemented or merged into master.

## 12. Reproducible specification checks

Run the read-only checker from a checkout that contains the cited implementation
commits (the `feature/options-surface-*` history):

```powershell
pwsh -NoProfile -File tools/ValidateModelOptionsSpec.ps1
```

It verifies the seven family names, paths, direct base contract and all 65 declared
shared properties against `944a1fd72`; checks the 1067/977 baseline constants at
`9f7fe07b6`/`671836a34`; and requires continuous phase transitions ending at zero.
Missing Git objects cause a failure rather than an invented or partial result.

Before/after control for this review:

```powershell
# Expected failure: the original roster/schedule cannot satisfy these source-backed contracts.
pwsh -NoProfile -File tools/ValidateModelOptionsSpec.ps1 -SpecRevision cea3a4578e3c14a23b90f4ea3d68b74cb473a36b
# Expected success: the revised specification.
pwsh -NoProfile -File tools/ValidateModelOptionsSpec.ps1
```

The original schedule fails continuity at phases 3 and 8 and finishes at 49;
the revised schedule has nine continuous transitions and finishes at zero.
This is proof of **document/source consistency and arithmetic**, not proof that
model migrations, explicitness tracking, paper corrections or runtime behavioral
tests are already implemented. Those require the phase-specific evidence above.
