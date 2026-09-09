# Model Options as the User-Facing Configuration Surface

**Issue:** #2090
**Date:** 2026-09-08
**Status:** Approved. Phase 1 landed 2026-09-08 (`9f7fe07b6`).

---

## 1. Summary

Model `Options` classes are intended to be the only user-facing configuration
surface for model-specific parameters, reached through the facade as:

```csharp
var model   = new BGE<double>(architecture, new BGEOptions { NumLayers = 24 });
var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>()
                  .ConfigureModel(model);
```

In `src/NeuralNetworks`, `src/Video` and `src/Document` that surface does not
work. The `Options` object is
accepted, stored, and returned by `GetOptions()`, but **no value is ever read
from it**. Every tunable value lives instead in a defaulted constructor
parameter that has no `Options` equivalent.

This design makes `Options` load-bearing across those three areas, adds a
compiler-checked ratchet so the two surfaces cannot drift apart again, and adds
paper-fidelity tests for the default values — which measurement shows are
currently placeholders, not paper values.

---

## 2. Measured baseline

All figures produced this session against `origin/master` by scanning
`src/NeuralNetworks/*.cs` and `src/**/*Options.cs`. Scripts are in the session
scratchpad; the metric is reproduced permanently by the ratchet test in §7.

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

Base classes do not rescue this. `ModelOptions` declares one property (`Seed`);
`NeuralNetworkOptions` declares one (`EncoderLayerCount`), and a repo-wide
search finds **no read of `EncoderLayerCount` anywhere** — every textual match
is an unrelated local variable.

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

**This is the load-bearing finding: the target design is not new work to invent.
It is already implemented, at scale, in five areas of this repository. #2090 is
the job of bringing one area up to the house standard.**

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

Phase 1's ratchet measured **1067**, against the file-based estimate of 806. The
261-parameter difference is not noise in the counting rules — it is models the
file scan never looked at, because §2.3 selected areas by asking *"do this area's
Options classes declare properties or set constructor defaults?"* That is a
different question from *"do this area's models read them"*, and the areas that
scored cleanest on the first question contain some of the worst offenders on the
second:

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

This is the strongest argument yet for the ratchet being reflection over the
built assembly rather than a source scan: the scan reproduced my assumptions
about where to look, and the reflection did not.

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

`RT2.cs:303` also contains a comment asking for exactly this feature
("extend `RT2Options` with an `EncoderLayerCount` property"), which is
independent evidence that the Options-driven route is the intended direction.

---

## 5. Design

### 5.1 Scope

Across the three areas: 470 parameters in `NeuralNetworks`, 134 in `Video`,
205 in `Document` — **809 total, of which 3 have an Options property, leaving
806 missing**. 806 is the figure the ratchet counts.

Of the 470 in `NeuralNetworks`:

| Group | Models | Params | Disposition |
| --- | ---: | ---: | --- |
| Architecture types (`NeuralNetworkArchitecture`, `Transformer…`, `DualStream…`, `TripleStream…`, `AudioTextDualStream…`) | 5 | 48 | **Out of scope.** Topology knobs stay on the architecture type per the locked bucket-B decision. |
| Infrastructure hosts (`CompiledModelHost`, `ChainedCompiledModelHost`) | 1 | 1 | **Out of scope.** `modelIdentity` / `shapeMode` are not hyperparameters. |
| **Vision-language / multimodal** | 11 | 107 | In scope → `VisionLanguageOptions` |
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
| `src/Document` | 29 | 205 | `DocumentModelOptions` under the existing `DocumentNeuralNetworkBase<T>` hierarchy |
| `src/Video` | 44 | 134 | `VideoModelOptions`, sub-split by task (segmentation, generation, super-resolution, tracking) |

**Grand total in scope: 163 models, 757 parameters.**

Arithmetic: 806 missing, less the 49 out-of-scope (48 architecture types plus
one compiled-model host), leaves 757. Within `NeuralNetworks` the five groups
sum to 98 + 107 + 77 + 41 + 97 = 420, which is 469 missing less the same 49.

The long tail is 40 models averaging 2.4 parameters each. It is the largest
model count and the smallest per-model effort, but it is also where §9.3 (family
grouping inferred from parameter names) is most likely to be wrong, so it is
sequenced last.

### 5.2 Family base classes

Seven base classes, each declaring the shared knobs once. Four are new under
`src/NeuralNetworks/Options/`; the rest live in `src/Models/Options/` or beside
their area. Two of them turned out to exist already, which changed the plan:

- **`DocumentNeuralNetworkOptions` already existed and all 29 Document options
  classes already derive from it.** It was empty. Extending it reaches the whole
  area without touching a single leaf — no new base was needed.
- **Video has no base in use.** 96 of its 108 options classes derive straight
  from `NeuralNetworkOptions`. A `VideoModelOptions<T>` exists but takes a type
  parameter it never uses, follows the nullable + `Effective*` pattern, and is
  derived from by exactly one class; `DocumentModelOptions<T>` is its twin and
  nothing derives from it at all. Both look like an earlier attempt at this same
  work that was never wired up. They are left alone here and removed in their
  areas' phases. The new base is named `VideoHyperparameterOptions` to avoid
  colliding with the abandoned one.

The seven:

- `SequenceModelOptions : NeuralNetworkOptions` — `VocabSize`, `ModelDimension`,
  `NumLayers`, `NumHeads`, `StateDimension`, `MaxSeqLength`, `ExpandFactor`,
  `AttentionInterval`, `FfnMultiplier`
- `VisionLanguageOptions : NeuralNetworkOptions` — `EmbeddingDimension`,
  `MaxSequenceLength`, `ImageSize`, `VisionEmbeddingDim`, `NumFrames`
- `GanOptions : NeuralNetworkOptions` — `LatentSize`, `GeneratorChannels`,
  `DiscriminatorChannels`, `CriticIterations`
- `EmbeddingModelOptions : NeuralNetworkOptions` — `VocabSize`,
  `EmbeddingDimension`, `MaxSequenceLength`, `NumLayers`, `NumHeads`,
  `FeedForwardDim`, `PoolingStrategy` — for the BGE/ColBERT/SGPT/SPLADE/SimCSE/
  Instructor/Matryoshka/FastText/GloVe/Word2Vec/TransformerEmbedding group

Each leaf `XxxOptions` sets its own paper defaults in its parameterless
constructor and adds only genuinely model-specific properties.

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
guessing when the two disagreed. A knob left at its default value is not a
conflict; only an explicitly-set contradicting value is. Explicitness is tracked
by a `HashSet<string>` of assigned property names maintained by the setters on
the family base.

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

- **Baseline: 1067, measured by the ratchet test on 2026-09-08.** This is the
  authoritative figure and it supersedes the 806 estimated below. **Out-of-scope
  floor: 49. In-scope target: 0.**
- Implemented as a reflection test over `AiDotNet.dll`, so it needs no
  documentation to exist and cannot be defeated by #2088's doc deletions.
- **The 806 figure was a file-based proxy, and it was low by 261.** As this
  section promised, the reflection number wins. See §2.6 for what the extra 261
  turned out to be.
- Stored as a single integer in
  `tests/AiDotNet.Tests/IntegrationTests/Configuration/OptionsSurfaceRatchet.txt`,
  alongside the existing `SourceGeneratorCoverageTests` convention. The test
  fails if the count rises, and fails with "lower the baseline" if it drops —
  so the number only moves deliberately.

**Exclusions, stated precisely** (each of these produced a false positive in the
baseline scan and must be excluded by the test, not by the scanner's accident):

- the `options` parameter itself
- parameters whose type is an interface or delegate (optimizer, loss function,
  tokenizer) — collaborators, not configuration
- the five architecture types and two compiled-model hosts of §5.1
- `modelIdentity`, and any parameter typed `string?` defaulting to `null` that
  names an artifact path rather than a hyperparameter

A second assertion pins the fix rather than the shape: for every in-scope model,
constructing it with a non-default Options value must produce a model whose
`GetOptions()` returns that value **and** whose layer stack differs from the
default. Without this, a model could satisfy the count by declaring properties it
still ignores — the exact failure mode of the current code.

---

## 8. Phasing

Each phase is independently mergeable and leaves the build green.

| Phase | Content | Ratchet |
| --- | --- | --- |
| ~~1~~ | ~~Seven family base classes; ratchet test establishing the authoritative baseline~~ **DONE `9f7fe07b6`, `944a1fd72`** | **1067 measured** |
| ~~2~~ | ~~`NeuralNetworks` — sequence / language models (17 models, 90 params)~~ **DONE `671836a34`** | **1067 → 977** |
| 3 | `NeuralNetworks` — vision-language / multimodal (11 models, 107 params) | 708 → 601 |
| 4 | `NeuralNetworks` — embedding & retrieval (11, 77) and GAN (10, 41) | 601 → 483 |
| 5 | `NeuralNetworks` — long tail (40 models, 97 params) | 483 → 386 |
| 6 | `src/Document` (29 models, 203 missing) | 386 → 183 |
| 7 | `src/Video` (44 models, 134 params) | 183 → 49 |
| 8 | `TextToSpeech`, `SpeechRecognition`, `Audio` (22 models, ~192 params) — added 2026-09-08, see §2.6 | 183 → 49 |
| 9 | `docs/model-paper-defaults.tsv`, `[PaperDefaults]`, fidelity test, correction of the §2.4 placeholder values | 49 |

The floor of 49 is the out-of-scope architecture types and compiled-model hosts
of §5.1, which the ratchet excludes from its in-scope count but which are listed
here so the arithmetic is checkable.

`Document` precedes `Video` because it is the denser defect (every one of its 29
models is affected, averaging seven parameters each) and because its models
already share `DocumentNeuralNetworkBase<T>`, so its family base is read off the
existing hierarchy rather than inferred.

**Phases 2-8 are renumbered against the measured 1067 rather than the estimated
806, and the 261 parameters in §2.6 need a phase of their own — see §11.4.** The
per-phase reductions below are unchanged; only the running total shifts.

Phase 8 is the only phase that changes numerical behaviour. Splitting it out
keeps the mechanical rewiring reviewable separately from the value changes.
Several phases exceed the 100-file PR limit; each will be split by family, not by
arbitrary file count.

---

## 9. Risks and open questions

1. **Default-value changes alter results.** Phase 5 changes trained-model
   behaviour for anyone relying on today's `modelDimension = 256`. Pre-v1, so
   acceptable, but it belongs in release notes and is the reason phase 5 is last.
2. **Paper values may be too large for CI.** A faithful Mamba default
   (768 × 24) instantiated in a unit test is much heavier than 256 × 4. Tests
   must construct explicitly-small Options rather than relying on defaults; if
   any test depends on the default being small it will surface in phase 5.
   *This is the risk most likely to force a design change* — if it turns out that
   many tests depend on small defaults, the alternative is a documented
   `XxxOptions.Small()` factory for test use, which I would rather add
   deliberately than discover under time pressure.
3. **Family-base grouping is inferred from parameter names**, not from a type
   hierarchy that exists today. If two models share a parameter name with
   different meanings, the shared property is wrong. Phase 2 must verify each
   model's usage before hoisting, not trust the name.
4. **Scope size.** Widening to `Video` and `Document` (decided 2026-09-08) takes
   this to 163 models and 757 parameters across three areas. That is a large
   change to land before v1, and it is the risk most likely to force a
   re-scoping. The phasing in §8 is ordered so each area is independently
   shippable: if time runs short, `NeuralNetworks` alone still closes the
   user-facing defect, and the ratchet baseline simply stops descending rather
   than the work being left half-migrated.
5. **Unverified:** I have not yet confirmed that every in-scope model's layer
   stack actually derives from the parameters being moved. If some model ignores
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
- An empty Options class can be legitimately correct (74 of 117 models have no
  tunable constructor parameters at all and need none)

## 11. Decisions resolved 2026-09-08

1. **§5.3 — remove the long constructor parameter lists**, with no `[Obsolete]`
   forwarding overloads. Pre-v1, and a surviving overload would keep the ratchet
   off zero permanently.
2. **§5.4 — adopt the RT2 pattern** (non-nullable property on the family base,
   paper default in the leaf constructor) for model hyperparameters. `CLAUDE.md`
   has been amended accordingly.
3. **§9.4 — widen #2090 to cover `Video` and `Document`** rather than filing a
   follow-up issue. One sweep, one ratchet, one consistent result; the cost is
   163 models and 757 in-scope parameters, tracked as the primary risk in §9.4.

4. **`TextToSpeech`, `SpeechRecognition` and `Audio` — decided 2026-09-08: give
   them their own phase (now phase 8).** They were excluded from §5.1 on the
   strength of a measurement that asked the wrong question (§2.6). Including them
   is what lets the ratchet actually reach its floor of 49 rather than stalling
   around 310, so the library ships v1 configured one way rather than two.
   `AudioHyperparameterOptions` was added to phase 1 as the seventh family base
   (`944a1fd72`): 22 models, 192 tunable parameters, dominated by signal settings
   rather than network shape — `sampleRate` alone appears in 21 of the 22.

Phase 1 landed 2026-09-08 as `9f7fe07b6`: six family bases, the ratchet at 1067,
and the behavioural assertion skipped until phase 2.
