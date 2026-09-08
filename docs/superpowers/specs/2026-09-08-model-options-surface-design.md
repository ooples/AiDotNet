# Model Options as the User-Facing Configuration Surface

**Issue:** #2090
**Date:** 2026-09-08
**Status:** Draft for review — no implementation until approved

---

## 1. Summary

Model `Options` classes are intended to be the only user-facing configuration
surface for model-specific parameters, reached through the facade as:

```csharp
var model   = new BGE<double>(architecture, new BGEOptions { NumLayers = 24 });
var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>()
                  .ConfigureModel(model);
```

In `src/NeuralNetworks` that surface does not work. The `Options` object is
accepted, stored, and returned by `GetOptions()`, but **no value is ever read
from it**. Every tunable value lives instead in a defaulted constructor
parameter that has no `Options` equivalent.

This design makes `Options` load-bearing for the 90 affected models, adds a
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
fully-inert classes in the entire repository, 99 are in `src/NeuralNetworks`.

**This is the load-bearing finding: the target design is not new work to invent.
It is already implemented, at scale, in five areas of this repository. #2090 is
the job of bringing one area up to the house standard.**

### 2.4 The defaults in `src/NeuralNetworks` are placeholders, not paper values

This contradicts an earlier conclusion of mine and is stated plainly because it
changes the scope of the work.

Among the 17 language models:

- **all 17** default `modelDimension = 256`
- **all 17** default `maxSeqLength = 512`
- **13 of 17** default `numLayers = 4`

Seventeen different papers cannot agree on `d_model = 256`. Mamba-130M is
768 × 24; RWKV-4 "Raven" is far larger; Griffin and RecurrentGemma are larger
still. These are demo-sized values shared by copy.

For contrast, where the value came from a real source it is visibly correct:

- `BGE` — `vocabSize 30522, embeddingDimension 768, numLayers 12, numHeads 12, feedForwardDim 3072` (BERT-base, exactly)
- `RT2Options` — `VisionDim 1024, DecoderDim 4096, NumVisionLayers 24, NumDecoderLayers 32, NumHeads 32` (PaLI-X)

So the repository has two tiers: paper-faithful defaults (`LayerHelper`
factories, `VisionLanguage`, `Audio`, `SpeechRecognition`) and placeholder
defaults (the `NeuralNetworks` sequence models). Paper verification is therefore
**not** a redundant audit of values that are already right — it has real defects
to find, and it is in scope for this work.

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

Of the 470 tunable parameters:

| Group | Models | Params | Disposition |
| --- | ---: | ---: | --- |
| Architecture types (`NeuralNetworkArchitecture`, `Transformer…`, `DualStream…`, `TripleStream…`, `AudioTextDualStream…`) | 5 | 48 | **Out of scope.** Topology knobs stay on the architecture type per the locked bucket-B decision. |
| Infrastructure hosts (`CompiledModelHost`, `ChainedCompiledModelHost`) | 1 | 1 | **Out of scope.** `modelIdentity` / `shapeMode` are not hyperparameters. |
| **Vision-language / multimodal** | 11 | 107 | In scope → `VisionLanguageOptions` |
| **Sequence / language models** | 18 | 98 | In scope → `SequenceModelOptions` |
| **Embedding & retrieval** (BGE, ColBERT, SGPT, SPLADE, SimCSE, Instructor, Matryoshka, FastText, GloVe, Word2Vec, TransformerEmbedding) | 11 | 77 | In scope → `EmbeddingModelOptions` |
| **GAN family** | 10 | 41 | In scope → `GanOptions` |
| **Long tail** (graph nets, classic CNN/RNN, autoencoders, RBM/DBM, spiking, mesh/voxel, …) | 40 | 97 | In scope; sub-clustered in phase 5 |
| **Total in scope** | **90** | **421** | |

The long tail is 40 models averaging 2.4 parameters each. It is the largest
model count and the smallest per-model effort, but it is also where §9.3 (family
grouping inferred from parameter names) is most likely to be wrong, so it is
sequenced last.

### 5.2 Family base classes

Four new base classes under `src/NeuralNetworks/Options/`, each declaring the
shared knobs once:

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

**Deviation flagged for your decision.** When you chose this option it was
described as keeping the long parameter lists as forwarding overloads. Your
reply — "moving all constructor parameters to that model options class" — reads
as removal, and removal is what makes the ratchet in §7 meaningful: a forwarding
overload keeps the second surface alive, which is the thing being fixed. AiDotNet
has not shipped v1, so there is no compatibility obligation. **If you want the
overloads retained for a release, say so and §5.3 changes to keep them marked
`[Obsolete]`.**

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
keeps the nullable pattern unchanged. **This is a deliberate deviation from a
standing rule and needs your explicit acceptance.**

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

- **Baseline today: 470. Out-of-scope floor: 49. In-scope target: 0.**
- Implemented as a reflection test over `AiDotNet.dll`, so it needs no
  documentation to exist and cannot be defeated by #2088's doc deletions.
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
- the four architecture types and two compiled-model hosts of §5.1
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
| 1 | Four family base classes; ratchet test with baseline 470; the §7 behavioural assertion marked `Skip` until phase 2 | 470 |
| 2 | Sequence / language models (18) — properties, defaults, constructor rewrite, `CreateNewInstance` | 470 → 372 |
| 3 | Vision-language / multimodal (11) | 372 → 265 |
| 4 | Embedding & retrieval (11) and GAN (10) | 265 → 147 |
| 5 | Long tail (40); every in-scope model now reads its Options | 147 → 49 |
| 6 | `docs/model-paper-defaults.tsv`, `[PaperDefaults]`, fidelity test, correction of the §2.4 placeholder values | 49 |

The floor of 49 is the out-of-scope architecture types and compiled-model hosts
of §5.1, which the ratchet excludes from its in-scope count but which are listed
here so the arithmetic is checkable.

Phase 6 is the only phase that changes numerical behaviour. Splitting it out
keeps the mechanical rewiring reviewable separately from the value changes.
Phases 2-5 are each large enough to exceed the 100-file PR limit for the bigger
groups; each will be split by family, not by arbitrary file count.

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
4. **`Video` (43 inert) and `Document` (27 inert)** have the same defect and are
   not in this spec's scope. They should get their own issue rather than be
   silently absorbed here.
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

## 11. Decisions still needed from you

1. §5.3 — remove the long constructor parameter lists, or keep them as
   `[Obsolete]` forwarding overloads for one release?
2. §5.4 — accept the deviation from CLAUDE.md's nullable + `GetEffectiveX()`
   rule for model hyperparameters?
3. §9.4 — file a separate issue for `Video` and `Document`, or widen this one?
