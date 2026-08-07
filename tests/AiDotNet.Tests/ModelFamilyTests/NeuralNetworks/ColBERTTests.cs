using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

// #1706/#1305: ColBERT is a full BERT-base encoder (12 layers / 12 heads / 768 hidden / 3072 FFN)
// plus a 768->128 projection. MoreData_ShouldNotDegrade runs 200 iterations of that BERT-scale
// training and is inherently >120s under the suite's single-threaded determinism BLAS even
// uncontended (confirmed: times out in a fully serialized run) — not a regression and not shrinkable.
// Tag HeavyTimeout so it runs full-fidelity nightly (deferred, not skipped). SimCSE/SPLADE/SGPT precedent.
//
// The nightly lane is still executed at full xUnit width, and OptimizerStep_ParamL2_DoesNotExplode was
// timing out THERE — not because it is slow, but because it was starved. Measured alone on the
// CI-matched Release build it takes 5 s against its 120 s gate (24x headroom), which is the
// innocent-bystander profile FoundationScaleSerialCollection exists for. Note the invariant runs a
// SINGLE Train call plus two whole-parameter L2 sweeps, so neither the float rung (already applied —
// this fixture is <float>) nor an iteration cap can reach it: there is no iteration count to trim.
// Serialize so that one step gets the whole machine; the BERT-base fixture stays paper-faithful.
[Trait("Category", "HeavyTimeout")]
[Collection("FoundationScaleSerial")] // dedicated cores (#1622 L4)
public class ColBERTTests : NeuralNetworkModelTestBase<float>
{
    // ColBERT (Khattab & Zaharia 2020) projects 768-dim BERT embeddings
    // down to 128-dim for late interaction retrieval.
    // Uses full paper parameters: 12 layers, 12 heads, 768 hidden, 3072 FFN.
    protected override int[] InputShape => [768];
    protected override int[] OutputShape => [128];

    protected override INeuralNetworkModel<float> CreateNetwork()
        => new ColBERT<float>();
}
