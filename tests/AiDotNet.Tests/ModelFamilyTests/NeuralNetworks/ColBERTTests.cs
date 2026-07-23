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
[Trait("Category", "HeavyTimeout")]
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
