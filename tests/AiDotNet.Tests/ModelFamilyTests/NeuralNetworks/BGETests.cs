using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

// #1706/#1305: BGE is a full BERT-base encoder (12 layers / 768 hidden / 3072 FFN). Its
// MoreData_ShouldNotDegrade runs 200 iterations of BERT-scale training and is inherently >120s
// under the suite's single-threaded determinism BLAS even uncontended (confirmed: it times out in a
// fully serialized run) — not a regression and not shrinkable (never-shrink rule). Tag HeavyTimeout
// so it runs full-fidelity in the nightly heavy lane (deferred, not skipped); it graduates back once
// BERT-scale training is fast enough. Matches the SimCSE/SPLADE/SGPT embedding-model precedent.
[Trait("Category", "HeavyTimeout")]
public class BGETests : NeuralNetworkModelTestBase<float>
{
    // BGE embedding model: input/output both use BERT-base 768-dim embeddings
    protected override int[] InputShape => [768];
    protected override int[] OutputShape => [768];

    protected override INeuralNetworkModel<float> CreateNetwork()
        => new BGE<float>();
}
