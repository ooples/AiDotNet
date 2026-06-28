using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

// #1706/#1305: BERT-scale encoder. The single-forward tests fit the timeout once serialized, but
// the training invariants (MoreData_ShouldNotDegrade = 200-iteration training, Training_ShouldReduceLoss)
// are inherently >120s under single-threaded determinism BLAS even uncontended — not a regression and
// not shrinkable. Tag HeavyTimeout so it runs full-fidelity in the nightly lane (deferred, not skipped);
// RequiresHeavySerialization additionally serializes it there so the heavy lane doesn't self-contend.
[Trait("Category", "HeavyTimeout")]
public class SimCSETests : NeuralNetworkModelTestBase<float>
{
    protected override bool RequiresHeavySerialization => true;

    // Per Gao et al. (2021): SimCSE outputs [CLS] embeddings of size embeddingDimension (768)
    protected override int[] InputShape => [1, 768];
    protected override int[] OutputShape => [1, 768];

    protected override INeuralNetworkModel<float> CreateNetwork()
        => new SimCSE<float>();
}
