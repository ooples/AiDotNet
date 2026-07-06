using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

// #1706/#1305: BERT-encoder + 30522-vocab MLM head. MoreData_ShouldNotDegrade (200-iteration
// training) is inherently >120s under single-threaded determinism BLAS even uncontended (verified:
// it times out even when serialized) — not a regression and not shrinkable. Tag HeavyTimeout so it
// runs full-fidelity nightly (deferred, not skipped); RequiresHeavySerialization serializes it there.
[Trait("Category", "HeavyTimeout")]
public class SPLADETests : NeuralNetworkModelTestBase<float>
{
    protected override bool RequiresHeavySerialization => true;

    // SPLADE produces a sparse vocab-sized [VocabSize=30522] activation
    // vector (BERT vocabulary), not the architecture.OutputSize=768 it
    // advertises. Test base shape must match the actual prediction shape.
    protected override int[] InputShape => [1];
    protected override int[] OutputShape => [30522];

    protected override INeuralNetworkModel<float> CreateNetwork()
        => new SPLADE<float>();
}
