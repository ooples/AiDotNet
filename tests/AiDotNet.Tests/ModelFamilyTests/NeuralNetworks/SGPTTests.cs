using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

// #1706/#1305: GPT-style sentence-embedding transformer. Training_ShouldReduceLoss /
// Clone_ShouldProduceIdenticalOutput run two full generations of training/inference and are
// inherently >120s under single-threaded determinism BLAS even uncontended — not a regression and
// not shrinkable. Tag HeavyTimeout so it runs full-fidelity nightly (deferred, not skipped);
// RequiresHeavySerialization serializes it there.
[Trait("Category", "HeavyTimeout")]
public class SGPTTests : NeuralNetworkModelTestBase<float>
{
    protected override bool RequiresHeavySerialization => true;

    protected override INeuralNetworkModel<float> CreateNetwork()
        => new SGPT<float>();
}
