using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

// UnifiedMultimodalNetwork is foundation-scale (768-dim embeddings, 12 transformer layers
// plus text/image/audio/video encoders + cross-modal attention + decoders — over the 500M
// streaming threshold). Its multi-iteration training invariants (MoreData_ShouldNotDegrade
// trains 50 + 200 steps across an original AND a clone) are CORRECT but inherently exceed
// the 120s default per-test gate under single-threaded determinism BLAS — verified a genuine
// timeout (the test runs the full 120s, no hang/exception), not a regression. Per the #1706
// strategy these are tagged HeavyTimeout: excluded from the default PR gate (Category!=HeavyTimeout)
// and run full-fidelity in the nightly heavy lane (deferred, not skipped; graduates back once
// the foundation forward is fast enough). The separate WeightRegistry single-tenant collision
// this model also hit (original + clone both auto-streaming) is fixed at the source in
// NeuralNetworkBase.TryAutoEnableWeightStreaming (decline streaming when the pool is occupied). #1738.
[Xunit.Trait("Category", "HeavyTimeout")]
public class UnifiedMultimodalNetworkTests : NeuralNetworkModelTestBase<float>
{
    // Default: inputSize=768 (embedding dim), outputSize=100
    protected override int[] InputShape => [768];
    protected override int[] OutputShape => [100];

    protected override INeuralNetworkModel<float> CreateNetwork()
        => new UnifiedMultimodalNetwork<float>();

    // UnifiedMultimodalNetwork is a MultiClassClassification model trained with
    // CrossEntropyWithLogitsLoss, so its paper-faithful target is a one-hot class
    // label, not a continuous-uniform tensor. The test base documents this exact
    // override hook for "multi-class classification with cross-entropy" families
    // (see CreateRandomTargetTensor remarks). A continuous-uniform target is not a
    // valid class distribution: cross-entropy can never drive the prediction onto
    // it, so the MSE proxy never reaches zero and two independent random runs are
    // not comparable. With a one-hot target the softmax prediction converges to the
    // label (MSE -> 0), which is the correct objective for this model.
    protected override Tensor<float> CreateRandomTargetTensor(int[] shape, System.Random rng)
    {
        var target = new Tensor<float>(shape);
        // Pick a single active class along the last axis for each row.
        int numClasses = shape[^1];
        int rows = target.Length / numClasses;
        for (int r = 0; r < rows; r++)
            target[r * numClasses + rng.Next(numClasses)] = 1.0f;
        return target;
    }
}
