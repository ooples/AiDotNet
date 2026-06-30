using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

// UnifiedMultimodalNetwork's default config is a foundation-scale multimodal model: 12 transformer
// layers (DEFAULT_NUM_LAYERS) at embeddingDim=768 with maxSeq=2048, PLUS text/image/audio/video
// encoders AND decoders AND a generation head. It auto-enables weight streaming (SupportsStreaming
// = true; CI observed 22 registered streaming entries), confirming it sits above the streaming
// parameter threshold. On the 16 GB CI runner (DOTNET_GCHeapHardLimit) its construction OOMs
// ("OutOfMemoryException: Array dimensions exceeded supported range"), and a timed-out/OOM'd
// predecessor leaves the process-global WeightRegistry contaminated so the next test fails with
// "existing streaming pool has N registered entries" — a downstream symptom of the size, not a
// separate leak. This is inherent to a paper-scale 4-modality transformer, not a regression and not
// shrinkable (never-shrink rule). Tag HeavyTimeout so the class is excluded from the default gate
// and runs full-fidelity in the nightly heavy lane (deferred, not skipped); #1706/#1305.
[Trait("Category", "HeavyTimeout")]
[Collection("FoundationScaleSerial")] // serialized so its forward gets the whole machine + the streaming reset can't race
public class UnifiedMultimodalNetworkTests : NeuralNetworkModelTestBase<float>
{
    // Auto-enables weight streaming (foundation-scale), registering its weights with the process-
    // global WeightRegistry. Reset that registry between tests so the next ctor doesn't fail on
    // leftover entries (#1706). Safe: the FoundationScaleSerial collection disables parallelization,
    // so nothing else runs concurrently with the reset.
    protected override bool ResetsWeightStreamingBetweenTests => true;

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
