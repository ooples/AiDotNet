using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// Regression tests for issue #1264 — Transformer must default to Adam, not
/// vanilla GradientDescent, on every code path that would otherwise pick a
/// fallback optimizer (constructor and deserialization).
/// </summary>
public class TransformerDefaultOptimizerTests
{
    private static TransformerArchitecture<float> MakeArch() =>
        new TransformerArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            numEncoderLayers: 1,
            numDecoderLayers: 1,
            numHeads: 2,
            modelDimension: 16,
            feedForwardDimension: 32,
            inputSize: 8,
            outputSize: 8,
            maxSequenceLength: 4,
            vocabularySize: 8);

    private static string GetOptimizerName<T>(Transformer<T> t)
    {
        // GetModelMetadata.AdditionalInfo["Optimizer"] holds the runtime
        // optimizer's GetType().Name — the only public surface to inspect
        // it (the field is private).
        var metadata = t.GetModelMetadata();
        return metadata.AdditionalInfo!["Optimizer"]?.ToString() ?? "";
    }

    [Fact]
    public void Constructor_NullOptimizer_DefaultsToAdam()
    {
        var transformer = new Transformer<float>(MakeArch(), optimizer: null);
        Assert.Contains("Adam", GetOptimizerName(transformer));
    }

    [Fact]
    public void Deserialize_MissingOptimizer_FallsBackToAdam_NotGradientDescent()
    {
        // Build a Transformer with no optimizer (so the wire format will not
        // carry one), serialize, then deserialize on a fresh instance and
        // assert the deserialized optimizer is Adam — NOT the legacy
        // GradientDescentOptimizer fallback that issue #1264 reported.
        var src = new Transformer<float>(MakeArch(), optimizer: null);

        // The public Serialize/Deserialize round-trip uses byte[] (not
        // BinaryReader/Writer); the per-network DeserializeNetworkSpecificData
        // hook still receives a BinaryReader internally, which is what the
        // PR's null-fallback fix lives in.
        var bytes = src.Serialize();
        var dst = new Transformer<float>(MakeArch(), optimizer: null);
        dst.Deserialize(bytes);

        var optimizerName = GetOptimizerName(dst);
        Assert.Contains("Adam", optimizerName);
        Assert.DoesNotContain("GradientDescent", optimizerName);
    }
}
