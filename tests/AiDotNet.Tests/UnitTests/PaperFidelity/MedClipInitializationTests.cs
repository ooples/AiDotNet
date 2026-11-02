using AiDotNet.Enums;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.VisionLanguage.Encoders;
using Xunit;

namespace AiDotNet.Tests.UnitTests.PaperFidelity;

public class MedClipInitializationTests
{
    private static MedCLIP<double> Create() => new(new NeuralNetworkArchitecture<double>(
        inputType: InputType.ThreeDimensional, taskType: NeuralNetworkTaskType.Embedding,
        inputHeight: 32, inputWidth: 32, inputDepth: 3, outputSize: 16) { RandomSeed = 42 },
        new MedCLIPOptions { ImageSize = 32, PatchSize = 2, VisionEmbeddingDim = 32,
            TextEmbeddingDim = 32, ProjectionDim = 16, NumVisionLayers = 2, NumTextLayers = 2,
            NumVisionHeads = 4, NumTextHeads = 4, VocabSize = 64, MaxSequenceLength = 8,
            DropoutRate = 0.0, Seed = 42 });

    private static MedCLIP<double> Rebuild()
    {
        using var shell = Create();
        return (MedCLIP<double>)CloneEngine.CopyConfiguration(shell);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void SeededModelsPreserveLazyProjectionWeightsAndOutputs(bool rebuild)
    {
        using var a = rebuild ? Rebuild() : Create();
        using var b = rebuild ? Rebuild() : Create();
        a.SetTrainingMode(false);
        b.SetTrainingMode(false);
        var input = new Tensor<double>(new[] { 1, 3, 32, 32 });
        for (int i = 0; i < input.Length; i++) input[i] = Math.Sin(i);
        var x = a.Predict(input);
        var y = b.Predict(input);
        for (int i = 0; i < a.Layers.Count; i++)
            Assert.True(a.Layers[i].GetParameters().ToArray().SequenceEqual(b.Layers[i].GetParameters().ToArray()),
                $"Layer {i}: {a.Layers[i].GetType().Name}");
        Assert.Equal(x.ToArray(), y.ToArray());
    }
}
