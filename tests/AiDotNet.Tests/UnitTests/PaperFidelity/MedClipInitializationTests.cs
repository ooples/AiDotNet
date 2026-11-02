using AiDotNet.Enums;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.VisionLanguage.Encoders;
using Xunit;

namespace AiDotNet.Tests.UnitTests.PaperFidelity;

public class MedClipInitializationTests
{
    private sealed class NormalizationProbe : MedCLIP<double>
    {
        public NormalizationProbe(double[]? std = null) : base(new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional, taskType: NeuralNetworkTaskType.Embedding,
            inputHeight: 32, inputWidth: 32, inputDepth: 3, outputSize: 16),
            new MedCLIPOptions { ImageSize = 32, VisionEmbeddingDim = 32, TextEmbeddingDim = 32,
                ProjectionDim = 16, NumVisionLayers = 2, NumTextLayers = 2,
                NumVisionHeads = 4, NumTextHeads = 4, VocabSize = 64, MaxSequenceLength = 8,
                ImageStd = std ?? new[] { 1.0, 1.0, 1.0 } }) { }

        public Tensor<double> Normalize(Tensor<double> image, double[] mean, double[] std)
            => NormalizeImage(image, mean, std);
    }

    [Fact]
    public void InvalidChannelStatisticsAreRejectedBeforeBuildingTheModel()
    {
        Assert.Throws<ArgumentException>(() => new NormalizationProbe(new[] { 1.0, 1.0 }));
    }

    [Fact]
    public void ImageNormalizationUsesEachImagesChannelStatistics()
    {
        using var model = new NormalizationProbe();
        var input = new Tensor<double>(new[] { 2, 3, 1, 1 });
        for (int i = 0; i < input.Length; i++) input[i] = (i + 1) * 10;
        var result = model.Normalize(input, new[] { 1.0, 2.0, 3.0 }, new[] { 1.0, 2.0, 3.0 });
        Assert.Equal(new[] { 9.0, 9.0, 9.0, 39.0, 24.0, 19.0 }, result.ToArray());
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void ImageNormalizationRejectsMissingChannelStatistics(bool shortenMean)
    {
        using var model = new NormalizationProbe();
        var mean = shortenMean ? new[] { 0.0, 0.0 } : new[] { 0.0, 0.0, 0.0 };
        var std = shortenMean ? new[] { 1.0, 1.0, 1.0 } : new[] { 1.0, 1.0 };
        Assert.Throws<ArgumentException>(() => model.Normalize(new Tensor<double>(new[] { 3, 1, 1 }), mean, std));
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(-1.0)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    public void ImageNormalizationRejectsInvalidStandardDeviation(double std)
    {
        using var model = new NormalizationProbe();
        Assert.Throws<ArgumentOutOfRangeException>(() => model.Normalize(
            new Tensor<double>(new[] { 3, 1, 1 }), new[] { 0.0, 0.0, 0.0 }, new[] { std, 1.0, 1.0 }));
    }

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
