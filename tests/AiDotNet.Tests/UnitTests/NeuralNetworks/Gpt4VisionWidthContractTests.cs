using System;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Tokenization;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

public class Gpt4VisionWidthContractTests
{
    public Gpt4VisionWidthContractTests() => TestModuleInitializer.EnsureInitialized();

    [Theory]
    [InlineData(8)]
    [InlineData(16)]
    public void SharedVisionWidth_ControlsActualNativePatchProjection(int width)
    {
        var options = new Gpt4VisionOptions
        {
            VisionDim = width, EmbeddingDimension = 16, HiddenDim = 16,
            VisionLayers = 1, NumLmLayers = 1, NumHeads = 2,
            ImageSize = 8, PatchSize = 4, MaxSequenceLength = 8,
            ContextWindowSize = 16, VocabSize = 512, MaxImagesPerRequest = 1
        };
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.ThreeDimensional, NeuralNetworkTaskType.ImageClassification,
            inputDepth: 3, inputHeight: 8, inputWidth: 8, outputSize: 16) { RandomSeed = 1337 };
        var tokenizer = ClipTokenizerFactory.CreateShapeCompatibleForTesting(512, new[] { "a" });
        using var model = new Gpt4VisionNeuralNetwork<float>(architecture, tokenizer, options);
        var projection = Assert.IsType<PatchEmbeddingLayer<float>>(model.Layers[0]);
        var image = new Tensor<float>(new[] { 3, 8, 8 });
        for (int index = 0; index < image.Length; index++)
            image[index] = (index % 17 + 1) / 17.0f;

        // Inspect the real tensor and learned projection, not only an options/metadata echo.
        var patches = projection.Forward(image);
        Assert.Equal(new[] { 4, width }, patches.Shape.ToArray());
        Assert.Equal((3 * 4 * 4 + 1) * width, projection.GetParameters().Length);
        for (int index = 0; index < patches.Length; index++)
            Assert.True(!float.IsNaN(patches[index]) && !float.IsInfinity(patches[index]));
        Assert.Equal(width, model.ImageEmbeddingDimension);
        Assert.Same(options, model.GetOptions());
    }
}
