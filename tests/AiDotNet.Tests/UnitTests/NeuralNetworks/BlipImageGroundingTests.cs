using System;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tokenization;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// BLIP's image-grounded text encoder (image-text matching) and image-grounded text decoder (captioning, VQA)
/// cross-attend to the image in every block (Li et al. 2022, §3.1).
/// </summary>
/// <remarks>Both native paths used to run their blocks on the text alone ("Simple implementation: just use text
/// features"), so the image never reached the matching score or the caption: these tests change only the image.</remarks>
public sealed class BlipImageGroundingTests
{
    private const int ImageSize = 16;

    private static BlipNeuralNetwork<float> CreateModel() => new(
        new NeuralNetworkArchitecture<float>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.ImageClassification,
            inputHeight: ImageSize, inputWidth: ImageSize, inputDepth: 3,
            outputSize: 3) { RandomSeed = 1337 },
        new BlipOptions
        {
            ImageSize = ImageSize, PatchSize = 8, Channels = 3, EmbeddingDimension = 8,
            HiddenDim = 8, MaxSequenceLength = 4, VocabSize = 512,
            NumEncoderLayers = 1, NumDecoderLayers = 1, NumHeads = 2, MlpDim = 16
        },
        tokenizer: ClipTokenizerFactory.CreateShapeCompatibleForTesting(512, new[] { "a" }));

    private static Tensor<float> Image(double phase)
    {
        var image = new Tensor<float>(new[] { 3, ImageSize, ImageSize });
        for (int i = 0; i < image.Length; i++) image[i] = (float)Math.Sin(0.37 * i + phase);
        return image;
    }

    [Fact(Timeout = 120000)]
    public async Task ImageTextMatch_DependsOnTheImage()
    {
        await Task.Yield();
        using var model = CreateModel();
        float first = model.ComputeImageTextMatch(Image(0.0), "a");
        float second = model.ComputeImageTextMatch(Image(1.3), "a");
        Assert.NotEqual(first, second);
    }

    [Fact(Timeout = 120000)]
    public async Task CaptionDecoder_DependsOnTheImage()
    {
        await Task.Yield();
        using var model = CreateModel();
        var tokens = new Tensor<float>(new[] { 1, 2 });
        tokens[0, 0] = 1; tokens[0, 1] = 2;
        var first = model.ForwardDecoderNative(tokens, model.GetImageFeaturesNative(Image(0.0)));
        var second = model.ForwardDecoderNative(tokens, model.GetImageFeaturesNative(Image(1.3)));
        Assert.Equal(first.Shape.ToArray(), second.Shape.ToArray());
        double largest = 0;
        for (int i = 0; i < first.Length; i++) largest = Math.Max(largest, Math.Abs(first[i] - second[i]));
        Assert.True(largest > 1e-6, "The decoder's logits ignored a change of image.");
    }
}
