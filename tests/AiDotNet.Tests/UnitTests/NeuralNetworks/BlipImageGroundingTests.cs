using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tokenization;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// BLIP's image-grounded text encoder (image-text matching) and image-grounded text decoder (captioning, VQA)
/// cross-attend to the image in every block, and the image-grounded encoder is the text encoder itself with that
/// cross-attention switched on (Li et al. 2022, §3.1). Issue #2297.
/// </summary>
/// <remarks>Both native paths used to run their blocks on the text alone ("Simple implementation: just use text
/// features"), so the image never reached the matching score or the caption. The image tests change only the image and
/// hold the text fixed.</remarks>
public sealed class BlipImageGroundingTests
{
    private const int ImageSize = 16;
    private const int EncoderLayers = 2;
    private const int DecoderLayers = 1;

    public BlipImageGroundingTests() => TestModuleInitializer.EnsureInitialized();

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
            NumEncoderLayers = EncoderLayers, NumDecoderLayers = DecoderLayers, NumHeads = 2, MlpDim = 16
        },
        tokenizer: ClipTokenizerFactory.CreateShapeCompatibleForTesting(512, new[] { "a" }));

    private static Tensor<float> Image(double phase)
    {
        var image = new Tensor<float>(new[] { 3, ImageSize, ImageSize });
        for (int i = 0; i < image.Length; i++) image[i] = (float)Math.Sin(0.37 * i + phase);
        return image;
    }

    private static double LargestDifference(Tensor<float> first, Tensor<float> second)
    {
        Assert.Equal(first.Shape.ToArray(), second.Shape.ToArray());
        double largest = 0;
        for (int i = 0; i < first.Length; i++) largest = Math.Max(largest, Math.Abs(first[i] - second[i]));
        return largest;
    }

    [Fact(Timeout = 120000)]
    public async Task ImageTextMatch_DependsOnTheImage()
    {
        await Task.Yield();
        using var model = CreateModel();
        float first = model.ComputeImageTextMatch(Image(0.0), "a");
        float second = model.ComputeImageTextMatch(Image(1.3), "a");
        // float.IsFinite does not exist on net471, so the check is spelled out.
        Assert.True(!float.IsNaN(first) && !float.IsInfinity(first), $"The first ITM score is not finite: {first}.");
        Assert.True(!float.IsNaN(second) && !float.IsInfinity(second), $"The second ITM score is not finite: {second}.");
        Assert.NotEqual(first, second);
    }

    [Fact(Timeout = 120000)]
    public async Task ImageGroundedTextEncoder_DependsOnTheImage()
    {
        await Task.Yield();
        using var model = CreateModel();
        var first = model.GetTextFeaturesNative("a", model.GetImageFeaturesNative(Image(0.0)));
        var second = model.GetTextFeaturesNative("a", model.GetImageFeaturesNative(Image(1.3)));
        Assert.True(LargestDifference(first, second) > 1e-6, "The image-grounded text encoder ignored a change of image.");
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
        Assert.True(LargestDifference(first, second) > 1e-6, "The decoder's logits ignored a change of image.");
    }

    [Fact(Timeout = 120000)]
    public async Task UnimodalTextEncoder_IsTheImageGroundedEncoderWithoutTheImage()
    {
        // ITC and ITM run the same blocks (§3.1): only the cross-attention differs, so the unimodal states differ
        // from the image-grounded ones, and the unimodal ones are the same whichever image the model saw last.
        await Task.Yield();
        using var model = CreateModel();
        var unimodalBefore = model.GetTextFeaturesNative("a", imageFeatures: null);
        var grounded = model.GetTextFeaturesNative("a", model.GetImageFeaturesNative(Image(0.0)));
        var unimodalAfter = model.GetTextFeaturesNative("a", imageFeatures: null);

        Assert.True(LargestDifference(unimodalBefore, grounded) > 1e-6,
            "The image-grounded encoder produced the unimodal encoder's states, so it did not attend to the image.");
        Assert.Equal(0.0, LargestDifference(unimodalBefore, unimodalAfter));
    }

    [Fact(Timeout = 120000)]
    public async Task TextBlocks_FollowThePaperStructure()
    {
        await Task.Yield();
        using var model = CreateModel();
        var blocks = model.Layers.OfType<TransformerDecoderLayer<float>>().ToArray();

        // One set of text-encoder blocks serves ITC and ITM; there is no separate set of matching blocks.
        Assert.Equal(EncoderLayers + DecoderLayers, blocks.Length);
        // The only plain encoder blocks left are the ViT's: the text encoder's blocks carry cross-attention.
        Assert.Equal(EncoderLayers, model.Layers.OfType<TransformerEncoderLayer<float>>().Count());
        Assert.All(blocks.Take(EncoderLayers), block => Assert.False(block.IsCausal));
        Assert.All(blocks.Skip(EncoderLayers), block => Assert.True(block.IsCausal));
    }

    [Theory(Timeout = 120000)]
    [InlineData(true)]
    [InlineData(false)]
    public async Task DecoderBlock_SelfAttentionDirectionFollowsTheCausalFlag(bool causal)
    {
        // A causal block's first position cannot see a change at the last position; a bidirectional block's can.
        await Task.Yield();
        var block = new TransformerDecoderLayer<float>(numHeads: 2, feedForwardDim: 16, causal: causal) { RandomSeed = 7 };
        var text = new Tensor<float>(new[] { 1, 3, 8 });
        for (int i = 0; i < text.Length; i++) text[i] = (float)Math.Cos(0.29 * i);
        var changed = text.Clone();
        for (int f = 0; f < 8; f++) changed[0, 2, f] += 0.75f;
        var image = new Tensor<float>(new[] { 1, 5, 8 });
        for (int i = 0; i < image.Length; i++) image[i] = (float)Math.Sin(0.41 * i);

        var before = block.Forward(text, image);
        var after = block.Forward(changed, image);
        double firstPositionChange = 0;
        for (int f = 0; f < 8; f++) firstPositionChange = Math.Max(firstPositionChange, Math.Abs(before[0, 0, f] - after[0, 0, f]));

        Assert.Equal(causal, block.IsCausal);
        if (causal)
            Assert.True(firstPositionChange < 1e-6, $"A causal block's first position saw the last token (change {firstPositionChange}).");
        else
            Assert.True(firstPositionChange > 1e-6, "A bidirectional block's first position ignored the last token.");
    }
}
