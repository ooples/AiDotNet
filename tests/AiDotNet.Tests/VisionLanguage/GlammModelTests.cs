using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.VisionLanguage.Grounding;
using Xunit;

namespace AiDotNet.Tests.VisionLanguage;

/// <summary>
/// GLaMM's public surface on a small configuration: generation with a mask per [SEG], the official prompts, training on a
/// conversation with masks (only the officially trained modules change), and the checks on its supervision.
/// </summary>
public class GlammModelTests
{
    private static GLaMM<double> Small(int seed = 7) => new(
        new NeuralNetworkArchitecture<double>(inputType: InputType.ThreeDimensional, taskType: NeuralNetworkTaskType.ImageSegmentation,
            inputHeight: 28, inputWidth: 28, inputDepth: 3, outputSize: 32),
        new GLaMMOptions
        {
            Seed = seed, TokenizerModel = GlammTokenizerReferenceTests.TestModel(),
            ClipImageSize = 28, ClipPatchSize = 7, ClipHidden = 64, ClipLayers = 10, ClipHeads = 4, ClipIntermediate = 128,
            LlmHidden = 32, LlmLayers = 2, LlmHeads = 4, LlmKeyValueHeads = 4, LlmIntermediate = 64, LoraRank = 4, LoraAlpha = 8, LoraDropout = 0,
            RegionWidth = 32, SamImageSize = 64, SamPatchSize = 16, SamEncoderDim = 32, SamEncoderDepth = 2, SamEncoderHeads = 2, SamWindowSize = 2,
            SamGlobalAttention = new[] { 1 }, SamPromptDim = 16, SamMaskInChannels = 8, SamDecoderHeads = 2, SamDecoderMlpDim = 32, SamIouHiddenDim = 16,
            ModelMaxLength = 256,
        });

    private static Tensor<double> Image(int h = 21, int w = 35)
    {
        var image = new Tensor<double>(new[] { 3, h, w });
        for (int c = 0; c < 3; c++)
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    image[c, y, x] = (x > w / 3 && y > h / 4 ? 0.8 : 0.2) + 0.05 * c;
        return image;
    }

    private static Tensor<double> Mask(int h = 21, int w = 35)
    {
        var mask = new Tensor<double>(new[] { h, w });
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                mask[y, x] = x > w / 3 && y > h / 4 ? 1 : 0;
        return mask;
    }

    [Fact]
    public void Construction_WithoutATokenizer_SaysWhereToGetOne()
    {
        var ex = Assert.Throws<ArgumentException>(() => new GLaMM<double>(new NeuralNetworkArchitecture<double>(inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.ImageSegmentation, inputHeight: 28, inputWidth: 28, inputDepth: 3, outputSize: 32)));
        Assert.Contains("TokenizerPath", ex.Message);
    }

    [Fact(Timeout = 300000)]
    public async Task Chat_ReturnsAMaskPerSegToken_AtTheImageSize()
    {
        await Task.Yield();
        using var model = Small();
        var image = Image();
        var answer = model.Chat(image, "Can you segment the dog in this image?", maxNewTokens: 6);
        Assert.NotNull(answer.Text);
        Assert.Equal(answer.Text.Split(new[] { "[SEG]" }, StringSplitOptions.None).Length - 1, answer.Masks.Count);
        Assert.All(answer.Masks, m => Assert.Equal(new[] { 21, 35 }, m.Shape.ToArray()));
        Assert.Equal(new[] { 21, 35 }, model.SegmentExpression(image, "Dog").Shape.ToArray());
        Assert.NotNull(model.CaptionRegion(image, new[] { 0.1, 0.2, 0.6, 0.9 }, maxNewTokens: 4));
        Assert.Equal(new[] { 16, 32 }, model.EncodeImage(image).Shape.ToArray());
    }

    [Fact(Timeout = 600000)]
    public async Task TrainConversation_LowersTheLoss_AndLeavesTheFrozenModulesAlone()
    {
        await Task.Yield();
        using var model = Small();
        var sample = new GlammSample<double>(Image(),
            new[] { new GlammTurn("Can you segment the dog near region1 <bbox>?", "Sure, it is [SEG].") },
            new[] { new[] { 0.3, 0.2, 0.9, 0.95 } }, new[] { Mask() });
        var clipBefore = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(model.Network.ClipLayers).Select(p => p.ToArray()).ToList();
        var samEncoderBefore = model.Network.Sam.PositionTable.GetParameters().ToArray();
        var loraBefore = model.Network.Llm.Layers[0].QueryLora!.B.GetParameters().ToArray();
        var regionBefore = model.Network.Region.UpDims.GetParameters().ToArray();
        var decoderBefore = model.Network.Sam.IouToken.GetParameters().ToArray();
        double before = model.EvaluateConversationLoss(sample);
        // The paper schedule warms up linearly over 100 steps from zero, so a few steps move the loss only a little.
        for (int step = 0; step < 15; step++) model.TrainConversation(sample);
        double after = model.EvaluateConversationLoss(sample);
        Assert.True(after < before, $"{before} -> {after}");
        Assert.NotEqual(loraBefore, model.Network.Llm.Layers[0].QueryLora!.B.GetParameters().ToArray());
        Assert.NotEqual(regionBefore, model.Network.Region.UpDims.GetParameters().ToArray());
        Assert.NotEqual(decoderBefore, model.Network.Sam.IouToken.GetParameters().ToArray());
        var clipAfter = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(model.Network.ClipLayers).Select(p => p.ToArray()).ToList();
        for (int i = 0; i < clipBefore.Count; i++) Assert.Equal(clipBefore[i], clipAfter[i]);
        Assert.Equal(samEncoderBefore, model.Network.Sam.PositionTable.GetParameters().ToArray());
    }

    [Fact]
    public void Supervision_MustMatchTheConversation()
    {
        using var model = Small();
        var turns = new[] { new GlammTurn("Can you segment the dog in this image?", "Sure, it is [SEG].") };
        Assert.Throws<ArgumentException>(() => model.TrainConversation(new GlammSample<double>(Image(), turns)));
        Assert.Throws<ArgumentException>(() => model.TrainConversation(new GlammSample<double>(Image(), turns, new[] { new[] { 0.1, 0.1, 0.5, 0.5 } }, new[] { Mask() })));
        Assert.Throws<ArgumentException>(() => model.TrainConversation(new GlammSample<double>(Image(), turns, null, new[] { Mask(10, 10) })));
        Assert.Throws<NotSupportedException>(() => model.Train(Image(), new Tensor<double>(new[] { 16, 32 })));
    }
}
