#nullable disable
using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.VisionLanguage.Medical;
using AiDotNet.VisionLanguage.Reasoning;
using AiDotNet.VisionLanguage.Robotics;
using AiDotNet.VisionLanguage.VideoLanguage;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.VisionLanguage;

/// <summary>
/// The documented prompt-conditioned calls of the vision-language models run in native mode: the prompt is embedded
/// through the learned prompt embedding and joined to the visual tokens on the sequence axis, and the decoder receives
/// a [tokens, DecoderDim] sequence. Before, the raw token ids were concatenated on the last axis (a rank mismatch that
/// threw), and the video, reasoning and action paths handed the decoder's attention a flat vector.
/// </summary>
public class PromptFusionIntegrationTests
{
    private const int Width = 16;

    private static NeuralNetworkArchitecture<double> Architecture(int size) => new(
        inputType: InputType.ThreeDimensional, taskType: NeuralNetworkTaskType.Regression,
        inputHeight: size, inputWidth: size, inputDepth: 3, outputSize: Width);

    private static Tensor<double> Image(int size, int seed)
    {
        var image = new Tensor<double>(new[] { 3, size, size });
        var random = RandomHelper.CreateSeededRandom(seed);
        for (int i = 0; i < image.Length; i++)
            image[i] = random.NextDouble();
        return image;
    }

    private static void AssertFinite(Tensor<double> output)
    {
        Assert.True(output.Length > 0);
        for (int i = 0; i < output.Length; i++)
            Assert.False(double.IsNaN(output[i]) || double.IsInfinity(output[i]), $"output[{i}] = {output[i]}");
    }

    private static bool Differ(Tensor<double> a, Tensor<double> b)
    {
        if (a.Length != b.Length) return true;
        for (int i = 0; i < a.Length; i++)
            if (a[i] != b[i]) return true;
        return false;
    }

    [Fact(Timeout = 120000)]
    public async Task DragonflyMed_AnswerMedicalQuestion_AppendsThePromptTokens()
    {
        await Task.Yield();
        var model = new DragonflyMed<double>(Architecture(28), new DragonflyMedOptions
        {
            ImageSize = 28, VisionDim = Width, DecoderDim = Width, NumVisionLayers = 1, NumDecoderLayers = 1,
            NumHeads = 2, VocabSize = 64, MaxSequenceLength = 16, DropoutRate = 0,
        });
        var image = Image(28, 1);

        var withoutPrompt = model.GenerateFromImage(image);
        var answer = model.AnswerMedicalQuestion(image, "Is there a lesion?");
        var other = model.AnswerMedicalQuestion(image, "Where is the fracture located?");

        AssertFinite(answer);
        Assert.Equal(Width, answer.Shape[answer.Rank - 1]);
        // The prompt adds tokens on the sequence axis, and its content reaches the output through the embedding.
        Assert.True(answer.Shape[answer.Rank - 2] > withoutPrompt.Shape[withoutPrompt.Rank - 2]);
        Assert.True(Differ(answer, other));
        Assert.Equal(model.ParameterCount, model.GetParameters().Length);
    }

    [Fact(Timeout = 120000)]
    public async Task KimiVL_ReasonWithChainOfThought_GivesTheDecoderATokenSequence()
    {
        await Task.Yield();
        var model = new KimiVL<double>(Architecture(28), new KimiVLOptions
        {
            ImageSize = 28, MaxVisualTokens = 4, VisionDim = Width, DecoderDim = Width, NumVisionLayers = 1,
            NumDecoderLayers = 1, NumHeads = 2, VocabSize = 64, MaxSequenceLength = 16, DropoutRate = 0,
        });

        var answer = model.ReasonWithChainOfThought(Image(28, 2), "How many apples are left if two are eaten?");

        AssertFinite(answer);
        Assert.True(answer.Rank >= 2);
        Assert.Equal(Width, answer.Shape[answer.Rank - 1]);
    }

    [Fact(Timeout = 120000)]
    public async Task Octo_PredictAction_RunsTheBackboneOnObservationAndTaskTokens()
    {
        await Task.Yield();
        var options = new OctoOptions
        {
            ImageSize = 28, VisionDim = Width, DecoderDim = Width, NumVisionLayers = 1, NumDecoderLayers = 1,
            NumHeads = 2, VocabSize = 64, MaxSequenceLength = 16, DropoutRate = 0,
        };
        var model = new Octo<double>(Architecture(28), options);
        var observation = Image(28, 3);

        var action = model.PredictAction(observation, "pick up the red cup");
        var other = model.PredictAction(observation, "open the top drawer");

        AssertFinite(action);
        Assert.Equal(options.ActionDimension * options.PredictionHorizon, action.Length);
        Assert.True(Differ(action, other));
    }

    [Fact(Timeout = 120000)]
    public async Task VideoLLaMA3_GenerateFromVideo_IsConditionedOnThePrompt()
    {
        await Task.Yield();
        var model = new VideoLLaMA3<double>(Architecture(32), new VideoLLaMA3Options
        {
            ImageSize = 32, VisionDim = Width, DecoderDim = Width, NumVisionLayers = 1, NumDecoderLayers = 1,
            NumHeads = 2, VocabSize = 64, MaxSequenceLength = 16, MaxFrames = 4, DropoutRate = 0,
        });
        var frames = new List<Tensor<double>> { Image(32, 4), Image(32, 5) };

        var silent = model.GenerateFromVideo(frames);
        var answer = model.GenerateFromVideo(frames, "What happens in this clip?");
        var other = model.GenerateFromVideo(frames, "Who enters the room?");

        AssertFinite(silent);
        AssertFinite(answer);
        Assert.Equal(Width, answer.Shape[answer.Rank - 1]);
        Assert.True(Differ(answer, other));
    }

    [Fact(Timeout = 120000)]
    public async Task VideoLLaMA3_GenerateFromImage_GivesTheDecoderATokenSequence()
    {
        await Task.Yield();
        var model = new VideoLLaMA3<double>(Architecture(32), new VideoLLaMA3Options
        {
            ImageSize = 32, VisionDim = Width, DecoderDim = Width, NumVisionLayers = 1, NumDecoderLayers = 1,
            NumHeads = 2, VocabSize = 64, MaxSequenceLength = 16, DropoutRate = 0,
        });

        var answer = model.GenerateFromImage(Image(32, 6), "Describe the scene.");

        AssertFinite(answer);
        Assert.Equal(Width, answer.Shape[answer.Rank - 1]);
    }
}
