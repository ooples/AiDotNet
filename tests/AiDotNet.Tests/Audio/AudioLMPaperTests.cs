using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Audio.Generation;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Audio;

/// <summary>
/// AudioLM (Borsos et al. 2022): the §IV-B configuration, the three stages' token streams of §III-C (repetitions removed,
/// coarse-then-fine flattening, fine stage independent of the semantic tokens), stage-wise training, and the generation
/// modes of §III-D. Google released no code, so these check the paper directly.
/// </summary>
public class AudioLMPaperTests
{
    private static AudioLMOptions Tiny() => new()
    {
        SemanticCodes = 16, SemanticFrameRate = 20, NumCodebooks = 3, CoarseCodebooks = 1, CodebookSize = 16,
        ModelDim = 16, NumLayers = 1, NumHeads = 2, KeyValueDim = 8, FeedForwardDim = 32, DropoutRate = 0.0,
        FineCropSeconds = 0.2, LearningRate = 3e-3, Seed = 42,
        SoundStream = new SoundStreamOptions
        {
            SampleRate = 16000, Filters = 4, Dimension = 8, Ratios = [8, 5, 5, 2], NumQuantizers = 3, CodebookSize = 16,
            TargetBandwidthKbps = 0.48,
        },
    };

    private static AudioLM<double> Create() =>
        new(new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Generative,
            inputSize: 8, outputSize: 6400) { RandomSeed = 3 }, Tiny());

    private static Tensor<double> SemanticInput(params int[] tokens)
    {
        var tensor = new Tensor<double>(new[] { tokens.Length });
        for (int i = 0; i < tokens.Length; i++) tensor[i] = tokens[i];
        return tensor;
    }

    private static Tensor<double> CodeTarget(int frames)
    {
        var codes = new Tensor<double>(new[] { frames, 3 });
        for (int f = 0; f < frames; f++)
            for (int q = 0; q < 3; q++) codes[f, q] = (3 * f + 7 * q + 1) % 16;
        return codes;
    }

    private static double[] Snapshot(IEnumerable<LayerBase<double>> layers) =>
        layers.SelectMany(l => l.GetParameters().ToArray()).ToArray();

    [Fact(Timeout = 60000)]
    public async Task Defaults_AreThePapers()
    {
        await Task.Yield();
        var o = new AudioLMOptions();
        // §IV-B: 12 layers, 16 heads, width 1024, feed-forward 4096, dropout 0.1; Q = 12 levels of 1024, Q′ = 4; K = 1024
        // semantic tokens at 25 Hz; crops of 30, 10 and 3 seconds; temperatures 0.6, 0.8, 0.6.
        Assert.Equal((12, 16, 1024, 4096, 0.1), (o.NumLayers, o.NumHeads, o.ModelDim, o.FeedForwardDim, o.DropoutRate));
        Assert.Equal((12, 4, 1024, 1024, 25), (o.NumCodebooks, o.CoarseCodebooks, o.CodebookSize, o.SemanticCodes, o.SemanticFrameRate));
        Assert.Equal((30.0, 10.0, 3.0), (o.SemanticCropSeconds, o.CoarseCropSeconds, o.FineCropSeconds));
        Assert.Equal((0.6, 0.8, 0.6), (o.SemanticTemperature, o.CoarseTemperature, o.FineTemperature));
        Assert.Equal(16000 / 320, o.SoundStream.SampleRate / o.SoundStream.Ratios.Aggregate(1, (a, b) => a * b));
    }

    [Fact(Timeout = 60000)]
    public async Task SemanticTokens_LoseConsecutiveRepetitions()
    {
        await Task.Yield();
        Assert.Equal(new[] { 4, 7, 4, 9 }, AudioLmCore<double>.Deduplicate(new[] { 4, 4, 7, 7, 7, 4, 9, 9 }));
    }

    [Fact(Timeout = 60000)]
    public async Task SemanticStageLoss_DependsOnlyOnTheDeduplicatedSequence()
    {
        await Task.Yield();
        using var model = Create();
        model.CurrentStage = AudioLMStage.Semantic;
        var once = model.EvaluateTrainingObjective(SemanticInput(1, 5, 9, 2), CodeTarget(8));
        var repeated = model.EvaluateTrainingObjective(SemanticInput(1, 1, 5, 9, 9, 9, 2), CodeTarget(14));
        Assert.Equal(once, repeated);
    }

    [Fact(Timeout = 60000)]
    public async Task FineStage_IgnoresTheSemanticTokens()
    {
        await Task.Yield();
        using var model = Create();
        model.CurrentStage = AudioLMStage.FineAcoustic;
        // §III-C: fine tokens are conditionally independent of the semantic tokens given the coarse ones.
        var a = model.EvaluateTrainingObjective(SemanticInput(1, 2, 3, 4), CodeTarget(8));
        var b = model.EvaluateTrainingObjective(SemanticInput(9, 8, 7, 6), CodeTarget(8));
        Assert.Equal(a, b);
        model.CurrentStage = AudioLMStage.CoarseAcoustic;
        Assert.NotEqual(model.EvaluateTrainingObjective(SemanticInput(1, 2, 3, 4), CodeTarget(8)),
            model.EvaluateTrainingObjective(SemanticInput(9, 8, 7, 6), CodeTarget(8)));
    }

    [Theory(Timeout = 120000)]
    [InlineData(AudioLMStage.Semantic)]
    [InlineData(AudioLMStage.CoarseAcoustic)]
    [InlineData(AudioLMStage.FineAcoustic)]
    public async Task EachStage_TrainsOnlyItsOwnTransformer(AudioLMStage stage)
    {
        await Task.Yield();
        using var model = Create();
        model.CurrentStage = stage;
        var before = Enum.GetValues(typeof(AudioLMStage)).Cast<AudioLMStage>().ToDictionary(s => s, s => Snapshot(model.StageLayers(s)));
        model.Train(SemanticInput(1, 5, 9, 2, 7, 3, 3, 8), CodeTarget(16));
        foreach (var s in before.Keys)
            Assert.Equal(s == stage, !before[s].SequenceEqual(Snapshot(model.StageLayers(s))));
    }

    [Fact(Timeout = 120000)]
    public async Task Continuation_KeepsThePromptAndAddsTheRequestedFrames()
    {
        await Task.Yield();
        using var model = Create();
        var prompt = new int[3, 4];
        for (int q = 0; q < 3; q++)
            for (int t = 0; t < 4; t++) prompt[q, t] = (5 * t + q) % 16;
        var codes = model.ContinueCodes(prompt, new[] { 3, 8 }, extensionSeconds: 0.2, seed: 1);
        Assert.Equal(3, codes.GetLength(0));
        Assert.InRange(codes.GetLength(1), 4, 4 + 8);
        for (int q = 0; q < 3; q++)
            for (int t = 0; t < codes.GetLength(1); t++)
            {
                if (t < 4) Assert.Equal(prompt[q, t], codes[q, t]);
                Assert.InRange(codes[q, t], 0, 15);
            }
        Assert.Equal(codes, model.ContinueCodes(prompt, new[] { 3, 8 }, extensionSeconds: 0.2, seed: 1));
    }

    [Fact(Timeout = 60000)]
    public async Task TextPrompts_AreRejected_BecauseAudioLMHasNoTextInput()
    {
        await Task.Yield();
        using var model = Create();
        Assert.Throws<NotSupportedException>(() => model.GenerateAudio("hello"));
        Assert.False(model.SupportsTextToAudio);
    }
}
