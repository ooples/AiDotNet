using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Audio.Generation;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.CodecBased;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// VALL-E 2 (Chen et al. 2024): grouped code modeling, repetition-aware sampling, the code-ID NAR, and synthesis decoded
/// by Vocos's EnCodec model. Microsoft released no code, so these check the paper's equations and algorithm directly.
/// </summary>
public class VALLE2PaperTests
{
    private static VALLE2Options Tiny(int groupSize = 2) => new()
    {
        HiddenDim = 16, NumHeads = 2, NumEncoderLayers = 1, NumDecoderLayers = 1, FeedForwardDim = 32, TextTokens = 128,
        NumCodebooks = 4, CodebookSize = 64, GroupSize = groupSize, MaxCodesPerTextToken = 2, MaxCodePositions = 512,
        LearningRate = 3e-3, WarmupSteps = 0, DropoutRate = 0.0, DecoderDim = 16, DecoderIntermediateDim = 32, DecoderLayers = 1,
        Codec = new EnCodecOptions
        {
            SampleRate = 24000, NumQuantizers = 4, CodebookSize = 64, Filters = 4, Ratios = [8, 5, 4, 2], Dimension = 8,
            ResidualKernelSizes = [3, 1], TargetBandwidthKbps = 4 * 75 * 6 / 1000.0,
        },
    };

    private static VALLE2<double> Create(VALLE2Options options) =>
        new(new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 1, outputSize: 1) { RandomSeed = 7 }, options);

    private static Tensor<double> Tone(int samples)
    {
        var audio = new Tensor<double>(new[] { samples });
        for (int i = 0; i < samples; i++)
            audio[i] = 0.4 * Math.Sin(2 * Math.PI * 180.0 * i / 24000) + 0.2 * Math.Sin(2 * Math.PI * 470.0 * i / 24000);
        return audio;
    }

    private static TtsTrainingSample<double> Sample(VALLE2<double> model, int frames = 13)
    {
        var codes = new Tensor<double>(new[] { frames, 4 });
        for (int f = 0; f < frames; f++)
            for (int q = 0; q < 4; q++) codes[f, q] = (5 * f + 11 * q) % 64;
        return new TtsTrainingSample<double> { Tokens = model.TextToTokens("hello there"), CodecTokens = codes };
    }

    private static double[] Snapshot(IEnumerable<LayerBase<double>> layers) =>
        layers.SelectMany(l => l.GetParameters().ToArray()).ToArray();

    [Theory(Timeout = 60000)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(4)]
    public async Task ArModel_PredictsOneGroupOfCodesPerStep(int groupSize)
    {
        await Task.Yield();
        using var model = Create(Tiny(groupSize));
        var text = new[] { 1, 20, 30, 40, 2 };
        var codes = Enumerable.Range(0, 3 * groupSize).Select(i => i % 64).ToArray();
        // <bos> and each of the three groups predict the next group: four steps of G codes over 64 codes and the end token.
        var logits = model.Core2ForTests.ArLogits(text, codes, false, new Random(0));
        Assert.Equal(new[] { 4, groupSize, 65 }, logits.Shape.ToArray());
        // Causality: a group's prediction does not depend on later groups.
        var earlier = model.Core2ForTests.ArLogits(text, codes.Take(groupSize).ToArray(), false, new Random(0));
        for (int s = 0; s < 2; s++)
            for (int k = 0; k < groupSize; k++)
                for (int v = 0; v < 65; v++) Assert.Equal(earlier[s, k, v], logits[s, k, v], 10);
    }

    [Fact(Timeout = 30000)]
    public async Task NucleusSampling_AtTopPZero_KeepsTheMostLikelyCode()
    {
        await Task.Yield();
        var probabilities = new[] { 0.1, 0.5, 0.15, 0.25 };
        var random = new Random(3);
        for (int i = 0; i < 50; i++) Assert.Equal(1, VallE2Core<double>.Nucleus(probabilities, 0.0, random));
        // At 0.8 the nucleus is {1, 3, 2} (0.5 + 0.25 + 0.15 ≥ 0.8): code 0 is never drawn.
        for (int i = 0; i < 200; i++) Assert.NotEqual(0, VallE2Core<double>.Nucleus(probabilities, 0.8, random));
    }

    [Fact(Timeout = 180000)]
    public async Task EachStage_TrainsItsOwnModelOnly()
    {
        await Task.Yield();
        using var model = Create(Tiny());
        var sample = Sample(model);
        foreach (var stage in new[] { VallETrainingStage.AutoRegressive, VallETrainingStage.NonAutoRegressive })
        {
            model.CurrentStage = stage;
            var own = stage == VallETrainingStage.AutoRegressive ? model.AutoRegressiveLayers : model.NonAutoRegressiveLayers;
            var other = stage == VallETrainingStage.AutoRegressive ? model.NonAutoRegressiveLayers : model.AutoRegressiveLayers;
            var ownBefore = Snapshot(own);
            var otherBefore = Snapshot(other);
            model.Train(sample);
            Assert.NotEqual(ownBefore, Snapshot(own));
            Assert.True(otherBefore.SequenceEqual(Snapshot(other)), $"{stage} changed the other model.");
        }
    }

    [Theory(Timeout = 300000)]
    [InlineData(VallETrainingStage.AutoRegressive)]
    [InlineData(VallETrainingStage.NonAutoRegressive)]
    public async Task Training_LowersTheStageObjective(VallETrainingStage stage)
    {
        await Task.Yield();
        using var model = Create(Tiny());
        model.CurrentStage = stage;
        var sample = Sample(model);
        double before = model.EvaluateTrainingObjective(sample);
        for (int i = 0; i < 30; i++) model.Train(sample);
        double after = model.EvaluateTrainingObjective(sample);
        Assert.True(after < 0.8 * before, $"{stage}: the objective did not fall ({before} -> {after}).");
    }

    [Fact(Timeout = 180000)]
    public async Task Synthesis_IsDeterministic_ReadsTheText_AndDecodesWithVocos()
    {
        await Task.Yield();
        using var model = Create(Tiny());
        var sample = Sample(model);
        for (int i = 0; i < 20; i++) model.Train(sample);
        model.Voice = model.CreateVoice(Tone(24000 / 3), "a prompt");
        var first = model.Synthesize("hello there");
        var second = model.Synthesize("hello there");
        Assert.Equal(first.ToArray(), second.ToArray());
        Assert.All(first.ToArray(), v => Assert.True(AiDotNet.Helpers.NumericalStabilityHelper.IsFinite(v)));
        // Vocos ("same" padding) gives frames · hop samples.
        Assert.True(first.Length > 0 && first.Length % 320 == 0, $"{first.Length} samples.");
        var other = model.Synthesize("a completely different sentence");
        Assert.True(other.Length != first.Length || Enumerable.Range(0, first.Length).Any(i => other[i] != first[i]),
            "Different text synthesized identical audio.");
    }

    [Fact(Timeout = 60000)]
    public async Task Configuration_NeedsAVocosBandwidth()
    {
        await Task.Yield();
        var options = Tiny();
        options.NumCodebooks = 3;
        options.Codec.NumQuantizers = 3;
        options.Codec.TargetBandwidthKbps = 3 * 75 * 6 / 1000.0;
        Assert.Throws<ArgumentException>(() => Create(options));
    }
}
