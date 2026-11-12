using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.Classic;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// PortaSpeech trains and synthesizes as its paper specifies (Ren et al. 2021, with NATSpeech for unstated details):
/// a mixture-alignment linguistic encoder, a VAE generator with a VP-flow prior trained on duration + MAE + KL, and a
/// grouped-sharing Glow post-net trained on its likelihood in a second phase.
/// </summary>
/// <remarks>
/// Before this change PortaSpeech had no VAE, no flow and no word-level alignment: synthesis ran a FastSpeech-style
/// layer stack once with durations from a fixed scale.
/// </remarks>
public class PortaSpeechPaperTests
{
    private const int MelBins = 8;

    private static PortaSpeech<double> CreateModel(Action<PortaSpeechOptions>? configure = null, bool paperOptimizer = false)
    {
        var options = new PortaSpeechOptions
        {
            HiddenDim = 16, EncoderDim = 16, NumHeads = 2, NumEncoderLayers = 1, NumWordEncoderLayers = 1, FilterChannels = 32,
            DurationPredictorDropout = 0.0, GeneratorChannels = 16, ProsodyDim = 4, GeneratorEncoderLayers = 2,
            GeneratorDecoderLayers = 2, PriorFlowSteps = 2, PriorFlowLayers = 2, PriorFlowChannels = 8, PostNetChannels = 16,
            PostNetLayers = 2, NumFlowLayers = 4, PostNetShareGroupSize = 2, MelChannels = MelBins, KlStartUpdates = 0,
        };
        configure?.Invoke(options);
        return new PortaSpeech<double>(
            new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
                inputSize: 8, outputSize: MelBins) { RandomSeed = 11 }, options,
            paperOptimizer ? null : new AiDotNet.Optimizers.AdamOptimizer<double, Tensor<double>, Tensor<double>>(null));
    }

    [Theory]
    [InlineData("EncoderDim", 8)]
    [InlineData("DecoderDim", 8)]
    [InlineData("NumDecoderLayers", 3)]
    public void SharedArchitectureFields_ChangeActualNetworkParameters(string field, int value)
    {
        using var control = CreateModel();
        using var changed = CreateModel(options => typeof(PortaSpeechOptions).GetProperty(field)!.SetValue(options, value));
        Assert.NotEqual(control.GetParameters().Length, changed.GetParameters().Length);
        var mel = changed.Synthesize("ab cd");
        Assert.Equal(MelBins, mel.Shape[1]);
        Assert.All(mel.ToArray(), value => Assert.True(AiDotNet.Helpers.NumericalStabilityHelper.IsFinite(value)));
    }

    [Theory]
    [InlineData("{\"HiddenDim\":16,\"EncoderDim\":192}", 16, 192, 4)]
    [InlineData("{\"EncoderDim\":192,\"HiddenDim\":16}", 16, 192, 4)]
    [InlineData("{\"GeneratorChannels\":16,\"DecoderDim\":192}", 192, 16, 4)]
    [InlineData("{\"DecoderDim\":192,\"GeneratorChannels\":16}", 192, 16, 4)]
    [InlineData("{\"GeneratorDecoderLayers\":2,\"NumDecoderLayers\":4}", 192, 192, 2)]
    [InlineData("{\"NumDecoderLayers\":4,\"GeneratorDecoderLayers\":2}", 192, 192, 2)]
    [InlineData("{\"EncoderDim\":16,\"DecoderDim\":8,\"NumDecoderLayers\":2}", 16, 8, 2)]
    public void SavedNativeArchitecture_WinsLegacyAliasesInEitherOrder(string json, int encoder, int generator, int layers)
    {
        var options = Newtonsoft.Json.JsonConvert.DeserializeObject<PortaSpeechOptions>(json)!;
        Assert.Equal(encoder, options.HiddenDim);
        Assert.Equal(generator, options.GeneratorChannels);
        Assert.Equal(layers, options.GeneratorDecoderLayers);
        var copied = new PortaSpeechOptions(options);
        Assert.Equal(encoder, copied.EncoderDim);
        Assert.Equal(generator, copied.DecoderDim);
        Assert.Equal(layers, copied.NumDecoderLayers);
    }

    [Fact]
    public void PostNetTemperature_ChangesOnlySecondPhaseSynthesis()
    {
        using var cool = CreateModel(options => options.PostNetTemperature = .4);
        using var warm = CreateModel(options => options.PostNetTemperature = .8);
        Assert.Equal(cool.Synthesize("ab cd").ToArray(), warm.Synthesize("ab cd").ToArray());
        cool.CurrentPhase = warm.CurrentPhase = PortaSpeechTrainingPhase.PostNet;
        Assert.False(cool.Synthesize("ab cd").ToArray().SequenceEqual(warm.Synthesize("ab cd").ToArray()));
    }

    [Fact]
    public void PostNetLearningRate_ChangesActualSecondPhaseOptimizerUpdate()
    {
        using var slow = CreateModel(options => options.PostNetLearningRate = .001, paperOptimizer: true);
        using var fast = CreateModel(options => options.PostNetLearningRate = .002, paperOptimizer: true);
        Assert.Equal(slow.GetParameters().ToArray(), fast.GetParameters().ToArray());
        slow.CurrentPhase = fast.CurrentPhase = PortaSpeechTrainingPhase.PostNet;
        Assert.True(AiDotNet.Helpers.NumericalStabilityHelper.IsFinite(slow.Train(Sample(slow))));
        Assert.True(AiDotNet.Helpers.NumericalStabilityHelper.IsFinite(fast.Train(Sample(fast))));
        Assert.False(slow.GetParameters().ToArray().SequenceEqual(fast.GetParameters().ToArray()));
        slow.CurrentPhase = fast.CurrentPhase = PortaSpeechTrainingPhase.VariationalGenerator;
        Assert.Equal(slow.Synthesize("ab cd").ToArray(), fast.Synthesize("ab cd").ToArray());
    }

    // "ab cd" as characters: words [a b] [space] [c d].
    private static TtsTrainingSample<double> Sample(PortaSpeech<double> model)
    {
        var tokens = new Tensor<double>(new[] { 5 });
        for (int i = 0; i < 5; i++) tokens[i] = 10 + i * 7;
        const int frames = 16;
        var mel = new Tensor<double>(new[] { frames, MelBins });
        for (int f = 0; f < frames; f++)
            for (int c = 0; c < MelBins; c++) mel[f, c] = Math.Sin(0.4 * f + 0.3 * c);
        return new TtsTrainingSample<double>
        {
            Tokens = tokens, Mel = mel, Durations = new[] { 3, 4, 2, 3, 4 }, WordLengths = new[] { 2, 1, 2 },
        };
    }

    [Fact(Timeout = 240000)]
    public async Task GeneratorPhase_ReducesTheDurationReconstructionAndKlObjective()
    {
        await Task.Yield();
        var model = CreateModel();
        var sample = Sample(model);
        double before = model.EvaluateTrainingObjective(sample);
        for (int i = 0; i < 30; i++) model.Train(sample);
        double after = model.EvaluateTrainingObjective(sample);
        Assert.True(after < before, $"Objective did not fall ({before} -> {after}).");
        Assert.Equal(30, model.GeneratorUpdates);
    }

    [Fact(Timeout = 240000)]
    public async Task PostNetPhase_ReducesItsNegativeLogLikelihood_AndTrainsOnlyThePostNet()
    {
        await Task.Yield();
        var model = CreateModel();
        var sample = Sample(model);
        var coarseBefore = model.Synthesize("ab cd").ToVector().ToArray();
        model.CurrentPhase = PortaSpeechTrainingPhase.PostNet;
        var parameters = model.GetParameters().ToArray();
        double before = model.EvaluateTrainingObjective(sample);
        for (int i = 0; i < 20; i++) model.Train(sample);
        double after = model.EvaluateTrainingObjective(sample);
        Assert.True(after < before, $"Post-net NLL did not fall ({before} -> {after}).");

        // Only the post-net moved: the synthesis path before the post-net is unchanged.
        var trained = model.GetParameters().ToArray();
        int changed = Enumerable.Range(0, parameters.Length).Count(i => parameters[i] != trained[i]);
        Assert.True(changed > 0 && changed < parameters.Length, $"{changed} of {parameters.Length} parameters changed.");
        model.CurrentPhase = PortaSpeechTrainingPhase.VariationalGenerator;
        Assert.Equal(coarseBefore, model.Synthesize("ab cd").ToVector().ToArray());
    }

    [Fact(Timeout = 120000)]
    public async Task Synthesis_IsRepeatable_WithFramesInMultiplesOfTheStride()
    {
        await Task.Yield();
        var model = CreateModel();
        var first = model.Synthesize("ab cd");
        Assert.Equal(first.ToVector().ToArray(), model.Synthesize("ab cd").ToVector().ToArray());
        Assert.Equal(2, first.Rank);
        Assert.Equal(MelBins, first.Shape[1]);
        Assert.Equal(0, first.Shape[0] % 4);
        for (int i = 0; i < first.Length; i++) Assert.True(AiDotNet.Helpers.NumericalStabilityHelper.IsFinite(first[i]), $"mel[{i}] = {first[i]}");

        model.CurrentPhase = PortaSpeechTrainingPhase.PostNet;
        var refined = model.Synthesize("ab cd");
        Assert.Equal(first.Shape, refined.Shape);
        for (int i = 0; i < refined.Length; i++) Assert.True(AiDotNet.Helpers.NumericalStabilityHelper.IsFinite(refined[i]), $"mel[{i}] = {refined[i]}");
    }

    [Fact(Timeout = 60000)]
    public async Task TokenMelTraining_Throws_BecauseThePaperTrainsOnDurations()
    {
        await Task.Yield();
        var model = CreateModel();
        var sample = Sample(model);
        Assert.Throws<NotSupportedException>(() => model.Train(sample.Tokens, sample.Mel!));
    }

    [Fact(Timeout = 60000)]
    public async Task WordLengths_MustCoverTheTokens()
    {
        await Task.Yield();
        var model = CreateModel();
        var sample = Sample(model);
        var bad = new TtsTrainingSample<double> { Tokens = sample.Tokens, Mel = sample.Mel, Durations = sample.Durations, WordLengths = new[] { 2, 2 } };
        Assert.Throws<ArgumentException>(() => model.Train(bad));
    }
}
