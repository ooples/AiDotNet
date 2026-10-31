using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.EndToEnd;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// Piper is VITS trained with Rhasspy's recipe (rhasspy/piper <c>piper_train</c>): the VITS network and objective, the
/// piper-phonemize id framing, and the x-low / medium / high model sizes.
/// </summary>
/// <remarks>
/// Before this change Piper ran a stack of generic layers once and trained it by regression at 1e-5; it had none of
/// VITS's posterior, flow, alignment search, duration model, decoder or discriminators.
/// </remarks>
public class PiperPaperTests
{
    private static PiperOptions Small() => new()
    {
        VocabSize = 32, HiddenDim = 16, InterChannels = 8, FilterChannels = 32, NumHeads = 2, NumEncoderLayers = 1,
        DropoutRate = 0.0, PosteriorLayers = 2, FlowLayers = 2, NumFlowSteps = 2, DurationPredictorDropout = 0.0,
        DurationPredictorFlows = 2, UpsampleRates = [4, 4], UpsampleKernelSizes = [8, 8], UpsampleInitialChannels = 16,
        ResblockKernelSizes = [3, 5], ResblockDilationSizes = [[1, 2], [2, 6]], DiscriminatorPeriods = [2, 3],
        DiscriminatorWidthDivisor = 32, FftSize = 64, WindowSize = 64, HopSize = 16, SampleRate = 4000, MelChannels = 8,
        SegmentSize = 128,
    };

    private static Piper<double> CreateModel(PiperOptions options) => new(
        new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 4, outputSize: 64) { RandomSeed = 11 },
        options,
        new AiDotNet.Optimizers.AdamOptimizer<double, Tensor<double>, Tensor<double>>(null));

    [Fact(Timeout = 60000)]
    public async Task Sizes_FollowPiperTrain()
    {
        await Task.Yield();
        var medium = PiperOptions.Medium();
        Assert.Equal(2, medium.ResblockType);
        Assert.Equal(new[] { 8, 8, 4 }, medium.UpsampleRates);
        Assert.Equal(256, medium.UpsampleInitialChannels);
        Assert.Equal(new[] { 16, 16, 8 }, medium.UpsampleKernelSizes);
        Assert.Equal(new[] { 3, 5, 7 }, medium.ResblockKernelSizes);
        Assert.Equal(192, medium.HiddenDim);
        Assert.Equal(512, medium.SpeakerEmbeddingDim);
        Assert.Equal(256, medium.UpsampleRates.Aggregate(1, (a, b) => a * b));

        var low = PiperOptions.XLow();
        Assert.Equal((96, 96, 384), (low.HiddenDim, low.InterChannels, low.FilterChannels));
        var high = PiperOptions.High();
        Assert.Equal(1, high.ResblockType);
        Assert.Equal(new[] { 8, 8, 2, 2 }, high.UpsampleRates);
        Assert.Equal(512, high.UpsampleInitialChannels);
    }

    [Fact(Timeout = 300000)]
    public async Task Training_ReducesTheVitsObjective()
    {
        await Task.Yield();
        var model = CreateModel(Small());
        var provider = (AiDotNet.Interfaces.ITrainingObjectiveProvider<double>)model;
        var tokens = new Tensor<double>(new[] { 3 });
        for (int i = 0; i < 3; i++) tokens[i] = 4 + i * 7;
        var audio = new Tensor<double>(new[] { 512 });
        for (int i = 0; i < 512; i++) audio[i] = 0.5 * Math.Sin(2 * Math.PI * (200 + i / 4.0) * i / 4000.0);
        double before = provider.EvaluateTrainingObjective(tokens, audio);
        for (int i = 0; i < 25; i++) model.Train(tokens, audio);
        double after = provider.EvaluateTrainingObjective(tokens, audio);
        Assert.True(after < before, $"Objective did not fall ({before} -> {after}).");
    }

    [Fact(Timeout = 120000)]
    public async Task Ids_AreFramedAsPiperPhonemizeFramesThem()
    {
        await Task.Yield();
        var probe = new Probe(Small());
        var ids = new Tensor<double>(new[] { 3 });
        ids[0] = 7; ids[1] = 9; ids[2] = 11;
        Assert.Equal(new double[] { 1, 0, 7, 0, 9, 0, 11, 0, 2 }, probe.Frame(ids).ToVector().ToArray());

        var options = Small();
        options.InterspersePad = false;
        Assert.Equal(new double[] { 1, 7, 9, 11, 2 }, new Probe(options).Frame(ids).ToVector().ToArray());
    }

    private sealed class Probe : Piper<double>
    {
        public Probe(PiperOptions options)
            : base(new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 4, outputSize: 64), options)
        {
        }

        public Tensor<double> Frame(Tensor<double> ids) => PrepareTokens(ids);
    }
}
