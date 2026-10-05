using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.EndToEnd;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// VITS2 trains and synthesizes as its paper specifies (Kong et al. 2023): the waveform networks first, with noise on the
/// alignment scores that decays to zero and a Transformer block in the flow, then the duration predictor on its own,
/// adversarially against a time-step-wise discriminator plus the log-duration MSE; no blank tokens; a speaker table that
/// also conditions the text encoder.
/// </summary>
/// <remarks>
/// Before this change VITS2 ran a stack of generic layers once and trained it by regression; it had no posterior, flow,
/// alignment search, duration model or discriminator, and its options carried an invented mixture count.
/// </remarks>
public class VITS2PaperTests
{
    private const int Vocab = 32;
    private const int Hidden = 16;

    private static VITS2Options Options(long acousticSteps = 800_000, int speakers = 0) => new()
    {
        VocabSize = Vocab, HiddenDim = Hidden, InterChannels = 8, FilterChannels = 32, NumHeads = 2, NumEncoderLayers = 3,
        DropoutRate = 0.0, PosteriorLayers = 2, FlowLayers = 2, NumFlowSteps = 2, DurationPredictorDropout = 0.0,
        DurationPredictorFilterChannels = 16, FlowTransformerDropout = 0.0,
        UpsampleRates = [4, 4], UpsampleKernelSizes = [8, 8], UpsampleInitialChannels = 16,
        ResblockKernelSizes = [3], ResblockDilationSizes = [[1, 3]], DiscriminatorPeriods = [2, 3], DiscriminatorWidthDivisor = 32,
        FftSize = 64, WindowSize = 64, HopSize = 16, SampleRate = 4000, MelChannels = 8, SegmentSize = 128,
        AcousticTrainingSteps = acousticSteps, NumSpeakers = speakers, SpeakerEmbeddingDim = 4,
    };

    private static VITS2<double> CreateModel(VITS2Options options) => new(
        new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 8, outputSize: 64) { RandomSeed = 11 },
        options,
        new AiDotNet.Optimizers.AdamOptimizer<double, Tensor<double>, Tensor<double>>(null));

    private static (Tensor<double> Tokens, Tensor<double> Audio) Example()
    {
        var tokens = new Tensor<double>(new[] { 4 });
        for (int i = 0; i < 4; i++) tokens[i] = 4 + i * 7;
        var audio = new Tensor<double>(new[] { 512 });
        for (int i = 0; i < 512; i++) audio[i] = 0.5 * Math.Sin(2 * Math.PI * (200 + i / 4.0) * i / 4000.0);
        return (tokens, audio);
    }

    [Fact(Timeout = 300000)]
    public async Task WaveformTraining_ReducesTheReconstructionAndKlObjective()
    {
        await Task.Yield();
        var model = CreateModel(Options());
        var provider = (AiDotNet.Interfaces.ITrainingObjectiveProvider<double>)model;
        var (tokens, audio) = Example();
        double before = provider.EvaluateTrainingObjective(tokens, audio);
        for (int i = 0; i < 25; i++) model.Train(tokens, audio);
        double after = provider.EvaluateTrainingObjective(tokens, audio);
        Assert.False(model.TrainsDurationPredictor);
        Assert.True(after < before, $"Objective did not fall ({before} -> {after}).");
    }

    [Fact(Timeout = 300000)]
    public async Task DurationPhase_TrainsOnlyTheDurationModel_AndReducesTheLogDurationError()
    {
        await Task.Yield();
        var model = CreateModel(Options(acousticSteps: 0));
        var provider = (AiDotNet.Interfaces.ITrainingObjectiveProvider<double>)model;
        var (tokens, audio) = Example();
        Assert.True(model.TrainsDurationPredictor);
        var start = model.GetParameters().ToArray();
        double before = provider.EvaluateTrainingObjective(tokens, audio);
        for (int i = 0; i < 30; i++) model.Train(tokens, audio);
        double after = provider.EvaluateTrainingObjective(tokens, audio);
        var end = model.GetParameters().ToArray();
        Assert.True(after < before, $"The log-duration MSE did not fall ({before} -> {after}).");

        // The text embedding (the first parameters) belongs to the waveform networks and stays fixed.
        for (int i = 0; i < Vocab * Hidden; i++) Assert.Equal(start[i], end[i]);
        Assert.Contains(Enumerable.Range(0, start.Length), i => start[i] != end[i]);
    }

    [Fact(Timeout = 60000)]
    public async Task AlignmentNoise_StartsAt001_AndFallsBy2e6PerStepToZero()
    {
        await Task.Yield();
        var probe = new Probe(Options());
        Assert.Equal(0.01, probe.Noise(0), 12);
        Assert.Equal(0.005, probe.Noise(2500), 12);
        Assert.Equal(0.0, probe.Noise(5000), 12);
        Assert.Equal(0.0, probe.Noise(1_000_000), 12);
    }

    [Fact(Timeout = 120000)]
    public async Task Synthesis_ProducesAWaveformOfWholeFrames_Repeatably()
    {
        await Task.Yield();
        var model = CreateModel(Options());
        var (tokens, _) = Example();
        var wave = model.Predict(tokens);
        Assert.Equal(0, wave.Length % 16);
        Assert.Equal(wave.ToVector().ToArray(), model.Predict(tokens).ToVector().ToArray());
    }

    [Fact(Timeout = 300000)]
    public async Task MultiSpeaker_TrainsOnNamedSpeakers_AndSpeaksInTheChosenVoice()
    {
        await Task.Yield();
        var model = CreateModel(Options(speakers: 3));
        var (tokens, audio) = Example();
        Assert.Throws<NotSupportedException>(() => model.Train(tokens, audio));
        Assert.Throws<ArgumentException>(() => model.Train(new TtsTrainingSample<double> { Tokens = tokens, Audio = audio }));
        double loss = model.Train(new TtsTrainingSample<double> { Tokens = tokens, Audio = audio, SpeakerId = 1 });
        Assert.False(double.IsNaN(loss));

        Assert.Throws<InvalidOperationException>(() => model.Predict(tokens));
        model.Voice = new TtsVoice<double> { SpeakerId = 0 };
        var first = model.Predict(tokens);
        model.Voice = new TtsVoice<double> { SpeakerId = 2 };
        var second = model.Predict(tokens);
        int shared = Math.Min(first.Length, second.Length);
        Assert.Contains(Enumerable.Range(0, shared), i => Math.Abs(first[i] - second[i]) > 1e-12);
    }

    [Fact(Timeout = 60000)]
    public async Task FlowTransformer_CarriesNoPosition_SoPermutingFramesPermutesItsOutput()
    {
        await Task.Yield();
        var block = new RelativePositionTransformerBlock<double>(4, 2, 4, 1, 0.0, 0, positional: false);
        var x = new Tensor<double>(new[] { 5, 4 });
        for (int i = 0; i < x.Length; i++) x[i] = Math.Sin(1.3 * i);
        int[] order = { 3, 0, 4, 1, 2 };
        var permuted = new Tensor<double>(new[] { 5, 4 });
        for (int t = 0; t < 5; t++)
            for (int c = 0; c < 4; c++) permuted[t, c] = x[order[t], c];
        var y = block.Forward(x);
        var yp = block.Forward(permuted);
        for (int t = 0; t < 5; t++)
            for (int c = 0; c < 4; c++) Assert.Equal(y[order[t], c], yp[t, c], 10);
    }

    private sealed class Probe : VITS2<double>
    {
        public Probe(VITS2Options options)
            : base(new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 8, outputSize: 64), options)
        {
        }

        public double Noise(long step) => AlignmentNoiseScale(step);
    }
}
