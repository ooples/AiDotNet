using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.Classic;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// Tacotron trains and synthesizes as its paper specifies (Wang et al. 2017): CBHG encoder, attention GRU with
/// content-based tanh attention, residual decoder GRUs emitting r frames per step, and a post-processing CBHG predicting
/// the linear spectrogram, trained with L1 on both spectrograms.
/// </summary>
/// <remarks>
/// Before this change Tacotron had no CBHG, no attention and no post-processing net: its synthesis read encoder values
/// through a fixed formula, and threw on every call because its encoder boundary pointed past the layer stack.
/// </remarks>
public class TacotronPaperTests
{
    private const int MelBins = 8;

    private static TacotronOptions SmallOptions() => new()
    {
        SampleRate = 16000, HopSize = 200, FftSize = 64, WindowSize = 64, MelChannels = MelBins, EmbeddingDim = 16,
        PrenetSizes = new[] { 16, 8 }, PrenetDropout = 0.0, EncoderBankSize = 3, PostBankSize = 2, CbhgChannels = 8,
        PostProjectionChannels = 16, AttentionDim = 16, DecoderRnnDim = 16, OutputsPerStep = 2, MaxDecoderSteps = 6,
        LearningRate = 2e-3, GriffinLimIterations = 4,
    };

    private static Tacotron<double> CreateModel() => new(
        new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 8, outputSize: MelBins) { RandomSeed = 11 },
        SmallOptions());

    private static Tensor<double> Tokens(int count)
    {
        var tokens = new Tensor<double>(new[] { count });
        for (int i = 0; i < count; i++) tokens[i] = 10 + (i * 7) % 60;
        return tokens;
    }

    private static TtsTrainingSample<double> Sample()
    {
        const int frames = 9, bins = 64 / 2 + 1;
        var mel = new Tensor<double>(new[] { frames, MelBins });
        var linear = new Tensor<double>(new[] { frames, bins });
        for (int f = 0; f < frames; f++)
        {
            for (int c = 0; c < MelBins; c++) mel[f, c] = Math.Sin(0.4 * f + 0.3 * c);
            for (int k = 0; k < bins; k++) linear[f, k] = Math.Cos(0.3 * f + 0.1 * k) - 1.0;
        }
        return new TtsTrainingSample<double> { Tokens = Tokens(5), Mel = mel, LinearSpectrogram = linear };
    }

    [Fact(Timeout = 60000)]
    public async Task GruCell_WithZeroWeights_HalvesTheState()
    {
        await Task.Yield();
        // r = z = sigmoid(0) = 0.5 and n = tanh(0) = 0, so h' = z h + (1 - z) n = h / 2.
        var cell = new GRUCellLayer<double>(inputSize: 3, hiddenSize: 4);
        var x = new Tensor<double>(new[] { 1, 3 });
        x[0, 0] = 1; x[0, 1] = -2; x[0, 2] = 0.5;
        var h = new Tensor<double>(new[] { 1, 4 });
        for (int i = 0; i < 4; i++) h[0, i] = i - 1.5;
        cell.Forward(x, h); // materialize the input projection
        cell.SetParameters(new Vector<double>((int)cell.ParameterCount));
        var next = cell.Forward(x, h);
        for (int i = 0; i < 4; i++) Assert.Equal(h[0, i] / 2, next[0, i], 12);
    }

    [Fact(Timeout = 60000)]
    public async Task Cbhg_KeepsTheTimeAxis_AndConcatenatesBothGruDirections()
    {
        await Task.Yield();
        var cbhg = new CbhgLayer<double>(inputChannels: 6, bankSize: 4, bankChannels: 5, projections: new[] { 7, 6 },
            highwayWidth: 8, gruUnits: 3);
        foreach (int time in new[] { 1, 2, 7 })
        {
            var x = new Tensor<double>(new[] { time, 6 });
            for (int i = 0; i < x.Length; i++) x[i] = Math.Sin(0.37 * i);
            var y = cbhg.Forward(x);
            Assert.Equal(new[] { time, 6 }, y.Shape);
            for (int i = 0; i < y.Length; i++) Assert.True(double.IsFinite(y[i]));
        }
    }

    [Fact(Timeout = 180000)]
    public async Task Training_ReducesTheMelAndLinearL1()
    {
        await Task.Yield();
        var model = CreateModel();
        var sample = Sample();
        double before = model.EvaluateTrainingObjective(sample);
        for (int i = 0; i < 30; i++) model.Train(sample);
        double after = model.EvaluateTrainingObjective(sample);
        Assert.True(after < before, $"Objective did not fall ({before} -> {after}).");
    }

    [Fact(Timeout = 60000)]
    public async Task TrainingTargets_AreDerivedFromTheRecording()
    {
        await Task.Yield();
        var model = CreateModel();
        var audio = new Tensor<double>(new[] { 1600 });
        for (int n = 0; n < audio.Length; n++) audio[n] = 0.3 * Math.Sin(2 * Math.PI * 220 * n / 16000.0);
        double loss = model.EvaluateTrainingObjective(new TtsTrainingSample<double> { Tokens = Tokens(5), Audio = audio });
        Assert.True(double.IsFinite(loss) && loss > 0, $"Objective on a recording was {loss}.");
    }

    [Fact(Timeout = 120000)]
    public async Task Synthesis_DecodesTheConfiguredSteps_AndAWaveform()
    {
        await Task.Yield();
        var model = CreateModel();
        var mel = model.Synthesize("hello");
        Assert.Equal(new[] { 6 * 2, MelBins }, mel.Shape);
        for (int i = 0; i < mel.Length; i++) Assert.True(double.IsFinite(mel[i]), $"mel[{i}] = {mel[i]}");

        var linear = model.PredictLinearSpectrogram("hello");
        Assert.Equal(new[] { 12, 33 }, linear.Shape);
        var waveform = model.SynthesizeWaveform("hello");
        Assert.True(waveform.Length > 0);
        for (int i = 0; i < waveform.Length; i++) Assert.True(double.IsFinite(waveform[i]), $"wave[{i}] = {waveform[i]}");
    }

    [Fact(Timeout = 60000)]
    public async Task TokenMelTraining_Throws_BecauseThePostNetTrainsOnTheLinearSpectrogram()
    {
        await Task.Yield();
        var model = CreateModel();
        Assert.Throws<NotSupportedException>(() => model.Train(Tokens(5), Sample().Mel!));
        Assert.Throws<ArgumentException>(() => model.Train(new TtsTrainingSample<double> { Tokens = Tokens(5), Mel = Sample().Mel }));
    }
}
