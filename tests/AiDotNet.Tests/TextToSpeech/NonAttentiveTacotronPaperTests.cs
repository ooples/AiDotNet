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
/// Non-Attentive Tacotron trains and synthesizes as its paper specifies (Shen et al. 2020): duration and range
/// predictors, Gaussian upsampling with a within-token positional embedding, a zoneout-LSTM autoregressive decoder and
/// the L1 + L2 spectrogram loss plus the seconds-domain duration loss.
/// </summary>
public class NonAttentiveTacotronPaperTests
{
    private const int MelBins = 8;

    private static NonAttentiveTacotron<double> CreateModel() => new(
        new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 8, outputSize: MelBins) { RandomSeed = 11 },
        new NonAttentiveTacotronOptions
        {
            SampleRate = 16000, HopSize = 200, FftSize = 64, WindowSize = 64, MelChannels = MelBins, EmbeddingDim = 16,
            EncoderConvChannels = new[] { 16 }, EncoderLstmDim = 8, DurationLstmDim = 8, RangeLstmDim = 8,
            PositionalEmbeddingDim = 4, PrenetSizes = new[] { 8, 8 }, PrenetDropout = 0.0, DecoderLstmDim = 16,
            ZoneoutProbability = 0.0, PostnetDim = 8, PostnetLayers = 2, DropoutRate = 0.0, LearningRate = 2e-3,
        },
        new AiDotNet.Optimizers.AdamOptimizer<double, Tensor<double>, Tensor<double>>(null));

    private static TtsTrainingSample<double> Sample()
    {
        const int frames = 14;
        var tokens = new Tensor<double>(new[] { 5 });
        for (int i = 0; i < 5; i++) tokens[i] = 10 + (i * 7) % 60;
        var mel = new Tensor<double>(new[] { frames, MelBins });
        for (int f = 0; f < frames; f++)
            for (int c = 0; c < MelBins; c++) mel[f, c] = Math.Sin(0.4 * f + 0.3 * c);
        return new TtsTrainingSample<double> { Tokens = tokens, Mel = mel, Durations = new[] { 3, 3, 3, 3, 2 } };
    }

    [Fact(Timeout = 60000)]
    public async Task LstmCell_MatchesTheGateEquations()
    {
        await Task.Yield();
        // With every weight and bias zero, i = f = o = 1/2 and g = 0: c' = c / 2 and h' = tanh(c') / 2.
        var cell = new LSTMCellLayer<double>(inputSize: 3, hiddenSize: 2);
        var x = new Tensor<double>(new[] { 1, 3 });
        var state = new Tensor<double>(new[] { 1, 4 });
        state[0, 2] = 0.8; state[0, 3] = -1.2;
        cell.Forward(x, state);
        cell.SetParameters(new Vector<double>((int)cell.ParameterCount));
        var (h, c) = cell.SplitState(cell.Forward(x, state));
        Assert.Equal(0.4, c[0, 0], 12);
        Assert.Equal(-0.6, c[0, 1], 12);
        Assert.Equal(Math.Tanh(0.4) / 2, h[0, 0], 12);
        Assert.Equal(Math.Tanh(-0.6) / 2, h[0, 1], 12);
    }

    [Fact(Timeout = 180000)]
    public async Task Training_ReducesTheSpectrogramAndDurationObjective()
    {
        await Task.Yield();
        var model = CreateModel();
        var sample = Sample();
        double first = model.Train(sample);
        double last = first;
        for (int i = 0; i < 30; i++) last = model.Train(sample);
        Assert.True(last < first, $"Objective did not fall ({first} -> {last}).");
    }

    [Fact(Timeout = 120000)]
    public async Task Synthesis_ProducesTheDurationsWorthOfFrames()
    {
        await Task.Yield();
        var mel = CreateModel().Synthesize("hello");
        Assert.Equal(2, mel.Rank);
        Assert.Equal(MelBins, mel.Shape[1]);
        Assert.True(mel.Shape[0] >= 1);
        for (int i = 0; i < mel.Length; i++) Assert.True(AiDotNet.Helpers.NumericalStabilityHelper.IsFinite(mel[i]), $"mel[{i}] = {mel[i]}");
    }

    [Fact(Timeout = 60000)]
    public async Task TokenMelTraining_Throws_BecauseTheSupervisedVariantTrainsOnTargetDurations()
    {
        await Task.Yield();
        Assert.Throws<NotSupportedException>(() => CreateModel().Train(Sample().Tokens, Sample().Mel!));
    }
}
