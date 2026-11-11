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

    private static NonAttentiveTacotron<double> CreateModel(Action<NonAttentiveTacotronOptions>? configure = null)
    {
        var options = new NonAttentiveTacotronOptions
        {
            SampleRate = 16000, HopSize = 200, FftSize = 64, WindowSize = 64, MelChannels = MelBins, EmbeddingDim = 16,
            EncoderConvChannels = new[] { 16 }, EncoderLstmDim = 8, DurationLstmDim = 8, RangeLstmDim = 8,
            PositionalEmbeddingDim = 4, PrenetSizes = new[] { 8, 8 }, PrenetDropout = 0.0, DecoderLstmDim = 16,
            ZoneoutProbability = 0.0, PostnetDim = 8, PostnetLayers = 2, DropoutRate = 0.0, LearningRate = 2e-3,
        };
        configure?.Invoke(options);
        return new NonAttentiveTacotron<double>(
            new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
                inputSize: 8, outputSize: MelBins) { RandomSeed = 11 },
            options, new AiDotNet.Optimizers.AdamOptimizer<double, Tensor<double>, Tensor<double>>(null));
    }

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

    [Fact(Timeout = 120000)]
    public async Task NativeSynthesis_RespectsTextAndMelCaps()
    {
        await Task.Yield();
        using var model = CreateModel(o =>
        {
            o.MaxTextLength = 3;
            o.DurationLossWeight = 100;
            o.DurationScale = 8;
        });
        var sample = Sample();
        var tokens = sample.Tokens;
        var prefix = new Tensor<double>(new[] { 3 }, new Vector<double>(tokens.ToArray().Take(3).ToArray()));
        var durations = sample.Durations!.Take(3).ToArray();
        int frames = durations.Sum();
        var trainingSample = new TtsTrainingSample<double>
        {
            Tokens = prefix,
            Durations = durations,
            Mel = new Tensor<double>(new[] { frames, MelBins },
                new Vector<double>(sample.Mel!.ToArray().Take(frames * MelBins).ToArray())),
        };
        for (int step = 0; step < 150; step++) model.Train(trainingSample);
        int uncappedFrames = model.Predict(prefix).Shape[0];
        Assert.True(uncappedFrames > 2, $"The trained duration predictor yielded {uncappedFrames} frames.");
        ((NonAttentiveTacotronOptions)model.GetOptions()).MaxMelLength = 2;
        var output = model.Predict(tokens);
        Assert.Equal(new[] { 2, MelBins }, output.Shape);
        Assert.Equal(model.Predict(prefix).ToArray(), output.ToArray());
    }

    [Fact]
    public void NativeTopology_RejectsAttentionAndAdditionalDecoderLayers()
    {
        Assert.Throws<ArgumentException>("NumHeads", () => CreateModel(o => o.NumHeads = 1));
        Assert.Throws<ArgumentException>("NumDecoderLayers", () => CreateModel(o => o.NumDecoderLayers = 3));
        using var legacy = CreateModel(o =>
        {
            o.NumHeads = 8;
            o.NumDecoderLayers = 6;
        });
        Assert.True(legacy.ParameterCount > 0);
    }

    [Fact]
    public void SavedNativeDimensions_DefineTheSharedOptions()
    {
        const string json = "{\"EmbeddingDim\":16,\"EncoderLstmDim\":8,\"DecoderLstmDim\":12,\"EncoderConvChannels\":[16],"
            + "\"HiddenDim\":256,\"EncoderDim\":256,\"DecoderDim\":80,\"NumEncoderLayers\":6}";
        var options = Newtonsoft.Json.JsonConvert.DeserializeObject<NonAttentiveTacotronOptions>(json)!;
        Assert.Equal(16, options.HiddenDim);
        Assert.Equal(8, options.EncoderDim);
        Assert.Equal(12, options.DecoderDim);
        Assert.Equal(1, options.NumEncoderLayers);
    }

    [Fact]
    public void EncoderLstm_ClipsBeforeTheOutputGateAndCarriesZoneoutState()
    {
        using var layer = new LSTMLayer<double>(3) { RandomSeed = 17 };
        var input = new Tensor<double>(new[] { 1, 3, 2 }, new Vector<double>(new[] { 0.8, -0.4, 0.2, 1.1, -0.9, 0.3 }));
        var first = Frame(0);
        layer.ForwardFromState(first, null, null, out var initialHidden, out var initialCell);
        const double clip = 0.001;
        Assert.Contains(initialCell.ToArray(), value => Math.Abs(value) > clip);
        var clipped = layer.ForwardWithStateRegularization(first, clip, null);
        for (int i = 0; i < initialCell.Length; i++)
        {
            double cell = initialCell[0, i];
            double expected = Math.Abs(cell) <= clip ? initialHidden[0, i]
                : initialHidden[0, i] * Math.Tanh(Math.Sign(cell) * clip) / Math.Tanh(cell);
            Assert.Equal(expected, clipped[0, 0, i], 10);
        }

        var hidden = new Tensor<double>(new[] { 1, 3 });
        var state = new Tensor<double>(new[] { 1, 3 });
        var expectedSequence = new List<double>();
        for (int step = 0; step < 3; step++)
        {
            layer.ForwardFromState(Frame(step), hidden, state, out var nextHidden, out var nextState);
            hidden = Mix(hidden, nextHidden);
            state = Mix(state, nextState);
            expectedSequence.AddRange(hidden.ToArray());
        }
        var actual = layer.ForwardWithStateRegularization(input, 0, Mix).ToArray();
        Assert.Equal(expectedSequence.Count, actual.Length);
        for (int i = 0; i < actual.Length; i++) Assert.Equal(expectedSequence[i], actual[i], 10);

        Tensor<double> Frame(int step) =>
            new(new[] { 1, 1, 2 }, new Vector<double>(new[] { input[0, step, 0], input[0, step, 1] }));

        static Tensor<double> Mix(Tensor<double> previous, Tensor<double> next) =>
            new(next.Shape.ToArray(), new Vector<double>(previous.ToArray().Zip(next.ToArray(), (old, current) => 0.1 * old + 0.9 * current).ToArray()));
    }

    [Fact]
    public void DurationSupervisedTraining_UpdatesTheEncoderWithZoneoutEnabled()
    {
        using var model = CreateModel(o => o.ZoneoutProbability = 0.1);
        var sample = Sample();
        var embedding = Assert.Single(model.Layers.OfType<EmbeddingLayer<double>>());
        var before = embedding.GetParameters().ToArray();
        Assert.True(AiDotNet.Helpers.NumericalStabilityHelper.IsFinite(model.Train(sample)));
        Assert.False(before.SequenceEqual(embedding.GetParameters().ToArray()));
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
