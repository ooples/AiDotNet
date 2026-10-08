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
/// ForwardTacotron trains and synthesizes as its reference implementation specifies (as-ideas/ForwardTacotron): series
/// predictors for duration, phoneme-level pitch and energy, a CBHG pre-net, length regulation, a bidirectional LSTM and a
/// CBHG post-net, with L1 losses on mel, post-net mel and (×0.1) the three series.
/// </summary>
/// <remarks>
/// Before this change ForwardTacotron cited Non-Attentive Tacotron while implementing neither model: its synthesis
/// derived durations with a fixed formula and threw on every call because its encoder boundary pointed past the layer
/// stack.
/// </remarks>
public class ForwardTacotronPaperTests
{
    private const int MelBins = 8;

    private static ForwardTacotron<double> CreateModel() => new(
        new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 8, outputSize: MelBins) { RandomSeed = 11 },
        new ForwardTacotronOptions
        {
            MelChannels = MelBins, EmbeddingDim = 16, SeriesEmbeddingDim = 8, DurationConvDim = 8, DurationRnnDim = 4,
            PitchConvDim = 8, PitchRnnDim = 4, EnergyConvDim = 8, EnergyRnnDim = 4, DurationDropout = 0.0, PitchDropout = 0.0,
            EnergyDropout = 0.0, PrenetDim = 8, PrenetBankSize = 3, PrenetDropout = 0.0, RnnDim = 8, PostnetChannels = 8,
            PostnetBankSize = 2, LearningRate = 2e-3,
        });

    private static TtsTrainingSample<double> Sample()
    {
        const int frames = 14;
        var tokens = new Tensor<double>(new[] { 5 });
        for (int i = 0; i < 5; i++) tokens[i] = 10 + (i * 7) % 60;
        var mel = new Tensor<double>(new[] { frames, MelBins });
        for (int f = 0; f < frames; f++)
            for (int c = 0; c < MelBins; c++) mel[f, c] = Math.Sin(0.4 * f + 0.3 * c) - 2.0;
        return new TtsTrainingSample<double>
        {
            Tokens = tokens, Mel = mel, Durations = new[] { 3, 3, 3, 3, 2 },
            Pitch = Enumerable.Range(0, frames).Select(f => f % 5 == 0 ? 0.0 : 140.0 + 10.0 * f).ToArray(),
        };
    }

    [Fact(Timeout = 180000)]
    public async Task Training_ReducesTheMelAndSeriesObjective()
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
    public async Task Synthesis_ProducesAFiniteMelSpectrogram()
    {
        await Task.Yield();
        var mel = CreateModel().Synthesize("hello");
        Assert.Equal(2, mel.Rank);
        Assert.Equal(MelBins, mel.Shape[1]);
        for (int i = 0; i < mel.Length; i++) Assert.True(AiDotNet.Helpers.NumericalStabilityHelper.IsFinite(mel[i]), $"mel[{i}] = {mel[i]}");
    }

    [Fact(Timeout = 60000)]
    public async Task TokenMelTraining_Throws_BecauseDurationsComeFromATacotronAlignment()
    {
        await Task.Yield();
        Assert.Throws<NotSupportedException>(() => CreateModel().Train(Sample().Tokens, Sample().Mel!));
    }
}
