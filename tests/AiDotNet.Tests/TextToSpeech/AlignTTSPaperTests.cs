using System;
using System.Collections.Generic;
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
/// AlignTTS learns its own alignment as the paper specifies (Zeng et al. 2020): a mix density network scored with the
/// Baum-Welch-style forward algorithm, Viterbi durations, and four training phases.
/// </summary>
/// <remarks>
/// Before this change AlignTTS had no mix density network and no alignment loss at all; its synthesis computed
/// durations with a fixed formula and threw on every call because its encoder boundary pointed past the layer stack.
/// </remarks>
public class AlignTTSPaperTests
{
    private static AlignTTS<double> CreateModel() => new(
        new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 8, outputSize: 16),
        new AlignTTSOptions
        {
            EncoderDim = 16, HiddenDim = 16, MelChannels = 8, NumEncoderLayers = 1, NumDecoderLayers = 1, NumHeads = 2,
            FftFilterSize = 32, DurationPredictorDim = 8, DurationPredictorLayers = 1, MixDensityHiddenSize = 16,
            DropoutRate = 0.0,
        },
        new AiDotNet.Optimizers.AdamOptimizer<double, Tensor<double>, Tensor<double>>(null));

    private static Tensor<double> Tokens(int count)
    {
        var t = new Tensor<double>(new[] { count });
        for (int i = 0; i < count; i++) t[i] = 5 + (i * 11) % 40;
        return t;
    }

    private static Tensor<double> Mel(int frames, int channels)
    {
        var mel = new Tensor<double>(new[] { frames, channels });
        for (int f = 0; f < frames; f++)
            for (int c = 0; c < channels; c++) mel[f, c] = Math.Sin(0.4 * f + 0.3 * c);
        return mel;
    }

    [Fact(Timeout = 60000)]
    public async Task MaximumPath_RecoversAPlantedAlignment()
    {
        await Task.Yield();
        int[] planted = { 2, 1, 3, 2 };
        int frames = planted.Sum();
        var scores = new double[planted.Length, frames];
        for (int s = 0; s < planted.Length; s++)
            for (int t = 0; t < frames; t++) scores[s, t] = -5.0;
        int frame = 0;
        for (int s = 0; s < planted.Length; s++)
            for (int r = 0; r < planted[s]; r++) scores[s, frame++] = 0.0;

        Assert.Equal(planted, MonotonicAlignment.MaximumPath(scores));
    }

    [Fact(Timeout = 60000)]
    public async Task LogLikelihood_EqualsTheSumOverEveryMonotonicPath()
    {
        await Task.Yield();
        const int tokens = 3, frames = 5;
        var rng = new Random(7);
        var logp = new Tensor<double>(new[] { tokens, frames });
        var values = new double[tokens, frames];
        for (int s = 0; s < tokens; s++)
            for (int t = 0; t < frames; t++) logp[s, t] = values[s, t] = -rng.NextDouble() * 3;

        // Brute force: every alignment that starts at token 0, ends at the last token, and advances by 0 or 1.
        var pathScores = new List<double>();
        void Walk(int t, int s, double acc)
        {
            acc += values[s, t];
            if (t == frames - 1) { if (s == tokens - 1) pathScores.Add(acc); return; }
            Walk(t + 1, s, acc);
            if (s + 1 < tokens) Walk(t + 1, s + 1, acc);
        }
        Walk(0, 0, 0);
        double max = pathScores.Max();
        double expected = max + Math.Log(pathScores.Sum(v => Math.Exp(v - max)));

        double actual = MonotonicAlignment.LogLikelihood(logp)[0];
        Assert.Equal(expected, actual, 9);
    }

    [Fact(Timeout = 120000)]
    public async Task EachPhase_ReducesItsObjective_AndTrainsOnlyItsParameters()
    {
        await Task.Yield();
        var model = CreateModel();
        var tokens = Tokens(5);
        var mel = Mel(14, 8);
        var provider = (AiDotNet.Interfaces.ITrainingObjectiveProvider<double>)model;

        foreach (var phase in new[] { AlignTTSTrainingPhase.Alignment, AlignTTSTrainingPhase.Decoder,
                     AlignTTSTrainingPhase.JointFineTuning, AlignTTSTrainingPhase.DurationPredictor })
        {
            model.CurrentPhase = phase;
            double before = provider.EvaluateTrainingObjective(tokens, mel);
            var encoderBefore = model.Layers[2].GetParameters().ToArray();
            for (int i = 0; i < 25; i++) model.Train(tokens, mel);
            double after = provider.EvaluateTrainingObjective(tokens, mel);
            var encoderAfter = model.Layers[2].GetParameters().ToArray();

            Assert.True(after < before, $"{phase}: objective did not fall ({before} -> {after}).");
            bool encoderMoved = !encoderBefore.SequenceEqual(encoderAfter);
            bool encoderTrains = phase is AlignTTSTrainingPhase.Alignment or AlignTTSTrainingPhase.JointFineTuning;
            Assert.True(encoderMoved == encoderTrains,
                $"{phase}: character-side block {(encoderMoved ? "moved" : "stayed")}, expected it to {(encoderTrains ? "train" : "stay frozen")}.");
        }
    }

    [Fact(Timeout = 60000)]
    public async Task Synthesis_ProducesAFiniteMelSpectrogram()
    {
        await Task.Yield();
        var mel = CreateModel().Synthesize("hello");
        Assert.Equal(2, mel.Rank);
        Assert.Equal(8, mel.Shape[1]);
        for (int i = 0; i < mel.Length; i++) Assert.True(double.IsFinite(mel[i]), $"mel[{i}] = {mel[i]}");
    }
}
