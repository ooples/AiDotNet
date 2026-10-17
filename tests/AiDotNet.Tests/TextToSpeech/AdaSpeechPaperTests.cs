using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.Classic;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// AdaSpeech trains, adapts and synthesizes as the paper specifies (Chen et al. 2021): acoustic conditions at speaker,
/// utterance and phoneme level, conditional layer normalization in the decoder, and three training phases.
/// </summary>
/// <remarks>
/// Before this change AdaSpeech had no speaker embedding, no acoustic encoders and no conditional layer normalization:
/// its synthesis averaged the first 16 encoder values into a "condition", computed durations as
/// <c>round(1 + 3|h + 0.1c|)</c>, and threw on every call because its encoder boundary pointed past the layer stack.
/// </remarks>
public class AdaSpeechPaperTests
{
    private const int MelBins = 8;

    private static AdaSpeech<double> CreateModel() => new(
        new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 8, outputSize: MelBins) { RandomSeed = 11 },
        new AdaSpeechOptions
        {
            EncoderDim = 16, HiddenDim = 16, MelChannels = MelBins, NumHeads = 2, NumEncoderLayers = 1,
            NumDecoderLayers = 1, FftFilterSize = 32, VariancePredictorFilterSize = 16, VariancePredictorDropout = 0.0,
            AcousticConditionFilterSize = 16, AcousticConditionDropout = 0.0, DropoutRate = 0.0, NumSpeakers = 3,
            LearningRate = 1e-3,
        });

    private static Tensor<double> Tokens(int count)
    {
        var tokens = new Tensor<double>(new[] { count });
        for (int i = 0; i < count; i++) tokens[i] = 10 + (i * 7) % 60;
        return tokens;
    }

    private static Tensor<double> Mel(int frames, double phase)
    {
        var mel = new Tensor<double>(new[] { frames, MelBins });
        for (int f = 0; f < frames; f++)
            for (int c = 0; c < MelBins; c++) mel[f, c] = Math.Sin(0.4 * f + 0.3 * c + phase);
        return mel;
    }

    private static TtsTrainingSample<double> Sample(int speaker, double phase = 0.0)
    {
        const int tokens = 5, frames = 14;
        return new TtsTrainingSample<double>
        {
            Tokens = Tokens(tokens),
            Mel = Mel(frames, phase),
            Durations = new[] { 3, 3, 3, 3, 2 },
            Pitch = Enumerable.Range(0, frames).Select(f => 140.0 + 30.0 * Math.Sin(0.5 * f)).ToArray(),
            Energy = Enumerable.Range(0, frames).Select(f => 5.0 + Math.Cos(0.3 * f)).ToArray(),
            SpeakerId = speaker,
        };
    }

    private static double[] Snapshot(IEnumerable<ILayer<double>> layers)
        => layers.SelectMany(l => l.GetParameters().ToArray()).ToArray();

    private static List<ILayer<double>> ConditionalProjections(AdaSpeech<double> model)
    {
        var found = new List<ILayer<double>>();
        void Walk(ILayer<double> layer)
        {
            if (layer is ConditionalLayerNormalizationLayer<double> cln)
            {
                found.Add(cln.ScaleProjection);
                found.Add(cln.BiasProjection);
            }
            else if (layer is LayerBase<double> composite)
                foreach (var child in composite.GetSubLayers()) Walk(child);
        }
        foreach (var layer in model.Layers) Walk(layer);
        return found;
    }

    [Fact(Timeout = 60000)]
    public async Task ConditionalLayerNorm_ScalesAndShiftsByTheConditionProjections()
    {
        await Task.Yield();
        var layer = new ConditionalLayerNormalizationLayer<double>(hiddenSize: 4, conditionSize: 3);
        var x = new Tensor<double>(new[] { 2, 4 });
        double[] xs = { 0.5, -1.0, 2.0, 0.25, 1.5, 0.0, -0.5, 3.0 };
        for (int i = 0; i < xs.Length; i++) x[i] = xs[i];
        var e = new Tensor<double>(new[] { 3 });
        e[0] = 0.3; e[1] = -0.7; e[2] = 1.1;

        var output = layer.Forward(x, e);

        // Eq. 1: gamma = E_s W_gamma, beta = E_s W_beta, applied to the standardized features.
        var wGamma = layer.ScaleProjection.GetParameters().ToArray();
        var wBeta = layer.BiasProjection.GetParameters().ToArray();
        for (int t = 0; t < 2; t++)
        {
            var row = Enumerable.Range(0, 4).Select(h => xs[t * 4 + h]).ToArray();
            double mean = row.Average();
            double variance = row.Select(v => (v - mean) * (v - mean)).Average();
            for (int h = 0; h < 4; h++)
            {
                double gamma = 0, beta = 0;
                for (int c = 0; c < 3; c++)
                {
                    gamma += e[c] * wGamma[c * 4 + h];
                    beta += e[c] * wBeta[c * 4 + h];
                }
                double expected = gamma * (row[h] - mean) / Math.Sqrt(variance + 1e-5) + beta;
                Assert.Equal(expected, output[t, h], 9);
            }
        }
    }

    [Fact(Timeout = 120000)]
    public async Task Pretraining_ReducesTheObjective_AndLeavesThePhonemePredictorUntouched()
    {
        await Task.Yield();
        var model = CreateModel();
        var sample = Sample(speaker: 1);
        double before = model.EvaluateTrainingObjective(sample);
        var predictorBefore = Snapshot(new ILayer<double>[] { model.PhonemePredictor! });

        for (int i = 0; i < 25; i++) model.Train(sample);

        double after = model.EvaluateTrainingObjective(sample);
        Assert.True(after < before, $"Pre-training objective did not fall ({before} -> {after}).");
        Assert.Equal(predictorBefore, Snapshot(new ILayer<double>[] { model.PhonemePredictor! }));
    }

    [Fact(Timeout = 120000)]
    public async Task JointTraining_TrainsThePhonemePredictor()
    {
        await Task.Yield();
        var model = CreateModel();
        var sample = Sample(speaker: 1);
        model.CurrentPhase = AdaSpeechTrainingPhase.SourceJoint;
        double before = model.EvaluateTrainingObjective(sample);
        var predictorBefore = Snapshot(new ILayer<double>[] { model.PhonemePredictor! });

        for (int i = 0; i < 25; i++) model.Train(sample);

        double after = model.EvaluateTrainingObjective(sample);
        Assert.True(after < before, $"Joint objective did not fall ({before} -> {after}).");
        Assert.NotEqual(predictorBefore, Snapshot(new ILayer<double>[] { model.PhonemePredictor! }));
    }

    [Fact(Timeout = 120000)]
    public async Task Adaptation_UpdatesOnlyTheSpeakerEmbeddingAndConditionalLayerNorms()
    {
        await Task.Yield();
        var model = CreateModel();
        for (int i = 0; i < 5; i++) model.Train(Sample(speaker: 0));

        model.CurrentPhase = AdaSpeechTrainingPhase.Adaptation;
        var adaptive = ConditionalProjections(model);
        Assert.Equal(2 * (2 * 1 + 1), adaptive.Count); // C = 2L + 1 conditional layer norms, two matrices each.
        adaptive.Add(model.SpeakerEmbeddingTable!);
        var adaptiveBefore = Snapshot(adaptive);
        var allBefore = model.GetParameters().ToArray();
        int adaptiveCount = adaptiveBefore.Length;

        var newVoice = Sample(speaker: 2, phase: 1.3);
        double before = model.EvaluateTrainingObjective(newVoice);
        for (int i = 0; i < 25; i++) model.Train(newVoice);
        double after = model.EvaluateTrainingObjective(newVoice);

        Assert.True(after < before, $"Adaptation objective did not fall ({before} -> {after}).");
        var allAfter = model.GetParameters().ToArray();
        int changed = allBefore.Zip(allAfter, (a, b) => a != b).Count(c => c);
        Assert.True(changed > 0, "Adaptation changed no parameter.");
        Assert.True(changed <= adaptiveCount,
            $"Adaptation changed {changed} parameters but only {adaptiveCount} belong to the speaker embedding and the conditional layer norms.");
        Assert.NotEqual(adaptiveBefore, Snapshot(adaptive));
    }

    [Fact(Timeout = 60000)]
    public async Task Synthesis_NeedsAVoice_AndFollowsIt()
    {
        await Task.Yield();
        var model = CreateModel();
        Assert.Throws<InvalidOperationException>(() => model.Synthesize("hello"));

        model.Voice = new TtsVoice<double> { SpeakerId = 0, Reference = Mel(12, 0.0) };
        var first = model.Synthesize("hello");
        Assert.Equal(2, first.Rank);
        Assert.Equal(MelBins, first.Shape[1]);
        for (int i = 0; i < first.Length; i++) Assert.True(AiDotNet.Helpers.NumericalStabilityHelper.IsFinite(first[i]), $"mel[{i}] = {first[i]}");

        model.Voice = new TtsVoice<double> { SpeakerId = 2, Reference = Mel(12, 0.0) };
        var otherSpeaker = model.Synthesize("hello");
        int n = Math.Min(first.Length, otherSpeaker.Length);
        Assert.True(Enumerable.Range(0, n).Any(i => Math.Abs(first[i] - otherSpeaker[i]) > 1e-9),
            "A different speaker produced the same spectrogram: the speaker never reaches the output.");
    }

    [Fact(Timeout = 60000)]
    public async Task TokenMelTraining_Throws_BecauseThePaperTrainsOnAlignmentsAndSpeakers()
    {
        await Task.Yield();
        var model = CreateModel();
        Assert.Throws<NotSupportedException>(() => model.Train(Tokens(5), Mel(14, 0.0)));
        Assert.Throws<ArgumentException>(() => model.Train(new TtsTrainingSample<double>
        {
            Tokens = Tokens(5), Mel = Mel(14, 0.0), Durations = new[] { 3, 3, 3, 3, 2 },
        }));
    }
}
