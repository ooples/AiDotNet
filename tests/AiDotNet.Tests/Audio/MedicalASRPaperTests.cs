using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.SpeechRecognition.Specialized;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Audio;

/// <summary>
/// Chiu et al. 2018's recognizers for medical conversations (sections 4 and 5): the 3800 Hz filterbank input, the LAS
/// model's multi-headed attention and scheduled sampling, and the unidirectional CTC model's output delay (#2397).
/// </summary>
public class MedicalASRPaperTests
{
    private static NeuralNetworkArchitecture<double> Architecture() =>
        new(InputType.OneDimensional, NeuralNetworkTaskType.SequenceToSequence, inputSize: 8000, outputSize: 16);

    private static MedicalASROptions Tiny(Action<MedicalASROptions>? tweak = null)
    {
        var o = new MedicalASROptions
        {
            MaxAudioLengthSeconds = 1, EncoderDim = 16, NumEncoderLayers = 2, DecoderDim = 16, NumDecoderLayers = 2,
            NumAttentionHeads = 4, NumMels = 16, VocabSize = 16, MaxTextLength = 8, Seed = 5,
        };
        tweak?.Invoke(o);
        return o;
    }

    private static Tensor<double> Tone(int samples, params double[] pitches)
    {
        // Broadband noise under the voice keeps every filterbank channel well above the log floor.
        var noise = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(17);
        var audio = new Tensor<double>(new[] { samples });
        for (int i = 0; i < samples; i++)
        {
            double t = i / 16000.0;
            audio[i] = 0.3 * Math.Sin(2 * Math.PI * 180 * t) * (1 + 0.5 * Math.Sin(2 * Math.PI * 5 * t)) + 0.05 * (noise.NextDouble() - 0.5);
            // Added tones fade in and out: a tone cut off at the clip's edges is broadband there, in any STFT.
            double envelope = Math.Pow(Math.Sin(Math.PI * i / samples), 2);
            foreach (double pitch in pitches) audio[i] += 0.2 * envelope * Math.Sin(2 * Math.PI * pitch * t);
        }
        return audio;
    }

    private static Tensor<double> Labels(params int[] labels) => new(new[] { labels.Length }, new Vector<double>(labels.Select(l => (double)l).ToArray()));

    [Fact(Timeout = 60000)]
    public async Task Defaults_AreSectionsFourAndFive()
    {
        await Task.Yield();
        var las = new MedicalASROptions();
        Assert.Equal((6, 768, true), (las.NumEncoderLayers, las.EncoderDim, las.Bidirectional));
        Assert.Equal((3, 768), (las.NumDecoderLayers, las.DecoderDim));
        Assert.Equal(0.3, las.ScheduledSamplingProbability);
        Assert.Equal((80, 3800.0, 3), (las.NumMels, las.MaxFrequency, las.StackedFrames));
        var ctc = MedicalASROptions.ForCtc();
        Assert.Equal((MedicalAsrDecoderType.Ctc, 5, 350, 8001), (ctc.DecoderType, ctc.NumEncoderLayers, ctc.EncoderDim, ctc.VocabSize));
        Assert.Equal(10, ctc.OutputDelay);
    }

    [Fact(Timeout = 120000)]
    public async Task Filterbanks_IgnoreFrequenciesAbove3800Hz()
    {
        await Task.Yield();
        using var model = new MedicalASR<double>(Architecture(), Tiny());
        var network = model.Network ?? throw new InvalidOperationException("native model");
        var plain = network.Features(Tone(8000));
        var high = network.Features(Tone(8000, 6000));
        var low = network.Features(Tone(8000, 1000));
        double Difference(Tensor<double> a) => Enumerable.Range(0, a.Length).Max(i => Math.Abs(a[i] - plain[i]));
        Assert.True(Difference(high) < 1e-3, $"a 6 kHz tone moved the features by {Difference(high)}");
        Assert.True(Difference(low) > 0.1, $"a 1 kHz tone moved the features by only {Difference(low)}");
    }

    [Fact(Timeout = 120000)]
    public async Task MultiHeadedAttention_GivesEachHeadADistribution()
    {
        await Task.Yield();
        using var model = new MedicalASR<double>(Architecture(), Tiny());
        var network = model.Network ?? throw new InvalidOperationException("native model");
        var encoded = network.Encode(network.Features(Tone(8000)));
        int steps = encoded.Shape[0];
        var keys = (network.Key ?? throw new InvalidOperationException("LAS")).Forward(encoded);
        var values = (network.Value ?? throw new InvalidOperationException("LAS")).Forward(encoded);
        var state = new Tensor<double>(new[] { 1, 16 });
        for (int k = 0; k < 16; k++) state[0, k] = Math.Sin(k + 1.0);
        var (context, weights) = network.Attend(keys, values, state);
        Assert.Equal(new[] { 1, 16 }, context.Shape.ToArray());
        Assert.Equal(new[] { 4, steps }, weights.Shape.ToArray());
        for (int h = 0; h < 4; h++) Assert.Equal(1.0, Enumerable.Range(0, steps).Sum(t => weights[h, t]), 9);
        // The heads attend differently.
        Assert.Contains(Enumerable.Range(0, steps), t => Math.Abs(weights[0, t] - weights[1, t]) > 1e-9);
    }

    [Fact(Timeout = 120000)]
    public async Task ScheduledSampling_RampsLinearlyTo30Percent_AndFeedsBackThePredictions()
    {
        await Task.Yield();
        using var model = new MedicalASR<double>(Architecture(), Tiny(o => o.ScheduledSamplingRampSteps = 4));
        var audio = Tone(8000);
        var labels = Labels(3, 7, 11);
        Assert.Equal(0.0, model.CurrentSamplingProbability, 12);
        model.Train(audio, labels);
        model.Train(audio, labels);
        Assert.Equal(0.15, model.CurrentSamplingProbability, 12);
        for (int i = 0; i < 4; i++) model.Train(audio, labels);
        Assert.Equal(0.3, model.CurrentSamplingProbability, 12);

        // Sampling every input from the model is greedy decoding.
        var network = model.Network ?? throw new InvalidOperationException("native model");
        var encoded = network.Encode(network.Features(audio));
        var sampled = network.DecoderLogits(encoded, new[] { 3, 7, 11 }, 1.0, AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(1));
        var greedy = network.Greedy(encoded, 4).Logits;
        for (int i = 0; i < sampled.Length; i++) Assert.Equal(greedy[i], sampled[i], 9);
        var forced = network.DecoderLogits(encoded, new[] { 3, 7, 11 });
        Assert.Contains(Enumerable.Range(16, sampled.Length - 16), i => Math.Abs(forced[i] - sampled[i]) > 1e-9);
    }

    [Fact(Timeout = 120000)]
    public async Task UnidirectionalCtc_OutputDelayLetsEachOutputSeeLaterInput()
    {
        await Task.Yield();
        MedicalASROptions Ctc(int delay) => Tiny(o => { o.DecoderType = MedicalAsrDecoderType.Ctc; o.Bidirectional = false; o.OutputDelay = delay; });
        foreach (int delay in new[] { 0, 2 })
        {
            using var model = new MedicalASR<double>(Architecture(), Ctc(delay));
            var network = model.Network ?? throw new InvalidOperationException("native model");
            var features = network.Features(Tone(8000));
            var encoded = network.Encode(features);
            Assert.Equal(features.Shape[0], encoded.Shape[0]);
            var changed = features.Clone();
            for (int b = 0; b < features.Shape[1]; b++) changed[2, b] += 2.0;
            var other = network.Encode(changed);
            bool firstOutputMoved = Enumerable.Range(0, encoded.Shape[1]).Any(k => Math.Abs(other[0, k] - encoded[0, k]) > 1e-9);
            // Without a delay the unidirectional output at step 0 has seen only step 0; with a delay of 2 it has seen
            // step 2.
            Assert.Equal(delay >= 2, firstOutputMoved);
        }
    }

    [Theory(Timeout = 600000)]
    [InlineData(MedicalAsrDecoderType.ListenAttendSpell)]
    [InlineData(MedicalAsrDecoderType.Ctc)]
    public async Task Training_LowersTheLoss_AndTranscribes(MedicalAsrDecoderType arm)
    {
        await Task.Yield();
        using var model = new MedicalASR<double>(Architecture(), Tiny(o => o.DecoderType = arm));
        var audio = Tone(8000);
        var labels = Labels(4, 9, 12);
        var objective = (AiDotNet.Interfaces.ITrainingObjectiveProvider<double>)model;
        double before = objective.EvaluateTrainingObjective(audio, labels);
        for (int step = 0; step < 150; step++) model.Train(audio, labels);
        double after = objective.EvaluateTrainingObjective(audio, labels);
        Assert.True(after < 0.5 * before, $"{arm} loss {before} -> {after}");
        Assert.NotNull(model.Transcribe(audio).Text);
    }
}
