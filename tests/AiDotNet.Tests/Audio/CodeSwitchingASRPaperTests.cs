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
/// Zeng et al. 2019's code-switching recognizer (section 4.3): the VGG2L + six-layer BLSTM encoder, the location-aware
/// LSTM decoder, the CTC layer, the LID decoder, and Equation 2's loss (#2397).
/// </summary>
public class CodeSwitchingASRPaperTests
{
    private static NeuralNetworkArchitecture<double> Architecture() =>
        new(InputType.OneDimensional, NeuralNetworkTaskType.SequenceToSequence, inputSize: 8000, outputSize: 16);

    private static CodeSwitchingASROptions Tiny() => new()
    {
        MaxAudioLengthSeconds = 1, EncoderDim = 16, NumEncoderLayers = 2, EncoderProjection = 16, NumMels = 16,
        MandarinCharVocabSize = 7, EnglishBpeVocabSize = 7, MaxTextLength = 8, DecoderDim = 16, AttentionDim = 16,
        AttentionConvChannels = 2, AttentionConvFilters = 3, Seed = 5,
    };

    private static Tensor<double> Voice(int samples, double pitch)
    {
        var audio = new Tensor<double>(new[] { samples });
        for (int i = 0; i < samples; i++)
        {
            double t = i / 16000.0;
            audio[i] = 0.3 * Math.Sin(2 * Math.PI * pitch * t) + 0.15 * Math.Sin(2 * Math.PI * 3 * pitch * t) * Math.Sin(2 * Math.PI * 4 * t);
        }
        return audio;
    }

    private static Tensor<double> Labels(params int[] labels) => new(new[] { labels.Length }, new Vector<double>(labels.Select(l => (double)l).ToArray()));

    [Fact(Timeout = 60000)]
    public async Task Defaults_AreSectionFourThree()
    {
        await Task.Yield();
        var o = new CodeSwitchingASROptions();
        Assert.Equal((320, 6, 320, 1), (o.EncoderDim, o.NumEncoderLayers, o.DecoderDim, o.NumDecoderLayers));
        Assert.Equal((0.2, 0.2), (o.CtcWeight, o.LidWeight));
        // The concatenated inventory and the boundary token: 1 + 3000 + 3000 + 1.
        Assert.Equal(6002, o.ToConfiguration().VocabSize);
    }

    [Fact(Timeout = 120000)]
    public async Task Encoder_IsVgg2lThenBidirectionalLstms()
    {
        await Task.Yield();
        using var model = new CodeSwitchingASR<double>(Architecture(), Tiny());
        var network = model.Network ?? throw new InvalidOperationException("native model");
        Assert.Equal(4, network.Convs.Count);
        Assert.Equal(2, network.EncoderLayers.Count);
        var features = network.Features(Voice(8000, 180));
        // 51 log-mel frames, then ceil(ceil(51 / 2) / 2) = 13 after the two poolings.
        Assert.Equal(51, features.Shape[0]);
        var encoded = network.Encode(features);
        Assert.Equal(new[] { 13, 16 }, encoded.Shape.ToArray());
        // Bidirectional: the first frame's encoding depends on the last frame's audio.
        var changed = features.Clone();
        for (int b = 0; b < 16; b++) changed[50, b] += 3.0;
        var other = network.Encode(changed);
        Assert.Contains(Enumerable.Range(0, 16), k => Math.Abs(other[0, k] - encoded[0, k]) > 1e-9);
    }

    [Fact(Timeout = 120000)]
    public async Task Loss_IsEquationTwo_WithLidLabelsFromTheTokenRanges()
    {
        await Task.Yield();
        using var model = new CodeSwitchingASR<double>(Architecture(), Tiny());
        var network = model.Network ?? throw new InvalidOperationException("native model");
        // Mandarin tokens are 1-7, English 8-14; the boundary is 15.
        Assert.Equal((0, 1, 15), (network.LanguageLabel(7), network.LanguageLabel(8), network.Boundary));
        var labels = new[] { 2, 9, 10, 3 };
        var features = network.Features(Voice(8000, 180));
        var (total, attention, ctc, lid) = network.Loss(features, labels);
        Assert.Equal(0.8 * attention[0] + 0.2 * ctc[0] + 0.2 * lid[0], total[0], 9);

        // The LID term is the cross-entropy of zh, en, en, zh, then the boundary.
        var (_, languages) = network.DecoderLogits(network.Encode(features), labels);
        var targets = new[] { 0, 1, 1, 0, 2 };
        double expected = 0;
        for (int i = 0; i < targets.Length; i++)
        {
            double max = Enumerable.Range(0, 3).Max(k => languages[i, k]);
            expected += max + Math.Log(Enumerable.Range(0, 3).Sum(k => Math.Exp(languages[i, k] - max))) - languages[i, targets[i]];
        }
        Assert.Equal(expected, lid[0], 9);
    }

    [Fact(Timeout = 120000)]
    public async Task LocationAwareAttention_IsANormalizedDistributionThatMoves()
    {
        await Task.Yield();
        using var model = new CodeSwitchingASR<double>(Architecture(), Tiny());
        var network = model.Network ?? throw new InvalidOperationException("native model");
        var encoded = network.Encode(network.Features(Voice(8000, 180)));
        var attention = network.Attention;
        int length = encoded.Shape[0];
        var uniform = new Tensor<double>(new[] { length });
        for (int t = 0; t < length; t++) uniform[t] = 1.0 / length;
        var projected = attention.Project(encoded);
        var state = new Tensor<double>(new[] { 1, 16 });
        var (context, weights) = attention.Attend(encoded, projected, state, uniform);
        Assert.Equal(1.0, Enumerable.Range(0, length).Sum(t => weights[t]), 9);
        Assert.Equal(new[] { 1, 16 }, context.Shape.ToArray());
        // The location term: different previous weights give different attention.
        var peaked = new Tensor<double>(new[] { length });
        peaked[0] = 1.0;
        var (_, moved) = attention.Attend(encoded, projected, state, peaked);
        Assert.Contains(Enumerable.Range(0, length), t => Math.Abs(moved[t] - weights[t]) > 1e-9);
    }

    [Fact(Timeout = 600000)]
    public async Task Training_LowersTheJointLoss_AndTheLidDecoderLearnsTheSwitch()
    {
        await Task.Yield();
        using var model = new CodeSwitchingASR<double>(Architecture(), Tiny());
        var audio = Voice(8000, 180);
        var labels = Labels(2, 9, 10, 3);
        var objective = (AiDotNet.Interfaces.ITrainingObjectiveProvider<double>)model;
        double before = objective.EvaluateTrainingObjective(audio, labels);
        for (int step = 0; step < 200; step++) model.Train(audio, labels);
        double after = objective.EvaluateTrainingObjective(audio, labels);
        Assert.True(after < 0.5 * before, $"joint loss {before} -> {after}");
        var result = model.Transcribe(audio);
        Assert.Contains(result.Language, model.SupportedLanguages);
        Assert.Equal(new[] { 13, 16 }, model.Predict(audio).Shape.ToArray());
    }
}
