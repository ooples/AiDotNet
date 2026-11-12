using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.SpeechRecognition.ProprietaryAPI;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Audio;

/// <summary>
/// Rev AI's Reverb ASR (Bhandari et al. 2024, section 2.2): an 18-layer joint CTC/attention Conformer whose attention
/// decoder has three left-to-right and three right-to-left layers, trained with WeNet U2++'s bidirectional loss (#2397).
/// </summary>
public class RevAIPaperTests
{
    private static NeuralNetworkArchitecture<double> Architecture() =>
        new(InputType.OneDimensional, NeuralNetworkTaskType.SequenceToSequence, inputSize: 8000, outputSize: 8);

    private static RevAIOptions Tiny() => new()
    {
        EncoderDim = 16, NumEncoderLayers = 1, NumAttentionHeads = 2, ConvKernel = 3, NumMels = 16, AttentionDecoderLayers = 1,
        ReverseDecoderLayers = 1, AttentionDecoderFeedForward = 16, VocabSize = 8, FrequencyMasks = 0, TimeMasks = 0,
        MaxAudioLengthSeconds = 1, MaxTextLength = 6, Seed = 5,
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

    [Fact(Timeout = 60000)]
    public async Task Defaults_AreSectionTwoTwo()
    {
        await Task.Yield();
        var o = new RevAIOptions();
        Assert.Equal((18, 3, 3), (o.NumEncoderLayers, o.AttentionDecoderLayers, o.ReverseDecoderLayers));
        Assert.Equal((640, 8, 2560, 15), (o.EncoderDim, o.NumAttentionHeads, o.AttentionDecoderFeedForward, o.ConvKernel));
        Assert.Equal((0.3, 0.3, 0.1), (o.CtcWeight, o.ReverseWeight, o.LabelSmoothing));
        // The copy constructor keeps the bidirectional decoder.
        var copy = new RevAIOptions(new RevAIOptions { ReverseDecoderLayers = 2, ReverseWeight = 0.4, Seed = 9 });
        Assert.Equal((2, 0.4, 9), (copy.ReverseDecoderLayers, copy.ReverseWeight, copy.Seed));
    }

    [Fact(Timeout = 120000)]
    public async Task AttentionLoss_MixesLeftToRightWithRightToLeftOverTheReversedTranscript()
    {
        await Task.Yield();
        var options = Tiny();
        options.LabelSmoothing = 0;
        using var model = new RevAI<double>(Architecture(), options);
        var network = model.Network ?? throw new InvalidOperationException("native model");
        Assert.Single(network.DecoderBlocks);
        Assert.Single(network.ReverseDecoderBlocks);
        var encoded = network.Encode(network.Features(Voice(8000, 180).ToArray()), training: false);
        var labels = new[] { 3, 5, 2 };

        // Unsmoothed, each direction's loss is the mean cross-entropy of its transcript then the end token.
        double Directional(int[] transcript, bool reverse)
        {
            var input = new[] { network.StartToken }.Concat(transcript).ToArray();
            var logits = network.DecoderLogits(input, encoded, reverse);
            double total = 0;
            for (int i = 0; i < input.Length; i++)
            {
                int target = i < transcript.Length ? transcript[i] : network.EndToken;
                double max = Enumerable.Range(0, 8).Max(v => logits[i, v]);
                total += max + Math.Log(Enumerable.Range(0, 8).Sum(v => Math.Exp(logits[i, v] - max))) - logits[i, target];
            }
            return total / input.Length;
        }

        double forward = Directional(labels, reverse: false), backward = Directional(Enumerable.Reverse(labels).ToArray(), reverse: true);
        Assert.NotEqual(forward, backward, 6);
        Assert.Equal(0.7 * forward + 0.3 * backward, network.AttentionLoss(encoded, labels)[0], 9);
    }

    [Fact(Timeout = 300000)]
    public async Task Training_LowersTheJointLoss_AndTrainsBothDecoders()
    {
        await Task.Yield();
        using var model = new RevAI<double>(Architecture(), Tiny());
        var network = model.Network ?? throw new InvalidOperationException("native model");
        var audio = Voice(8000, 180);
        var labels = new[] { 3, 5, 2 };
        var target = new Tensor<double>(new[] { 3 }, new Vector<double>(labels.Select(l => (double)l).ToArray()));
        var features = network.Features(audio.ToArray());
        var reverse = network.ReverseDecoderBlocks[0].SelfAttention.GetParameters().ToArray();
        double Loss() => network.CtcTrainingLoss(features, labels, training: false)[0];
        double before = Loss();
        for (int step = 0; step < 30; step++) model.Train(audio, target);
        Assert.True(Loss() < 0.5 * before, $"joint loss {before} -> {Loss()}");
        Assert.NotEqual(reverse, network.ReverseDecoderBlocks[0].SelfAttention.GetParameters().ToArray());
        Assert.True(model.Transcribe(audio).Text.Length <= 6);
    }
}
