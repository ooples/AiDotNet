using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Audio.SpeechRecognition;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.SpeechRecognition.ConformerFamily;
using AiDotNet.SpeechRecognition.Streaming;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Audio;

/// <summary>
/// Gulati et al. 2020's Conformer recognizers: Table 1's sizes, the 4x convolution subsampling, and training with the CTC
/// and RNN-T objectives (#2397).
/// </summary>
public class ConformerFamilyPaperTests
{
    private static NeuralNetworkArchitecture<double> Architecture() =>
        new(InputType.OneDimensional, NeuralNetworkTaskType.SequenceToSequence, inputSize: 8000, outputSize: 8);

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
    public async Task Defaults_AreTableOnesSizes()
    {
        await Task.Yield();
        // Table 1: Conformer (M) is 16 blocks, width 256, 4 heads, kernel 32, with a 640-unit LSTM; (L) is 17, 512, 8.
        var m = new ConformerOptions();
        Assert.Equal((16, 256, 4, 32, 640, 1), (m.NumEncoderLayers, m.EncoderDim, m.NumAttentionHeads, m.ConvKernel, m.PredictionDim, m.PredictionLayers));
        var l = new ConformerTransducerOptions();
        Assert.Equal((17, 512, 8, 32), (l.NumEncoderLayers, l.EncoderDim, l.NumAttentionHeads, l.ConvKernel));
        // Section 3.1: SpecAugment with 2 frequency masks (F = 27) and 10 time masks (pS = 0.05).
        Assert.Equal((2, 27, 10, 0.05), (m.FrequencyMasks, m.FrequencyMaskWidth, m.TimeMasks, m.TimeMaskRatio));
        Assert.Equal(80, m.NumMels);
    }

    private static ConformerCTCOptions TinyCtc() => new()
    {
        EncoderDim = 16, NumEncoderLayers = 1, NumAttentionHeads = 2, ConvKernel = 3, NumMels = 16, VocabSize = 8,
        FrequencyMasks = 0, TimeMasks = 0, MaxAudioLengthSeconds = 1, Seed = 5,
    };

    [Fact(Timeout = 120000)]
    public async Task Subsampling_ReducesTheFrameRateFourTimes()
    {
        await Task.Yield();
        using var model = new ConformerCTC<double>(Architecture(), TinyCtc());
        // 0.5 s is 51 10 ms frames; two stride-2 3x3 convolutions give 12 frames of 40 ms.
        Assert.Equal(new[] { 12, 8 }, model.Predict(Voice(8000, 180)).Shape.ToArray());
    }

    [Fact(Timeout = 300000)]
    public async Task CtcTraining_LowersTheCtcLoss()
    {
        await Task.Yield();
        using var model = new ConformerCTC<double>(Architecture(), TinyCtc());
        var network = model.Network ?? throw new InvalidOperationException("native model");
        var audio = Voice(8000, 180);
        var labels = new[] { 3, 5, 2 };
        var target = new Tensor<double>(new[] { 3 }, new Vector<double>(labels.Select(l => (double)l).ToArray()));
        double Loss()
        {
            var logits = network.CtcLogits(network.Encode(network.Features(audio.ToArray()), training: false));
            var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
            return engine.TensorCTCLoss(engine.Reshape(engine.TensorLogSoftmax(logits, axis: -1), new[] { logits.Shape[0], 1, 8 }),
                new Tensor<int>(labels, new[] { 3 }), new[] { logits.Shape[0] }, new[] { 3 }, 0)[0];
        }
        double before = Loss();
        for (int step = 0; step < 30; step++) model.Train(audio, target);
        double after = Loss();
        Assert.True(after < 0.5 * before, $"CTC loss {before} -> {after}");
    }

    [Fact(Timeout = 300000)]
    public async Task TransducerTraining_LowersTheTransducerLoss_AndGreedyDecodingRuns()
    {
        await Task.Yield();
        using var model = new ConformerTransducer<double>(Architecture(), new ConformerTransducerOptions
        {
            EncoderDim = 16, NumEncoderLayers = 1, NumAttentionHeads = 2, ConvKernel = 3, NumMels = 16, PredictionDim = 16,
            JointDim = 16, VocabSize = 8, FrequencyMasks = 0, TimeMasks = 0, MaxAudioLengthSeconds = 1, Seed = 5,
        });
        var network = model.Network ?? throw new InvalidOperationException("native model");
        var audio = Voice(8000, 180);
        var labels = new[] { 3, 5, 2 };
        var target = new Tensor<double>(new[] { 3 }, new Vector<double>(labels.Select(l => (double)l).ToArray()));
        double Loss() => network.TransducerLoss(network.Encode(network.Features(audio.ToArray()), training: false), labels)[0];
        double before = Loss();
        for (int step = 0; step < 30; step++) model.Train(audio, target);
        double after = Loss();
        Assert.True(after < 0.5 * before, $"transducer loss {before} -> {after}");
        _ = model.Transcribe(audio);
    }

    [Fact(Timeout = 120000)]
    public async Task StreamingConformer_AttendsWithinChunks()
    {
        await Task.Yield();
        Assert.Equal(10, new StreamingConformerOptions().AttentionChunkSize);
        Assert.Equal(10, new StreamingConformerOptions().ToConfiguration().AttentionChunkSize);
    }
}
