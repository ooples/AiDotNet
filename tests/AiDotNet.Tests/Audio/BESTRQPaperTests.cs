using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.SpeechRecognition.Foundation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Audio;

/// <summary>
/// BEST-RQ (Chiu et al. 2022): the 0.6B configuration of section 4.1, the random-projection quantizer of section 3.1
/// (fixed, on normalized input), the masking of section 3 and the masked-prediction pre-training of section 3.2. Google
/// released no code, so these check the paper directly (#2321).
/// </summary>
public class BESTRQPaperTests
{
    private static BESTRQOptions Tiny() => new()
    {
        EncoderDim = 16, NumEncoderLayers = 1, NumAttentionHeads = 2, FeedForwardMultiplier = 2, NumMels = 16,
        CodebookSize = 32, CodeDim = 8, MaskProbability = 0.2, MaskLength = 4, VocabSize = 8, MaxAudioLengthSeconds = 1,
        Seed = 5,
    };

    private static BESTRQ<double> Create(BESTRQOptions? options = null,
        IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>>? optimizer = null) =>
        new(new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.SequenceToSequence,
            inputSize: 4000, outputSize: 8), options ?? Tiny(), optimizer);

    private static Tensor<double> Voice(int samples, double pitch, double gain = 1.0)
    {
        var audio = new Tensor<double>(new[] { samples });
        for (int i = 0; i < samples; i++)
        {
            double t = i / 16000.0;
            audio[i] = gain * (0.3 * Math.Sin(2 * Math.PI * pitch * t) + 0.15 * Math.Sin(2 * Math.PI * 3 * pitch * t) * Math.Sin(2 * Math.PI * 4 * t));
        }
        return audio;
    }

    private static BestRqNetwork<double> Network(BESTRQ<double> model) =>
        model.Network ?? throw new InvalidOperationException("native model");

    private static double[] Samples(Tensor<double> audio) => Enumerable.Range(0, audio.Length).Select(i => audio[i]).ToArray();

    [Fact(Timeout = 60000)]
    public async Task Defaults_AreThePaperZeroPointSixBModel()
    {
        await Task.Yield();
        var o = new BESTRQOptions();
        // Section 4.1.1 with Conformer XL (Zhang et al. 2020, Table 1): 24 blocks, width 1024, 8 heads, kernel 5; an
        // 8192-entry, 16-dimensional codebook over 4 stacked frames; masks of 400 ms with probability 0.01; section 3:
        // N(0, 0.1) noise; section 4.1: 80-bin log-mel at a 10 ms stride and a 1024-token fine-tuning vocabulary.
        Assert.Equal((24, 1024, 8, 5), (o.NumEncoderLayers, o.EncoderDim, o.NumAttentionHeads, o.ConvKernel));
        Assert.Equal((8192, 16, 4), (o.CodebookSize, o.CodeDim, o.StackedFrames));
        Assert.Equal((0.01, 40, 0.1), (o.MaskProbability, o.MaskLength, o.MaskNoiseStd));
        Assert.Equal((80, 160, 1024), (o.NumMels, o.HopLength, o.VocabSize));
    }

    [Fact(Timeout = 60000)]
    public async Task Encoder_ReducesTenMillisecondFramesFourTimes_AndCtcCoversTheVocabulary()
    {
        await Task.Yield();
        using var model = Create();
        var network = Network(model);
        var features = network.Features(Samples(Voice(4000, 180)));
        int frames = features.GetLength(0);
        Assert.InRange(frames, 24, 27);                                                          // 0.25 s at 10 ms
        var logits = model.Predict(Voice(4000, 180));
        Assert.Equal(new[] { network.EncodedLength(frames), 8 }, logits.Shape.ToArray());
        Assert.Equal(network.EncodedLength(frames), ((frames - 3) / 2 + 1 - 3) / 2 + 1);
    }

    [Fact(Timeout = 60000)]
    public async Task Features_AreNormalizedPerBin()
    {
        await Task.Yield();
        using var model = Create();
        var features = Network(model).Features(Samples(Voice(4000, 220)));
        int frames = features.GetLength(0);
        for (int b = 0; b < features.GetLength(1); b++)
        {
            var column = Enumerable.Range(0, frames).Select(t => features[t, b]).ToArray();
            double mean = column.Average(), std = Math.Sqrt(column.Select(v => (v - mean) * (v - mean)).Average());
            Assert.True(Math.Abs(mean) < 1e-9, $"bin {b} mean {mean}");
            Assert.True(std < 1e-9 || Math.Abs(std - 1) < 1e-9, $"bin {b} std {std}");
        }
    }

    [Fact(Timeout = 60000)]
    public async Task Quantizer_LabelsEveryFourFrames_AndIgnoresTheInputScale()
    {
        await Task.Yield();
        using var model = Create();
        var network = Network(model);
        var features = network.Features(Samples(Voice(4000, 200)));
        var labels = network.Quantize(features);
        Assert.Equal(features.GetLength(0) / 4, labels.Length);
        Assert.All(labels, l => Assert.InRange(l, 0, 31));
        Assert.True(labels.Distinct().Count() > 1, "a voiced input maps to more than one code");
        // The normalized input and Equation 1's unit-norm comparison make the labels independent of the input's gain.
        Assert.Equal(labels, network.Quantize(network.Features(Samples(Voice(4000, 200, gain: 3.0)))));
        Assert.NotEqual(labels, network.Quantize(network.Features(Samples(Voice(4000, 470)))));
    }

    [Fact(Timeout = 60000)]
    public async Task Masking_ReplacesSpansWithSmallNoise()
    {
        await Task.Yield();
        var options = Tiny();
        options.MaskProbability = 0.05;
        options.MaskLength = 10;
        options.MaxAudioLengthSeconds = 4;
        using var model = Create(options);
        var network = Network(model);
        var features = network.Features(Samples(Voice(64000, 190)));
        int frames = features.GetLength(0), bins = features.GetLength(1);
        // Each frame is covered unless none of the 10 frames before it (itself included) started a span: 1 − 0.95^10,
        // averaged over 20 draws.
        double covered = Enumerable.Range(0, 20).Average(seed =>
            network.MaskInput(features, Tensors.Helpers.RandomHelper.CreateSeededRandom(100 + seed)).Mask.Count(m => m) / (double)frames);
        double expected = 1 - Math.Pow(0.95, 10);
        Assert.InRange(covered, expected - 0.05, expected + 0.05);
        var (masked, mask) = network.MaskInput(features, Tensors.Helpers.RandomHelper.CreateSeededRandom(3));
        var noise = Enumerable.Range(0, frames).Where(t => mask[t]).SelectMany(t => Enumerable.Range(0, bins).Select(b => masked[t, b])).ToArray();
        double std = Math.Sqrt(noise.Select(v => v * v).Average());
        Assert.InRange(std, 0.08, 0.12);
        for (int t = 0; t < frames; t++)
            if (!mask[t])
                for (int b = 0; b < bins; b++) Assert.Equal(features[t, b], masked[t, b]);
    }

    [Fact(Timeout = 120000)]
    public async Task Pretraining_LowersTheMaskedPredictionLoss_AndLeavesTheQuantizerFixed()
    {
        await Task.Yield();
        var optimizer = new AdamOptimizer<double, Tensor<double>, Tensor<double>>(null,
            new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>> { InitialLearningRate = 3e-3 });
        using var model = Create(optimizer: optimizer);
        var network = Network(model);
        var audio = Voice(4000, 210);
        var labels = network.Quantize(network.Features(Samples(audio)));
        double Loss() => model.PretrainingLoss(audio, Tensors.Helpers.RandomHelper.CreateSeededRandom(11));
        double before = Loss();
        Assert.True(before > 0 && !double.IsNaN(before) && !double.IsInfinity(before), $"initial loss {before}");
        for (int step = 0; step < 40; step++) model.Pretrain(audio);
        double after = Loss();
        Assert.True(after < 0.7 * before, $"masked-prediction loss {before} -> {after}");
        Assert.Equal(labels, network.Quantize(network.Features(Samples(audio))));
    }
}
