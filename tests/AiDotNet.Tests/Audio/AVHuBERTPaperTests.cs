using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.SpeechRecognition.Robust;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Audio;

/// <summary>
/// AV-HuBERT (Shi et al. 2022): the BASE configuration of section 4 and appendix B, the 25 Hz audio and lip streams, the
/// three recognition modes that modality dropout makes possible, and masked multimodal cluster prediction (#2321).
/// </summary>
public class AVHuBERTPaperTests
{
    private static AVHuBERTOptions Tiny() => new()
    {
        EncoderDim = 16, NumEncoderLayers = 1, NumAttentionHeads = 2, FeedForwardDim = 32, FinalDim = 8, NumClusters = 6,
        VocabSize = 8, DropoutRate = 0, LayerDrop = 0, LipSize = 32, MaxAudioLengthSeconds = 1, Seed = 5,
    };

    private static AVHuBERT<double> Create(IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>>? optimizer = null) =>
        new(new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.SequenceToSequence,
            inputSize: 8000, outputSize: 8), Tiny(), optimizer);

    private static Tensor<double> Voice(int samples)
    {
        var audio = new Tensor<double>(new[] { samples });
        for (int i = 0; i < samples; i++)
        {
            double f = 150 * (1 + i / 2000 * 0.37), t = i / 16000.0;
            audio[i] = 0.3 * Math.Sin(2 * Math.PI * f * t) + 0.15 * Math.Sin(2 * Math.PI * 3 * f * t);
        }
        return audio;
    }

    // A mouth that opens and closes: a dark ellipse whose height follows the frame index.
    private static Tensor<double> Lips(int frames, int size)
    {
        var lips = new Tensor<double>(new[] { frames, size, size });
        for (int f = 0; f < frames; f++)
        {
            double open = 0.15 + 0.1 * Math.Sin(f * 0.9);
            for (int y = 0; y < size; y++)
                for (int x = 0; x < size; x++)
                {
                    double dx = (x - size / 2.0) / (0.35 * size), dy = (y - size / 2.0) / (open * size);
                    lips[f, y, x] = dx * dx + dy * dy < 1 ? 0.1 : 0.8;
                }
        }
        return lips;
    }

    [Fact(Timeout = 60000)]
    public async Task Defaults_AreTheBaseModel()
    {
        await Task.Yield();
        var o = new AVHuBERTOptions();
        // Section 4: BASE is 12 layers, 768 wide, ffn 3072, 12 heads. Appendix B.4: p_m = p_a = 0.5, audio masks of 10
        // frames, video masks of 5, 2000 clusters in the last iteration, 46 CTC units. Appendix B.2: 88x88 mouth crops.
        Assert.Equal((12, 768, 3072, 12), (o.NumEncoderLayers, o.EncoderDim, o.FeedForwardDim, o.NumAttentionHeads));
        Assert.Equal((0.5, 0.5, 10, 5), (o.ModalityDropout, o.AudioDropout, o.AudioMaskLength, o.VideoMaskLength));
        Assert.Equal((2000, 46, 88), (o.NumClusters, o.VocabSize, o.LipSize));
    }

    [Fact(Timeout = 120000)]
    public async Task AudioAndLips_ShareTheTwentyFiveHertzFrameRate_AndEveryModeRecognizes()
    {
        await Task.Yield();
        using var model = Create();
        var network = model.Network ?? throw new InvalidOperationException("native model");
        var audio = Voice(8000);
        var features = network.AudioFeatures(Enumerable.Range(0, audio.Length).Select(i => audio[i]).ToArray());
        Assert.Equal(104, features.GetLength(1));
        Assert.InRange(features.GetLength(0), 12, 13);                                             // 0.5 s at 25 Hz
        var lips = Lips(features.GetLength(0), 32);
        var lipFeatures = network.EncodeVideo(lips, null, Tensors.Helpers.RandomHelper.CreateSeededRandom(1), training: false);
        Assert.Equal(new[] { features.GetLength(0), 16 }, lipFeatures.Shape.ToArray());
        Assert.Equal(features.GetLength(0), model.Predict(audio).Shape[0]);
        _ = model.TranscribeLips(lips);
        _ = model.TranscribeAudioVisual(audio, lips);
        // Equation 2: the lips change the fused features.
        var withLips = network.Fuse(network.EncodeAudio(features, null), lipFeatures, false, Tensors.Helpers.RandomHelper.CreateSeededRandom(1));
        var withoutLips = network.Fuse(network.EncodeAudio(features, null), null, false, Tensors.Helpers.RandomHelper.CreateSeededRandom(1));
        Assert.NotEqual(withLips.ToArray(), withoutLips.ToArray());
    }

    [Fact(Timeout = 300000)]
    public async Task MaskedClusterPrediction_LowersTheLoss()
    {
        await Task.Yield();
        var optimizer = new AdamOptimizer<double, Tensor<double>, Tensor<double>>(null,
            new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>> { InitialLearningRate = 3e-4 });
        using var model = Create(optimizer);
        var audio = Voice(8000);
        var lips = Lips(12, 32);
        var targets = Enumerable.Range(0, 12).Select(t => (t / 2) % 6).ToArray();
        // Measured in inference mode, so BatchNorm uses its running statistics; at 2e-3 those lag the lip encoder's
        // fast-moving weights after 30 steps (the training-mode loss still falls 16.9 -> 2.1), at 3e-4 they keep up.
        double Loss() => model.PretrainingLoss(audio, lips, targets, Tensors.Helpers.RandomHelper.CreateSeededRandom(11));
        double before = Loss();
        Assert.True(before > 0 && !double.IsNaN(before) && !double.IsInfinity(before), $"initial loss {before}");
        for (int step = 0; step < 30; step++) model.Pretrain(audio, lips, targets);
        double after = Loss();
        Assert.True(after < 0.8 * before, $"masked cluster prediction loss {before} -> {after}");
    }
}
