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
/// SPIRAL (Huang et al. 2022): the BASE configuration of Table 1 and section 4.2, the encoder's 10 → 40 → 80 ms frame
/// rates, the convolutional CTC classifier of section 3.5, and the teacher-student pre-training of section 3.1 (#2321).
/// </summary>
public class SPIRALPaperTests
{
    private static SPIRALOptions Tiny() => new()
    {
        NumMels = 16, FirstBlockDim = 16, FirstBlockLayers = 1, FirstBlockHeads = 2, FirstBlockFeedForwardDim = 32,
        EncoderDim = 16, NumEncoderLayers = 1, NumAttentionHeads = 2, FeedForwardDim = 32, LayerDrop = 0, DropoutRate = 0,
        ProjectionDim = 8, PredictorChannels = 8, ClassifierChannels = 8, VocabSize = 8, Negatives = 6, MaxAudioLengthSeconds = 2,
        TimeMaskProbability = 0.05, TimeMaskLength = 5, FrequencyMaskProbability = 0.05, FrequencyMaskLength = 3,
        PretrainingSteps = 100, Seed = 5,
    };

    private static SPIRAL<double> Create(SPIRALOptions? options = null,
        IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>>? optimizer = null) =>
        new(new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.SequenceToSequence,
            inputSize: 16000, outputSize: 8), options ?? Tiny(), optimizer);

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

    [Fact(Timeout = 60000)]
    public async Task Defaults_AreTheBaseModel()
    {
        await Task.Yield();
        var o = new SPIRALOptions();
        // Table 1 BASE: block 1 is 2 layers of width 512, 8 heads, ffn 2048, no LayerDrop; block 2 is 10 layers of width
        // 768, 12 heads, ffn 3072, LayerDrop 0.05; a 256-wide projection and predictor. Section 4.2: 128-bin log-mel with a
        // 20 ms window and 10 ms stride, SpecAugment p = 0.025 / L = 20 in time and p = 0.02 / L = 20 in frequency, teacher
        // momentum 0.995 → 1.0. Section 3.5: a 512-channel classifier and 1024 subwords.
        Assert.Equal((2, 512, 8, 2048, 0.0), (o.FirstBlockLayers, o.FirstBlockDim, o.FirstBlockHeads, o.FirstBlockFeedForwardDim, o.FirstBlockLayerDrop));
        Assert.Equal((10, 768, 12, 3072, 0.05), (o.NumEncoderLayers, o.EncoderDim, o.NumAttentionHeads, o.FeedForwardDim, o.LayerDrop));
        Assert.Equal((256, 256, 512, 1024), (o.ProjectionDim, o.PredictorChannels, o.ClassifierChannels, o.VocabSize));
        Assert.Equal((128, 320, 160), (o.NumMels, o.WindowLength, o.HopLength));
        Assert.Equal((0.025, 20, 0.02, 20), (o.TimeMaskProbability, o.TimeMaskLength, o.FrequencyMaskProbability, o.FrequencyMaskLength));
        Assert.Equal((0.995, 1.0), (o.TeacherMomentumStart, o.TeacherMomentumEnd));
        var c = o.ToConfiguration();
        Assert.Equal(new[] { (5, 384, 2), (5, 512, 2), (1, 512, 1) }, c.FirstConvolutions);
        Assert.Equal(new[] { (5, 1536, 2), (1, 768, 1) }, c.SecondConvolutions);
    }

    [Fact(Timeout = 60000)]
    public async Task Encoder_RunsAtEightyMilliseconds_AndTheClassifierCoversTheVocabulary()
    {
        await Task.Yield();
        using var model = Create();
        var network = model.Network ?? throw new InvalidOperationException("native model");
        Assert.Equal(8, network.TotalStride);
        var audio = Voice(16000);
        var features = network.Features(Enumerable.Range(0, audio.Length).Select(i => audio[i]).ToArray());
        Assert.InRange(features.GetLength(0), 99, 102);                                           // 1 s at 10 ms
        var logits = model.Predict(audio);
        Assert.Equal(new[] { (features.GetLength(0) + 7) / 8, 8 }, logits.Shape.ToArray());     // 80 ms frames
    }

    [Fact(Timeout = 300000)]
    public async Task Pretraining_LowersTheContrastiveLoss_AndTheTeacherFollowsTheStudent()
    {
        await Task.Yield();
        var optimizer = new AdamOptimizer<double, Tensor<double>, Tensor<double>>(null,
            new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>> { InitialLearningRate = 2e-3 });
        using var model = Create(optimizer: optimizer);
        var audio = Voice(16000);
        double Loss() => model.PretrainingLoss(audio, Tensors.Helpers.RandomHelper.CreateSeededRandom(11));
        double before = Loss();
        Assert.True(before > 0 && !double.IsNaN(before) && !double.IsInfinity(before), $"initial loss {before}");
        var teacher = model.Teacher ?? throw new InvalidOperationException("the first loss builds the teacher");
        var network = model.Network ?? throw new InvalidOperationException("native model");
        // The teacher starts as the student.
        Assert.Equal(network.ProjectionHead.GetParameters().ToArray(), teacher.ProjectionHead.GetParameters().ToArray());
        for (int step = 0; step < 30; step++) model.Pretrain(audio);
        double after = Loss();
        Assert.True(after < 0.8 * before, $"contrastive loss {before} -> {after}");
        // Equation 1 with α near 0.995: the teacher moved toward the student but lags it.
        var s = network.ProjectionHead.GetParameters().ToArray();
        var t = teacher.ProjectionHead.GetParameters().ToArray();
        Assert.NotEqual(s, t);
    }
}
