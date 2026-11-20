using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.SpeechRecognition.AlibabaASR;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Audio;

/// <summary>
/// Paraformer (Gao et al. 2022) as FunASR builds it: the AISHELL-1 recipe and the released Paraformer-large, CIF
/// integration matching the release's loop, and training on the decoder, MAE and CTC losses (#2397).
/// </summary>
public class ParaformerPaperTests
{
    private static NeuralNetworkArchitecture<double> Architecture() =>
        new(InputType.OneDimensional, NeuralNetworkTaskType.SequenceToSequence, inputSize: 8000, outputSize: 16);

    private static ParaformerOptions Tiny() => new()
    {
        EncoderDim = 16, NumEncoderLayers = 1, NumDecoderLayers = 1, NumAttentionHeads = 2, FeedForwardDim = 32, ConvKernel = 3,
        NumMels = 16, VocabSize = 16, MaxTextLength = 8, MaxAudioLengthSeconds = 1, FrequencyMasks = 0, TimeMasks = 0, Seed = 5,
    };

    private static ParaformerLargeOptions TinyLarge() => new()
    {
        NumMels = 16, EncoderDim = 16, NumEncoderLayers = 1, NumDecoderLayers = 1, NumAttentionHeads = 2, FeedForwardDim = 32,
        FsmnKernelSize = 3, VocabSize = 16, MaxTextLength = 8, MaxAudioLengthSeconds = 1, Seed = 5,
    };

    private static Tensor<double> Voice(int samples, double pitch)
    {
        var audio = new Tensor<double>(new[] { samples });
        for (int i = 0; i < samples; i++)
        {
            double t = i / 16000.0;
            audio[i] = 0.4 * Math.Sin(2 * Math.PI * pitch * t) + 0.2 * Math.Sin(2 * Math.PI * 2 * pitch * t);
        }
        return audio;
    }

    [Fact(Timeout = 60000)]
    public async Task Defaults_AreTheFunAsrConfigurations()
    {
        await Task.Yield();
        var o = new ParaformerOptions();
        // paraformer_conformer_12e_6d_2048_256.yaml
        Assert.Equal(256, o.EncoderDim);
        Assert.Equal(4, o.NumAttentionHeads);
        Assert.Equal(2048, o.FeedForwardDim);
        Assert.Equal(12, o.NumEncoderLayers);
        Assert.Equal(6, o.NumDecoderLayers);
        Assert.Equal(15, o.ConvKernel);
        Assert.Equal(0.3, o.CtcWeight);
        Assert.Equal(0.1, o.LabelSmoothing);
        Assert.Equal(0.4, o.SamplingRatio);
        Assert.Equal(0.45, o.TailThreshold);
        Assert.Equal(30000, o.WarmupSteps);
        var l = new ParaformerLargeOptions();
        // speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404 config.yaml
        Assert.Equal(512, l.EncoderDim);
        Assert.Equal(4, l.NumAttentionHeads);
        Assert.Equal(2048, l.FeedForwardDim);
        Assert.Equal(50, l.NumEncoderLayers);
        Assert.Equal(16, l.NumDecoderLayers);
        Assert.Equal(11, l.FsmnKernelSize);
        Assert.Equal(7, l.LfrM);
        Assert.Equal(6, l.LfrN);
        Assert.Equal(8404, l.VocabSize);
        Assert.Equal(0.75, l.SamplingRatio);
    }

    [Fact(Timeout = 60000)]
    public async Task Integrate_MatchesTheReleaseCifLoop()
    {
        await Task.Yield();
        using var model = new Paraformer<double>(Architecture(), Tiny());
        var network = model.Network ?? throw new InvalidOperationException("native model");
        double[] alphas = { 0.3, 0.5, 0.4, 0.9, 0.2, 0.7 };
        int t = alphas.Length, d = 3;
        var hidden = new Tensor<double>(new[] { t, d });
        for (int i = 0; i < t; i++) for (int k = 0; k < d; k++) hidden[i, k] = (i + 1) * (k + 2) * 0.1;
        // FunASR's cif(): integrate until the weight reaches 1, split the firing frame's weight.
        var expected = new System.Collections.Generic.List<double[]>();
        double integrate = 0;
        var frame = new double[d];
        for (int i = 0; i < t; i++)
        {
            double completion = 1 - integrate;
            integrate += alphas[i];
            bool fire = integrate >= 1;
            if (fire) integrate -= 1;
            double cur = fire ? completion : alphas[i], rest = alphas[i] - cur;
            for (int k = 0; k < d; k++) frame[k] += cur * hidden[i, k];
            if (fire) { expected.Add((double[])frame.Clone()); for (int k = 0; k < d; k++) frame[k] = rest * hidden[i, k]; }
        }
        var weights = new Tensor<double>(new[] { 1, t }, new Vector<double>(alphas));
        var actual = network.Integrate(hidden, weights, expected.Count);
        Assert.Equal(3, expected.Count);
        for (int n = 0; n < expected.Count; n++)
            for (int k = 0; k < d; k++) Assert.Equal(expected[n][k], actual[n, k], 12);
    }

    [Fact(Timeout = 300000)]
    public async Task Training_LowersTheRecipeLoss()
    {
        await Task.Yield();
        using var model = new Paraformer<double>(Architecture(), Tiny());
        var audio = Voice(8000, 180);
        var target = new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 3, 5, 7 }));
        var objective = (ITrainingObjectiveProvider<double>)model;
        double before = objective.EvaluateTrainingObjective(audio, target);
        // A constant rate: the recipe's 30,000-step warmup would leave it near zero for the test's steps.
        model.SetBaseTrainOptimizer(new AdamOptimizer<double, Tensor<double>, Tensor<double>>(model,
            new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>> { InitialLearningRate = 1e-3 }));
        var trace = new System.Collections.Generic.List<double>();
        for (int step = 0; step < 40; step++) { model.Train(audio, target); trace.Add(objective.EvaluateTrainingObjective(audio, target)); }
        double after = objective.EvaluateTrainingObjective(audio, target);
        Assert.True(after < 0.5 * before, $"loss {before} -> {after}: {string.Join(", ", trace.Select(x => x.ToString("F3")))}");
    }

    [Fact(Timeout = 300000)]
    public async Task Large_TrainsAndCountsTheEndToken()
    {
        await Task.Yield();
        using var model = new ParaformerLarge<double>(Architecture(), TinyLarge());
        var audio = Voice(8000, 180);
        var target = new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 3, 5, 7 }));
        var objective = (ITrainingObjectiveProvider<double>)model;
        double before = objective.EvaluateTrainingObjective(audio, target);
        // A constant rate: the recipe's 30,000-step warmup would leave it near zero for the test's steps.
        model.SetBaseTrainOptimizer(new AdamOptimizer<double, Tensor<double>, Tensor<double>>(model,
            new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>> { InitialLearningRate = 1e-3 }));
        var trace = new System.Collections.Generic.List<double>();
        for (int step = 0; step < 40; step++) { model.Train(audio, target); trace.Add(objective.EvaluateTrainingObjective(audio, target)); }
        double after = objective.EvaluateTrainingObjective(audio, target);
        Assert.True(after < 0.5 * before, $"loss {before} -> {after}: {string.Join(", ", trace.Select(x => x.ToString("F3")))}");
        // The predictor learns to count the three tokens plus the end token.
        var network = model.Network ?? throw new InvalidOperationException("native model");
        var encoder = model.Encoder ?? throw new InvalidOperationException("native model");
        var alphas = network.Alphas(encoder.Encode(audio));
        double count = 0;
        for (int i = 0; i < alphas.Length; i++) count += alphas.GetFlat(i);
        Assert.InRange(count, 3.0, 5.0);
    }
}
