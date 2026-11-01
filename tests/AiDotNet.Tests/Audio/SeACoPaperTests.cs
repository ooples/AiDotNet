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
/// SeACo-Paraformer (Shi et al. 2023) as FunASR builds it: the configuration, the paper's hotword sampling and
/// position-aware targets, the bias stage training the bias modules over a frozen backbone, and the inference merge
/// (#2397).
/// </summary>
[Collection("SeACo staged training")]
public class SeACoPaperTests
{
    private static NeuralNetworkArchitecture<double> Architecture() =>
        new(InputType.OneDimensional, NeuralNetworkTaskType.SequenceToSequence, inputSize: 8000, outputSize: 16);

    private static SeACoOptions Tiny(SeACoTrainingStage stage) => new()
    {
        NumMels = 16, EncoderDim = 16, NumEncoderLayers = 1, NumDecoderLayers = 1, NumAttentionHeads = 2, FeedForwardDim = 32,
        FsmnKernelSize = 3, NumBiasEncoderLayers = 1, NumBiasDecoderLayers = 1, BiasFeedForwardDim = 32, BiasFsmnKernelSize = 3,
        VocabSize = 16, MaxTextLength = 8, MaxAudioLengthSeconds = 1, Seed = 5, TrainingStage = stage,
        HotwordBatchRatio = 1, HotwordUtteranceRatio = 1, HotwordMinLength = 2, HotwordMaxLength = 2,
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
    public async Task Defaults_AreTheReleaseConfiguration()
    {
        await Task.Yield();
        var o = new SeACoOptions();
        // FunASR's seaco config and the paper's section 4.2.
        Assert.Equal(512, o.EncoderDim);
        Assert.Equal(50, o.NumEncoderLayers);
        Assert.Equal(16, o.NumDecoderLayers);
        Assert.Equal(2048, o.FeedForwardDim);
        Assert.Equal(11, o.FsmnKernelSize);
        Assert.Equal(8404, o.VocabSize);
        Assert.Equal(0.75, o.SamplingRatio);
        Assert.Equal(2, o.NumBiasEncoderLayers);
        Assert.Equal(4, o.NumBiasDecoderLayers);
        Assert.Equal(1024, o.BiasFeedForwardDim);
        Assert.Equal(21, o.BiasFsmnKernelSize);
        Assert.Equal(0.75, o.HotwordBatchRatio);
        Assert.Equal(0.75, o.HotwordUtteranceRatio);
        Assert.Equal(2, o.HotwordMinLength);
        Assert.Equal(8, o.HotwordMaxLength);
        Assert.Equal(1.0, o.BiasMergeLambda);
        Assert.Equal(50, o.AttentionFilterTopK);
        Assert.Equal(8404, o.ResolveHotwordMaskTokenId());
        Assert.Equal(SeACoTrainingStage.Backbone, o.TrainingStage);
    }

    [Fact(Timeout = 60000)]
    public async Task SampledHotwords_AreTranscriptSpansMarkedWhereverTheyOccur()
    {
        await Task.Yield();
        using var model = new SeACo<double>(Architecture(), Tiny(SeACoTrainingStage.Bias));
        int[][] labels = { new[] { 3, 4, 5, 6 }, new[] { 7, 8, 9 } };
        var (hotwords, positions) = model.SampleHotwords(labels, Tensors.Helpers.RandomHelper.CreateSeededRandom(11));
        // r_b = r_u = 1: one two-token span per utterance, then the default hotword.
        Assert.Equal(3, hotwords.Count);
        Assert.Equal(new[] { 1 }, hotwords[2]);
        for (int r = 0; r < 2; r++)
        {
            var hotword = hotwords[r];
            Assert.Equal(2, hotword.Length);
            int start = Enumerable.Range(0, labels[r].Length - 1).Single(s => labels[r][s] == hotword[0] && labels[r][s + 1] == hotword[1]);
            for (int i = 0; i < labels[r].Length; i++) Assert.Equal(i == start || i == start + 1, positions[r][i]);
        }
    }

    [Fact(Timeout = 300000)]
    public async Task BiasStage_LowersTheBiasLossAndLeavesTheBackbone()
    {
        await Task.Yield();
        using var model = new SeACo<double>(Architecture(), Tiny(SeACoTrainingStage.Bias));
        var audio = Voice(8000, 180);
        var target = new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 3, 5, 7 }));
        var objective = (ITrainingObjectiveProvider<double>)model;
        double before = objective.EvaluateTrainingObjective(audio, target);
        var parameters = model.GetParameters().ToArray();
        int backbone = checked((int)model.BackboneParameterCount);
        Assert.InRange(backbone, 1, parameters.Length - 1);
        // A constant rate: the recipe's 30,000-step warmup would leave it near zero for the test's steps.
        model.SetBaseTrainOptimizer(new AdamOptimizer<double, Tensor<double>, Tensor<double>>(model,
            new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>> { InitialLearningRate = 1e-2 }));
        var trace = new System.Collections.Generic.List<double>();
        for (int step = 0; step < 40; step++) { model.Train(audio, target); trace.Add(objective.EvaluateTrainingObjective(audio, target)); }
        double after = objective.EvaluateTrainingObjective(audio, target);
        Assert.True(after < 0.5 * before, $"loss {before} -> {after}: {string.Join(", ", trace.Select(x => x.ToString("F3")))}");
        var trained = model.GetParameters().ToArray();
        for (int i = 0; i < backbone; i++) Assert.Equal(parameters[i], trained[i]);
    }

    [Fact(Timeout = 60000)]
    public async Task Merge_KeepsTheAsrOutputUnlessAHotwordIsDetected()
    {
        await Task.Yield();
        var options = Tiny(SeACoTrainingStage.Backbone);
        options.VocabSize = 4;
        options.BiasMergeLambda = 0.5;
        using var model = new SeACo<double>(new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
            NeuralNetworkTaskType.SequenceToSequence, inputSize: 8000, outputSize: 4), options);
        var asr = new Tensor<double>(new[] { 3, 4 }, new Vector<double>(new double[]
        {
            0.1, 0.6, 0.2, 0.1,
            0.1, 0.6, 0.2, 0.1,
            0.1, 0.4, 0.35, 0.15,
        }));
        // Columns: the four tokens, then '#'.
        var biased = new Tensor<double>(new[] { 3, 5 }, new Vector<double>(new double[]
        {
            0.0, 0.1, 0.3, 0.0, 0.6,                       // '#': the ASR token (1)
            0.0, 0.0, 0.0, 0.9, 0.1,                       // token 3: 0.5 * 0.9 + 0.5 * 0.1 = 0.5 beats 0.3 for token 1
            0.0, 0.0, 0.5, 0.0, 0.0,                       // token 2: 0.425 beats 0.2 for token 1
        }));
        Assert.Equal(new[] { 1, 3, 2 }, model.Merge(asr, biased));
    }

    [Fact(Timeout = 300000)]
    public async Task Transcribe_WithHotwords_RunsTheBiasModules()
    {
        await Task.Yield();
        using var model = new SeACo<double>(Architecture(), Tiny(SeACoTrainingStage.Backbone));
        var audio = Voice(8000, 180);
        Assert.NotNull(model.Transcribe(audio).Text);
        var biased = model.Transcribe(audio, new[] { "", "" });
        Assert.NotNull(biased.Text);
        var bias = model.Bias ?? throw new InvalidOperationException("native model");
        Assert.True(bias.Output.ParameterCount > 0);
    }
}
