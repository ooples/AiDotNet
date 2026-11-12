using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.SpeechRecognition.NeMo;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Audio;

/// <summary>
/// Nemotron Speech Streaming (the release's <c>model_config.yaml</c>; Noroozi et al. 2024): the released configuration,
/// the causal encoder (causal subsampling, causal convolutions, chunked attention with limited left context), the
/// release's tokenizer, the Noam schedule with its factor, and the transducer's training (#2397).
/// </summary>
public class NemotronSpeechPaperTests
{
    private static NeuralNetworkArchitecture<double> Architecture() =>
        new(InputType.OneDimensional, NeuralNetworkTaskType.SequenceToSequence, inputSize: 16000, outputSize: 16);

    private static NemotronSpeechOptions Tiny() => new()
    {
        NumMels = 16, EncoderDim = 16, NumEncoderLayers = 2, NumAttentionHeads = 2, ConvKernel = 3, SubsamplingChannels = 4,
        LookaheadFrames = 1, LeftContextFrames = 2, PredictionDim = 16, PredictionLayers = 1, JointDim = 16, VocabSize = 16,
        FrequencyMasks = 0, TimeMasks = 0, Seed = 5,
    };

    private static double[] Voice(int samples, double pitch)
    {
        var audio = new double[samples];
        for (int i = 0; i < samples; i++)
        {
            double t = i / 16000.0;
            audio[i] = 0.4 * Math.Sin(2 * Math.PI * pitch * t) + 0.2 * Math.Sin(2 * Math.PI * 2 * pitch * t);
        }
        return audio;
    }

    private static Tensor<double> Waveform(double[] samples) => new(new[] { samples.Length }, new Vector<double>(samples));

    private static Tensor<double> Transcript(params double[] ids) => new(new[] { ids.Length }, new Vector<double>(ids));

    [Fact(Timeout = 60000)]
    public async Task Defaults_AreTheRelease()
    {
        await Task.Yield();
        var o = new NemotronSpeechOptions();
        Assert.Equal((128, 1024, 24, 8, 4, 9, 256), (o.NumMels, o.EncoderDim, o.NumEncoderLayers, o.NumAttentionHeads, o.FeedForwardMultiplier,
            o.ConvKernel, o.SubsamplingChannels));
        Assert.Equal((70, 13, 640, 2, 640, 1025, 0.005), (o.LeftContextFrames, o.LookaheadFrames, o.PredictionDim, o.PredictionLayers, o.JointDim,
            o.VocabSize, o.FastEmitLambda));
        Assert.Equal((2, 27, 10, 0.05), (o.FrequencyMasks, o.FrequencyMaskWidth, o.TimeMasks, o.TimeMaskRatio));
        Assert.Equal((8000, 3.0, 1e-3), (o.WarmupSteps, o.NoamFactor, o.WeightDecay));
        // [70, 13]: 14-frame chunks (1.12 s) and the five chunks (70 frames) before each.
        var configuration = o.ToConfiguration();
        Assert.Equal((14, 5, true, true), (configuration.AttentionChunkSize, configuration.AttentionLeftChunks, configuration.CausalConvolution,
            configuration.CausalDownsampling));
    }

    [Fact(Timeout = 60000)]
    public async Task Tokenizer_IsTheRelease()
    {
        await Task.Yield();
        var tokenizer = NemotronSpeech<double>.Tokenizer;
        Assert.Equal(1024, tokenizer.Count);
        // Ids from the release's tokenizer.model through Python's sentencepiece.
        Assert.Equal(new[] { 77, 32, 944, 178, 79 }, tokenizer.Encode("hello world"));
        Assert.Equal(new[] { 126, 246, 335, 362, 279, 361, 971, 128, 191, 597, 8, 70, 5, 509, 990, 956, 105, 957, 962 },
            tokenizer.Encode("The quick brown fox jumps over the lazy dog."));
        Assert.Equal(new[] { 208, 966, 948, 941, 994, 980, 994, 1005, 961, 57, 946, 966, 943, 56, 977 }, tokenizer.Encode("It's 2026, isn't it?"));
        Assert.Equal("hello world", tokenizer.Decode(tokenizer.Encode("hello world")));
    }

    [Fact(Timeout = 120000)]
    public async Task Subsampling_IsCausalAt80Milliseconds()
    {
        await Task.Yield();
        using var model = new NemotronSpeech<double>(Architecture(), Tiny());
        var network = model.Network ?? throw new InvalidOperationException("native model");
        var features = network.Features(Voice(16000, 180));
        int frames = features.GetLength(0);
        // Each causal stride-2 stage pads two frames before and one after: n / 2 + 1 frames.
        int expected = frames;
        for (int i = 0; i < 3; i++) expected = expected / 2 + 1;
        Assert.Equal(expected, network.EncodedLength(frames));
        Assert.Equal(new[] { expected, 16 }, network.Encode(features, training: false).Shape.ToArray());
    }

    [Fact(Timeout = 120000)]
    public async Task Encoder_NeverSeesPastItsLookahead()
    {
        await Task.Yield();
        // No lookahead: one-frame chunks, so every stage is causal and the features are not normalized over the utterance.
        var options = Tiny();
        options.LookaheadFrames = 0;
        using var model = new NemotronSpeech<double>(Architecture(), options);
        var network = model.Network ?? throw new InvalidOperationException("native model");
        var clean = Voice(16000, 180);
        var changed = (double[])clean.Clone();
        for (int i = 12000; i < changed.Length; i++) changed[i] += 0.3 * Math.Sin(i * 0.37);
        var a = network.Encode(network.Features(clean), training: false);
        var b = network.Encode(network.Features(changed), training: false);
        // Samples from 0.75 s reach log-mel frames from 73 on (25 ms windows every 10 ms); encoder frame t reads log-mel
        // frames up to 8t, so frames 0 to 9 cannot see the change.
        for (int t = 0; t < 10; t++)
            for (int j = 0; j < 16; j++) Assert.Equal(a[t, j], b[t, j], 12);
        int last = a.Shape[0] - 1;
        Assert.Contains(Enumerable.Range(0, 16), j => Math.Abs(a[last, j] - b[last, j]) > 1e-9);
    }

    [Fact(Timeout = 60000)]
    public async Task Attention_IsChunkedWithLimitedLeftContext()
    {
        await Task.Yield();
        // Two-frame chunks, one chunk of left context, a kernel-1 convolution so attention alone carries context.
        var block = new ConformerBlockLayer<double>(8, 2, 2, 1, attentionChunkSize: 2, attentionLeftChunks: 1, causalConvolution: true);
        block.SetTrainingMode(false);
        var random = Tensors.Helpers.RandomHelper.CreateSeededRandom(11);
        var input = new Tensor<double>(new[] { 1, 6, 8 });
        for (int i = 0; i < input.Length; i++) input.SetFlat(i, random.NextDouble() - 0.5);
        var baseline = block.Forward(input);
        Tensor<double> Perturbed(int row)
        {
            var x = input.Clone();
            // Not a constant shift, which the pre-attention LayerNorm would remove.
            for (int j = 0; j < 8; j++) x[0, row, j] += j % 2 == 0 ? 1.0 : -0.5 * j;
            return block.Forward(x);
        }
        bool Moved(Tensor<double> y, int row) => Enumerable.Range(0, 8).Any(j => Math.Abs(y[0, row, j] - baseline[0, row, j]) > 1e-9);
        // Frame 0 (chunk 0) reaches chunks 0 and 1, not chunk 2 beyond the left context.
        var first = Perturbed(0);
        Assert.Equal(new[] { 0, 1, 2, 3 }, Enumerable.Range(0, 6).Where(r => Moved(first, r)).ToArray());
        // Frame 3 (chunk 1) reaches chunk 1 and chunk 2, never chunk 0: no frame reads a later chunk.
        var middle = Perturbed(3);
        Assert.Equal(new[] { 2, 3, 4, 5 }, Enumerable.Range(0, 6).Where(r => Moved(middle, r)).ToArray());
    }

    [Fact(Timeout = 60000)]
    public async Task Optimizer_IsAdamWUnderNoamWithTheFactor()
    {
        await Task.Yield();
        using var model = new NemotronSpeech<double>(Architecture(), Tiny());
        var reports = PaperOptimizerFactory.ReportsFor(model).ToList();
        var report = Assert.Single(reports);
        Assert.Equal(OptimizerKind.AdamW, report.PaperOptimizer);
        Assert.Contains("AdamWOptimizer", report.AppliedOptimizer);
        Assert.True(report.Fidelity == RecipeFidelity.Exact, string.Join("; ", report.Unhonoured));
        // lr(t) = 3.0 * 16^-0.5 * min(t^-0.5, t * 8000^-1.5): at the end of warm-up, 3.0 / sqrt(16 * 8000).
        var schedule = new NoamSchedule(16, 8000, 3.0);
        Assert.Equal(3.0 / Math.Sqrt(16.0 * 8000), schedule.BaseLearningRate, 12);
    }

    [Fact(Timeout = 300000)]
    public async Task Training_LowersTheTransducerLoss()
    {
        await Task.Yield();
        using var model = new NemotronSpeech<double>(Architecture(), Tiny());
        var audio = Waveform(Voice(16000, 180));
        var target = Transcript(3, 5, 7);
        var objective = (ITrainingObjectiveProvider<double>)model;
        double before = objective.EvaluateTrainingObjective(audio, target);
        model.SetBaseTrainOptimizer(new AdamOptimizer<double, Tensor<double>, Tensor<double>>(model,
            new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>> { InitialLearningRate = 1e-2 }));
        var trace = new List<double>();
        for (int step = 0; step < 100; step++) { model.Train(audio, target); trace.Add(objective.EvaluateTrainingObjective(audio, target)); }
        double after = objective.EvaluateTrainingObjective(audio, target);
        Assert.True(after < 0.5 * before, $"loss {before} -> {after}: {string.Join(", ", trace.Select(t => t.ToString("F3")))}");
        string text = model.Transcribe(audio).Text;
        Assert.True(text == "\u0003\u0005\u0007", $"transcript [{string.Join(",", text.Select(c => (int)c))}] at loss {after}");
    }
}
