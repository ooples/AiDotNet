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
using AiDotNet.Optimizers;
using AiDotNet.SpeechRecognition.Specialized;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Audio;

/// <summary>
/// VoxtLM (Maiti et al. 2024; ESPnet's <c>egs2/voxtlm_v1</c>): the k50 small configuration, the speech tokenizer, the
/// ASR data format of Table 1, the causal position-free language model, the masked label-smoothed loss, and the frozen
/// tokenizer (#2397).
/// </summary>
public class VoxtLMPaperTests
{
    private static NeuralNetworkArchitecture<double> Architecture() =>
        new(InputType.OneDimensional, NeuralNetworkTaskType.SequenceToSequence, inputSize: 16000, outputSize: 23);

    private static VoxtLMOptions Tiny() => new()
    {
        NumSpeechUnits = 8, TextVocabSize = 8, EmbeddingDim = 16, ModelDim = 16, NumLayers = 2, NumAttentionHeads = 2, FeedForwardDim = 32,
        DropoutRate = 0, HubertWidth = 16, HubertLayer = 1, HubertHeads = 2, HubertIntermediate = 32, HubertConvChannels = 16,
        MaxTextLength = 8, Seed = 5,
    };

    private static double[] Voice(int samples, double pitch)
    {
        var audio = new double[samples];
        for (int i = 0; i < samples; i++)
        {
            double t = i / 16000.0;
            audio[i] = 0.4 * Math.Sin(2 * Math.PI * pitch * t) + 0.2 * Math.Sin(2 * Math.PI * (2.7 * pitch + 900 * t) * t);
        }
        return audio;
    }

    private static Tensor<double> Waveform(double[] samples) => new(new[] { samples.Length }, new Vector<double>(samples));

    private static Tensor<double> Transcript(params double[] ids) => new(new[] { ids.Length }, new Vector<double>(ids));

    [Fact(Timeout = 60000)]
    public async Task Defaults_AreK50Small()
    {
        await Task.Yield();
        var o = new VoxtLMOptions();
        Assert.Equal((50, 2000, 12, 768, 12, 3072, 768), (o.NumSpeechUnits, o.TextVocabSize, o.NumLayers, o.ModelDim, o.NumAttentionHeads,
            o.FeedForwardDim, o.EmbeddingDim));
        Assert.Equal((0.1, 0.1, 6, 768), (o.DropoutRate, o.LabelSmoothing, o.HubertLayer, o.HubertWidth));
        Assert.Equal((1e-3, 1e-6, 10000, 5.0, 150), (o.LearningRate, o.WeightDecay, o.WarmupSteps, o.MaxGradientNorm, o.MaxTextLength));
    }

    [Fact(Timeout = 120000)]
    public async Task AsrCondition_IsStartSpeechUnitsGenerateText()
    {
        await Task.Yield();
        using var model = new VoxtLM<double>(Architecture(), Tiny());
        var network = model.Network ?? throw new InvalidOperationException("native model");
        Assert.Equal(6 + 8 + 8 + 1, network.VocabSize);
        var audio = Voice(16000, 180);
        var prompt = network.AsrPrompt(audio);
        Assert.Equal(VoxtLMNetwork<double>.StartSpeech, prompt[0]);
        Assert.Equal(VoxtLMNetwork<double>.GenerateText, prompt[^1]);
        // One unit per 20 ms HuBERT frame, none merged, each a unit token.
        int frames = network.Units.Network.Frames(audio.Length);
        Assert.Equal(frames + 2, prompt.Length);
        Assert.All(prompt.Skip(1).Take(frames), id => Assert.InRange(id, VoxtLMNetwork<double>.UnitOffset, network.TextOffset - 1));
    }

    [Fact(Timeout = 60000)]
    public async Task LanguageModel_IsCausal()
    {
        await Task.Yield();
        using var model = new VoxtLM<double>(Architecture(), Tiny());
        var network = model.Network ?? throw new InvalidOperationException("native model");
        var random = Tensors.Helpers.RandomHelper.CreateSeededRandom(1);
        var a = network.Forward(new[] { 2, 7, 9, 3, 15, 16 }, training: false, random);
        var b = network.Forward(new[] { 2, 7, 9, 3, 18, 20 }, training: false, random);
        for (int t = 0; t < 4; t++)
            for (int v = 0; v < network.VocabSize; v++) Assert.Equal(a[t, v], b[t, v], 12);
        Assert.Contains(Enumerable.Range(0, network.VocabSize), v => Math.Abs(a[4, v] - b[4, v]) > 1e-9);
    }

    [Fact(Timeout = 120000)]
    public async Task Loss_IsTheTranscriptAndEndAfterTheCondition()
    {
        await Task.Yield();
        var options = Tiny();
        options.LabelSmoothing = 0;
        using var model = new VoxtLM<double>(Architecture(), options);
        var network = model.Network ?? throw new InvalidOperationException("native model");
        var random = Tensors.Helpers.RandomHelper.CreateSeededRandom(1);
        var prompt = network.AsrPrompt(Voice(16000, 180));
        var transcript = new[] { 3, 5 };
        double loss = network.AsrLoss(prompt, transcript, training: false, random)[0];
        // Unsmoothed, the loss is the summed cross-entropy of <generatetext> -> 3, 3 -> 5 and 5 -> <sos/eos>.
        var ids = prompt.Concat(transcript.Select(t => network.TextOffset + t)).ToArray();
        var logits = network.Forward(ids, training: false, random);
        var targets = transcript.Select(t => network.TextOffset + t).Concat(new[] { network.EndToken }).ToArray();
        double expected = 0;
        for (int i = 0; i < targets.Length; i++)
        {
            int row = prompt.Length - 1 + i;
            double max = Enumerable.Range(0, network.VocabSize).Max(v => logits[row, v]);
            double logSum = max + Math.Log(Enumerable.Range(0, network.VocabSize).Sum(v => Math.Exp(logits[row, v] - max)));
            expected += logSum - logits[row, targets[i]];
        }
        Assert.Equal(expected, loss, 9);
    }

    [Fact(Timeout = 60000)]
    public async Task Optimizer_IsAdamUnderWarmupLr()
    {
        await Task.Yield();
        using var model = new VoxtLM<double>(Architecture(), Tiny());
        var report = Assert.Single(PaperOptimizerFactory.ReportsFor(model));
        Assert.Equal(OptimizerKind.Adam, report.PaperOptimizer);
        Assert.True(report.Fidelity == RecipeFidelity.Exact, string.Join("; ", report.Unhonoured));
        // warmuplr peaks at the learning rate at the end of warm-up.
        var schedule = new NoamSchedule(768, 10000, 1e-3 * Math.Sqrt(768.0 * 10000));
        Assert.Equal(1e-3, schedule.BaseLearningRate, 12);
    }

    [Fact(Timeout = 300000)]
    public async Task Training_LowersTheLoss_AndLeavesTheTokenizerFixed()
    {
        await Task.Yield();
        using var model = new VoxtLM<double>(Architecture(), Tiny());
        var network = model.Network ?? throw new InvalidOperationException("native model");
        var audio = Waveform(Voice(16000, 180));
        var target = Transcript(3, 5, 7);
        var hubert = network.Units.Network.Projection.GetParameters().ToArray();
        var decoder = network.Decoder.GetParameters().ToArray();
        var objective = (ITrainingObjectiveProvider<double>)model;
        double before = objective.EvaluateTrainingObjective(audio, target);
        model.SetBaseTrainOptimizer(new AdamOptimizer<double, Tensor<double>, Tensor<double>>(model,
            new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>> { InitialLearningRate = 1e-2 }));
        var trace = new List<double>();
        for (int step = 0; step < 60; step++) { model.Train(audio, target); trace.Add(objective.EvaluateTrainingObjective(audio, target)); }
        double after = objective.EvaluateTrainingObjective(audio, target);
        Assert.True(after < 0.5 * before, $"loss {before} -> {after}: {string.Join(", ", trace.Select(t => t.ToString("F3")))}");
        string text = model.Transcribe(audio).Text;
        Assert.True(text == "\u0003\u0005\u0007", $"transcript [{string.Join(",", text.Select(c => (int)c))}] at loss {after}");
        Assert.Equal(hubert, network.Units.Network.Projection.GetParameters().ToArray());
        Assert.NotEqual(decoder, network.Decoder.GetParameters().ToArray());
    }

    [Fact(Timeout = 300000)]
    public async Task FittedUnits_SpreadOverTheClusters()
    {
        await Task.Yield();
        using var model = new VoxtLM<double>(Architecture(), Tiny());
        var network = model.Network ?? throw new InvalidOperationException("native model");
        var recordings = new[] { 140.0, 220.0, 330.0, 500.0 }.Select(p => Waveform(Voice(16000, p))).ToList();
        model.FitSpeechUnits(recordings, iterations: 10);
        var used = recordings.SelectMany(r => network.SpeechTokens(Enumerable.Range(0, r.Length).Select(i => r[i]).ToArray())).Distinct().Count();
        Assert.True(used >= 4, $"the fitted tokenizer uses {used} of 8 units");
    }
}
