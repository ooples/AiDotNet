using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.SpeechRecognition.NeMo;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Audio;

/// <summary>
/// Canary-Qwen-2.5B (the release's <c>config.json</c> and model card): the released configuration, the 80 ms speech
/// frames, NeMo's <c>qwen</c> prompt with the audio in the user turn, LoRA on q and v, and the frozen LLM (#2397).
/// </summary>
public class CanaryQwenPaperTests
{
    private static NeuralNetworkArchitecture<double> Architecture() =>
        new(InputType.OneDimensional, NeuralNetworkTaskType.SequenceToSequence, inputSize: 16000, outputSize: 16);

    private static CanaryQwenOptions Tiny() => new()
    {
        NumMels = 16, EncoderDim = 16, NumEncoderLayers = 1, NumAttentionHeads = 2, ConvKernel = 3, SubsamplingChannels = 4,
        DecoderDim = 16, DecoderFeedForwardDim = 32, NumDecoderLayers = 1, NumDecoderHeads = 2, NumDecoderKvHeads = 1, DecoderHeadDim = 8,
        VocabSize = 16, LoraRank = 2, LoraAlpha = 4, LoraDropout = 0, WarmupSteps = 0, MaxTextLength = 8, Seed = 5,
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

    private static double[] Samples(Tensor<double> audio) => Enumerable.Range(0, audio.Length).Select(i => audio[i]).ToArray();

    private static Tensor<double> Transcript(params double[] ids) => new(new[] { ids.Length }, new Vector<double>(ids));

    [Fact(Timeout = 60000)]
    public async Task Defaults_AreTheRelease()
    {
        await Task.Yield();
        var o = new CanaryQwenOptions();
        Assert.Equal((128, 1024, 32, 8, 4, 9, 256), (o.NumMels, o.EncoderDim, o.NumEncoderLayers, o.NumAttentionHeads, o.FeedForwardMultiplier,
            o.ConvKernel, o.SubsamplingChannels));
        Assert.Equal((2048, 6144, 28, 16, 8, 128, 151936), (o.DecoderDim, o.DecoderFeedForwardDim, o.NumDecoderLayers, o.NumDecoderHeads,
            o.NumDecoderKvHeads, o.DecoderHeadDim, o.VocabSize));
        Assert.Equal((128, 256.0, 0.01, 1000), (o.LoraRank, o.LoraAlpha, o.LoraDropout, o.WarmupSteps));
        Assert.Equal("Transcribe the following:", o.Instruction);
    }

    [Fact(Timeout = 120000)]
    public async Task Speech_IsOneFrameEvery80Milliseconds()
    {
        await Task.Yield();
        using var model = new CanaryQwen<double>(Architecture(), Tiny());
        var network = model.Network ?? throw new InvalidOperationException("native model");
        var audio = Samples(Voice(16000, 180));
        int frames = network.Encoder.Features(audio).GetLength(0);
        int expected = frames;
        for (int i = 0; i < 3; i++) expected = (expected - 1) / 2 + 1;
        // 12.5 frames a second, projected to the decoder's width.
        Assert.InRange(expected, 12, 13);
        Assert.Equal(new[] { expected, 16 }, network.Speech(audio, training: false).Shape.ToArray());
    }

    [Fact(Timeout = 120000)]
    public async Task Prompt_IsNemosQwenFormat()
    {
        await Task.Yield();
        var options = Tiny();
        options.VocabSize = 151936;
        using var model = new CanaryQwen<double>(Architecture(), options);
        var network = model.Network ?? throw new InvalidOperationException("native model");
        Assert.Equal(151645, network.EndToken);
        Assert.Equal(151669, network.Tokenizer.AddedTokenId(CanaryQwenNetwork<double>.AudioLocator));
        var speech = network.Speech(Samples(Voice(16000, 180)), training: false);
        var prompt = network.Prompt(speech, options.Instruction);
        var head = network.Tokenizer.Encode("<|im_start|>user\nTranscribe the following: ");
        var tail = network.Tokenizer.Encode("<|im_end|>\n<|im_start|>assistant\n");
        Assert.Equal(151644, head[0]);
        Assert.Equal(new[] { 151645, 198, 151644, 77091, 198 }, tail);
        Assert.Equal(head.Length + speech.Shape[0] + tail.Length, prompt.Shape[0]);
        for (int j = 0; j < 16; j++) Assert.Equal(speech[0, j], prompt[head.Length, j], 12);
    }

    [Fact(Timeout = 120000)]
    public async Task Lora_IsOnQueryAndValue()
    {
        await Task.Yield();
        using var model = new CanaryQwen<double>(Architecture(), Tiny());
        var network = model.Network ?? throw new InvalidOperationException("native model");
        var layer = network.Llm.Layers[0];
        Assert.Equal(2, layer.Loras.Count());
        Assert.NotNull(layer.QueryLora);
        Assert.NotNull(layer.ValueLora);
        Assert.NotNull(layer.QueryNorm);
        Assert.Null(network.Llm.LmHead);
    }

    [Fact(Timeout = 300000)]
    public async Task Training_LowersTheTranscriptLoss()
    {
        await Task.Yield();
        using var model = new CanaryQwen<double>(Architecture(), Tiny());
        var audio = Voice(16000, 180);
        var target = Transcript(3, 5, 7);
        var objective = (ITrainingObjectiveProvider<double>)model;
        double before = objective.EvaluateTrainingObjective(audio, target);
        model.SetBaseTrainOptimizer(new AdamOptimizer<double, Tensor<double>, Tensor<double>>(model,
            new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>> { InitialLearningRate = 1e-2 }));
        var trace = new List<double>();
        for (int step = 0; step < 40; step++) { model.Train(audio, target); trace.Add(objective.EvaluateTrainingObjective(audio, target)); }
        double after = objective.EvaluateTrainingObjective(audio, target);
        Assert.True(after < 0.5 * before, $"loss {before} -> {after}: {string.Join(", ", trace.Select(t => t.ToString("F3")))}");
        Assert.Equal("\u0003\u0005\u0007", model.Transcribe(audio).Text);
    }

    [Fact(Timeout = 300000)]
    public async Task ReleaseScope_FreezesTheLlm()
    {
        await Task.Yield();
        var options = Tiny();
        options.TrainingScope = SpeechLlmTrainingScope.EncoderAdapterAndLora;
        using var model = new CanaryQwen<double>(Architecture(), options);
        var network = model.Network ?? throw new InvalidOperationException("native model");
        var layer = network.Llm.Layers[0];
        var query = layer.Query.GetParameters().ToArray();
        var projection = network.Projection.GetParameters().ToArray();
        var lora = layer.QueryLora ?? throw new InvalidOperationException("q_proj LoRA");
        var loraB = lora.B.GetParameters().ToArray();
        model.SetBaseTrainOptimizer(new AdamOptimizer<double, Tensor<double>, Tensor<double>>(model,
            new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>> { InitialLearningRate = 1e-2 }));
        model.Train(Voice(16000, 180), Transcript(3, 5, 7));
        Assert.Equal(query, layer.Query.GetParameters().ToArray());
        Assert.NotEqual(projection, network.Projection.GetParameters().ToArray());
        Assert.NotEqual(loraB, lora.B.GetParameters().ToArray());
    }
}
