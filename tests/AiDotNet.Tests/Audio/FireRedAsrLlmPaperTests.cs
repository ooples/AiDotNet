using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.SpeechRecognition.LLMIntegrated;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Audio;

/// <summary>
/// FireRedASR-LLM (FireRed Team 2025, §2.2): the released configuration, the two-frame adapter, the release's prompt
/// around the speech, and training on the transcript's cross-entropy with the paper's encoder + adapter + LoRA scope
/// (#2397).
/// </summary>
public class FireRedAsrLlmPaperTests
{
    private static NeuralNetworkArchitecture<double> Architecture() =>
        new(InputType.OneDimensional, NeuralNetworkTaskType.SequenceToSequence, inputSize: 8000, outputSize: 16);

    private static FireRedASRLLMOptions Tiny() => new()
    {
        EncoderDim = 16, EncoderFeedForwardDim = 64, NumEncoderLayers = 1, NumAttentionHeads = 2, ConvKernel = 3, NumMels = 32,
        LlmDim = 16, LlmFeedForwardDim = 32, NumLlmLayers = 1, NumLlmAttentionHeads = 2, NumLlmKvHeads = 1, VocabSize = 16,
        LoraRank = 2, LoraDropout = 0, MaxTextLength = 8, Seed = 5,
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

    [Fact(Timeout = 60000)]
    public async Task Defaults_AreTheReleasedLargeModel()
    {
        await Task.Yield();
        var o = new FireRedASRLLMOptions();
        Assert.Equal((80, 1280, 5120, 16, 20, 33), (o.NumMels, o.EncoderDim, o.EncoderFeedForwardDim, o.NumEncoderLayers, o.NumAttentionHeads, o.ConvKernel));
        Assert.Equal(2, o.AdapterFrameSplicingFactor);
        Assert.Equal((3584, 18944, 28, 28, 4, 152064), (o.LlmDim, o.LlmFeedForwardDim, o.NumLlmLayers, o.NumLlmAttentionHeads, o.NumLlmKvHeads, o.VocabSize));
        Assert.Equal((64, 16.0, 0.05), (o.LoraRank, o.LoraAlpha, o.LoraDropout));
        Assert.Equal("请转写音频为文字", o.Instruction);
    }

    [Fact(Timeout = 120000)]
    public async Task Adapter_SplicesPairsOfEncoderFrames()
    {
        await Task.Yield();
        using var model = new FireRedASRLLM<double>(Architecture(), Tiny());
        var network = model.Network ?? throw new InvalidOperationException("native model");
        var samples = Samples(Voice(8000, 180));
        int frames = network.Encoder.Encode(network.Encoder.Features(samples), training: false).Shape[0];
        var speech = network.Speech(samples, training: false);
        // The odd frame is discarded (adapter.py), then every two frames make one LLM input.
        Assert.Equal(new[] { frames / 2, 16 }, speech.Shape.ToArray());
    }

    [Fact(Timeout = 120000)]
    public async Task Prompt_IsTheReleasesTemplate()
    {
        await Task.Yield();
        var options = Tiny();
        options.VocabSize = 152064;
        using var model = new FireRedASRLLM<double>(Architecture(), options);
        var network = model.Network ?? throw new InvalidOperationException("native model");
        // Qwen2-7B-Instruct's three added tokens, then the release's <speech>.
        Assert.Equal(151646, network.Tokenizer.AddedTokenId("<speech>"));
        Assert.Equal(151645, network.EndToken);
        var speech = network.Speech(Samples(Voice(8000, 180)), training: false);
        var prompt = network.Prompt(speech, options.Instruction);
        int head = network.Tokenizer.Encode("<|im_start|>user\n").Length;
        int tail = network.Tokenizer.Encode("请转写音频为文字<|im_end|>\n<|im_start|>assistant\n").Length;
        Assert.Equal(head + speech.Shape[0] + tail, prompt.Shape[0]);
        for (int j = 0; j < 16; j++) Assert.Equal(speech[0, j], prompt[head, j], 12);
    }

    [Fact(Timeout = 300000)]
    public async Task Training_LowersTheTranscriptLoss()
    {
        await Task.Yield();
        using var model = new FireRedASRLLM<double>(Architecture(), Tiny());
        var audio = Voice(8000, 180);
        var target = new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 3, 5, 7 }));
        var objective = (ITrainingObjectiveProvider<double>)model;
        double before = objective.EvaluateTrainingObjective(audio, target);
        model.SetBaseTrainOptimizer(new AdamOptimizer<double, Tensor<double>, Tensor<double>>(model,
            new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>> { InitialLearningRate = 1e-2 }));
        var trace = new System.Collections.Generic.List<double>();
        for (int step = 0; step < 40; step++) { model.Train(audio, target); trace.Add(objective.EvaluateTrainingObjective(audio, target)); }
        double after = objective.EvaluateTrainingObjective(audio, target);
        Assert.True(after < 0.5 * before, $"loss {before} -> {after}: {string.Join(", ", trace.Select(x => x.ToString("F3")))}");
        Assert.Equal("\u0003\u0005\u0007", model.Transcribe(audio).Text);
    }

    [Fact(Timeout = 300000)]
    public async Task PaperScope_TrainsEncoderAdapterAndEveryLoraBranch()
    {
        await Task.Yield();
        var options = Tiny();
        options.TrainingScope = SpeechLlmTrainingScope.EncoderAdapterAndLora;
        using var model = new FireRedASRLLM<double>(Architecture(), options);
        var network = model.Network ?? throw new InvalidOperationException("native model");
        var layer = network.Llm.Layers[0];
        // LoRA on q, k, v, o, gate, up and down (the release's target_modules).
        Assert.Equal(7, layer.Loras.Count());
        var fixedWeights = layer.Gate.GetParameters().ToArray();
        var adapter = network.Adapter1.GetParameters().ToArray();
        var downLora = layer.DownLora ?? throw new InvalidOperationException("down_proj LoRA");
        var downB = downLora.B.GetParameters().ToArray();
        model.SetBaseTrainOptimizer(new AdamOptimizer<double, Tensor<double>, Tensor<double>>(model,
            new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>> { InitialLearningRate = 1e-2 }));
        model.Train(Voice(8000, 180), new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 3, 5, 7 })));
        Assert.Equal(fixedWeights, layer.Gate.GetParameters().ToArray());
        Assert.NotEqual(adapter, network.Adapter1.GetParameters().ToArray());
        Assert.NotEqual(downB, downLora.B.GetParameters().ToArray());
    }
}
