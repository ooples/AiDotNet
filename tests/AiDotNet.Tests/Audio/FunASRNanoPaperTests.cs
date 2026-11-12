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
/// Fun-ASR-nano (Fun-ASR report, 2025): the release's configuration, the ChatML transcription prompt around the adaptor's
/// output, and the report's staged training of the LLM path and the CTC decoder (#2397).
/// </summary>
public class FunASRNanoPaperTests
{
    private static NeuralNetworkArchitecture<double> Architecture() =>
        new(InputType.OneDimensional, NeuralNetworkTaskType.SequenceToSequence, inputSize: 8000, outputSize: 16);

    private static FunASRNanoOptions Tiny() => new()
    {
        NumMels = 16, EncoderDim = 16, NumEncoderLayers = 1, NumTpEncoderLayers = 1, NumAttentionHeads = 2, EncoderFeedForwardDim = 32,
        FsmnKernelSize = 3, AdaptorFeedForwardDim = 32, NumAdaptorLayers = 1, AdaptorHeads = 2, LlmHiddenDim = 16, NumLlmLayers = 1,
        LlmHeads = 2, LlmKeyValueHeads = 1, LlmHeadDim = 8, LlmIntermediateDim = 32, VocabSize = 16, LoraRank = 4, LoraDropout = 0,
        CtcDecoderDim = 16, CtcFeedForwardDim = 32, NumCtcDecoderLayers = 1, CtcVocabSize = 16, MaxTextLength = 8, Seed = 5,
    };

    // Fun-ASR starts from a pretrained Qwen3 whose token rows have norms far above a fresh N(0, 0.02²) table's; through the
    // frozen tied head such a fresh table caps every logit near zero, so no trainable module could sharpen the answer.
    // The fixture loads rows at a pretrained scale (unit-variance entries) in place of a checkpoint.
    private static void LoadPretrainedScaleEmbeddings(FunAsrNanoNetwork<double> network)
    {
        var table = network.Llm.Embeddings ?? throw new InvalidOperationException("token table");
        var values = table.GetParameters();
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(11);
        var rows = new double[values.Length];
        for (int i = 0; i < rows.Length; i++)
            rows[i] = Math.Sqrt(-2.0 * Math.Log(1.0 - random.NextDouble())) * Math.Cos(2.0 * Math.PI * random.NextDouble());
        table.SetParameters(new Vector<double>(rows));
    }

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
    public async Task Defaults_AreTheReleasedConfiguration()
    {
        await Task.Yield();
        var o = new FunASRNanoOptions();
        // SenseVoiceEncoderSmall, the Transformer adaptor and CTC decoder, Qwen3-0.6B, and the release's training recipe.
        Assert.Equal((80, 7, 6, 512, 50, 20, 4, 2048, 11), (o.NumMels, o.LfrM, o.LfrN, o.EncoderDim, o.NumEncoderLayers, o.NumTpEncoderLayers,
            o.NumAttentionHeads, o.EncoderFeedForwardDim, o.FsmnKernelSize));
        Assert.Equal((1, 2048, 2, 8), (o.AdaptorDownsampleRate, o.AdaptorFeedForwardDim, o.NumAdaptorLayers, o.AdaptorHeads));
        Assert.Equal((1024, 28, 16, 8, 128, 3072, 151936), (o.LlmHiddenDim, o.NumLlmLayers, o.LlmHeads, o.LlmKeyValueHeads, o.LlmHeadDim,
            o.LlmIntermediateDim, o.VocabSize));
        Assert.Equal(1_000_000, o.LlmRopeTheta);
        Assert.Equal((16, 32.0), (o.LoraRank, o.LoraAlpha));
        Assert.Equal((512, 2048, 5, 60515), (o.CtcDecoderDim, o.CtcFeedForwardDim, o.NumCtcDecoderLayers, o.CtcVocabSize));
        Assert.Equal(2500, o.WarmupSteps);
        Assert.Equal(FunASRNanoTrainingStage.Joint, o.TrainingStage);
    }

    [Fact(Timeout = 60000)]
    public async Task Instruction_IsTheReleasesPrompt()
    {
        await Task.Yield();
        Assert.Equal("语音转写：", FunAsrNanoNetwork<double>.Instruction(null, null, true));
        Assert.Equal("语音转写成英文，不进行文本规整：", FunAsrNanoNetwork<double>.Instruction(null, "英文", false));
        string withHotwords = FunAsrNanoNetwork<double>.Instruction(new[] { "通义", "千问" }, null, true);
        Assert.StartsWith("请结合上下文信息", withHotwords);
        Assert.EndsWith("热词列表：[通义, 千问]\n语音转写：", withHotwords);
    }

    [Fact(Timeout = 120000)]
    public async Task Prompt_WrapsTheSpeechInChatML()
    {
        await Task.Yield();
        var options = Tiny();
        options.VocabSize = 151936;
        using var model = new FunASRNano<double>(Architecture(), options);
        var network = model.Network ?? throw new InvalidOperationException("native model");
        Assert.True(network.UsesQwenVocabulary);
        Assert.Equal(network.Tokenizer.AddedTokenId("<|im_end|>"), network.EndToken);
        var encoded = network.Encode(Voice(8000, 180));
        var speech = network.Adaptor.Forward(encoded);
        var prompt = network.Prompt(encoded, "语音转写：");
        int head = network.Tokenizer.Encode("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n语音转写：").Length;
        int tail = network.Tokenizer.Encode("<|im_end|>\n<|im_start|>assistant\n").Length;
        Assert.Equal(head + speech.Shape[0] + tail, prompt.Shape[0]);
        // The adaptor's output sits between the instruction and the assistant turn, unchanged.
        for (int j = 0; j < 16; j++) Assert.Equal(speech[0, j], prompt[head, j], 12);
    }

    [Fact(Timeout = 300000)]
    public async Task Training_LowersTheLlmLossAndKeepsTheLlmFrozen()
    {
        await Task.Yield();
        var options = Tiny();
        options.TrainingStage = FunASRNanoTrainingStage.FullFineTuning;
        using var model = new FunASRNano<double>(Architecture(), options);
        var network = model.Network ?? throw new InvalidOperationException("native model");
        var audio = Voice(8000, 180);
        var target = new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 3, 5, 7 }));
        var objective = (ITrainingObjectiveProvider<double>)model;
        LoadPretrainedScaleEmbeddings(network);
        var frozen = network.Llm.Layers[0].Gate.GetParameters().ToArray();
        double before = objective.EvaluateTrainingObjective(audio, target);
        model.SetBaseTrainOptimizer(new AdamOptimizer<double, Tensor<double>, Tensor<double>>(model,
            new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>> { InitialLearningRate = 1e-2 }));
        var trace = new System.Collections.Generic.List<double>();
        for (int step = 0; step < 200; step++) { model.Train(audio, target); trace.Add(objective.EvaluateTrainingObjective(audio, target)); }
        double after = objective.EvaluateTrainingObjective(audio, target);
        Assert.True(after < 0.1 * before, $"loss {before} -> {after}: {string.Join(", ", trace.Select(x => x.ToString("F3")))}");
        // Stage 4 trains the encoder, the adaptor and the LoRA adapters; the LLM's own weights stay as loaded.
        Assert.Equal(frozen, network.Llm.Layers[0].Gate.GetParameters().ToArray());
        Assert.Equal("\u0003\u0005\u0007", model.Transcribe(audio).Text);
    }

    [Fact(Timeout = 300000)]
    public async Task CtcStage_TrainsOnlyTheCtcDecoder()
    {
        await Task.Yield();
        var options = Tiny();
        options.TrainingStage = FunASRNanoTrainingStage.CtcDecoder;
        using var model = new FunASRNano<double>(Architecture(), options);
        var network = model.Network ?? throw new InvalidOperationException("native model");
        var audio = Voice(8000, 180);
        var target = new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 2, 4, 6 }));
        var objective = (ITrainingObjectiveProvider<double>)model;
        var adaptor = network.Adaptor.Linear1.GetParameters().ToArray();
        double before = objective.EvaluateTrainingObjective(audio, target);
        model.SetBaseTrainOptimizer(new AdamOptimizer<double, Tensor<double>, Tensor<double>>(model,
            new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>> { InitialLearningRate = 1e-2 }));
        for (int step = 0; step < 80; step++) model.Train(audio, target);
        double after = objective.EvaluateTrainingObjective(audio, target);
        Assert.True(after < 0.5 * before, $"CTC loss {before} -> {after}");
        Assert.Equal(adaptor, network.Adaptor.Linear1.GetParameters().ToArray());
        Assert.Equal(new[] { 2, 4, 6 }, model.FirstPassTokens(audio));
    }
}
