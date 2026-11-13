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
/// Granite-speech (Saon et al. 2025, §II): the released configuration, the stacked log-mel front end, the window
/// Q-Former's 5× downsampling, the chat prompt around the audio, Granite's multipliers, and the paper's two training
/// stages (#2397).
/// </summary>
public class GraniteSpeechPaperTests
{
    private static NeuralNetworkArchitecture<double> Architecture() =>
        new(InputType.OneDimensional, NeuralNetworkTaskType.SequenceToSequence, inputSize: 8000, outputSize: 16);

    private static GraniteSpeechOptions Tiny() => new()
    {
        NumMels = 16, EncoderDim = 16, NumEncoderLayers = 2, NumAttentionHeads = 2, EncoderHeadDim = 8, AttentionContextSize = 8,
        MaxRelativePosition = 8, ConvKernel = 3, CtcVocabSize = 8, ProjectorDim = 16, NumProjectorLayers = 1, NumProjectorHeads = 2,
        ProjectorFeedForwardDim = 32, LlmDim = 16, LlmFeedForwardDim = 32, NumLlmLayers = 1, NumLlmAttentionHeads = 2, NumLlmKvHeads = 1,
        VocabSize = 16, LoraRank = 2, MaxTextLength = 8, Seed = 5,
        // A width-16 tied head cannot lift logits divided by 16 (the 8B value) within a short test.
        LogitsScaling = 1,
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
    public async Task Defaults_AreTheReleased8BModel()
    {
        await Task.Yield();
        var o = new GraniteSpeechOptions();
        Assert.Equal((80, 1024, 16, 8, 128, 4), (o.NumMels, o.EncoderDim, o.NumEncoderLayers, o.NumAttentionHeads, o.EncoderHeadDim, o.FeedForwardMultiplier));
        Assert.Equal((200, 512, 15, 2, 256), (o.AttentionContextSize, o.MaxRelativePosition, o.ConvKernel, o.ConvExpansionFactor, o.CtcVocabSize));
        Assert.Equal((1024, 2, 16, 4096, 15, 5), (o.ProjectorDim, o.NumProjectorLayers, o.NumProjectorHeads, o.ProjectorFeedForwardDim, o.ProjectorWindowSize, o.ProjectorDownsampleRate));
        Assert.Equal((4096, 12800, 40, 32, 8, 49160), (o.LlmDim, o.LlmFeedForwardDim, o.NumLlmLayers, o.NumLlmAttentionHeads, o.NumLlmKvHeads, o.VocabSize));
        Assert.Equal((12.0, 0.22, 0.0078125, 16.0, 1e7), (o.EmbeddingMultiplier, o.ResidualMultiplier, o.AttentionMultiplier, o.LogitsScaling, o.LlmRopeTheta));
        Assert.Equal((64, 32.0, 0.0), (o.LoraRank, o.LoraAlpha, o.LoraDropout));
        Assert.Equal(GraniteSpeechTrainingStage.Joint, o.TrainingStage);
    }

    [Fact(Timeout = 120000)]
    public async Task Features_StackPairsOfNormalizedLogMelFrames()
    {
        await Task.Yield();
        using var model = new GraniteSpeech<double>(Architecture(), Tiny());
        var network = model.Network ?? throw new InvalidOperationException("native model");
        var features = network.Features(Samples(Voice(8000, 180)));
        // 1 + 8000 / 160 = 51 centred frames; the odd one is dropped and pairs are stacked: 25 × (2 · 16).
        Assert.Equal(new[] { 25, 32 }, features.Shape.ToArray());
        var values = Enumerable.Range(0, features.Length).Select(i => features.GetFlat(i)).ToArray();
        // log10, floored 8 below the maximum, then / 4 + 1: a span of at most 2.
        Assert.True(values.Max() - values.Min() <= 2.0 + 1e-12, $"span {values.Max() - values.Min()}");
    }

    [Fact(Timeout = 120000)]
    public async Task Projector_ReadsEachWindowOf15FramesWith3Queries()
    {
        await Task.Yield();
        using var model = new GraniteSpeech<double>(Architecture(), Tiny());
        var network = model.Network ?? throw new InvalidOperationException("native model");
        var encoded = network.Encode(network.Features(Samples(Voice(8000, 180))), training: false).Hidden;
        Assert.Equal(new[] { 25, 16 }, encoded.Shape.ToArray());
        // 25 frames pad to two windows of 15; each yields 15 / 5 = 3 vectors in the LLM's width.
        Assert.Equal(new[] { 6, 16 }, network.Project(encoded).Shape.ToArray());
    }

    [Fact(Timeout = 120000)]
    public async Task Prompt_IsTheModelCardsChatTemplate()
    {
        await Task.Yield();
        var options = Tiny();
        options.VocabSize = 49160;
        using var model = new GraniteSpeech<double>(Architecture(), options);
        var network = model.Network ?? throw new InvalidOperationException("native model");
        Assert.Equal(49159, network.Tokenizer.TokenId("<|audio|>"));
        Assert.Equal(0, network.EndToken);
        var speech = network.Speech(Samples(Voice(8000, 180)), training: false);
        var prompt = network.Prompt(speech, options.SystemPrompt, options.Instruction);
        int head = network.Tokenizer.Encode($"<|start_of_role|>system<|end_of_role|>{options.SystemPrompt}<|end_of_text|>\n<|start_of_role|>user<|end_of_role|>").Length;
        int tail = network.Tokenizer.Encode($"{options.Instruction}<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>").Length;
        Assert.Equal(head + speech.Shape[0] + tail, prompt.Shape[0]);
        for (int j = 0; j < 16; j++) Assert.Equal(speech[0, j], prompt[head, j], 12);
    }

    [Fact(Timeout = 120000)]
    public async Task LogitsScaling_DividesTheLogits()
    {
        await Task.Yield();
        var plain = Tiny();
        plain.LogitsScaling = 1;
        var scaled = Tiny();
        scaled.LogitsScaling = 4;
        using var a = new GraniteSpeech<double>(Architecture(), plain);
        using var b = new GraniteSpeech<double>(Architecture(), scaled);
        var audio = Voice(8000, 180);
        var x = a.Predict(audio);
        var y = b.Predict(audio);
        for (int v = 0; v < 16; v++) Assert.Equal(x.GetFlat(v) / 4, y.GetFlat(v), 10);
    }

    [Fact(Timeout = 300000)]
    public async Task Training_LowersTheTranscriptLoss()
    {
        await Task.Yield();
        using var model = new GraniteSpeech<double>(Architecture(), Tiny());
        var audio = Voice(8000, 180);
        var target = Transcript(3, 5, 7);
        var objective = (ITrainingObjectiveProvider<double>)model;
        double before = objective.EvaluateTrainingObjective(audio, target);
        model.SetBaseTrainOptimizer(new AdamOptimizer<double, Tensor<double>, Tensor<double>>(model,
            new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>> { InitialLearningRate = 1e-2 }));
        var trace = new System.Collections.Generic.List<double>();
        for (int step = 0; step < 40; step++) { model.Train(audio, target); trace.Add(objective.EvaluateTrainingObjective(audio, target)); }
        double after = objective.EvaluateTrainingObjective(audio, target);
        Assert.True(after < 0.5 * before, $"loss {before} -> {after}: {string.Join(", ", trace.Select(t => t.ToString("F3")))}");
        Assert.Equal("\u0003\u0005\u0007", model.Transcribe(audio).Text);
    }

    [Fact(Timeout = 300000)]
    public async Task EncoderCtcStage_TrainsTheEncoderAlone()
    {
        await Task.Yield();
        var options = Tiny();
        options.TrainingStage = GraniteSpeechTrainingStage.EncoderCtc;
        using var model = new GraniteSpeech<double>(Architecture(), options);
        var network = model.Network ?? throw new InvalidOperationException("native model");
        var audio = Voice(8000, 180);
        var labels = Transcript(2, 4, 6);
        var objective = (ITrainingObjectiveProvider<double>)model;
        var projector = network.ProjectorLinear.GetParameters().ToArray();
        var input = network.InputLinear.GetParameters().ToArray();
        double before = objective.EvaluateTrainingObjective(audio, labels);
        model.SetBaseTrainOptimizer(new AdamOptimizer<double, Tensor<double>, Tensor<double>>(model,
            new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>> { InitialLearningRate = 1e-2 }));
        for (int step = 0; step < 30; step++) model.Train(audio, labels);
        double after = objective.EvaluateTrainingObjective(audio, labels);
        Assert.True(after < 0.5 * before, $"CTC {before} -> {after}");
        Assert.Equal(projector, network.ProjectorLinear.GetParameters().ToArray());
        Assert.NotEqual(input, network.InputLinear.GetParameters().ToArray());
    }

    [Fact(Timeout = 300000)]
    public async Task ProjectorAndLoraStage_KeepsTheEncoderAndLlmFixed()
    {
        await Task.Yield();
        var options = Tiny();
        options.TrainingStage = GraniteSpeechTrainingStage.ProjectorAndLora;
        using var model = new GraniteSpeech<double>(Architecture(), options);
        var network = model.Network ?? throw new InvalidOperationException("native model");
        var layer = network.Llm.Layers[0];
        // LoRA on q_proj and v_proj only (the release's adapter_config).
        Assert.Equal(2, layer.Loras.Count());
        var query = layer.Query.GetParameters().ToArray();
        var encoder = network.InputLinear.GetParameters().ToArray();
        var projector = network.ProjectorLinear.GetParameters().ToArray();
        var valueLora = layer.ValueLora ?? throw new InvalidOperationException("v_proj LoRA");
        var valueB = valueLora.B.GetParameters().ToArray();
        model.SetBaseTrainOptimizer(new AdamOptimizer<double, Tensor<double>, Tensor<double>>(model,
            new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>> { InitialLearningRate = 1e-2 }));
        model.Train(Voice(8000, 180), Transcript(3, 5, 7));
        Assert.Equal(query, layer.Query.GetParameters().ToArray());
        Assert.Equal(encoder, network.InputLinear.GetParameters().ToArray());
        Assert.NotEqual(projector, network.ProjectorLinear.GetParameters().ToArray());
        Assert.NotEqual(valueB, valueLora.B.GetParameters().ToArray());
    }
}
