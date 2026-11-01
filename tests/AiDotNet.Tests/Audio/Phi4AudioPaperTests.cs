using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.SpeechRecognition.LLMIntegrated;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Audio;

/// <summary>
/// Phi-4-multimodal's speech path (Abdin et al. 2025, §2.2): the released configuration, SpeechLib filterbanks, the 8×
/// subsampling that sets the release's audio token count, the ASR prompt, Phi-3's partial rotary embedding with LongRoPE,
/// the fused speech LoRA, and the paper's training stages (#2397).
/// </summary>
public class Phi4AudioPaperTests
{
    private static NeuralNetworkArchitecture<double> Architecture() =>
        new(InputType.OneDimensional, NeuralNetworkTaskType.SequenceToSequence, inputSize: 8000, outputSize: 16);

    private static Phi4AudioOptions Tiny() => new()
    {
        NumMels = 16, EncoderDim = 16, NumEncoderLayers = 1, NumAttentionHeads = 2, EncoderFeedForwardDim = 24, SubsamplingChannels = 4,
        RelativeBiasMaxDistance = 8, LlmDim = 16, LlmFeedForwardDim = 32, NumLlmLayers = 1, NumLlmAttentionHeads = 2, NumLlmKvHeads = 1,
        UseLongRope = false, VocabSize = 16, LoraRank = 2, LoraAlpha = 4, LoraDropout = 0, MaxTextLength = 8, Seed = 5,
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
    public async Task Defaults_AreTheReleasedModel()
    {
        await Task.Yield();
        var o = new Phi4AudioOptions();
        Assert.Equal((80, 1024, 24, 16, 1536, 1024, 3, 500), (o.NumMels, o.EncoderDim, o.NumEncoderLayers, o.NumAttentionHeads,
            o.EncoderFeedForwardDim, o.SubsamplingChannels, o.ConvKernel, o.RelativeBiasMaxDistance));
        Assert.Equal((3072, 8192, 32, 24, 8, 200064), (o.LlmDim, o.LlmFeedForwardDim, o.NumLlmLayers, o.NumLlmAttentionHeads, o.NumLlmKvHeads, o.VocabSize));
        Assert.Equal((0.75, 10000.0, true), (o.PartialRotaryFactor, o.LlmRopeTheta, o.UseLongRope));
        Assert.Equal((320, 640.0, 0.01), (o.LoraRank, o.LoraAlpha, o.LoraDropout));
        Assert.Equal("Transcribe the audio clip into text.", o.Instruction);
    }

    [Fact(Timeout = 60000)]
    public async Task SpeechLibFilterbank_SpansBinsOneTo245()
    {
        await Task.Yield();
        var bank = Phi4AudioNetwork<double>.SpeechLibMel(16000, 512, 80, 0.0, 7690.0);
        // f2bin(0) + 1 = 1 through f2bin(7690) = 246, exclusive.
        for (int m = 0; m < 80; m++)
        {
            Assert.Equal(0.0, bank[m, 0]);
            for (int k = 246; k < 257; k++) Assert.Equal(0.0, bank[m, k]);
            for (int k = 0; k < 257; k++) Assert.InRange(bank[m, k], 0.0, 1.0);
        }
        Assert.True(Enumerable.Range(1, 245).All(k => Enumerable.Range(0, 80).Any(m => bank[m, k] > 0)));
    }

    [Fact(Timeout = 120000)]
    public async Task Encoder_SubsamplesToTheReleasesAudioTokenCount()
    {
        await Task.Yield();
        using var model = new Phi4Audio<double>(Architecture(), Tiny());
        var network = model.Network ?? throw new InvalidOperationException("native model");
        var features = network.Features(Samples(Voice(8000, 180)));
        // ⌊(8000 − 400) / 160⌋ + 1 frames of 16 log bands, each at least log(1).
        Assert.Equal((48, 16), (features.GetLength(0), features.GetLength(1)));
        Assert.True(features.Cast<double>().All(v => v >= 0));
        // _compute_audio_embed_size: ⌈48 / 8⌉ = 6 tokens of the projector's output.
        Assert.Equal(new[] { 6, 16 }, network.Speech(Samples(Voice(8000, 180))).Shape.ToArray());
    }

    [Fact(Timeout = 120000)]
    public async Task Prompt_IsTheModelCardsAsrPrompt()
    {
        await Task.Yield();
        var options = Tiny();
        options.VocabSize = 200064;
        using var model = new Phi4Audio<double>(Architecture(), options);
        var network = model.Network ?? throw new InvalidOperationException("native model");
        Assert.Equal(200011, network.AudioTokenId);
        Assert.Equal(new[] { 200020, 199999 }, network.EndTokens);
        Assert.Equal(new[] { 200021 }, network.Tokenizer.Encode("<|user|>"));
        var speech = network.Speech(Samples(Voice(8000, 180)));
        var prompt = network.Prompt(speech, options.Instruction);
        int tail = network.Tokenizer.Encode("Transcribe the audio clip into text.<|end|><|assistant|>").Length;
        Assert.Equal(1 + speech.Shape[0] + tail, prompt.Shape[0]);
        for (int j = 0; j < 16; j++) Assert.Equal(speech[0, j], prompt[1, j], 12);
    }

    [Fact(Timeout = 120000)]
    public async Task Tokenizer_MatchesTheRelease()
    {
        await Task.Yield();
        using var model = new Phi4Audio<double>(Architecture(), Tiny());
        var network = model.Network ?? throw new InvalidOperationException("native model");
        // Hugging Face tokenizers on the release's tokenizer.json: the chat tokens strip the whitespace after them
        // (rstrip), not before, and the o200k split keeps digits in threes.
        Assert.Equal(new[] { 3248, 4981, 290, 11065, 19925, 1511, 2201, 13, 200020, 200019 },
            network.Tokenizer.Encode("Transcribe the audio clip into text.<|end|><|assistant|>"));
        Assert.Equal(new[] { 200021, 13225, 2375, 11, 4275, 220, 7633, 2548, 293, 141801, 1703, 3443, 2543, 220, 200020, 87 },
            network.Tokenizer.Encode("<|user|> Hello world, it's 12345 o'clock!\n\nNew line <|end|>  x"));
    }

    [Fact(Timeout = 60000)]
    public async Task PartialRotaryAndLongRope_FollowPhi3()
    {
        await Task.Yield();
        double[] longFactors = { 1, 2, 4 }, shortFactors = { 1, 1, 1 };
        var configuration = new LlamaConfiguration
        {
            VocabSize = 8, Hidden = 16, Layers = 1, Heads = 2, KeyValueHeads = 1, Intermediate = 16, RopeTheta = 10000,
            PartialRotaryFactor = 0.75, RopeOriginalMaxPositions = 4096, LongRopeShortFactors = shortFactors,
            LongRopeLongFactors = longFactors, LongRopeMaxPositions = 131072,
        };
        // Three quarters of a head of 8: 6 rotary dimensions, 3 frequencies of base^(−2k/6).
        Assert.Equal(6, configuration.RotaryDim);
        var network = new LlamaNetwork<double>(new CpuEngine(), new List<LayerBase<double>>(), configuration);
        var shortRange = network.InverseFrequencies(100);
        var longRange = network.InverseFrequencies(5000);
        for (int k = 0; k < 3; k++)
        {
            double plain = 1.0 / Math.Pow(10000, 2.0 * k / 6);
            Assert.Equal(plain, shortRange[k], 12);
            Assert.Equal(plain / longFactors[k], longRange[k], 12);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task SpeechLora_IsFusedOverQkvAndGateUp()
    {
        await Task.Yield();
        using var model = new Phi4Audio<double>(Architecture(), Tiny());
        var network = model.Network ?? throw new InvalidOperationException("native model");
        var layer = network.Llm.Layers[0];
        // peft on qkv_proj, o_proj, gate_up_proj and down_proj: four adapters, two of them fused.
        Assert.Equal(4, layer.Loras.Count());
        var qkv = layer.QkvLora ?? throw new InvalidOperationException("qkv_proj LoRA");
        var gateUp = layer.GateUpLora ?? throw new InvalidOperationException("gate_up_proj LoRA");
        Assert.Equal(16 + 2 * 8, qkv.B.OutputSize);
        Assert.Equal(2 * 32, gateUp.B.OutputSize);
        Assert.Null(layer.QueryLora);
    }

    [Fact(Timeout = 300000)]
    public async Task Training_LowersTheTranscriptLoss()
    {
        await Task.Yield();
        using var model = new Phi4Audio<double>(Architecture(), Tiny());
        var audio = Voice(8000, 180);
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
    public async Task PostTrainingStage_KeepsTheEncoderAndDecoderFixed()
    {
        await Task.Yield();
        var options = Tiny();
        options.TrainingStage = Phi4AudioTrainingStage.ProjectorAndLora;
        using var model = new Phi4Audio<double>(Architecture(), options);
        var network = model.Network ?? throw new InvalidOperationException("native model");
        var layer = network.Llm.Layers[0];
        var query = layer.Query.GetParameters().ToArray();
        var encoder = network.SubsamplingOut.GetParameters().ToArray();
        var projector = network.Projection1.GetParameters().ToArray();
        var qkv = layer.QkvLora ?? throw new InvalidOperationException("qkv_proj LoRA");
        var qkvB = qkv.B.GetParameters().ToArray();
        model.SetBaseTrainOptimizer(new AdamOptimizer<double, Tensor<double>, Tensor<double>>(model,
            new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>> { InitialLearningRate = 1e-2 }));
        model.Train(Voice(8000, 180), Transcript(3, 5, 7));
        Assert.Equal(query, layer.Query.GetParameters().ToArray());
        Assert.Equal(encoder, network.SubsamplingOut.GetParameters().ToArray());
        Assert.NotEqual(projector, network.Projection1.GetParameters().ToArray());
        Assert.NotEqual(qkvB, qkv.B.GetParameters().ToArray());
    }
}
