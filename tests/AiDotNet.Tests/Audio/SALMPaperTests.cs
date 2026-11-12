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
/// SALM (Chen et al. 2023, §3.1 and §4): the paper's configuration, the 320 ms speech prompt from the encoder's 8x and the
/// adapter's 4x subsampling, the release's <c>Q: {input}\nA: {output}</c> template over Megatron's SentencePiece
/// vocabulary, Megatron GPT's LayerNorm, the fused query-key-value LoRA, and the frozen-LLM training scope (#2397).
/// </summary>
public class SALMPaperTests
{
    private static NeuralNetworkArchitecture<double> Architecture() =>
        new(InputType.OneDimensional, NeuralNetworkTaskType.SequenceToSequence, inputSize: 16000, outputSize: 16);

    private static SALMOptions Tiny() => new()
    {
        NumMels = 16, EncoderDim = 16, NumEncoderLayers = 1, NumAttentionHeads = 2, ConvKernel = 3, SubsamplingChannels = 4,
        AdapterDim = 16, NumAdapterLayers = 1, NumAdapterHeads = 2, LlmDim = 16, LlmFeedForwardDim = 32, NumLlmLayers = 1,
        NumLlmAttentionHeads = 2, VocabSize = 16, LoraRank = 2, LoraAlpha = 2, WarmupSteps = 0, MaxTextLength = 8, Seed = 5,
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
    public async Task Defaults_AreThePapersModel()
    {
        await Task.Yield();
        var o = new SALMOptions();
        Assert.Equal((80, 512, 17, 8, 4, 9, 256), (o.NumMels, o.EncoderDim, o.NumEncoderLayers, o.NumAttentionHeads, o.FeedForwardMultiplier,
            o.ConvKernel, o.SubsamplingChannels));
        Assert.Equal((512, 2, 8), (o.AdapterDim, o.NumAdapterLayers, o.NumAdapterHeads));
        Assert.Equal((2048, 5440, 24, 16, 0.5, 256000), (o.LlmDim, o.LlmFeedForwardDim, o.NumLlmLayers, o.NumLlmAttentionHeads,
            o.PartialRotaryFactor, o.VocabSize));
        Assert.Equal((128, SpeechLlmTrainingScope.AllParameters, 2000), (o.LoraRank, o.TrainingScope, o.WarmupSteps));
    }

    [Fact(Timeout = 120000)]
    public async Task SpeechPrompt_IsSubsampled32Times()
    {
        await Task.Yield();
        using var model = new SALM<double>(Architecture(), Tiny());
        var network = model.Network ?? throw new InvalidOperationException("native model");
        var audio = Samples(Voice(16000, 180));
        int frames = network.Encoder.Features(audio).GetLength(0);
        // Five stride-2, padding-1, width-3 convolutions: three in the encoder's 8x, two in the adapter's 4x.
        int expected = frames;
        for (int i = 0; i < 5; i++) expected = (expected - 1) / 2 + 1;
        Assert.Equal(new[] { expected, 16 }, network.Speech(audio, training: false).Shape.ToArray());
    }

    [Fact(Timeout = 120000)]
    public async Task Prompt_IsTheReleasesTemplateOverMegatronsVocabulary()
    {
        await Task.Yield();
        var options = Tiny();
        options.VocabSize = 256000;
        using var model = new SALM<double>(Architecture(), options);
        var network = model.Network ?? throw new InvalidOperationException("native model");
        // sentencepiece on GPT-2B-001's tokenizer model: </s> is 3, and the template has no <s>.
        Assert.Equal(3, network.EndToken);
        Assert.Equal(new[] { 877, 251541, 4137, 5600, 276, 9967, 812, 5170, 2618, 251511, 14, 251519, 251541, 251490 },
            network.PromptTokens("Transcribe the audio into English text."));
        Assert.Equal(new[] { 41533, 1349 }, network.Tokenizer.Encode("hello world"));
        var speech = network.Speech(Samples(Voice(16000, 180)), training: false);
        var prompt = network.Prompt(speech, options.Instruction);
        Assert.Equal(speech.Shape[0] + 14, prompt.Shape[0]);
        for (int j = 0; j < 16; j++) Assert.Equal(speech[0, j], prompt[0, j], 12);
    }

    [Fact(Timeout = 60000)]
    public async Task MegatronLayerNorm_CentresAndShifts()
    {
        await Task.Yield();
        var configuration = new LlamaConfiguration
        {
            VocabSize = 8, Hidden = 8, Layers = 1, Heads = 2, KeyValueHeads = 2, Intermediate = 8, PartialRotaryFactor = 0.5, LayerNorm = true,
            // An epsilon far below the variance, so the normalized rows have unit variance.
            RmsEpsilon = 1e-14,
        };
        var network = new LlamaNetwork<double>(new CpuEngine(), new List<LayerBase<double>>(), configuration);
        // The reference initialization: LayerNorm weights 1 and biases 0.
        network.InitializeLikeReference(AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(3));
        var layer = network.Layers[0];
        Assert.NotNull(layer.InputNormBias);
        Assert.NotNull(layer.PostNormBias);
        var bias = network.FinalNormBias ?? throw new InvalidOperationException("final LayerNorm bias");
        var shift = Enumerable.Range(0, 8).Select(i => 0.1 * i).ToArray();
        bias.SetParameters(new Vector<double>(shift));
        var (hidden, _) = network.Forward(network.Embed(new[,] { { 1, 4, 6 } }));
        // With unit weights, each position's output less the bias has mean 0 and variance 1 over the width.
        for (int t = 0; t < 3; t++)
        {
            var row = Enumerable.Range(0, 8).Select(i => hidden[0, t, i] - shift[i]).ToArray();
            Assert.Equal(0.0, row.Average(), 9);
            Assert.Equal(1.0, row.Select(v => v * v).Average(), 3);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task Lora_IsOnTheFusedQueryKeyValueProjection()
    {
        await Task.Yield();
        using var model = new SALM<double>(Architecture(), Tiny());
        var network = model.Network ?? throw new InvalidOperationException("native model");
        var layer = network.Llm.Layers[0];
        var qkv = layer.QkvLora ?? throw new InvalidOperationException("query_key_value LoRA");
        Assert.Single(layer.Loras);
        Assert.Equal((2, 3 * 16), (qkv.A.OutputSize, qkv.B.OutputSize));
    }

    [Fact(Timeout = 300000)]
    public async Task Training_LowersTheTranscriptLoss()
    {
        await Task.Yield();
        using var model = new SALM<double>(Architecture(), Tiny());
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
    public async Task PaperScope_FreezesTheLlmAndTrainsTheRest()
    {
        await Task.Yield();
        var options = Tiny();
        options.TrainingScope = SpeechLlmTrainingScope.EncoderAdapterAndLora;
        using var model = new SALM<double>(Architecture(), options);
        var network = model.Network ?? throw new InvalidOperationException("native model");
        var layer = network.Llm.Layers[0];
        var query = layer.Query.GetParameters().ToArray();
        var projection = network.Projection.GetParameters().ToArray();
        var qkv = layer.QkvLora ?? throw new InvalidOperationException("query_key_value LoRA");
        var qkvB = qkv.B.GetParameters().ToArray();
        model.SetBaseTrainOptimizer(new AdamOptimizer<double, Tensor<double>, Tensor<double>>(model,
            new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>> { InitialLearningRate = 1e-2 }));
        model.Train(Voice(16000, 180), Transcript(3, 5, 7));
        Assert.Equal(query, layer.Query.GetParameters().ToArray());
        Assert.NotEqual(projection, network.Projection.GetParameters().ToArray());
        Assert.NotEqual(qkvB, qkv.B.GetParameters().ToArray());
    }
}
