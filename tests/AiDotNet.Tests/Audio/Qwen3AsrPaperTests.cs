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
/// Qwen3-ASR-1.7B and -0.6B (Qwen3-ASR technical report, 2026): the releases' configurations, the AuT encoder's 12.5 Hz
/// token rate, the chat template around the audio, and fine-tuning on the answer's cross-entropy (#2397).
/// </summary>
public class Qwen3AsrPaperTests
{
    private static NeuralNetworkArchitecture<double> Architecture() =>
        new(InputType.OneDimensional, NeuralNetworkTaskType.SequenceToSequence, inputSize: 8000, outputSize: 16);

    private static Qwen3ASRSmallOptions Tiny() => new()
    {
        NumMels = 80, EncoderDim = 16, NumEncoderLayers = 1, NumAttentionHeads = 2, EncoderFeedForwardDim = 32, DownsampleChannels = 4,
        LlmHiddenDim = 16, NumLlmLayers = 1, LlmHeads = 2, LlmKeyValueHeads = 1, LlmHeadDim = 8, LlmIntermediateDim = 32, VocabSize = 16,
        MaxTextLength = 8, Seed = 5,
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
    public async Task Defaults_AreTheReleasedConfigurations()
    {
        await Task.Yield();
        var large = new Qwen3ASROptions();
        Assert.Equal((128, 1024, 24, 16, 4096, 480), (large.NumMels, large.EncoderDim, large.NumEncoderLayers, large.NumAttentionHeads,
            large.EncoderFeedForwardDim, large.DownsampleChannels));
        Assert.Equal((2048, 28, 16, 8, 128, 6144, 151936), (large.LlmHiddenDim, large.NumLlmLayers, large.LlmHeads, large.LlmKeyValueHeads,
            large.LlmHeadDim, large.LlmIntermediateDim, large.VocabSize));
        var small = new Qwen3ASRSmallOptions();
        Assert.Equal((128, 896, 18, 14, 3584, 480), (small.NumMels, small.EncoderDim, small.NumEncoderLayers, small.NumAttentionHeads,
            small.EncoderFeedForwardDim, small.DownsampleChannels));
        Assert.Equal((1024, 28, 16, 8, 128, 3072, 151936), (small.LlmHiddenDim, small.NumLlmLayers, small.LlmHeads, small.LlmKeyValueHeads,
            small.LlmHeadDim, small.LlmIntermediateDim, small.VocabSize));
        Assert.Equal((100, 8, 1_000_000.0), (small.ChunkFrames, small.WindowChunks, small.LlmRopeTheta));
    }

    [Fact(Timeout = 120000)]
    public async Task Encoder_EmitsThirteenTokensPerSecond()
    {
        await Task.Yield();
        using var model = new Qwen3ASRSmall<double>(Architecture(), Tiny());
        var network = model.Network ?? throw new InvalidOperationException("native model");
        // 2.5 s: 250 frames, two full chunks of 13 tokens and a 50-frame tail of 7 (_get_feat_extract_output_lengths).
        var spectrogram = network.Spectrogram(Voice(40000, 180));
        Assert.Equal(250, spectrogram.Shape[1]);
        Assert.Equal(33, network.Tokens(250));
        var encoded = network.Encode(spectrogram);
        Assert.Equal(new[] { 33, 16 }, encoded.Shape.ToArray());
    }

    [Fact(Timeout = 120000)]
    public async Task Prompt_IsTheReleasesChatTemplate()
    {
        await Task.Yield();
        var options = Tiny();
        options.VocabSize = 151936;
        using var model = new Qwen3ASRSmall<double>(Architecture(), options);
        var network = model.Network ?? throw new InvalidOperationException("native model");
        // The release's tokenizer ids.
        Assert.Equal(151669, network.Tokenizer.AddedTokenId("<|audio_start|>"));
        Assert.Equal(151670, network.Tokenizer.AddedTokenId("<|audio_end|>"));
        Assert.Equal(151676, network.Tokenizer.AddedTokenId("<|audio_pad|>"));
        Assert.Equal(151704, network.TextMarker);
        Assert.Equal(151645, network.EndToken);
        var audio = network.Encode(network.Spectrogram(Voice(16000, 180)));
        var prompt = network.Prompt(audio, "", "English");
        int head = network.Tokenizer.Encode("<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>").Length;
        int tail = network.Tokenizer.Encode("<|audio_end|><|im_end|>\n<|im_start|>assistant\nlanguage English<asr_text>").Length;
        Assert.Equal(head + 13 + tail, prompt.Shape[0]);
        for (int j = 0; j < 16; j++) Assert.Equal(audio[0, j], prompt[head, j], 12);
    }

    [Fact(Timeout = 300000)]
    public async Task FineTuning_LowersTheAnswerLoss()
    {
        await Task.Yield();
        using var model = new Qwen3ASRSmall<double>(Architecture(), Tiny());
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

    [Fact(Timeout = 120000)]
    public async Task LargeModel_BuildsTheSameNetworkAtItsOwnWidths()
    {
        await Task.Yield();
        var tiny = Tiny();
        var options = new Qwen3ASROptions
        {
            NumMels = 80, EncoderDim = 24, NumEncoderLayers = 1, NumAttentionHeads = 2, EncoderFeedForwardDim = 32, DownsampleChannels = 4,
            LlmHiddenDim = 32, NumLlmLayers = 1, LlmHeads = 2, LlmKeyValueHeads = 1, LlmHeadDim = 8, LlmIntermediateDim = 32,
            VocabSize = tiny.VocabSize, Seed = 5,
        };
        using var model = new Qwen3ASR<double>(Architecture(), options);
        var network = model.Network ?? throw new InvalidOperationException("native model");
        // The projector maps the encoder's width to the decoder's.
        Assert.Equal(new[] { 13, 32 }, network.Encode(network.Spectrogram(Voice(16000, 180))).Shape.ToArray());
        Assert.Equal(new[] { 1, 16 }, model.Predict(Voice(8000, 180)).Shape.ToArray());
    }
}
