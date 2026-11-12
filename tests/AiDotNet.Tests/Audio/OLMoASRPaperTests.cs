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
using AiDotNet.Tokenization.Algorithms;
using Xunit;

namespace AiDotNet.Tests.Audio;

/// <summary>
/// OLMoASR (Ngo et al. 2025, §3): Whisper's English-only architecture and tokenizer, the English-only decoder prompt,
/// and training on the transcript's teacher-forced cross-entropy (#2397).
/// </summary>
public class OLMoASRPaperTests
{
    private static NeuralNetworkArchitecture<double> Architecture() =>
        new(InputType.OneDimensional, NeuralNetworkTaskType.SequenceToSequence, inputSize: 8000, outputSize: 16);

    private static OLMoASROptions Tiny() => new()
    {
        ModelDim = 16, NumAttentionHeads = 2, NumEncoderLayers = 1, NumDecoderLayers = 1, VocabSize = 16, MaxTextLength = 8, Seed = 5,
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
    public async Task Defaults_AreTheReleasedLargeEnV2()
    {
        await Task.Yield();
        var o = new OLMoASROptions();
        Assert.Equal((80, 1280, 20, 32, 32, 51864), (o.NumMels, o.ModelDim, o.NumAttentionHeads, o.NumEncoderLayers, o.NumDecoderLayers, o.VocabSize));
        Assert.Equal(224, o.MaxTextLength);
    }

    [Fact(Timeout = 60000)]
    public async Task EnglishEncoding_IsWhispersGpt2Vocabulary()
    {
        await Task.Yield();
        var english = WhisperTiktoken.English;
        Assert.Equal(51864, english.VocabularySize);
        Assert.Equal(50256, english.SpecialTokenId("<|endoftext|>"));
        Assert.Equal(50257, english.SpecialTokenId("<|startoftranscript|>"));
        Assert.Equal(50362, english.SpecialTokenId("<|notimestamps|>"));
        // GPT-2's ids: "Hello" is 15496 and " world" is 995.
        Assert.Equal(new[] { 15496, 995 }, english.Encode("Hello world"));
        Assert.Equal("Hello world", english.Decode(english.Encode("Hello world")));
    }

    [Fact(Timeout = 120000)]
    public async Task Prompt_IsStartOfTranscriptThenNoTimestamps()
    {
        await Task.Yield();
        var options = Tiny();
        options.VocabSize = 51864;
        using var model = new OLMoASR<double>(Architecture(), options);
        // English-only models have no language or task tokens.
        Assert.Equal(new[] { 50257, 50362 }, model.Prompt());
        Assert.Equal(new[] { 2, 51864 }, model.Predict(Voice(8000, 180)).Shape.ToArray());
    }

    [Fact(Timeout = 300000)]
    public async Task Training_LowersTheTranscriptLoss()
    {
        await Task.Yield();
        using var model = new OLMoASR<double>(Architecture(), Tiny());
        var audio = Voice(8000, 180);
        var target = new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 3, 5, 7 }));
        var objective = (ITrainingObjectiveProvider<double>)model;
        double before = objective.EvaluateTrainingObjective(audio, target);
        model.SetBaseTrainOptimizer(new AdamOptimizer<double, Tensor<double>, Tensor<double>>(model,
            new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>> { InitialLearningRate = 1e-2 }));
        var trace = new System.Collections.Generic.List<double>();
        for (int step = 0; step < 30; step++) { model.Train(audio, target); trace.Add(objective.EvaluateTrainingObjective(audio, target)); }
        double after = objective.EvaluateTrainingObjective(audio, target);
        Assert.True(after < 0.5 * before, $"loss {before} -> {after}: {string.Join(", ", trace.Select(t => t.ToString("F3")))}");
        Assert.Equal("\u0003\u0005\u0007", model.Transcribe(audio).Text);
    }
}
