using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text.Json;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.VoiceCloning;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public class XttsSamplingReferenceTests
{
    [Fact]
    public void NativeOptionValidation_AcceptsTheOriginalHfZeroNucleusBoundary()
    {
        using var document = Read();
        var reference = document.RootElement.GetProperty("cases")[5];
        Assert.Equal(0, reference.GetProperty("top_p").GetDouble());
        typeof(XTTSv2<double>).GetMethod("ValidateOptions", BindingFlags.Static | BindingFlags.NonPublic)!
            .Invoke(null, new object[] { Options(reference) });
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    [InlineData(6)]
    [InlineData(7)]
    [InlineData(8)]
    [InlineData(9)]
    [InlineData(10)]
    [InlineData(11)]
    public void Sampling_MatchesOriginalSupportedHfPolicy(int index)
    {
        using var document = Read();
        var reference = document.RootElement.GetProperty("cases")[index];
        var options = Options(reference);
        var model = new XttsLanguageModel<double>(AiDotNetEngine.Current,
            new List<LayerBase<double>>(), options, new Random(1));
        var values = reference.GetProperty("logits").EnumerateArray().Select(x => x.GetDouble()).ToArray();
        var logits = new Tensor<double>(new[] { 1, values.Length });
        for (int i = 0; i < values.Length; i++) logits[0, i] = values[i];
        var history = reference.GetProperty("history").EnumerateArray().Select(x => x.GetInt32()).ToArray();
        var probabilities = reference.GetProperty("probabilities").EnumerateArray().Select(x => x.GetDouble()).ToArray();
        var ordered = Enumerable.Range(0, probabilities.Length).Where(i => probabilities[i] > 0)
            .OrderByDescending(i => probabilities[i]).ThenBy(i => i).ToArray();
        int Expected(double uniform)
        {
            double cumulative = 0;
            foreach (int token in ordered)
            {
                cumulative += probabilities[token];
                if (uniform <= cumulative) return token;
            }
            return ordered[ordered.Length - 1];
        }
        void Check(double uniform)
        {
            int actual = (int)typeof(XttsLanguageModel<double>).GetMethod("Sample",
                BindingFlags.Instance | BindingFlags.NonPublic)!.Invoke(model,
                    new object[] { logits, 0, history, new FixedUniform(uniform) })!;
            Assert.Equal(Expected(uniform), actual);
        }
        foreach (var value in reference.GetProperty("uniforms").EnumerateArray()) Check(value.GetDouble());
        double boundary = 0;
        for (int i = 0; i < ordered.Length - 1; i++)
        {
            boundary += probabilities[ordered[i]];
            if (boundary > 1e-9 && boundary < 1 - 1e-9)
            {
                Check(boundary - 1e-9);
                Check(boundary + 1e-9);
            }
        }
    }

    [Theory]
    [InlineData(6)]
    [InlineData(7)]
    [InlineData(9)]
    [InlineData(10)]
    [InlineData(11)]
    public void RealGeneration_UsesOriginalPrefixHistoryAndRetainsTheEosPredictionLatent(int index)
    {
        using var document = Read();
        var reference = document.RootElement.GetProperty("cases")[index];
        var options = Options(reference);
        var replay = reference.GetProperty("generation_trace");
        int firstCode = replay.GetProperty("generated_codes")[0].GetInt32();
        Assert.Equal(1, reference.GetProperty("probabilities")[firstCode].GetDouble());
        var model = new XttsLanguageModel<double>(AiDotNetEngine.Current,
            new List<LayerBase<double>>(), options, new Random(1));
        model.LoadTorchWeights((name, shape) => name == "gpt.mel_head.bias"
            ? reference.GetProperty("logits").EnumerateArray().Select(x => x.GetDouble()).ToArray()
            : new double[shape.Aggregate(1, (a, b) => a * b)], includeConditioning: false);
        var result = model.Generate(new[] { 2, 3 }, new Tensor<double>(new[] { 2, options.LLMDim }), new FixedUniform(.5));
        Assert.Equal(replay.GetProperty("generated_codes").EnumerateArray().Select(x => x.GetInt32()), result.Codes);
        Assert.Equal(replay.GetProperty("latent_frames").GetInt32(), result.Latents.Shape[0]);
    }

    private static XTTSv2Options Options(JsonElement reference) => new()
    {
        LLMDim = 16, NumLLMLayers = 1, NumHeads = 4, ConditioningBlocks = 1, ConditioningHeads = 4,
        PerceiverDepth = 1, PerceiverLatents = 2, PerceiverHeads = 1, DropoutRate = 0,
        TextVocabularySize = 10, AudioVocabularySize = 8, CodebookSize = 6,
        StartTextToken = 8, StopTextToken = 9, StartAudioToken = 6, StopAudioToken = 7,
        MaxTextLength = 8, MaxCodecFrames = 3,
        Temperature = reference.GetProperty("temperature").GetDouble(),
        TopK = reference.GetProperty("top_k").GetInt32(), TopP = reference.GetProperty("top_p").GetDouble(),
        RepetitionPenalty = reference.GetProperty("repetition_penalty").GetDouble(),
    };

    private static JsonDocument Read() => JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
        "TextToSpeech", "ReferenceData", "xtts_sampling_reference.json")));

    private sealed class FixedUniform : Random
    {
        private readonly double _value;
        internal FixedUniform(double value) => _value = value;
        public override double NextDouble() => _value;
    }
}
