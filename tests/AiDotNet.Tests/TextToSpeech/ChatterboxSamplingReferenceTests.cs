using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text.Json;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.VoiceCloning;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public class ChatterboxSamplingReferenceTests
{
    [Fact]
    public void GenerationLoop_UsesTheAuthorsSingleTokenNucleusAtAnExactBoundary()
    {
        var options = SmallOptions();
        options.MaxGeneratedSpeechTokens = 3;
        options.MinP = 0;
        options.GenerationTopP = .125;
        var model = new ChatterboxT3<double>(AiDotNetEngine.Current, new List<LayerBase<double>>(), options, new Random(1));
        TorchParameters.Linear(model.SpeechHead, options.LLMDim, options.SpeechVocabularySize,
            new double[options.LLMDim * options.SpeechVocabularySize]);
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
            "TextToSpeech", "ReferenceData", "chatterbox_sampling_reference.json")));
        // The pinned author policy keeps only stop ID 7 for this uniform
        // eight-token distribution. The real transformer runs with a zero head.
        Assert.All(document.RootElement.GetProperty("cases")[5].GetProperty("draws").EnumerateArray(),
            draw => Assert.Equal(options.SpeechStopToken, draw.GetProperty("token").GetInt32()));
        var speaker = new Tensor<double>(new[] { options.T3SpeakerEmbeddingDim });
        speaker[0] = 1;
        var condition = new ChatterboxT3Condition<double>(speaker, new[] { 1, 2 }, .5);
        Assert.Empty(model.Generate(new[] { 1 }, condition, new Random(5)));
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
    public void PostLogitSampling_MatchesTheAuthorsActualInferenceStatements(int index)
    {
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
            "TextToSpeech", "ReferenceData", "chatterbox_sampling_reference.json")));
        var expected = document.RootElement.GetProperty("cases")[index];
        var options = SmallOptions();
        options.ClassifierFreeGuidanceWeight = expected.GetProperty("cfg").GetDouble();
        options.Temperature = expected.GetProperty("temperature").GetDouble();
        options.RepetitionPenalty = expected.GetProperty("repetition").GetDouble();
        options.MinP = expected.GetProperty("min_p").GetDouble();
        options.GenerationTopP = expected.GetProperty("top_p").GetDouble();
        var model = new ChatterboxT3<double>(AiDotNetEngine.Current, new List<LayerBase<double>>(), options, new Random(1));
        var values = expected.GetProperty("conditional").EnumerateArray().Concat(
            expected.GetProperty("unconditional").EnumerateArray()).Select(x => x.GetDouble()).ToArray();
        var logits = new Tensor<double>(new[] { 2, 1, 8 }, new Vector<double>(values));
        var history = expected.GetProperty("history").EnumerateArray().Select(x => x.GetInt32()).ToArray();
        var sample = typeof(ChatterboxT3<double>).GetMethod("Sample", BindingFlags.Instance | BindingFlags.NonPublic)!;
        int TokenAt(double uniform) => (int)sample.Invoke(model,
            new object[] { logits, history, true, new FixedUniform(uniform) })!;
        foreach (var draw in expected.GetProperty("draws").EnumerateArray())
            Assert.Equal(draw.GetProperty("token").GetInt32(), TokenAt(draw.GetProperty("uniform").GetDouble()));
        // Probe both sides of every resolved CDF boundary, checking the full
        // active distribution to 1e-9 rather than only six categorical outcomes.
        var order = expected.GetProperty("probabilities").EnumerateArray()
            .Select((value, token) => (Token: token, Probability: value.GetDouble()))
            .Where(item => item.Probability > 0).OrderByDescending(item => item.Probability)
            .ThenBy(item => item.Token).ToArray();
        double cumulative = 0;
        const double epsilon = 1e-9;
        for (int i = 0; i < order.Length - 1; i++)
        {
            cumulative += order[i].Probability;
            if (order[i].Probability < 2 * epsilon || order[i + 1].Probability < 2 * epsilon) continue;
            Assert.Equal(order[i].Token, TokenAt(cumulative - epsilon));
            Assert.Equal(order[i + 1].Token, TokenAt(cumulative + epsilon));
        }
    }

    private static ChatterboxOptions SmallOptions() => new()
    {
        VocabSize = 32, SpeechVocabularySize = 8, CodebookSize = 6,
        LLMDim = 16, NumLLMLayers = 1, NumHeads = 2, KeyValueHeads = 2,
        LmFeedForwardDim = 32, MaxTextLength = 16, MaxSpeechTokens = 16,
        T3SpeakerEmbeddingDim = 6, PerceiverQueries = 2, PerceiverHeads = 2, PerceiverDropout = 0,
        TextStartToken = 30, TextStopToken = 31, SpeechStartToken = 6, SpeechStopToken = 7,
        RopeTheta = 10000, RopeScalingFactor = 1
    };

    private sealed class FixedUniform : Random
    {
        private readonly double _value;
        internal FixedUniform(double value) => _value = value;
        public override double NextDouble() => _value;
    }
}
