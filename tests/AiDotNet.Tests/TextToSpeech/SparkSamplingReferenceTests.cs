using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text.Json;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.CodecBased;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public class SparkSamplingReferenceTests
{
    [Fact]
    public void FullVocabularyGeneration_RetainsAllTopKTiesAndTerminatesOnEos()
    {
        var options = SmallOptions();
        options.TopK = 1;
        options.TopP = 1;
        options.Temperature = 1;
        options.MaxCodecFrames = 3;
        var model = new SparkTokenLanguageModel<double>(AiDotNetEngine.Current, new List<LayerBase<double>>(), options);
        model.Transformer.LoadTorchWeights((name, shape) => new double[shape.Aggregate(1, (a, b) => a * b)]);
        using var document = Read();
        Assert.All(document.RootElement.GetProperty("uniform_topk_one").EnumerateArray(),
            value => Assert.Equal(.125, value.GetDouble()));
        Assert.Empty(model.GenerateAll(new[] { 1 }, new FixedUniform(.999999)));
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
    public void SamplingDistribution_MatchesTheAuthorsPinnedTransformersPolicy(int index)
    {
        using var document = Read();
        var expected = document.RootElement.GetProperty("cases")[index];
        var options = SmallOptions();
        options.Temperature = expected.GetProperty("temperature").GetDouble();
        options.TopK = expected.GetProperty("top_k").GetInt32();
        options.TopP = expected.GetProperty("top_p").GetDouble();
        var model = new SparkTokenLanguageModel<double>(AiDotNetEngine.Current, new List<LayerBase<double>>(), options);
        var logits = new Tensor<double>(new[] { 1, 1, 8 }, new Vector<double>(expected.GetProperty("logits")
            .EnumerateArray().Select(x => x.GetDouble()).ToArray()));
        var sample = typeof(SparkTokenLanguageModel<double>).GetMethod("Sample", BindingFlags.Instance | BindingFlags.NonPublic)!;
        int TokenAt(double uniform) => (int)sample.Invoke(model,
            new object[] { logits, Enumerable.Range(0, 8), new FixedUniform(uniform) })!;
        foreach (var draw in expected.GetProperty("draws").EnumerateArray())
            Assert.Equal(draw.GetProperty("token").GetInt32(), TokenAt(draw.GetProperty("uniform").GetDouble()));
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

    private static JsonDocument Read() => JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
        "TextToSpeech", "ReferenceData", "spark_sampling_reference.json")));

    private static SparkTTSOptions SmallOptions() => new()
    {
        VocabSize = 8, LLMDim = 16, NumLLMLayers = 1, NumHeads = 2,
        KeyValueHeads = 2, LmIntermediateDim = 32, EosToken = 7, GenerationSeed = 1
    };

    private sealed class FixedUniform : Random
    {
        private readonly double _value;
        internal FixedUniform(double value) => _value = value;
        public override double NextDouble() => _value;
    }
}
