using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using AiDotNet.TextToSpeech.CodecBased;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public class CosyVoiceSamplingReferenceTests
{
    [Theory]
    [InlineData(0, 0)]
    [InlineData(0, 1)]
    [InlineData(0, 2)]
    [InlineData(0, 3)]
    [InlineData(1, 0)]
    [InlineData(1, 1)]
    [InlineData(1, 2)]
    [InlineData(1, 3)]
    public void InitialEosController_MatchesTheSelectedOriginalAuthorVersion(int source, int index)
    {
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
            "TextToSpeech", "ReferenceData", "cosyvoice_sampling_reference.json")));
        var expected = document.RootElement.GetProperty("sources")[source].GetProperty("generation")[index];
        var scores = expected.GetProperty("masses").EnumerateArray()
            .Select(x => x.GetDouble() > 0 ? Math.Log(x.GetDouble()) : double.NegativeInfinity).ToArray();
        var uniforms = expected.GetProperty("uniforms").EnumerateArray().Select(x => x.GetDouble()).ToArray();
        var random = new ReplayUniforms(uniforms);
        bool ignore = expected.GetProperty("ignore_eos").GetBoolean();
        int Draw()
        {
            if (source == 0)
            {
                var config = new CosyVoiceLmConfiguration(10, 4, 4, 4, 2,
                    new WenetEncoderConfiguration(4, 4, 1, 4, 1, WenetInputLayer.Linear, true, true),
                    new WenetEncoderConfiguration(4, 4, 1, 4, 1, WenetInputLayer.LinearLegacy, false, false));
                return CosyVoiceLm<double>.SampleForGeneration(scores, Array.Empty<int>(), config, random, ignore);
            }
            return CosyVoiceLm<double>.SampleForGeneration(scores, Array.Empty<int>(), .8, 25, 10, .1, random, 4, ignore);
        }
        if (expected.GetProperty("failed").GetBoolean()) Assert.Throws<InvalidOperationException>(() => Draw());
        else Assert.Equal(expected.GetProperty("token").GetInt32(), Draw());
        Assert.Equal(uniforms.Length, random.Consumed);
        Assert.Equal(ReadScores(expected.GetProperty("post_scores")), scores);
    }

    [Theory]
    [InlineData(0, 0)]
    [InlineData(0, 1)]
    [InlineData(0, 2)]
    [InlineData(0, 3)]
    [InlineData(0, 4)]
    [InlineData(0, 5)]
    [InlineData(0, 6)]
    [InlineData(0, 7)]
    [InlineData(0, 8)]
    [InlineData(0, 9)]
    [InlineData(0, 10)]
    [InlineData(1, 0)]
    [InlineData(1, 1)]
    [InlineData(1, 2)]
    [InlineData(1, 3)]
    [InlineData(1, 4)]
    [InlineData(1, 5)]
    [InlineData(1, 6)]
    [InlineData(1, 7)]
    [InlineData(1, 8)]
    [InlineData(1, 9)]
    [InlineData(1, 10)]
    public void RepetitionAwareSampler_MatchesTheSelectedOriginalAuthorVersion(int source, int index)
    {
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
            "TextToSpeech", "ReferenceData", "cosyvoice_sampling_reference.json")));
        var expected = document.RootElement.GetProperty("sources")[source].GetProperty("cases")[index];
        var scores = expected.GetProperty("scores").EnumerateArray()
            .Select(x => x.ValueKind == JsonValueKind.Null ? double.NegativeInfinity : x.GetDouble()).ToArray();
        var decoded = expected.GetProperty("decoded").EnumerateArray().Select(x => x.GetInt32()).ToArray();
        double topP = expected.GetProperty("top_p").GetDouble(), ratio = expected.GetProperty("ratio").GetDouble();
        int topK = expected.GetProperty("top_k").GetInt32(), window = expected.GetProperty("window").GetInt32();
        var uniforms = expected.GetProperty("uniforms").EnumerateArray().Select(x => x.GetDouble()).ToArray();
        var random = new ReplayUniforms(uniforms);
        int actual;
        if (source == 0)
        {
            var config = new CosyVoiceLmConfiguration(10, 6, 4, 4, 2,
                new WenetEncoderConfiguration(4, 4, 1, 4, 1, WenetInputLayer.Linear, true, true),
                new WenetEncoderConfiguration(4, 4, 1, 4, 1, WenetInputLayer.LinearLegacy, false, false),
                TopP: topP, TopK: topK, RepetitionWindow: window, RepetitionRatio: ratio);
            actual = CosyVoiceLm<double>.SampleRepetitionAware(scores, decoded, config, random);
        }
        else
        {
            // CosyVoice2/3 call the scalar overload with the current author's
            // exclusion-on-fallback rule. Original 300M uses its own config.
            actual = CosyVoiceLm<double>.SampleRepetitionAware(scores, decoded, topP, topK, window, ratio, random);
        }
        Assert.Equal(expected.GetProperty("token").GetInt32(), actual);
        Assert.Equal(uniforms.Length, random.Consumed);
        Assert.Equal(ReadScores(expected.GetProperty("post_scores")), scores);
    }

    private static double[] ReadScores(JsonElement scores) => scores.EnumerateArray()
        .Select(x => x.ValueKind == JsonValueKind.Null ? double.NegativeInfinity : x.GetDouble()).ToArray();

    private sealed class ReplayUniforms : Random
    {
        private readonly Queue<double> _values;
        internal int Consumed { get; private set; }
        internal ReplayUniforms(IEnumerable<double> values) => _values = new Queue<double>(values);
        public override double NextDouble()
        {
            Consumed++;
            Assert.NotEmpty(_values);
            return _values.Dequeue();
        }
    }
}
