using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using System.Reflection;
using System.Text.Json;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.CodecBased;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public class TortoiseCandidateReferenceTests
{
    [Fact]
    public void GenerationDefault_PreservesTheAuthorsApiCapAndCheckpointCapacitySeparately()
    {
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
            "TextToSpeech", "ReferenceData", "tortoise_candidate_reference.json")));
        var options = new TortoiseTTSOptions();
        Assert.Equal(document.RootElement.GetProperty("default_maximum").GetInt32(), options.MaximumAutoregressiveTokens);
        Assert.Equal(604, options.MaxCodecFrames);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void SamplingDistribution_MatchesHfPrefixPenaltyNucleusAndMinimumLength(int index)
    {
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
            "TextToSpeech", "ReferenceData", "tortoise_candidate_reference.json")));
        var expected = document.RootElement.GetProperty("sampling")[index];
        var options = new TortoiseTTSOptions
        {
            LLMDim = 8, NumHeads = 2, NumLLMLayers = 1, NumEncoderLayers = 1,
            VocabSize = 16, CodebookSize = 6, SpeechStartToken = 6, SpeechStopToken = 7,
            AutoregressiveMelChannels = 4, AutoregressiveFeedForwardDim = 16,
            MaxTextLength = 8, MaxCodecFrames = 8, DropoutRate = 0,
            Temperature = expected.GetProperty("temperature").GetDouble(), RepetitionPenalty = 2,
            TopP = expected.GetProperty("top_p").GetDouble(),
            MinimumAutoregressiveTokens = expected.GetProperty("minimum").GetInt32()
        };
        var model = new TortoiseAutoregressive<double>(AiDotNetEngine.Current, new List<LayerBase<double>>(), options);
        var method = typeof(TortoiseAutoregressive<double>).GetMethod("Sample", BindingFlags.Instance | BindingFlags.NonPublic)!;
        var logits = new Tensor<double>(new[] { 1, 8 }, new Vector<double>(expected.GetProperty("logits")
            .EnumerateArray().Select(x => x.GetDouble()).ToArray()));
        var previous = expected.GetProperty("previous").EnumerateArray().Select(x => x.GetInt32()).ToArray();
        foreach (var draw in expected.GetProperty("draws").EnumerateArray())
            Assert.Equal(draw.GetProperty("token").GetInt32(), (int)method.Invoke(model,
                new object[] { logits, previous, new FixedUniform(draw.GetProperty("uniform").GetDouble()) })!);
    }

    private sealed class FixedUniform : Random
    {
        private readonly double _value;
        internal FixedUniform(double value) => _value = value;
        public override double NextDouble() => _value;
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
    public void SilenceTail_PreservesTheAuthorsEightLatentFrames(int index)
    {
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
            "TextToSpeech", "ReferenceData", "tortoise_candidate_reference.json")));
        var expected = document.RootElement.GetProperty("cases")[index];
        var raw = expected.GetProperty("raw").EnumerateArray().Select(x => x.GetInt32()).ToArray();
        var codes = TortoiseCandidateTail.Prepare(raw, expected.GetProperty("maximum").GetInt32(), 8193, 8192);
        Assert.Equal(expected.GetProperty("repaired").EnumerateArray().Select(x => x.GetInt32()).ToArray(), codes);
        Assert.Equal(expected.GetProperty("latent_length").GetInt32(), TortoiseCandidateTail.LatentLength(codes, 8192));
    }
}
