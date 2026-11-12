using System;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text.Json;
using AiDotNet.Audio.Generation;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.CodecBased;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public class VALLEXGenerationReferenceTests
{
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void CanonicalCloningGeneration_MatchesOriginalBosDependentLengthLimit(int index)
    {
        using var document = Read();
        var reference = document.RootElement.GetProperty("controller_cases")[index];
        var options = VALLEXOptions.CommunityCheckpoint();
        options.HiddenDim = 16; options.NumHeads = 2;
        options.NumEncoderLayers = options.NumDecoderLayers = 1; options.FeedForwardDim = 32;
        options.NumCodebooks = 3; options.CodebookSize = 4;
        options.PrependBos = reference.GetProperty("prepend_bos").GetBoolean();
        options.Temperature = 1; options.TopK = 1; options.MaxCodesPerTextToken = 16;
        options.Codec = new EnCodecOptions { SampleRate = 24000, NumQuantizers = 3, CodebookSize = 4,
            Filters = 4, Ratios = new[] { 8, 5, 4, 2 }, Dimension = 8, TargetBandwidthKbps = .45 };
        using var model = new VALLEX<double>(new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
            NeuralNetworkTaskType.Generative, inputSize: 1, outputSize: 1), options);
        var core = (VallECore<double>)typeof(VallEModelBase<double>)
            .GetField("_core", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)!;
        core.LoadTorchWeights((name, shape) =>
        {
            var values = new double[shape.Aggregate(1, (a, b) => a * b)];
            if (name is "nar_text_position.alpha" or "nar_audio_position.alpha") values[0] = 1;
            if (name == "ar_decoder.norm.bias") values[0] = 1;
            if (name == "ar_predict_layer.weight") values[0] = 10;
            return values;
        });
        var target = Enumerable.Repeat(1, reference.GetProperty("text_length").GetInt32()).ToArray();
        using var inference = new NoGradScope<double>();
        var codes = (int[,])typeof(VALLEX<double>).GetMethod("Generate", BindingFlags.Instance | BindingFlags.NonPublic)!
            .Invoke(model, new object[] { target, Array.Empty<int>(), new int[1, 3],
                new TtsVoice<double> { LanguageId = 0 }, new FixedUniform(.5) })!;
        Assert.All(Enumerable.Range(0, codes.GetLength(0)), frame => Assert.Equal(0, codes[frame, 0]));
        Assert.Equal(reference.GetProperty("generated_codes").GetArrayLength(), codes.GetLength(0));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void TopKSampling_MatchesOriginalDistributionAndNeverSelectsMaskedZeroMass(int index)
    {
        using var document = Read();
        var reference = document.RootElement.GetProperty("sampling_cases")[index];
        var logits = reference.GetProperty("logits").EnumerateArray().Select(x => x.GetDouble()).ToArray();
        var probabilities = reference.GetProperty("probabilities").EnumerateArray().Select(x => x.GetDouble()).ToArray();
        foreach (var value in reference.GetProperty("uniforms").EnumerateArray())
        {
            double uniform = value.GetDouble(), cumulative = 0;
            int expected = -1;
            for (int token = 0; token < probabilities.Length; token++)
            {
                cumulative += probabilities[token];
                if (uniform < cumulative) { expected = token; break; }
            }
            int actual = VallECore<double>.SampleTopK(logits, reference.GetProperty("temperature").GetDouble(),
                reference.GetProperty("top_k").GetInt32(), new FixedUniform(uniform));
            Assert.True(probabilities[actual] > 0, "Sampling must not return an original zero-probability token.");
            Assert.Equal(expected, actual);
        }
    }

    private static JsonDocument Read() => JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
        "TextToSpeech", "ReferenceData", "vallex_generation_reference.json")));

    private sealed class FixedUniform : Random
    {
        private readonly double _value;
        internal FixedUniform(double value) => _value = value;
        public override double NextDouble() => _value;
    }
}
