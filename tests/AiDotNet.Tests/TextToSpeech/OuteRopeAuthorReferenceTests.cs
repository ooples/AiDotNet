using System;
using System.IO;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech.Latest;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public class OuteRopeAuthorReferenceTests
{
    [Fact]
    public void ReleasedLlamaScalingMatchesUnmodifiedTransformers4483()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", "outetts_rope_reference.json");
        var reference = JObject.Parse(File.ReadAllText(path));
        Assert.Equal("b2d343e0237fa0492208e1a00ec899f548d7d5e4", (string)reference["git_blob"]!);
        foreach (var c in reference["cases"]!)
        {
            var options = new OuteTTSOptions
            {
                HiddenDim = 16, VocabSize = 64, Layers = 1, Heads = 2, KeyValueHeads = 1, HeadDim = 8, Intermediate = 32,
                CodecEncoderDim = 2, CodecEncoderRates = new[] { 4, 8, 10 }, CodecLatentDim = 16,
                CodecDecoderDim = 8, CodecDecoderRates = new[] { 10, 8, 4 }, CodecCodebookDim = 4,
                AlignerMels = 80, AlignerWidth = 16, AlignerHeads = 2, AlignerEncoderLayers = 1, AlignerDecoderLayers = 1,
                RopeLowFrequencyFactor = (double)c["low"]!, RopeHighFrequencyFactor = (double)c["high"]!,
                RopeOriginalMaxPositions = (int)c["original"]!,
            };
            options.HeadDim = (int)c["head_dim"]!; // Both the compact fixture and the release's actual 64-wide heads.
            var model = new OuteTTS<double>(new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
                NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1) { RandomSeed = 11 }, options);
            var actual = model.LanguageModel.InverseFrequencies();
            var expected = c["frequencies"]!.Select(f => (double)f).ToArray();
            Assert.Equal(expected.Length, actual.Length);
            // The author explicitly emits float32 for every model dtype. Keep the same fixed tolerance for portable
            // CPU power kernels and rounding through the exponent, wavelength, interpolation and final blend.
            for (int i = 0; i < expected.Length; i++)
            {
                Assert.True(TtsPaperNumerics.IsFinite(actual[i]));
                Assert.InRange(Math.Abs(actual[i] - expected[i]), 0, 3e-7 * Math.Abs(expected[i]));
            }
        }
    }
}
