using System.Text.Json;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.VoiceCloning;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public sealed class ChatterboxVoiceReferenceTests
{
    [Fact]
    public void OriginalPowerMelAndTrimBoundaries_MatchIncludingShortAndQuietRecordings()
    {
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
            "TextToSpeech", "ReferenceData", "chatterbox_voice_reference.json")));
        var transform = new AiDotNet.TextToSpeech.Vocoders.CenteredLogMel<double>(AiDotNetEngine.Current,
            16000, 400, 160, 400, 4, 0, 8000, 1e-5, spectrumPower: 2, logarithm: false);
        foreach (var item in document.RootElement.GetProperty("cases").EnumerateArray())
        {
            var audio = item.GetProperty("audio").EnumerateArray().Select(value => value.GetDouble()).ToArray();
            var trimmed = ChatterboxVoiceEncoder<double>.TrimSilence(audio, 20);
            Assert.Equal(item.GetProperty("trimmed_length").GetInt32(), trimmed.Length);
            var actual = transform.Forward(new Tensor<double>(new[] { trimmed.Length }, new Vector<double>(trimmed)));
            int frames = item.GetProperty("mel_frames").GetInt32();
            Assert.Equal(new[] { 1, 4, frames }, actual.Shape.ToArray());
            var expected = item.GetProperty("mel").EnumerateArray().Select(value => value.GetDouble()).ToArray();
            double error = 0;
            for (int frame = 0; frame < frames; frame++)
                for (int channel = 0; channel < 4; channel++)
                    error = Math.Max(error, Math.Abs(actual[0, channel, frame] - expected[frame * 4 + channel]));
            Assert.True(error < 2e-6, $"Input {audio.Length}: power-mel error {error:R}.");
        }
    }


    [Fact]
    public void WaveformMelTrimmingPartialsAndVoiceEmbedding_MatchTheOriginalPipeline()
    {
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
            "TextToSpeech", "ReferenceData", "chatterbox_voice_reference.json")));
        var reference = document.RootElement;
        double[] Values(JsonElement values) => values.EnumerateArray().Select(value => value.GetDouble()).ToArray();
        foreach (var item in reference.GetProperty("cases").EnumerateArray())
        {
            var options = new ChatterboxOptions
            {
                VoiceEncoderMelChannels = 4, VoiceEncoderLayers = 3, VoiceEncoderHiddenDim = 8, T3SpeakerEmbeddingDim = 6,
                VoiceEncoderPartialFrames = item.GetProperty("partial_frames").GetInt32(),
                VoiceEncoderPartialRate = item.GetProperty("partial_rate").GetDouble()
            };
            var encoder = new ChatterboxVoiceEncoder<double>(AiDotNetEngine.Current, new List<LayerBase<double>>(), options);
            encoder.LoadTorchWeights((name, shape) =>
            {
                Assert.Equal(shape, reference.GetProperty("weight_shapes").GetProperty(name).EnumerateArray().Select(value => value.GetInt32()).ToArray());
                return Values(reference.GetProperty("weights").GetProperty(name));
            });
            var audio = Values(item.GetProperty("audio"));
            var embedding = encoder.Embed(new Tensor<double>(new[] { audio.Length }, new Vector<double>(audio)), 16000);
            var expected = Values(item.GetProperty("embedding"));
            double error = embedding.ToArray().Zip(expected, (a, b) => Math.Abs(a - b)).Max();
            Assert.True(error < 2e-6, $"Audio {audio.Length}, partial frames {options.VoiceEncoderPartialFrames}: embedding error {error:R}.");
        }
    }
}
