using System.Text.Json;
using AiDotNet.Audio.Effects;
using AiDotNet.TextToSpeech.StyleEmotion;
using AiDotNet.TextToSpeech.VoiceCloning;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public sealed class MetaVoiceOutputPolicyTests
{
    private static JsonDocument Reference() => JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
        "TextToSpeech", "ReferenceData", "metavoice_output_policy_reference.json")));
    private static double[] Values(JsonElement item) => item.EnumerateArray().Select(x => x.GetDouble()).ToArray();

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void OriginalAudioCraftPolicy_MatchesLoudnessClampedBiquadsCompressorAndWrittenPcm(int index)
    {
        using var document = Reference(); var entry = document.RootElement.GetProperty("cases")[index];
        var audio = Values(entry.GetProperty("audio"));
        if (entry.GetProperty("loudness").ValueKind != JsonValueKind.Null)
        {
            double actual = LoudnessMeter.IntegratedTorchaudioLoudness(audio, 24000);
            Assert.True(Math.Abs(actual - entry.GetProperty("loudness").GetDouble()) < 2e-10,
                "Original torchaudio LUFS mismatch: " + actual.ToString("R"));
        }
        var normalized = MetaVoiceOutputPolicy.Normalize(audio);
        double error = normalized.Zip(Values(entry.GetProperty("normalized")), (a, b) => Math.Abs(a - b)).Max();
        Assert.True(error < 2e-10, "Original normalized waveform error " + error.ToString("R"));
        Assert.Equal(entry.GetProperty("first_pcm").EnumerateArray().Select(x => (short)x.GetInt32()).ToArray(), MetaVoiceOutputPolicy.DecoderPcm(normalized));
    }

    [Fact]
    public void OriginalDfSaveAudio_TruncatesAndWrapsRatherThanUsingFfmpegSaturation()
    {
        using var document = Reference(); var reference = document.RootElement;
        Assert.Equal(reference.GetProperty("boundary_pcm").EnumerateArray().Select(x => (short)x.GetInt32()).ToArray(),
            MetaVoiceOutputPolicy.EnhancedPcm(Values(reference.GetProperty("boundary_audio"))));
    }

    [Fact]
    public void OriginalSincFastResampler_PreservesThePinnedLegacyAliasBehavior()
    {
        AiDotNetEngine.ResetToCpu(); using var document = Reference(); var reference = document.RootElement;
        var audio = Values(reference.GetProperty("resample_audio"));
        var result = new MetaVoiceSincFastResample<double>(AiDotNetEngine.Current)
            .Forward(new Tensor<double>(new[] { audio.Length }, new Vector<double>(audio)));
        double error = result.ToArray().Zip(Values(reference.GetProperty("resampled")), (a, b) => Math.Abs(a - b)).Max();
        Assert.True(error < 2e-13, "Original sinc_fast handoff error " + error.ToString("R"));
    }

    [Fact]
    public void LoudnessPolicy_PreservesTheOriginalShortNonquietInputFailure()
        => Assert.Throws<ArgumentException>(() => MetaVoiceOutputPolicy.Normalize(Enumerable.Repeat(.1, 1377).ToArray()));
}
