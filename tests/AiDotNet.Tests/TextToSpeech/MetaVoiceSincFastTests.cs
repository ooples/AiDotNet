using System.Text.Json;
using AiDotNet.TextToSpeech.StyleEmotion;
using AiDotNet.TextToSpeech.VoiceCloning;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public sealed class MetaVoiceSincFastTests
{
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void PinnedLegacyAlias_MatchesBothPrecisionsOnMultitoneImpulseAndRampAndRejectsModernHann(int index)
    {
        AiDotNetEngine.ResetToCpu();
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
            "TextToSpeech", "ReferenceData", "metavoice_sinc_fast_reference.json")));
        var reference = document.RootElement.GetProperty("cases")[index];
        static double[] Values(JsonElement x) => x.EnumerateArray().Select(value => value.GetDouble()).ToArray();
        var input = Values(reference.GetProperty("audio")); var expected = Values(reference.GetProperty("output"));
        var audio = new Tensor<double>(new[] { input.Length }, new Vector<double>(input));
        var output = new MetaVoiceSincFastResample<double>(AiDotNetEngine.Current).Forward(audio);
        Assert.Equal(expected.Length, output.Length);
        double error = output.ToArray().Zip(expected, (a, b) => Math.Abs(a - b)).Max();
        Assert.True(error < 2e-13, "Pinned double filter error " + error.ToString("R"));
        var single = new MetaVoiceSincFastResample<float>(AiDotNetEngine.Current).Forward(
            new Tensor<float>(new[] { input.Length }, new Vector<float>(input.Select(x => (float)x))));
        double floatError = single.ToArray().Zip(Values(reference.GetProperty("float_output")), (a, b) => Math.Abs(a - b)).Max();
        Assert.True(floatError < 3e-7, "Pinned float filter error " + floatError.ToString("R"));
        var hann = new TorchaudioResample<double>(AiDotNetEngine.Current, 24000, 48000, 16).Forward(audio);
        double wrongError = hann.ToArray().Zip(expected, (a, b) => Math.Abs(a - b)).Max();
        Assert.True(wrongError > .001, "The negative control did not distinguish the modern Hann branch.");
        Assert.True(Math.Abs(wrongError - reference.GetProperty("wrong_hann_error").GetDouble()) < 2e-13);
    }
}
