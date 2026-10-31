using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.Vocoders;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// GPT-SoVITS's v4/V5 vocoder (the release's Generator with is_bias, as init_hifigan builds it, narrower) and the v3/v4
/// reference mels (mel_fn, mel_fn_v4, norm_spec) against the release in float64
/// (tools/reference-data/gptsovits_vocoder_reference.py).
/// </summary>
public class GptSoVitsVocoderReferenceTests
{
    private static readonly Lazy<JsonElement> Reference = new(() =>
    {
        const string name = "gptsovits_vocoder_reference.json";
        string relative = Path.Combine("TextToSpeech", "ReferenceData", name);
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", relative))) dir = dir.Parent;
        string path = Path.Combine(dir?.FullName ?? throw new FileNotFoundException(name), "tests", "AiDotNet.Tests", relative);
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement;
    });

    private static double[] Doubles(JsonElement e) => e.EnumerateArray().Select(v => v.GetDouble()).ToArray();

    private static void Close(double[] expected, Tensor<double> actual, double tolerance, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual.GetFlat(i)) <= tolerance * (1 + Math.Abs(expected[i])), $"{what}[{i}]: {expected[i]} vs {actual.GetFlat(i)}");
    }

    [Fact(Timeout = 300000)]
    public async Task V4Vocoder_MatchesGenerator()
    {
        await Task.Yield();
        var r = Reference.Value;
        var state = r.GetProperty("state");
        var generator = new HiFiGanGenerator<double>(AiDotNetEngine.Current, 100, 32, new[] { 10, 6 }, new[] { 20, 12 }, new[] { 3, 7 },
            new[] { new[] { 1, 3, 5 }, new[] { 1, 3, 5 } }, resblockType1: true, plainEnds: true, plainOutputBias: true);
        generator.LoadTorch(name => state.TryGetProperty(name, out var v) ? Doubles(v) : null);
        var audio = generator.Forward(new Tensor<double>(new[] { 1, 100, 7 }, new Vector<double>(Doubles(r.GetProperty("mel")))));
        Close(Doubles(r.GetProperty("audio")), audio, 1e-9, "audio");
    }

    [Theory(Timeout = 300000)]
    [InlineData("v3")]
    [InlineData("v4")]
    public async Task ReferenceMel_MatchesMelFn(string version)
    {
        await Task.Yield();
        var r = Reference.Value;
        var wav = Doubles(r.GetProperty("wav"));
        var mel = GptSoVitsCfmSynthesizer<double>.ReferenceMel(AiDotNetEngine.Current, version, new Tensor<double>(new[] { wav.Length }, new Vector<double>(wav)));
        var e = r.GetProperty(version);
        Assert.Equal(e.GetProperty("shape").EnumerateArray().Select(v => v.GetInt32()).ToArray(), mel.Shape.ToArray());
        Close(Doubles(e.GetProperty("mel")), mel, 1e-9, "mel");
    }
}
