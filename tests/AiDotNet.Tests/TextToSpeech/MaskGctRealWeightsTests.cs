using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech.FlowDiffusion;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// MaskGCT with the released weights against the release's own classes run in float64
/// (tools/reference-data/maskgct_realweights_reference.py), where the checkpoints are present: <c>MASKGCT_DIR</c>
/// (amphion/MaskGCT at 265c6ce), <c>W2VBERT_DIR</c> (facebook/w2v-bert-2.0) and <c>MASKGCT_STATS</c> (Amphion's
/// wav2vec2bert_stats.pt). Every checkpoint tensor must find its place; the prompt's semantic tokens and acoustic codes,
/// the audio decoded from the codes and one forward of each generator are compared. The model runs in float (the full
/// model in double would not fit beside the checkpoints), so codes are compared as a share and values relatively.
/// </summary>
public class MaskGctRealWeightsTests
{
    private readonly ITestOutputHelper _output;

    public MaskGctRealWeightsTests(ITestOutputHelper output) => _output = output;

    private static JsonElement Load(string name)
    {
        string relative = Path.Combine("TextToSpeech", "ReferenceData", name);
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", relative))) dir = dir.Parent;
        string path = Path.Combine(dir?.FullName ?? throw new FileNotFoundException(name), "tests", "AiDotNet.Tests", relative);
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement;
    }

    private static double[] Doubles(JsonElement e) => e.EnumerateArray().Select(v => v.GetDouble()).ToArray();
    private static int[] Ints(JsonElement e) => e.EnumerateArray().Select(v => v.GetInt32()).ToArray();

    private static double RelativeError(double[] expected, float[] actual)
    {
        double scale = expected.Max(Math.Abs), worst = 0;
        for (int i = 0; i < expected.Length; i++) worst = Math.Max(worst, Math.Abs(expected[i] - actual[i]));
        return worst / scale;
    }

    [Fact(Timeout = 3600000)]
    public async Task ReleasedWeights_MatchTheRelease_WhenPresent()
    {
        await Task.Yield();
        string? dir = Environment.GetEnvironmentVariable("MASKGCT_DIR");
        string? w2v = Environment.GetEnvironmentVariable("W2VBERT_DIR");
        string? stats = Environment.GetEnvironmentVariable("MASKGCT_STATS");
        if (dir is null || w2v is null || stats is null || !Directory.Exists(dir) || !File.Exists(Path.Combine(w2v, "model.safetensors")) || !File.Exists(stats))
            return;
        var reference = Load("maskgct_realweights_reference.json");
        var architecture = new NeuralNetworkArchitecture<float>(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);
        var model = new MaskGCT<float>(architecture, new MaskGCTOptions());
        model.LoadReferenceWeights(dir, Path.Combine(w2v, "model.safetensors"), stats);

        var semantic = model.SemanticTokens(Doubles(reference.GetProperty("speech16")));
        var expectedSemantic = Ints(reference.GetProperty("semantic"));
        Assert.Equal(expectedSemantic.Length, semantic.Length);
        int semanticSame = semantic.Zip(expectedSemantic, (a, b) => a == b ? 1 : 0).Sum();
        _output.WriteLine($"semantic tokens: {semanticSame}/{semantic.Length}");

        var codes = model.AcousticCodec.Encode(Doubles(reference.GetProperty("speech24")));
        var expectedCodes = reference.GetProperty("acoustic").EnumerateArray().Select(Ints).ToArray();
        Assert.Equal(expectedCodes.Length, codes.GetLength(0));
        int codesSame = 0;
        for (int t = 0; t < expectedCodes.Length; t++)
            for (int q = 0; q < 12; q++) codesSame += codes[t, q] == expectedCodes[t][q] ? 1 : 0;
        _output.WriteLine($"acoustic codes: {codesSame}/{expectedCodes.Length * 12}");

        var given = new int[expectedCodes.Length, 12];
        for (int t = 0; t < expectedCodes.Length; t++)
            for (int q = 0; q < 12; q++) given[t, q] = expectedCodes[t][q];
        var audio = model.AcousticCodec.Decode(given);
        Assert.Equal(reference.GetProperty("audio_length").GetInt32(), audio.Length);
        var head = Doubles(reference.GetProperty("audio_head"));
        double audioError = RelativeError(head, audio.Take(head.Length).Select(v => (float)v).ToArray());
        _output.WriteLine($"decoded audio: relative error {audioError:E2}");

        int prompt = reference.GetProperty("prompt").GetInt32();
        var rows = Ints(reference.GetProperty("rows"));
        var phones = Ints(reference.GetProperty("phones"));
        int n = expectedSemantic.Length;
        var masked = Enumerable.Range(0, n - prompt).Select(i => i % 2 == 0).ToArray();
        var t2s = model.TextToSemantic.Logits(phones, expectedSemantic.Take(prompt).ToArray(), expectedSemantic.Skip(prompt).ToArray(), masked, 0.5);
        double t2sError = 0;
        var t2sExpected = reference.GetProperty("t2s_logits").EnumerateArray().ToArray();
        for (int r = 0; r < rows.Length; r++)
        {
            var expected = Doubles(t2sExpected[r]);
            t2sError = Math.Max(t2sError, RelativeError(expected, Enumerable.Range(0, expected.Length).Select(j => t2s[rows[r] - prompt, j]).ToArray()));
        }
        _output.WriteLine($"text-to-semantic logits: relative error {t2sError:E2}");

        int frames = Math.Min(n, expectedCodes.Length), layer = reference.GetProperty("layer").GetInt32();
        var frameCodes = new int[frames, 12];
        for (int t = 0; t < frames; t++)
            for (int q = 0; q < 12; q++) frameCodes[t, q] = expectedCodes[t][q];
        var layerMask = Enumerable.Range(0, frames).Select(i => i >= prompt && (i - prompt) % 2 == 0).ToArray();
        var s2a = model.FullModel.TrainingLogits(expectedSemantic.Take(frames).ToArray(), frameCodes, layer, prompt, layerMask, 0.5);
        double s2aError = 0;
        var s2aExpected = reference.GetProperty("s2a_logits").EnumerateArray().ToArray();
        for (int r = 0; r < rows.Length; r++)
        {
            var expected = Doubles(s2aExpected[r]);
            s2aError = Math.Max(s2aError, RelativeError(expected, Enumerable.Range(0, expected.Length).Select(j => s2a[rows[r], j]).ToArray()));
        }
        _output.WriteLine($"semantic-to-acoustic logits: relative error {s2aError:E2}");

        Assert.True(semanticSame >= semantic.Length * 0.98, $"semantic tokens {semanticSame}/{semantic.Length}");
        Assert.True(codesSame >= expectedCodes.Length * 12 * 0.98, $"acoustic codes {codesSame}/{expectedCodes.Length * 12}");
        Assert.True(audioError < 1e-3, $"decoded audio relative error {audioError}");
        Assert.True(t2sError < 1e-3, $"text-to-semantic logits relative error {t2sError}");
        Assert.True(s2aError < 1e-3, $"semantic-to-acoustic logits relative error {s2aError}");
    }
}
