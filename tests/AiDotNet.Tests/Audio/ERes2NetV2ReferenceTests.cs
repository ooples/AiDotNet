using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using AiDotNet.Audio.Speaker;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Audio;

/// <summary>
/// ERes2NetV2's forward3 with Kaldi fbank against GPT-SoVITS's vendored 3D-Speaker modules in float64
/// (tools/reference-data/eres2netv2_reference.py): a small encoder in eval mode.
/// </summary>
public class ERes2NetV2ReferenceTests
{
    private static JsonElement Load()
    {
        const string name = "eres2netv2_reference.json";
        string relative = Path.Combine("Audio", "ReferenceData", name);
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", relative))) dir = dir.Parent;
        string path = Path.Combine(dir?.FullName ?? throw new FileNotFoundException(name), "tests", "AiDotNet.Tests", relative);
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement;
    }

    private static double[] Doubles(JsonElement e) => e.EnumerateArray().Select(v => v.GetDouble()).ToArray();

    [Fact(Timeout = 300000)]
    public async Task FbankAndForward3_MatchTheReference()
    {
        await Task.Yield();
        var r = Load();
        var state = r.GetProperty("state");
        var encoder = new ERes2NetV2Encoder<double>(AiDotNetEngine.Current, name => state.TryGetProperty(name, out var v) ? Doubles(v) : null,
            channels: 4, blocks: new[] { 1, 1, 1, 1 });
        var wav = Doubles(r.GetProperty("wav"));
        var shape = r.GetProperty("fbank_shape").EnumerateArray().Select(v => v.GetInt32()).ToArray();
        var expectedFbank = Doubles(r.GetProperty("fbank"));
        var fbank = new AiDotNet.Audio.Features.KaldiFbank<double>(AiDotNetEngine.Current, 16000, 80, AiDotNet.Audio.Features.WindowType.Povey)
            .Forward(new Tensor<double>(new[] { wav.Length }, new Vector<double>(wav)));
        Assert.Equal(shape, fbank.Shape.ToArray());
        for (int i = 0; i < expectedFbank.Length; i++)
            Assert.True(Math.Abs(expectedFbank[i] - fbank.GetFlat(i)) < 1e-8 * (1 + Math.Abs(expectedFbank[i])), $"fbank[{i}]: {expectedFbank[i]} vs {fbank.GetFlat(i)}");
        var embedding = encoder.Embed(new Tensor<double>(shape, new Vector<double>(expectedFbank)));
        var expected = Doubles(r.GetProperty("embedding"));
        Assert.Equal(expected.Length, embedding.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - embedding.GetFlat(i)) < 1e-9 * (1 + Math.Abs(expected[i])), $"embedding[{i}]: {expected[i]} vs {embedding.GetFlat(i)}");
    }
}
