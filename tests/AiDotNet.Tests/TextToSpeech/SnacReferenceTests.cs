using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using AiDotNet.Audio.Codecs;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// SNAC, Orpheus's codec, against snac.SNAC with the released 24 kHz configuration (tools/reference-data/snac_reference.py):
/// every checkpoint name has a place, the codes of a signal match, and decoding them (with the reference's noise draws)
/// matches to 1e-9.
/// </summary>
public class SnacReferenceTests
{
    private static JsonElement Load(string name)
    {
        string relative = Path.Combine("TextToSpeech", "ReferenceData", name);
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", relative))) dir = dir.Parent;
        string path = Path.Combine(dir?.FullName ?? throw new FileNotFoundException(name), "tests", "AiDotNet.Tests", relative);
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement;
    }

    internal static double[] ClosedForm(string name, int count) => Enumerable.Range(0, count).Select(i =>
    {
        double s = Math.Sin(0.37 * i + 0.11 * name.Length);
        return name.EndsWith(".alpha", StringComparison.Ordinal) ? 1 + 0.5 * s
            : name.EndsWith("original0", StringComparison.Ordinal) ? 0.5 + 0.2 * s
            : 0.05 * s;
    }).ToArray();

    [Fact(Timeout = 300000)]
    public async Task EncodeAndDecode_MatchSnac()
    {
        await Task.Yield();
        var reference = Load("snac_reference.json");
        var names = reference.GetProperty("names").EnumerateObject().ToDictionary(p => p.Name, p => p.Value.GetInt32(), StringComparer.Ordinal);
        var snac = new Snac<double>(new CpuEngine(), new SnacConfiguration(), new List<LayerBase<double>>());
        var used = snac.Load(name => names.TryGetValue(name, out int count) ? ClosedForm(name, count) : null);
        Assert.DoesNotContain(names.Keys, n => !used.Contains(n));

        var audio = reference.GetProperty("audio").EnumerateArray().Select(v => v.GetDouble()).ToArray();
        var expectedCodes = reference.GetProperty("codes").EnumerateArray().Select(r => r.EnumerateArray().Select(v => v.GetInt32()).ToArray()).ToArray();
        var codes = snac.Encode(audio);
        Assert.Equal(expectedCodes.Length, codes.Length);
        for (int q = 0; q < codes.Length; q++) Assert.Equal(expectedCodes[q], codes[q]);

        var draws = new Queue<double[]>(reference.GetProperty("noise").EnumerateArray().Select(r => r.EnumerateArray().Select(v => v.GetDouble()).ToArray()));
        var decoded = snac.Decode(expectedCodes, _ => draws.Dequeue());
        Assert.Empty(draws);
        var expected = reference.GetProperty("decoded").EnumerateArray().Select(v => v.GetDouble()).ToArray();
        Assert.Equal(expected.Length, decoded.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - decoded[i]) <= 1e-9, $"sample {i}: {expected[i]} vs {decoded[i]}");
    }
}
