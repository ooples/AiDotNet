using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Audio.Foundations;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>SenseVoice-Small's SAN-M encoder and FunASR's front end against the release
/// (<c>tools/reference-data/sensevoice_encoder_reference.py</c>).</summary>
public class SenseVoiceEncoderReferenceTests
{
    [Fact(Timeout = 120000)]
    public async Task Encoder_MatchesTheRelease()
    {
        await Task.Yield();
        const string name = "sensevoice_encoder_reference.json";
        string path = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", name);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(path) && dir is not null; dir = dir.Parent)
            path = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", name);
        var f = JObject.Parse(File.ReadAllText(path));
        double[] Doubles(JToken t) => t.Select(v => (double)v).ToArray();
        var state = ((JObject)f["state"]!).Properties().ToDictionary(p => p.Name, p => Doubles(p.Value), StringComparer.Ordinal);
        var encoder = new SanmEncoder<double>(new CpuEngine(), new List<LayerBase<double>>(),
            new SanmEncoderConfiguration(Mels: 8, Width: 16, Heads: 2, FeedForward: 32, Blocks: 3, TpBlocks: 2), new Random(0));
        var used = new HashSet<string>(StringComparer.Ordinal);
        encoder.LoadTorchWeights(n =>
        {
            used.Add(n);
            return state.TryGetValue(n, out var v) ? v : null;
        }, "");
        Assert.Empty(state.Keys.Where(k => !used.Contains(k)));
        encoder.Normalization = (Doubles(f["shift"]!), Doubles(f["scale"]!));
        var audio = Doubles(f["audio"]!);
        var features = encoder.Features(new Tensor<double>(new[] { audio.Length }, new Vector<double>(audio)));
        Close(Doubles(f["features"]!), features.ToArray(), "features", 1e-9);
        var output = encoder.Forward(features);
        Close(Doubles(f["output"]!), output.ToArray(), "output", 1e-9);
    }

    private static void Close(double[] expected, double[] actual, string what, double tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance * Math.Max(1, Math.Abs(expected[i])), $"{what}[{i}]: {expected[i]:R} vs {actual[i]:R}");
    }
}
