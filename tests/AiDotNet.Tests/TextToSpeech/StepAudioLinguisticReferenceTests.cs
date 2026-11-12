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

/// <summary>Step-Audio's linguistic tokenizer: the streaming Paraformer encoder (FunASR's WavFrontendOnline and
/// SANMEncoderChunkOpt, chunk [0, 4, 5], look-back 4) and the k-means codes, against the release
/// (<c>tools/reference-data/stepaudio_linguistic_reference.py</c>).</summary>
public class StepAudioLinguisticReferenceTests
{
    [Fact(Timeout = 120000)]
    public async Task StreamingEncoder_MatchesTheRelease()
    {
        await Task.Yield();
        const string name = "stepaudio_linguistic_reference.json";
        string path = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", name);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(path) && dir is not null; dir = dir.Parent)
            path = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", name);
        var f = JObject.Parse(File.ReadAllText(path));
        double[] Doubles(JToken t) => t.Select(v => (double)v).ToArray();
        var state = ((JObject)f["state"]!).Properties().ToDictionary(p => p.Name, p => Doubles(p.Value), StringComparer.Ordinal);
        var encoder = new StreamingParaformerEncoder<double>(new CpuEngine(), new List<LayerBase<double>>(),
            new SanmEncoderConfiguration(Mels: 8, Width: 16, Heads: 2, FeedForward: 32, Blocks: 3, TpBlocks: 0, NormEpsilon: 1e-12), new Random(0))
        {
            Dither = 0,
        };
        var used = new HashSet<string>(StringComparer.Ordinal);
        encoder.Encoder.LoadTorchWeights(n =>
        {
            used.Add(n);
            return state.TryGetValue(n, out var v) ? v : null;
        }, "");
        Assert.Empty(state.Keys.Where(k => !used.Contains(k)));
        encoder.Encoder.Normalization = (Doubles(f["shift"]!), Doubles(f["scale"]!));
        var encoded = encoder.Encode(Doubles(f["audio"]!));
        var expected = Doubles(f["encoded"]!);
        var actual = encoded.ToArray();
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-9 * Math.Max(1, Math.Abs(expected[i])), $"encoded[{i}]: {expected[i]:R} vs {actual[i]:R}");
        var means = Doubles(f["means"]!);
        int width = encoded.Shape[1], centroids = means.Length / width;
        var codes = Enumerable.Range(0, encoded.Shape[0]).Select(t => Enumerable.Range(0, centroids)
            .OrderBy(k => Enumerable.Range(0, width).Sum(j => Math.Pow(encoded[t, j] - means[k * width + j], 2))).First()).ToArray();
        Assert.Equal(f["codes"]!.Select(v => (int)v).ToArray(), codes);
    }
}
