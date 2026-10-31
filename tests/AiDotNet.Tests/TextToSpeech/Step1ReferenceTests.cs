using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>Step-1 (LlamaNetwork with √-distance ALiBi) against the release's <c>modeling_step1.py</c>
/// (<c>tools/reference-data/step1_reference.py</c>): full-sequence logits and cached incremental steps.</summary>
public class Step1ReferenceTests
{
    [Fact(Timeout = 120000)]
    public async Task Step1_MatchesTheRelease()
    {
        await Task.Yield();
        const string name = "step1_reference.json";
        string path = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", name);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(path) && dir is not null; dir = dir.Parent)
            path = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", name);
        var f = JObject.Parse(File.ReadAllText(path));
        double[] Doubles(JToken t) => t.Select(v => (double)v).ToArray();
        var state = ((JObject)f["state"]!).Properties().ToDictionary(p => p.Name, p => Doubles(p.Value), StringComparer.Ordinal);
        var engine = new CpuEngine();
        var network = new LlamaNetwork<double>(engine, new List<LayerBase<double>>(), new LlamaConfiguration
        {
            VocabSize = 50, Hidden = 24, Layers = 2, Heads = 6, KeyValueHeads = 2, Intermediate = 40, RmsEpsilon = 1e-5, SqrtAlibi = true,
        });
        var used = new HashSet<string>(StringComparer.Ordinal);
        network.LoadTorchWeights((n, _) =>
        {
            used.Add(n);
            return state[n];
        });
        Assert.Empty(state.Keys.Where(k => !used.Contains(k)));
        var ids = f["ids"]!.Select(v => (int)v).ToArray();
        int[,] Grid(int start, int end)
        {
            var g = new int[1, end - start];
            for (int i = start; i < end; i++) g[0, i - start] = ids[i];
            return g;
        }
        var (_, logits) = network.Forward(network.Embed(Grid(0, ids.Length)));
        Close(Doubles(f["logits"]!), logits!.ToArray(), "logits");
        var cache = new LlamaCache<double>();
        var spans = new[] { (0, 4), (4, 5), (5, 6), (6, 7) };
        var steps = f["steps"]!.ToArray();
        for (int s = 0; s < spans.Length; s++)
        {
            var (_, stepLogits) = network.Forward(network.Embed(Grid(spans[s].Item1, spans[s].Item2)), null, null, cache, lastLogitsOnly: false);
            Close(Doubles(steps[s]), stepLogits!.ToArray(), $"step {s}");
        }
    }

    private static void Close(double[] expected, double[] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-10 * Math.Max(1, Math.Abs(expected[i])), $"{what}[{i}]: {expected[i]:R} vs {actual[i]:R}");
    }
}
