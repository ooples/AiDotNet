using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.VisionLanguage;

/// <summary>
/// The LLaMA decoder in its Llama 3 form (rotary scaling, tied head) and its Qwen3 form (a separate head width, per-head
/// query and key RMSNorm, tied head) against Hugging Face's LlamaForCausalLM and Qwen3ForCausalLM
/// (tools/reference-data/llama3_qwen3_reference.py), including generation one position at a time through the KV cache.
/// </summary>
public class Llama3Qwen3ReferenceTests
{
    private static JsonElement Fixture()
    {
        const string name = "llama3_qwen3_reference.json";
        string relative = Path.Combine("NeuralNetworks", "ReferenceData", name);
        string path = Path.Combine(AppContext.BaseDirectory, relative);
        if (!File.Exists(path))
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", relative))) dir = dir.Parent;
            path = Path.Combine(dir?.FullName ?? throw new FileNotFoundException(name), "tests", "AiDotNet.Tests", relative);
        }
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement;
    }

    private static double[] Values(JsonElement e) => e.EnumerateArray().Select(v => v.GetDouble()).ToArray();

    [Fact(Timeout = 300000)]
    public async Task Llama3_MatchesTheReference()
    {
        await Task.Yield();
        RunCase("llama3", new LlamaConfiguration
        {
            VocabSize = 128, Hidden = 32, Layers = 2, Heads = 4, KeyValueHeads = 2, Intermediate = 64, RopeTheta = 500000,
            RopeScalingFactor = 32, RopeLowFrequencyFactor = 1, RopeHighFrequencyFactor = 4, RopeOriginalMaxPositions = 16,
            TieWordEmbeddings = true,
        });
    }

    [Fact(Timeout = 300000)]
    public async Task Qwen3_MatchesTheReference()
    {
        await Task.Yield();
        RunCase("qwen3", new LlamaConfiguration
        {
            VocabSize = 128, Hidden = 32, Layers = 2, Heads = 4, KeyValueHeads = 2, HeadDim = 16, Intermediate = 64, RopeTheta = 1000000,
            RmsEpsilon = 1e-6, QueryKeyNorm = true, TieWordEmbeddings = true,
        });
    }

    private static void RunCase(string kind, LlamaConfiguration configuration)
    {
        var fixture = Fixture();
        var k = fixture.GetProperty(kind);
        var layers = new List<LayerBase<double>>();
        var net = new LlamaNetwork<double>(AiDotNetEngine.Current, layers, configuration);
        var state = k.GetProperty("state");
        var used = new HashSet<string>();
        net.LoadTorchWeights((name, shape) =>
        {
            var v = Values(state.GetProperty(name));
            Assert.Equal(shape.Aggregate(1, (a, b) => a * b), v.Length);
            used.Add(name);
            return v;
        });
        Assert.Empty(state.EnumerateObject().Select(p => p.Name).Where(n => !used.Contains(n)));
        var idRows = fixture.GetProperty("ids").EnumerateArray().Select(r => r.EnumerateArray().Select(v => v.GetInt32()).ToArray()).ToArray();
        var maskRows = fixture.GetProperty("mask").EnumerateArray().Select(r => r.EnumerateArray().Select(v => v.GetInt32() != 0).ToArray()).ToArray();
        int b = idRows.Length, l = idRows[0].Length;
        var ids = new int[b, l];
        var mask = new bool[b, l];
        var labels = new int[b, l];
        for (int i = 0; i < b; i++)
            for (int j = 0; j < l; j++)
            {
                ids[i, j] = idRows[i][j];
                mask[i, j] = maskRows[i][j];
                labels[i, j] = mask[i, j] ? ids[i, j] : -100;
            }
        void Close(double[] expected, Tensor<double> actual, string what)
        {
            Assert.Equal(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
                Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-10 * Math.Max(1, Math.Abs(expected[i])), $"{what}[{i}]: {expected[i]} vs {actual[i]}");
        }
        using (new NoGradScope<double>())
        {
            var (hidden, logits) = net.ForwardWithLogits(net.Embed(ids), mask);
            Close(Values(k.GetProperty("hidden")), hidden, "hidden");
            Close(Values(k.GetProperty("logits")), logits, "logits");
            double loss = net.CausalLmLoss(logits, labels)[0];
            Assert.True(Math.Abs(k.GetProperty("loss").GetDouble() - loss) < 1e-10, $"loss {k.GetProperty("loss").GetDouble()} vs {loss}");

            var cache = new LlamaCache<double>();
            Tensor<double>? last = null;
            for (int t = 0; t < l; t++)
                last = net.Forward(net.Embed(new[,] { { ids[0, t] } }), null, null, cache, lastLogitsOnly: true).Logits;
            Assert.Equal(l, cache.Length);
            Close(Values(k.GetProperty("cached_last_logits")), last!, "cached last logits");
        }
    }
}
