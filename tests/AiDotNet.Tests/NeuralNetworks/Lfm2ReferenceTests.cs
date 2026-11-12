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

namespace AiDotNet.Tests.NeuralNetworks;

/// <summary>
/// LFM2 against transformers 4.57.1's Lfm2ForCausalLM and peft's rank-stabilized LoRA
/// (tools/reference-data/lfm2_reference.py): a tiny model of convolution and attention layers, its logits, greedy
/// generation through the hybrid cache (keys and values, and the convolutions' states), and with LoRA the logits, the
/// causal LM loss and the adapters' gradients.
/// </summary>
public class Lfm2ReferenceTests
{
    private static JsonElement Fixture()
    {
        const string name = "lfm2_reference.json";
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

    private static Lfm2Configuration Configuration(JsonElement f, bool lora)
    {
        var c = f.GetProperty("config");
        var l = f.GetProperty("lora_config");
        return new Lfm2Configuration
        {
            VocabSize = c.GetProperty("vocab_size").GetInt32(), Hidden = c.GetProperty("hidden_size").GetInt32(),
            AttentionLayers = c.GetProperty("layer_types").EnumerateArray().Select(t => t.GetString() == "full_attention").ToArray(),
            Heads = c.GetProperty("num_attention_heads").GetInt32(), KeyValueHeads = c.GetProperty("num_key_value_heads").GetInt32(),
            Intermediate = c.GetProperty("intermediate_size").GetInt32(), FeedForwardMultipleOf = c.GetProperty("block_multiple_of").GetInt32(),
            ConvKernel = c.GetProperty("conv_L_cache").GetInt32(), NormEpsilon = c.GetProperty("norm_eps").GetDouble(),
            RopeTheta = c.GetProperty("rope_theta").GetDouble(),
            LoraRank = lora ? l.GetProperty("r").GetInt32() : 0, LoraAlpha = l.GetProperty("lora_alpha").GetDouble(), LoraDropout = 0,
            LoraTargets = l.GetProperty("target_modules").EnumerateArray().Select(t => t.GetString()!).ToArray(),
        };
    }

    private static Lfm2Network<double> Network(JsonElement f, bool lora, List<LayerBase<double>> layers)
    {
        var net = new Lfm2Network<double>(AiDotNetEngine.Current, layers, Configuration(f, lora));
        var state = f.GetProperty("state");
        var adapters = f.GetProperty("lora");
        var used = new HashSet<string>();
        net.LoadTorchWeights((name, shape) =>
        {
            var v = Values(name.Contains("lora_") ? adapters.GetProperty(name) : state.GetProperty(name));
            Assert.Equal(shape.Aggregate(1, (a, b) => a * b), v.Length);
            used.Add(name);
            return v;
        });
        Assert.Empty(state.EnumerateObject().Select(p => p.Name).Where(n => !used.Contains(n)));
        if (lora) Assert.Empty(adapters.EnumerateObject().Select(p => p.Name).Where(n => !used.Contains(n)));
        return net;
    }

    private static void Close(double[] expected, Tensor<double> actual, string what, double tolerance = 1e-10)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance * Math.Max(1, Math.Abs(expected[i])), $"{what}[{i}]: {expected[i]} vs {actual[i]}");
    }

    private static int[,] Row(IReadOnlyList<int> ids)
    {
        var row = new int[1, ids.Count];
        for (int i = 0; i < ids.Count; i++) row[0, i] = ids[i];
        return row;
    }

    [Fact(Timeout = 300000)]
    public async Task LogitsAndCachedGreedyGeneration_MatchTheReference()
    {
        await Task.Yield();
        var f = Fixture();
        var net = Network(f, lora: false, new List<LayerBase<double>>());
        var ids = f.GetProperty("ids").EnumerateArray().Select(v => v.GetInt32()).ToList();
        using var _ = new NoGradScope<double>();
        Close(Values(f.GetProperty("logits")), net.Forward(net.Embed(Row(ids))).Logits, "logits");

        var cache = new Lfm2Cache<double>();
        var logits = net.Forward(net.Embed(Row(ids)), cache, lastLogitsOnly: true).Logits;
        var generated = new List<int>();
        for (int step = 0; step < 6; step++)
        {
            int next = 0;
            for (int v = 1; v < logits.Length; v++) if (logits[v] > logits[next]) next = v;
            generated.Add(next);
            logits = net.Forward(net.Embed(Row(new[] { next })), cache, lastLogitsOnly: true).Logits;
        }
        Assert.Equal(f.GetProperty("generated").EnumerateArray().Select(v => v.GetInt32()), generated);
        Assert.Equal(ids.Count + 6, cache.Length);
    }

    [Fact(Timeout = 300000)]
    public async Task LoraLogitsLossAndGradients_MatchPeft()
    {
        await Task.Yield();
        var f = Fixture();
        var layers = new List<LayerBase<double>>();
        var net = Network(f, lora: true, layers);
        var ids = f.GetProperty("ids").EnumerateArray().Select(v => v.GetInt32()).ToArray();
        var labels = Row(ids);
        var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(layers);
        Dictionary<Tensor<double>, Tensor<double>> grads;
        Tensor<double> logits, loss;
        using (var tape = new GradientTape<double>())
        {
            logits = net.Forward(net.Embed(labels)).Logits;
            loss = net.CausalLmLoss(logits, labels);
            grads = tape.ComputeGradients(loss, sources: parameters);
        }
        Close(Values(f.GetProperty("lora_logits")), logits, "LoRA logits");
        double reference = f.GetProperty("loss").GetDouble();
        Assert.True(Math.Abs(reference - loss[0]) <= 1e-12 * Math.Abs(reference), $"loss {reference} vs {loss[0]}");

        // The torch weights are [out, in]; the layers hold [in, out].
        void Gradient(string name, BiasFreeLinearLayer<double> layer)
        {
            var expected = Values(f.GetProperty("gradients").GetProperty(name));
            var g = grads[layer.Weights];
            int inputs = layer.Weights.Shape[0], outputs = layer.Weights.Shape[1];
            for (int o = 0; o < outputs; o++)
                for (int i = 0; i < inputs; i++)
                {
                    double e = expected[o * inputs + i], a = g[i * outputs + o];
                    Assert.True(Math.Abs(e - a) <= 1e-9 * Math.Max(1e-3, Math.Abs(e)), $"{name}[{o},{i}]: {e} vs {a}");
                }
        }
        Gradient("model.layers.0.conv.in_proj.lora_A.weight", net.Layers[0].ConvIn!.LoraA!);
        Gradient("model.layers.1.self_attn.q_proj.lora_B.weight", net.Layers[1].Query!.LoraB!);
    }
}
