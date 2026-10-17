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
/// The LLaMA decoder (GLaMM's language model) against Hugging Face's LlamaForCausalLM (tools/reference-data/llama_reference.py):
/// grouped-query attention with rotary embeddings, SwiGLU, RMSNorm and a right-padded batch; the final hidden states, the
/// logits and the causal language-model loss, before and after one SGD step.
/// </summary>
public class LlamaReferenceTests
{
    private static JsonElement Fixture()
    {
        const string name = "llama_reference.json";
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
    public async Task Decoder_MatchesTheReference_AndAfterOneSgdStep()
    {
        await Task.Yield();
        RunCase(Fixture(), 0);
    }

    /// <summary>The same with peft LoRA on the query and value projections (rank 4, alpha 8), as GLaMM fine-tunes Vicuna.</summary>
    [Fact(Timeout = 300000)]
    public async Task LoraDecoder_MatchesPeft_AndAfterOneSgdStep()
    {
        await Task.Yield();
        RunCase(Fixture().GetProperty("lora"), 4, Fixture());
    }

    private static void RunCase(JsonElement k, int loraRank, JsonElement? inputs = null)
    {
        var engine = AiDotNetEngine.Current;
        var source = inputs ?? k;
        var layers = new List<LayerBase<double>>();
        var net = new LlamaNetwork<double>(engine, layers, new LlamaConfiguration
        {
            VocabSize = 128, Hidden = 32, Layers = 2, Heads = 4, KeyValueHeads = 2, Intermediate = 64,
            LoraRank = loraRank, LoraAlpha = 8, LoraDropout = 0,
        });
        var state = k.GetProperty("state");
        net.LoadTorchWeights((name, shape) =>
        {
            var v = Values(state.GetProperty(name));
            Assert.Equal(shape.Aggregate(1, (a, b) => a * b), v.Length);
            return v;
        });
        var idRows = source.GetProperty("ids").EnumerateArray().Select(r => r.EnumerateArray().Select(v => v.GetInt32()).ToArray()).ToArray();
        var maskRows = source.GetProperty("mask").EnumerateArray().Select(r => r.EnumerateArray().Select(v => v.GetInt32() != 0).ToArray()).ToArray();
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

        var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(layers);
        Tensor<double> hidden, logits, loss;
        Dictionary<Tensor<double>, Tensor<double>> grads;
        using (var tape = new GradientTape<double>())
        {
            (hidden, logits) = net.Forward(net.Embed(ids), mask);
            loss = net.CausalLmLoss(logits, labels);
            grads = tape.ComputeGradients(loss, sources: parameters);
        }
        void Close(double[] expected, Tensor<double> actual, string what)
        {
            Assert.Equal(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
                Assert.True(Math.Abs(expected[i] - actual[i]) < 1e-10, $"{what}[{i}]: {expected[i]} vs {actual[i]}");
        }
        // The padded positions' outputs depend only on the unpadded keys, as in the reference.
        Close(Values(k.GetProperty("hidden")), hidden, "hidden");
        Close(Values(k.GetProperty("logits")), logits, "logits");
        Assert.True(Math.Abs(k.GetProperty("loss").GetDouble() - loss[0]) < 1e-10, $"{k.GetProperty("loss").GetDouble()} vs {loss[0]}");
        foreach (var p in parameters)
        {
            if (!grads.TryGetValue(p, out var g)) continue;
            for (int i = 0; i < p.Length; i++) p[i] -= 0.01 * g[i];
            engine.InvalidatePersistentTensor(p);
        }
        double after;
        using (new NoGradScope<double>())
            after = net.CausalLmLoss(net.Forward(net.Embed(ids), mask).Logits, labels)[0];
        Assert.True(Math.Abs(k.GetProperty("loss_after_sgd").GetDouble() - after) < 1e-10, $"{k.GetProperty("loss_after_sgd").GetDouble()} vs {after}");
    }
}
