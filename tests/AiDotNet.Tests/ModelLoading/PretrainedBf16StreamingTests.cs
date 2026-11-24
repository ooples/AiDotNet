using System;
using System.Collections.Generic;
using AiDotNet.Agentic.Models.Local;
using AiDotNet.Deployment.Configuration;
using AiDotNet.ModelLoading.Pretrained;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelLoading;

/// <summary>
/// A pretrained checkpoint loaded through a model builder (outside <c>AiModelBuilder</c>) can be put into a bf16
/// streaming store for inference: a bf16 checkpoint then needs half its fp32 size, with no loss. Before, a network
/// built this way had no public way to stream, and a bf16 store on the CPU engine threw on the first forward
/// (FusedLinear read the bias through the write-intent accessor). The weights here are exact in bf16, so the
/// streamed logits equal the resident ones.
/// </summary>
[Collection("WeightStreaming-Singleton")]
public sealed class PretrainedBf16StreamingTests : IDisposable
{
    public PretrainedBf16StreamingTests() => NeuralNetworkBase<double>.ResetWeightStreamingForTests();
    public void Dispose() => NeuralNetworkBase<double>.ResetWeightStreamingForTests();

    private static double[] Fill(int n, int seed)
    {
        var a = new double[n];
        for (int i = 0; i < n; i++) a[i] = (((i + seed) % 7) - 3) / 64.0;   // exact in bf16
        return a;
    }

    private sealed class InMemorySource : INamedTensorSource
    {
        private readonly Dictionary<string, double[]> _tensors;
        public InMemorySource(Dictionary<string, double[]> tensors) => _tensors = tensors;
        public IReadOnlyCollection<string> TensorNames => _tensors.Keys;
        public double[] ReadAsDouble(string name) => _tensors[name];
        public void Remove(string name) => _tensors.Remove(name);
    }

    [Fact]
    public void A_loaded_MoE_checkpoint_streams_in_bf16_and_gives_the_resident_logits()
    {
        var config = HuggingFaceConfig.Parse(@"{ ""architectures"": [""MixtralForCausalLM""],
            ""model_type"": ""mixtral"", ""hidden_size"": 8, ""intermediate_size"": 16,
            ""num_hidden_layers"": 1, ""num_attention_heads"": 2, ""num_key_value_heads"": 2,
            ""vocab_size"": 6, ""num_local_experts"": 4, ""num_experts_per_tok"": 2,
            ""tie_word_embeddings"": false }");
        int h = 8, ffn = 16, vocab = 6, heads = 2, headDim = 4, E = 4;
        var t = new Dictionary<string, double[]>
        {
            ["model.embed_tokens.weight"] = Fill(vocab * h, 50),
            ["model.layers.0.input_layernorm.weight"] = Fill(h, 51),
            ["model.layers.0.post_attention_layernorm.weight"] = Fill(h, 52),
            ["model.layers.0.self_attn.q_proj.weight"] = Fill(heads * headDim * h, 53),
            ["model.layers.0.self_attn.k_proj.weight"] = Fill(heads * headDim * h, 54),
            ["model.layers.0.self_attn.v_proj.weight"] = Fill(heads * headDim * h, 55),
            ["model.layers.0.self_attn.o_proj.weight"] = Fill(h * heads * headDim, 56),
            ["model.layers.0.block_sparse_moe.gate.weight"] = Fill(E * h, 57),
            ["model.norm.weight"] = Fill(h, 90),
            ["lm_head.weight"] = Fill(vocab * h, 91),
        };
        for (int e = 0; e < E; e++)
        {
            string ep = $"model.layers.0.block_sparse_moe.experts.{e}.";
            t[ep + "w1.weight"] = Fill(ffn * h, 60 + e);
            t[ep + "w3.weight"] = Fill(ffn * h, 70 + e);
            t[ep + "w2.weight"] = Fill(h * ffn, 80 + e);
        }
        var net = MoEModelBuilder<double>.Build(config, new InMemorySource(t));
        net.SetTrainingMode(false);
        var ids = new Tensor<double>(new[] { 1, 4 });
        for (int i = 0; i < 4; i++) ids[0, i] = i + 1;
        var resident = net.Predict(ids).ToVector();

        net.ConfigureWeightStreaming(new WeightStreamingConfig { Enabled = true, StoreDtype = StreamingStoreDtype.Bf16 });
        var streamed = net.Predict(ids).ToVector();

        var report = WeightRegistry.GetStreamingReport();
        Assert.True(report.RegisteredEntryCount > 0, "no weights registered with the streaming store");
        // bf16: 2 bytes per parameter (fp32 would be 4).
        Assert.True(report.ResidentBytes <= 2L * net.ParameterCount,
            $"resident {report.ResidentBytes} B for {net.ParameterCount} parameters");
        Assert.Equal(resident.Length, streamed.Length);
        for (int i = 0; i < resident.Length; i++)
            Assert.True(Math.Abs(resident[i] - streamed[i]) <= 1e-12, $"[{i}] {streamed[i]} vs {resident[i]}");
    }
}
