using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using AiDotNet.Agentic.Models.Local;
using AiDotNet.ModelLoading.Pretrained;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.ModelLoading;

/// <summary>
/// Loading an MoE checkpoint from safetensors decodes each weight straight into one reused buffer. Before, every
/// expert matrix was read as a double[] (twice its fp32 size), converted, transposed into a third array and
/// concatenated with its bias into a fourth: about 20 bytes of short-lived large-object-heap garbage per parameter,
/// so a bf16 OLMoE-1B-7B load left gigabytes of dead arrays behind and failed under a GC heap limit near the
/// model's size.
/// </summary>
[Collection("WeightStreaming-Singleton")]
public sealed class MoELoadStagingTests : IDisposable
{
    public MoELoadStagingTests() => NeuralNetworkBase<float>.ResetWeightStreamingForTests();
    public void Dispose() => NeuralNetworkBase<float>.ResetWeightStreamingForTests();

    private const int Hidden = 64, Ffn = 128, Vocab = 16, Heads = 4, HeadDim = 16, Experts = 8, Layers = 2;

    private static HuggingFaceConfig Config() => HuggingFaceConfig.Parse(@"{ ""architectures"": [""MixtralForCausalLM""],
        ""model_type"": ""mixtral"", ""hidden_size"": " + Hidden + @", ""intermediate_size"": " + Ffn + @",
        ""num_hidden_layers"": " + Layers + @", ""num_attention_heads"": " + Heads + @", ""num_key_value_heads"": " + Heads + @",
        ""vocab_size"": " + Vocab + @", ""num_local_experts"": " + Experts + @", ""num_experts_per_tok"": 2,
        ""tie_word_embeddings"": false }");

    // Values exact in bf16, and distinct across rows and columns so a wrong transpose changes the logits.
    private static float[] Fill(int n, int seed)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (((i * 3 + seed) % 11) - 5) / 64f;
        return a;
    }

    private static List<(string Name, int[] Shape, float[] Data)> Tensors()
    {
        var t = new List<(string, int[], float[])>
        {
            ("model.embed_tokens.weight", new[] { Vocab, Hidden }, Fill(Vocab * Hidden, 1)),
            ("model.norm.weight", new[] { Hidden }, Fill(Hidden, 2)),
            ("lm_head.weight", new[] { Vocab, Hidden }, Fill(Vocab * Hidden, 3)),
        };
        int q = Heads * HeadDim;
        for (int l = 0; l < Layers; l++)
        {
            string p = $"model.layers.{l}.";
            t.Add((p + "input_layernorm.weight", new[] { Hidden }, Fill(Hidden, 10 + l)));
            t.Add((p + "post_attention_layernorm.weight", new[] { Hidden }, Fill(Hidden, 12 + l)));
            t.Add((p + "self_attn.q_proj.weight", new[] { q, Hidden }, Fill(q * Hidden, 14 + l)));
            t.Add((p + "self_attn.k_proj.weight", new[] { q, Hidden }, Fill(q * Hidden, 16 + l)));
            t.Add((p + "self_attn.v_proj.weight", new[] { q, Hidden }, Fill(q * Hidden, 18 + l)));
            t.Add((p + "self_attn.o_proj.weight", new[] { Hidden, q }, Fill(Hidden * q, 20 + l)));
            t.Add((p + "block_sparse_moe.gate.weight", new[] { Experts, Hidden }, Fill(Experts * Hidden, 22 + l)));
            for (int e = 0; e < Experts; e++)
            {
                string ep = p + $"block_sparse_moe.experts.{e}.";
                t.Add((ep + "w1.weight", new[] { Ffn, Hidden }, Fill(Ffn * Hidden, 30 + e + l)));
                t.Add((ep + "w3.weight", new[] { Ffn, Hidden }, Fill(Ffn * Hidden, 40 + e + l)));
                t.Add((ep + "w2.weight", new[] { Hidden, Ffn }, Fill(Hidden * Ffn, 50 + e + l)));
            }
        }
        return t;
    }

    // A bf16 safetensors file, in memory.
    private static SafetensorsFile Bf16File()
    {
        var header = new JObject();
        long offset = 0;
        var payload = new MemoryStream();
        foreach (var (name, shape, data) in Tensors())
        {
            header[name] = new JObject
            {
                ["dtype"] = "BF16",
                ["shape"] = new JArray(shape.Select(d => (object)d).ToArray()),
                ["data_offsets"] = new JArray(offset, offset + data.Length * 2L),
            };
            foreach (var v in data)
            {
                uint bits = BitConverter.ToUInt32(BitConverter.GetBytes(v), 0);
                payload.WriteByte((byte)(bits >> 16));
                payload.WriteByte((byte)(bits >> 24));
            }
            offset += data.Length * 2L;
        }
        byte[] headerBytes = Encoding.UTF8.GetBytes(header.ToString(Newtonsoft.Json.Formatting.None));
        var file = new MemoryStream();
        file.Write(BitConverter.GetBytes((long)headerBytes.Length), 0, 8);
        file.Write(headerBytes, 0, headerBytes.Length);
        payload.Position = 0;
        payload.CopyTo(file);
        return SafetensorsReader.Read(file.ToArray());
    }

    /// <summary>The same file seen only through <see cref="INamedTensorSource.ReadAsDouble"/>: whole-tensor reads.</summary>
    private sealed class WholeTensorSource : INamedTensorSource
    {
        private readonly INamedTensorSource _inner;
        public WholeTensorSource(INamedTensorSource inner) => _inner = inner;
        public IReadOnlyCollection<string> TensorNames => _inner.TensorNames;
        public double[] ReadAsDouble(string name) => _inner.ReadAsDouble(name);
    }

    private static Tensor<float> Ids()
    {
        var ids = new Tensor<float>(new[] { 1, 4 });
        for (int i = 0; i < 4; i++) ids[0, i] = i + 1;
        return ids;
    }

    /// <summary>The streamed decode writes the same transposed weights as the whole-tensor read: identical logits.</summary>
    [Fact]
    public void Decoding_through_the_staging_buffer_gives_the_whole_tensor_reads_logits()
    {
        var file = Bf16File();
        var reference = MoEModelBuilder<float>.Build(Config(), new WholeTensorSource(file));
        reference.SetTrainingMode(false);
        var expected = reference.Predict(Ids()).ToVector();

        var staged = MoEModelBuilder<float>.Build(Config(), file);
        staged.SetTrainingMode(false);
        var actual = staged.Predict(Ids()).ToVector();

        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(expected[i] == actual[i], $"[{i}] {actual[i]} vs {expected[i]}");
    }

#if NET
    /// <summary>
    /// The streamed bf16 load from safetensors allocates at least 8 bytes per parameter less than the same load through
    /// whole-tensor reads (the double[] each weight was read into, twice its fp32 size, is what the staging removes;
    /// the converted copy goes too). Before the change both arms took the whole-tensor path and allocated the same.
    /// The rest of either load (building the network, each block's fp32 materialization, the warmup forward) is the
    /// same in both arms, so the difference is the per-weight staging alone.
    /// </summary>
    [Fact]
    public void A_streamed_safetensors_MoE_load_allocates_no_whole_tensor_copies()
    {
        var file = Bf16File();
        long fp32Bytes = 4L * Tensors().Sum(t => (long)t.Data.Length);

        long before = GC.GetAllocatedBytesForCurrentThread();
        var net = MoEModelBuilder<float>.BuildStreamed(Config(), file, StreamingStoreDtype.Bf16, warmupSequenceLength: 1);
        long staged = GC.GetAllocatedBytesForCurrentThread() - before;
        GC.KeepAlive(net);
        NeuralNetworkBase<float>.ResetWeightStreamingForTests();

        before = GC.GetAllocatedBytesForCurrentThread();
        var wholeNet = MoEModelBuilder<float>.BuildStreamed(Config(), new WholeTensorSource(file), StreamingStoreDtype.Bf16, warmupSequenceLength: 1);
        long whole = GC.GetAllocatedBytesForCurrentThread() - before;
        GC.KeepAlive(wholeNet);

        Assert.True(whole - staged >= 2 * fp32Bytes,
            $"staged load allocated {staged} bytes, whole-tensor reads {whole}, for a {fp32Bytes}-byte fp32 model: "
            + $"the staging saved {whole - staged} bytes, under 8 per parameter");
    }
#endif
}
