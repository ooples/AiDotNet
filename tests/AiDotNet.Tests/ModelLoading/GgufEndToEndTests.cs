using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using AiDotNet.ModelLoading.Pretrained;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelLoading
{
    /// <summary>
    /// End-to-end tests for GGUF import: synthesize a tiny llama.cpp-format GGUF in memory, then load it
    /// through the full facade path (<see cref="PretrainedLoader{T}"/> / <see cref="GgufModelSource"/>) and
    /// run a forward pass. Exercises metadata→config, GGUF→Hugging Face name remapping, tie-detection, and a
    /// non-llama-named (qwen2) architecture.
    /// </summary>
    public class GgufEndToEndTests
    {
        [Fact]
        public void Gguf_EndToEnd_Llama_TiedHead_BuildsAndForwards()
        {
            string path = WriteTempGguf("llama", includeOutput: false);
            try
            {
                using (var src = GgufModelSource.Open(path))
                {
                    Assert.Equal(8, src.Config.HiddenSize);
                    Assert.Equal(1, src.Config.NumHiddenLayers);
                    Assert.Equal(2, src.Config.NumAttentionHeads);
                    Assert.Equal(1, src.Config.NumKeyValueHeads);
                    Assert.Equal(6, src.Config.VocabSize);
                    Assert.True(src.Config.TieWordEmbeddings); // no output.weight -> tied
                    Assert.Contains("model.embed_tokens.weight", src.TensorNames);
                    Assert.Contains("model.layers.0.self_attn.q_proj.weight", src.TensorNames);
                    Assert.DoesNotContain("lm_head.weight", src.TensorNames); // tied
                }

                // Full facade resolution: GGUF -> config -> registry -> builder -> weight-loaded decoder.
                var model = PretrainedLoader<double>.Load(PretrainedSource.Gguf(path));
                var net = Assert.IsType<NeuralNetwork<double>>(model);
                Assert.Equal(4, net.Layers.Count); // embed + 1 block + final norm + lm head

                var tokens = new Tensor<double>(new[] { 1, 3 });
                tokens[0, 0] = 1; tokens[0, 1] = 4; tokens[0, 2] = 2;
                var logits = net.Predict(tokens);
                Assert.Equal(new[] { 1, 3, 6 }, logits.Shape);
            }
            finally { File.Delete(path); }
        }

        [Fact]
        public void Gguf_EndToEnd_UntiedHead_LoadsSeparateLmHead()
        {
            string path = WriteTempGguf("llama", includeOutput: true);
            try
            {
                using var src = GgufModelSource.Open(path);
                Assert.False(src.Config.TieWordEmbeddings);
                Assert.Contains("lm_head.weight", src.TensorNames);

                var model = PretrainedLoader<double>.Load(PretrainedSource.Gguf(path));
                Assert.NotNull(model);
            }
            finally { File.Delete(path); }
        }

        [Fact]
        public void Gguf_EndToEnd_NonLlamaArch_Qwen2_Resolves()
        {
            // A GGUF whose general.architecture is "qwen2" (same llama-family layer stack) loads through the
            // SAME registry + builder — proving non-llama-named families are supported.
            string path = WriteTempGguf("qwen2", includeOutput: false);
            try
            {
                using var src = GgufModelSource.Open(path);
                Assert.Equal("qwen2", src.Config.ModelType);

                var model = PretrainedLoader<double>.Load(PretrainedSource.Gguf(path));
                var net = Assert.IsType<NeuralNetwork<double>>(model);
                var tokens = new Tensor<double>(new[] { 1, 2 });
                tokens[0, 0] = 0; tokens[0, 1] = 3;
                Assert.Equal(new[] { 1, 2, 6 }, net.Predict(tokens).Shape);
            }
            finally { File.Delete(path); }
        }

        // ---- minimal GGUF writer (F32) for a tiny GQA llama-family decoder ----

        private static string WriteTempGguf(string arch, bool includeOutput)
        {
            const int hidden = 8, heads = 2, kvHeads = 1, ffn = 16, vocab = 6, layers = 1;
            int headDim = hidden / heads;      // 4
            int qDim = heads * headDim;        // 8
            int kvDim = kvHeads * headDim;     // 4

            // GGUF tensor: (name, dims as ne[] with ne[0] fastest, F32 data). Data order is the raw sequential
            // bytes GgufFile returns; the builder handles the [out,in]->[in,out] transpose itself.
            var tensors = new List<(string name, long[] dims, float[] data)>
            {
                ("token_embd.weight", new long[] { hidden, vocab }, Seq(vocab * hidden, 1)),
                ("blk.0.attn_norm.weight", new long[] { hidden }, Seq(hidden, 2)),
                ("blk.0.attn_q.weight", new long[] { hidden, qDim }, Seq(qDim * hidden, 3)),
                ("blk.0.attn_k.weight", new long[] { hidden, kvDim }, Seq(kvDim * hidden, 4)),
                ("blk.0.attn_v.weight", new long[] { hidden, kvDim }, Seq(kvDim * hidden, 5)),
                ("blk.0.attn_output.weight", new long[] { qDim, hidden }, Seq(hidden * qDim, 6)),
                ("blk.0.ffn_norm.weight", new long[] { hidden }, Seq(hidden, 7)),
                ("blk.0.ffn_gate.weight", new long[] { hidden, ffn }, Seq(ffn * hidden, 8)),
                ("blk.0.ffn_up.weight", new long[] { hidden, ffn }, Seq(ffn * hidden, 9)),
                ("blk.0.ffn_down.weight", new long[] { ffn, hidden }, Seq(hidden * ffn, 10)),
                ("output_norm.weight", new long[] { hidden }, Seq(hidden, 11)),
            };
            if (includeOutput)
                tensors.Add(("output.weight", new long[] { hidden, vocab }, Seq(vocab * hidden, 12)));

            var meta = new List<(string key, uint type, object val)>
            {
                ("general.architecture", 8u, arch),
                ("general.alignment", 4u, 32u),
                ($"{arch}.embedding_length", 4u, (uint)hidden),
                ($"{arch}.block_count", 4u, (uint)layers),
                ($"{arch}.attention.head_count", 4u, (uint)heads),
                ($"{arch}.attention.head_count_kv", 4u, (uint)kvHeads),
                ($"{arch}.feed_forward_length", 4u, (uint)ffn),
                ($"{arch}.context_length", 4u, 64u),
                ($"{arch}.attention.layer_norm_rms_epsilon", 6u, 1e-5f),
                ($"{arch}.rope.freq_base", 6u, 10000.0f),
                ($"{arch}.vocab_size", 4u, (uint)vocab),
            };

            using var ms = new MemoryStream();
            using (var w = new BinaryWriter(ms, Encoding.UTF8, leaveOpen: true))
            {
                w.Write(0x46554747u);              // "GGUF"
                w.Write(3u);                       // version
                w.Write((ulong)tensors.Count);
                w.Write((ulong)meta.Count);

                foreach (var (key, type, val) in meta)
                {
                    WriteString(w, key);
                    w.Write(type);
                    switch (type)
                    {
                        case 8u: WriteString(w, (string)val); break;
                        case 4u: w.Write((uint)val); break;
                        case 6u: w.Write((float)val); break;
                        default: throw new InvalidOperationException($"unhandled meta type {type}");
                    }
                }

                // Tensor infos with contiguous data-section offsets.
                ulong offset = 0;
                foreach (var (name, dims, data) in tensors)
                {
                    WriteString(w, name);
                    w.Write((uint)dims.Length);
                    foreach (var d in dims) w.Write((ulong)d);
                    w.Write(0u);          // ggml type F32
                    w.Write(offset);
                    offset += (ulong)data.Length * 4u;
                }

                w.Flush();
                long pos = ms.Position;
                int pad = (int)((32 - (pos % 32)) % 32);
                for (int i = 0; i < pad; i++) w.Write((byte)0);

                foreach (var (_, _, data) in tensors)
                    foreach (var v in data) w.Write(v);
            }

            string path = Path.Combine(Path.GetTempPath(), "adn_gguf_" + Guid.NewGuid().ToString("N") + ".gguf");
            File.WriteAllBytes(path, ms.ToArray());
            return path;
        }

        private static void WriteString(BinaryWriter w, string s)
        {
            var bytes = Encoding.UTF8.GetBytes(s);
            w.Write((ulong)bytes.Length);
            w.Write(bytes);
        }

        private static float[] Seq(int n, int seed)
        {
            var a = new float[n];
            for (int i = 0; i < n; i++) a[i] = (((i + seed) % 11) - 5) * 0.02f;
            return a;
        }
    }
}
