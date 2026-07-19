using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using AiDotNet.ModelLoading.Pretrained;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tokenization.Models;
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
        /// <summary>
        /// Parity check: the tokenizer built from a real SmolLM2 GGUF must produce byte-for-byte the same
        /// token ids as llama.cpp for the same text (ground truth captured from llama.cpp's /tokenize on
        /// SmolLM2-135M-Instruct-Q8_0). Gated on the model file being present locally (set
        /// AIDOTNET_SMOLLM2_GGUF), since the multi-hundred-MB checkpoint is not committed.
        /// </summary>
        [Fact]
        public void Gguf_Tokenizer_MatchesLlamaCpp_SmolLM2()
        {
            string? path = Environment.GetEnvironmentVariable("AIDOTNET_SMOLLM2_GGUF");
            if (string.IsNullOrEmpty(path) || !File.Exists(path))
            {
                return; // local-only: requires the real SmolLM2 GGUF checkpoint.
            }

            using var src = GgufModelSource.Open(path);
            var tokenizer = src.BuildTokenizer();
            var opts = new EncodingOptions { AddSpecialTokens = false };

            Assert.Equal(
                new[] { 19556, 905 },
                tokenizer.Encode("Hello world", opts).TokenIds.ToArray());
            Assert.Equal(
                new[] { 504, 2365, 6354, 16438, 27003, 690, 260, 23790, 2767, 30 },
                tokenizer.Encode("The quick brown fox jumps over the lazy dog.", opts).TokenIds.ToArray());
            Assert.Equal(
                new[] { 1604, 3987, 46477, 24, 94, 727 },
                tokenizer.Encode("def fibonacci(n):", opts).TokenIds.ToArray());

            // Round-trip: ids -> exact original text (byte-level BPE is lossless).
            Assert.Equal("Hello world", tokenizer.Decode(new List<int> { 19556, 905 }, skipSpecialTokens: true));
        }

        /// <summary>
        /// End-to-end generation parity: load the real SmolLM2 GGUF (decoder + tokenizer), encode a prompt
        /// with no added special tokens, and assert (a) the token ids match llama.cpp and (b) the greedy
        /// next token is the one llama.cpp predicts (" Paris", id 7042, for "The capital of France is").
        /// This is the generation-correctness check the shape-only GGUF tests never had. Gated on the model
        /// file (set AIDOTNET_SMOLLM2_GGUF).
        /// </summary>
        [Fact]
        public void Gguf_Generation_MatchesLlamaCpp_SmolLM2()
        {
            string? path = Environment.GetEnvironmentVariable("AIDOTNET_SMOLLM2_GGUF");
            if (string.IsNullOrEmpty(path) || !File.Exists(path))
            {
                return;
            }

            var (model, tokenizer) = PretrainedLoader<float>.LoadGgufWithTokenizer(path);

            // Tokenization must match llama.cpp exactly first (isolates tokenizer vs forward).
            var ids = tokenizer.Encode(
                "The capital of France is", new EncodingOptions { AddSpecialTokens = false }).TokenIds;
            Assert.Equal(new[] { 504, 3575, 282, 4649, 314 }, ids.ToArray());

            // Single-token forward is RoPE-independent (position 0 => rotation angle 0), so this isolates the
            // non-positional path (embedding/attention/FFN/norm/dequant) from RoPE. llama.cpp greedy next
            // token for "Hello" (id 19556) is '\n' (id 198).
            Assert.Equal(198, GreedyNext(model, new[] { 19556 }));

            // Full prompt: llama.cpp greedy next is " Paris" (id 7042).
            Assert.Equal(7042, GreedyNext(model, ids.ToArray()));
        }

        private static int GreedyNext(NeuralNetworkBase<float> model, int[] tokenIds)
        {
            int seq = tokenIds.Length;
            var input = new Tensor<float>(new[] { 1, seq });
            for (int i = 0; i < seq; i++) input[0, i] = tokenIds[i];

            var logits = model.Predict(input); // [1, seq, vocab]
            int vocab = logits.Shape[2];
            int best = 0;
            float bestVal = float.NegativeInfinity;
            for (int v = 0; v < vocab; v++)
            {
                float x = Convert.ToSingle(logits[0, seq - 1, v]);
                if (x > bestVal) { bestVal = x; best = v; }
            }

            return best;
        }

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
                Assert.Equal(new[] { 1, 3, 6 }, logits.Shape.ToArray());
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
                Assert.Equal(new[] { 1, 2, 6 }, net.Predict(tokens).Shape.ToArray());
            }
            finally { File.Delete(path); }
        }

        [Fact]
        public void Gguf_EndToEnd_Gemma2_SandwichNorms_HeadDim_Softcap()
        {
            // Gemma-2 exercises the family-specific pieces the generic llama map cannot: four sandwiched
            // norms (attn_norm/attn_post_norm/ffn_norm/ffn_post_norm), an explicit head_dim (key_length)
            // that differs from hidden/heads, and both logit soft-caps read from GGUF metadata.
            string path = WriteGemma2Gguf();
            try
            {
                using (var src = GgufModelSource.Open(path))
                {
                    Assert.Equal("gemma2", src.Config.ModelType);
                    Assert.Equal(6, src.Config.HeadDim);                 // from gemma2.attention.key_length
                    Assert.Equal(50.0, src.Config.AttnLogitSoftcapping);
                    Assert.Equal(30.0, src.Config.FinalLogitSoftcapping);
                    // The GGUF ffn_norm must map to the PRE-feedforward norm, not post-attention.
                    Assert.Contains("model.layers.0.pre_feedforward_layernorm.weight", src.TensorNames);
                    Assert.Contains("model.layers.0.post_attention_layernorm.weight", src.TensorNames);
                    Assert.Contains("model.layers.0.post_feedforward_layernorm.weight", src.TensorNames);
                }

                var model = PretrainedLoader<double>.Load(PretrainedSource.Gguf(path));
                var net = Assert.IsType<NeuralNetwork<double>>(model);
                // embed + block + final norm + lm head + final-logit soft-cap = 5.
                Assert.Equal(5, net.Layers.Count);

                // The attention soft-cap threaded from GGUF metadata into the layer.
                var attn = Assert.IsType<GroupedQueryAttentionLayer<double>>(
                    Assert.IsType<Gemma2DecoderBlock<double>>(net.Layers[1]).AttentionLayer);
                Assert.Equal(50.0, attn.AttnLogitSoftcap);
                Assert.Equal(6, attn.HeadDimension);

                var tokens = new Tensor<double>(new[] { 1, 3 });
                tokens[0, 0] = 1; tokens[0, 1] = 4; tokens[0, 2] = 2;
                var logits = net.Predict(tokens);
                Assert.Equal(new[] { 1, 3, 6 }, logits.Shape.ToArray());
                for (int s = 0; s < 3; s++)
                    for (int v = 0; v < 6; v++)
                        Assert.True(Math.Abs(logits[0, s, v]) < 30.0); // final soft-cap bounds
            }
            finally { File.Delete(path); }
        }

        [Fact]
        public void Gguf_EndToEnd_StarCoder2_BiasedLayerNorm_NonGatedMlp()
        {
            // StarCoder2 exercises biased LayerNorm (weight + bias), a non-gated c_fc/c_proj MLP mapped from
            // ffn_up/ffn_down, biased attention, and the plain layer_norm_epsilon key (not the rms variant).
            string path = WriteStarCoder2Gguf();
            try
            {
                using (var src = GgufModelSource.Open(path))
                {
                    Assert.Equal("starcoder2", src.Config.ModelType);
                    Assert.Contains("model.layers.0.input_layernorm.bias", src.TensorNames);
                    Assert.Contains("model.layers.0.post_attention_layernorm.bias", src.TensorNames);
                    Assert.Contains("model.layers.0.mlp.c_fc.weight", src.TensorNames);
                    Assert.Contains("model.layers.0.mlp.c_proj.bias", src.TensorNames);
                    Assert.Contains("model.norm.bias", src.TensorNames);
                    // Non-gated MLP: the gated gate_proj name must NOT be produced.
                    Assert.DoesNotContain("model.layers.0.mlp.gate_proj.weight", src.TensorNames);
                }

                var model = PretrainedLoader<double>.Load(PretrainedSource.Gguf(path));
                var net = Assert.IsType<NeuralNetwork<double>>(model);
                var tokens = new Tensor<double>(new[] { 1, 3 });
                tokens[0, 0] = 1; tokens[0, 1] = 4; tokens[0, 2] = 2;
                Assert.Equal(new[] { 1, 3, 6 }, net.Predict(tokens).Shape.ToArray());
            }
            finally { File.Delete(path); }
        }

        [Fact]
        public void Gguf_EndToEnd_Mixtral_StackedExperts_SlicedPerExpert()
        {
            // Mixtral ships under the generic "llama" arch with expert_count > 0; the loader must normalize
            // model_type to mixtral, expose the router (ffn_gate_inp) and slice each expert out of the stacked
            // ffn_*_exps tensors under the Mixtral block_sparse_moe.experts.{e}.w1/w3/w2 names.
            string path = WriteMixtralGguf();
            try
            {
                using (var src = GgufModelSource.Open(path))
                {
                    Assert.Equal("mixtral", src.Config.ModelType);
                    Assert.Equal(4, src.Config.NumLocalExperts);
                    Assert.Equal(2, src.Config.NumExpertsPerTok);
                    Assert.Contains("model.layers.0.block_sparse_moe.gate.weight", src.TensorNames);
                    Assert.Contains("model.layers.0.block_sparse_moe.experts.0.w1.weight", src.TensorNames);
                    Assert.Contains("model.layers.0.block_sparse_moe.experts.3.w2.weight", src.TensorNames);
                    // Each expert slice is a distinct quarter of the stacked tensor.
                    var e0 = src.ReadAsDouble("model.layers.0.block_sparse_moe.experts.0.w1.weight");
                    var e1 = src.ReadAsDouble("model.layers.0.block_sparse_moe.experts.1.w1.weight");
                    Assert.Equal(16 * 8, e0.Length); // ffn(16) * hidden(8)
                    Assert.NotEqual(e0, e1);
                }

                var model = PretrainedLoader<double>.Load(PretrainedSource.Gguf(path));
                var net = Assert.IsType<NeuralNetwork<double>>(model);
                var tokens = new Tensor<double>(new[] { 1, 3 });
                tokens[0, 0] = 1; tokens[0, 1] = 4; tokens[0, 2] = 2;
                Assert.Equal(new[] { 1, 3, 6 }, net.Predict(tokens).Shape.ToArray());
            }
            finally { File.Delete(path); }
        }

        [Fact]
        public void Gguf_EndToEnd_Qwen2Moe_SharedExpert_BiasedAttention()
        {
            // Qwen2-MoE (GGUF arch "qwen2moe") normalizes to qwen2_moe, shares the stacked-expert slicing under
            // the mlp.experts.{e}.gate_proj/up_proj/down_proj names, and adds an always-on shared expert plus
            // q/k/v attention biases.
            string path = WriteQwen2MoeGguf();
            try
            {
                using (var src = GgufModelSource.Open(path))
                {
                    Assert.Equal("qwen2_moe", src.Config.ModelType);
                    Assert.Equal(4, src.Config.NumLocalExperts);
                    Assert.Contains("model.layers.0.mlp.gate.weight", src.TensorNames);
                    Assert.Contains("model.layers.0.mlp.experts.0.gate_proj.weight", src.TensorNames);
                    Assert.Contains("model.layers.0.mlp.shared_expert.gate_proj.weight", src.TensorNames);
                    Assert.Contains("model.layers.0.mlp.shared_expert_gate.weight", src.TensorNames);
                    Assert.Contains("model.layers.0.self_attn.q_proj.bias", src.TensorNames);
                }

                var model = PretrainedLoader<double>.Load(PretrainedSource.Gguf(path));
                var net = Assert.IsType<NeuralNetwork<double>>(model);
                var tokens = new Tensor<double>(new[] { 1, 3 });
                tokens[0, 0] = 1; tokens[0, 1] = 4; tokens[0, 2] = 2;
                Assert.Equal(new[] { 1, 3, 6 }, net.Predict(tokens).Shape.ToArray());
            }
            finally { File.Delete(path); }
        }

        [Fact]
        public void Gguf_EndToEnd_Dbrx_FusedQkv_StackedExperts_LayerNorm()
        {
            // DBRX uses its own nested config schema, a fused Wqkv, LayerNorm norms, and stacked experts. The
            // loader must emit the DBRX-shaped config, present standard HF names (so DbrxModelBuilder skips its
            // safetensors DbrxTensorSource translation), split the fused qkv, and slice each expert.
            string path = WriteDbrxGguf();
            try
            {
                using (var src = GgufModelSource.Open(path))
                {
                    Assert.Equal("dbrx", src.Config.ModelType);
                    Assert.Equal(4, src.Config.NumLocalExperts);
                    Assert.Equal(2, src.Config.NumExpertsPerTok);
                    Assert.Contains("model.layers.0.self_attn.qkv_proj.weight", src.TensorNames); // fused Wqkv
                    Assert.Contains("model.layers.0.mlp.gate.weight", src.TensorNames);
                    Assert.Contains("model.layers.0.mlp.experts.0.gate_proj.weight", src.TensorNames);
                    Assert.Contains("model.layers.0.input_layernorm.weight", src.TensorNames);
                }

                var model = PretrainedLoader<double>.Load(PretrainedSource.Gguf(path));
                var net = Assert.IsType<NeuralNetwork<double>>(model);
                var tokens = new Tensor<double>(new[] { 1, 3 });
                tokens[0, 0] = 1; tokens[0, 1] = 4; tokens[0, 2] = 2;
                Assert.Equal(new[] { 1, 3, 6 }, net.Predict(tokens).Shape.ToArray());
            }
            finally { File.Delete(path); }
        }

        // ---- family-specific GGUF writers (F32) ----

        private static string WriteGemma2Gguf()
        {
            const int hidden = 8, heads = 2, kvHeads = 1, headDim = 6, ffn = 16, vocab = 6;
            int qDim = heads * headDim;   // 12
            int kvDim = kvHeads * headDim; // 6
            var tensors = new List<(string, long[], float[])>
            {
                ("token_embd.weight", new long[] { hidden, vocab }, Seq(vocab * hidden, 1)),
                ("blk.0.attn_norm.weight", new long[] { hidden }, Seq(hidden, 2)),
                ("blk.0.attn_post_norm.weight", new long[] { hidden }, Seq(hidden, 3)),
                ("blk.0.ffn_norm.weight", new long[] { hidden }, Seq(hidden, 4)),
                ("blk.0.ffn_post_norm.weight", new long[] { hidden }, Seq(hidden, 5)),
                ("blk.0.attn_q.weight", new long[] { hidden, qDim }, Seq(qDim * hidden, 6)),
                ("blk.0.attn_k.weight", new long[] { hidden, kvDim }, Seq(kvDim * hidden, 7)),
                ("blk.0.attn_v.weight", new long[] { hidden, kvDim }, Seq(kvDim * hidden, 8)),
                ("blk.0.attn_output.weight", new long[] { qDim, hidden }, Seq(hidden * qDim, 9)),
                ("blk.0.ffn_gate.weight", new long[] { hidden, ffn }, Seq(ffn * hidden, 10)),
                ("blk.0.ffn_up.weight", new long[] { hidden, ffn }, Seq(ffn * hidden, 11)),
                ("blk.0.ffn_down.weight", new long[] { ffn, hidden }, Seq(hidden * ffn, 12)),
                ("output_norm.weight", new long[] { hidden }, Seq(hidden, 13)),
            };
            const string a = "gemma2";
            var meta = new List<(string, uint, object)>
            {
                ("general.architecture", 8u, a),
                ("general.alignment", 4u, 32u),
                ($"{a}.embedding_length", 4u, (uint)hidden),
                ($"{a}.block_count", 4u, 1u),
                ($"{a}.attention.head_count", 4u, (uint)heads),
                ($"{a}.attention.head_count_kv", 4u, (uint)kvHeads),
                ($"{a}.attention.key_length", 4u, (uint)headDim),
                ($"{a}.feed_forward_length", 4u, (uint)ffn),
                ($"{a}.context_length", 4u, 64u),
                ($"{a}.attention.layer_norm_rms_epsilon", 6u, 1e-5f),
                ($"{a}.rope.freq_base", 6u, 10000.0f),
                ($"{a}.vocab_size", 4u, (uint)vocab),
                ($"{a}.attn_logit_softcapping", 6u, 50.0f),
                ($"{a}.final_logit_softcapping", 6u, 30.0f),
            };
            return SerializeGguf(tensors, meta);
        }

        private static string WriteStarCoder2Gguf()
        {
            const int hidden = 8, heads = 2, kvHeads = 1, headDim = 4, ffn = 16, vocab = 6;
            int qDim = heads * headDim;   // 8
            int kvDim = kvHeads * headDim; // 4
            var tensors = new List<(string, long[], float[])>
            {
                ("token_embd.weight", new long[] { hidden, vocab }, Seq(vocab * hidden, 1)),
                ("blk.0.attn_norm.weight", new long[] { hidden }, Seq(hidden, 2)),
                ("blk.0.attn_norm.bias", new long[] { hidden }, Seq(hidden, 3)),
                ("blk.0.ffn_norm.weight", new long[] { hidden }, Seq(hidden, 4)),
                ("blk.0.ffn_norm.bias", new long[] { hidden }, Seq(hidden, 5)),
                ("blk.0.attn_q.weight", new long[] { hidden, qDim }, Seq(qDim * hidden, 6)),
                ("blk.0.attn_q.bias", new long[] { qDim }, Seq(qDim, 7)),
                ("blk.0.attn_k.weight", new long[] { hidden, kvDim }, Seq(kvDim * hidden, 8)),
                ("blk.0.attn_k.bias", new long[] { kvDim }, Seq(kvDim, 9)),
                ("blk.0.attn_v.weight", new long[] { hidden, kvDim }, Seq(kvDim * hidden, 10)),
                ("blk.0.attn_v.bias", new long[] { kvDim }, Seq(kvDim, 11)),
                ("blk.0.attn_output.weight", new long[] { qDim, hidden }, Seq(hidden * qDim, 12)),
                ("blk.0.attn_output.bias", new long[] { hidden }, Seq(hidden, 13)),
                ("blk.0.ffn_up.weight", new long[] { hidden, ffn }, Seq(ffn * hidden, 14)),   // -> c_fc
                ("blk.0.ffn_up.bias", new long[] { ffn }, Seq(ffn, 15)),
                ("blk.0.ffn_down.weight", new long[] { ffn, hidden }, Seq(hidden * ffn, 16)), // -> c_proj
                ("blk.0.ffn_down.bias", new long[] { hidden }, Seq(hidden, 17)),
                ("output_norm.weight", new long[] { hidden }, Seq(hidden, 18)),
                ("output_norm.bias", new long[] { hidden }, Seq(hidden, 19)),
            };
            const string a = "starcoder2";
            var meta = new List<(string, uint, object)>
            {
                ("general.architecture", 8u, a),
                ("general.alignment", 4u, 32u),
                ($"{a}.embedding_length", 4u, (uint)hidden),
                ($"{a}.block_count", 4u, 1u),
                ($"{a}.attention.head_count", 4u, (uint)heads),
                ($"{a}.attention.head_count_kv", 4u, (uint)kvHeads),
                ($"{a}.feed_forward_length", 4u, (uint)ffn),
                ($"{a}.context_length", 4u, 64u),
                ($"{a}.attention.layer_norm_epsilon", 6u, 1e-5f), // plain LayerNorm eps (not rms)
                ($"{a}.rope.freq_base", 6u, 10000.0f),
                ($"{a}.vocab_size", 4u, (uint)vocab),
            };
            return SerializeGguf(tensors, meta);
        }

        private static string WriteMixtralGguf()
        {
            const int hidden = 8, heads = 2, kvHeads = 2, headDim = 4, ffn = 16, nExp = 4, vocab = 6;
            int qDim = heads * headDim;      // 8
            int kvDim = kvHeads * headDim;   // 8
            int expElems = ffn * hidden;     // 128 per expert (each of gate/up/down)
            var tensors = new List<(string, long[], float[])>
            {
                ("token_embd.weight", new long[] { hidden, vocab }, Seq(vocab * hidden, 1)),
                ("blk.0.attn_norm.weight", new long[] { hidden }, Seq(hidden, 2)),
                ("blk.0.ffn_norm.weight", new long[] { hidden }, Seq(hidden, 3)),
                ("blk.0.attn_q.weight", new long[] { hidden, qDim }, Seq(qDim * hidden, 4)),
                ("blk.0.attn_k.weight", new long[] { hidden, kvDim }, Seq(kvDim * hidden, 5)),
                ("blk.0.attn_v.weight", new long[] { hidden, kvDim }, Seq(kvDim * hidden, 6)),
                ("blk.0.attn_output.weight", new long[] { qDim, hidden }, Seq(hidden * qDim, 7)),
                // Router [n_expert, hidden] and the three stacked expert tensors ne = [in, out, n_expert].
                ("blk.0.ffn_gate_inp.weight", new long[] { hidden, nExp }, Seq(nExp * hidden, 8)),
                ("blk.0.ffn_gate_exps.weight", new long[] { hidden, ffn, nExp }, Seq(expElems * nExp, 9)),
                ("blk.0.ffn_up_exps.weight", new long[] { hidden, ffn, nExp }, Seq(expElems * nExp, 10)),
                ("blk.0.ffn_down_exps.weight", new long[] { ffn, hidden, nExp }, Seq(expElems * nExp, 11)),
                ("output_norm.weight", new long[] { hidden }, Seq(hidden, 12)),
            };
            const string a = "llama"; // GGUF ships Mixtral under the generic llama arch
            var meta = new List<(string, uint, object)>
            {
                ("general.architecture", 8u, a),
                ("general.alignment", 4u, 32u),
                ($"{a}.embedding_length", 4u, (uint)hidden),
                ($"{a}.block_count", 4u, 1u),
                ($"{a}.attention.head_count", 4u, (uint)heads),
                ($"{a}.attention.head_count_kv", 4u, (uint)kvHeads),
                ($"{a}.feed_forward_length", 4u, (uint)ffn),
                ($"{a}.context_length", 4u, 64u),
                ($"{a}.attention.layer_norm_rms_epsilon", 6u, 1e-5f),
                ($"{a}.rope.freq_base", 6u, 10000.0f),
                ($"{a}.vocab_size", 4u, (uint)vocab),
                ($"{a}.expert_count", 4u, (uint)nExp),
                ($"{a}.expert_used_count", 4u, 2u),
                ($"{a}.expert_feed_forward_length", 4u, (uint)ffn),
            };
            return SerializeGguf(tensors, meta);
        }

        private static string WriteQwen2MoeGguf()
        {
            const int hidden = 8, heads = 2, kvHeads = 2, headDim = 4, routedFfn = 16, sharedFfn = 16, nExp = 4, vocab = 6;
            int qDim = heads * headDim;      // 8
            int kvDim = kvHeads * headDim;   // 8
            int expElems = routedFfn * hidden; // 128
            var tensors = new List<(string, long[], float[])>
            {
                ("token_embd.weight", new long[] { hidden, vocab }, Seq(vocab * hidden, 1)),
                ("blk.0.attn_norm.weight", new long[] { hidden }, Seq(hidden, 2)),
                ("blk.0.ffn_norm.weight", new long[] { hidden }, Seq(hidden, 3)),
                ("blk.0.attn_q.weight", new long[] { hidden, qDim }, Seq(qDim * hidden, 4)),
                ("blk.0.attn_q.bias", new long[] { qDim }, Seq(qDim, 5)),        // Qwen2 biases q/k/v
                ("blk.0.attn_k.weight", new long[] { hidden, kvDim }, Seq(kvDim * hidden, 6)),
                ("blk.0.attn_k.bias", new long[] { kvDim }, Seq(kvDim, 7)),
                ("blk.0.attn_v.weight", new long[] { hidden, kvDim }, Seq(kvDim * hidden, 8)),
                ("blk.0.attn_v.bias", new long[] { kvDim }, Seq(kvDim, 9)),
                ("blk.0.attn_output.weight", new long[] { qDim, hidden }, Seq(hidden * qDim, 10)),
                ("blk.0.ffn_gate_inp.weight", new long[] { hidden, nExp }, Seq(nExp * hidden, 11)),
                ("blk.0.ffn_gate_exps.weight", new long[] { hidden, routedFfn, nExp }, Seq(expElems * nExp, 12)),
                ("blk.0.ffn_up_exps.weight", new long[] { hidden, routedFfn, nExp }, Seq(expElems * nExp, 13)),
                ("blk.0.ffn_down_exps.weight", new long[] { routedFfn, hidden, nExp }, Seq(expElems * nExp, 14)),
                // Always-on shared expert + its sigmoid gate logit ([1, hidden]).
                ("blk.0.ffn_gate_shexp.weight", new long[] { hidden, sharedFfn }, Seq(sharedFfn * hidden, 15)),
                ("blk.0.ffn_up_shexp.weight", new long[] { hidden, sharedFfn }, Seq(sharedFfn * hidden, 16)),
                ("blk.0.ffn_down_shexp.weight", new long[] { sharedFfn, hidden }, Seq(hidden * sharedFfn, 17)),
                ("blk.0.ffn_gate_inp_shexp.weight", new long[] { hidden, 1 }, Seq(hidden, 18)),
                ("output_norm.weight", new long[] { hidden }, Seq(hidden, 19)),
            };
            const string a = "qwen2moe";
            var meta = new List<(string, uint, object)>
            {
                ("general.architecture", 8u, a),
                ("general.alignment", 4u, 32u),
                ($"{a}.embedding_length", 4u, (uint)hidden),
                ($"{a}.block_count", 4u, 1u),
                ($"{a}.attention.head_count", 4u, (uint)heads),
                ($"{a}.attention.head_count_kv", 4u, (uint)kvHeads),
                ($"{a}.feed_forward_length", 4u, (uint)routedFfn),
                ($"{a}.context_length", 4u, 64u),
                ($"{a}.attention.layer_norm_rms_epsilon", 6u, 1e-5f),
                ($"{a}.rope.freq_base", 6u, 10000.0f),
                ($"{a}.vocab_size", 4u, (uint)vocab),
                ($"{a}.expert_count", 4u, (uint)nExp),
                ($"{a}.expert_used_count", 4u, 2u),
                ($"{a}.expert_feed_forward_length", 4u, (uint)routedFfn),
                ($"{a}.expert_shared_feed_forward_length", 4u, (uint)sharedFfn),
            };
            return SerializeGguf(tensors, meta);
        }

        private static string WriteDbrxGguf()
        {
            const int hidden = 8, heads = 2, kvHeads = 2, headDim = 4, ffn = 16, nExp = 4, vocab = 6;
            int qDim = heads * headDim;      // 8
            int kvDim = kvHeads * headDim;   // 8
            int qkvRows = qDim + 2 * kvDim;  // 24 (fused Q;K;V)
            int expElems = ffn * hidden;     // 128
            var tensors = new List<(string, long[], float[])>
            {
                ("token_embd.weight", new long[] { hidden, vocab }, Seq(vocab * hidden, 1)),
                ("blk.0.attn_norm.weight", new long[] { hidden }, Seq(hidden, 2)),
                ("blk.0.ffn_norm.weight", new long[] { hidden }, Seq(hidden, 3)),
                ("blk.0.attn_qkv.weight", new long[] { hidden, qkvRows }, Seq(qkvRows * hidden, 4)), // fused Wqkv
                ("blk.0.attn_output.weight", new long[] { qDim, hidden }, Seq(hidden * qDim, 5)),
                ("blk.0.ffn_gate_inp.weight", new long[] { hidden, nExp }, Seq(nExp * hidden, 6)),
                ("blk.0.ffn_gate_exps.weight", new long[] { hidden, ffn, nExp }, Seq(expElems * nExp, 7)),
                ("blk.0.ffn_up_exps.weight", new long[] { hidden, ffn, nExp }, Seq(expElems * nExp, 8)),
                ("blk.0.ffn_down_exps.weight", new long[] { ffn, hidden, nExp }, Seq(expElems * nExp, 9)),
                ("output_norm.weight", new long[] { hidden }, Seq(hidden, 10)),
            };
            const string a = "dbrx";
            var meta = new List<(string, uint, object)>
            {
                ("general.architecture", 8u, a),
                ("general.alignment", 4u, 32u),
                ($"{a}.embedding_length", 4u, (uint)hidden),
                ($"{a}.block_count", 4u, 1u),
                ($"{a}.attention.head_count", 4u, (uint)heads),
                ($"{a}.attention.head_count_kv", 4u, (uint)kvHeads),
                ($"{a}.feed_forward_length", 4u, (uint)ffn),
                ($"{a}.context_length", 4u, 64u),
                ($"{a}.rope.freq_base", 6u, 500000.0f),
                ($"{a}.vocab_size", 4u, (uint)vocab),
                ($"{a}.expert_count", 4u, (uint)nExp),
                ($"{a}.expert_used_count", 4u, 2u),
            };
            return SerializeGguf(tensors, meta);
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

            return SerializeGguf(tensors, meta);
        }

        // Serializes a GGUF v3 file (F32 tensors) from tensor + metadata lists. Metadata value types:
        // 4u = uint32, 6u = float32, 8u = string.
        private static string SerializeGguf(
            List<(string name, long[] dims, float[] data)> tensors,
            List<(string key, uint type, object val)> meta)
        {
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
