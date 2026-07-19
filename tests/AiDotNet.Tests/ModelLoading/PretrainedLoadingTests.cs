using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using AiDotNet.Agentic.Models.Local;
using AiDotNet.ModelLoading.Pretrained;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.ModelLoading
{
    /// <summary>
    /// Tests for the facade-wired pretrained-loading path: config.json parsing, the LLaMA-family
    /// decoder builder + Hugging Face weight mapping, and the sharded safetensors source.
    /// </summary>
    public class PretrainedLoadingTests
    {
        // ---- HuggingFaceConfig ----

        [Fact]
        public void Config_Parse_ReadsLlamaFields()
        {
            string json = @"{
                ""architectures"": [""LlamaForCausalLM""],
                ""model_type"": ""llama"",
                ""hidden_size"": 32,
                ""intermediate_size"": 64,
                ""num_hidden_layers"": 2,
                ""num_attention_heads"": 4,
                ""num_key_value_heads"": 2,
                ""vocab_size"": 100,
                ""rms_norm_eps"": 1e-5,
                ""rope_theta"": 500000.0,
                ""max_position_embeddings"": 8192,
                ""tie_word_embeddings"": true,
                ""hidden_act"": ""silu"",
                ""torch_dtype"": ""bfloat16"",
                ""bos_token_id"": 1,
                ""eos_token_id"": 2
            }";

            var config = HuggingFaceConfig.Parse(json);

            Assert.Equal("LlamaForCausalLM", config.Architectures[0]);
            Assert.Equal("llama", config.ModelType);
            Assert.Equal(32, config.HiddenSize);
            Assert.Equal(64, config.IntermediateSize);
            Assert.Equal(2, config.NumHiddenLayers);
            Assert.Equal(4, config.NumAttentionHeads);
            Assert.Equal(2, config.NumKeyValueHeads);
            Assert.Equal(100, config.VocabSize);
            Assert.Equal(8, config.HeadDim); // 32 / 4
            Assert.Equal(500000.0, config.RopeTheta);
            Assert.True(config.TieWordEmbeddings);
            Assert.Equal(1, config.BosTokenId);
            Assert.Equal(2, config.EosTokenId);
        }

        [Fact]
        public void Config_DefaultsKvHeadsToAttentionHeads_WhenAbsent()
        {
            string json = @"{ ""hidden_size"": 16, ""intermediate_size"": 32, ""num_hidden_layers"": 1,
                ""num_attention_heads"": 4, ""vocab_size"": 50 }";
            var config = HuggingFaceConfig.Parse(json);
            Assert.Equal(4, config.NumKeyValueHeads); // MHA fallback
        }

        [Fact]
        public void Config_Throws_OnInvalidGqaHeadDivisibility()
        {
            string json = @"{ ""hidden_size"": 16, ""intermediate_size"": 32, ""num_hidden_layers"": 1,
                ""num_attention_heads"": 4, ""num_key_value_heads"": 3, ""vocab_size"": 50 }";
            Assert.Throws<InvalidDataException>(() => HuggingFaceConfig.Parse(json));
        }

        [Fact]
        public void Config_Throws_OnMissingRequiredField()
        {
            string json = @"{ ""num_attention_heads"": 4, ""vocab_size"": 50 }"; // no hidden_size
            Assert.Throws<InvalidDataException>(() => HuggingFaceConfig.Parse(json));
        }

        // ---- LlamaModelBuilder ----

        private static HuggingFaceConfig TinyConfig(bool tie) => HuggingFaceConfig.Parse($@"{{
            ""architectures"": [""LlamaForCausalLM""], ""model_type"": ""llama"",
            ""hidden_size"": 8, ""intermediate_size"": 16, ""num_hidden_layers"": 2,
            ""num_attention_heads"": 2, ""num_key_value_heads"": 1, ""vocab_size"": 10,
            ""rms_norm_eps"": 1e-5, ""rope_theta"": 10000.0, ""max_position_embeddings"": 64,
            ""tie_word_embeddings"": {(tie ? "true" : "false")} }}");

        // Builds an in-memory tensor source with every tensor a tiny Llama needs, filled deterministically.
        private static InMemorySource TinyWeights(HuggingFaceConfig c, bool includeLmHead)
        {
            int h = c.HiddenSize, ffn = c.IntermediateSize, heads = c.NumAttentionHeads;
            int kv = c.NumKeyValueHeads, headDim = h / heads, kvDim = kv * headDim;
            var t = new Dictionary<string, double[]>();
            t["model.embed_tokens.weight"] = Fill(c.VocabSize * h, 1);
            for (int i = 0; i < c.NumHiddenLayers; i++)
            {
                string p = $"model.layers.{i}.";
                t[p + "input_layernorm.weight"] = Fill(h, i + 2);
                t[p + "post_attention_layernorm.weight"] = Fill(h, i + 3);
                t[p + "self_attn.q_proj.weight"] = Fill(heads * headDim * h, i + 4);
                t[p + "self_attn.k_proj.weight"] = Fill(kvDim * h, i + 5);
                t[p + "self_attn.v_proj.weight"] = Fill(kvDim * h, i + 6);
                t[p + "self_attn.o_proj.weight"] = Fill(h * heads * headDim, i + 7);
                t[p + "mlp.gate_proj.weight"] = Fill(ffn * h, i + 8);
                t[p + "mlp.up_proj.weight"] = Fill(ffn * h, i + 9);
                t[p + "mlp.down_proj.weight"] = Fill(h * ffn, i + 10);
            }
            t["model.norm.weight"] = Fill(h, 99);
            if (includeLmHead)
                t["lm_head.weight"] = Fill(c.VocabSize * h, 42);
            return new InMemorySource(t);
        }

        private static double[] Fill(int n, int seed)
        {
            var a = new double[n];
            for (int i = 0; i < n; i++) a[i] = (((i + seed) % 7) - 3) * 0.01;
            return a;
        }

        [Fact]
        public void Builder_BuildsCanonicalStack_AndProducesLogits()
        {
            var config = TinyConfig(tie: false);
            var weights = TinyWeights(config, includeLmHead: true);

            NeuralNetwork<double> net = LlamaModelBuilder<double>.Build(config, weights);

            // Canonical stack: Embedding + 2 blocks + final RMSNorm + LM head = 5 layers.
            Assert.Equal(5, net.Layers.Count);

            // Forward: [1, seq] token ids -> [1, seq, vocab].
            var tokens = new Tensor<double>(new[] { 1, 3 });
            tokens[0, 0] = 1; tokens[0, 1] = 5; tokens[0, 2] = 2;
            var logits = net.Predict(tokens);

            Assert.Equal(3, logits.Shape.Length);
            Assert.Equal(1, logits.Shape[0]);
            Assert.Equal(3, logits.Shape[1]);
            Assert.Equal(config.VocabSize, logits.Shape[2]);
        }

        [Fact]
        public void Builder_IsDeterministic_ForSameWeights()
        {
            var config = TinyConfig(tie: false);
            var a = LlamaModelBuilder<double>.Build(config, TinyWeights(config, true));
            var b = LlamaModelBuilder<double>.Build(config, TinyWeights(config, true));

            var tokens = new Tensor<double>(new[] { 1, 2 });
            tokens[0, 0] = 3; tokens[0, 1] = 7;
            var la = a.Predict(tokens);
            var lb = b.Predict(tokens);

            Assert.Equal(la.Shape.Length, lb.Shape.Length);
            for (int s = 0; s < la.Shape[1]; s++)
                for (int v = 0; v < la.Shape[2]; v++)
                    Assert.Equal(la[0, s, v], lb[0, s, v], 10);
        }

        [Fact]
        public void Builder_TieWordEmbeddings_LoadsWithoutLmHeadTensor()
        {
            var config = TinyConfig(tie: true);
            var weights = TinyWeights(config, includeLmHead: false); // no lm_head.weight

            var net = LlamaModelBuilder<double>.Build(config, weights);
            Assert.Equal(5, net.Layers.Count);
        }

        [Fact]
        public void Builder_Throws_WhenLmHeadMissingAndNotTied()
        {
            var config = TinyConfig(tie: false);
            var weights = TinyWeights(config, includeLmHead: false);

            Assert.Throws<InvalidDataException>(() => LlamaModelBuilder<double>.Build(config, weights));
        }

        [Fact]
        public void Builder_Throws_WhenRequiredTensorMissing()
        {
            var config = TinyConfig(tie: false);
            var weights = TinyWeights(config, includeLmHead: true);
            weights.Remove("model.layers.0.self_attn.q_proj.weight");

            Assert.Throws<InvalidDataException>(() => LlamaModelBuilder<double>.Build(config, weights));
        }

        [Fact]
        public void Builder_ExplicitHeadDim_BuildsAndForwards()
        {
            // head_dim (6) != hidden/num_heads (8/2=4): numHeads*headDim = 12 != hidden. The GQA layer is
            // built with the explicit head dim and the projections follow from it.
            var config = HuggingFaceConfig.Parse(@"{
                ""architectures"": [""LlamaForCausalLM""], ""model_type"": ""llama"",
                ""hidden_size"": 8, ""intermediate_size"": 16, ""num_hidden_layers"": 1,
                ""num_attention_heads"": 2, ""num_key_value_heads"": 1, ""head_dim"": 6,
                ""vocab_size"": 10, ""tie_word_embeddings"": false }");
            Assert.Equal(6, config.HeadDim);

            int h = 8, ffn = 16, vocab = 10, heads = 2, kvHeads = 1, headDim = 6;
            int qDim = heads * headDim, kvDim = kvHeads * headDim;
            var t = new Dictionary<string, double[]>
            {
                ["model.embed_tokens.weight"] = Fill(vocab * h, 1),
                ["model.layers.0.input_layernorm.weight"] = Fill(h, 2),
                ["model.layers.0.post_attention_layernorm.weight"] = Fill(h, 3),
                ["model.layers.0.self_attn.q_proj.weight"] = Fill(qDim * h, 4),
                ["model.layers.0.self_attn.k_proj.weight"] = Fill(kvDim * h, 5),
                ["model.layers.0.self_attn.v_proj.weight"] = Fill(kvDim * h, 6),
                ["model.layers.0.self_attn.o_proj.weight"] = Fill(h * qDim, 7),
                ["model.layers.0.mlp.gate_proj.weight"] = Fill(ffn * h, 8),
                ["model.layers.0.mlp.up_proj.weight"] = Fill(ffn * h, 9),
                ["model.layers.0.mlp.down_proj.weight"] = Fill(h * ffn, 10),
                ["model.norm.weight"] = Fill(h, 11),
                ["lm_head.weight"] = Fill(vocab * h, 12),
            };

            var net = LlamaModelBuilder<double>.Build(config, new InMemorySource(t));
            var tokens = new Tensor<double>(new[] { 1, 3 });
            tokens[0, 0] = 1; tokens[0, 1] = 5; tokens[0, 2] = 2;
            var logits = net.Predict(tokens);
            Assert.Equal(new[] { 1, 3, vocab }, logits.Shape);
        }

        [Fact]
        public void Builder_Gemma_AppliesNormOffset_EmbedScale_AndGeglu()
        {
            var config = HuggingFaceConfig.Parse(@"{ ""architectures"": [""GemmaForCausalLM""],
                ""model_type"": ""gemma"", ""hidden_size"": 8, ""intermediate_size"": 16,
                ""num_hidden_layers"": 1, ""num_attention_heads"": 2, ""num_key_value_heads"": 2,
                ""vocab_size"": 6, ""tie_word_embeddings"": true, ""hidden_act"": ""gelu"" }");

            int h = 8, ffn = 16, vocab = 6, heads = 2, kvHeads = 2, headDim = 4;
            var inNorm = Fill(h, 20);
            var embed = Fill(vocab * h, 21);
            var t = new Dictionary<string, double[]>
            {
                ["model.embed_tokens.weight"] = embed,
                ["model.layers.0.input_layernorm.weight"] = inNorm,
                ["model.layers.0.post_attention_layernorm.weight"] = Fill(h, 22),
                ["model.layers.0.self_attn.q_proj.weight"] = Fill(heads * headDim * h, 23),
                ["model.layers.0.self_attn.k_proj.weight"] = Fill(kvHeads * headDim * h, 24),
                ["model.layers.0.self_attn.v_proj.weight"] = Fill(kvHeads * headDim * h, 25),
                ["model.layers.0.self_attn.o_proj.weight"] = Fill(h * heads * headDim, 26),
                ["model.layers.0.mlp.gate_proj.weight"] = Fill(ffn * h, 27),
                ["model.layers.0.mlp.up_proj.weight"] = Fill(ffn * h, 28),
                ["model.layers.0.mlp.down_proj.weight"] = Fill(h * ffn, 29),
                ["model.norm.weight"] = Fill(h, 30),
            };

            var net = LlamaModelBuilder<double>.Build(config, new InMemorySource(t), DecoderOptions<double>.Gemma);

            var tokens = new Tensor<double>(new[] { 1, 2 });
            tokens[0, 0] = 1; tokens[0, 1] = 3;
            Assert.Equal(new[] { 1, 2, vocab }, net.Predict(tokens).Shape);

            // (1 + weight) RMSNorm: the block's pre-attention gamma is the loaded weight + 1.
            var block = Assert.IsType<PreLNTransformerBlock<double>>(net.Layers[1]);
            var gamma = block.Norm1.GetGammaTensor();
            for (int i = 0; i < h; i++)
                Assert.Equal(inNorm[i] + 1.0, gamma[i], 9);

            // sqrt(hidden) embedding scale baked into the table (row 0, col 0).
            var emb = Assert.IsType<EmbeddingLayer<double>>(net.Layers[0]);
            Assert.Equal(embed[0] * Math.Sqrt(h), emb.GetParameters()[0], 9);
        }

        [Fact]
        public void Builder_Phi3_SplitsFusedProjections_AndForwards()
        {
            var config = HuggingFaceConfig.Parse(@"{ ""architectures"": [""Phi3ForCausalLM""],
                ""model_type"": ""phi3"", ""hidden_size"": 8, ""intermediate_size"": 16,
                ""num_hidden_layers"": 1, ""num_attention_heads"": 2, ""num_key_value_heads"": 2,
                ""vocab_size"": 6, ""tie_word_embeddings"": false, ""hidden_act"": ""silu"" }");

            int h = 8, ffn = 16, vocab = 6, heads = 2, kvHeads = 2, headDim = 4;
            int qkvRows = (heads + 2 * kvHeads) * headDim; // 24
            var qkv = Fill(qkvRows * h, 43);
            var t = new Dictionary<string, double[]>
            {
                ["model.embed_tokens.weight"] = Fill(vocab * h, 40),
                ["model.layers.0.input_layernorm.weight"] = Fill(h, 41),
                ["model.layers.0.post_attention_layernorm.weight"] = Fill(h, 42),
                ["model.layers.0.self_attn.qkv_proj.weight"] = qkv,           // fused
                ["model.layers.0.self_attn.o_proj.weight"] = Fill(h * heads * headDim, 44),
                ["model.layers.0.mlp.gate_up_proj.weight"] = Fill(2 * ffn * h, 45), // fused
                ["model.layers.0.mlp.down_proj.weight"] = Fill(h * ffn, 46),
                ["model.norm.weight"] = Fill(h, 47),
                ["lm_head.weight"] = Fill(vocab * h, 48),
            };

            var fused = new FusedProjectionSource(new InMemorySource(t), config);
            Assert.Contains("model.layers.0.self_attn.q_proj.weight", fused.TensorNames);
            Assert.Contains("model.layers.0.mlp.up_proj.weight", fused.TensorNames);

            // q_proj is the first numHeads*headDim rows of qkv_proj.
            var q = fused.ReadAsDouble("model.layers.0.self_attn.q_proj.weight");
            Assert.Equal(heads * headDim * h, q.Length);
            for (int i = 0; i < q.Length; i++) Assert.Equal(qkv[i], q[i]);
            // v_proj is the last numKVHeads*headDim rows.
            var v = fused.ReadAsDouble("model.layers.0.self_attn.v_proj.weight");
            int vStart = (heads + kvHeads) * headDim * h;
            for (int i = 0; i < v.Length; i++) Assert.Equal(qkv[vStart + i], v[i]);

            var net = LlamaModelBuilder<double>.Build(config, fused);
            var tokens = new Tensor<double>(new[] { 1, 2 });
            tokens[0, 0] = 0; tokens[0, 1] = 5;
            Assert.Equal(new[] { 1, 2, vocab }, net.Predict(tokens).Shape);
        }

        [Fact]
        public void Builder_Mixtral_MoE_BuildsAndForwards()
        {
            var config = HuggingFaceConfig.Parse(@"{ ""architectures"": [""MixtralForCausalLM""],
                ""model_type"": ""mixtral"", ""hidden_size"": 8, ""intermediate_size"": 16,
                ""num_hidden_layers"": 1, ""num_attention_heads"": 2, ""num_key_value_heads"": 2,
                ""vocab_size"": 6, ""num_local_experts"": 4, ""num_experts_per_tok"": 2,
                ""tie_word_embeddings"": false }");
            Assert.Equal(4, config.NumLocalExperts);
            Assert.Equal(2, config.NumExpertsPerTok);

            int h = 8, ffn = 16, vocab = 6, heads = 2, kvHeads = 2, headDim = 4, E = 4;
            var t = new Dictionary<string, double[]>
            {
                ["model.embed_tokens.weight"] = Fill(vocab * h, 50),
                ["model.layers.0.input_layernorm.weight"] = Fill(h, 51),
                ["model.layers.0.post_attention_layernorm.weight"] = Fill(h, 52),
                ["model.layers.0.self_attn.q_proj.weight"] = Fill(heads * headDim * h, 53),
                ["model.layers.0.self_attn.k_proj.weight"] = Fill(kvHeads * headDim * h, 54),
                ["model.layers.0.self_attn.v_proj.weight"] = Fill(kvHeads * headDim * h, 55),
                ["model.layers.0.self_attn.o_proj.weight"] = Fill(h * heads * headDim, 56),
                ["model.layers.0.block_sparse_moe.gate.weight"] = Fill(E * h, 57), // router
                ["model.norm.weight"] = Fill(h, 90),
                ["lm_head.weight"] = Fill(vocab * h, 91),
            };
            for (int e = 0; e < E; e++)
            {
                string ep = $"model.layers.0.block_sparse_moe.experts.{e}.";
                t[ep + "w1.weight"] = Fill(ffn * h, 60 + e); // gate
                t[ep + "w3.weight"] = Fill(ffn * h, 70 + e); // up
                t[ep + "w2.weight"] = Fill(h * ffn, 80 + e); // down
            }

            var net = MoEModelBuilder<double>.Build(config, new InMemorySource(t));
            Assert.Equal(4, net.Layers.Count); // embed + 1 MoE block + final norm + lm head

            var tokens = new Tensor<double>(new[] { 1, 3 });
            tokens[0, 0] = 1; tokens[0, 1] = 4; tokens[0, 2] = 2;
            Assert.Equal(new[] { 1, 3, vocab }, net.Predict(tokens).Shape);

            var block = Assert.IsType<MoEDecoderBlock<double>>(net.Layers[1]);
            Assert.Equal(4, block.Moe.NumExperts);
            Assert.Equal(2, block.Moe.TopK);

            Assert.True(PretrainedArchitectures<double>.TryResolve(config, null, out _));
        }

        [Fact]
        public void MoE_Routing_MixesTopKExpertsByRenormalizedWeights()
        {
            // Two experts, top-1: with a strongly separating router, each token goes to exactly one expert,
            // so the layer output equals that expert's SwiGLU output (renormalized weight = 1).
            int h = 4, ffn = 4, E = 2, topK = 1;
            var moe = new MoEFeedForwardLayer<double>(h, ffn, E, topK, new AiDotNet.ActivationFunctions.SiLUActivation<double>());
            moe.Materialize();

            // Router: expert 0 fires on positive first feature, expert 1 otherwise (large logits => hard top-1).
            SetDense(moe.Router, new double[,] { { 100, 0, 0, 0 }, { -100, 0, 0, 0 } }); // [E, h] -> [h,E] internally
            // Give the two experts different, identifiable weights.
            SetDense(moe.ExpertGate(0), Const(ffn, h, 0.0)); // gate(x)=0 -> silu(0)=0 -> expert 0 output is 0
            SetDense(moe.ExpertUp(0), Const(ffn, h, 0.0));
            SetDense(moe.ExpertDown(0), Const(h, ffn, 0.0));
            SetDense(moe.ExpertGate(1), Identityish(ffn, h));
            SetDense(moe.ExpertUp(1), Identityish(ffn, h));
            SetDense(moe.ExpertDown(1), Identityish(h, ffn));

            var x = new Tensor<double>(new[] { 1, h });
            x[0, 0] = 1.0; x[0, 1] = 0.5; x[0, 2] = -0.5; x[0, 3] = 0.25; // positive first feature -> expert 0
            var y = moe.Forward(x);
            // Expert 0 has all-zero weights -> output is exactly zero.
            for (int j = 0; j < h; j++) Assert.Equal(0.0, y[0, j], 9);
        }

        private static void SetDense(DenseLayer<double> dense, double[,] outIn)
        {
            int outDim = outIn.GetLength(0), inDim = outIn.GetLength(1);
            var w = new double[inDim * outDim];               // layer stores [in, out]
            for (int o = 0; o < outDim; o++)
                for (int i = 0; i < inDim; i++)
                    w[i * outDim + o] = outIn[o, i];
            var full = new double[w.Length + outDim];          // + zero bias
            Array.Copy(w, full, w.Length);
            dense.SetParameters(new Vector<double>(full));
        }

        private static double[,] Const(int outDim, int inDim, double v)
        {
            var m = new double[outDim, inDim];
            for (int o = 0; o < outDim; o++) for (int i = 0; i < inDim; i++) m[o, i] = v;
            return m;
        }

        private static double[,] Identityish(int outDim, int inDim)
        {
            var m = new double[outDim, inDim];
            for (int o = 0; o < outDim; o++) m[o, o % inDim] = 1.0;
            return m;
        }

        [Fact]
        public void Builder_Gemma2_DualNorms_Softcap_BuildsAndForwards()
        {
            var config = HuggingFaceConfig.Parse(@"{ ""architectures"": [""Gemma2ForCausalLM""],
                ""model_type"": ""gemma2"", ""hidden_size"": 8, ""intermediate_size"": 16,
                ""num_hidden_layers"": 1, ""num_attention_heads"": 2, ""num_key_value_heads"": 2,
                ""vocab_size"": 6, ""tie_word_embeddings"": true, ""hidden_act"": ""gelu_pytorch_tanh"",
                ""final_logit_softcapping"": 30.0 }");
            Assert.Equal(30.0, config.FinalLogitSoftcapping);

            int h = 8, ffn = 16, vocab = 6, heads = 2, kvHeads = 2, headDim = 4;
            var inNorm = Fill(h, 60);
            var t = new Dictionary<string, double[]>
            {
                ["model.embed_tokens.weight"] = Fill(vocab * h, 61),
                ["model.layers.0.input_layernorm.weight"] = inNorm,
                ["model.layers.0.post_attention_layernorm.weight"] = Fill(h, 62),
                ["model.layers.0.pre_feedforward_layernorm.weight"] = Fill(h, 63),
                ["model.layers.0.post_feedforward_layernorm.weight"] = Fill(h, 64),
                ["model.layers.0.self_attn.q_proj.weight"] = Fill(heads * headDim * h, 65),
                ["model.layers.0.self_attn.k_proj.weight"] = Fill(kvHeads * headDim * h, 66),
                ["model.layers.0.self_attn.v_proj.weight"] = Fill(kvHeads * headDim * h, 67),
                ["model.layers.0.self_attn.o_proj.weight"] = Fill(h * heads * headDim, 68),
                ["model.layers.0.mlp.gate_proj.weight"] = Fill(ffn * h, 69),
                ["model.layers.0.mlp.up_proj.weight"] = Fill(ffn * h, 70),
                ["model.layers.0.mlp.down_proj.weight"] = Fill(h * ffn, 71),
                ["model.norm.weight"] = Fill(h, 72),
            };

            var net = Gemma2ModelBuilder<double>.Build(config, new InMemorySource(t));
            Assert.Equal(5, net.Layers.Count); // embed + block + final norm + lm head + softcap

            var tokens = new Tensor<double>(new[] { 1, 3 });
            tokens[0, 0] = 1; tokens[0, 1] = 4; tokens[0, 2] = 2;
            var logits = net.Predict(tokens);
            Assert.Equal(new[] { 1, 3, vocab }, logits.Shape);
            // Final soft-capping bounds every logit strictly within (-30, 30).
            for (int s = 0; s < 3; s++)
                for (int vv = 0; vv < vocab; vv++)
                    Assert.True(Math.Abs(logits[0, s, vv]) < 30.0);

            // Sandwich norm uses the Gemma (1 + weight) convention.
            var block = Assert.IsType<Gemma2DecoderBlock<double>>(net.Layers[1]);
            var gamma = block.NormInput.GetGammaTensor();
            for (int i = 0; i < h; i++) Assert.Equal(inNorm[i] + 1.0, gamma[i], 9);

            Assert.True(PretrainedArchitectures<double>.TryResolve(config, null, out _));
        }

        [Fact]
        public void Builder_Cohere_LayerNormParallelResidual_BuildsAndForwards()
        {
            var config = HuggingFaceConfig.Parse(@"{ ""architectures"": [""CohereForCausalLM""],
                ""model_type"": ""cohere"", ""hidden_size"": 8, ""intermediate_size"": 16,
                ""num_hidden_layers"": 1, ""num_attention_heads"": 2, ""num_key_value_heads"": 2,
                ""vocab_size"": 6, ""tie_word_embeddings"": true, ""logit_scale"": 0.5 }");
            Assert.Equal(0.5, config.LogitScale);

            int h = 8, ffn = 16, vocab = 6, heads = 2, kvHeads = 2, headDim = 4;
            var t = new Dictionary<string, double[]>
            {
                ["model.embed_tokens.weight"] = Fill(vocab * h, 73),
                ["model.layers.0.input_layernorm.weight"] = Fill(h, 74), // single norm (parallel residual)
                ["model.layers.0.self_attn.q_proj.weight"] = Fill(heads * headDim * h, 75),
                ["model.layers.0.self_attn.k_proj.weight"] = Fill(kvHeads * headDim * h, 76),
                ["model.layers.0.self_attn.v_proj.weight"] = Fill(kvHeads * headDim * h, 77),
                ["model.layers.0.self_attn.o_proj.weight"] = Fill(h * heads * headDim, 78),
                ["model.layers.0.mlp.gate_proj.weight"] = Fill(ffn * h, 79),
                ["model.layers.0.mlp.up_proj.weight"] = Fill(ffn * h, 80),
                ["model.layers.0.mlp.down_proj.weight"] = Fill(h * ffn, 81),
                ["model.norm.weight"] = Fill(h, 82),
            };

            var net = CohereModelBuilder<double>.Build(config, new InMemorySource(t));
            Assert.Equal(4, net.Layers.Count); // embed + block + final layernorm + lm head
            Assert.IsType<CohereDecoderBlock<double>>(net.Layers[1]);

            var tokens = new Tensor<double>(new[] { 1, 3 });
            tokens[0, 0] = 1; tokens[0, 1] = 4; tokens[0, 2] = 2;
            Assert.Equal(new[] { 1, 3, vocab }, net.Predict(tokens).Shape);

            Assert.True(PretrainedArchitectures<double>.TryResolve(config, null, out _));
        }

        [Fact]
        public void GQA_ProjectionBias_ChangesParamCount_ButDefaultIsUnchanged()
        {
            // Default (bias-free): 4 weight matrices [8x8]=256 + output bias 8 = 264.
            var plain = new GroupedQueryAttentionLayer<double>(sequenceLength: 4, embeddingDimension: 8, numHeads: 2, numKVHeads: 2);
            plain.Forward(new Tensor<double>(new[] { 1, 4, 8 }));
            Assert.Equal(264, plain.ParameterCount);

            // With projection biases: + q/k/v biases (8 each) = 264 + 24 = 288.
            var biased = new GroupedQueryAttentionLayer<double>(sequenceLength: 4, embeddingDimension: 8, numHeads: 2, numKVHeads: 2, useProjectionBias: true);
            biased.Forward(new Tensor<double>(new[] { 1, 4, 8 }));
            Assert.Equal(288, biased.ParameterCount);

            // SetParameters round-trips through the biased layout.
            var p = biased.GetParameters();
            Assert.Equal(288, p.Length);
            for (int i = 0; i < p.Length; i++) p[i] = (i % 5) * 0.1;
            biased.SetParameters(p);
            var q = biased.GetParameters();
            for (int i = 0; i < p.Length; i++) Assert.Equal(p[i], q[i], 9);
        }

        [Fact]
        public void Builder_StarCoder2_LayerNormBias_BiasedAttn_NonGatedMlp_BuildsAndForwards()
        {
            var config = HuggingFaceConfig.Parse(@"{ ""architectures"": [""Starcoder2ForCausalLM""],
                ""model_type"": ""starcoder2"", ""hidden_size"": 8, ""intermediate_size"": 16,
                ""num_hidden_layers"": 1, ""num_attention_heads"": 2, ""num_key_value_heads"": 2,
                ""vocab_size"": 6, ""tie_word_embeddings"": false }");

            int h = 8, ffn = 16, vocab = 6, heads = 2, kvHeads = 2, headDim = 4;
            int qDim = heads * headDim, kvDim = kvHeads * headDim;
            var t = new Dictionary<string, double[]>
            {
                ["model.embed_tokens.weight"] = Fill(vocab * h, 83),
                ["model.layers.0.input_layernorm.weight"] = Fill(h, 84),
                ["model.layers.0.input_layernorm.bias"] = Fill(h, 85),
                ["model.layers.0.self_attn.q_proj.weight"] = Fill(qDim * h, 86),
                ["model.layers.0.self_attn.q_proj.bias"] = Fill(qDim, 87),
                ["model.layers.0.self_attn.k_proj.weight"] = Fill(kvDim * h, 88),
                ["model.layers.0.self_attn.k_proj.bias"] = Fill(kvDim, 89),
                ["model.layers.0.self_attn.v_proj.weight"] = Fill(kvDim * h, 90),
                ["model.layers.0.self_attn.v_proj.bias"] = Fill(kvDim, 91),
                ["model.layers.0.self_attn.o_proj.weight"] = Fill(h * qDim, 92),
                ["model.layers.0.self_attn.o_proj.bias"] = Fill(h, 93),
                ["model.layers.0.post_attention_layernorm.weight"] = Fill(h, 94),
                ["model.layers.0.post_attention_layernorm.bias"] = Fill(h, 95),
                ["model.layers.0.mlp.c_fc.weight"] = Fill(ffn * h, 96),
                ["model.layers.0.mlp.c_fc.bias"] = Fill(ffn, 97),
                ["model.layers.0.mlp.c_proj.weight"] = Fill(h * ffn, 98),
                ["model.layers.0.mlp.c_proj.bias"] = Fill(h, 99),
                ["model.norm.weight"] = Fill(h, 100),
                ["model.norm.bias"] = Fill(h, 101),
                ["lm_head.weight"] = Fill(vocab * h, 102),
            };

            var net = StarCoder2ModelBuilder<double>.Build(config, new InMemorySource(t));
            Assert.Equal(4, net.Layers.Count); // embed + block + final layernorm + lm head
            Assert.IsType<StarCoder2DecoderBlock<double>>(net.Layers[1]);

            var tokens = new Tensor<double>(new[] { 1, 3 });
            tokens[0, 0] = 1; tokens[0, 1] = 4; tokens[0, 2] = 2;
            Assert.Equal(new[] { 1, 3, vocab }, net.Predict(tokens).Shape);

            Assert.True(PretrainedArchitectures<double>.TryResolve(config, null, out _));
        }

        [Fact]
        public void Builder_Qwen2_LoadsQkvBiases()
        {
            var config = HuggingFaceConfig.Parse(@"{ ""architectures"": [""Qwen2ForCausalLM""],
                ""model_type"": ""qwen2"", ""hidden_size"": 8, ""intermediate_size"": 16,
                ""num_hidden_layers"": 1, ""num_attention_heads"": 2, ""num_key_value_heads"": 2,
                ""vocab_size"": 6, ""tie_word_embeddings"": false }");

            int h = 8, ffn = 16, vocab = 6, heads = 2, kvHeads = 2, headDim = 4;
            int qDim = heads * headDim, kvDim = kvHeads * headDim;
            var t = new Dictionary<string, double[]>
            {
                ["model.embed_tokens.weight"] = Fill(vocab * h, 103),
                ["model.layers.0.input_layernorm.weight"] = Fill(h, 104),
                ["model.layers.0.post_attention_layernorm.weight"] = Fill(h, 105),
                ["model.layers.0.self_attn.q_proj.weight"] = Fill(qDim * h, 106),
                ["model.layers.0.self_attn.q_proj.bias"] = Fill(qDim, 107),
                ["model.layers.0.self_attn.k_proj.weight"] = Fill(kvDim * h, 108),
                ["model.layers.0.self_attn.k_proj.bias"] = Fill(kvDim, 109),
                ["model.layers.0.self_attn.v_proj.weight"] = Fill(kvDim * h, 110),
                ["model.layers.0.self_attn.v_proj.bias"] = Fill(kvDim, 111),
                ["model.layers.0.self_attn.o_proj.weight"] = Fill(h * qDim, 112), // no o bias (Qwen2)
                ["model.layers.0.mlp.gate_proj.weight"] = Fill(ffn * h, 113),
                ["model.layers.0.mlp.up_proj.weight"] = Fill(ffn * h, 114),
                ["model.layers.0.mlp.down_proj.weight"] = Fill(h * ffn, 115),
                ["model.norm.weight"] = Fill(h, 116),
                ["lm_head.weight"] = Fill(vocab * h, 117),
            };

            Assert.True(PretrainedArchitectures<double>.TryResolve(config, null, out var factory));
            var net = factory(config, new InMemorySource(t));
            var block = Assert.IsType<PreLNTransformerBlock<double>>(net.Layers[1]);
            // Biased attention: param count includes q/k/v biases (+ output bias 8) = weights(256) + 24 + 8 = 288.
            Assert.Equal(288, block.AttentionLayer.ParameterCount);

            var tokens = new Tensor<double>(new[] { 1, 2 });
            tokens[0, 0] = 1; tokens[0, 1] = 3;
            Assert.Equal(new[] { 1, 2, vocab }, net.Predict(tokens).Shape);
        }

        [Fact]
        public void Builder_Qwen2Moe_SharedExpert_BuildsAndForwards()
        {
            var config = HuggingFaceConfig.Parse(@"{ ""architectures"": [""Qwen2MoeForCausalLM""],
                ""model_type"": ""qwen2_moe"", ""hidden_size"": 8, ""intermediate_size"": 32,
                ""moe_intermediate_size"": 16, ""shared_expert_intermediate_size"": 20,
                ""num_hidden_layers"": 1, ""num_attention_heads"": 2, ""num_key_value_heads"": 2,
                ""vocab_size"": 6, ""num_experts"": 4, ""num_experts_per_tok"": 2, ""tie_word_embeddings"": false }");
            Assert.Equal(4, config.NumLocalExperts);
            Assert.Equal(16, config.MoeIntermediateSize);
            Assert.Equal(20, config.SharedExpertIntermediateSize);

            int h = 8, routed = 16, shared = 20, vocab = 6, heads = 2, kvHeads = 2, headDim = 4, E = 4;
            int qDim = heads * headDim, kvDim = kvHeads * headDim;
            var t = new Dictionary<string, double[]>
            {
                ["model.embed_tokens.weight"] = Fill(vocab * h, 120),
                ["model.layers.0.input_layernorm.weight"] = Fill(h, 121),
                ["model.layers.0.post_attention_layernorm.weight"] = Fill(h, 122),
                ["model.layers.0.self_attn.q_proj.weight"] = Fill(qDim * h, 123),
                ["model.layers.0.self_attn.q_proj.bias"] = Fill(qDim, 124),
                ["model.layers.0.self_attn.k_proj.weight"] = Fill(kvDim * h, 125),
                ["model.layers.0.self_attn.k_proj.bias"] = Fill(kvDim, 126),
                ["model.layers.0.self_attn.v_proj.weight"] = Fill(kvDim * h, 127),
                ["model.layers.0.self_attn.v_proj.bias"] = Fill(kvDim, 128),
                ["model.layers.0.self_attn.o_proj.weight"] = Fill(h * qDim, 129),
                ["model.layers.0.mlp.gate.weight"] = Fill(E * h, 130), // router
                ["model.layers.0.mlp.shared_expert.gate_proj.weight"] = Fill(shared * h, 131),
                ["model.layers.0.mlp.shared_expert.up_proj.weight"] = Fill(shared * h, 132),
                ["model.layers.0.mlp.shared_expert.down_proj.weight"] = Fill(h * shared, 133),
                ["model.layers.0.mlp.shared_expert_gate.weight"] = Fill(1 * h, 134),
                ["model.norm.weight"] = Fill(h, 150),
                ["lm_head.weight"] = Fill(vocab * h, 151),
            };
            for (int e = 0; e < E; e++)
            {
                string ep = $"model.layers.0.mlp.experts.{e}.";
                t[ep + "gate_proj.weight"] = Fill(routed * h, 135 + e);
                t[ep + "up_proj.weight"] = Fill(routed * h, 140 + e);
                t[ep + "down_proj.weight"] = Fill(h * routed, 145 + e);
            }

            var net = Qwen2MoEModelBuilder<double>.Build(config, new InMemorySource(t));
            var block = Assert.IsType<MoEDecoderBlock<double>>(net.Layers[1]);
            Assert.True(block.Moe.HasSharedExpert);

            var tokens = new Tensor<double>(new[] { 1, 3 });
            tokens[0, 0] = 1; tokens[0, 1] = 4; tokens[0, 2] = 2;
            Assert.Equal(new[] { 1, 3, vocab }, net.Predict(tokens).Shape);

            Assert.True(PretrainedArchitectures<double>.TryResolve(config, null, out _));
        }

        [Fact]
        public void Architectures_ResolvesGemmaAndPhi3()
        {
            var gemma = HuggingFaceConfig.Parse(@"{ ""architectures"": [""GemmaForCausalLM""], ""model_type"": ""gemma"",
                ""hidden_size"": 8, ""intermediate_size"": 16, ""num_hidden_layers"": 1,
                ""num_attention_heads"": 2, ""vocab_size"": 6 }");
            var phi3 = HuggingFaceConfig.Parse(@"{ ""architectures"": [""Phi3ForCausalLM""], ""model_type"": ""phi3"",
                ""hidden_size"": 8, ""intermediate_size"": 16, ""num_hidden_layers"": 1,
                ""num_attention_heads"": 2, ""vocab_size"": 6 }");
            Assert.True(PretrainedArchitectures<double>.TryResolve(gemma, null, out _));
            Assert.True(PretrainedArchitectures<double>.TryResolve(phi3, null, out _));
        }

        // ---- PretrainedSource descriptor ----

        [Fact]
        public void Source_Factories_SetKindAndLocator()
        {
            Assert.Equal(PretrainedModelKind.HuggingFace, PretrainedSource.HuggingFace("org/model").Kind);
            Assert.Equal(PretrainedModelKind.Safetensors, PretrainedSource.Safetensors("/m").Kind);
            Assert.Equal(PretrainedModelKind.Onnx, PretrainedSource.Onnx("/m.onnx").Kind);
            Assert.Equal(PretrainedModelKind.Gguf, PretrainedSource.Gguf("/m.gguf").Kind);
        }

        [Fact]
        public void Source_FluentOptions_Chain()
        {
            var s = PretrainedSource.HuggingFace("org/model")
                .WithRevision("v2").Dtype(PretrainedDType.BFloat16).WithCacheDirectory("/cache");

            Assert.Equal("org/model", s.Locator);
            Assert.Equal("v2", s.Revision);
            Assert.Equal(PretrainedDType.BFloat16, s.DType);
            Assert.Equal("/cache", s.CacheDirectory);
        }

        [Fact]
        public void Architectures_RecognizesLlamaFamily()
        {
            var config = TinyConfig(tie: false);
            Assert.True(PretrainedArchitectures<double>.TryResolve(config, null, out _));
        }

        [Fact]
        public void Loader_OnnxSource_IsWired_ReachesFileOpen()
        {
            // Wired: a missing file throws FileNotFound (the resolver reached the ONNX adapter),
            // NOT NotSupported (which was the pre-wiring behavior).
            Assert.Throws<FileNotFoundException>(() =>
                PretrainedLoader<double>.Load(PretrainedSource.Onnx(
                    Path.Combine(Path.GetTempPath(), "adn_missing_" + Guid.NewGuid().ToString("N") + ".onnx"))));
        }

        [Fact]
        public void Loader_GgufSource_IsWired_ReachesFileOpen()
        {
            Assert.Throws<FileNotFoundException>(() =>
                PretrainedLoader<double>.Load(PretrainedSource.Gguf(
                    Path.Combine(Path.GetTempPath(), "adn_missing_" + Guid.NewGuid().ToString("N") + ".gguf"))));
        }

        [Fact]
        public void Architectures_ReturnsFalse_ForUnknown()
        {
            var config = HuggingFaceConfig.Parse(@"{ ""architectures"": [""SomeUnknownForCausalLM""],
                ""model_type"": ""unknownarch"", ""hidden_size"": 8, ""intermediate_size"": 16,
                ""num_hidden_layers"": 1, ""num_attention_heads"": 2, ""vocab_size"": 10 }");
            Assert.False(PretrainedArchitectures<double>.TryResolve(config, null, out _));
        }

        // ---- ShardedSafetensorsSource (round-trips a real single-file safetensors) ----

        [Fact]
        public void ShardedSource_ReadsSingleFile()
        {
            string dir = Path.Combine(Path.GetTempPath(), "adn_st_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(dir);
            try
            {
                var values = new float[] { 1.5f, -2.25f, 3.0f, 0.5f };
                WriteSafetensors(Path.Combine(dir, "model.safetensors"),
                    new Dictionary<string, (int[] shape, float[] data)> { ["w"] = (new[] { 2, 2 }, values) });

                using var src = ShardedSafetensorsSource.Open(dir);
                Assert.Contains("w", src.TensorNames);
                var read = src.ReadAsDouble("w");
                Assert.Equal(values.Length, read.Length);
                for (int i = 0; i < values.Length; i++)
                    Assert.Equal(values[i], read[i], 5);
            }
            finally { Directory.Delete(dir, recursive: true); }
        }

        [Fact]
        public void ShardedSource_ReadsMultiShardViaIndex()
        {
            string dir = Path.Combine(Path.GetTempPath(), "adn_st_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(dir);
            try
            {
                WriteSafetensors(Path.Combine(dir, "model-00001-of-00002.safetensors"),
                    new Dictionary<string, (int[], float[])> { ["a"] = (new[] { 2 }, new[] { 10f, 20f }) });
                WriteSafetensors(Path.Combine(dir, "model-00002-of-00002.safetensors"),
                    new Dictionary<string, (int[], float[])> { ["b"] = (new[] { 2 }, new[] { 30f, 40f }) });

                var index = new JObject
                {
                    ["weight_map"] = new JObject
                    {
                        ["a"] = "model-00001-of-00002.safetensors",
                        ["b"] = "model-00002-of-00002.safetensors",
                    }
                };
                File.WriteAllText(Path.Combine(dir, "model.safetensors.index.json"), index.ToString());

                using var src = ShardedSafetensorsSource.Open(dir);
                Assert.Equal(new[] { 10.0, 20.0 }, src.ReadAsDouble("a"));
                Assert.Equal(new[] { 30.0, 40.0 }, src.ReadAsDouble("b"));
            }
            finally { Directory.Delete(dir, recursive: true); }
        }

        // Minimal safetensors writer (F32) for the round-trip tests.
        private static void WriteSafetensors(string path, Dictionary<string, (int[] shape, float[] data)> tensors)
        {
            var header = new JObject();
            long offset = 0;
            var order = new List<(string name, float[] data)>();
            foreach (var kvp in tensors)
            {
                int count = kvp.Value.data.Length;
                long byteLen = count * 4L;
                header[kvp.Key] = new JObject
                {
                    ["dtype"] = "F32",
                    ["shape"] = new JArray(kvp.Value.shape.Select(d => (object)d).ToArray()),
                    ["data_offsets"] = new JArray(offset, offset + byteLen),
                };
                offset += byteLen;
                order.Add((kvp.Key, kvp.Value.data));
            }

            byte[] headerBytes = Encoding.UTF8.GetBytes(header.ToString(Newtonsoft.Json.Formatting.None));
            using var fs = File.Create(path);
            using var bw = new BinaryWriter(fs);
            bw.Write((long)headerBytes.Length);
            bw.Write(headerBytes);
            foreach (var (_, data) in order)
                foreach (var v in data)
                    bw.Write(v);
        }

        private sealed class InMemorySource : INamedTensorSource
        {
            private readonly Dictionary<string, double[]> _tensors;
            public InMemorySource(Dictionary<string, double[]> tensors) => _tensors = tensors;
            public IReadOnlyCollection<string> TensorNames => _tensors.Keys;
            public double[] ReadAsDouble(string name) => _tensors[name];
            public void Remove(string name) => _tensors.Remove(name);
        }
    }
}
