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
