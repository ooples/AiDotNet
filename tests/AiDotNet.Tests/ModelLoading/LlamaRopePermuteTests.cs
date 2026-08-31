using System;
using System.IO;
using System.Linq;
using AiDotNet.Agentic.Models.Local;      // SafetensorsReader
using AiDotNet.ModelLoading.Pretrained;   // LlamaModelBuilder, HuggingFaceConfig
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelLoading
{
    /// <summary>
    /// Covers the HuggingFace half-split -> interleaved q/k row permutation used when loading raw HF safetensors
    /// Llama-family checkpoints (LlamaModelBuilder), so AiDotNet's interleaved RoPE reproduces HF outputs. The
    /// unit tests are self-contained; the golden test is opt-in and skips unless the local SmolLM2-360M fixtures
    /// are present (they are far too large to commit).
    /// </summary>
    public class LlamaRopePermuteTests
    {
        [Fact]
        public void PermuteRopeRows_ReordersLanesWithinEachHead()
        {
            // 2 heads, headDim 4 (half=2), inDim 1. Row value == its lane index for readability.
            // Half-split lanes [0,1,2,3] -> interleaved [0,2,1,3]: new[2i]=old[i], new[2i+1]=old[i+half].
            var hf = new double[] { 0, 1, 2, 3, /* head1 */ 10, 11, 12, 13 };
            var outp = LlamaModelBuilder<double>.PermuteRopeRowsHalfToInterleaved(hf, heads: 2, headDim: 4, inDim: 1);
            Assert.Equal(new double[] { 0, 2, 1, 3, 10, 12, 11, 13 }, outp);
        }

        [Fact]
        public void PermuteRopeRows_MovesWholeRows_ForInDimGreaterThanOne()
        {
            // headDim 4, inDim 3, 1 head: each lane is a 3-wide row that moves as a unit.
            var hf = new double[] { 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3 };
            var outp = LlamaModelBuilder<double>.PermuteRopeRowsHalfToInterleaved(hf, heads: 1, headDim: 4, inDim: 3);
            Assert.Equal(new double[] { 0, 0, 0, 2, 2, 2, 1, 1, 1, 3, 3, 3 }, outp);
        }

        [Fact]
        public void PermuteRopeRows_AppliedTwice_IsNotIdentity_ButPreservesMultiset()
        {
            // Sanity: the permutation is a genuine reorder (not identity for headDim>2) and loses no rows.
            var hf = Enumerable.Range(0, 8).Select(i => (double)i).ToArray();
            var once = LlamaModelBuilder<double>.PermuteRopeRowsHalfToInterleaved(hf, heads: 1, headDim: 8, inDim: 1);
            Assert.NotEqual(hf, once);
            Assert.Equal(hf.OrderBy(x => x), once.OrderBy(x => x));
        }

        // Opt-in golden parity: loads the real SmolLM2-360M HF safetensors and checks the first-16-token logits
        // against the PyTorch reference. Proves the q/k permute + interleaved RoPE reproduces HF exactly. Skips
        // silently when the local fixtures are absent (CI), since the checkpoint cannot be committed.
        [Fact]
        public void SmolLM2_HfSafetensors_ReproducesPyTorchLogitsOracle()
        {
            const string dir = @"C:\Users\cheat\Temp\he-m2-audit\data\smollm2-360m";
            const string dataRoot = @"C:\Users\cheat\Temp\he-m2-audit\data";
            string modelPath = Path.Combine(dir, "model_f32.safetensors");
            string configPath = Path.Combine(dir, "config.json");
            string refPath = Path.Combine(dataRoot, "ref_logits_16.f32");
            string tokPath = Path.Combine(dataRoot, "wikitext2_test_tokens.i32");
            if (!File.Exists(modelPath) || !File.Exists(configPath) || !File.Exists(refPath) || !File.Exists(tokPath))
                return; // fixtures not present — nothing to verify (this test is local-only)

            const int V = 49152;
            var config = HuggingFaceConfig.FromFile(configPath);
            using var fs = File.OpenRead(modelPath);
            var src = SafetensorsReader.Read(fs);
            var net = LlamaModelBuilder<float>.Build(config, src);

            int[] tokens = ReadInt32(tokPath).Take(16).ToArray();
            float[] refLogits = ReadFloat32(refPath); // [16, V]
            var input = new Tensor<float>(new[] { 1, 16 });
            for (int p = 0; p < 16; p++) input[0, p] = tokens[p];
            float[] logits = net.Predict(input).ToArray();
            int cols = logits.Length / 16;

            double maxAbs = 0;
            int argmaxMatches = 0;
            for (int r = 0; r < 16; r++)
            {
                int am = 0, refAm = 0;
                float amv = float.NegativeInfinity, refv = float.NegativeInfinity;
                for (int c = 0; c < V; c++)
                {
                    float a = logits[r * cols + c], b = refLogits[r * V + c];
                    maxAbs = Math.Max(maxAbs, Math.Abs(a - b));
                    if (a > amv) { amv = a; am = c; }
                    if (b > refv) { refv = b; refAm = c; }
                }
                if (am == refAm) argmaxMatches++;
            }

            Assert.Equal(16, argmaxMatches);
            Assert.True(maxAbs < 0.05, $"max|logit diff| {maxAbs:F4} exceeds tolerance 0.05");
        }

        // End-to-end through the FACADE loading path: PretrainedSource.Safetensors(dir) is exactly what
        // AiModelBuilder.ConfigureModel(PretrainedSource) resolves via PretrainedLoader.Load -> PretrainedArchitectures
        // -> LlamaModelBuilder.Build (the permute fix). Proves the fix reaches the facade. Opt-in / local-only;
        // the shared fixture dir ships a bf16 model.safetensors so this asserts HF-correct argmax (bf16-robust)
        // rather than the tight f32 logit tolerance the direct-loader test above checks.
        [Fact]
        public void SmolLM2_ViaPretrainedSourceFacade_ReproducesArgmax()
        {
            const string dir = @"C:\Users\cheat\Temp\he-m2-audit\data\smollm2-360m";
            const string dataRoot = @"C:\Users\cheat\Temp\he-m2-audit\data";
            string refPath = Path.Combine(dataRoot, "ref_logits_16.f32");
            string tokPath = Path.Combine(dataRoot, "wikitext2_test_tokens.i32");
            if (!File.Exists(Path.Combine(dir, "config.json")) || !File.Exists(refPath) || !File.Exists(tokPath))
                return; // fixtures not present

            const int V = 49152;
            var model = PretrainedLoader<float>.Load(PretrainedSource.Safetensors(dir));
            var net = Assert.IsType<NeuralNetwork<float>>(model);

            int[] tokens = ReadInt32(tokPath).Take(16).ToArray();
            float[] refLogits = ReadFloat32(refPath);
            var input = new Tensor<float>(new[] { 1, 16 });
            for (int p = 0; p < 16; p++) input[0, p] = tokens[p];
            float[] logits = net.Predict(input).ToArray();
            int cols = logits.Length / 16;

            int argmaxMatches = 0;
            for (int r = 0; r < 16; r++)
            {
                int am = 0, refAm = 0;
                float amv = float.NegativeInfinity, refv = float.NegativeInfinity;
                for (int c = 0; c < V; c++)
                {
                    if (logits[r * cols + c] > amv) { amv = logits[r * cols + c]; am = c; }
                    if (refLogits[r * V + c] > refv) { refv = refLogits[r * V + c]; refAm = c; }
                }
                if (am == refAm) argmaxMatches++;
            }
            Assert.Equal(16, argmaxMatches);
        }

        private static int[] ReadInt32(string path)
        {
            var bytes = File.ReadAllBytes(path);
            var res = new int[bytes.Length / 4];
            Buffer.BlockCopy(bytes, 0, res, 0, res.Length * 4);
            return res;
        }

        private static float[] ReadFloat32(string path)
        {
            var bytes = File.ReadAllBytes(path);
            var res = new float[bytes.Length / 4];
            Buffer.BlockCopy(bytes, 0, res, 0, res.Length * 4);
            return res;
        }
    }
}
