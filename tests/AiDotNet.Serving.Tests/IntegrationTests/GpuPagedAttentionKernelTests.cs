using System;
using AiDotNet.DistributedTraining;
using AiDotNet.Inference;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Serving.Tests.IntegrationTests;

/// <summary>
/// GPU tensor-parallel serving: verifies (1) the Tensors GPU paged-attention DECODE kernel matches a CPU
/// scaled-dot-product-attention oracle, and (2) a <see cref="TensorParallelPagedModel{T}"/> run in GPU mode
/// (per-rank device paged attention) matches the CPU (double) model within float tolerance. Skips when no GPU is
/// present on the box.
/// </summary>
public sealed class GpuPagedAttentionKernelTests
{
    [Fact]
    public void GpuPagedAttentionDecode_MatchesCpuOracle()
    {
        if (AiDotNetEngine.Current is not DirectGpuTensorEngine gpu || !gpu.IsGpuAvailable)
            return; // no GPU on this box — nothing to verify

        var backend = gpu.GetBackend();

        const int heads = 2, headDim = 4, blockSize = 16, seqLen = 5;
        int dim = heads * headDim;
        float scale = 1.0f / MathF.Sqrt(headDim);
        var rng = new Random(42);

        var cache = new DevicePagedKVCache(backend, maxBlocks: 8, blockSize: blockSize, heads: heads, headDim: headDim);
        int seqId = 1;

        var keys = new float[seqLen][];
        var vals = new float[seqLen][];
        for (int t = 0; t < seqLen; t++)
        {
            keys[t] = RandVec(rng, dim);
            vals[t] = RandVec(rng, dim);
            cache.Append(seqId, keys[t], vals[t]);
        }

        var query = RandVec(rng, dim);
        var qBuf = backend.AllocateBuffer(query);
        // Kernel via the public IDirectGpuBackend interface (0.116.0); it RETURNS the output buffer (q is input-only).
        var outBuf = backend.PagedAttentionDecode(
            qBuf, cache.KeyBlocks, cache.ValueBlocks, cache.GetBlockTableBuffer(seqId),
            heads, headDim, blockSize, seqLen, scale);
        var gpuOut = backend.DownloadBuffer(outBuf);

        // CPU oracle: per head, softmax(scale * q·k_j) · v_j over j.
        var cpuOut = new float[dim];
        for (int h = 0; h < heads; h++)
        {
            int off = h * headDim;
            var scores = new float[seqLen];
            float max = float.NegativeInfinity;
            for (int j = 0; j < seqLen; j++)
            {
                float dot = 0f;
                for (int d = 0; d < headDim; d++) dot += query[off + d] * keys[j][off + d];
                scores[j] = dot * scale;
                if (scores[j] > max) max = scores[j];
            }
            float sum = 0f;
            for (int j = 0; j < seqLen; j++) { scores[j] = MathF.Exp(scores[j] - max); sum += scores[j]; }
            for (int d = 0; d < headDim; d++)
            {
                float acc = 0f;
                for (int j = 0; j < seqLen; j++) acc += scores[j] / sum * vals[j][off + d];
                cpuOut[off + d] = acc;
            }
        }

        for (int i = 0; i < dim; i++)
            Assert.True(MathF.Abs(gpuOut[i] - cpuOut[i]) < 1e-4f,
                $"index {i}: gpu={gpuOut[i]} cpu={cpuOut[i]}");
    }

    private static float[] RandVec(Random rng, int n)
    {
        var v = new float[n];
        for (int i = 0; i < n; i++) v[i] = (float)(rng.NextDouble() * 2 - 1);
        return v;
    }

    private const int Vocab = 20, EmbedDim = 16, NumHeads = 4, NumLayers = 2, FfnDim = 32;

    [Fact]
    public void GpuTensorParallelModel_MatchesCpuModel_WithinFloatTolerance()
    {
        if (!GpuPagedAttention.IsAvailable)
            return; // no GPU on this box

        var rng = RandomHelper.CreateSeededRandom(31415);
        var embedding = RandT(rng, Vocab, EmbedDim);
        var lmHead = RandT(rng, Vocab, EmbedDim);
        var finalGamma = RandGamma(rng, EmbedDim);
        var layers = new TensorParallelLayerWeights<double>[NumLayers];
        for (int l = 0; l < NumLayers; l++)
            layers[l] = new TensorParallelLayerWeights<double>
            {
                QWeight = RandT(rng, EmbedDim, EmbedDim), QBias = RandT(rng, EmbedDim),
                KWeight = RandT(rng, EmbedDim, EmbedDim), KBias = RandT(rng, EmbedDim),
                VWeight = RandT(rng, EmbedDim, EmbedDim), VBias = RandT(rng, EmbedDim),
                OWeight = RandT(rng, EmbedDim, EmbedDim), OBias = RandT(rng, EmbedDim),
                UpWeight = RandT(rng, FfnDim, EmbedDim), UpBias = RandT(rng, FfnDim),
                DownWeight = RandT(rng, EmbedDim, FfnDim), DownBias = RandT(rng, EmbedDim),
                Norm1Gamma = RandGamma(rng, EmbedDim), Norm2Gamma = RandGamma(rng, EmbedDim),
            };

        var prompt = new Tensor<double>(new[] { 1, 4 });
        prompt[0, 0] = 1; prompt[0, 1] = 5; prompt[0, 2] = 2; prompt[0, 3] = 9;
        var ctx = new InferenceForwardContext(sequenceId: 1, position: 0);

        int[] gpuArgmax = RunModel(worldSize: 2, useGpu: true, embedding, lmHead, layers, finalGamma, prompt, ctx, out bool gpuUsed);
        if (!gpuUsed) return; // GPU path did not engage
        int[] cpuArgmax = RunModel(worldSize: 2, useGpu: false, embedding, lmHead, layers, finalGamma, prompt, ctx, out _);

        Assert.Equal(cpuArgmax, gpuArgmax); // same next-token choices (FP32 attention vs FP64 CPU, tolerance in argmax)
    }

    private static int[] RunModel(
        int worldSize, bool useGpu, Tensor<double> embedding, Tensor<double> lmHead,
        TensorParallelLayerWeights<double>[] layers, Tensor<double> finalGamma,
        Tensor<double> prompt, InferenceForwardContext ctx, out bool gpuUsed)
    {
        Func<double, double> gelu = v => 0.5 * v * (1.0 + Math.Tanh(0.7978845608028654 * (v + 0.044715 * v * v * v)));
        var model = new TensorParallelPagedModel<double>(
            worldSize, EmbedDim, NumHeads, NumLayers, FfnDim, Vocab,
            useRmsNorm: true, finalNormGamma: finalGamma, ffnActivation: gelu, useGpu: useGpu);
        model.SetFromFullWeights(embedding, lmHead, layers);
        gpuUsed = model.GpuActive;
        if (!useGpu)
            for (int r = 0; r < model.RankCaches.Length; r++) model.RankCaches[r].AllocateSequence(1, prompt.Shape[1]);
        var logits = model.PredictWithContext(prompt, ctx);
        int seq = logits.Shape[1], vocab = logits.Shape[^1];
        var argmax = new int[seq];
        for (int s = 0; s < seq; s++)
        {
            int best = 0; double bv = double.NegativeInfinity;
            for (int vch = 0; vch < vocab; vch++) if (logits[0, s, vch] > bv) { bv = logits[0, s, vch]; best = vch; }
            argmax[s] = best;
        }
        return argmax;
    }

    private static Tensor<double> RandT(System.Random rng, params int[] shape)
    {
        var t = new Tensor<double>(shape);
        if (shape.Length == 1) for (int i = 0; i < shape[0]; i++) t[i] = rng.NextDouble() * 2 - 1;
        else for (int i = 0; i < shape[0]; i++) for (int j = 0; j < shape[1]; j++) t[i, j] = rng.NextDouble() * 2 - 1;
        return t;
    }

    private static Tensor<double> RandGamma(System.Random rng, int dim)
    {
        var t = new Tensor<double>(new[] { dim });
        for (int i = 0; i < dim; i++) t[i] = 0.5 + rng.NextDouble();
        return t;
    }
}
