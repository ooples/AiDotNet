using System;
using System.Collections.Generic;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.DistributedTraining;

/// <summary>
/// Builds a tensor-parallel served model (<see cref="TensorParallelPagedModel{T}"/>) from a recognized trained
/// transformer, extracting and sharding its per-layer attention/FFN weights across the ranks. Recognizes the
/// canonical pre-LN decoder stack:
/// <c>[EmbeddingLayer, PreLNTransformerBlock × N, (optional final RMSNormalizationLayer), DenseLayer(lm head)]</c>
/// whose blocks use <see cref="MultiHeadAttentionLayer{T}"/> self-attention. Returns <c>null</c> for any other
/// structure so the serving layer can fall back to the normal (non-tensor-parallel) path.
/// </summary>
internal static class TensorParallelPartitioner<T>
{
    /// <summary>
    /// Attempts to partition <paramref name="model"/> into a tensor-parallel served model of the given world size.
    /// Returns null (with <paramref name="reason"/> set) when the model is not a recognized tensor-parallelizable
    /// transformer, or when its head count / FFN width are not divisible by <paramref name="worldSize"/>.
    /// </summary>
    public static TensorParallelPagedModel<T>? TryBuild(
        NeuralNetworkBase<T> model, int worldSize, int blockSize, int numBlocks, out string reason, bool useGpu = false)
    {
        reason = string.Empty;
        if (model is null) { reason = "model is null"; return null; }
        if (worldSize < 2) { reason = "worldSize < 2"; return null; }

        var layers = model.Layers;
        if (layers is null || layers.Count < 3)
        {
            reason = "too few layers for a transformer stack";
            return null;
        }

        if (layers[0] is not EmbeddingLayer<T> embedding)
        {
            reason = "first layer is not an EmbeddingLayer";
            return null;
        }

        var blocks = new List<PreLNTransformerBlock<T>>();
        int idx = 1;
        while (idx < layers.Count && layers[idx] is PreLNTransformerBlock<T> block)
        {
            blocks.Add(block);
            idx++;
        }
        if (blocks.Count == 0)
        {
            reason = "no PreLNTransformerBlock layers found";
            return null;
        }

        RMSNormalizationLayer<T>? finalNorm = null;
        if (idx < layers.Count && layers[idx] is RMSNormalizationLayer<T> fn)
        {
            finalNorm = fn;
            idx++;
        }

        if (idx >= layers.Count || layers[idx] is not DenseLayer<T> head)
        {
            reason = "no DenseLayer language-model head after the blocks";
            return null;
        }
        idx++;
        if (idx != layers.Count)
        {
            reason = $"unexpected trailing layers ({layers.Count - idx})";
            return null;
        }

        // Every block's attention must be a plain multi-head OR grouped-query attention with consistent geometry.
        int embedDim = blocks[0].HiddenSize;
        int ffnDim = blocks[0].FfnDim;
        if (!TryReadHeads(blocks[0].AttentionLayer, out int numHeads, out int numKVHeads))
        {
            reason = "block attention is not MultiHeadAttentionLayer or GroupedQueryAttentionLayer";
            return null;
        }
        if (numHeads <= 0 || embedDim % numHeads != 0)
        {
            reason = "invalid head geometry";
            return null;
        }
        if (numKVHeads <= 0 || numHeads % numKVHeads != 0)
        {
            reason = $"numHeads ({numHeads}) must be a multiple of numKVHeads ({numKVHeads})";
            return null;
        }
        if (numHeads % worldSize != 0)
        {
            reason = $"numHeads ({numHeads}) not divisible by worldSize ({worldSize})";
            return null;
        }
        if (numKVHeads % worldSize != 0)
        {
            reason = $"numKVHeads ({numKVHeads}) not divisible by worldSize ({worldSize})";
            return null;
        }
        if (ffnDim % worldSize != 0)
        {
            reason = $"ffnDim ({ffnDim}) not divisible by worldSize ({worldSize})";
            return null;
        }
        int headDim = embedDim / numHeads;
        int kvDim = numKVHeads * headDim;

        int vocab = MaxTokenId(embedding) + 1;

        // Extract embedding + lm-head matrices (both [vocab, embedDim] in the TP model's convention).
        var embMatrix = ExtractEmbedding(embedding, vocab, embedDim);
        if (embMatrix is null) { reason = "could not read embedding matrix"; return null; }
        var headW = head.GetWeights();
        var lmHead = TransposeToOutIn(headW, vocab, embedDim);
        if (lmHead is null)
        {
            reason = $"lm head weight shape mismatch: got [{string.Join(",", headW.Shape)}], expected [{embedDim},{vocab}] (in,out); vocab={vocab}";
            return null;
        }

        var perLayer = new TensorParallelLayerWeights<T>[blocks.Count];
        for (int l = 0; l < blocks.Count; l++)
        {
            var b = blocks[l];
            if (b.HiddenSize != embedDim || b.FfnDim != ffnDim)
            {
                reason = $"block {l} geometry differs from block 0";
                return null;
            }
            if (!TryReadHeads(b.AttentionLayer, out int bHeads, out int bKVHeads) || bHeads != numHeads || bKVHeads != numKVHeads)
            {
                reason = $"block {l} attention differs";
                return null;
            }

            AttnWeights(b.AttentionLayer, out var qWraw, out var kWraw, out var vWraw, out var oWraw, out var oBias);
            var qW = TransposeToOutIn(qWraw, embedDim, embedDim);   // Q: [embDim, numHeads*headDim=embDim]
            var kW = TransposeToOutIn(kWraw, kvDim, embedDim);      // K: [embDim, numKVHeads*headDim=kvDim]
            var vW = TransposeToOutIn(vWraw, kvDim, embedDim);      // V: same as K
            var oW = TransposeToOutIn(oWraw, embedDim, embedDim);   // O: [numHeads*headDim=embDim, embDim]
            var upW = TransposeToOutIn(b.FfnUp.GetWeights(), ffnDim, embedDim);
            var downW = TransposeToOutIn(b.FfnDown.GetWeights(), embedDim, ffnDim);
            if (qW is null || kW is null || vW is null || oW is null || upW is null || downW is null)
            {
                reason = $"block {l} weight shape mismatch";
                return null;
            }

            perLayer[l] = new TensorParallelLayerWeights<T>
            {
                QWeight = qW, QBias = Zeros(embedDim),
                KWeight = kW, KBias = Zeros(kvDim),
                VWeight = vW, VBias = Zeros(kvDim),
                OWeight = oW, OBias = oBias,
                UpWeight = upW, UpBias = DenseBias(b.FfnUp, ffnDim),
                DownWeight = downW, DownBias = DenseBias(b.FfnDown, embedDim),
                Norm1Gamma = b.Norm1.GetGammaTensor(),
                Norm2Gamma = b.Norm2.GetGammaTensor(),
            };
        }

        // Use the block's REAL activation function so the sharded FFN matches the trained model exactly.
        var numOps = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        var activationFn = blocks[0].FfnActivation;
        Func<double, double> ffnActivation = v => Convert.ToDouble(activationFn.Activate(numOps.FromDouble(v)));

        var tp = new TensorParallelPagedModel<T>(
            worldSize, embedDim, numHeads, blocks.Count, ffnDim, vocab,
            blockSize: blockSize, numBlocks: numBlocks,
            useRmsNorm: true,
            finalNormGamma: finalNorm?.GetGammaTensor(),
            ffnActivation: ffnActivation,
            rmsNormEpsilon: Convert.ToDouble(blocks[0].Norm1.GetEpsilon()),
            lmHeadBias: DenseBias(head, vocab),
            useGpu: useGpu,
            numKVHeads: numKVHeads);

        tp.SetFromFullWeights(embMatrix, lmHead, perLayer);
        return tp;
    }

    // Reads (query heads, KV heads) from a multi-head or grouped-query attention layer; false for other types.
    private static bool TryReadHeads(LayerBase<T> attention, out int numHeads, out int numKVHeads)
    {
        switch (attention)
        {
            case MultiHeadAttentionLayer<T> mha:
                numHeads = mha.HeadCount; numKVHeads = mha.HeadCount; return true;
            case GroupedQueryAttentionLayer<T> gqa:
                numHeads = gqa.NumHeads; numKVHeads = gqa.NumKVHeads; return true;
            default:
                numHeads = 0; numKVHeads = 0; return false;
        }
    }

    // Extracts the Q/K/V/O weight tensors (stored [in, out]) + output bias from either attention type.
    private static void AttnWeights(
        LayerBase<T> attention, out Tensor<T> qW, out Tensor<T> kW, out Tensor<T> vW, out Tensor<T> oW, out Tensor<T> oBias)
    {
        switch (attention)
        {
            case MultiHeadAttentionLayer<T> mha:
                qW = mha.GetQueryWeights(); kW = mha.GetKeyWeights(); vW = mha.GetValueWeights(); oW = mha.GetOutputWeights();
                oBias = ParamTail(mha.GetParameters(), mha.GetOutputWeights().Shape[1]);
                return;
            case GroupedQueryAttentionLayer<T> gqa:
                qW = gqa.GetQueryWeights(); kW = gqa.GetKeyWeights(); vW = gqa.GetValueWeights(); oW = gqa.GetOutputWeights();
                oBias = ParamTail(gqa.GetParameters(), gqa.GetOutputWeights().Shape[1]);
                return;
            default:
                throw new System.ArgumentException($"Unsupported attention type {attention.GetType().Name}.");
        }
    }

    // The output bias is the final `count` entries of the layer's parameter vector (after the weight matrices).
    private static Tensor<T> ParamTail(Vector<T> p, int count)
    {
        var t = new Tensor<T>(new[] { count });
        int start = p.Length - count;
        if (start < 0) return Zeros(count);
        for (int i = 0; i < count; i++) t[i] = p[start + i];
        return t;
    }

    // Reads the full [vocab, embedDim] embedding matrix by looking up every token id.
    private static Tensor<T>? ExtractEmbedding(EmbeddingLayer<T> embedding, int vocab, int embedDim)
    {
        var ids = new int[vocab];
        for (int i = 0; i < vocab; i++) ids[i] = i;
        Matrix<T> table;
        try { table = embedding.GetTokenEmbeddings(ids); }
        catch { return null; }
        if (table.Rows != vocab || table.Columns != embedDim) return null;
        var t = new Tensor<T>(new[] { vocab, embedDim });
        for (int v = 0; v < vocab; v++)
            for (int d = 0; d < embedDim; d++)
                t[v, d] = table[v, d];
        return t;
    }

    // Transposes a stored [in, out] weight matrix into the TP model's [out, in] row-major convention.
    private static Tensor<T>? TransposeToOutIn(Tensor<T> weight, int outDim, int inDim)
    {
        if (weight.Rank != 2) return null;
        // Stored convention is [inputSize, outputSize].
        if (weight.Shape[0] != inDim || weight.Shape[1] != outDim) return null;
        var t = new Tensor<T>(new[] { outDim, inDim });
        for (int o = 0; o < outDim; o++)
            for (int i = 0; i < inDim; i++)
                t[o, i] = weight[i, o];
        return t;
    }

    private static Tensor<T> Zeros(int n)
    {
        var t = new Tensor<T>(new[] { n });
        var numOps = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < n; i++) t[i] = numOps.Zero;
        return t;
    }

    // A DenseLayer bias is the final outDim entries of its parameter vector (after the weight matrix).
    private static Tensor<T> DenseBias(DenseLayer<T> dense, int outDim)
    {
        var p = dense.GetParameters();
        var t = new Tensor<T>(new[] { outDim });
        int start = p.Length - outDim;
        if (start < 0) return Zeros(outDim);
        for (int i = 0; i < outDim; i++) t[i] = p[start + i];
        return t;
    }

    private static int MaxTokenId(EmbeddingLayer<T> embedding)
    {
        // VocabularySize is exposed via the internal metadata (the layer has no public int accessor).
        var meta = embedding.GetMetadata();
        if (meta.TryGetValue("VocabularySize", out var vs) && int.TryParse(vs, out var v) && v > 0)
            return v - 1;
        return -1;
    }
}
