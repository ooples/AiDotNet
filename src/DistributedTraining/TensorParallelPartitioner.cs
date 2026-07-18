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

        // Every block's attention must be a plain multi-head attention with a consistent head/dim geometry.
        int embedDim = blocks[0].HiddenSize;
        int ffnDim = blocks[0].FfnDim;
        if (blocks[0].AttentionLayer is not MultiHeadAttentionLayer<T> firstMha)
        {
            reason = "block attention is not MultiHeadAttentionLayer";
            return null;
        }
        int numHeads = firstMha.HeadCount;
        if (numHeads <= 0 || embedDim % numHeads != 0)
        {
            reason = "invalid head geometry";
            return null;
        }
        if (numHeads % worldSize != 0)
        {
            reason = $"numHeads ({numHeads}) not divisible by worldSize ({worldSize})";
            return null;
        }
        if (ffnDim % worldSize != 0)
        {
            reason = $"ffnDim ({ffnDim}) not divisible by worldSize ({worldSize})";
            return null;
        }

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
            if (b.AttentionLayer is not MultiHeadAttentionLayer<T> mha || mha.HeadCount != numHeads)
            {
                reason = $"block {l} attention differs";
                return null;
            }

            var qW = TransposeToOutIn(mha.GetQueryWeights(), embedDim, embedDim);
            var kW = TransposeToOutIn(mha.GetKeyWeights(), embedDim, embedDim);
            var vW = TransposeToOutIn(mha.GetValueWeights(), embedDim, embedDim);
            var oW = TransposeToOutIn(mha.GetOutputWeights(), embedDim, embedDim);
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
                KWeight = kW, KBias = Zeros(embedDim),
                VWeight = vW, VBias = Zeros(embedDim),
                OWeight = oW, OBias = OutputBias(mha, embedDim),
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
            useGpu: useGpu);

        tp.SetFromFullWeights(embMatrix, lmHead, perLayer);
        return tp;
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

    // The MHA output bias is the final embedDim entries of its parameter vector (after Q/K/V/O weights).
    private static Tensor<T> OutputBias(MultiHeadAttentionLayer<T> mha, int embedDim)
    {
        var p = mha.GetParameters();
        var t = new Tensor<T>(new[] { embedDim });
        int start = p.Length - embedDim;
        if (start < 0) return Zeros(embedDim);
        for (int i = 0; i < embedDim; i++) t[i] = p[start + i];
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
