using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Kairos Mixture-of-Size patch embedder: emits N parallel patch-size paths through the
/// SAME transformer backbone shape (numPatches varies per path but hiddenDim is fixed),
/// then combines them via a learned router that weights each path per-input. This is the
/// Mixture-of-Size analog of Mixture-of-Experts: the "experts" are alternative
/// tokenization granularities rather than alternative FFNs.
/// </summary>
/// <remarks>
/// <para>
/// Forward:
/// </para>
/// <list type="number">
/// <item><description>Compute mean of the raw input signal as router input (a global
///   summary that does not depend on patch size).</description></item>
/// <item><description>Router Dense(contextLength -> N) + softmax produces per-path
///   weights <c>w[B, N]</c>.</description></item>
/// <item><description>For each path <c>k</c>: reshape input to
///   <c>[B, patches_k, patchSize_k]</c> → Dense(patchSize_k -> hiddenDim) → pool over
///   patches → <c>[B, hiddenDim]</c>. The pooling collapses variable-length patch
///   sequences into a fixed [B, hiddenDim] representation so the router can combine
///   them.</description></item>
/// <item><description>Weighted combine: output = Σ_k w[k] · pooled_k →
///   <c>[B, hiddenDim]</c>.</description></item>
/// </list>
/// <para>
/// This layer outputs a single summarized [B, hiddenDim] tensor per input, ready to feed
/// a downstream transformer stack that treats each input as a single token (since
/// multi-size pooling has already collapsed the sequence). For the full per-patch
/// sequence interpretation, the caller should use only one patch size.
/// </para>
/// </remarks>
/// <typeparam name="T">Numeric element type.</typeparam>
[LayerCategory(LayerCategory.MixtureOfExperts)]
[LayerCategory(LayerCategory.Structural)]
[LayerTask(LayerTask.Routing)]
[LayerTask(LayerTask.Projection)]
[LayerProperty(IsTrainable = true, ChangesShape = true, TestInputShape = "16", TestConstructorArgs = "16, 4, new int[] { 4, 8 }")]
public class KairosMultiSizePatchLayer<T> : LayerBase<T>
{
    private readonly int _contextLength;
    private readonly int _hiddenDim;
    private readonly int[] _patchSizes;

    private readonly List<DenseLayer<T>> _patchEmbeddings;
    private readonly DenseLayer<T> _router;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new <see cref="KairosMultiSizePatchLayer{T}"/>.
    /// </summary>
    /// <param name="contextLength">Input sequence length.</param>
    /// <param name="hiddenDim">Fixed hidden dimension of every patch-size path.</param>
    /// <param name="patchSizes">
    /// Array of patch sizes to route over. Must have at least two entries. Each patch size
    /// must be positive and must divide evenly into <paramref name="contextLength"/>.
    /// </param>
    public KairosMultiSizePatchLayer(int contextLength, int hiddenDim, int[] patchSizes)
        : base(new[] { contextLength }, new[] { hiddenDim })
    {
        if (contextLength < 1) throw new ArgumentOutOfRangeException(nameof(contextLength));
        if (hiddenDim < 1) throw new ArgumentOutOfRangeException(nameof(hiddenDim));
        if (patchSizes is null) throw new ArgumentNullException(nameof(patchSizes));
        // Accept the one-path fallback: LayerHelper.CreateDefaultKairosLayers filters
        // patch sizes larger than contextLength, so small contexts can legitimately
        // reduce the default [8, 16, 32, 64] down to a single surviving size. The
        // forward path already degenerates correctly when patchSizes.Length == 1
        // (router collapses to a single active expert).
        if (patchSizes.Length < 1)
            throw new ArgumentException("At least one patch size is required.", nameof(patchSizes));

        _contextLength = contextLength;
        _hiddenDim = hiddenDim;
        _patchSizes = (int[])patchSizes.Clone();

        // Per-path patch embedding Dense(patchSize → hiddenDim). DenseLayer operates
        // on the last axis of its input, so we feed [B, patches_k, patchSize_k] and
        // get [B, patches_k, hiddenDim].
        _patchEmbeddings = new List<DenseLayer<T>>(patchSizes.Length);
        foreach (int ps in _patchSizes)
        {
            if (ps < 1)
                throw new ArgumentOutOfRangeException(nameof(patchSizes),
                    $"All patch sizes must be positive; got {ps}.");
            if (contextLength % ps != 0)
                throw new ArgumentException(
                    $"contextLength ({contextLength}) must be divisible by every patch size; {ps} does not divide.",
                    nameof(patchSizes));

            _patchEmbeddings.Add(new DenseLayer<T>(
                outputSize: hiddenDim,
                activationFunction: null));
        }

        // Router: [B, contextLength] -> [B, numPatchSizes] softmax weights.
        _router = new DenseLayer<T>(
            outputSize: _patchSizes.Length,
            activationFunction: new SoftmaxActivation<T>());

        foreach (var emb in _patchEmbeddings)
            RegisterSubLayer(emb);
        RegisterSubLayer(_router);
    }

    /// <inheritdoc/>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Support both [B, contextLength] and [contextLength] inputs.
        bool addedBatch = false;
        if (input.Rank == 1)
        {
            input = Engine.Reshape(input, new[] { 1, input.Shape[0] });
            addedBatch = true;
        }
        int batchSize = input.Shape[0];

        // Router over flat input.
        var routerWeights = _router.Forward(input); // [B, numPatchSizes]

        // For each patch size: reshape input → [B, patches_k, patchSize_k], embed →
        // [B, patches_k, hiddenDim], mean-pool over patches → [B, hiddenDim].
        var pathOutputs = new List<Tensor<T>>(_patchSizes.Length);
        for (int k = 0; k < _patchSizes.Length; k++)
        {
            int ps = _patchSizes[k];
            int patches_k = _contextLength / ps;
            var reshaped = Engine.Reshape(input, new[] { batchSize, patches_k, ps });
            var embedded = _patchEmbeddings[k].Forward(reshaped); // [B, patches_k, hiddenDim]
            var pooled = Engine.ReduceMean(embedded, new[] { 1 }, keepDims: false); // [B, hiddenDim]
            pathOutputs.Add(pooled);
        }

        // Weighted combine: output = Σ_k routerWeights[:, k] · pathOutputs[k].
        // Built entirely out of Engine ops so the gradient tape records the mixture
        // and gradients flow back into both the router and every patch-embedding
        // Dense. Previously the loop used .Data.Span reads/writes which produced
        // a fresh Tensor disconnected from the tape — _router and
        // _patchEmbeddings would stop learning through this layer.
        Tensor<T>? output = null;
        for (int k = 0; k < _patchSizes.Length; k++)
        {
            // Select per-path router weight slice [B, 1] and multiply-broadcast
            // with pathOutputs[k] ([B, hiddenDim]). Engine.TensorNarrow keeps
            // rank so the result is [B, 1] which broadcasts correctly across
            // hiddenDim.
            var routerSlice = Engine.TensorNarrow(routerWeights, dim: 1, start: k, length: 1); // [B, 1]
            var weighted = Engine.TensorMultiply(routerSlice, pathOutputs[k]); // [B, hiddenDim]
            output = output is null ? weighted : Engine.TensorAdd(output, weighted);
        }

        if (output is null)
            throw new InvalidOperationException(
                "KairosMultiSizePatchLayer forward produced no weighted paths — patchSizes was empty (this should have been rejected in the constructor).");

        if (addedBatch)
            return Engine.Reshape(output, new[] { _hiddenDim });
        return output;
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        _router.UpdateParameters(learningRate);
        foreach (var emb in _patchEmbeddings)
            emb.UpdateParameters(learningRate);
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        var parts = new List<Vector<T>> { _router.GetParameters() };
        foreach (var emb in _patchEmbeddings)
            parts.Add(emb.GetParameters());

        int total = 0;
        foreach (var p in parts) total += p.Length;
        var combined = new T[total];
        int offset = 0;
        foreach (var p in parts)
        {
            for (int i = 0; i < p.Length; i++)
                combined[offset + i] = p[i];
            offset += p.Length;
        }
        return new Vector<T>(combined);
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        _router.ResetState();
        foreach (var emb in _patchEmbeddings)
            emb.ResetState();
    }
}
