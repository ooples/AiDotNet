using AiDotNet.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.Layers.SSM;

/// <summary>
/// The per-sequence decoding state of a <see cref="MultiLatentAttentionLayer{T}"/>: the compressed latent of every
/// position consumed so far (the MLA cache, <c>latentDim</c> values per position instead of full keys and values).
/// Create it with <see cref="MultiLatentAttentionLayer{T}.CreateStepState"/>.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public sealed class MultiLatentAttentionStepState<T>
{
    internal MultiLatentAttentionStepState(int batchSize, int latentDimension)
    {
        BatchSize = batchSize;
        LatentDimension = latentDimension;
    }

    /// <summary>The number of sequences this state decodes.</summary>
    public int BatchSize { get; }

    /// <summary>The number of positions cached so far.</summary>
    public int Position => Latents?.Shape[1] ?? 0;

    internal int LatentDimension { get; }

    // [B, positions, latentDim], or null before the first position.
    internal Tensor<T>? Latents { get; set; }
}

public partial class MultiLatentAttentionLayer<T>
{
    /// <summary>
    /// Creates an empty latent cache for <paramref name="batchSize"/> sequences, for use with <see cref="Step"/>.
    /// </summary>
    /// <param name="batchSize">The number of sequences decoded together.</param>
    /// <returns>An empty cache, equivalent to the start of a sequence.</returns>
    public MultiLatentAttentionStepState<T> CreateStepState(int batchSize = 1)
    {
        if (batchSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(batchSize), batchSize, "Batch size must be positive.");
        return new MultiLatentAttentionStepState<T>(batchSize, _latentDimension);
    }

    /// <summary>
    /// Runs the layer on ONE new position per sequence against the cached latents of the earlier positions, giving the
    /// same output the full causal <c>Forward</c> produces at that position, and appends this position's latent.
    /// </summary>
    /// <param name="input">This position's input, <c>[batch, 1, modelDim]</c> or <c>[batch, modelDim]</c>.</param>
    /// <param name="state">The cache from <see cref="CreateStepState"/>, advanced in place.</param>
    /// <returns>The output at this position, in the input's shape.</returns>
    /// <remarks>
    /// <para>Uses the absorbed form of multi-head latent attention (DeepSeek-V2 §2.1.2): with latents <c>c_s</c> and
    /// per-head up-projections <c>W_K^h</c>, <c>W_V^h</c>, the score is <c>q_hᵀ W_K^hᵀ c_s = (W_K^h q_h)ᵀ c_s</c>,
    /// and the head output is <c>W_V^hᵀ Σ_s p_s c_s</c>. So each step reads only the cached latents: per-token cost
    /// O(positions × heads × latentDim), and keys and values are never re-expanded.</para>
    /// <para><b>For Beginners:</b> Instead of remembering every past word's full key and value, this layer remembers
    /// a small compressed summary of each word and does the attention directly against those summaries.</para>
    /// </remarks>
    public Tensor<T> Step(Tensor<T> input, MultiLatentAttentionStepState<T> state)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (state is null) throw new ArgumentNullException(nameof(state));
        int b = state.BatchSize, d = _modelDimension, h = _numHeads, hd = _headDimension, l = _latentDimension;
        bool rank3 = input.Shape.Length == 3;
        if ((rank3 && (input.Shape[0] != b || input.Shape[1] != 1 || input.Shape[2] != d))
            || (!rank3 && (input.Shape.Length != 2 || input.Shape[0] != b || input.Shape[1] != d)))
            throw new ArgumentException(
                $"Step expects one position per sequence, [{b}, 1, {d}] or [{b}, {d}]; got [{string.Join(",", input.Shape)}].", nameof(input));
        if (state.LatentDimension != l)
            throw new ArgumentException("The state was created by a layer of a different shape.", nameof(state));

        var x = Engine.Reshape(input, new[] { b, d });
        var latent = Engine.TensorAdd(Engine.TensorMatMul(x, _compressWeights), Engine.Reshape(_compressBias, new[] { 1, l }));
        var latent3 = Engine.Reshape(latent, new[] { b, 1, l });
        state.Latents = state.Latents is null ? latent3 : Engine.TensorConcatenate(new[] { state.Latents, latent3 }, 1);
        var cache = state.Latents;                                                   // [B, t, L]

        // Queries absorbed into latent space: qLat[b, h] = W_K^h q[b, h], W_K^h = keyUp[:, h·hd:(h+1)·hd] ([L, hd]).
        var q = Engine.TensorPermute(Engine.Reshape(Engine.TensorMatMul(x, _queryWeights), new[] { b, h, hd }), new[] { 1, 0, 2 });   // [H, B, hd]
        var keyUp = Engine.TensorPermute(Engine.Reshape(_keyUpWeights, new[] { l, h, hd }), new[] { 1, 2, 0 });                     // [H, hd, L]
        var qLat = Engine.TensorPermute(Engine.BatchMatMul(q, keyUp), new[] { 1, 0, 2 });                                           // [B, H, L]

        var scores = Engine.TensorMultiplyScalar(Engine.BatchMatMul(qLat, Engine.TensorPermute(cache, new[] { 0, 2, 1 })),
            NumOps.FromDouble(1.0 / Math.Sqrt(hd)));                                                                                    // [B, H, t]
        var weights = Engine.Softmax(scores, -1);
        var context = Engine.TensorPermute(Engine.BatchMatMul(weights, cache), new[] { 1, 0, 2 });                                  // [H, B, L]
        var valueUp = Engine.TensorPermute(Engine.Reshape(_valueUpWeights, new[] { l, h, hd }), new[] { 1, 0, 2 });                 // [H, L, hd]
        var attn = Engine.Reshape(Engine.TensorPermute(Engine.BatchMatMul(context, valueUp), new[] { 1, 0, 2 }), new[] { b, d });   // [B, D]

        var gate = Engine.Swish(Engine.TensorAdd(Engine.TensorMatMul(x, _outputGateWeights), Engine.Reshape(_outputGateBias, new[] { 1, d })));
        var projected = Engine.TensorAdd(Engine.TensorMatMul(Engine.TensorMultiply(attn, gate), _outputProjectionWeights),
            Engine.Reshape(_outputProjectionBias, new[] { 1, d }));
        return Engine.Reshape(ApplyActivation(projected), input._shape);
    }
}
