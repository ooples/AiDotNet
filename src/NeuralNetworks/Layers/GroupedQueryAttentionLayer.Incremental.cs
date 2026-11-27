using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// The per-sequence decoding state of a <see cref="GroupedQueryAttentionLayer{T}"/>: the keys and values of every
/// position consumed so far, <c>numKVHeads</c> of each per position (the grouped heads are shared, so the cache is
/// <c>numHeads / numKVHeads</c> times smaller than a multi-head one). The cache grows by the positions each
/// <see cref="GroupedQueryAttentionLayer{T}.Step"/> consumes, as a PyTorch dynamic key-value cache does. Create it
/// with <see cref="GroupedQueryAttentionLayer{T}.CreateStepState(int)"/>.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public sealed class GroupedQueryAttentionStepState<T>
{
    internal GroupedQueryAttentionStepState(int batchSize, int kvHeads, int headDimension)
    {
        BatchSize = batchSize;
        KVHeads = kvHeads;
        HeadDimension = headDimension;
    }

    /// <summary>The number of sequences this state decodes.</summary>
    public int BatchSize { get; }

    /// <summary>The number of positions cached: the position the next step's first input takes.</summary>
    public int Position { get; internal set; }

    internal int KVHeads { get; }

    internal int HeadDimension { get; }

    // [batch, numKVHeads, Position, headDim], rotary-encoded keys as attention reads them; null before the first step.
    internal Tensor<T>? Keys { get; set; }

    // [batch, numKVHeads, Position, headDim]; null before the first step.
    internal Tensor<T>? Values { get; set; }
}

public partial class GroupedQueryAttentionLayer<T>
{
    /// <summary>
    /// Creates an empty key-value cache for <paramref name="batchSize"/> sequences, for use with <see cref="Step"/>.
    /// </summary>
    /// <param name="batchSize">The number of sequences decoded together.</param>
    /// <returns>An empty cache, equivalent to the start of a sequence.</returns>
    public GroupedQueryAttentionStepState<T> CreateStepState(int batchSize = 1)
    {
        if (batchSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(batchSize), batchSize, "Batch size must be positive.");
        return new GroupedQueryAttentionStepState<T>(batchSize, _numKVHeads, _headDimension);
    }

    /// <summary>
    /// Runs the layer on the next positions of each sequence against the cached keys and values of the earlier ones,
    /// giving the output the full <c>Forward</c> over the whole sequence produces at those positions, and appends
    /// their keys and values to the cache.
    /// </summary>
    /// <param name="input">The new positions' input, <c>[batch, newPositions, modelDim]</c>, or <c>[batch, modelDim]</c>
    /// for one position. A prompt can go in one step and each generated token in the next ones.</param>
    /// <param name="state">The cache from <see cref="CreateStepState"/>, advanced in place.</param>
    /// <returns>The output at the new positions, in the input's shape.</returns>
    /// <remarks>
    /// <para>The new positions' rotary encoding starts at <see cref="GroupedQueryAttentionStepState{T}.Position"/>; a
    /// causal layer lets new position <c>i</c> attend to every cached position and to new positions up to <c>i</c>,
    /// and a non-causal one to all of them. The cache keeps the <c>numKVHeads</c> key and value heads; the query heads
    /// of a group share them in the attention kernel, so they are never repeated.</para>
    /// <para><b>For Beginners:</b> When a model writes text one word at a time, each new word only needs its own
    /// keys and values computed; the earlier words' ones are remembered here instead of being recomputed every time.</para>
    /// </remarks>
    /// <exception cref="NotSupportedException">The layer uses an ALiBi positional bias.</exception>
    public Tensor<T> Step(Tensor<T> input, GroupedQueryAttentionStepState<T> state)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (state is null) throw new ArgumentNullException(nameof(state));
        if (_alibiLayer is not null)
            throw new NotSupportedException("Step does not support an ALiBi positional bias; use Forward over the whole sequence.");
        EnsureWeightsMaterialized();
        int b = state.BatchSize, d = _embeddingDimension;
        bool rank3 = input.Shape.Length == 3 && input.Shape[2] == d && input.Shape[1] > 0;
        bool rank2 = input.Shape.Length == 2 && input.Shape[1] == d;
        if (input.Shape[0] != b || !(rank3 || rank2))
            throw new ArgumentException(
                $"Step expects [{b}, positions, {d}] or [{b}, {d}]; got [{string.Join(",", input.Shape)}].", nameof(input));
        if (state.KVHeads != _numKVHeads || state.HeadDimension != _headDimension)
            throw new ArgumentException("The state was created by a layer of a different shape.", nameof(state));
        int s = rank3 ? input.Shape[1] : 1;
        int start = state.Position;

        var x = Engine.Reshape(input, new[] { b * s, d });
        var qFlat = Engine.TensorMatMul(x, _queryWeights);
        var kFlat = Engine.TensorMatMul(x, _keyWeights);
        var vFlat = Engine.TensorMatMul(x, _valueWeights);
        if (_queryBias.Length > 0)
        {
            qFlat = Engine.TensorAdd(qFlat, Engine.Reshape(_queryBias, new[] { 1, _numHeads * _headDimension }));
            kFlat = Engine.TensorAdd(kFlat, Engine.Reshape(_keyBias, new[] { 1, _numKVHeads * _headDimension }));
            vFlat = Engine.TensorAdd(vFlat, Engine.Reshape(_valueBias, new[] { 1, _numKVHeads * _headDimension }));
        }
        if (_queryProjectionNorm is not null && _keyProjectionNorm is not null)
        {
            qFlat = Engine.RMSNorm(qFlat, _queryProjectionNorm, _projectionNormEpsilon, out _);
            kFlat = Engine.RMSNorm(kFlat, _keyProjectionNorm, _projectionNormEpsilon, out _);
        }

        var queries = Engine.TensorPermute(Engine.Reshape(qFlat, new[] { b, s, _numHeads, _headDimension }), new[] { 0, 2, 1, 3 });
        var keys = Engine.TensorPermute(Engine.Reshape(kFlat, new[] { b, s, _numKVHeads, _headDimension }), new[] { 0, 2, 1, 3 });
        var values = Engine.TensorPermute(Engine.Reshape(vFlat, new[] { b, s, _numKVHeads, _headDimension }), new[] { 0, 2, 1, 3 });
        if (_ropeLayer is not null)
            (queries, keys) = _ropeLayer.ApplyRoPE(queries, keys, startPosition: start);

        var allKeys = state.Keys is null ? keys : Engine.TensorConcatenate(new[] { state.Keys, keys }, 2);
        var allValues = state.Values is null ? values : Engine.TensorConcatenate(new[] { state.Values, values }, 2);

        // Key j is visible to new query i iff j <= i + (cached + new - new): the cached positions and the new ones up
        // to i, the causal offset the kernel applies when there are more keys than queries.
        var context = Engine.ScaledDotProductAttentionGqa(
            queries, allKeys, allValues,
            scale: 1.0 / Math.Sqrt(_headDimension),
            isCausal: _useCausalMask,
            softcap: _attnLogitSoftcap);

        var flat = Engine.Reshape(Engine.TensorPermute(context, new[] { 0, 2, 1, 3 }), new[] { b * s, _numHeads * _headDimension });
        var projected = Engine.TensorAdd(
            Engine.Reshape(Engine.TensorMatMul(flat, _outputWeights), new[] { b, s, d }),
            Engine.Reshape(_outputBias, new[] { 1, 1, d }));
        var output = StepActivation(projected);

        state.Keys = allKeys;
        state.Values = allValues;
        state.Position = start + s;
        return Engine.Reshape(output, input._shape);
    }

    // A step is never followed by the eager backward that drains the layer's caches, and a new layer is in training
    // mode by default, so outside a tape the activation runs in a no-grad scope, which retains nothing. Under a tape the
    // step stays differentiable.
    private Tensor<T> StepActivation(Tensor<T> projected)
    {
        if (GradientTape<T>.Current is not null) return ApplyActivation(projected);
        using var noGrad = new NoGradScope<T>();
        return ApplyActivation(projected);
    }
}
