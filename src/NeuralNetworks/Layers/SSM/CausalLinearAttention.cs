namespace AiDotNet.NeuralNetworks.Layers.SSM;

/// <summary>
/// Shared tape-aware compositions for causal linear-attention layers.
/// </summary>
internal static class CausalLinearAttention
{
    /// <summary>
    /// Computes multi-head scaled dot-product attention from [B,T,D] projections.
    /// </summary>
    internal static Tensor<T> ScaledDotProduct<T>(
        IEngine engine,
        Tensor<T> query,
        Tensor<T> key,
        Tensor<T> value,
        int numHeads,
        bool causal,
        int? windowSize = null)
    {
        int batch = query.Shape[0];
        int sequence = query.Shape[1];
        int modelDimension = query.Shape[2];
        int headDimension = modelDimension / numHeads;
        var q = engine.TensorPermute(
            engine.Reshape(query, new[] { batch, sequence, numHeads, headDimension }),
            new[] { 0, 2, 1, 3 });
        var k = engine.TensorPermute(
            engine.Reshape(key, new[] { batch, sequence, numHeads, headDimension }),
            new[] { 0, 2, 1, 3 });
        var v = engine.TensorPermute(
            engine.Reshape(value, new[] { batch, sequence, numHeads, headDimension }),
            new[] { 0, 2, 1, 3 });
        Tensor<bool>? mask = causal
            ? CreateCausalMask(batch, numHeads, sequence, windowSize)
            : null;
        var attended = engine.ScaledDotProductAttention(
            q, k, v, mask, 1.0 / Math.Sqrt(headDimension), out _);
        return engine.Reshape(
            engine.TensorPermute(attended, new[] { 0, 2, 1, 3 }),
            new[] { batch, sequence, modelDimension });
    }

    /// <summary>
    /// Computes normalized causal linear attention without a time-step operation loop:
    /// S_t = sum_{j&lt;=t} k_j v_j^T, z_t = sum_{j&lt;=t} k_j,
    /// y_t = (q_t^T S_t) / (q_t^T z_t + epsilon).
    /// </summary>
    internal static Tensor<T> Normalized<T>(
        IEngine engine,
        Tensor<T> query,
        Tensor<T> key,
        Tensor<T> value,
        int numHeads,
        T epsilon)
    {
        int batch = query.Shape[0];
        int sequence = query.Shape[1];
        int modelDimension = query.Shape[2];
        int headDimension = modelDimension / numHeads;

        var q = engine.Reshape(query, new[] { batch, sequence, numHeads, headDimension });
        var k = engine.Reshape(key, new[] { batch, sequence, numHeads, headDimension });
        var v = engine.Reshape(value, new[] { batch, sequence, numHeads, headDimension });

        return engine.Reshape(
            NormalizedHeads(engine, q, k, v, epsilon),
            new[] { batch, sequence, modelDimension });
    }

    /// <summary>
    /// Computes normalized causal linear attention for explicit head tensors.
    /// Query/key feature width may differ from value width: q,k [B,T,H,F],
    /// v [B,T,H,V], result [B,T,H,V].
    /// </summary>
    internal static Tensor<T> NormalizedHeads<T>(
        IEngine engine,
        Tensor<T> query,
        Tensor<T> key,
        Tensor<T> value,
        T epsilon)
    {
        var q = query;
        var k = key;
        var v = value;

        var keyColumns = engine.TensorExpandDims(k, axis: 4);
        var valueRows = engine.TensorExpandDims(v, axis: 3);
        var keyValue = engine.TensorMultiply(keyColumns, valueRows);
        var state = engine.TensorCumSum(keyValue, axis: 1);
        var normalizer = engine.TensorCumSum(k, axis: 1);

        var weightedState = engine.TensorMultiply(
            state,
            engine.TensorExpandDims(q, axis: 4));
        var numerator = engine.ReduceSum(weightedState, new[] { 3 }, keepDims: false);
        var denominator = engine.ReduceSum(
            engine.TensorMultiply(normalizer, q),
            new[] { 3 },
            keepDims: true);
        var safeDenominator = engine.TensorAddScalar(denominator, epsilon);
        var output = engine.TensorDivide(numerator, safeDenominator);

        return output;
    }

    private static Tensor<bool> CreateCausalMask(
        int batch,
        int heads,
        int sequence,
        int? windowSize)
    {
        int planeLength = sequence * sequence;
        var values = new bool[batch * heads * planeLength];
        for (int bh = 0; bh < batch * heads; bh++)
        {
            int planeOffset = bh * planeLength;
            for (int query = 0; query < sequence; query++)
            {
                int rowOffset = planeOffset + query * sequence;
                int firstKey = windowSize.HasValue
                    ? Math.Max(0, query - windowSize.Value + 1)
                    : 0;
                for (int key = firstKey; key <= query; key++)
                    values[rowOffset + key] = true;
            }
        }

        return new Tensor<bool>(values, new[] { batch, heads, sequence, sequence });
    }
}
