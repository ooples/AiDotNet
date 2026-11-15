namespace AiDotNet.Audio.Speaker;

/// <summary>
/// Cosine scores and L2 normalization of speaker embeddings that stay inside the element type's range.
/// </summary>
/// <remarks>
/// A norm reduced in double is still returned as <typeparamref name="T"/>: in Half an embedding whose entries are
/// around 4e4 has a norm above 65504, which rounds to infinity, and dividing by it turns every entry (and the cosine)
/// into zero. Cosine and direction are scale invariant, so each vector is first divided by its largest magnitude:
/// the entries are then at most 1, the squares at most 1 and the norm at most sqrt(length).
/// </remarks>
internal static class SpeakerScoring
{
    /// <summary>
    /// <paramref name="values"/> divided by its largest absolute entry, and that entry. A vector whose largest entry is
    /// zero, infinite or NaN is returned as it is with a scale of one.
    /// </summary>
    internal static Tensor<T> ScaleToUnitMax<T>(IEngine engine, Tensor<T> values, out T scale)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        scale = numOps.One;
        if (values.Length == 0)
        {
            return values;
        }

        T largest = engine.ReduceMax(engine.TensorAbs(values.Reshape(new[] { values.Length })), new[] { 0 }, false).GetFlat(0);
        if (!numOps.GreaterThan(largest, numOps.Zero) || numOps.IsInfinity(largest))
        {
            return values;
        }

        scale = largest;
        return engine.TensorDivideScalar(values, largest);
    }

    /// <summary>The L2 norm of a vector whose entries are at most 1 in magnitude (the squares cannot overflow).</summary>
    private static T UnitMaxNorm<T>(IEngine engine, Tensor<T> scaled)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        return numOps.Sqrt(engine.TensorSum(engine.TensorSquare(scaled)));
    }

    /// <summary>
    /// The cosine of two equally long vectors and the product of their norms (in double, which holds it where
    /// <typeparamref name="T"/> may not). The cosine is zero when either vector is zero.
    /// </summary>
    internal static T Cosine<T>(IEngine engine, Tensor<T> a, Tensor<T> b, out double normProduct)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var scaledA = ScaleToUnitMax(engine, a, out T scaleA);
        var scaledB = ScaleToUnitMax(engine, b, out T scaleB);
        T normA = UnitMaxNorm(engine, scaledA);
        T normB = UnitMaxNorm(engine, scaledB);
        normProduct = numOps.ToDouble(scaleA) * numOps.ToDouble(normA) * numOps.ToDouble(scaleB) * numOps.ToDouble(normB);
        if (numOps.Equals(normA, numOps.Zero) || numOps.Equals(normB, numOps.Zero))
        {
            return numOps.Zero;
        }

        // The dot product of the two unit vectors, which stays inside [-1, 1].
        var unitA = engine.TensorDivideScalar(scaledA, normA);
        var unitB = engine.TensorDivideScalar(scaledB, normB);
        return engine.TensorSum(engine.TensorMultiply(unitA, unitB));
    }

    /// <summary>
    /// A new tensor of the same shape holding <paramref name="embedding"/> divided by its L2 norm (a copy when the
    /// norm is zero).
    /// </summary>
    internal static Tensor<T> L2Normalize<T>(IEngine engine, Tensor<T> embedding)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var copy = new Tensor<T>(embedding.ToArray(), embedding.Shape.ToArray());
        var scaled = ScaleToUnitMax(engine, copy, out _);
        T norm = UnitMaxNorm(engine, scaled);
        if (numOps.Equals(norm, numOps.Zero))
        {
            return copy;
        }

        return engine.TensorDivideScalar(scaled, norm);
    }
}
