namespace AiDotNet.Audio.Speaker;

/// <summary>
/// Cosine scores and L2 normalization of speaker embeddings that stay inside the element type's range.
/// </summary>
/// <remarks>
/// A norm reduced in double is still returned as <typeparamref name="T"/>: in Half an embedding whose entries are
/// around 4e4 has a norm above 65504, which rounds to infinity, and dividing by it turns every entry (and the cosine)
/// into zero. Cosine and direction are scale invariant, so each vector is first divided by its largest magnitude:
/// the entries are then at most 1, the squares at most 1 and the norm at most sqrt(length).
/// <para>
/// The whole computation runs in double and is rounded to T once, at the end. Reduced in T, the sums of squares and
/// the dot product carry T's rounding at every addition: over 4096 entries in Half the cosine of a vector with itself
/// came out as 1.0009765625, and the L2-normalized vector had length 1.0005. The cosine is clamped to [-1, 1], so the
/// last rounding of the double reduction cannot leave the range either.
/// </para>
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
    private static double UnitMaxNorm(IEngine engine, Tensor<double> scaled)
    {
        return Math.Sqrt(engine.TensorSum(engine.TensorSquare(scaled)));
    }

    /// <summary>
    /// <paramref name="values"/> as a new flat double tensor. The values are copied out before the cast: a cast taken
    /// straight off a view that starts inside its buffer reads from the start of that buffer.
    /// </summary>
    internal static Tensor<double> InDouble<T>(Tensor<T> values)
    {
        return new Tensor<T>(values.ToArray(), new[] { values.Length }).Cast<double>();
    }

    /// <summary>
    /// The cosine of two equally long vectors, in [-1, 1], and the product of their norms, both computed in double
    /// whatever <typeparamref name="T"/> is. The cosine is zero when either vector is zero or empty.
    /// </summary>
    internal static double CosineInDouble<T>(IEngine engine, Tensor<T> a, Tensor<T> b, out double normProduct)
    {
        normProduct = 0.0;
        if (a.Length == 0 || b.Length == 0)
        {
            return 0.0;
        }

        var scaledA = ScaleToUnitMax(engine, InDouble(a), out double scaleA);
        var scaledB = ScaleToUnitMax(engine, InDouble(b), out double scaleB);
        double normA = UnitMaxNorm(engine, scaledA);
        double normB = UnitMaxNorm(engine, scaledB);
        normProduct = scaleA * normA * scaleB * normB;
        if (normA == 0.0 || normB == 0.0)
        {
            return 0.0;
        }

        // The dot product of the two unit vectors. It is in [-1, 1] up to the last rounding of the reduction, which
        // the clamp removes (a NaN stays a NaN).
        var unitA = engine.TensorDivideScalar(scaledA, normA);
        var unitB = engine.TensorDivideScalar(scaledB, normB);
        double cosine = engine.TensorSum(engine.TensorMultiply(unitA, unitB));
        return Math.Max(-1.0, Math.Min(1.0, cosine));
    }

    /// <summary>
    /// <see cref="CosineInDouble{T}"/> rounded to <typeparamref name="T"/> once (a value in [-1, 1] stays in it).
    /// </summary>
    internal static T Cosine<T>(IEngine engine, Tensor<T> a, Tensor<T> b, out double normProduct)
    {
        return MathHelper.GetNumericOperations<T>().FromDouble(CosineInDouble(engine, a, b, out normProduct));
    }

    /// <summary>
    /// A new tensor of the same shape holding <paramref name="embedding"/> divided by its L2 norm (a copy when the
    /// norm is zero). The norm and the division are taken in double and the result rounded to
    /// <typeparamref name="T"/> once.
    /// </summary>
    internal static Tensor<T> L2Normalize<T>(IEngine engine, Tensor<T> embedding)
    {
        var shape = embedding.Shape.ToArray();
        var copy = new Tensor<T>(embedding.ToArray(), shape);
        if (copy.Length == 0)
        {
            return copy;
        }

        var scaled = ScaleToUnitMax(engine, InDouble(copy), out double _);
        double norm = UnitMaxNorm(engine, scaled);
        if (norm == 0.0)
        {
            return copy;
        }

        return engine.TensorDivideScalar(scaled, norm).Cast<T>().Reshape(shape);
    }
}
