namespace AiDotNet.FederatedLearning.Unlearning;

/// <summary>
/// Whole-tensor helpers shared by the federated unlearners.
/// </summary>
internal static class UnlearningEngineOps
{
    /// <summary>
    /// A new <c>[length]</c> tensor holding the first entries of <paramref name="source"/> (read flat,
    /// at most <c>Shape[0]</c> of them), zero padded when the source is shorter.
    /// </summary>
    internal static Tensor<T> FitLength<T>(Tensor<T> source, int length)
    {
        var fitted = new Tensor<T>(new[] { Math.Max(0, length) });
        int sourceRows = source.Shape.Length > 0 ? source.Shape[0] : 0;
        int available = Math.Max(0, Math.Min(length, Math.Min(sourceRows, source.Length)));
        if (available > 0)
        {
            source.AsSpan().Slice(0, available).CopyTo(fitted.AsWritableSpan());
        }

        return fitted;
    }

    /// <summary>
    /// Sum of <c>FitLength(update, length)</c> over <paramref name="updates"/>.
    /// </summary>
    internal static Tensor<T> SumFitted<T>(IEnumerable<Tensor<T>> updates, int length)
    {
        var engine = AiDotNetEngine.Current;
        var sum = new Tensor<T>(new[] { Math.Max(0, length) });
        foreach (var update in updates)
        {
            sum = engine.TensorAdd(sum, FitLength(update, length));
        }

        return sum;
    }

    /// <summary>
    /// Euclidean distance between the first <c>min(a.Shape[0], b.Shape[0])</c> entries of the two tensors.
    /// </summary>
    internal static double L2Distance<T>(Tensor<T> a, Tensor<T> b)
    {
        int size = Math.Min(a.Shape[0], b.Shape[0]);
        if (size <= 0)
        {
            return 0.0;
        }

        var engine = AiDotNetEngine.Current;
        var diff = engine.TensorSubtract(FitLength(a, size).Cast<double>(), FitLength(b, size).Cast<double>());
        return Math.Sqrt(Dot(diff, diff));
    }

    /// <summary>
    /// sum_i a_i b_i, reduced in double. The certificate metrics are doubles, and in T the products and
    /// long sums overflow or saturate (Half: 65504, and integer spacing 2 above 2048).
    /// </summary>
    internal static double Dot<T>(Tensor<T> a, Tensor<T> b)
    {
        var engine = AiDotNetEngine.Current;
        return engine.TensorSum(engine.TensorMultiply(a.Cast<double>(), b.Cast<double>()));
    }

    /// <summary>sum_i a_i, reduced in double (see <see cref="Dot{T}"/>).</summary>
    internal static double Sum<T>(Tensor<T> a)
    {
        return AiDotNetEngine.Current.TensorSum(a.Cast<double>());
    }

    /// <summary>
    /// Per-parameter influence of a client from its update history: |mean| * (1 + std) over the rounds, scaled to
    /// [0, 1] by the largest score (left as is when that is not positive or any score is NaN). The variance is the
    /// two-pass mean of squared deviations: E[x^2] - E[x]^2 cancels catastrophically in float and Half when the
    /// updates are large next to their spread.
    /// </summary>
    internal static Tensor<T> InfluenceScores<T>(IReadOnlyList<Tensor<T>> updates, int modelSize)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        var influence = new Tensor<T>(new[] { Math.Max(0, modelSize) });
        if (updates.Count > 0)
        {
            T rounds = numOps.FromDouble(updates.Count);
            var fitted = new List<Tensor<T>>(updates.Count);
            foreach (var update in updates)
            {
                fitted.Add(FitLength(update, modelSize));
            }

            var mean = engine.TensorDivideScalar(SumFitted(fitted, modelSize), rounds);
            var squaredDeviation = new Tensor<T>(new[] { Math.Max(0, modelSize) });
            foreach (var update in fitted)
            {
                squaredDeviation = engine.TensorAdd(squaredDeviation, engine.TensorSquare(engine.TensorSubtract(update, mean)));
            }

            var std = engine.TensorSqrt(engine.TensorDivideScalar(squaredDeviation, rounds));
            influence = engine.TensorMultiply(engine.TensorAbs(mean), engine.TensorAddScalar(std, numOps.One));
        }

        // A NaN score leaves the scores unnormalized, as the Math.Max fold did.
        if (modelSize > 0 && !double.IsNaN(Sum(influence)))
        {
            T maxInfluence = engine.ReduceMax(influence, new[] { 0 }, false).GetFlat(0);
            if (numOps.GreaterThan(maxInfluence, numOps.Zero))
            {
                influence = engine.TensorDivideScalar(influence, maxInfluence);
            }
        }

        return influence;
    }
}
