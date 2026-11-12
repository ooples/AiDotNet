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
        var numOps = MathHelper.GetNumericOperations<T>();
        var diff = engine.TensorSubtract(FitLength(a, size), FitLength(b, size));
        return Math.Sqrt(numOps.ToDouble(engine.TensorSum(engine.TensorSquare(diff))));
    }
}
