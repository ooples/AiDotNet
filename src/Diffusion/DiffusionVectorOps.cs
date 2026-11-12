namespace AiDotNet.Diffusion;

/// <summary>
/// Whole-vector helpers shared by the diffusion alignment, guidance and distillation components.
/// </summary>
internal static class DiffusionVectorOps
{
    /// <summary>
    /// Returns the first <paramref name="length"/> elements of <paramref name="values"/>, padded with zeros when the
    /// vector is shorter.
    /// </summary>
    internal static Vector<T> Fit<T>(Vector<T> values, int length)
    {
        if (values.Length == length)
            return values;
        if (values.Length > length)
            return values.GetSubVector(0, length);
        return Vector<T>.Concatenate(values, new Vector<T>(length - values.Length));
    }

    /// <summary>
    /// The per-sample reduction axes of a batched tensor: every axis but the first (all axes for a rank-1 tensor).
    /// </summary>
    internal static int[] SampleAxes(int rank)
    {
        if (rank <= 1)
            return new[] { 0 };
        var axes = new int[rank - 1];
        for (int a = 1; a < rank; a++)
            axes[a - 1] = a;
        return axes;
    }

    /// <summary>
    /// Population standard deviation over <paramref name="axes"/> (kept as size-1 dimensions).
    /// </summary>
    internal static Tensor<T> Std<T>(Tensor<T> values, int[] axes)
    {
        var engine = AiDotNetEngine.Current;
        var mean = engine.ReduceMean(values, axes, true);
        var centered = engine.TensorSubtract(values, engine.TensorBroadcastTo(mean, values._shape));
        return engine.TensorSqrt(engine.ReduceMean(engine.TensorSquare(centered), axes, true));
    }

    /// <summary>
    /// weight * (a - b) over the first <paramref name="length"/> elements, each operand zero-padded or truncated.
    /// </summary>
    internal static Vector<T> ScaledDifference<T>(Vector<T> a, Vector<T> b, int length, T weight)
    {
        var engine = AiDotNetEngine.Current;
        return engine.Multiply(engine.Subtract(Fit(a, length), Fit(b, length)), weight);
    }

    /// <summary>
    /// Sum of the pseudo-Huber terms sqrt((a - b)^2 + c^2) - c over the common prefix of the two vectors.
    /// </summary>
    internal static T PseudoHuberSum<T>(Vector<T> a, Vector<T> b, T c)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int length = Math.Min(a.Length, b.Length);
        var diff = engine.Subtract(Fit(a, length), Fit(b, length));
        var root = engine.Sqrt(engine.Add(engine.Multiply(diff, diff), Vector<T>.Default(length, ops.Multiply(c, c))));
        return ops.Subtract(engine.Sum(root), ops.Multiply(c, ops.FromDouble(length)));
    }

    /// <summary>
    /// Sum of squared differences over the common prefix of the two vectors.
    /// </summary>
    internal static T SquaredDistance<T>(Vector<T> a, Vector<T> b)
    {
        var engine = AiDotNetEngine.Current;
        int length = Math.Min(a.Length, b.Length);
        var diff = engine.Subtract(Fit(a, length), Fit(b, length));
        return engine.DotProduct(diff, diff);
    }
}
