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
    /// Vector of <paramref name="length"/> elements all equal to <paramref name="value"/>.
    /// </summary>
    internal static Vector<T> Full<T>(int length, T value)
    {
        var result = new Vector<T>(length);
        result.Fill(value);
        return result;
    }

    /// <summary>
    /// Vector of <paramref name="length"/> ones.
    /// </summary>
    internal static Vector<T> Ones<T>(int length) => Full(length, MathHelper.GetNumericOperations<T>().One);

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
        var root = engine.Sqrt(engine.Add(engine.Multiply(diff, diff), Full(length, ops.Multiply(c, c))));
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

    /// <summary>
    /// Euclidean norm of <paramref name="values"/>.
    /// </summary>
    /// <remarks>
    /// Half squares overflow above 255 and the engine accumulates Half in Half, so Half is reduced in double (the
    /// wide accumulation the converted loops used) and returned as T.
    /// </remarks>
    internal static T Norm<T>(Vector<T> values)
    {
        var engine = AiDotNetEngine.Current;
#if NET5_0_OR_GREATER
        if (typeof(T) == typeof(Half))
            return MathHelper.GetNumericOperations<T>().FromDouble(engine.Norm(values.AsTensor().Cast<double>().ToVector()));
#endif
        return engine.Norm(values);
    }

    /// <summary>
    /// sum((reference - estimate)^2) / sum(reference^2) over two tensors of equal length, or 0 when the reference has
    /// no energy.
    /// </summary>
    /// <remarks>
    /// Half squares overflow above 255 (an ordinary STFT magnitude), so Half is reduced in double and returned as T.
    /// </remarks>
    internal static T RelativeSquaredError<T>(Tensor<T> reference, Tensor<T> estimate)
    {
        var ops = MathHelper.GetNumericOperations<T>();
#if NET5_0_OR_GREATER
        if (typeof(T) == typeof(Half))
            return ops.FromDouble(RelativeSquaredError(reference.Cast<double>(), estimate.Cast<double>()));
#endif
        var engine = AiDotNetEngine.Current;
        T error = engine.TensorSum(engine.TensorSquare(engine.TensorSubtract(reference, estimate)));
        T energy = engine.TensorSum(engine.TensorSquare(reference));
        return ops.GreaterThan(energy, ops.Zero) ? ops.Divide(error, energy) : ops.Zero;
    }

    /// <summary>
    /// Sinusoidal timestep embedding (Vaswani 2017 / DDPM) of <paramref name="size"/> elements:
    /// [sin(t f_0), ..., sin(t f_(h-1)), cos(t f_0), ..., cos(t f_(h-1))] with h = size / 2 and
    /// f_j = exp(-ln(10000) j / max(1, h - 1)), followed by one zero when the size is odd.
    /// </summary>
    /// <remarks>
    /// The angles are geometry and are computed in double: in Half, t f_j for an ordinary timestep is only known to a
    /// fraction of a radian, which scrambles the sines.
    /// </remarks>
    internal static Tensor<T> SinusoidalTimestepEmbedding<T>(int timestep, int size)
    {
        var engine = AiDotNetEngine.Current;
        int half = size / 2;
        var j = engine.TensorArange<double>(0, half, 1);
        var freq = engine.TensorExp(engine.TensorMultiplyScalar(j, -Math.Log(10000.0) / Math.Max(1, half - 1)));
        var arg = engine.TensorMultiplyScalar(freq, (double)timestep);
        var parts = size == 2 * half
            ? new[] { engine.TensorSin(arg), engine.TensorCos(arg) }
            : new[] { engine.TensorSin(arg), engine.TensorCos(arg), new Tensor<double>(new[] { 1 }) };
        return engine.TensorConcatenate(parts, 0).Cast<T>();
    }

    /// <summary>
    /// Interleaved sinusoidal position table [count, size]: channel 2k of position p holds
    /// sin(p 10000^(-2k / size)) and channel 2k + 1 the cosine of the same angle.
    /// </summary>
    /// <remarks>
    /// The angles are geometry and are computed in double: Half cannot represent positions above 2048, nor the
    /// angles to a useful fraction of a radian.
    /// </remarks>
    internal static Tensor<T> InterleavedPositionEmbedding<T>(int count, int size)
    {
        var engine = AiDotNetEngine.Current;
        int pairs = (size + 1) / 2;
        var positions = engine.Reshape(engine.TensorArange<double>(0, count, 1), new[] { count, 1 });
        var freq = engine.TensorExp(engine.TensorMultiplyScalar(engine.TensorArange<double>(0, 2 * pairs, 2),
            -Math.Log(10000.0) / size));
        var angles = engine.TensorMultiply(
            engine.TensorBroadcastTo(positions, new[] { count, pairs }),
            engine.TensorBroadcastTo(engine.Reshape(freq, new[] { 1, pairs }), new[] { count, pairs }));
        var pairShape = new[] { count, pairs, 1 };
        var interleaved = engine.Reshape(engine.TensorConcatenate(new[]
        {
            engine.Reshape(engine.TensorSin(angles), pairShape),
            engine.Reshape(engine.TensorCos(angles), pairShape),
        }, 2), new[] { count, 2 * pairs });
        return engine.TensorNarrow(interleaved, 1, 0, size).Contiguous().Cast<T>();
    }
}
