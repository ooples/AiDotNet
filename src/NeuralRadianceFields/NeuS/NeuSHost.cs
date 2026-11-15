using AiDotNet.Attributes;

namespace AiDotNet.NeuralRadianceFields.NeuS;

/// <summary>
/// Host-side helpers shared by the NeuS and SparseNeuS components: moving sampling constants between
/// <c>double</c> arrays and tensors, type-safe epsilons, and seeded normal draws for initialization.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// The ray geometry, sample depths, interpolation corner weights and validity masks of NeuS are constants of the
/// sampling (the reference computes them under <c>torch.no_grad()</c> and detaches them), so they are built on the
/// host in double precision and enter the differentiable graph once per batch as tensors.
/// </remarks>
internal static class NeuSHost<T>
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    // The smallest positive normal half-precision value; an epsilon that rounds to zero in T falls back to it.
    private const double SmallestHalfNormal = 6.103515625e-5;

    /// <summary>Copies host values into a new tensor of the given shape.</summary>
    /// <param name="values">The values, row-major.</param>
    /// <param name="shape">The tensor shape; its element count must equal the value count.</param>
    [ScalarKernel("Converts host-built sampling constants (ray geometry, sample depths, interpolation weights) once per batch.")]
    public static Tensor<T> FromDoubles(double[] values, params int[] shape)
    {
        var tensor = new Tensor<T>(shape);
        if (tensor.Length != values.Length)
            throw new ArgumentException($"Shape holds {tensor.Length} elements but {values.Length} values were given.", nameof(values));
        for (int i = 0; i < values.Length; i++) tensor[i] = Ops.FromDouble(values[i]);
        return tensor;
    }

    /// <summary>Copies a tensor's values to a host array, row-major.</summary>
    /// <param name="tensor">The tensor.</param>
    [ScalarKernel("Reads detached sampling quantities (coarse SDF values, weights) back for host-side importance sampling.")]
    public static double[] ToDoubles(Tensor<T> tensor)
    {
        var values = new double[tensor.Length];
        for (int i = 0; i < values.Length; i++) values[i] = Ops.ToDouble(tensor[i]);
        return values;
    }

    /// <summary>An integer index tensor for gathers.</summary>
    /// <param name="indices">The indices.</param>
    public static Tensor<int> Indices(int[] indices)
        => Tensor<int>.FromVector(Vector<int>.Wrap(indices), new[] { indices.Length });

    /// <summary>
    /// <paramref name="requested"/> in T, or the smallest positive normal half-precision value when the requested
    /// epsilon rounds to zero in T (so a guard such as <c>p + 1e-7</c> never becomes <c>p + 0</c>).
    /// </summary>
    /// <param name="requested">The epsilon the reference uses.</param>
    public static T Epsilon(double requested)
    {
        var value = Ops.FromDouble(requested);
        return Ops.ToDouble(value) > 0.0 ? value : Ops.FromDouble(SmallestHalfNormal);
    }

    /// <summary>A scalar constant in T.</summary>
    /// <param name="value">The value.</param>
    public static T Scalar(double value) => Ops.FromDouble(value);

    /// <summary>A tensor of shape <paramref name="shape"/> filled with <paramref name="value"/>.</summary>
    /// <param name="value">The fill value.</param>
    /// <param name="shape">The shape.</param>
    public static Tensor<T> Filled(double value, params int[] shape)
    {
        int count = 1;
        foreach (int d in shape) count *= d;
        var values = new double[count];
        for (int i = 0; i < count; i++) values[i] = value;
        return FromDoubles(values, shape);
    }

    /// <summary>
    /// Samples <paramref name="query"/> on the dense grid of <paramref name="bounds"/>, in chunks of
    /// <paramref name="chunkSize"/> points, without recording a gradient tape (NeuS's <c>extract_fields</c>).
    /// </summary>
    /// <param name="engine">The engine.</param>
    /// <param name="query">Maps <c>[n, 3]</c> points to <c>[n]</c> signed distances.</param>
    /// <param name="resolution">Samples per axis.</param>
    /// <param name="bounds">The sampled box.</param>
    /// <param name="chunkSize">Points per query.</param>
    /// <returns>The grid, <c>[R, R, R]</c>.</returns>
    public static Tensor<T> SampleGrid(IEngine engine, Func<Tensor<T>, Tensor<T>> query, int resolution,
        SdfGridBounds bounds, int chunkSize)
    {
        if (query is null) throw new ArgumentNullException(nameof(query));
        if (bounds is null) throw new ArgumentNullException(nameof(bounds));
        if (chunkSize <= 0) throw new ArgumentOutOfRangeException(nameof(chunkSize), "The chunk size must be positive.");
        var points = bounds.GridPoints(resolution);
        int total = points.Length / 3;
        var parts = new List<Tensor<T>>();
        using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
        {
            for (int start = 0; start < total; start += chunkSize)
            {
                int count = Math.Min(chunkSize, total - start);
                var chunk = new double[count * 3];
                Array.Copy(points, start * 3, chunk, 0, count * 3);
                var sdf = query(FromDoubles(chunk, count, 3));
                // Copied out of the query's arena: the next chunk may reuse its buffers.
                parts.Add(FromDoubles(ToDoubles(sdf), count));
            }
        }

        var grid = parts.Count == 1 ? parts[0] : engine.TensorConcatenate(parts.ToArray(), 0);
        return engine.Reshape(grid, new[] { resolution, resolution, resolution });
    }

    /// <summary>A standard normal draw by the Box-Muller transform.</summary>
    /// <param name="random">The generator.</param>
    public static double NextGaussian(Random random)
    {
        double u1 = 1.0 - random.NextDouble();
        double u2 = random.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }
}
