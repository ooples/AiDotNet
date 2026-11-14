namespace AiDotNet.FederatedLearning.Infrastructure;

/// <summary>
/// Engine-op helpers shared by the federated learning components that stay correct for narrow element types.
/// </summary>
/// <remarks>
/// <para>Half overflows at 65504 and stops counting at 2048 (2048 + 1 rounds back to 2048), so a sum of squares, a
/// count of 0/1 flags or a client-weighted sum formed directly in T can be wrong long before the values themselves
/// are out of range. These helpers pre-scale before squaring, count in int and normalize weights first.</para>
/// </remarks>
/// <typeparam name="T">Numeric type.</typeparam>
internal static class FederatedTensorMath<T>
{
    private static readonly AiDotNet.Tensors.Interfaces.INumericOperations<T> NumOps =
        AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();

    private static AiDotNet.Tensors.Engines.IEngine Engine => AiDotNet.Tensors.Engines.AiDotNetEngine.Current;

    /// <summary>
    /// True for Half and BFloat16, whose sums stop growing once the running total dwarfs the next term (adding a
    /// square of at most 1 to a Half total of 2048 or more rounds it away), so their sums of squares are
    /// accumulated in double.
    /// </summary>
    private static readonly bool AccumulateInDouble =
        typeof(T) == typeof(Half) || typeof(T) == typeof(AiDotNet.Tensors.NumericOperations.BFloat16);

    /// <summary>
    /// The L2 norm m * ||x / m|| with m = max |x| (the scaling LAPACK's nrm2 uses), returned in double so a norm
    /// above the range of T is still finite. A NaN entry gives NaN and an infinite entry gives infinity.
    /// </summary>
    internal static double ScaledNorm(Vector<T> vector)
    {
        if (vector.Length == 0)
        {
            return 0;
        }

        T scale = Engine.TensorMaxValue(Engine.TensorAbs(vector.AsTensor()));
        double max = NumOps.ToDouble(scale);
        if (!(max > 0) || double.IsInfinity(max))
        {
            // All zeros (norm 0), a NaN maximum or an infinite entry: the plain norm already gives the answer.
            return NumOps.ToDouble(Engine.Norm(vector));
        }

        var unitScaled = Engine.Divide(vector, scale);
        return max * (AccumulateInDouble
            ? Math.Sqrt(Engine.TensorSum(Engine.TensorSquare(unitScaled.AsTensor().Cast<double>())))
            : NumOps.ToDouble(Engine.Norm(unitScaled)));
    }

    /// <summary>
    /// Σ x² as m² · Σ (x / m)² with m = max |x|, returned in double; for Half and BFloat16 the scaled squares are
    /// also summed in double.
    /// </summary>
    internal static double ScaledSumOfSquares(Tensor<T> values)
    {
        if (values.Length == 0)
        {
            return 0;
        }

        double max = NumOps.ToDouble(Engine.TensorMaxValue(Engine.TensorAbs(values)));
        if (!(max > 0) || double.IsInfinity(max))
        {
            return max > 0 || double.IsNaN(max) ? max : 0;
        }

        var scaled = Engine.TensorDivideScalar(values, NumOps.FromDouble(max));
        double sum = AccumulateInDouble
            ? Engine.TensorSum(Engine.TensorSquare(scaled.Cast<double>()))
            : NumOps.ToDouble(Engine.TensorSum(Engine.TensorSquare(scaled)));
        return max * max * sum;
    }

    /// <summary>
    /// Splits every row of a [rows, width] tensor into its L2 norm (in double) and its unit direction. Each row is
    /// divided by its largest magnitude before squaring, so neither the norm nor the direction overflows T. Zero
    /// rows get norm 0 and a zero direction.
    /// </summary>
    internal static (Tensor<T> Units, double[] Norms) NormalizeRows(Tensor<T> rows)
    {
        int count = rows.Shape[0];
        int width = rows.Shape[1];
        var shape = new[] { count, width };
        var one = Engine.TensorFullLike(new Tensor<T>(new[] { count, 1 }), NumOps.One);

        var rowMax = Engine.ReduceMax(Engine.TensorAbs(rows), new[] { 1 }, true); // [count, 1]
        var safeMax = Engine.TensorWhere(Engine.TensorGreaterThan(rowMax, NumOps.Zero), rowMax, one);
        var scaled = Engine.TensorDivide(rows, Engine.TensorBroadcastTo(safeMax, shape));
        var scaledNorm = AccumulateInDouble
            ? Engine.TensorSqrt(Engine.ReduceSum(Engine.TensorSquare(scaled.Cast<double>()), new[] { 1 }, true)).Cast<T>()
            : Engine.TensorSqrt(Engine.ReduceSum(Engine.TensorSquare(scaled), new[] { 1 }, true));
        var safeNorm = Engine.TensorWhere(Engine.TensorGreaterThan(scaledNorm, NumOps.Zero), scaledNorm, one);
        var units = Engine.TensorDivide(scaled, Engine.TensorBroadcastTo(safeNorm, shape));

        // The norm max * ||row / max|| is formed in double: the product can exceed the range of T.
        var maxValues = rowMax.Cast<double>().ToArray();
        var scaledValues = AccumulateInDouble
            ? Engine.TensorSqrt(Engine.ReduceSum(Engine.TensorSquare(scaled.Cast<double>()), new[] { 1 }, false)).ToArray()
            : scaledNorm.Cast<double>().ToArray();
        var norms = new double[count];
        for (int i = 0; i < count; i++)
        {
            norms[i] = maxValues[i] * scaledValues[i];
        }

        return (units, norms);
    }

    /// <summary>
    /// Σ_k w_k x_k over tensors of one shape, as a single batched multiply-and-reduce. The weights are divided by
    /// <paramref name="normalizer"/> in double first, so a client weight above the range of T does not overflow.
    /// </summary>
    internal static Tensor<T> WeightedSum(IReadOnlyList<Tensor<T>> items, IReadOnlyList<double> weights, double normalizer)
    {
        if (items.Count == 0)
        {
            throw new ArgumentException("At least one tensor is required.", nameof(items));
        }

        if (weights.Count != items.Count)
        {
            throw new ArgumentException("One weight is required per tensor.", nameof(weights));
        }

        var normalized = new double[weights.Count];
        for (int k = 0; k < normalized.Length; k++)
        {
            normalized[k] = weights[k] / normalizer;
        }

        var stacked = Engine.TensorStack(items.ToArray(), 0); // [K, ...]
        var weightShape = new int[stacked.Rank];
        weightShape[0] = items.Count;
        for (int d = 1; d < weightShape.Length; d++)
        {
            weightShape[d] = 1;
        }

        var weightTensor = new Tensor<double>(normalized, new[] { normalized.Length }).Cast<T>().Reshape(weightShape);
        var weighted = Engine.TensorMultiply(stacked, Engine.TensorBroadcastTo(weightTensor, stacked.Shape.ToArray()));
        return Engine.ReduceSum(weighted, new[] { 0 }, false);
    }

    /// <summary>
    /// The number of non-zero entries in every row of a [rows, columns] tensor, counted in int (a 0/1 sum in Half
    /// stops increasing at 2048).
    /// </summary>
    internal static int[] CountNonzeroPerRow(Tensor<T> rows)
    {
        int count = rows.Shape[0];
        if (count == 0)
        {
            return Array.Empty<int>();
        }

        var coordinates = Engine.TensorNonzero(rows); // [nnz, 2], row-major
        int nonzero = coordinates.Shape[0];
        if (nonzero == 0)
        {
            return new int[count];
        }

        var rowIndices = Engine.TensorSlice(coordinates, new[] { 0, 0 }, new[] { nonzero, 1 }).Reshape(new[] { nonzero });
        return Engine.TensorBinCount(rowIndices, count).ToArray();
    }
}
