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

    /// <summary>True for Half and BFloat16, whose intermediates the callers form in float or double.</summary>
    internal static bool IsNarrow => AccumulateInDouble;

    /// <summary>
    /// x * factor. Half and BFloat16 multiply in double: a factor below the smallest Half (a clip scale of 1e-8)
    /// would round to zero before it reached a value it scales back into range.
    /// </summary>
    internal static Tensor<T> Scale(Tensor<T> values, double factor)
    {
        return AccumulateInDouble
            ? Engine.TensorMultiplyScalar(values.Cast<double>(), factor).Cast<T>()
            : Engine.TensorMultiplyScalar(values, NumOps.FromDouble(factor));
    }

    /// <summary>Σ x in double; Half and BFloat16 are widened first (a Half total overflows at 65504).</summary>
    internal static double Sum(Tensor<T> values)
    {
        return AccumulateInDouble
            ? Engine.TensorSum(values.Cast<double>())
            : NumOps.ToDouble(Engine.TensorSum(values));
    }

    /// <summary>a · b in double; Half and BFloat16 are widened before the products are formed.</summary>
    internal static double Dot(Vector<T> a, Vector<T> b)
    {
        return AccumulateInDouble
            ? Engine.TensorSum(Engine.TensorMultiply(a.AsTensor().Cast<double>(), b.AsTensor().Cast<double>()))
            : NumOps.ToDouble(Engine.DotProduct(a, b));
    }

    /// <summary>
    /// Σ w (a - b)² in double (w = 1 when <paramref name="importance"/> is null). Half and BFloat16 subtract,
    /// square and sum in double: a difference of 300 squares past Half's 65504.
    /// </summary>
    internal static double WeightedSquaredDistance(Vector<T> a, Vector<T> b, Vector<T>? importance)
    {
        if (AccumulateInDouble)
        {
            var squared = Engine.TensorSquare(Engine.TensorSubtract(a.AsTensor().Cast<double>(), b.AsTensor().Cast<double>()));
            return Engine.TensorSum(importance is null
                ? squared
                : Engine.TensorMultiply(importance.AsTensor().Cast<double>(), squared));
        }

        var diff = Engine.Subtract(a, b);
        return NumOps.ToDouble(importance is null
            ? Engine.DotProduct(diff, diff)
            : Engine.Sum(Engine.Multiply(importance, Engine.Multiply(diff, diff))));
    }

    /// <summary>
    /// ||a - b|| for two flat tensors of one length, in double. Half and BFloat16 subtract in double: the
    /// difference of two in-range values (40000 and -40000) is out of range.
    /// </summary>
    internal static double Distance(Tensor<T> a, Tensor<T> b)
    {
        return AccumulateInDouble
            ? FederatedTensorMath<double>.ScaledNorm(Engine.TensorSubtract(a.Cast<double>(), b.Cast<double>()).ToVector())
            : ScaledNorm(Engine.TensorSubtract(a, b).ToVector());
    }

    /// <summary>
    /// The pseudo-Fisher diagonal theta_j² * mean_i(x_{i, j mod C}²) of a [N, C] sample block. Callers pass
    /// float tensors for Half and BFloat16, where x² overflows at 256 and 0 * infinity would be NaN.
    /// </summary>
    internal static Tensor<TV> PseudoFisherDiagonal<TV>(Tensor<TV> samples, Tensor<TV> parameters, int[] featureIndex)
    {
        var columnMeanSquares = Engine.ReduceMean(Engine.TensorSquare(samples), new[] { 0 }, false); // [C]
        var perParameter = Engine.TensorGather(columnMeanSquares, new Tensor<int>(featureIndex, new[] { featureIndex.Length }), 0);
        return Engine.TensorMultiply(Engine.TensorSquare(parameters), perParameter);
    }

    /// <summary>
    /// Σ_k (w_k / W) x_k over the clients' vectors (their first <paramref name="length"/> entries), W = Σ w_k.
    /// The weights are normalized in double before they touch T and the sum runs through
    /// <see cref="Accumulator"/>; a total that is not positive gives zeros.
    /// </summary>
    internal static Vector<T> WeightedAverage(Dictionary<int, Vector<T>> items, Dictionary<int, double>? weights, int length)
    {
        double total = 0;
        foreach (var id in items.Keys)
        {
            total += weights?.GetValueOrDefault(id, 1.0) ?? 1.0;
        }

        var sum = new Accumulator(length);
        if (total > 0)
        {
            foreach (var (id, item) in items)
            {
                var head = item.Length == length ? item : item.GetSubVector(0, length);
                sum.Add(head.AsTensor(), (weights?.GetValueOrDefault(id, 1.0) ?? 1.0) / total);
            }
        }

        return sum.ToTensor().ToVector();
    }

    /// <summary>
    /// A running Σ c_k x_k over flat tensors of one length. Half and BFloat16 accumulate in double: a product
    /// below 2^-24 rounds to zero and a Half running sum stops growing once it dwarfs the next term (0.5 plus
    /// 1/4096 is 0.5), so the mean of thousands of clients is wrong although it is in range.
    /// </summary>
    internal sealed class Accumulator
    {
        private Tensor<T>? _sum;
        private Tensor<double>? _wideSum;

        internal Accumulator(int length)
        {
            if (AccumulateInDouble)
            {
                _wideSum = new Tensor<double>(new[] { length });
            }
            else
            {
                _sum = new Tensor<T>(new[] { length });
            }
        }

        internal void Add(Tensor<T> values, double coefficient)
        {
            if (_wideSum is not null)
            {
                _wideSum = Engine.TensorAdd(_wideSum, Engine.TensorMultiplyScalar(values.Cast<double>(), coefficient));
            }
            else if (_sum is not null)
            {
                _sum = Engine.TensorAdd(_sum, Engine.TensorMultiplyScalar(values, NumOps.FromDouble(coefficient)));
            }
        }

        internal Tensor<T> ToTensor() => _wideSum is not null ? _wideSum.Cast<T>() : _sum ?? new Tensor<T>(new[] { 0 });
    }

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

        if (AccumulateInDouble)
        {
            // Half and BFloat16: a coefficient below the smallest Half rounds to zero, so multiply in double.
            var wide = Engine.TensorMultiplyScalar(items[0].Cast<double>(), normalized[0]);
            for (int k = 1; k < normalized.Length; k++)
            {
                wide = Engine.TensorAdd(wide, Engine.TensorMultiplyScalar(items[k].Cast<double>(), normalized[k]));
            }

            return wide.Cast<T>();
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
