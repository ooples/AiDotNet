namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Whole-vector engine helpers for the per-task parameter modulation that several adapted
/// meta-learning models apply to their backbone parameters at prediction time.
/// </summary>
/// <remarks>
/// <para>
/// A modulation vector m of length L scales a parameter vector p of length n cyclically:
/// p'[i] = p[i] * m[i mod L] (numpy: <c>p * np.resize(m, n)</c>). The factors are kept in the
/// model's numeric type T so the scaling never round-trips through double.
/// </para>
/// </remarks>
internal static class ParameterModulation
{
    /// <summary>
    /// Scales <paramref name="parameters"/> by <paramref name="factors"/> repeated cyclically.
    /// </summary>
    /// <param name="engine">The engine that runs the vector operations.</param>
    /// <param name="parameters">The backbone parameters p.</param>
    /// <param name="factors">The non-empty modulation factors m.</param>
    /// <returns>A new vector p * resize(m, p.Length).</returns>
    internal static Vector<T> Apply<T>(IEngine engine, Vector<T> parameters, Vector<T> factors)
    {
        if (factors.Length == 0)
        {
            throw new ArgumentException("Modulation factors must not be empty.", nameof(factors));
        }

        return engine.Multiply(parameters, Cycle(engine, factors, parameters.Length));
    }

    /// <summary>
    /// Repeats <paramref name="factors"/> cyclically to exactly <paramref name="length"/> entries.
    /// </summary>
    internal static Vector<T> Cycle<T>(IEngine engine, Vector<T> factors, int length)
    {
        int m = factors.Length;
        if (m == length)
        {
            return factors;
        }

        int repeats = (length + m - 1) / m;
        var tiled = engine.TensorTile(Tensor<T>.FromVector(factors), new[] { repeats }).ToVector();
        return tiled.Length == length ? tiled : tiled.GetSubVector(0, length);
    }

    /// <summary>
    /// The magnitude gate 0.5 + 0.5 * sigmoid(mean|w| - 1) as a one-element modulation vector:
    /// larger task-specific weights push the backbone scale from 0.5 towards 1.
    /// </summary>
    internal static Vector<T> MeanAbsGate<T>(IEngine engine, Vector<T> weights)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        T half = numOps.FromDouble(0.5);
        T meanAbs = engine.Mean(engine.Abs(weights));
        var gate = engine.Sigmoid(new Vector<T>(new[] { numOps.Subtract(meanAbs, numOps.One) }));
        return engine.Add(engine.Multiply(gate, half), engine.Fill(1, half));
    }

    /// <summary>
    /// The mean of clamp(adapted[i] / raw[i], lower, upper) over the entries with |raw[i]| &gt; threshold,
    /// taken over the first min(raw.Length, adapted.Length) entries, as a one-element modulation vector.
    /// </summary>
    /// <returns>The one-element factor vector, or null when no entry passes the threshold.</returns>
    internal static Vector<T>? MeanClampedRatio<T>(
        IEngine engine, Vector<T> raw, Vector<T> adapted, double lower, double upper, double threshold)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int n = Math.Min(raw.Length, adapted.Length);
        if (n == 0)
        {
            return null;
        }

        var rawT = Tensor<T>.FromVector(raw.Length == n ? raw : raw.GetSubVector(0, n));
        var adaptedT = Tensor<T>.FromVector(adapted.Length == n ? adapted : adapted.GetSubVector(0, n));

        // 1 where |raw| > threshold (NaN compares false, so NaN entries are skipped like the scalar check).
        var mask = engine.TensorGreaterThan(engine.TensorAbs(rawT), numOps.FromDouble(threshold));
        T count = engine.TensorSum(mask);
        if (!numOps.GreaterThan(count, numOps.Zero))
        {
            return null;
        }

        var safeRaw = engine.TensorWhere(mask, rawT, engine.TensorOnesLike(rawT));
        var ratio = engine.TensorClamp(
            engine.TensorDivide(adaptedT, safeRaw), numOps.FromDouble(lower), numOps.FromDouble(upper));
        T sum = engine.TensorSum(engine.TensorWhere(mask, ratio, engine.TensorZerosLike(rawT)));
        return new Vector<T>(new[] { numOps.Divide(sum, count) });
    }

    /// <summary>
    /// Scales the model's parameters in place by <see cref="MeanClampedRatio"/> of the raw and adapted features,
    /// leaving them unchanged when no raw entry passes the threshold.
    /// </summary>
    internal static void ScaleByMeanClampedRatio<T, TInput, TOutput>(
        IEngine engine, IParameterizable<T, TInput, TOutput> model, Vector<T> raw, Vector<T> adapted,
        double lower, double upper, double threshold)
    {
        var ratio = MeanClampedRatio(engine, raw, adapted, lower, upper, threshold);
        if (ratio is not null)
        {
            model.SetParameters(Apply(engine, model.GetParameters(), ratio));
        }
    }

    /// <summary>
    /// Reads <paramref name="parameters"/> cyclically as a [rows, columns] table in row-major order, so entry
    /// (r, c) is parameters[(r * columns + c) mod parameters.Length].
    /// </summary>
    internal static Tensor<T> Table<T>(IEngine engine, Vector<T> parameters, int rows, int columns)
    {
        return engine.Reshape(Tensor<T>.FromVector(Cycle(engine, parameters, rows * columns)), new[] { rows, columns });
    }

    /// <summary>
    /// Reads <paramref name="parameters"/> in order, without wrapping, as a [rows, defaults.Length] table; once
    /// the parameters run out every remaining entry takes the default for its column.
    /// </summary>
    internal static Tensor<T> DefaultedTable<T>(IEngine engine, Vector<T> parameters, int rows, Vector<T> defaults)
    {
        int columns = defaults.Length;
        int needed = rows * columns;
        int available = Math.Min(parameters.Length, needed);
        var values = available == needed
            ? parameters.GetSubVector(0, needed)
            : Vector<T>.Concatenate(
                available > 0 ? parameters.GetSubVector(0, available) : new Vector<T>(0),
                Cycle(engine, defaults, needed).GetSubVector(available, needed - available));
        return engine.Reshape(Tensor<T>.FromVector(values), new[] { rows, columns });
    }

    /// <summary>Columns [start, start + count) of a [rows, columns] table, shape [rows, count].</summary>
    internal static Tensor<T> Columns<T>(IEngine engine, Tensor<T> table, int start, int count)
    {
        return engine.TensorSlice(table, new[] { 0, start }, new[] { table.Shape[0], count });
    }

    /// <summary>Column <paramref name="column"/> of a [rows, columns] table as a vector.</summary>
    internal static Vector<T> Column<T>(IEngine engine, Tensor<T> table, int column)
    {
        return Columns(engine, table, column, 1).ToVector();
    }
}
