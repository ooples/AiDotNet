using AiDotNet.Enums;

namespace AiDotNet.CausalDiscovery;

/// <summary>
/// Engine-op helpers shared by the causal discovery algorithms: dense [rows x cols] weight matrices built
/// from per-edge statistics with one scatter, and scalar tests on tensors without per-element access.
/// </summary>
internal static class CausalTensorHelpers
{
    /// <summary>
    /// A zero matrix holding <c>weights[p]</c> at the row-major position <c>p</c> of each entry.
    /// </summary>
    internal static Matrix<T> FromStatistics<T>(int rows, int cols, Dictionary<int, double> weights)
    {
        var flat = new Tensor<T>(new[] { rows * cols });
        if (weights.Count > 0)
        {
            var positions = new List<int>(weights.Keys);
            var values = new List<double>(positions.Count);
            values.AddRange(positions.Select(p => weights[p]));
            flat = AiDotNetEngine.Current.TensorScatter(flat,
                new Tensor<int>(positions.ToArray(), new[] { positions.Count }),
                Vector<double>.FromList(values).AsTensor().Cast<T>(), 0);
        }

        return flat.Reshape(new[] { rows, cols }).ToMatrix();
    }

    /// <summary>
    /// A zero matrix holding <c>weights[p]</c> at the row-major position <c>p</c> of each entry.
    /// </summary>
    internal static Matrix<T> FromValues<T>(int rows, int cols, Dictionary<int, T> weights)
    {
        var flat = new Tensor<T>(new[] { rows * cols });
        if (weights.Count > 0)
        {
            var positions = new List<int>(weights.Keys);
            var values = new List<T>(positions.Count);
            values.AddRange(positions.Select(p => weights[p]));
            flat = AiDotNetEngine.Current.TensorScatter(flat,
                new Tensor<int>(positions.ToArray(), new[] { positions.Count }),
                Vector<T>.FromList(values).AsTensor(), 0);
        }

        return flat.Reshape(new[] { rows, cols }).ToMatrix();
    }

    /// <summary>
    /// Overwrites <paramref name="destination"/> ([rows x cols]) with <c>weights[p]</c> at each row-major position <c>p</c>.
    /// </summary>
    internal static Tensor<T> OverwriteStatistics<T>(Tensor<T> destination, Dictionary<int, double> weights)
    {
        if (weights.Count == 0)
            return destination;

        int rows = destination.Shape[0], cols = destination.Shape[1];
        var positions = new List<int>(weights.Keys);
        var values = new List<double>(positions.Count);
        values.AddRange(positions.Select(p => weights[p]));
        return AiDotNetEngine.Current.TensorScatter(destination.Reshape(new[] { rows * cols }),
            new Tensor<int>(positions.ToArray(), new[] { positions.Count }),
            Vector<double>.FromList(values).AsTensor().Cast<T>(), 0).Reshape(new[] { rows, cols });
    }

    /// <summary>
    /// Indices that sort the 1-D <paramref name="keys"/> (which must not contain NaN), with equal keys kept
    /// in ascending index order - the order a stable sort, or a comparison that breaks ties by index, gives.
    /// </summary>
    internal static int[] StableArgsort<T>(Tensor<T> keys, bool descending)
    {
        var engine = AiDotNetEngine.Current;
        int k = keys.Length;
        if (k == 0)
            return Array.Empty<int>();

        int[] order = engine.TensorArgsort(keys, -1, descending).ToArray();
        if (k == 1)
            return order;

        // Any sort leaves equal keys adjacent; re-order each run of equal keys by index.
        var sorted = engine.TensorIndexSelect(keys, new Tensor<int>((int[])order.Clone(), new[] { k }), 0);
        var equalToNext = engine.TensorEquals(
            engine.TensorSlice(sorted, new[] { 1 }, new[] { k - 1 }), engine.TensorSlice(sorted, new[] { 0 }, new[] { k - 1 }));
        int[] ties = engine.TensorNonzero(equalToNext).ToArray();
        for (int t = 0; t < ties.Length; t++)
        {
            int start = ties[t], end = start + 1;
            while (t + 1 < ties.Length && ties[t + 1] == end)
            {
                t++;
                end++;
            }

            Array.Sort(order, start, end - start + 1);
        }

        return order;
    }

    /// <summary>
    /// Pearson correlation of two columns: sum(dx dy) / sqrt(sum(dx^2) sum(dy^2)) over the centered
    /// columns, or 0 unless both centered sums of squares are above 1e-10.
    /// </summary>
    internal static T PearsonCorrelation<T>(Matrix<T> data, int col1, int col2)
    {
        var engine = AiDotNetEngine.Current;
        return PearsonCorrelation(engine.GetColumn(data, col1).AsTensor(), engine.GetColumn(data, col2).AsTensor());
    }

    /// <summary>
    /// Pearson correlation of two 1-D tensors: sum(dx dy) / sqrt(sum(dx^2) sum(dy^2)) over the centered
    /// values, or 0 unless both centered sums of squares are above 1e-10.
    /// </summary>
    internal static T PearsonCorrelation<T>(Tensor<T> x, Tensor<T> y)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        var dx = engine.TensorSubtractScalar(x, engine.TensorMean(x));
        var dy = engine.TensorSubtractScalar(y, engine.TensorMean(y));
        T sxy = engine.TensorSum(engine.TensorMultiply(dx, dy));
        T sxx = engine.TensorSum(engine.TensorSquare(dx));
        T syy = engine.TensorSum(engine.TensorSquare(dy));
        T tiny = ops.FromDouble(1e-10);
        return ops.GreaterThan(sxx, tiny) && ops.GreaterThan(syy, tiny)
            ? ops.Divide(sxy, ops.Sqrt(ops.Multiply(sxx, syy)))
            : ops.Zero;
    }

    /// <summary>
    /// sum(a b) / sqrt(sum(a^2) sum(b^2)) without centering, or 0 unless both sums of squares are above 1e-10.
    /// </summary>
    internal static T UncenteredCorrelation<T>(Tensor<T> a, Tensor<T> b)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        T sab = engine.TensorSum(engine.TensorMultiply(a, b));
        T saa = engine.TensorSum(engine.TensorSquare(a));
        T sbb = engine.TensorSum(engine.TensorSquare(b));
        T tiny = ops.FromDouble(1e-10);
        return ops.GreaterThan(saa, tiny) && ops.GreaterThan(sbb, tiny)
            ? ops.Divide(sab, ops.Sqrt(ops.Multiply(saa, sbb)))
            : ops.Zero;
    }

    /// <summary>
    /// Least-squares slope of <paramref name="child"/> on <paramref name="parent"/>: cov / var over the centered
    /// columns, or 0 unless the parent's centered sum of squares is above 1e-10.
    /// </summary>
    internal static T RegressionSlope<T>(Matrix<T> data, int parent, int child)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        var p = engine.GetColumn(data, parent).AsTensor();
        var c = engine.GetColumn(data, child).AsTensor();
        var dp = engine.TensorSubtractScalar(p, engine.TensorMean(p));
        var dc = engine.TensorSubtractScalar(c, engine.TensorMean(c));
        T cov = engine.TensorSum(engine.TensorMultiply(dp, dc));
        T varP = engine.TensorSum(engine.TensorSquare(dp));
        return ops.GreaterThan(varP, ops.FromDouble(1e-10)) ? ops.Divide(cov, varP) : ops.Zero;
    }

    /// <summary>
    /// Gaussian mutual information -0.5 log(1 - r^2) of two 1-D tensors, with r the correlation of the
    /// population (1/n) moments (0 unless both variances are above 1e-15) clamped to [-0.9999, 0.9999].
    /// </summary>
    internal static T GaussianMutualInformation<T>(Tensor<T> x, Tensor<T> y)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        var dx = engine.TensorSubtractScalar(x, engine.TensorMean(x));
        var dy = engine.TensorSubtractScalar(y, engine.TensorMean(y));
        T sxx = engine.TensorMean(engine.TensorSquare(dx));
        T syy = engine.TensorMean(engine.TensorSquare(dy));
        T sxy = engine.TensorMean(engine.TensorMultiply(dx, dy));
        T tiny = ops.FromDouble(1e-15);
        T corr = ops.GreaterThan(sxx, tiny) && ops.GreaterThan(syy, tiny)
            ? ops.Divide(sxy, ops.Sqrt(ops.Multiply(sxx, syy)))
            : ops.Zero;
        T bound = ops.FromDouble(0.9999);
        if (ops.GreaterThan(corr, bound)) corr = bound;
        if (ops.LessThan(corr, ops.Negate(bound))) corr = ops.Negate(bound);
        return ops.Multiply(ops.FromDouble(-0.5), ops.Log(ops.Subtract(ops.One, ops.Multiply(corr, corr))));
    }

    /// <summary>
    /// Sum of squared deviations of a column from its mean.
    /// </summary>
    internal static T CenteredSumOfSquares<T>(Matrix<T> data, int column)
    {
        var engine = AiDotNetEngine.Current;
        var x = engine.GetColumn(data, column).AsTensor();
        return engine.TensorSum(engine.TensorSquare(engine.TensorSubtractScalar(x, engine.TensorMean(x))));
    }

    /// <summary>
    /// Residual sum of squares of the least-squares fit of <paramref name="target"/> on an intercept and
    /// <paramref name="predictors"/>: normal equations (Z'Z + ridge I) beta = Z'y over Z = [1 | X], solved by LU.
    /// </summary>
    internal static T InterceptOlsRss<T>(Matrix<T> data, int target, IReadOnlyList<int> predictors, double ridge)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = data.Rows;
        int dim = predictors.Count + 1;
        var x = Tensor<T>.FromMatrix(data);
        var y = engine.TensorSliceAxis(x, 1, target).Reshape(new[] { n, 1 });
        var z = engine.TensorConcatenate(new[] { engine.TensorOnesLike(y), SelectColumns(x, predictors) }, 1);
        var zt = engine.TensorTranspose(z);
        var gram = Symmetrize(engine.TensorMatMul(zt, z));
        gram = engine.TensorAdd(gram, engine.TensorMultiplyScalar(engine.TensorEye<T>(dim), ops.FromDouble(ridge)));
        var zty = engine.TensorMatMul(zt, y).Reshape(new[] { dim }).ToVector();

        var beta = MatrixSolutionHelper.SolveLinearSystem<T>(gram.ToMatrix(), zty, MatrixDecompositionType.Lu);
        var prediction = engine.TensorMatMul(z, beta.AsTensor().Reshape(new[] { dim, 1 }));
        return engine.TensorSum(engine.TensorSquare(engine.TensorSubtract(y, prediction)));
    }

    /// <summary>
    /// Residual sum of squares of the least-squares fit of <paramref name="target"/> on the centered
    /// <paramref name="predictors"/> with an implicit intercept: (Xc'Xc + ridge I) beta = Xc'yc solved by LU,
    /// prediction mean(y) + Xc beta.
    /// </summary>
    internal static T CenteredOlsRss<T>(Matrix<T> data, int target, IReadOnlyList<int> predictors, double ridge)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = data.Rows;
        int k = predictors.Count;
        var x = Tensor<T>.FromMatrix(data);
        var y = engine.TensorSliceAxis(x, 1, target).Reshape(new[] { n, 1 });
        var yc = engine.TensorSubtractScalar(y, engine.TensorMean(y));
        var predictorColumns = SelectColumns(x, predictors);
        var xc = engine.TensorSubtract(predictorColumns,
            engine.TensorBroadcastTo(engine.ReduceMean(predictorColumns, new[] { 0 }, true), new[] { n, k }));
        var xct = engine.TensorTranspose(xc);
        var gram = Symmetrize(engine.TensorMatMul(xct, xc));
        gram = engine.TensorAdd(gram, engine.TensorMultiplyScalar(engine.TensorEye<T>(k), ops.FromDouble(ridge)));
        var xty = engine.TensorMatMul(xct, yc).Reshape(new[] { k }).ToVector();

        var beta = MatrixSolutionHelper.SolveLinearSystem<T>(gram.ToMatrix(), xty, MatrixDecompositionType.Lu);
        var residual = engine.TensorSubtract(yc, engine.TensorMatMul(xc, beta.AsTensor().Reshape(new[] { k, 1 })));
        return engine.TensorSum(engine.TensorSquare(residual));
    }

    /// <summary>
    /// Gathers the given columns of an [n x d] tensor into an [n x k] tensor.
    /// </summary>
    internal static Tensor<T> SelectColumns<T>(Tensor<T> source, IReadOnlyList<int> columns)
    {
        var index = columns.ToArray();
        return AiDotNetEngine.Current.TensorIndexSelect(source, new Tensor<int>(index, new[] { index.Length }), 1);
    }

    /// <summary>
    /// The square matrix with its strict lower triangle replaced by the mirrored upper triangle, so a Gram
    /// matrix is exactly symmetric.
    /// </summary>
    internal static Tensor<T> Symmetrize<T>(Tensor<T> matrix)
    {
        var engine = AiDotNetEngine.Current;
        return engine.TensorAdd(engine.TensorTriu(matrix, 0), engine.TensorTranspose(engine.TensorTriu(matrix, 1)));
    }

    /// <summary>
    /// True when any entry of <paramref name="values"/> is below <paramref name="bound"/> (NaN never is).
    /// </summary>
    internal static bool AnyBelow<T>(Tensor<T> values, T bound)
    {
        var engine = AiDotNetEngine.Current;
        return engine.TensorCountNonzero(engine.TensorLessThan(values, bound)) > 0;
    }

    /// <summary>
    /// True when any entry of <paramref name="values"/> is above <paramref name="bound"/> (NaN never is).
    /// </summary>
    internal static bool AnyAbove<T>(Tensor<T> values, T bound)
    {
        var engine = AiDotNetEngine.Current;
        return engine.TensorCountNonzero(engine.TensorGreaterThan(values, bound)) > 0;
    }

    /// <summary>
    /// A zero [rows x cols] matrix holding the concatenated <paramref name="chunks"/> at the row-major
    /// <paramref name="positions"/> (one position per chunk element, in order).
    /// </summary>
    internal static Matrix<T> FromTensorValues<T>(int rows, int cols, List<int> positions, List<Tensor<T>> chunks)
    {
        var engine = AiDotNetEngine.Current;
        var flat = new Tensor<T>(new[] { rows * cols });
        if (positions.Count > 0)
        {
            var values = chunks.Count == 1 ? chunks[0].Reshape(new[] { positions.Count })
                : engine.TensorConcatenate(chunks.Select(c => c.Reshape(new[] { c.Length })).ToArray(), 0);
            flat = engine.TensorScatter(flat, new Tensor<int>(positions.ToArray(), new[] { positions.Count }), values, 0);
        }

        return flat.Reshape(new[] { rows, cols }).ToMatrix();
    }

    /// <summary>
    /// Keeps <paramref name="source"/> ([rows x cols]) at the given row-major positions and zeroes the rest.
    /// </summary>
    internal static Matrix<T> KeepPositions<T>(Tensor<T> source, IEnumerable<int> positions)
    {
        var engine = AiDotNetEngine.Current;
        int rows = source.Shape[0], cols = source.Shape[1];
        var flatSource = source.Reshape(new[] { rows * cols });
        var flat = new Tensor<T>(new[] { rows * cols });
        var distinct = new SortedSet<int>(positions);
        if (distinct.Count == 0)
            return flat.Reshape(new[] { rows, cols }).ToMatrix();

        if (distinct.Count > 0)
        {
            var index = new Tensor<int>(distinct.ToArray(), new[] { distinct.Count });
            flat = engine.TensorScatter(flat, index, engine.TensorIndexSelect(flatSource, index, 0), 0);
        }

        return flat.Reshape(new[] { rows, cols }).ToMatrix();
    }
}
