using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection;

/// <summary>
/// Whole-array engine helpers shared by the feature selectors (pairwise distances, first-index arg-min/arg-max per
/// row, per-group means and per-group best members). Ties always resolve to the lowest index, like the
/// strict-comparison loops and stable LINQ orderings they replace.
/// </summary>
internal static class FeatureSelectionTensorOps
{
    /// <summary>
    /// Sum of squared differences between every row of <paramref name="a"/> ([m, p]) and every row of
    /// <paramref name="b"/> ([k, p]), as [m, k]. Uses the difference form (not the expanded dot-product form) so
    /// the values match a direct per-pair loop.
    /// </summary>
    public static Tensor<T> SquaredDistances<T>(Tensor<T> a, Tensor<T> b)
    {
        var engine = AiDotNetEngine.Current;
        int m = a.Shape[0];
        int k = b.Shape[0];
        int p = a.Shape[1];
        var shape = new[] { m, k, p };
        var diff = engine.TensorSubtract(
            engine.TensorBroadcastTo(engine.Reshape(a, new[] { m, 1, p }), shape),
            engine.TensorBroadcastTo(engine.Reshape(b, new[] { 1, k, p }), shape));
        return engine.ReduceSum(engine.TensorSquare(diff), new[] { 2 }, false);
    }

    /// <summary>
    /// For every row of a 2-D tensor, the column of its first (lowest-index) minimum; a row with no minimum
    /// (all NaN) gets 0.
    /// </summary>
    public static int[] FirstArgMinPerRow<T>(Tensor<T> values)
    {
        var engine = AiDotNetEngine.Current;
        return FirstArgMaxPerRow(engine.TensorNegate(values));
    }

    /// <summary>
    /// For every row of a 2-D tensor, the column of its first (lowest-index) maximum; a row with no maximum
    /// (all NaN) gets 0.
    /// </summary>
    public static int[] FirstArgMaxPerRow<T>(Tensor<T> values)
    {
        var engine = AiDotNetEngine.Current;
        int rows = values.Shape[0];
        int cols = values.Shape[1];
        if (rows == 0)
            return Array.Empty<int>();
        if (cols == 0)
            return new int[rows];

        var rowMax = engine.TensorBroadcastTo(engine.ReduceMax(values, new[] { 1 }, true), new[] { rows, cols });
        var first = PreprocessingTensorOps.FirstNonZeroPerRow(engine.TensorEquals(values, rowMax));
        return Array.ConvertAll(first, c => c == cols ? 0 : c);
    }

    /// <summary>
    /// Per-group column means: <paramref name="membership"/> is a 0/1 [n, k] one-hot of each row's group and
    /// <paramref name="x"/> is [n, p]. Returns [k, p]; an empty group's mean is 0. <paramref name="counts"/> gets
    /// the [k] group sizes.
    /// </summary>
    public static Tensor<T> GroupMeans<T>(Tensor<T> membership, Tensor<T> x, out Tensor<T> counts)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int k = membership.Shape[1];
        counts = engine.ReduceSum(membership, new[] { 0 }, false);
        var sums = engine.TensorMatMul(engine.TensorTranspose(membership), x);
        var divisor = engine.Reshape(engine.TensorMax(counts, ops.One), new[] { k, 1 });
        return engine.TensorDivide(sums, engine.TensorBroadcastTo(divisor, sums.Shape.ToArray()));
    }

    /// <summary>
    /// Column sums of squared deviations from the column means of a 2-D tensor ([n, p] to [p]).
    /// </summary>
    public static Tensor<T> ColumnSumSquaredDeviations<T>(Tensor<T> x)
    {
        var engine = AiDotNetEngine.Current;
        var centered = engine.TensorSubtract(x, engine.TensorBroadcastTo(
            engine.ReduceMean(x, new[] { 0 }, true), x.Shape.ToArray()));
        return engine.ReduceSum(engine.TensorSquare(centered), new[] { 0 }, false);
    }

    /// <summary>
    /// For each non-empty group (in group order), the first (lowest-index) member with the largest score.
    /// <paramref name="groups"/> holds each position's group in 0..<paramref name="groupCount"/>-1.
    /// </summary>
    public static int[] BestPerGroup<T>(Tensor<T> scores, int[] groups, int groupCount)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int p = scores.Length;
        if (p == 0 || groupCount == 0)
            return Array.Empty<int>();

        var shape = new[] { groupCount, p };
        var member = engine.TensorTranspose(PreprocessingTensorOps.ClassOneHot<T>(groups, groupCount));
        var broadcast = engine.TensorBroadcastTo(engine.Reshape(scores, new[] { 1, p }), shape);
        var candidates = engine.TensorWhere(member, broadcast, PreprocessingTensorOps.Filled(shape, ops.MinValue));
        var groupMax = engine.TensorBroadcastTo(engine.ReduceMax(candidates, new[] { 1 }, true), shape);
        var isBest = engine.TensorMultiply(member, engine.TensorEquals(candidates, groupMax));
        var best = PreprocessingTensorOps.FirstNonZeroPerRow(isBest);
        var nonEmpty = PreprocessingTensorOps.MaskPositions(
            engine.TensorGreaterThan(engine.ReduceSum(member, new[] { 1 }, false), ops.Zero));
        return Array.ConvertAll(nonEmpty, g => best[g]);
    }

    /// <summary>
    /// Per-column least-squares slope of <paramref name="ys"/> ([count, p]) on <paramref name="xs"/> ([count]),
    /// using only the points where the 0/1 <paramref name="valid"/> mask ([count, p]) is set.
    /// <paramref name="fitted"/> marks the columns with at least two valid points and a centred x sum of squares
    /// of at least <paramref name="minDenominator"/>; other columns' slopes are meaningless.
    /// </summary>
    public static Tensor<T> MaskedSlope<T>(Tensor<T> xs, Tensor<T> ys, Tensor<T> valid, double minDenominator, out Tensor<T> fitted)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int count = ys.Shape[0];
        int p = ys.Shape[1];
        var shape = new[] { count, p };
        var x = engine.TensorBroadcastTo(engine.Reshape(xs, new[] { count, 1 }), shape);
        var y = engine.TensorWhere(valid, ys, new Tensor<T>(shape));
        var points = engine.ReduceSum(valid, new[] { 0 }, true);
        var meanX = engine.TensorDivide(engine.ReduceSum(engine.TensorMultiply(x, valid), new[] { 0 }, true), points);
        var meanY = engine.TensorDivide(engine.ReduceSum(y, new[] { 0 }, true), points);
        var dx = engine.TensorMultiply(engine.TensorSubtract(x, engine.TensorBroadcastTo(meanX, shape)), valid);
        var dy = engine.TensorMultiply(engine.TensorSubtract(y, engine.TensorBroadcastTo(meanY, shape)), valid);
        var numerator = engine.ReduceSum(engine.TensorMultiply(dx, dy), new[] { 0 }, false);
        var denominator = engine.ReduceSum(engine.TensorMultiply(dx, dx), new[] { 0 }, false);
        fitted = engine.TensorMultiply(
            PreprocessingTensorOps.AtLeast(engine.Reshape(points, new[] { p }), ops.FromDouble(2)),
            PreprocessingTensorOps.Not(engine.TensorLessThan(denominator, ops.FromDouble(minDenominator))));
        return engine.TensorDivide(numerator, denominator);
    }

    /// <summary>
    /// Shannon entropy in bits of every row of histogram counts ([rows, bins] to [rows]) over
    /// <paramref name="total"/> samples: <c>-sum p log2 p</c> over the non-empty bins.
    /// </summary>
    public static Tensor<T> EntropyBits<T>(Tensor<T> counts, int total)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        var probabilities = engine.TensorDivideScalar(counts, ops.FromDouble(total));
        var terms = engine.TensorWhere(engine.TensorGreaterThan(counts, ops.Zero),
            engine.TensorMultiply(probabilities, engine.TensorLog(probabilities)), new Tensor<T>(counts.Shape.ToArray()));
        return engine.TensorDivideScalar(
            engine.TensorNegate(engine.ReduceSum(terms, new[] { 1 }, false)), ops.FromDouble(Math.Log(2)));
    }

    /// <summary>
    /// Per-column bin counts of a one-hot binning ([n, p * bins] to [p, bins]).
    /// </summary>
    public static Tensor<T> BinCounts<T>(Tensor<T> oneHot, int bins)
    {
        var engine = AiDotNetEngine.Current;
        int p = oneHot.Shape[1] / bins;
        return engine.Reshape(engine.ReduceSum(oneHot, new[] { 0 }, false), new[] { p, bins });
    }

    /// <summary>
    /// Gaussian kernel sums for 1-D kernel density estimates of every column: for each grid point
    /// (<paramref name="grid"/>, [g, p]) and column, <c>sum_i exp(-0.5 ((grid - values[i]) / bandwidth)^2)</c> over the
    /// rows of <paramref name="values"/> ([m, p]). Returns [g, p] (unnormalized).
    /// </summary>
    public static Tensor<T> GaussianKernelSums<T>(Tensor<T> grid, Tensor<T> values, T bandwidth)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int g = grid.Shape[0];
        int m = values.Shape[0];
        int p = grid.Shape[1];
        var shape = new[] { g, m, p };
        var u = engine.TensorDivideScalar(engine.TensorSubtract(
            engine.TensorBroadcastTo(engine.Reshape(grid, new[] { g, 1, p }), shape),
            engine.TensorBroadcastTo(engine.Reshape(values, new[] { 1, m, p }), shape)), bandwidth);
        var kernel = engine.TensorExp(engine.TensorMultiplyScalar(engine.TensorSquare(u), ops.FromDouble(-0.5)));
        return engine.ReduceSum(kernel, new[] { 1 }, false);
    }

    /// <summary>The positions (ascending) whose group in <paramref name="groups"/> is <paramref name="group"/>.</summary>
    public static int[] GroupMembers(int[] groups, int group) =>
        Enumerable.Range(0, groups.Length).Where(i => groups[i] == group).ToArray();

    /// <summary>The rows of a 2-D tensor at the given positions, in order.</summary>
    public static Tensor<T> Rows<T>(Tensor<T> x, int[] rows) =>
        AiDotNetEngine.Current.TensorIndexSelect(x, new Tensor<int>((int[])rows.Clone(), new[] { rows.Length }), 0);

    /// <summary>
    /// A [count, p] grid whose column j runs evenly from <c>low[j]</c> (row 0) to <c>low[j] + span[j]</c>
    /// (row count - 1): row i is <c>low + span * i / (count - 1)</c>.
    /// </summary>
    public static Tensor<T> EvenGrid<T>(Tensor<T> low, Tensor<T> span, int count)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int p = low.Length;
        var shape = new[] { count, p };
        var steps = engine.TensorBroadcastTo(engine.Reshape(PreprocessingTensorOps.Range<T>(count), new[] { count, 1 }), shape);
        var scaled = engine.TensorDivideScalar(
            engine.TensorMultiply(steps, engine.TensorBroadcastTo(engine.Reshape(span, new[] { 1, p }), shape)),
            ops.FromDouble(count - 1));
        return engine.TensorAdd(engine.TensorBroadcastTo(engine.Reshape(low, new[] { 1, p }), shape), scaled);
    }

    /// <summary>
    /// Top eigenpairs of a symmetric [p, p] matrix by power iteration with deflation: each component starts from
    /// the uniform vector 1/p, runs <paramref name="iterations"/> matrix-vector products (normalizing whenever the
    /// norm is above 1e-10), takes the Rayleigh quotient as its eigenvalue (0 when the vector's squared norm is not
    /// above 1e-10; floored at 0 with <paramref name="clampNegative"/>) and deflates the matrix by
    /// <c>lambda v v^T</c>. Returns the eigenvectors as rows ([components, p]).
    /// </summary>
    public static Tensor<T> PowerIterationComponents<T>(
        Tensor<T> matrix, int components, int iterations, bool clampNegative, out Tensor<T> eigenvalues)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int p = matrix.Shape[0];
        T floor = ops.FromDouble(1e-10);
        T zero = ops.Zero;
        var one = PreprocessingTensorOps.Filled(new[] { 1, 1 }, ops.One);
        var noValue = new Tensor<T>(new[] { 1, 1 });
        var start = PreprocessingTensorOps.Filled(new[] { p, 1 }, ops.Divide(ops.One, ops.FromDouble(p)));
        var deflated = matrix;
        var vectors = new Tensor<T>[components];
        var values = new Tensor<T>[components];

        for (int k = 0; k < components; k++)
        {
            var v = start;
            for (int iter = 0; iter < iterations; iter++)
            {
                var next = engine.TensorMatMul(deflated, v);
                var norm = engine.TensorSqrt(engine.ReduceSum(engine.TensorSquare(next), new[] { 0 }, true));
                var divisor = engine.TensorWhere(engine.TensorGreaterThan(norm, floor), norm, one);
                v = engine.TensorDivide(next, engine.TensorBroadcastTo(divisor, new[] { p, 1 }));
            }

            var mv = engine.TensorMatMul(deflated, v);
            var numerator = engine.ReduceSum(engine.TensorMultiply(v, mv), new[] { 0 }, true);
            var denominator = engine.ReduceSum(engine.TensorSquare(v), new[] { 0 }, true);
            var eigenvalue = engine.TensorWhere(engine.TensorGreaterThan(denominator, floor),
                engine.TensorDivide(numerator, denominator), noValue);
            if (clampNegative)
                eigenvalue = engine.TensorMax(eigenvalue, zero);

            var vector = engine.Reshape(v, new[] { p });
            vectors[k] = engine.Reshape(v, new[] { 1, p });
            values[k] = engine.Reshape(eigenvalue, new[] { 1 });

            // Deflate
            deflated = engine.TensorSubtract(deflated, engine.TensorMultiply(
                engine.TensorOuter(vector, vector), engine.TensorBroadcastTo(eigenvalue, new[] { p, p })));
        }

        if (components == 0)
        {
            eigenvalues = new Tensor<T>(new[] { 0 });
            return new Tensor<T>(new[] { 0, p });
        }

        eigenvalues = engine.TensorConcatenate(values, 0);
        return engine.TensorConcatenate(vectors, 0);
    }

    /// <summary>
    /// Feature importances from component loadings ([components, p]) and component weights ([components]):
    /// <c>sum_k weights[k] * loadings[k, j]^2</c> for every feature j.
    /// </summary>
    public static Tensor<T> WeightedSquaredLoadings<T>(Tensor<T> loadings, Tensor<T> weights)
    {
        var engine = AiDotNetEngine.Current;
        int components = loadings.Shape[0];
        int p = loadings.Shape[1];
        if (components == 0)
            return new Tensor<T>(new[] { p });
        return engine.Reshape(engine.TensorMatMul(
            engine.Reshape(weights, new[] { 1, components }), engine.TensorSquare(loadings)), new[] { p });
    }

    /// <summary>
    /// The average over all pairs of rows (r1 &lt; r2) of <c>|rows[r1] - rows[r2]|</c> (or its square with
    /// <paramref name="squared"/>), per column: [k, p] to [p]; 0 when there are fewer than two rows.
    /// </summary>
    public static Tensor<T> MeanPairwiseDifference<T>(Tensor<T> rows, bool squared)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int k = rows.Shape[0];
        int p = rows.Shape[1];
        if (k < 2)
            return new Tensor<T>(new[] { p });

        // Every ordered pair once in each order; the diagonal is exactly 0, so the c1 &lt; c2 sum is half the total
        var shape = new[] { k, k, p };
        var diff = engine.TensorSubtract(
            engine.TensorBroadcastTo(engine.Reshape(rows, new[] { k, 1, p }), shape),
            engine.TensorBroadcastTo(engine.Reshape(rows, new[] { 1, k, p }), shape));
        var magnitude = squared ? engine.TensorSquare(diff) : engine.TensorAbs(diff);
        int pairCount = k * (k - 1) / 2;
        return engine.TensorDivideScalar(engine.ReduceSum(magnitude, new[] { 0, 1 }, false), ops.FromDouble(2.0 * pairCount));
    }

    /// <summary>Column-wise median of a 2-D tensor ([m, p] to [p]); the mean of the two middle values for even m.</summary>
    public static Tensor<T> ColumnMedian<T>(Tensor<T> x)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int m = x.Shape[0];
        int mid = m / 2;
        if (m % 2 != 0)
            return PreprocessingTensorOps.ColumnOrderStatistic(x, mid);

        return engine.TensorDivideScalar(engine.TensorAdd(
            PreprocessingTensorOps.ColumnOrderStatistic(x, mid - 1), PreprocessingTensorOps.ColumnOrderStatistic(x, mid)),
            ops.FromDouble(2));
    }

    /// <summary>
    /// Range-scaled histogram counts of every column of <paramref name="x"/> ([n, p]):
    /// <c>bin = min(trunc((x - min) / (max - min) * (bins - 1)), bins - 1)</c>. Returns [p, bins];
    /// <paramref name="constant"/> is the [p] 0/1 mask of columns whose range is below 1e-10 (their counts are all
    /// in bin 0 and should be ignored).
    /// </summary>
    public static Tensor<T> RangeBinCounts<T>(Tensor<T> x, int bins, out Tensor<T> constant)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = x.Shape[0];
        int p = x.Shape[1];
        var shape = new[] { n, p };
        var min = PreprocessingTensorOps.ColumnMin(x);
        var range = engine.TensorSubtract(PreprocessingTensorOps.ColumnMax(x), min);
        constant = engine.TensorLessThan(range, ops.FromDouble(1e-10));
        var safeRange = engine.TensorWhere(constant, PreprocessingTensorOps.Filled(new[] { p }, ops.One), range);
        var scaled = engine.TensorMultiplyScalar(engine.TensorDivide(
            engine.TensorSubtract(x, engine.TensorBroadcastTo(engine.Reshape(min, new[] { 1, p }), shape)),
            engine.TensorBroadcastTo(engine.Reshape(safeRange, new[] { 1, p }), shape)), ops.FromDouble(bins - 1));
        var binIndex = engine.TensorMin(engine.TensorTrunc(scaled), ops.FromDouble(bins - 1));
        binIndex = engine.TensorWhere(engine.TensorBroadcastTo(engine.Reshape(constant, new[] { 1, p }), shape),
            new Tensor<T>(shape), binIndex);
        return BinCounts(PreprocessingTensorOps.OneHotBins(binIndex, bins), bins);
    }

    /// <summary>
    /// Cyclic coordinate descent on a linear model: for every iteration and every column j in order,
    /// <c>rho = x_j . (residual + beta_j x_j)</c> (the unnormalized partial-residual correlation, as a [1] tensor) is
    /// passed to <paramref name="update"/> together with j, the returned [1] value becomes <c>beta_j</c> and the
    /// residual is updated. Stops after an iteration whose largest coefficient change is below
    /// <paramref name="tolerance"/>. <paramref name="target"/> ([n]) is the starting residual (the centred target);
    /// <paramref name="residual"/> receives the final residual. Returns the coefficients ([p]).
    /// </summary>
    public static Tensor<T> CoordinateDescent<T>(
        Tensor<T> x, Tensor<T> target, int maxIterations, double tolerance,
        Func<Tensor<T>, int, Tensor<T>> update, out Tensor<T> residual)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = x.Shape[0];
        int p = x.Shape[1];
        var columns = new Tensor<T>[p];
        var selectors = new Tensor<T>[p];
        var identity = engine.TensorEye<T>(p);
        for (int j = 0; j < p; j++)
        {
            columns[j] = PreprocessingTensorOps.ColumnVector(x, j);
            selectors[j] = PreprocessingTensorOps.RowVector(identity, j);
        }

        var squaredNorms = engine.ReduceSum(engine.TensorSquare(x), new[] { 0 }, false);
        var beta = new Tensor<T>(new[] { p });
        var current = target;
        T tol = ops.FromDouble(tolerance);
        var columnShape = new[] { n };
        var betaShape = new[] { p };

        for (int iter = 0; iter < maxIterations; iter++)
        {
            var maxChange = new Tensor<T>(new[] { 1 });
            for (int j = 0; j < p; j++)
            {
                var old = engine.TensorSlice(beta, new[] { j }, new[] { 1 });
                var rho = engine.TensorAdd(
                    engine.ReduceSum(engine.TensorMultiply(columns[j], current), new[] { 0 }, true),
                    engine.TensorMultiply(old, engine.TensorSlice(squaredNorms, new[] { j }, new[] { 1 })));
                var updated = update(rho, j);
                var delta = engine.TensorSubtract(updated, old);
                current = engine.TensorSubtract(current,
                    engine.TensorMultiply(columns[j], engine.TensorBroadcastTo(delta, columnShape)));
                beta = engine.TensorWhere(selectors[j], engine.TensorBroadcastTo(updated, betaShape), beta);
                maxChange = engine.TensorMax(maxChange, engine.TensorAbs(delta));
            }

            if (PreprocessingTensorOps.IsBelow(engine.TensorMaxValue(maxChange), tol))
                break;
        }

        residual = current;
        return beta;
    }

    /// <summary>Element-wise soft thresholding: <c>sign(z) * max(|z| - threshold, 0)</c>.</summary>
    public static Tensor<T> SoftThreshold<T>(Tensor<T> z, T threshold)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        return engine.TensorMultiply(engine.TensorSign(z),
            engine.TensorMax(engine.TensorSubtractScalar(engine.TensorAbs(z), threshold), ops.Zero));
    }

    /// <summary>
    /// Population standardization of every column: <c>(x - mean) / std</c>, with a std below 1e-10 replaced by 1.
    /// <paramref name="stds"/> receives the (replaced) [p] standard deviations.
    /// </summary>
    public static Tensor<T> Standardize<T>(Tensor<T> x, out Tensor<T> stds)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = x.Shape[0];
        int p = x.Shape[1];
        var shape = new[] { n, p };
        var centered = engine.TensorSubtract(x, engine.TensorBroadcastTo(engine.ReduceMean(x, new[] { 0 }, true), shape));
        var std = engine.TensorSqrt(engine.ReduceMean(engine.TensorSquare(centered), new[] { 0 }, false));
        stds = engine.TensorWhere(engine.TensorLessThan(std, ops.FromDouble(1e-10)), PreprocessingTensorOps.Filled(new[] { p }, ops.One), std);
        return engine.TensorDivide(centered, engine.TensorBroadcastTo(engine.Reshape(stds, new[] { 1, p }), shape));
    }

    /// <summary><c>y - mean(y)</c>.</summary>
    public static Tensor<T> Center<T>(Tensor<T> y)
    {
        var engine = AiDotNetEngine.Current;
        return engine.TensorSubtractScalar(y, engine.TensorMean(y));
    }

    /// <summary>
    /// MCP thresholding (Zhang 2010) for a standardized column: <c>S(z, lambda) * gamma / (gamma - 1)</c> for
    /// <c>|z| &lt;= gamma * lambda</c>, and <c>z</c> (no shrinkage) beyond.
    /// </summary>
    public static Tensor<T> McpThreshold<T>(Tensor<T> z, double lambda, double gamma)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        var shrunk = engine.TensorMultiplyScalar(SoftThreshold(z, ops.FromDouble(lambda)), ops.FromDouble(gamma / (gamma - 1)));
        var inside = PreprocessingTensorOps.Not(engine.TensorGreaterThan(engine.TensorAbs(z), ops.FromDouble(gamma * lambda)));
        return engine.TensorWhere(inside, shrunk, z);
    }

    /// <summary>
    /// SCAD thresholding (Fan and Li 2001) for a standardized column: <c>S(z, lambda)</c> for
    /// <c>|z| &lt;= 2 lambda</c>, <c>((a - 1) z - sign(z) a lambda) / (a - 2)</c> for <c>2 lambda &lt; |z| &lt;= a lambda</c>,
    /// and <c>z</c> beyond.
    /// </summary>
    public static Tensor<T> ScadThreshold<T>(Tensor<T> z, double lambda, double a)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        var magnitude = engine.TensorAbs(z);
        var soft = SoftThreshold(z, ops.FromDouble(lambda));
        var transition = engine.TensorDivideScalar(engine.TensorSubtract(
            engine.TensorMultiplyScalar(z, ops.FromDouble(a - 1)),
            engine.TensorMultiplyScalar(engine.TensorSign(z), ops.FromDouble(a * lambda))), ops.FromDouble(a - 2));
        var inSoft = PreprocessingTensorOps.Not(engine.TensorGreaterThan(magnitude, ops.FromDouble(2 * lambda)));
        var inTransition = PreprocessingTensorOps.Not(engine.TensorGreaterThan(magnitude, ops.FromDouble(a * lambda)));
        return engine.TensorWhere(inSoft, soft, engine.TensorWhere(inTransition, transition, z));
    }
}
