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

    /// <summary>
    /// Ridge coefficients on centered data: solves <c>(Xc'Xc + alpha I) beta = Xc'yc</c> by Cholesky factorization
    /// (the system is symmetric positive definite for <c>alpha &gt; 0</c>), as sklearn Ridge with fit_intercept.
    /// </summary>
    public static Tensor<T> RidgeCoefficients<T>(Tensor<T> x, Tensor<T> y, double alpha)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = x.Shape[0];
        int p = x.Shape[1];
        var centered = engine.TensorSubtract(x, engine.TensorBroadcastTo(engine.ReduceMean(x, new[] { 0 }, true), new[] { n, p }));
        var transposed = engine.TensorTranspose(centered);
        var gram = engine.TensorAdd(engine.TensorMatMul(transposed, centered),
            engine.TensorMultiplyScalar(engine.TensorEye<T>(p), ops.FromDouble(alpha)));
        var rhs = engine.TensorMatMul(transposed, engine.Reshape(Center(y), new[] { n, 1 }));
        var factor = AiDotNet.Helpers.LinalgDispatch<T>.CholeskyEx(gram).L;
        return engine.Reshape(AiDotNet.Helpers.LinalgDispatch<T>.CholeskySolve(rhs, factor), new[] { p });
    }

    /// <summary>
    /// Solves <c>(X'X + alpha I) beta = X'y</c> for an already standardized/centered <paramref name="x"/> and
    /// <paramref name="y"/>. Uses LU with partial pivoting; when a pivot falls to 1e-10 or below (singular system,
    /// possible for <c>alpha = 0</c>) it returns the minimum-norm least-squares solution through the SVD
    /// pseudo-inverse, as numpy.linalg.lstsq / sklearn do.
    /// </summary>
    public static Tensor<T> RegularizedNormalSolve<T>(Tensor<T> x, Tensor<T> y, double alpha)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = x.Shape[0];
        int p = x.Shape[1];
        var transposed = engine.TensorTranspose(x);
        var gram = engine.TensorAdd(engine.TensorMatMul(transposed, x),
            engine.TensorMultiplyScalar(engine.TensorEye<T>(p), ops.FromDouble(alpha)));
        var rhs = engine.TensorMatMul(transposed, engine.Reshape(y, new[] { n, 1 }));
        T tolerance = ops.FromDouble(1e-10);

        var (lu, pivots) = AiDotNet.Helpers.LinalgDispatch<T>.LuFactor(gram);
        var pivotMagnitudes = engine.TensorAbs(AiDotNet.Helpers.LinalgDispatch<T>.Diagonal(lu));
        if (PreprocessingTensorOps.CountNonZero(engine.TensorGreaterThan(pivotMagnitudes, tolerance)) == p)
            return engine.Reshape(AiDotNet.Helpers.LinalgDispatch<T>.LuSolve(lu, pivots, rhs), new[] { p });

        var (u, s, vh) = AiDotNet.Helpers.LinalgDispatch<T>.Svd(gram, false);
        var kept = engine.TensorGreaterThan(s, tolerance);
        var safe = engine.TensorWhere(kept, s, PreprocessingTensorOps.Filled(new[] { p }, ops.One));
        var inverse = engine.TensorWhere(kept, engine.TensorDivide(PreprocessingTensorOps.Filled(new[] { p }, ops.One), safe),
            PreprocessingTensorOps.Filled(new[] { p }, ops.Zero));
        var projected = engine.TensorMultiply(engine.Reshape(engine.TensorMatMul(engine.TensorTranspose(u), rhs), new[] { p }), inverse);
        return engine.Reshape(engine.TensorMatMul(engine.TensorTranspose(vh), engine.Reshape(projected, new[] { p, 1 })), new[] { p });
    }

    /// <summary>
    /// Best Gini impurity reduction of a single-threshold split (decision stump) on every column of
    /// <paramref name="x"/> ([m, q]) for the class positions <paramref name="classIndex"/> (one per row): thresholds
    /// lie between consecutive sorted values that differ by at least 1e-10; the result is floored at 0, and is 0
    /// for fewer than two rows.
    /// </summary>
    public static Tensor<T> BestGiniSplitReduction<T>(Tensor<T> x, int[] classIndex, int classCount)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int m = x.Shape[0];
        int q = x.Shape[1];
        if (m < 2)
            return new Tensor<T>(new[] { q });

        var sorted = engine.TensorSort(x, 0, false).Item1;
        var order = engine.TensorArgsort(x, 0, false);
        var oneHot = PreprocessingTensorOps.ClassOneHot<T>(classIndex, classCount);
        var flatOrder = engine.Reshape(engine.TensorTranspose(order), new[] { q * m });
        var sortedClasses = engine.Reshape(engine.TensorIndexSelect(oneHot, flatOrder, 0), new[] { q, m, classCount });
        var left = engine.TensorSlice(engine.TensorCumSum(sortedClasses, 1), new[] { 0, 0, 0 }, new[] { q, m - 1, classCount });
        var totals = engine.ReduceSum(oneHot, new[] { 0 }, true);
        var splitShape = new[] { q, m - 1, classCount };
        var right = engine.TensorSubtract(engine.TensorBroadcastTo(engine.Reshape(totals, new[] { 1, 1, classCount }), splitShape), left);

        // Left sizes 1..m-1 and right sizes m-1..1 for split positions i = 1..m-1
        var leftSizes = engine.TensorAddScalar(PreprocessingTensorOps.Range<T>(m - 1), ops.One);
        var rightSizes = engine.TensorSubtract(PreprocessingTensorOps.Filled(new[] { m - 1 }, ops.FromDouble(m)), leftSizes);
        var sizeShape = new[] { q, m - 1 };
        var leftSize3 = engine.TensorBroadcastTo(engine.Reshape(leftSizes, new[] { 1, m - 1, 1 }), splitShape);
        var rightSize3 = engine.TensorBroadcastTo(engine.Reshape(rightSizes, new[] { 1, m - 1, 1 }), splitShape);
        var one = PreprocessingTensorOps.Filled(sizeShape, ops.One);
        var leftGini = engine.TensorSubtract(one, engine.ReduceSum(engine.TensorSquare(engine.TensorDivide(left, leftSize3)), new[] { 2 }, false));
        var rightGini = engine.TensorSubtract(one, engine.ReduceSum(engine.TensorSquare(engine.TensorDivide(right, rightSize3)), new[] { 2 }, false));
        var weighted = engine.TensorDivideScalar(engine.TensorAdd(
            engine.TensorMultiply(leftGini, engine.TensorBroadcastTo(engine.Reshape(leftSizes, new[] { 1, m - 1 }), sizeShape)),
            engine.TensorMultiply(rightGini, engine.TensorBroadcastTo(engine.Reshape(rightSizes, new[] { 1, m - 1 }), sizeShape))),
            ops.FromDouble(m));
        T overall = ops.Subtract(ops.One, engine.TensorSumOfSquares(engine.TensorDivideScalar(totals, ops.FromDouble(m))));
        var reduction = engine.TensorAddScalar(engine.TensorNegate(weighted), overall);

        // Valid thresholds: consecutive sorted values at least 1e-10 apart
        var sortedT = engine.TensorTranspose(sorted);
        var gaps = engine.TensorAbs(engine.TensorSubtract(
            engine.TensorSlice(sortedT, new[] { 0, 1 }, new[] { q, m - 1 }),
            engine.TensorSlice(sortedT, new[] { 0, 0 }, new[] { q, m - 1 })));
        var valid = PreprocessingTensorOps.Not(engine.TensorLessThan(gaps, ops.FromDouble(1e-10)));
        var candidates = engine.TensorWhere(valid, reduction, new Tensor<T>(sizeShape));
        return engine.TensorMax(engine.ReduceMax(candidates, new[] { 1 }, false), ops.Zero);
    }

    /// <summary>
    /// Best single-threshold split of a tree node. <paramref name="nodeX"/> ([m, f]) holds the node's rows of the
    /// candidate features (in candidate order). For <see cref="SplitCriterion.SumOfSquares"/>,
    /// <paramref name="rowValues"/> is the [m] regression target and the gain is
    /// <c>parentImpurity - SSE_left - SSE_right</c>; for <see cref="SplitCriterion.Gini"/> it is the [m, K] class
    /// one-hot and the gain is <c>parentImpurity - (m_l gini_l + m_r gini_r) / m</c>. Thresholds are midpoints
    /// between consecutive distinct values and both sides need at least <paramref name="minLeaf"/> rows. The
    /// first (candidate order, then ascending threshold) strictly positive maximum gain wins, like the strict
    /// <c>gain &gt; bestGain</c> scan starting at 0. Returns false when no split has a positive gain.
    /// </summary>
    public static bool TryBestSplit<T>(
        Tensor<T> nodeX, Tensor<T> rowValues, SplitCriterion criterion, T parentImpurity, int minLeaf,
        out int column, out T threshold, out Tensor<T> gain)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int m = nodeX.Shape[0];
        int f = nodeX.Shape[1];
        column = -1;
        threshold = ops.Zero;
        gain = new Tensor<T>(new[] { 1 });
        if (m < 2 || f == 0)
            return false;

        int splits = m - 1;
        var sizeShape = new[] { f, splits };
        var sorted = engine.TensorTranspose(engine.TensorSort(nodeX, 0, false).Item1);
        var flatOrder = engine.Reshape(engine.TensorTranspose(engine.TensorArgsort(nodeX, 0, false)), new[] { f * m });
        var leftSizes = engine.TensorAddScalar(PreprocessingTensorOps.Range<T>(splits), ops.One);
        var rightSizes = engine.TensorSubtract(PreprocessingTensorOps.Filled(new[] { splits }, ops.FromDouble(m)), leftSizes);
        var leftSize2 = engine.TensorBroadcastTo(engine.Reshape(leftSizes, new[] { 1, splits }), sizeShape);
        var rightSize2 = engine.TensorBroadcastTo(engine.Reshape(rightSizes, new[] { 1, splits }), sizeShape);

        Tensor<T> gains;
        if (criterion == SplitCriterion.SumOfSquares)
        {
            var values = engine.Reshape(engine.TensorIndexSelect(rowValues, flatOrder, 0), new[] { f, m });
            var sums = engine.TensorCumSum(values, 1);
            var squares = engine.TensorCumSum(engine.TensorSquare(values), 1);
            var leftSum = engine.TensorSlice(sums, new[] { 0, 0 }, sizeShape);
            var leftSq = engine.TensorSlice(squares, new[] { 0, 0 }, sizeShape);
            var rightSum = engine.TensorSubtract(
                engine.TensorBroadcastTo(engine.TensorSlice(sums, new[] { 0, splits }, new[] { f, 1 }), sizeShape), leftSum);
            var rightSq = engine.TensorSubtract(
                engine.TensorBroadcastTo(engine.TensorSlice(squares, new[] { 0, splits }, new[] { f, 1 }), sizeShape), leftSq);
            var leftSse = engine.TensorSubtract(leftSq, engine.TensorDivide(engine.TensorSquare(leftSum), leftSize2));
            var rightSse = engine.TensorSubtract(rightSq, engine.TensorDivide(engine.TensorSquare(rightSum), rightSize2));
            gains = engine.TensorAddScalar(engine.TensorNegate(engine.TensorAdd(leftSse, rightSse)), parentImpurity);
        }
        else
        {
            int classCount = rowValues.Shape[1];
            var classShape = new[] { f, splits, classCount };
            var classes = engine.Reshape(engine.TensorIndexSelect(rowValues, flatOrder, 0), new[] { f, m, classCount });
            var left = engine.TensorSlice(engine.TensorCumSum(classes, 1), new[] { 0, 0, 0 }, classShape);
            var totals = engine.Reshape(engine.ReduceSum(rowValues, new[] { 0 }, false), new[] { 1, 1, classCount });
            var right = engine.TensorSubtract(engine.TensorBroadcastTo(totals, classShape), left);
            var leftSize3 = engine.TensorBroadcastTo(engine.Reshape(leftSizes, new[] { 1, splits, 1 }), classShape);
            var rightSize3 = engine.TensorBroadcastTo(engine.Reshape(rightSizes, new[] { 1, splits, 1 }), classShape);
            var one = PreprocessingTensorOps.Filled(sizeShape, ops.One);
            var leftGini = engine.TensorSubtract(one,
                engine.ReduceSum(engine.TensorSquare(engine.TensorDivide(left, leftSize3)), new[] { 2 }, false));
            var rightGini = engine.TensorSubtract(one,
                engine.ReduceSum(engine.TensorSquare(engine.TensorDivide(right, rightSize3)), new[] { 2 }, false));
            var weighted = engine.TensorDivideScalar(engine.TensorAdd(
                engine.TensorMultiply(leftGini, leftSize2), engine.TensorMultiply(rightGini, rightSize2)), ops.FromDouble(m));
            gains = engine.TensorAddScalar(engine.TensorNegate(weighted), parentImpurity);
        }

        // Valid splits: a boundary between distinct values with at least minLeaf rows on each side
        T minimum = ops.FromDouble(minLeaf);
        var lower = engine.TensorSlice(sorted, new[] { 0, 0 }, sizeShape);
        var upper = engine.TensorSlice(sorted, new[] { 0, 1 }, sizeShape);
        var valid = engine.TensorMultiply(engine.TensorGreaterThan(upper, lower), engine.TensorMultiply(
            PreprocessingTensorOps.Not(engine.TensorLessThan(leftSize2, minimum)),
            PreprocessingTensorOps.Not(engine.TensorLessThan(rightSize2, minimum))));
        var candidates = engine.Reshape(engine.TensorWhere(valid, gains, new Tensor<T>(sizeShape)), new[] { f * splits });
        if (!PreprocessingTensorOps.IsAbove(engine.TensorMaxValue(candidates), ops.Zero))
            return false;

        int best = PreprocessingTensorOps.FirstArgMax(candidates);
        column = best / splits;
        int position = best % splits;
        gain = engine.TensorSlice(candidates, new[] { best }, new[] { 1 });
        threshold = ops.Divide(
            engine.TensorSum(engine.TensorSlice(sorted, new[] { column, position }, new[] { 1, 2 })), ops.FromDouble(2));
        return true;
    }

    /// <summary>
    /// Splits the node <paramref name="rows"/> by their values in <paramref name="nodeColumn"/>: at most
    /// <paramref name="threshold"/> goes left, the rest right, each in node order.
    /// </summary>
    public static void PartitionRows<T>(Tensor<T> nodeColumn, T threshold, int[] rows, out int[] left, out int[] right)
    {
        var engine = AiDotNetEngine.Current;
        var above = engine.TensorGreaterThan(nodeColumn, threshold);
        left = PreprocessingTensorOps.MaskPositions(PreprocessingTensorOps.Not(above)).Select(i => rows[i]).ToArray();
        right = PreprocessingTensorOps.MaskPositions(above).Select(i => rows[i]).ToArray();
    }

    /// <summary>
    /// Sums [1] contributions into a [length] tensor at the given positions (repeated positions add up).
    /// </summary>
    public static Tensor<T> ScatterAdd<T>(IReadOnlyList<int> positions, IReadOnlyList<Tensor<T>> values, int length)
    {
        var engine = AiDotNetEngine.Current;
        int count = positions.Count;
        if (count == 0)
            return new Tensor<T>(new[] { length });

        var oneHot = engine.TensorOneHot<T>(new Tensor<int>(positions.ToArray(), new[] { count }), length);
        var column = engine.Reshape(engine.TensorConcatenate(values.ToArray(), 0), new[] { count, 1 });
        return engine.ReduceSum(
            engine.TensorMultiply(oneHot, engine.TensorBroadcastTo(column, new[] { count, length })), new[] { 0 }, false);
    }

    /// <summary>Gini impurity <c>1 - sum_k p_k^2</c> of the rows of a [m, K] class one-hot (0 for no rows).</summary>
    public static T GiniImpurity<T>(Tensor<T> oneHot)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int m = oneHot.Shape[0];
        if (m == 0)
            return ops.Zero;
        var proportions = engine.TensorDivideScalar(engine.ReduceSum(oneHot, new[] { 0 }, false), ops.FromDouble(m));
        return ops.Subtract(ops.One, engine.TensorSumOfSquares(proportions));
    }

    /// <summary>Sum of squared deviations from the mean of a 1-D tensor (two-pass).</summary>
    public static T SumSquaredDeviations<T>(Tensor<T> values)
    {
        var engine = AiDotNetEngine.Current;
        return engine.TensorSumOfSquares(engine.TensorSubtractScalar(values, engine.TensorMean(values)));
    }

    /// <summary>
    /// Double-centred absolute-distance matrix of a 1-D tensor (Szekely, Rizzo and Bakirov 2007):
    /// <c>A_ik = d_ik - mean_k d_ik - mean_i d_ik + mean d</c> with <c>d_ik = |v_i - v_k|</c>.
    /// </summary>
    public static Tensor<T> DoubleCentredDistances<T>(Tensor<T> values)
    {
        var engine = AiDotNetEngine.Current;
        int s = values.Length;
        var shape = new[] { s, s };
        var distances = engine.TensorAbs(engine.TensorSubtract(
            engine.TensorBroadcastTo(engine.Reshape(values, new[] { s, 1 }), shape),
            engine.TensorBroadcastTo(engine.Reshape(values, new[] { 1, s }), shape)));
        var rowMeans = engine.TensorBroadcastTo(engine.ReduceMean(distances, new[] { 1 }, true), shape);
        var colMeans = engine.TensorBroadcastTo(engine.ReduceMean(distances, new[] { 0 }, true), shape);
        return engine.TensorAddScalar(engine.TensorSubtract(engine.TensorSubtract(distances, rowMeans), colMeans),
            engine.TensorMean(distances));
    }

    /// <summary>
    /// Squared sample distance covariances (V-statistics) of every column of <paramref name="x"/> ([s, p]) with
    /// <paramref name="y"/> ([s]): <c>dCov^2(x_j, y) = mean(A_j * B)</c>; <paramref name="xVariances"/> receives
    /// <c>dVar^2(x_j) = mean(A_j^2)</c> and <paramref name="yVariance"/> <c>dVar^2(y)</c>. Returns [p].
    /// </summary>
    public static Tensor<T> SquaredDistanceCovariances<T>(Tensor<T> x, Tensor<T> y, out Tensor<T> xVariances, out T yVariance)
    {
        var engine = AiDotNetEngine.Current;
        int p = x.Shape[1];
        var b = DoubleCentredDistances(y);
        yVariance = engine.TensorMean(engine.TensorSquare(b));
        var covariances = new Tensor<T>[p];
        var variances = new Tensor<T>[p];
        for (int j = 0; j < p; j++)
        {
            var a = DoubleCentredDistances(PreprocessingTensorOps.ColumnVector(x, j));
            covariances[j] = engine.ReduceMean(engine.Reshape(engine.TensorMultiply(a, b), new[] { a.Length }), new[] { 0 }, true);
            variances[j] = engine.ReduceMean(engine.Reshape(engine.TensorSquare(a), new[] { a.Length }), new[] { 0 }, true);
        }

        xVariances = p == 0 ? new Tensor<T>(new[] { 0 }) : engine.TensorConcatenate(variances, 0);
        return p == 0 ? new Tensor<T>(new[] { 0 }) : engine.TensorConcatenate(covariances, 0);
    }

    /// <summary>
    /// Range bins of every column of <paramref name="x"/> ([n, p]):
    /// <c>bin = min(trunc((x - min) / (max - min) * scale), bins - 1)</c>, 0 for a column whose range is not above
    /// 1e-10. Returns the bins one-hot as [n, p * bins].
    /// </summary>
    public static Tensor<T> RangeBinOneHot<T>(Tensor<T> x, int bins, double scale)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = x.Shape[0];
        int p = x.Shape[1];
        var shape = new[] { n, p };
        var min = PreprocessingTensorOps.ColumnMin(x);
        var range = engine.TensorSubtract(PreprocessingTensorOps.ColumnMax(x), min);
        var varying = engine.TensorGreaterThan(range, ops.FromDouble(1e-10));
        var safeRange = engine.TensorWhere(varying, range, PreprocessingTensorOps.Filled(new[] { p }, ops.One));
        var scaled = engine.TensorMultiplyScalar(engine.TensorDivide(
            engine.TensorSubtract(x, engine.TensorBroadcastTo(engine.Reshape(min, new[] { 1, p }), shape)),
            engine.TensorBroadcastTo(engine.Reshape(safeRange, new[] { 1, p }), shape)), ops.FromDouble(scale));
        var binIndex = engine.TensorMin(engine.TensorTrunc(scaled), ops.FromDouble(bins - 1));
        binIndex = engine.TensorWhere(engine.TensorBroadcastTo(engine.Reshape(varying, new[] { 1, p }), shape),
            binIndex, new Tensor<T>(shape));
        return PreprocessingTensorOps.OneHotBins(binIndex, bins);
    }

    /// <summary>
    /// Joint bin counts of every feature (one-hot [n, p * bins]) with a target (one-hot [n, targetBins]):
    /// [p, bins, targetBins].
    /// </summary>
    public static Tensor<T> JointBinCounts<T>(Tensor<T> featureOneHot, Tensor<T> targetOneHot, int bins)
    {
        var engine = AiDotNetEngine.Current;
        int p = featureOneHot.Shape[1] / bins;
        int targetBins = targetOneHot.Shape[1];
        return engine.Reshape(engine.TensorMatMul(engine.TensorTranspose(featureOneHot), targetOneHot),
            new[] { p, bins, targetBins });
    }

    /// <summary>
    /// 0/1 matrix [count, count] marking the template pairs (i, k) whose length-<paramref name="m"/> windows of
    /// <paramref name="values"/> starting at i and k differ by at most <paramref name="tolerance"/> ([1]) in every
    /// position (Chebyshev distance), as used by approximate and sample entropy.
    /// </summary>
    public static Tensor<T> TemplateMatches<T>(Tensor<T> values, int m, int count, Tensor<T> tolerance)
    {
        var engine = AiDotNetEngine.Current;
        var shape = new[] { count, count };
        var distance = new Tensor<T>(shape);
        for (int l = 0; l < m; l++)
        {
            var window = engine.TensorSlice(values, new[] { l }, new[] { count });
            distance = engine.TensorMax(distance, engine.TensorAbs(engine.TensorSubtract(
                engine.TensorBroadcastTo(engine.Reshape(window, new[] { count, 1 }), shape),
                engine.TensorBroadcastTo(engine.Reshape(window, new[] { 1, count }), shape))));
        }

        return PreprocessingTensorOps.Not(engine.TensorGreaterThan(distance,
            engine.TensorBroadcastTo(engine.Reshape(tolerance, new[] { 1, 1 }), shape)));
    }

    /// <summary>Population standard deviation of every column of <paramref name="x"/> ([n, p]).</summary>
    public static Tensor<T> ColumnStd<T>(Tensor<T> x)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        return engine.TensorSqrt(engine.TensorDivideScalar(ColumnSumSquaredDeviations(x), ops.FromDouble(x.Shape[0])));
    }

    /// <summary>
    /// Pearson chi-squared statistic of every feature's bins (one-hot [n, p * bins]) against the classes (one-hot
    /// [n, K]): <c>sum (O - E)^2 / E</c> over the cells with <c>E = n_bin * n_class / n &gt; 0</c>.
    /// <paramref name="occupiedBins"/> receives the number of non-empty bins per feature. Returns [p].
    /// </summary>
    public static Tensor<T> ContingencyChiSquared<T>(Tensor<T> binOneHot, Tensor<T> classOneHot, int bins, out Tensor<T> occupiedBins)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = binOneHot.Shape[0];
        int k = classOneHot.Shape[1];
        var observed = JointBinCounts(binOneHot, classOneHot, bins);
        int p = observed.Shape[0];
        var cellShape = new[] { p, bins, k };
        var binCounts = engine.ReduceSum(observed, new[] { 2 }, true);
        var classCounts = engine.Reshape(engine.ReduceSum(classOneHot, new[] { 0 }, false), new[] { 1, 1, k });
        var expected = engine.TensorDivideScalar(engine.TensorMultiply(
            engine.TensorBroadcastTo(binCounts, cellShape), engine.TensorBroadcastTo(classCounts, cellShape)), ops.FromDouble(n));
        var positive = engine.TensorGreaterThan(expected, ops.Zero);
        var safe = engine.TensorWhere(positive, expected, PreprocessingTensorOps.Filled(cellShape, ops.One));
        var terms = engine.TensorWhere(positive,
            engine.TensorDivide(engine.TensorSquare(engine.TensorSubtract(observed, expected)), safe), new Tensor<T>(cellShape));
        occupiedBins = engine.ReduceSum(engine.TensorGreaterThan(engine.Reshape(binCounts, new[] { p, bins }), ops.Zero),
            new[] { 1 }, false);
        return engine.ReduceSum(terms, new[] { 1, 2 }, false);
    }

    /// <summary>
    /// Chi-squared survival function <c>P(X &gt;= statistic)</c> for X ~ chi2(df), i.e. the regularized upper incomplete
    /// gamma <c>Q(df / 2, statistic / 2)</c> (scipy.stats.chi2.sf); 1 where <c>df &lt;= 0</c> or the statistic is not
    /// positive.
    /// </summary>
    public static Tensor<T> ChiSquaredSurvival<T>(Tensor<T> statistic, Tensor<T> df)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        var shape = statistic.Shape.ToArray();
        var one = PreprocessingTensorOps.Filled(shape, ops.One);
        var defined = engine.TensorMultiply(engine.TensorGreaterThan(df, ops.Zero), engine.TensorGreaterThan(statistic, ops.Zero));
        T half = ops.FromDouble(0.5);
        var q = engine.TensorIgammac(engine.TensorMultiplyScalar(engine.TensorWhere(defined, df, one), half),
            engine.TensorMultiplyScalar(engine.TensorWhere(defined, statistic, one), half));
        return engine.TensorWhere(defined, q, one);
    }

    /// <summary>
    /// Element-wise regularized incomplete beta function <c>I_x(a, b)</c> (scipy.special.betainc) by the continued
    /// fraction of Numerical Recipes (modified Lentz, up to 300 terms), using the symmetry
    /// <c>I_x(a, b) = 1 - I_{1-x}(b, a)</c> where the fraction converges faster. <c>x</c> is clamped to [0, 1].
    /// </summary>
    public static Tensor<T> RegularizedIncompleteBeta<T>(Tensor<T> a, Tensor<T> b, Tensor<T> x)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        var shape = x.Shape.ToArray();
        var zero = new Tensor<T>(shape);
        var one = PreprocessingTensorOps.Filled(shape, ops.One);
        var clamped = engine.TensorClamp(x, ops.Zero, ops.One);
        var interior = engine.TensorMultiply(engine.TensorGreaterThan(clamped, ops.Zero), engine.TensorLessThan(clamped, ops.One));
        var safeX = engine.TensorWhere(interior, clamped, PreprocessingTensorOps.Filled(shape, ops.FromDouble(0.5)));
        var complement = PreprocessingTensorOps.OneMinus(safeX);

        // Front factor x^a (1 - x)^b / (a B(a, b))
        var logFront = engine.TensorAdd(engine.TensorSubtract(engine.TensorSubtract(
            engine.TensorLgamma(engine.TensorAdd(a, b)), engine.TensorLgamma(a)), engine.TensorLgamma(b)),
            engine.TensorAdd(engine.TensorMultiply(a, engine.TensorLog(safeX)), engine.TensorMultiply(b, engine.TensorLog(complement))));
        var front = engine.TensorExp(logFront);

        // Direct fraction where x < (a + 1) / (a + b + 2), else the symmetric one
        var direct = engine.TensorLessThan(safeX, engine.TensorDivide(engine.TensorAddScalar(a, ops.One),
            engine.TensorAddScalar(engine.TensorAdd(a, b), ops.FromDouble(2))));
        var fa = engine.TensorWhere(direct, a, b);
        var fb = engine.TensorWhere(direct, b, a);
        var fx = engine.TensorWhere(direct, safeX, complement);
        var fraction = engine.TensorDivide(engine.TensorMultiply(front, BetaContinuedFraction(fa, fb, fx)), fa);
        var value = engine.TensorWhere(direct, fraction, PreprocessingTensorOps.OneMinus(fraction));

        // x = 0 gives 0 and x = 1 gives 1
        return engine.TensorWhere(interior, value, engine.TensorWhere(engine.TensorGreaterThan(clamped, ops.Zero), one, zero));
    }

    private static Tensor<T> BetaContinuedFraction<T>(Tensor<T> a, Tensor<T> b, Tensor<T> x)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        var shape = x.Shape.ToArray();
        T tiny = ops.FromDouble(1e-30);
        T epsilon = ops.FromDouble(3e-16);
        T one = ops.One;
        var tinyTensor = PreprocessingTensorOps.Filled(shape, tiny);

        Tensor<T> AwayFromZero(Tensor<T> t) =>
            engine.TensorWhere(engine.TensorLessThan(engine.TensorAbs(t), tiny), tinyTensor, t);

        var qab = engine.TensorAdd(a, b);
        var qap = engine.TensorAddScalar(a, one);
        var qam = engine.TensorSubtractScalar(a, one);
        var c = PreprocessingTensorOps.Filled(shape, one);
        var d = engine.TensorReciprocal(AwayFromZero(PreprocessingTensorOps.OneMinus(
            engine.TensorDivide(engine.TensorMultiply(qab, x), qap))));
        var h = d;

        for (int m = 1; m <= 300; m++)
        {
            T mT = PreprocessingTensorOps.CountAsScalar<T>(m);
            T m2 = PreprocessingTensorOps.CountAsScalar<T>(2 * m);

            // Even step: m (b - m) x / ((a - 1 + 2m)(a + 2m))
            var even = engine.TensorDivide(engine.TensorMultiplyScalar(engine.TensorMultiply(engine.TensorSubtractScalar(b, mT), x), mT),
                engine.TensorMultiply(engine.TensorAddScalar(qam, m2), engine.TensorAddScalar(a, m2)));
            d = engine.TensorReciprocal(AwayFromZero(engine.TensorAddScalar(engine.TensorMultiply(even, d), one)));
            c = AwayFromZero(engine.TensorAddScalar(engine.TensorDivide(even, c), one));
            h = engine.TensorMultiply(h, engine.TensorMultiply(d, c));

            // Odd step: -(a + m)(a + b + m) x / ((a + 2m)(a + 1 + 2m))
            var odd = engine.TensorNegate(engine.TensorDivide(
                engine.TensorMultiply(engine.TensorMultiply(engine.TensorAddScalar(a, mT), engine.TensorAddScalar(qab, mT)), x),
                engine.TensorMultiply(engine.TensorAddScalar(a, m2), engine.TensorAddScalar(qap, m2))));
            d = engine.TensorReciprocal(AwayFromZero(engine.TensorAddScalar(engine.TensorMultiply(odd, d), one)));
            c = AwayFromZero(engine.TensorAddScalar(engine.TensorDivide(odd, c), one));
            var delta = engine.TensorMultiply(d, c);
            h = engine.TensorMultiply(h, delta);

            if (PreprocessingTensorOps.IsBelow(engine.TensorMaxValue(engine.TensorAbs(engine.TensorSubtractScalar(delta, one))), epsilon))
                break;
        }

        return h;
    }

    /// <summary>
    /// F-distribution survival function <c>P(F &gt;= f)</c> for F ~ F(df1, df2) (scipy.stats.f.sf):
    /// <c>I_{df2 / (df2 + df1 f)}(df2 / 2, df1 / 2)</c>; 1 where <c>f &lt;= 0</c> or a degree of freedom is not positive.
    /// </summary>
    public static Tensor<T> FSurvival<T>(Tensor<T> f, double df1, double df2)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        var shape = f.Shape.ToArray();
        var one = PreprocessingTensorOps.Filled(shape, ops.One);
        if (df1 <= 0 || df2 <= 0)
            return one;

        var positive = engine.TensorGreaterThan(f, ops.Zero);
        var safeF = engine.TensorWhere(positive, f, one);
        var x = engine.TensorDivide(PreprocessingTensorOps.Filled(shape, ops.FromDouble(df2)),
            engine.TensorAddScalar(engine.TensorMultiplyScalar(safeF, ops.FromDouble(df1)), ops.FromDouble(df2)));
        var tail = RegularizedIncompleteBeta(PreprocessingTensorOps.Filled(shape, ops.FromDouble(df2 / 2)),
            PreprocessingTensorOps.Filled(shape, ops.FromDouble(df1 / 2)), x);
        return engine.TensorWhere(positive, tail, one);
    }

    /// <summary>
    /// Pearson correlation matrix of the columns of <paramref name="x"/> ([n, p] to [p, p]): unit diagonal,
    /// 0 for any off-diagonal pair involving a column whose centred sum of squares is not above
    /// <paramref name="epsilon"/> (the undefined constant-column case).
    /// </summary>
    public static Tensor<T> CorrelationMatrix<T>(Tensor<T> x, double epsilon = 1e-10)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int p = x.Shape[1];
        var xc = engine.TensorSubtract(x, engine.ReduceMean(x, new[] { 0 }, true));
        var cross = engine.TensorMatMul(engine.TensorTranspose(xc), xc);
        var ss = engine.ReduceSum(engine.TensorSquare(xc), new[] { 0 }, false);
        var varying = engine.TensorGreaterThan(ss, ops.FromDouble(epsilon));
        var scale = engine.TensorSqrt(engine.TensorWhere(varying, ss, PreprocessingTensorOps.Filled(new[] { p }, ops.One)));
        var r = engine.TensorDivide(cross, engine.TensorOuter(scale, scale));
        var defined = engine.TensorOuter(varying, varying);
        var eye = engine.TensorEye<T>(p);
        return engine.TensorWhere(eye, eye, engine.TensorWhere(defined, r, new Tensor<T>(new[] { p, p })));
    }
}

/// <summary>Impurity used by <see cref="FeatureSelectionTensorOps.TryBestSplit{T}"/>.</summary>
internal enum SplitCriterion
{
    /// <summary>Regression: sum of squared deviations from the node mean.</summary>
    SumOfSquares,

    /// <summary>Classification: Gini impurity.</summary>
    Gini
}
