using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection;

/// <summary>
/// Whole-array engine helpers shared by the feature selectors (column statistics, rank transforms and
/// class-wise moments), generic in <typeparamref name="T"/> and free of per-element loops.
/// </summary>
internal static class SelectorTensorOps
{
    /// <summary>Column medians of an [n, p] tensor (mean of the two middle values for even n), as [p].</summary>
    public static Tensor<T> ColumnMedian<T>(Tensor<T> x, bool keepDims = false) =>
        AiDotNetEngine.Current.TensorQuantile(x, 0.5, 0, keepDims, QuantileInterpolation.Linear);

    /// <summary>Column median absolute deviation (unscaled, scipy <c>median_abs_deviation(scale=1)</c>), as [p].</summary>
    public static Tensor<T> ColumnMedianAbsoluteDeviation<T>(Tensor<T> x)
    {
        var engine = AiDotNetEngine.Current;
        var deviations = engine.TensorAbs(engine.TensorSubtract(x, ColumnMedian(x, keepDims: true)));
        return ColumnMedian(deviations);
    }

    /// <summary>
    /// Power spectrum <c>|DFT_k|^2</c>, k = 0 .. w/2, of every column of a [w, m] tensor by the direct DFT
    /// (angle <c>2 pi k i / w</c>), as [w/2 + 1, m].
    /// </summary>
    public static Tensor<T> DftPower<T>(Tensor<T> x)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int w = x.Shape[0];
        int bins = w / 2 + 1;
        var angle = engine.TensorMultiplyScalar(
            engine.TensorOuter(engine.TensorArange<T>(0, bins, 1), engine.TensorArange<T>(0, w, 1)),
            ops.FromDouble(2 * Math.PI / w));
        var re = engine.TensorMatMul(engine.TensorCos(angle), x);
        var im = engine.TensorMatMul(engine.TensorSin(angle), x);
        return engine.TensorAdd(engine.TensorMultiply(re, re), engine.TensorMultiply(im, im));
    }

    /// <summary>
    /// Shannon entropy in bits of each column of a non-negative [k, m] tensor after normalising the column to sum
    /// to 1, counting only probabilities above <paramref name="minProbability"/>; a column whose total is not above
    /// <paramref name="minTotal"/> scores 0. Returns [m].
    /// </summary>
    public static Tensor<T> ColumnEntropyBits<T>(Tensor<T> weights, double minTotal, double minProbability)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        var total = engine.ReduceSum(weights, new[] { 0 }, true);
        var probability = engine.TensorDivide(weights, total);
        var counted = engine.TensorGreaterThan(probability, PreprocessingTensorOps.Tolerance<T>(minProbability));
        var zeros = PreprocessingTensorOps.Filled(weights.Shape.ToArray(), ops.Zero);
        var terms = engine.TensorWhere(counted, engine.TensorMultiply(probability, engine.TensorLog2(probability)), zeros);
        var entropy = engine.TensorNegate(engine.ReduceSum(terms, new[] { 0 }, false));
        var valid = engine.TensorGreaterThan(engine.Reshape(total, new[] { total.Length }), PreprocessingTensorOps.Tolerance<T>(minTotal));
        return engine.TensorWhere(valid, entropy, PreprocessingTensorOps.Filled(new[] { total.Length }, ops.Zero));
    }

    /// <summary>
    /// Squared DFT magnitude <c>|sum_i x_i e^(-2 pi f i)|^2</c> of every column of an [n, m] tensor at each
    /// frequency of <paramref name="frequencies"/> (cycles per sample, shape [f]), as [f, m].
    /// </summary>
    public static Tensor<T> FrequencyPower<T>(Tensor<T> x, Tensor<T> frequencies)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = x.Shape[0];
        var angle = engine.TensorMultiplyScalar(
            engine.TensorOuter(frequencies, engine.TensorArange<T>(0, n, 1)), ops.FromDouble(2 * Math.PI));
        var re = engine.TensorMatMul(engine.TensorCos(angle), x);
        var im = engine.TensorMatMul(engine.TensorSin(angle), x);
        return engine.TensorAdd(engine.TensorMultiply(re, re), engine.TensorMultiply(im, im));
    }

    /// <summary>Column sums of squares of an [n, m] tensor, as [m].</summary>
    public static Tensor<T> ColumnSumOfSquares<T>(Tensor<T> x)
    {
        var engine = AiDotNetEngine.Current;
        return engine.ReduceSum(engine.TensorMultiply(x, x), new[] { 0 }, false);
    }

    /// <summary>Column sample variances (divisor n - 1) of an [n, m] tensor, as [m].</summary>
    public static Tensor<T> ColumnSampleVariance<T>(Tensor<T> x)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        return AiDotNetEngine.Current.TensorDivideScalar(
            ColumnSumOfSquares(FilterStatisticTensorOps.CenterColumns(x)), ops.FromDouble(x.Shape[0] - 1));
    }

    /// <summary>Column population variances (divisor n) of an [n, m] tensor, as [m].</summary>
    public static Tensor<T> ColumnVariance<T>(Tensor<T> x)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        return AiDotNetEngine.Current.TensorDivideScalar(
            ColumnSumOfSquares(FilterStatisticTensorOps.CenterColumns(x)), ops.FromDouble(x.Shape[0]));
    }

    /// <summary>
    /// Lag products <c>sum_{i &lt; n - lag} x[i, j] * x[i + lag, j]</c> of every column of an [n, m] tensor, as [m].
    /// </summary>
    public static Tensor<T> LagProducts<T>(Tensor<T> x, int lag)
    {
        var engine = AiDotNetEngine.Current;
        int n = x.Shape[0];
        int m = x.Shape[1];
        var head = engine.TensorSlice(x, new[] { 0, 0 }, new[] { n - lag, m });
        var tail = engine.TensorSlice(x, new[] { lag, 0 }, new[] { n - lag, m });
        return engine.ReduceSum(engine.TensorMultiply(head, tail), new[] { 0 }, false);
    }

    /// <summary>Entries of <paramref name="values"/> where <paramref name="condition"/> is non-zero, 0 elsewhere.</summary>
    public static Tensor<T> ZeroUnless<T>(Tensor<T> condition, Tensor<T> values)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        return AiDotNetEngine.Current.TensorWhere(condition, values,
            PreprocessingTensorOps.Filled(values.Shape.ToArray(), ops.Zero));
    }

    /// <summary>A [p, p] 0/1 mask of the strictly upper-triangular entries (column &gt; row).</summary>
    public static Tensor<T> StrictUpperMask<T>(int p)
    {
        var engine = AiDotNetEngine.Current;
        var index = engine.TensorArange<T>(0, p, 1);
        var rows = engine.TensorBroadcastTo(engine.Reshape(index, new[] { p, 1 }), new[] { p, p });
        var columns = engine.TensorBroadcastTo(engine.Reshape(index, new[] { 1, p }), new[] { p, p });
        return engine.TensorGreaterThan(columns, rows);
    }

    /// <summary>
    /// Pearson correlation matrix of the columns of an [n, p] tensor using sample covariances and sample
    /// standard deviations offset by <paramref name="stdEpsilon"/>, with the diagonal set to 1. Returns [p, p].
    /// </summary>
    public static Tensor<T> SampleCorrelationMatrix<T>(Tensor<T> x, double stdEpsilon)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = x.Shape[0];
        int p = x.Shape[1];
        var centered = FilterStatisticTensorOps.CenterColumns(x);
        var covariance = engine.TensorDivideScalar(
            engine.TensorMatMul(engine.TensorTranspose(centered), centered), ops.FromDouble(n - 1));
        var std = engine.TensorAddScalar(
            engine.TensorSqrt(engine.TensorDivideScalar(ColumnSumOfSquares(centered), ops.FromDouble(n - 1))),
            PreprocessingTensorOps.Tolerance<T>(stdEpsilon));
        var correlation = engine.TensorDivide(covariance, engine.TensorOuter(std, std));
        return engine.TensorWhere(engine.TensorEye<T>(p), PreprocessingTensorOps.Filled(new[] { p, p }, ops.One), correlation);
    }

    /// <summary>
    /// Greedy redundancy pruning: walking the features in order, each kept feature removes every later feature
    /// whose entry in the [p, p] 0/1 <paramref name="redundant"/> matrix is set. Returns the kept indices, ascending.
    /// </summary>
    public static int[] KeepFirstOfRedundantPairs<T>(Tensor<T> redundant)
    {
        int p = redundant.Shape[0];
        var pairs = PreprocessingTensorOps.MaskPositions(
            AiDotNetEngine.Current.TensorMultiply(redundant, StrictUpperMask<T>(p)));
        var removed = new bool[p];
        foreach (int position in pairs)
        {
            int first = position / p;
            int second = position % p;
            if (!removed[first])
                removed[second] = true;
        }

        return Enumerable.Range(0, p).Where(j => !removed[j]).ToArray();
    }

    /// <summary>
    /// One Haar analysis step on every column of an [len, m] tensor (len &gt;= 2; a trailing odd row is dropped):
    /// approximation <c>(x[2i] + x[2i+1]) / sqrt(2)</c> and detail <c>(x[2i] - x[2i+1]) / sqrt(2)</c>, each [len/2, m].
    /// </summary>
    public static (Tensor<T> Approximation, Tensor<T> Detail) HaarStep<T>(Tensor<T> x)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int half = x.Shape[0] / 2;
        int m = x.Shape[1];
        var pairs = engine.Reshape(engine.TensorSlice(x, new[] { 0, 0 }, new[] { 2 * half, m }), new[] { half, 2, m });
        var even = engine.TensorSliceAxis(pairs, 1, 0);
        var odd = engine.TensorSliceAxis(pairs, 1, 1);
        T scale = ops.FromDouble(Math.Sqrt(2));
        return (engine.TensorDivideScalar(engine.TensorAdd(even, odd), scale),
            engine.TensorDivideScalar(engine.TensorSubtract(even, odd), scale));
    }

    /// <summary>
    /// Columns of an [n, m] tensor centred and divided by their standard deviation (population, or sample when
    /// <paramref name="sample"/> is set); a deviation below <paramref name="minStd"/> is replaced by 1.
    /// </summary>
    public static Tensor<T> StandardizeColumns<T>(Tensor<T> x, bool sample = false, double minStd = 1e-10)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = x.Shape[0];
        int m = x.Shape[1];
        var centered = FilterStatisticTensorOps.CenterColumns(x);
        var std = engine.TensorSqrt(engine.TensorDivideScalar(
            ColumnSumOfSquares(centered), ops.FromDouble(sample ? n - 1 : n)));
        std = engine.TensorWhere(engine.TensorLessThan(std, PreprocessingTensorOps.Tolerance<T>(minStd)),
            PreprocessingTensorOps.Filled(new[] { m }, ops.One), std);
        return engine.TensorDivide(centered, engine.Reshape(std, new[] { 1, m }));
    }

    /// <summary>A vector centred on its mean, as an [n, 1] column.</summary>
    public static Tensor<T> CenteredColumn<T>(Vector<T> y)
    {
        var engine = AiDotNetEngine.Current;
        var yt = Tensor<T>.FromVector(y);
        return engine.Reshape(engine.TensorSubtractScalar(yt, engine.TensorMean(yt)), new[] { y.Length, 1 });
    }

    /// <summary>Ridge coefficients <c>(X'X + alpha I)^-1 X'y</c> for an [n, p] X and an [n, 1] y, as [p].</summary>
    public static Tensor<T> RidgeCoefficients<T>(Tensor<T> x, Tensor<T> y, T alpha)
    {
        var engine = AiDotNetEngine.Current;
        int p = x.Shape[1];
        var xt = engine.TensorTranspose(x);
        var gram = engine.TensorAdd(engine.TensorMatMul(xt, x), engine.TensorMultiplyScalar(engine.TensorEye<T>(p), alpha));
        return engine.Reshape(LinalgDispatch<T>.Solve(gram, engine.TensorMatMul(xt, y)), new[] { p });
    }

    /// <summary>
    /// Average ranks (1-based; tied values share the mean of their positions, scipy <c>rankdata(method="average")</c>)
    /// of every column of an [n, m] tensor, as [n, m].
    /// </summary>
    public static Tensor<T> AverageRanks<T>(Tensor<T> x)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = x.Shape[0];
        int m = x.Shape[1];
        var columns = new Tensor<T>[m];
        for (int j = 0; j < m; j++)
        {
            var column = engine.Reshape(engine.TensorSliceAxis(x, 1, j), new[] { n });
            var sorted = engine.TensorSort(column, 0, false).Values;
            var below = engine.TensorSearchSorted(sorted, column, false);
            var atMost = engine.TensorSearchSorted(sorted, column, true);
            columns[j] = engine.Reshape(engine.TensorAdd(below, atMost).Cast<T>(), new[] { n, 1 });
        }

        // (below + atMost + 1) / 2 = mean of the 1-based positions below + 1 .. atMost
        var sum = m == 0 ? new Tensor<T>(new[] { n, 0 }) : engine.TensorConcatenate(columns, 1);
        return engine.TensorDivideScalar(engine.TensorAddScalar(sum, ops.One), ops.FromDouble(2));
    }

    /// <summary>Pearson correlation of every column of an [n, m] tensor with an [n, 1] column, 0 where either
    /// sum of squares is not above <paramref name="epsilon"/>. Returns [m].</summary>
    public static Tensor<T> PearsonWithColumn<T>(Tensor<T> x, Tensor<T> y, double epsilon = 1e-10)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int m = x.Shape[1];
        var xc = FilterStatisticTensorOps.CenterColumns(x);
        var yc = FilterStatisticTensorOps.CenterColumns(y);
        var sxx = ColumnSumOfSquares(xc);
        T syy = engine.TensorSum(engine.TensorMultiply(yc, yc));
        // r from max-magnitude-scaled columns: sxx and sxx syy overflow Half (the ranks of 100 samples suffice)
        var r = PreprocessingTensorOps.CenteredCorrelations(xc, yc);
        var valid = engine.TensorMultiply(
            engine.TensorGreaterThan(sxx, PreprocessingTensorOps.Tolerance<T>(epsilon)),
            PreprocessingTensorOps.Filled(new[] { m }, ops.GreaterThan(syy, PreprocessingTensorOps.Tolerance<T>(epsilon)) ? ops.One : ops.Zero));
        return ZeroUnless(valid, r);
    }

    /// <summary>
    /// Robust z-scores of every column of an [n, m] tensor: <c>(x - median) / (1.4826 * MAD + epsilon)</c>.
    /// </summary>
    public static Tensor<T> RobustZScores<T>(Tensor<T> x, double epsilon = 1e-10)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        var deviation = engine.TensorSubtract(x, ColumnMedian(x, keepDims: true));
        var scale = engine.TensorAddScalar(
            engine.TensorMultiplyScalar(ColumnMedian(engine.TensorAbs(deviation), keepDims: true), ops.FromDouble(1.4826)),
            PreprocessingTensorOps.Tolerance<T>(epsilon));
        return engine.TensorDivide(deviation, scale);
    }

    /// <summary>Huber weights <c>min(1, delta / |z|)</c> of every element.</summary>
    public static Tensor<T> HuberWeights<T>(Tensor<T> z, T delta)
    {
        var engine = AiDotNetEngine.Current;
        return engine.TensorDivide(PreprocessingTensorOps.Filled(z.Shape.ToArray(), delta), engine.TensorMax(engine.TensorAbs(z), delta));
    }

    /// <summary>
    /// Every column of an [n, m] tensor clipped to its order statistics at <c>(int)(n * lower)</c> and
    /// <c>min((int)(n * upper), n - 1)</c> (0-based positions in the sorted column).
    /// </summary>
    public static Tensor<T> WinsorizeColumns<T>(Tensor<T> x, double lower, double upper)
    {
        var engine = AiDotNetEngine.Current;
        int n = x.Shape[0];
        int m = x.Shape[1];
        int lowerIndex = (int)(n * lower);
        int upperIndex = Math.Min((int)(n * upper), n - 1);
        var shape = new[] { n, m };
        var low = engine.TensorBroadcastTo(
            engine.Reshape(PreprocessingTensorOps.ColumnOrderStatistic(x, lowerIndex), new[] { 1, m }), shape);
        var high = engine.TensorBroadcastTo(
            engine.Reshape(PreprocessingTensorOps.ColumnOrderStatistic(x, upperIndex), new[] { 1, m }), shape);
        return engine.TensorMin(engine.TensorMax(x, low), high);
    }

    /// <summary>
    /// Pearson correlation of every column of an [n, m] tensor with an [n, 1] column over the rows where the [n, m]
    /// 0/1 <paramref name="mask"/> is set (each column its own rows), 0 where <c>sqrt(sxx * syy)</c> is not above
    /// <paramref name="epsilon"/> or fewer than <paramref name="minRows"/> rows are kept. Returns [m].
    /// </summary>
    public static Tensor<T> MaskedPearson<T>(Tensor<T> x, Tensor<T> y, Tensor<T> mask, int minRows, double epsilon = 1e-10)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = x.Shape[0];
        int m = x.Shape[1];
        var yFull = engine.TensorBroadcastTo(y, new[] { n, m });
        // Row counts in int: 0/1 sums in a narrow T (Half) stop growing at 2048
        var count = engine.ReduceSum(mask.Cast<int>(), new[] { 0 }, true).Cast<T>();
        var xMean = engine.TensorDivide(engine.ReduceSum(engine.TensorMultiply(mask, x), new[] { 0 }, true), count);
        var yMean = engine.TensorDivide(engine.ReduceSum(engine.TensorMultiply(mask, yFull), new[] { 0 }, true), count);
        var xd = engine.TensorMultiply(mask, engine.TensorSubtract(x, xMean));
        var yd = engine.TensorMultiply(mask, engine.TensorSubtract(yFull, yMean));
        var sxy = engine.ReduceSum(engine.TensorMultiply(xd, yd), new[] { 0 }, false);
        var denominator = engine.TensorSqrt(engine.TensorMultiply(ColumnSumOfSquares(xd), ColumnSumOfSquares(yd)));
        var valid = engine.TensorMultiply(
            engine.TensorGreaterThan(denominator, PreprocessingTensorOps.Tolerance<T>(epsilon)),
            AtLeastCount(engine.Reshape(count, new[] { m }), minRows));
        return ZeroUnless(valid, engine.TensorDivide(sxy, denominator));
    }

    /// <summary>0/1 mask of the entries that are at least <paramref name="minimum"/>.</summary>
    public static Tensor<T> AtLeastCount<T>(Tensor<T> counts, int minimum) =>
        PreprocessingTensorOps.AtLeast(counts, MathHelper.GetNumericOperations<T>().FromDouble(minimum));

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
    /// Per-group column means: <paramref name="membership"/> is a 0/1 [n, k] one-hot of each row's group and
    /// <paramref name="x"/> is [n, p]. Returns [k, p]; an empty group's mean is 0. <paramref name="counts"/> gets
    /// the [k] group sizes.
    /// </summary>
    public static Tensor<T> GroupMeans<T>(Tensor<T> membership, Tensor<T> x, out Tensor<T> counts)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int k = membership.Shape[1];
        // Group sizes in int: 0/1 sums in a narrow T (Half) stop growing at 2048
        counts = engine.ReduceSum(membership.Cast<int>(), new[] { 0 }, false).Cast<T>();
        var sums = engine.TensorMatMul(engine.TensorTranspose(membership), x);
        var divisor = engine.Reshape(engine.TensorMax(counts, ops.One), new[] { k, 1 });
        return engine.TensorDivide(sums, engine.TensorBroadcastTo(divisor, sums.Shape.ToArray()));
    }

    /// <summary>
    /// Kendall rank correlation of every column of <paramref name="x"/> ([n, p]) with <paramref name="y"/> ([n]).
    /// A pair counts as tied in a variable when its difference is below 1e-10 in magnitude. Tau-b is
    /// <c>(C - D) / sqrt((n0 - n1)(n0 - n2))</c> with n1 / n2 the pairs tied in x / y (joint ties included in both,
    /// as in scipy.stats.kendalltau), 0 when the denominator is 0; tau-a is <c>(C - D) / n0</c>. Returns [p].
    /// </summary>
    public static Tensor<T> KendallTau<T>(Tensor<T> x, Tensor<T> y, bool tauB)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = x.Shape[0];
        int p = x.Shape[1];
        var zeros = new Tensor<T>(new[] { p });
        if (p == 0)
            return zeros;

        // Pair counts are integers (summed in int: 0/+-1 sums in a narrow T such as Half stop growing at 2048) and
        // tau is their ratio, formed in double and converted to T once: the counts themselves reach n (n - 1), past
        // Half's 65504 from n = 257 (and 2 (C - D) past it from n = 182), so in Half they would overflow to infinity.
        T tolerance = PreprocessingTensorOps.Tolerance<T>(1e-10);
        var ySigns = PairSigns(y, tolerance);
        var ySignsInt = ySigns.Cast<int>();
        var signSums = new int[p];
        var untied = new int[p];
        for (int j = 0; j < p; j++)
        {
            var xSigns = PairSigns(PreprocessingTensorOps.ColumnVector(x, j), tolerance);
            signSums[j] = engine.TensorSum(engine.TensorMultiply(xSigns.Cast<int>(), ySignsInt));
            untied[j] = engine.TensorCountNonzero(xSigns);
        }

        // Over ordered pairs: sum(sx * sy) = 2 (C - D) and sum|sx| = 2 (n0 - n1)
        var differences = new Tensor<int>(signSums, new[] { p }).Cast<double>();
        if (tauB)
        {
            // sqrt(a) * sqrt(b) rather than sqrt(a * b): the product of two pair counts overflows int
            double yUntied = Math.Sqrt(engine.TensorCountNonzero(ySigns));
            var denominators = engine.TensorMultiplyScalar(
                engine.TensorSqrt(new Tensor<int>(untied, new[] { p }).Cast<double>()), yUntied);
            var usable = engine.TensorGreaterThan(denominators, 0.0);
            return engine.TensorWhere(usable, engine.TensorDivide(differences,
                engine.TensorWhere(usable, denominators, PreprocessingTensorOps.Filled(new[] { p }, 1.0))),
                new Tensor<double>(new[] { p })).Cast<T>();
        }

        long pairs = (long)n * (n - 1) / 2;
        return pairs > 0 ? engine.TensorDivideScalar(differences, 2.0 * pairs).Cast<T>() : zeros;
    }

    /// <summary>
    /// Signs of all ordered pairwise differences <c>v_i - v_k</c> of a 1-D tensor, flattened to [s * s];
    /// differences below <paramref name="tolerance"/> in magnitude count as ties (0).
    /// </summary>
    public static Tensor<T> PairSigns<T>(Tensor<T> values, T tolerance)
    {
        var engine = AiDotNetEngine.Current;
        int s = values.Length;
        var shape = new[] { s, s };
        var differences = engine.Reshape(engine.TensorSubtract(
            engine.TensorBroadcastTo(engine.Reshape(values, new[] { s, 1 }), shape),
            engine.TensorBroadcastTo(engine.Reshape(values, new[] { 1, s }), shape)), new[] { s * s });
        return engine.TensorWhere(engine.TensorLessThan(engine.TensorAbs(differences), tolerance),
            new Tensor<T>(new[] { s * s }), engine.TensorSign(differences));
    }

    /// <summary>
    /// Class scatter of every column of <paramref name="x"/> ([n, p]) for rows grouped by
    /// <paramref name="classIndex"/> (positions in 0..<paramref name="classCount"/>-1): returns the between-class sum
    /// of squares <c>sum_c n_c (mean_c - mean)^2</c> and, in <paramref name="within"/>, the within-class sum of
    /// squares <c>sum_i (x_i - mean_class(i))^2</c>. Both [p].
    /// </summary>
    public static Tensor<T> ClassScatter<T>(Tensor<T> x, Tensor<int> classIndex, int classCount, out Tensor<T> within)
    {
        var engine = AiDotNetEngine.Current;
        int p = x.Shape[1];
        var shape = new[] { classCount, p };
        var membership = engine.TensorOneHot<T>(classIndex, classCount);
        var classMeans = GroupMeans(membership, x, out var counts);
        var overallMean = engine.TensorBroadcastTo(engine.ReduceMean(x, new[] { 0 }, true), shape);
        within = engine.ReduceSum(engine.TensorSquare(engine.TensorSubtract(x,
            engine.TensorIndexSelect(classMeans, classIndex, 0))), new[] { 0 }, false);
        return engine.ReduceSum(engine.TensorMultiply(engine.TensorSquare(engine.TensorSubtract(classMeans, overallMean)),
            engine.TensorBroadcastTo(engine.Reshape(counts, new[] { classCount, 1 }), shape)), new[] { 0 }, false);
    }

    /// <summary>
    /// ANOVA F-value of every column of <paramref name="x"/> ([n, p]) against the integer classes of
    /// <paramref name="target"/> (scikit-learn <c>f_classif</c>): <c>F = (SSB / (k - 1)) / (SSW / (n - k))</c>.
    /// A feature with no within-class scatter whose class means differ separates the classes perfectly and scores
    /// +infinity (p-value 0); a constant feature scores 0 (p-value 1). Degenerate degrees of freedom give 0 / 1.
    /// </summary>
    public static Tensor<T> FClassif<T>(Tensor<T> x, Vector<T> target, out Tensor<T> pValues)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = x.Shape[0];
        int p = x.Shape[1];
        PreprocessingTensorOps.ClassLabels(target, out var classes, out var classIndex);
        int k = classes.Length;
        int dfBetween = k - 1;
        int dfWithin = n - k;
        if (dfBetween <= 0 || dfWithin <= 0)
        {
            pValues = PreprocessingTensorOps.Filled(new[] { p }, ops.One);
            return PreprocessingTensorOps.Filled(new[] { p }, ops.Zero);
        }

        var between = ClassScatter(x, new Tensor<int>(classIndex, new[] { n }), k, out var within);
        var msb = engine.TensorDivideScalar(between, ops.FromDouble(dfBetween));
        var msw = engine.TensorDivideScalar(within, ops.FromDouble(dfWithin));
        var scores = SeparationRatio(msb, msw, 1e-10);
        pValues = FilterStatisticTensorOps.FSurvival(scores, dfBetween, dfWithin);
        return scores;
    }

    /// <summary>
    /// F-value of the univariate least-squares regression of the target on every column (scikit-learn
    /// <c>f_regression</c>): <c>F = SS_regression / (SS_residual / (n - 2))</c>, +infinity for a perfect fit,
    /// 0 for a constant feature or target; p-values from F(1, n - 2).
    /// </summary>
    public static Tensor<T> FRegression<T>(Tensor<T> x, Vector<T> target, out Tensor<T> pValues)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = x.Shape[0];
        int p = x.Shape[1];
        int dfResidual = n - 2;
        if (dfResidual <= 0)
        {
            pValues = PreprocessingTensorOps.Filled(new[] { p }, ops.One);
            return PreprocessingTensorOps.Filled(new[] { p }, ops.Zero);
        }

        var xc = FilterStatisticTensorOps.CenterColumns(x);
        var yc = CenteredColumn(target);
        var sxy = engine.ReduceSum(engine.TensorMultiply(xc, yc), new[] { 0 }, false);
        var sxx = ColumnSumOfSquares(xc);
        T ssTotal = engine.TensorSum(engine.TensorMultiply(yc, yc));

        // Slope (0 for a near-constant feature) and the residual sum of squares of y - beta x
        var beta = ZeroUnless(engine.TensorGreaterThan(sxx, PreprocessingTensorOps.Tolerance<T>(1e-10)), engine.TensorDivide(sxy, sxx));
        var residual = engine.TensorSubtract(yc, engine.TensorMultiply(xc, engine.Reshape(beta, new[] { 1, p })));
        var ssResidual = ColumnSumOfSquares(residual);
        var ssRegression = engine.TensorMax(engine.TensorSubtract(
            PreprocessingTensorOps.Filled(new[] { p }, ssTotal), ssResidual), ops.Zero);
        var msResidual = engine.TensorDivideScalar(ssResidual, ops.FromDouble(dfResidual));

        var scores = SeparationRatio(ssRegression, msResidual, 1e-10);
        pValues = FilterStatisticTensorOps.FSurvival(scores, 1, dfResidual);
        return scores;
    }

    /// <summary>
    /// <c>numerator / denominator</c> where the denominator is above <paramref name="epsilon"/>; otherwise +infinity
    /// when the numerator is above <paramref name="epsilon"/> (a perfect separation or fit) and 0 when it is not.
    /// </summary>
    public static Tensor<T> SeparationRatio<T>(Tensor<T> numerator, Tensor<T> denominator, double epsilon)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        var shape = numerator.Shape.ToArray();
        T eps = PreprocessingTensorOps.Tolerance<T>(epsilon);
        var usable = engine.TensorGreaterThan(denominator, eps);
        var safeDenominator = engine.TensorWhere(usable, denominator, PreprocessingTensorOps.Filled(shape, ops.One));
        var degenerate = engine.TensorWhere(engine.TensorGreaterThan(numerator, eps),
            PreprocessingTensorOps.Filled(shape, ops.FromDouble(double.PositiveInfinity)),
            PreprocessingTensorOps.Filled(shape, ops.Zero));
        return engine.TensorWhere(usable, engine.TensorDivide(numerator, safeDenominator), degenerate);
    }

    /// <summary>
    /// Chi-squared statistic of every non-negative column of <paramref name="x"/> against the integer classes of
    /// <paramref name="target"/> (scikit-learn <c>chi2</c>): observed class sums against
    /// <c>expected = total * classCount / n</c>, p-values from chi2(k - 1).
    /// </summary>
    /// <exception cref="ArgumentException">Any feature value is negative.</exception>
    public static Tensor<T> Chi2<T>(Tensor<T> x, Vector<T> target, out Tensor<T> pValues)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = x.Shape[0];
        int p = x.Shape[1];
        if (x.Length > 0 && PreprocessingTensorOps.IsBelow(engine.TensorMinValue(x), ops.Zero))
            throw new ArgumentException("Chi-squared scoring requires non-negative feature values (counts or frequencies).");

        PreprocessingTensorOps.ClassLabels(target, out var classes, out var classIndex);
        int k = classes.Length;
        var membership = PreprocessingTensorOps.ClassOneHot<T>(classIndex, k);
        var observed = engine.TensorMatMul(engine.TensorTranspose(membership), x);
        // Class sizes counted in int: 0/1 sums in a narrow T (Half) stop growing at 2048
        var classSizes = PreprocessingTensorOps.CategoryCounts<T>(
            new Tensor<int>((int[])classIndex.Clone(), new[] { classIndex.Length }), k);
        var classShare = engine.TensorDivideScalar(classSizes, ops.FromDouble(n));
        var expected = engine.TensorOuter(classShare, engine.ReduceSum(x, new[] { 0 }, false));
        var difference = engine.TensorSubtract(observed, expected);
        var counted = engine.TensorGreaterThan(expected, PreprocessingTensorOps.Tolerance<T>(1e-10));
        var safeExpected = engine.TensorWhere(counted, expected, PreprocessingTensorOps.Filled(new[] { k, p }, ops.One));
        var terms = ZeroUnless(counted, engine.TensorDivide(engine.TensorMultiply(difference, difference), safeExpected));
        var scores = engine.ReduceSum(terms, new[] { 0 }, false);
        pValues = ChiSquaredSurvival(scores, PreprocessingTensorOps.Filled(new[] { p }, ops.FromDouble(k - 1)));
        return scores;
    }

    /// <summary>
    /// Mutual information (nats) between every continuous column of <paramref name="x"/> and the integer classes of
    /// <paramref name="target"/> by the nearest-neighbour estimator of Ross (2014), as scikit-learn
    /// <c>mutual_info_classif</c> with <c>n_neighbors = 3</c>:
    /// <c>psi(N) + mean psi(k_i) - mean psi(N_c(i)) - mean psi(m_i)</c>, clamped at 0, where k_i = min(3, N_c - 1),
    /// d_i is the distance to the k_i-th nearest same-class sample and m_i counts all samples closer than d_i
    /// (itself included). Samples whose class has a single member are ignored. The 1e-10 tie-breaking noise of
    /// scikit-learn is not added, so tie-free data reproduces it exactly.
    /// </summary>
    /// <remarks>
    /// One-dimensional neighbour search on sorted columns: the k_i nearest same-class samples lie within k_i places
    /// of each sample in its class's sorted order, and the samples closer than d_i form a contiguous run of the
    /// sorted column found by binary search, so memory is O(n p) and time O(n log n) per column. Class sizes and
    /// neighbour counts are integers; only the final digamma terms are in <typeparamref name="T"/>.
    /// </remarks>
    public static Tensor<T> MutualInfoClassif<T>(Tensor<T> x, Vector<T> target)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int p = x.Shape[1];
        var zeros = PreprocessingTensorOps.Filled(new[] { p }, ops.Zero);
        PreprocessingTensorOps.ClassLabels(target, out var classes, out var classIndex);
        int k = classes.Length;
        if (p == 0 || classIndex.Length == 0)
            return zeros;

        // Keep the samples whose class has more than one member (class sizes counted in int)
        var allClasses = new Tensor<int>((int[])classIndex.Clone(), new[] { classIndex.Length });
        var classSizes = engine.TensorBinCount(allClasses, k);
        var rowClassCount = engine.TensorIndexSelect(classSizes, allClasses, 0);
        var kept = PreprocessingTensorOps.MaskPositions(engine.TensorGreaterThan(rowClassCount, 1));
        int m = kept.Length;
        if (m == 0)
            return zeros;

        var keptIndex = new Tensor<int>(kept, new[] { m });
        var labelCounts = engine.TensorIndexSelect(rowClassCount, keptIndex, 0);
        var neighbours = engine.TensorMin(engine.TensorSubtractScalar(labelCounts, 1), 3);
        T constant = ops.Subtract(
            ops.Add(engine.TensorSum(engine.TensorDigamma(PreprocessingTensorOps.Filled(new[] { 1 }, ops.FromDouble(m)))),
                engine.TensorMean(engine.TensorDigamma(neighbours.Cast<T>()))),
            engine.TensorMean(engine.TensorDigamma(labelCounts.Cast<T>())));
        var constantTensor = PreprocessingTensorOps.Filled(new[] { 1 }, constant);

        // Group the kept rows by class; the order inside a class is irrelevant because each block is sorted next
        var keptRows = engine.TensorIndexSelect(x, keptIndex, 0);
        var byClass = engine.TensorIndexSelect(
            keptRows, engine.TensorArgsort(engine.TensorIndexSelect(allClasses, keptIndex, 0), 0, false), 0);
        int[] sizes = classSizes.ToArray();
        var classValues = new List<Tensor<T>>();
        var classRadii = new List<Tensor<T>>();
        int offset = 0;
        for (int c = 0; c < k; c++)
        {
            if (sizes[c] < 2)
                continue;
            var sortedClass = engine.TensorSort(engine.TensorNarrow(byClass, 0, offset, sizes[c]), 0, false).Values;
            offset += sizes[c];
            classValues.Add(sortedClass);
            classRadii.Add(KthNeighbourDistance(sortedClass, Math.Min(3, sizes[c] - 1)));
        }

        // Every sample's own value and radius, in class-sorted order (the estimate is a mean over samples)
        var centres = engine.TensorConcatenate(classValues.ToArray(), 0);
        var radii = engine.TensorConcatenate(classRadii.ToArray(), 0);
        var positive = engine.TensorGreaterThan(radii, ops.Zero);
        var sortedColumns = engine.TensorSort(keptRows, 0, false).Values;
        var results = new Tensor<T>[p];
        for (int j = 0; j < p; j++)
        {
            var counts = CountWithinRadius(engine.TensorSliceAxis(sortedColumns, 1, j),
                engine.TensorSliceAxis(centres, 1, j), engine.TensorSliceAxis(radii, 1, j),
                engine.TensorSliceAxis(positive, 1, j));
            results[j] = engine.TensorSubtract(constantTensor,
                engine.Reshape(engine.ReduceMean(engine.TensorDigamma(counts.Cast<T>()), new[] { 0 }, false), new[] { 1 }));
        }

        return engine.TensorMax(engine.TensorConcatenate(results, 0), ops.Zero);
    }

    /// <summary>
    /// For an [s, p] tensor sorted along axis 0, the distance from every entry to its <paramref name="k"/>-th
    /// nearest other entry of the same column (1 &lt;= k &lt; s): the k-th smallest of the gaps to the entries up to
    /// k places before and after it, which in one dimension hold its k nearest neighbours. Returns [s, p].
    /// </summary>
    private static Tensor<T> KthNeighbourDistance<T>(Tensor<T> sorted, int k)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int s = sorted.Shape[0];
        int p = sorted.Shape[1];
        var candidates = new Tensor<T>[2 * k];
        T infinity = ops.FromDouble(double.PositiveInfinity);
        for (int step = 1; step <= k; step++)
        {
            // gap[r] = sorted[r + step] - sorted[r]: the forward gap of row r and the backward gap of row r + step
            var gap = engine.TensorSubtract(
                engine.TensorNarrow(sorted, 0, step, s - step), engine.TensorNarrow(sorted, 0, 0, s - step));
            var missing = PreprocessingTensorOps.Filled(new[] { step, p }, infinity);
            candidates[2 * step - 2] = engine.TensorConcatenate(new[] { gap, missing }, 0);
            candidates[2 * step - 1] = engine.TensorConcatenate(new[] { missing, gap }, 0);
        }

        var nearest = engine.TensorSort(engine.TensorStack(candidates, 2), 2, false).Values;
        return engine.TensorSliceAxis(nearest, 2, k - 1);
    }

    /// <summary>
    /// For every centre (a value of the 1-D ascending <paramref name="sequence"/>), the number of entries of the
    /// sequence at distance below its radius, or at distance 0 where <paramref name="positive"/> is 0. That set is a
    /// contiguous run of the sequence containing the centre: it is located by binary search on
    /// <c>centre -/+ radius</c>, then each end is moved a whole value at a time until the entry just inside it passes
    /// the distance test and the entry just beyond it fails, so the count matches the distance test exactly even
    /// where <c>centre + radius</c> rounds.
    /// </summary>
    private static Tensor<int> CountWithinRadius<T>(Tensor<T> sequence, Tensor<T> centre, Tensor<T> radius, Tensor<T> positive)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = sequence.Length;
        var origin = PreprocessingTensorOps.Filled(new[] { centre.Length }, ops.Zero);

        // 1 where the sequence entry at each position passes the distance test of its centre
        Tensor<int> Inside(Tensor<int> position)
        {
            var value = engine.TensorIndexSelect(sequence, engine.TensorClamp(position, 0, n - 1), 0);
            var distance = engine.TensorAbs(engine.TensorSubtract(value, centre));
            return engine.TensorWhere(positive, engine.TensorLessThan(distance, radius),
                engine.TensorLessEqual(distance, origin)).Cast<int>();
        }

        // The first position (right: one past the last) holding the value found at each position
        Tensor<int> ValueEdge(Tensor<int> position, bool right) => engine.TensorSearchSorted(sequence,
            engine.TensorIndexSelect(sequence, engine.TensorClamp(position, 0, n - 1), 0), right);

        // Initial ends, widened to cover every copy of the centre itself
        var upper = engine.TensorMax(engine.TensorSearchSorted(sequence, engine.TensorAdd(centre, radius), false),
            engine.TensorSearchSorted(sequence, centre, true));
        var lower = engine.TensorMin(engine.TensorSearchSorted(sequence, engine.TensorSubtract(centre, radius), true),
            engine.TensorSearchSorted(sequence, centre, false));

        // Every move lands on a value boundary between the current end and the true one, so this terminates
        bool moved = true;
        while (moved)
        {
            var grow = engine.TensorMultiply(engine.TensorLessThan(upper, n), Inside(upper));
            var nextUpper = engine.TensorWhere(grow, ValueEdge(upper, true), upper);
            var last = engine.TensorSubtractScalar(nextUpper, 1);
            var shrink = engine.TensorMultiply(engine.TensorGreaterThan(nextUpper, 0), engine.TensorEquals(Inside(last), 0));
            nextUpper = engine.TensorWhere(shrink, ValueEdge(last, false), nextUpper);

            var below = engine.TensorSubtractScalar(lower, 1);
            grow = engine.TensorMultiply(engine.TensorGreaterThan(lower, 0), Inside(below));
            var nextLower = engine.TensorWhere(grow, ValueEdge(below, false), lower);
            shrink = engine.TensorMultiply(engine.TensorLessThan(nextLower, n), engine.TensorEquals(Inside(nextLower), 0));
            nextLower = engine.TensorWhere(shrink, ValueEdge(nextLower, true), nextLower);

            moved = engine.TensorCountNonzero(engine.TensorSubtract(nextUpper, upper)) > 0
                || engine.TensorCountNonzero(engine.TensorSubtract(nextLower, lower)) > 0;
            upper = nextUpper;
            lower = nextLower;
        }

        return engine.TensorSubtract(upper, lower);
    }

    /// <summary>
    /// R^2 (against the centred total sum of squares; 0 when that is not positive) of the no-intercept ridge
    /// regression of the [n, 1] <paramref name="y"/> on the <paramref name="columns"/> of the [n, p]
    /// <paramref name="x"/>, with <paramref name="ridge"/> added to the Gram diagonal. An empty subset scores 0.
    /// </summary>
    public static T RidgeRSquared<T>(Tensor<T> x, Tensor<T> y, IReadOnlyCollection<int> columns, double ridge)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        if (columns.Count == 0)
            return ops.Zero;

        var indices = columns.ToArray();
        var xs = engine.TensorIndexSelect(x, new Tensor<int>(indices, new[] { indices.Length }), 1);
        var coefficients = RidgeCoefficients(xs, y, ops.FromDouble(ridge));
        var predictions = engine.TensorMatMul(xs, engine.Reshape(coefficients, new[] { indices.Length, 1 }));
        var residual = engine.TensorSubtract(y, predictions);
        T ssRes = engine.TensorSum(engine.TensorMultiply(residual, residual));
        var centered = FilterStatisticTensorOps.CenterColumns(y);
        T ssTot = engine.TensorSum(engine.TensorMultiply(centered, centered));
        return ops.GreaterThan(ssTot, ops.Zero) ? ops.Subtract(ops.One, ops.Divide(ssRes, ssTot)) : ops.Zero;
    }

    /// <summary>A [length] tensor holding <paramref name="values"/> at their keys and 0 elsewhere.</summary>
    public static Tensor<T> FromSparse<T>(IReadOnlyDictionary<int, T> values, int length)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var result = PreprocessingTensorOps.Filled(new[] { length }, ops.Zero);
        if (values.Count == 0)
            return result;

        var keys = values.Keys.ToArray();
        var entries = new Tensor<T>(keys.Select(key => values[key]).ToArray(), new[] { keys.Length });
        return AiDotNetEngine.Current.TensorIndexCopy(result, 0, new Tensor<int>(keys, new[] { keys.Length }), entries);
    }

    /// <summary>The values divided by their maximum when that maximum is positive, unchanged otherwise.</summary>
    public static Tensor<T> NormalizeByPositiveMax<T>(Tensor<T> values)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        if (values.Length == 0)
            return values;

        T max = engine.TensorMaxValue(values);
        return ops.GreaterThan(max, ops.Zero) ? engine.TensorDivideScalar(values, max) : values;
    }

    /// <summary>The scalar difference <c>a - b</c> (for once-per-iteration bookkeeping in wrapper searches).</summary>
    public static T Difference<T>(T a, T b) => MathHelper.GetNumericOperations<T>().Subtract(a, b);
}
