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
        var counted = engine.TensorGreaterThan(probability, ops.FromDouble(minProbability));
        var zeros = PreprocessingTensorOps.Filled(weights.Shape.ToArray(), ops.Zero);
        var terms = engine.TensorWhere(counted, engine.TensorMultiply(probability, engine.TensorLog2(probability)), zeros);
        var entropy = engine.TensorNegate(engine.ReduceSum(terms, new[] { 0 }, false));
        var valid = engine.TensorGreaterThan(engine.Reshape(total, new[] { total.Length }), ops.FromDouble(minTotal));
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

    /// <summary>Column means subtracted from an [n, m] tensor.</summary>
    public static Tensor<T> CenterColumns<T>(Tensor<T> x)
    {
        var engine = AiDotNetEngine.Current;
        return engine.TensorSubtract(x, engine.ReduceMean(x, new[] { 0 }, true));
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
            ColumnSumOfSquares(CenterColumns(x)), ops.FromDouble(x.Shape[0] - 1));
    }

    /// <summary>Column population variances (divisor n) of an [n, m] tensor, as [m].</summary>
    public static Tensor<T> ColumnVariance<T>(Tensor<T> x)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        return AiDotNetEngine.Current.TensorDivideScalar(
            ColumnSumOfSquares(CenterColumns(x)), ops.FromDouble(x.Shape[0]));
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
        var centered = CenterColumns(x);
        var covariance = engine.TensorDivideScalar(
            engine.TensorMatMul(engine.TensorTranspose(centered), centered), ops.FromDouble(n - 1));
        var std = engine.TensorAddScalar(
            engine.TensorSqrt(engine.TensorDivideScalar(ColumnSumOfSquares(centered), ops.FromDouble(n - 1))),
            ops.FromDouble(stdEpsilon));
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
        var centered = CenterColumns(x);
        var std = engine.TensorSqrt(engine.TensorDivideScalar(
            ColumnSumOfSquares(centered), ops.FromDouble(sample ? n - 1 : n)));
        std = engine.TensorWhere(engine.TensorLessThan(std, ops.FromDouble(minStd)),
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
        var xc = CenterColumns(x);
        var yc = CenterColumns(y);
        var sxy = engine.ReduceSum(engine.TensorMultiply(xc, yc), new[] { 0 }, false);
        var sxx = ColumnSumOfSquares(xc);
        T syy = engine.TensorSum(engine.TensorMultiply(yc, yc));
        var r = engine.TensorDivide(sxy, engine.TensorSqrt(engine.TensorMultiplyScalar(sxx, syy)));
        var valid = engine.TensorMultiply(
            engine.TensorGreaterThan(sxx, ops.FromDouble(epsilon)),
            PreprocessingTensorOps.Filled(new[] { m }, ops.GreaterThan(syy, ops.FromDouble(epsilon)) ? ops.One : ops.Zero));
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
            ops.FromDouble(epsilon));
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
        var count = engine.ReduceSum(mask, new[] { 0 }, true);
        var xMean = engine.TensorDivide(engine.ReduceSum(engine.TensorMultiply(mask, x), new[] { 0 }, true), count);
        var yMean = engine.TensorDivide(engine.ReduceSum(engine.TensorMultiply(mask, yFull), new[] { 0 }, true), count);
        var xd = engine.TensorMultiply(mask, engine.TensorSubtract(x, xMean));
        var yd = engine.TensorMultiply(mask, engine.TensorSubtract(yFull, yMean));
        var sxy = engine.ReduceSum(engine.TensorMultiply(xd, yd), new[] { 0 }, false);
        var denominator = engine.TensorSqrt(engine.TensorMultiply(ColumnSumOfSquares(xd), ColumnSumOfSquares(yd)));
        var valid = engine.TensorMultiply(
            engine.TensorGreaterThan(denominator, ops.FromDouble(epsilon)),
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

        T tolerance = ops.FromDouble(1e-10);
        var ySigns = PairSigns(y, tolerance);
        var signSums = new Tensor<T>[p];
        var untied = new Tensor<T>[p];
        for (int j = 0; j < p; j++)
        {
            var xSigns = PairSigns(PreprocessingTensorOps.ColumnVector(x, j), tolerance);
            signSums[j] = engine.ReduceSum(engine.TensorMultiply(xSigns, ySigns), new[] { 0 }, true);
            untied[j] = engine.ReduceSum(engine.TensorAbs(xSigns), new[] { 0 }, true);
        }

        // Over ordered pairs: sum(sx * sy) = 2 (C - D) and sum|sx| = 2 (n0 - n1)
        var differences = engine.TensorConcatenate(signSums, 0);
        if (tauB)
        {
            var denominators = engine.TensorSqrt(engine.TensorMultiplyScalar(
                engine.TensorConcatenate(untied, 0), engine.TensorSum(engine.TensorAbs(ySigns))));
            var usable = engine.TensorGreaterThan(denominators, ops.Zero);
            return engine.TensorWhere(usable, engine.TensorDivide(differences,
                engine.TensorWhere(usable, denominators, PreprocessingTensorOps.Filled(new[] { p }, ops.One))), zeros);
        }

        long pairs = (long)n * (n - 1) / 2;
        return pairs > 0 ? engine.TensorDivideScalar(differences, ops.FromDouble(2.0 * pairs)) : zeros;
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
        pValues = FSurvival(scores, dfBetween, dfWithin);
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

        var xc = CenterColumns(x);
        var yc = CenteredColumn(target);
        var sxy = engine.ReduceSum(engine.TensorMultiply(xc, yc), new[] { 0 }, false);
        var sxx = ColumnSumOfSquares(xc);
        T ssTotal = engine.TensorSum(engine.TensorMultiply(yc, yc));

        // Slope (0 for a near-constant feature) and the residual sum of squares of y - beta x
        var beta = ZeroUnless(engine.TensorGreaterThan(sxx, ops.FromDouble(1e-10)), engine.TensorDivide(sxy, sxx));
        var residual = engine.TensorSubtract(yc, engine.TensorMultiply(xc, engine.Reshape(beta, new[] { 1, p })));
        var ssResidual = ColumnSumOfSquares(residual);
        var ssRegression = engine.TensorMax(engine.TensorSubtract(
            PreprocessingTensorOps.Filled(new[] { p }, ssTotal), ssResidual), ops.Zero);
        var msResidual = engine.TensorDivideScalar(ssResidual, ops.FromDouble(dfResidual));

        var scores = SeparationRatio(ssRegression, msResidual, 1e-10);
        pValues = FSurvival(scores, 1, dfResidual);
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
        T eps = ops.FromDouble(epsilon);
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
        var classShare = engine.TensorDivideScalar(engine.ReduceSum(membership, new[] { 0 }, false), ops.FromDouble(n));
        var expected = engine.TensorOuter(classShare, engine.ReduceSum(x, new[] { 0 }, false));
        var difference = engine.TensorSubtract(observed, expected);
        var counted = engine.TensorGreaterThan(expected, ops.FromDouble(1e-10));
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

        // Keep the samples whose class has more than one member
        var classCounts = engine.ReduceSum(PreprocessingTensorOps.ClassOneHot<T>(classIndex, k), new[] { 0 }, false);
        var allClasses = new Tensor<int>(classIndex, new[] { classIndex.Length });
        var rowClassCount = engine.TensorIndexSelect(classCounts, allClasses, 0);
        var kept = PreprocessingTensorOps.MaskPositions(engine.TensorGreaterThan(rowClassCount, ops.One));
        int m = kept.Length;
        if (m == 0)
            return zeros;

        var keptIndex = new Tensor<int>(kept, new[] { m });
        var labelCounts = engine.TensorIndexSelect(rowClassCount, keptIndex, 0);
        var neighbours = engine.TensorMin(engine.TensorSubtractScalar(labelCounts, ops.One), ops.FromDouble(3));

        // Same-class pairs (self excluded) and the per-sample constant part of the estimate
        var classColumn = engine.TensorIndexSelect(allClasses, keptIndex, 0).Cast<T>();
        var shape = new[] { m, m };
        var sameClass = engine.TensorEquals(
            engine.TensorBroadcastTo(engine.Reshape(classColumn, new[] { m, 1 }), shape),
            engine.TensorBroadcastTo(engine.Reshape(classColumn, new[] { 1, m }), shape));
        sameClass = engine.TensorSubtract(sameClass, engine.TensorEye<T>(m));
        var infinity = PreprocessingTensorOps.Filled(shape, ops.FromDouble(double.PositiveInfinity));
        var origin = PreprocessingTensorOps.Filled(shape, ops.Zero);
        T constant = ops.Subtract(
            ops.Add(engine.TensorSum(engine.TensorDigamma(PreprocessingTensorOps.Filled(new[] { 1 }, ops.FromDouble(m)))),
                engine.TensorMean(engine.TensorDigamma(neighbours))),
            engine.TensorMean(engine.TensorDigamma(labelCounts)));
        var constantTensor = PreprocessingTensorOps.Filled(new[] { 1 }, constant);
        int columnsNeeded = Math.Min(3, m);
        var isFirst = engine.TensorEquals(neighbours, ops.One);
        var isSecond = engine.TensorEquals(neighbours, ops.FromDouble(2));
        var positiveRadius = PreprocessingTensorOps.Filled(new[] { m }, ops.Zero);

        var keptRows = engine.TensorIndexSelect(x, keptIndex, 0);
        var results = new Tensor<T>[p];
        for (int j = 0; j < p; j++)
        {
            var column = engine.TensorSliceAxis(keptRows, 1, j);
            var distances = engine.TensorAbs(engine.TensorSubtract(
                engine.TensorBroadcastTo(engine.Reshape(column, new[] { m, 1 }), shape),
                engine.TensorBroadcastTo(engine.Reshape(column, new[] { 1, m }), shape)));

            // Distance to the k_i-th nearest same-class neighbour
            var nearest = engine.TensorSort(engine.TensorWhere(sameClass, distances, infinity), 1, false).Values;
            var first = engine.TensorSliceAxis(nearest, 1, 0);
            var second = columnsNeeded > 1 ? engine.TensorSliceAxis(nearest, 1, 1) : first;
            var third = columnsNeeded > 2 ? engine.TensorSliceAxis(nearest, 1, 2) : second;
            var radius = engine.TensorWhere(isFirst, first, engine.TensorWhere(isSecond, second, third));

            // Samples strictly within the radius (at distance 0 when the radius is 0), itself included
            var radiusColumn = engine.TensorBroadcastTo(engine.Reshape(radius, new[] { m, 1 }), shape);
            var hasRadius = engine.TensorBroadcastTo(
                engine.Reshape(engine.TensorGreaterThan(radius, positiveRadius), new[] { m, 1 }), shape);
            var within = engine.TensorWhere(hasRadius,
                engine.TensorLessThan(distances, radiusColumn), engine.TensorLessEqual(distances, origin));
            var counts = engine.ReduceSum(within, new[] { 1 }, false);

            results[j] = engine.TensorSubtract(constantTensor,
                engine.Reshape(engine.ReduceMean(engine.TensorDigamma(counts), new[] { 0 }, false), new[] { 1 }));
        }

        return engine.TensorMax(engine.TensorConcatenate(results, 0), ops.Zero);
    }
}
