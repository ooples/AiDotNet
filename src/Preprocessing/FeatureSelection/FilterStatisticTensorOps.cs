using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection;

/// <summary>
/// Whole-array engine helpers shared by the filter feature selectors: column gathers, arg-min/arg-max picks,
/// class one-hot encodings, per-feature test statistics (ANOVA, simple regression, two-sample moments)
/// and the survival functions of the F, Student t, chi-squared and normal distributions.
/// </summary>
/// <remarks>
/// Every helper works on whole tensors through <see cref="AiDotNetEngine.Current"/>, generic in T. The
/// distribution functions follow scipy.stats (<c>f.sf</c>, <c>t.sf</c>, <c>chi2.sf</c>, <c>norm.sf</c>), which
/// is what scikit-learn's univariate scorers (<c>f_classif</c>, <c>f_regression</c>, <c>chi2</c>) use.
/// </remarks>
internal static class FilterStatisticTensorOps
{
    // Lentz's method replaces a vanishing denominator with this value (Numerical Recipes, betacf FPMIN).
    private const double ContinuedFractionFloor = 1e-30;

    // Upper bound on continued-fraction terms; the expansion needs O(sqrt(max(a, b))) terms, so this
    // covers degrees of freedom far beyond any realistic sample size.
    private const int ContinuedFractionMaxTerms = 10000;

    // Relative change below which a continued-fraction term no longer moves the result (raised to a few ulps
    // of T for low-precision types, where rounding noise alone keeps the term ratio that far from 1).
    private const double ContinuedFractionTolerance = 1e-14;

    /// <summary>The data matrix as an [n, p] tensor.</summary>
    public static Tensor<T> AsTensor<T>(Matrix<T> data) => Tensor<T>.FromMatrix(data);

    /// <summary>The vector as an [n] tensor.</summary>
    public static Tensor<T> AsTensor<T>(Vector<T> values) => Tensor<T>.FromVector(values);

    /// <summary>Gathers the given columns of <paramref name="data"/> into a new matrix, in the given order.</summary>
    public static Matrix<T> SelectColumns<T>(Matrix<T> data, int[] columns)
    {
        if (columns.Length == 0 || data.Rows == 0)
            return new Matrix<T>(data.Rows, columns.Length);

        var indices = new Tensor<int>((int[])columns.Clone(), new[] { columns.Length });
        return AiDotNetEngine.Current.TensorIndexSelect(Tensor<T>.FromMatrix(data), indices, 1).ToMatrix();
    }

    /// <summary>A tensor of the given shape with every element set to <paramref name="value"/>.</summary>
    public static Tensor<T> Filled<T>(int[] shape, T value)
    {
        var result = new Tensor<T>(shape);
        AiDotNetEngine.Current.TensorFill(result, value);
        return result;
    }

    /// <summary>A tensor shaped like <paramref name="like"/> with every element set to <paramref name="value"/>.</summary>
    public static Tensor<T> FilledLike<T>(Tensor<T> like, T value) => Filled(like.Shape.ToArray(), value);

    /// <summary>
    /// The values with NaN replaced by the smallest representable value, so a NaN orders below every number the
    /// way <see cref="Comparer{T}.Default"/> orders a NaN double (LINQ <c>OrderBy</c> puts it first, <c>OrderByDescending</c> last).
    /// </summary>
    public static Tensor<T> NaNAsLowest<T>(Tensor<T> values)
    {
        var engine = AiDotNetEngine.Current;
        return engine.TensorWhere(engine.TensorIsNan(values), FilledLike(values, MathHelper.GetNumericOperations<T>().MinValue), values);
    }

    /// <summary>The positions of the non-zero entries of a 0/1 mask, ascending.</summary>
    public static int[] MaskPositions<T>(Tensor<T> mask)
    {
        var engine = AiDotNetEngine.Current;
        var flat = mask.Rank == 1 ? mask : engine.Reshape(mask, new[] { mask.Length });
        return engine.TensorNonzero(flat).ToArray();
    }

    /// <summary>The index of the first (lowest-index) largest entry of a 1-D tensor; NaN never wins over a number.</summary>
    public static int FirstArgMax<T>(Tensor<T> values) => PreprocessingTensorOps.StableOrder(values, descending: true)[0];

    /// <summary>The index of the first (lowest-index) smallest entry of a 1-D tensor; NaN never wins over a number.</summary>
    public static int FirstArgMin<T>(Tensor<T> values) => PreprocessingTensorOps.StableOrder(values, descending: false)[0];

    /// <summary>
    /// The common "significant features first" rule: when at least <paramref name="k"/> features have a p-value
    /// below <paramref name="alpha"/>, the <paramref name="k"/> of them with the smallest p-values (ties to the lower
    /// index); otherwise the <paramref name="k"/> features with the largest <paramref name="scores"/>. Returned in
    /// ascending index order.
    /// </summary>
    public static int[] SignificantOrTopK<T>(Tensor<T> pValues, T alpha, Tensor<T> scores, int k)
    {
        var engine = AiDotNetEngine.Current;
        var significant = engine.TensorLessThan(pValues, alpha);
        if (engine.TensorCountNonzero(significant) >= k)
        {
            var key = engine.TensorWhere(significant, pValues, FilledLike(pValues, MathHelper.GetNumericOperations<T>().MaxValue));
            return PreprocessingTensorOps.BottomKIndicesSorted(key, k);
        }

        return PreprocessingTensorOps.TopKIndicesSorted(scores, k);
    }

    /// <summary>
    /// The "candidates first" rule: when at least <paramref name="k"/> scores are at or above
    /// <paramref name="minimum"/>, the <paramref name="k"/> best of those candidates; otherwise the
    /// <paramref name="k"/> best scores overall (ties to the lower index). Returned in ascending index order.
    /// </summary>
    public static int[] CandidatesOrTopK<T>(Tensor<T> scores, T minimum, int k)
    {
        var engine = AiDotNetEngine.Current;
        var candidates = PreprocessingTensorOps.OneMinus(engine.TensorLessThan(NaNAsLowest(scores), minimum));
        if (engine.TensorCountNonzero(candidates) >= k)
            return PreprocessingTensorOps.TopKIndicesSorted(
                engine.TensorWhere(candidates, scores, FilledLike(scores, MathHelper.GetNumericOperations<T>().MinValue)), k);

        return PreprocessingTensorOps.TopKIndicesSorted(scores, k);
    }

    /// <summary>
    /// Integer class labels of a target, each value rounded to the nearest integer like <c>(int)Math.Round(value)</c>:
    /// returns, per row, the position of its class among the sorted distinct labels.
    /// </summary>
    public static int[] ClassIndex<T>(Vector<T> target, out int classCount)
    {
        var labels = AiDotNetEngine.Current.TensorRound(AsTensor(target)).Cast<int>().ToArray();
        var classes = labels.Distinct().OrderBy(c => c).ToArray();
        var classIndex = new int[labels.Length];
        for (int i = 0; i < labels.Length; i++)
            classIndex[i] = Array.BinarySearch(classes, labels[i]);
        classCount = classes.Length;
        return classIndex;
    }

    /// <summary>One-hot class membership [n, classCount] (0/1 in T) from per-row class positions.</summary>
    public static Tensor<T> ClassOneHot<T>(int[] classIndex, int classCount)
    {
        if (classIndex.Length == 0)
            return new Tensor<T>(new[] { 0, classCount });
        return AiDotNetEngine.Current.TensorOneHot<T>(
            new Tensor<int>((int[])classIndex.Clone(), new[] { classIndex.Length }), classCount);
    }

    /// <summary>Broadcasts a [p] row across <paramref name="rows"/> rows ([rows, p]).</summary>
    public static Tensor<T> RowBroadcast<T>(Tensor<T> row, int rows)
    {
        var engine = AiDotNetEngine.Current;
        return engine.TensorBroadcastTo(engine.Reshape(row, new[] { 1, row.Length }), new[] { rows, row.Length });
    }

    /// <summary>Broadcasts an [n] column across <paramref name="columns"/> columns ([n, columns]).</summary>
    public static Tensor<T> ColumnBroadcast<T>(Tensor<T> column, int columns)
    {
        var engine = AiDotNetEngine.Current;
        return engine.TensorBroadcastTo(engine.Reshape(column, new[] { column.Length, 1 }), new[] { column.Length, columns });
    }

    /// <summary>Column means of an [n, p] tensor ([p]).</summary>
    public static Tensor<T> ColumnMeans<T>(Tensor<T> x) => AiDotNetEngine.Current.ReduceMean(x, new[] { 0 }, false);

    /// <summary>
    /// The <paramref name="q"/>-quantile of every column of an [n, p] tensor ([p]) with linear interpolation between
    /// the order statistics at position <c>q (n - 1)</c> (numpy.quantile's default); q = 0.5 is the median, the
    /// mean of the two middle values when n is even.
    /// </summary>
    public static Tensor<T> ColumnQuantile<T>(Tensor<T> x, double q) =>
        AiDotNetEngine.Current.TensorQuantile(x, q, 0, false, QuantileInterpolation.Linear);

    /// <summary>Column variances of an [n, p] tensor with divisor n (numpy.var, ddof = 0) ([p]).</summary>
    public static Tensor<T> ColumnPopulationVariance<T>(Tensor<T> x) =>
        AiDotNetEngine.Current.TensorDivideScalar(
            ColumnSumOfSquaredDeviations(x), MathHelper.GetNumericOperations<T>().FromDouble(x.Shape[0]));

    // Upper bound on the elements of one block of the DFT basis built by ColumnDftPower.
    private const int DftBlockElements = 1 << 22;

    /// <summary>
    /// Squared DFT magnitudes <c>|sum_i x[i, j] exp(-2 pi i k i / n)|^2</c> of every column of an [n, p] tensor at the
    /// frequencies <c>k = 1 .. kCount</c> ([kCount, p]), evaluated as cos/sin basis matrix products built in blocks of
    /// frequencies so the basis never holds more than a few million elements.
    /// </summary>
    public static Tensor<T> ColumnDftPower<T>(Tensor<T> x, int kCount)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = x.Shape[0];
        int p = x.Shape[1];
        if (kCount <= 0 || n == 0)
            return new Tensor<T>(new[] { Math.Max(kCount, 0), p });

        var sampleIndex = engine.Reshape(engine.TensorArange<T>(0, n, 1), new[] { 1, n });
        int block = Math.Max(1, Math.Min(kCount, DftBlockElements / n));
        var blocks = new List<Tensor<T>>();
        for (int start = 1; start <= kCount; start += block)
        {
            int count = Math.Min(block, kCount - start + 1);
            var frequency = engine.Reshape(engine.TensorArange<T>(start, start + count, 1), new[] { count, 1 });

            // angle[k, i] = 2 pi k i / n
            var angle = engine.TensorMultiplyScalar(
                engine.TensorMatMul(frequency, sampleIndex), ops.FromDouble(2.0 * Math.PI / n));
            var real = engine.TensorMatMul(engine.TensorCos(angle), x);
            var imaginary = engine.TensorMatMul(engine.TensorSin(angle), x);
            blocks.Add(engine.TensorAdd(engine.TensorSquare(real), engine.TensorSquare(imaginary)));
        }

        return blocks.Count == 1 ? blocks[0] : engine.TensorConcatenate(blocks.ToArray(), 0);
    }

    /// <summary>
    /// The number of positive frequencies below the Nyquist frequency for <paramref name="n"/> samples,
    /// <c>ceil(n / 2) - 1</c>: the bins <c>numpy.fft.rfft</c> returns after the DC term, without the Nyquist bin
    /// of an even length.
    /// </summary>
    public static int PositiveFrequencyCount(int n) => Math.Max(0, (n + 1) / 2 - 1);

    /// <summary>
    /// Every column of an [n, p] tensor (or every entry of an [n] tensor) minus its mean along the first axis.
    /// </summary>
    public static Tensor<T> CenterColumns<T>(Tensor<T> x)
    {
        var engine = AiDotNetEngine.Current;
        var means = engine.ReduceMean(x, new[] { 0 }, true);
        return engine.TensorSubtract(x, engine.TensorBroadcastTo(means, x.Shape.ToArray()));
    }

    /// <summary>Column sums of squared deviations from the column mean ([n, p] to [p]).</summary>
    public static Tensor<T> ColumnSumOfSquaredDeviations<T>(Tensor<T> x)
    {
        var engine = AiDotNetEngine.Current;
        return engine.ReduceSum(engine.TensorSquare(CenterColumns(x)), new[] { 0 }, false);
    }

    /// <summary>
    /// Centred cross sums of every column of <paramref name="x"/> ([n, p]) with <paramref name="y"/> ([n]):
    /// <c>Sxx = sum (x - x̄)^2</c> and <c>Sxy = sum (x - x̄)(y - ȳ)</c> per column, and <c>Syy = sum (y - ȳ)^2</c>.
    /// </summary>
    public static (Tensor<T> Sxx, Tensor<T> Sxy, T Syy) CenteredCrossSums<T>(Tensor<T> x, Tensor<T> y)
    {
        var engine = AiDotNetEngine.Current;
        int n = x.Shape[0];
        int p = x.Shape[1];
        var xc = CenterColumns(x);
        var yc = engine.Reshape(engine.TensorSubtractScalar(y, engine.TensorMean(y)), new[] { n, 1 });
        var sxy = engine.Reshape(engine.TensorMatMul(engine.TensorTranspose(xc), yc), new[] { p });
        var sxx = engine.ReduceSum(engine.TensorSquare(xc), new[] { 0 }, false);
        return (sxx, sxy, engine.TensorSumOfSquares(yc));
    }

    /// <summary>
    /// One-way ANOVA sums of squares of every column of <paramref name="x"/> ([n, p]) across the classes of
    /// <paramref name="classOneHot"/> ([n, k]): <c>SSB = sum_c n_c (mean_c - mean)^2</c> and
    /// <c>SSW = sum_i (x_i - mean_{class(i)})^2</c>, both [p].
    /// </summary>
    public static (Tensor<T> SsBetween, Tensor<T> SsWithin) AnovaSums<T>(Tensor<T> x, Tensor<T> classOneHot)
    {
        var engine = AiDotNetEngine.Current;
        int p = x.Shape[1];
        int k = classOneHot.Shape[1];
        // Class sizes counted in int: 0/1 sums in a narrow T (Half) stop growing at 2048
        var counts = engine.ReduceSum(classOneHot.Cast<int>(), new[] { 0 }, false).Cast<T>();
        var classMeans = engine.TensorDivide(
            engine.TensorMatMul(engine.TensorTranspose(classOneHot), x), ColumnBroadcast(counts, p));
        var deviation = engine.TensorSubtract(classMeans, RowBroadcast(ColumnMeans(x), k));
        var ssBetween = engine.ReduceSum(
            engine.TensorMultiply(engine.TensorSquare(deviation), ColumnBroadcast(counts, p)), new[] { 0 }, false);
        var residual = engine.TensorSubtract(x, engine.TensorMatMul(classOneHot, classMeans));
        var ssWithin = engine.ReduceSum(engine.TensorSquare(residual), new[] { 0 }, false);
        return (ssBetween, ssWithin);
    }

    /// <summary>
    /// Mean and sum of squared deviations of every column over the rows selected by a 0/1 row mask
    /// (<paramref name="rowMask"/> [n], <paramref name="count"/> selected rows): both [p].
    /// </summary>
    public static (Tensor<T> Mean, Tensor<T> SumSquares) MaskedColumnMoments<T>(Tensor<T> x, Tensor<T> rowMask, int count)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = x.Shape[0];
        int p = x.Shape[1];
        var maskRow = engine.Reshape(rowMask, new[] { 1, n });
        var mean = engine.TensorDivideScalar(
            engine.Reshape(engine.TensorMatMul(maskRow, x), new[] { p }), ops.FromDouble(count));
        var squared = engine.TensorSquare(engine.TensorSubtract(x, RowBroadcast(mean, n)));
        var sumSquares = engine.Reshape(engine.TensorMatMul(maskRow, squared), new[] { p });
        return (mean, sumSquares);
    }

    /// <summary>
    /// The regularized incomplete beta function <c>I_x(a, b)</c> element-wise (scipy.special.betainc), by the
    /// Numerical Recipes continued fraction evaluated with the modified Lentz method on every element at once,
    /// using the symmetry <c>I_x(a, b) = 1 - I_{1-x}(b, a)</c> where the fraction converges faster.
    /// <paramref name="oneMinusX"/> is <c>1 - x</c>, passed separately so callers that can form it without
    /// cancellation (an F or t statistic near 0 puts x within rounding of 1) keep full precision.
    /// </summary>
    public static Tensor<T> RegularizedIncompleteBeta<T>(Tensor<T> a, Tensor<T> b, Tensor<T> x, Tensor<T> oneMinusX)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        T one = ops.One;

        // direct where x < (a + 1) / (a + b + 2), otherwise evaluate I_{1-x}(b, a)
        var split = engine.TensorDivide(
            engine.TensorAddScalar(a, one), engine.TensorAddScalar(engine.TensorAdd(a, b), ops.FromDouble(2)));
        var direct = engine.TensorLessThan(x, split);
        var aa = engine.TensorWhere(direct, a, b);
        var bb = engine.TensorWhere(direct, b, a);
        var xx = engine.TensorWhere(direct, x, oneMinusX);
        var yy = engine.TensorWhere(direct, oneMinusX, x);

        // front factor x^a (1-x)^b / (a B(a, b)), in logs
        var logFront = engine.TensorAdd(
            engine.TensorSubtract(
                engine.TensorSubtract(engine.TensorLgamma(engine.TensorAdd(aa, bb)), engine.TensorLgamma(aa)),
                engine.TensorLgamma(bb)),
            engine.TensorAdd(engine.TensorXlogy(aa, xx), engine.TensorXlogy(bb, yy)));
        var partial = engine.TensorDivide(
            engine.TensorMultiply(engine.TensorExp(logFront), BetaContinuedFraction(aa, bb, xx)), aa);
        return engine.TensorWhere(direct, partial, PreprocessingTensorOps.OneMinus(partial));
    }

    // Numerical Recipes betacf, every element advanced together; stops once every element has converged.
    private static Tensor<T> BetaContinuedFraction<T>(Tensor<T> a, Tensor<T> b, Tensor<T> x)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        T one = ops.One;
        T tolerance = ops.FromDouble(Math.Max(ContinuedFractionTolerance, 16.0 * Math.Pow(2.0, -ops.PrecisionBits)));
        var floor = FilledLike(x, ops.FromDouble(ContinuedFractionFloor));

        var qab = engine.TensorAdd(a, b);
        var qap = engine.TensorAddScalar(a, one);
        var qam = engine.TensorSubtractScalar(a, one);
        var c = FilledLike(x, one);
        var d = AwayFromZero(PreprocessingTensorOps.OneMinus(engine.TensorDivide(engine.TensorMultiply(qab, x), qap)), floor);
        d = engine.TensorReciprocal(d);
        var h = d;
        var m = FilledLike(x, ops.Zero);

        for (int term = 0; term < ContinuedFractionMaxTerms; term++)
        {
            m = engine.TensorAddScalar(m, one);
            var m2 = engine.TensorAdd(m, m);

            // even step: m (b - m) x / ((a - 1 + 2m)(a + 2m))
            var even = engine.TensorDivide(
                engine.TensorMultiply(engine.TensorMultiply(m, engine.TensorSubtract(b, m)), x),
                engine.TensorMultiply(engine.TensorAdd(qam, m2), engine.TensorAdd(a, m2)));
            d = engine.TensorReciprocal(AwayFromZero(engine.TensorAddScalar(engine.TensorMultiply(even, d), one), floor));
            c = AwayFromZero(engine.TensorAddScalar(engine.TensorDivide(even, c), one), floor);
            h = engine.TensorMultiply(h, engine.TensorMultiply(d, c));

            // odd step: -(a + m)(a + b + m) x / ((a + 2m)(a + 1 + 2m))
            var odd = engine.TensorNegate(engine.TensorDivide(
                engine.TensorMultiply(engine.TensorMultiply(engine.TensorAdd(a, m), engine.TensorAdd(qab, m)), x),
                engine.TensorMultiply(engine.TensorAdd(a, m2), engine.TensorAdd(qap, m2))));
            d = engine.TensorReciprocal(AwayFromZero(engine.TensorAddScalar(engine.TensorMultiply(odd, d), one), floor));
            c = AwayFromZero(engine.TensorAddScalar(engine.TensorDivide(odd, c), one), floor);
            var delta = engine.TensorMultiply(d, c);
            h = engine.TensorMultiply(h, delta);

            var moving = engine.TensorGreaterThan(engine.TensorAbs(engine.TensorSubtractScalar(delta, one)), tolerance);
            if (engine.TensorCountNonzero(moving) == 0)
                break;
        }

        return h;
    }

    // Lentz guard: replaces entries whose magnitude is below the floor with the floor.
    private static Tensor<T> AwayFromZero<T>(Tensor<T> values, Tensor<T> floor)
    {
        var engine = AiDotNetEngine.Current;
        return engine.TensorWhere(engine.TensorLessThan(engine.TensorAbs(values), floor), floor, values);
    }

    /// <summary>
    /// Upper-tail probability of the F distribution, <c>P(F(d1, d2) &gt; f)</c> (scipy.stats.f.sf), element-wise:
    /// <c>I_{d2 / (d2 + d1 f)}(d2 / 2, d1 / 2)</c>. An infinite statistic gives 0 and NaN stays NaN.
    /// </summary>
    public static Tensor<T> FSurvival<T>(Tensor<T> f, double d1, double d2)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        T one = ops.One;

        // x = d2 / (d2 + d1 f) = 1 / (1 + d1 f / d2) and 1 - x = 1 / (1 + d2 / (d1 f)): finite for f = 0 and f = inf
        var scaled = engine.TensorMultiplyScalar(f, ops.FromDouble(d1 / d2));
        var x = engine.TensorReciprocal(engine.TensorAddScalar(scaled, one));
        var oneMinusX = engine.TensorReciprocal(engine.TensorAddScalar(engine.TensorReciprocal(scaled), one));
        return RegularizedIncompleteBeta(
            FilledLike(f, ops.FromDouble(d2 / 2.0)), FilledLike(f, ops.FromDouble(d1 / 2.0)), x, oneMinusX);
    }

    /// <summary>
    /// Two-sided p-value of a Student t statistic, <c>2 P(T(df) &gt; |t|)</c> (2 * scipy.stats.t.sf(|t|, df)),
    /// element-wise with per-element degrees of freedom: <c>I_{df / (df + t^2)}(df / 2, 1 / 2)</c>.
    /// </summary>
    public static Tensor<T> StudentTTwoSided<T>(Tensor<T> t, Tensor<T> df)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        T one = ops.One;

        // x = df / (df + t^2) = 1 / (1 + t^2 / df) and 1 - x = 1 / (1 + df / t^2): finite for t = 0 and |t| = inf
        var ratio = engine.TensorDivide(engine.TensorSquare(t), df);
        var x = engine.TensorReciprocal(engine.TensorAddScalar(ratio, one));
        var oneMinusX = engine.TensorReciprocal(engine.TensorAddScalar(engine.TensorReciprocal(ratio), one));
        return RegularizedIncompleteBeta(
            engine.TensorMultiplyScalar(df, ops.FromDouble(0.5)), FilledLike(t, ops.FromDouble(0.5)), x, oneMinusX);
    }

    /// <summary>Two-sided p-value of a Student t statistic with a common number of degrees of freedom.</summary>
    public static Tensor<T> StudentTTwoSided<T>(Tensor<T> t, double df) =>
        StudentTTwoSided(t, FilledLike(t, MathHelper.GetNumericOperations<T>().FromDouble(df)));

    /// <summary>
    /// Upper-tail probability of the chi-squared distribution, <c>P(chi2(df) &gt; x)</c> (scipy.stats.chi2.sf),
    /// element-wise: the regularized upper incomplete gamma <c>Q(df / 2, x / 2)</c>.
    /// </summary>
    public static Tensor<T> ChiSquareSurvival<T>(Tensor<T> x, double df)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        T half = ops.FromDouble(0.5);
        return engine.TensorIgammac(FilledLike(x, ops.FromDouble(df / 2.0)), engine.TensorMultiplyScalar(x, half));
    }

    /// <summary>Upper-tail probability of the standard normal, <c>P(Z &gt; z) = erfc(z / sqrt 2) / 2</c> (scipy.stats.norm.sf).</summary>
    public static Tensor<T> NormalSurvival<T>(Tensor<T> z)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        return engine.TensorMultiplyScalar(
            engine.TensorErfc(engine.TensorMultiplyScalar(z, ops.FromDouble(1.0 / Math.Sqrt(2.0)))), ops.FromDouble(0.5));
    }

    /// <summary>Two-sided p-value of a standard normal statistic, <c>2 P(Z &gt; |z|) = erfc(|z| / sqrt 2)</c>.</summary>
    public static Tensor<T> NormalTwoSided<T>(Tensor<T> z)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        return engine.TensorErfc(engine.TensorMultiplyScalar(engine.TensorAbs(z), ops.FromDouble(1.0 / Math.Sqrt(2.0))));
    }

    /// <summary>
    /// A variance ratio (F or similar statistic) that stays meaningful when the error term vanishes:
    /// <c>numerator / denominator</c> where the denominator is above <paramref name="epsilon"/>; otherwise
    /// +infinity where the numerator is above <paramref name="epsilon"/> (the effect is exact, as scikit-learn's
    /// <c>f_classif</c> reports it) and 0 where both vanish (a constant feature carries no signal).
    /// </summary>
    public static Tensor<T> SeparationAwareRatio<T>(Tensor<T> numerator, Tensor<T> denominator, double epsilon)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        T eps = ops.FromDouble(epsilon);
        var separated = engine.TensorWhere(
            engine.TensorGreaterThan(numerator, eps),
            FilledLike(numerator, ops.FromDouble(double.PositiveInfinity)),
            FilledLike(numerator, ops.Zero));
        return engine.TensorWhere(
            engine.TensorGreaterThan(denominator, eps), GuardedDivide(numerator, denominator, epsilon, ops.Zero), separated);
    }

    /// <summary>
    /// <see cref="SeparationAwareRatio{T}"/> of a signed numerator: <c>numerator / denominator</c>, or
    /// <c>sign(numerator) * infinity</c> when the denominator vanishes but the numerator does not (a t statistic
    /// of two groups that do not overlap at all), or 0 when both vanish.
    /// </summary>
    public static Tensor<T> SignedSeparationAwareRatio<T>(Tensor<T> numerator, Tensor<T> denominator, double epsilon)
    {
        var engine = AiDotNetEngine.Current;
        return engine.TensorMultiply(
            engine.TensorSign(numerator), SeparationAwareRatio(engine.TensorAbs(numerator), denominator, epsilon));
    }

    /// <summary>
    /// The rows of a target below <paramref name="threshold"/> as a 0/1 [n] mask (the "class 0" rows of a binary
    /// target), with its row count in <paramref name="count"/>.
    /// </summary>
    public static Tensor<T> BelowMask<T>(Vector<T> target, double threshold, out int count)
    {
        var engine = AiDotNetEngine.Current;
        var mask = engine.TensorLessThan(AsTensor(target), MathHelper.GetNumericOperations<T>().FromDouble(threshold));
        count = engine.TensorCountNonzero(mask);
        return mask;
    }

    /// <summary>
    /// <c>numerator / denominator</c> where the denominator is above <paramref name="epsilon"/>, otherwise
    /// <paramref name="fallback"/>.
    /// </summary>
    public static Tensor<T> GuardedDivide<T>(Tensor<T> numerator, Tensor<T> denominator, double epsilon, T fallback)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        var safe = engine.TensorGreaterThan(denominator, ops.FromDouble(epsilon));
        var safeDenominator = engine.TensorWhere(safe, denominator, FilledLike(denominator, ops.One));
        return engine.TensorWhere(safe, engine.TensorDivide(numerator, safeDenominator), FilledLike(numerator, fallback));
    }
}
