using AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Helpers;

/// <summary>
/// The univariate scoring functions of scikit-learn's <c>sklearn.feature_selection</c> module, generic in T and
/// computed for every feature at once with engine operations.
/// </summary>
/// <remarks>
/// <list type="bullet">
/// <item><see cref="SelectKBestScoreFunc.FClassif"/>: one-way ANOVA F of each feature across the classes of the
/// target (<c>f_classif</c>), p-value from the F distribution.</item>
/// <item><see cref="SelectKBestScoreFunc.FRegression"/>: F of the univariate linear regression of the target on each
/// feature, <c>r^2 / (1 - r^2) (n - 2)</c> (<c>f_regression</c>), p-value from F(1, n - 2).</item>
/// <item><see cref="SelectKBestScoreFunc.Chi2"/>: chi-squared statistic of the class-wise feature sums against
/// their expectation (<c>chi2</c>), p-value from the chi-squared distribution with (classes - 1) degrees of
/// freedom. Features must be non-negative.</item>
/// <item><see cref="SelectKBestScoreFunc.MutualInfoRegression"/>: the Kraskov k-nearest-neighbour estimate of the
/// mutual information between each feature and a continuous target (<c>mutual_info_regression</c>, 3 neighbours).
/// It has no p-values.</item>
/// </list>
/// </remarks>
internal static class StatisticalTestHelper
{
    // Below this, a mean square of a feature scaled to unit largest deviation is treated as zero: the cut-off is
    // relative to the feature's own spread (an absolute 1e-10 on raw sums of squares scored a feature that varies
    // at 1e-6 as constant).
    private const double ZeroVariance = 1e-10;

    // numpy.isclose(r^2, 1) defaults, the "perfect fit" test of scikit-learn's f_regression(force_finite=True).
    private const double IsCloseAbsoluteTolerance = 1e-8;
    private const double IsCloseRelativeTolerance = 1e-5;

    // scikit-learn's mutual_info_regression default neighbour count.
    private const int MutualInfoNeighbors = 3;

    // Euler-Mascheroni constant: digamma(m) = H(m - 1) - gamma for a positive integer m.
    private const double EulerGamma = 0.57721566490153286060651209;

    // Scale of the tie-breaking noise of scikit-learn's mutual_info_regression.
    private const double TieNoiseScale = 1e-10;

    /// <summary>
    /// Scores every feature (column) of <paramref name="data"/> against <paramref name="target"/>.
    /// </summary>
    /// <returns>The scores and the p-values; the p-values are null for mutual information, which has none.</returns>
    public static (Vector<T> Scores, Vector<T>? PValues) ComputeScores<T>(
        Matrix<T> data, Vector<T> target, SelectKBestScoreFunc scoringFunction)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match the number of rows in data.");

        switch (scoringFunction)
        {
            case SelectKBestScoreFunc.FRegression:
                return FRegression(data, target);
            case SelectKBestScoreFunc.MutualInfoRegression:
                return (MutualInfoRegression(data, target), null);
            case SelectKBestScoreFunc.FClassif:
                return FClassif(data, target);
            case SelectKBestScoreFunc.Chi2:
                return Chi2(data, target);
            default:
                throw new ArgumentOutOfRangeException(nameof(scoringFunction), scoringFunction, "Unknown scoring function.");
        }
    }

    /// <summary>
    /// Scores every feature and requires p-values, as the p-value based selectors (false positive rate, false
    /// discovery rate, family-wise error) do; scikit-learn rejects a scorer without p-values there too.
    /// </summary>
    public static (Vector<T> Scores, Vector<T> PValues) ComputeScoresWithPValues<T>(
        Matrix<T> data, Vector<T> target, SelectKBestScoreFunc scoringFunction)
    {
        var (scores, pValues) = ComputeScores(data, target, scoringFunction);
        if (pValues is null)
            throw new ArgumentException(
                $"The {scoringFunction} scoring function produces no p-values; p-value based selection needs one that does.",
                nameof(scoringFunction));
        return (scores, pValues);
    }

    /// <summary>
    /// The class of every row as its position among the sorted distinct target values (numpy.unique), and the
    /// one-hot [n, classes] membership.
    /// </summary>
    public static Tensor<T> ClassOneHot<T>(Vector<T> target, out int classCount)
    {
        var engine = AiDotNetEngine.Current;
        if (target.Length == 0)
        {
            classCount = 0;
            return new Tensor<T>(new[] { 0, 0 });
        }

        var (classes, inverse, _) = engine.TensorUniqueWithInfo(FilterStatisticTensorOps.AsTensor(target), true, true, false);
        if (inverse is null)
            throw new InvalidOperationException("The engine returned no inverse map for the class labels.");
        classCount = classes.Length;
        return FilterStatisticTensorOps.ClassOneHot<T>(inverse.ToArray(), classCount);
    }

    // sklearn.feature_selection.f_classif
    private static (Vector<T> Scores, Vector<T>? PValues) FClassif<T>(Matrix<T> data, Vector<T> target)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = data.Rows;
        int p = data.Columns;

        var oneHot = ClassOneHot(target, out int k);
        int dfBetween = k - 1;
        int dfWithin = n - k;

        // every distinct target value is a class: a constant target has no second class, and a continuous one (all
        // values distinct) leaves no within-class degrees of freedom, so f_classif has no statistic at all
        if (dfBetween <= 0)
            throw new ArgumentException(
                "The FClassif scoring function needs a target with at least two classes; this target has one value.",
                nameof(target));
        if (dfWithin <= 0)
            throw new ArgumentException(
                $"The FClassif scoring function needs more rows ({n}) than classes ({k}); every distinct target value " +
                "is a class, so a continuous target needs the FRegression scoring function.", nameof(target));

        // F is unchanged by scaling a feature, so every feature is centred and divided by its largest deviation
        // first: the sums of squares then stay at most n (raw ones overflow Half past 256) and the zero-scatter
        // cut-off below is relative to the feature's spread; a constant feature (all zero) scores 0 with p = 1
        var scaled = FeatureScoringTensorOps.UnitDeviationColumns(FilterStatisticTensorOps.AsTensor(data), out _);
        var (ssBetween, ssWithin) = FilterStatisticTensorOps.AnovaSums(scaled, oneHot);
        var msBetween = engine.TensorDivideScalar(ssBetween, ops.FromDouble(dfBetween));
        var msWithin = engine.TensorDivideScalar(ssWithin, ops.FromDouble(dfWithin));

        // no within-class scatter: class means that differ separate the classes exactly (F = +inf)
        var f = FilterStatisticTensorOps.SeparationAwareRatio(msBetween, msWithin, ZeroVariance);
        return (f.ToVector(), FilterStatisticTensorOps.FSurvival(f, dfBetween, dfWithin).ToVector());
    }

    // sklearn.feature_selection.f_regression (center=True, force_finite=True)
    private static (Vector<T> Scores, Vector<T>? PValues) FRegression<T>(Matrix<T> data, Vector<T> target)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = data.Rows;
        int p = data.Columns;
        int df = n - 2;
        if (df <= 0)
            return Uninformative<T>(p);

        var x = FilterStatisticTensorOps.AsTensor(data);
        var y = FilterStatisticTensorOps.AsTensor(target);

        // a constant target leaves r undefined for every feature (scikit-learn: NaN, forced to F = 0 with p = 1)
        if (ops.Equals(engine.TensorMaxValue(y), engine.TensorMinValue(y)))
            return Uninformative<T>(p);

        // r of every feature (scale-free, so its sums stay in range for Half), squared; a constant feature (max = min)
        // has an undefined correlation, scored 0 with p = 1
        var r2 = engine.TensorSquare(FeatureScoringTensorOps.Correlation(x, y));
        var constantFeature = engine.TensorEquals(PreprocessingTensorOps.ColumnMax(x), PreprocessingTensorOps.ColumnMin(x));
        var informative = PreprocessingTensorOps.OneMinus(constantFeature);

        // F = r^2 / (1 - r^2) (n - 2). scikit-learn's force_finite replaces scores only when some F is not finite: an
        // exact fit (1 - r^2 = 0, or below 0 by rounding) or a constant feature (NaN). Then every feature whose r^2
        // is numpy.isclose to 1 takes the largest finite value with p = 0; otherwise near-perfect fits keep their
        // finite F, so a closer fit still ranks above a looser one
        var oneMinusR2 = PreprocessingTensorOps.OneMinus(r2);
        var f = engine.TensorMultiplyScalar(
            FilterStatisticTensorOps.GuardedDivide(r2, oneMinusR2, 0.0, ops.Zero), ops.FromDouble(df));
        var pValues = FilterStatisticTensorOps.FSurvival(f, 1, df);
        bool forceFinite = engine.TensorCountNonzero(engine.TensorLessThan(oneMinusR2, ops.Zero)) > 0
            || engine.TensorCountNonzero(engine.TensorEquals(oneMinusR2, ops.Zero)) > 0
            || engine.TensorCountNonzero(constantFeature) > 0;
        if (forceFinite)
        {
            var perfect = engine.TensorLessThan(
                engine.TensorAbs(oneMinusR2), ops.FromDouble(IsCloseAbsoluteTolerance + IsCloseRelativeTolerance));
            f = engine.TensorWhere(perfect, FilterStatisticTensorOps.FilledLike(f, ops.MaxValue), f);
            pValues = engine.TensorWhere(perfect, FilterStatisticTensorOps.FilledLike(pValues, ops.Zero), pValues);
        }

        f = engine.TensorWhere(informative, f, FilterStatisticTensorOps.FilledLike(f, ops.Zero));
        pValues = engine.TensorWhere(informative, pValues, FilterStatisticTensorOps.FilledLike(pValues, ops.One));
        return (f.ToVector(), pValues.ToVector());
    }

    // sklearn.feature_selection.chi2
    private static (Vector<T> Scores, Vector<T>? PValues) Chi2<T>(Matrix<T> data, Vector<T> target)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = data.Rows;
        int p = data.Columns;
        var x = FilterStatisticTensorOps.AsTensor(data);

        if (n > 0 && engine.TensorCountNonzero(engine.TensorLessThan(x, ops.Zero)) > 0)
            throw new ArgumentException("The chi-squared score needs non-negative feature values (counts or frequencies).");

        var oneHot = ClassOneHot(target, out int k);
        if (k < 2)
            return Uninformative<T>(p);

        // observed[c, j] = sum of feature j over class c; expected = P(class c) * total of feature j
        var observed = engine.TensorMatMul(engine.TensorTranspose(oneHot), x);
        var classProbability = engine.TensorDivideScalar(FeatureScoringTensorOps.Counts(oneHot, 0), ops.FromDouble(n));
        var featureTotal = engine.ReduceSum(x, new[] { 0 }, false);
        var expected = engine.TensorOuter(classProbability, featureTotal);

        // a feature that is zero everywhere has no expected mass: scored 0 with p = 1
        // (O - E) * ((O - E) / E): (O - E)^2 overflows Half once |O - E| passes 256
        var difference = engine.TensorSubtract(observed, expected);
        var terms = engine.TensorMultiply(
            difference, FilterStatisticTensorOps.GuardedDivide(difference, expected, 0.0, ops.Zero));
        var chi2 = engine.ReduceSum(terms, new[] { 0 }, false);
        var pValues = FilterStatisticTensorOps.ChiSquareSurvival(chi2, k - 1);
        var informative = engine.TensorGreaterThan(featureTotal, ops.Zero);
        pValues = engine.TensorWhere(informative, pValues, FilterStatisticTensorOps.FilledLike(pValues, ops.One));
        return (chi2.ToVector(), pValues.ToVector());
    }

    // sklearn.feature_selection.mutual_info_regression (n_neighbors = 3): MI = psi(n) + psi(k) - mean psi(n_x + 1)
    // - mean psi(n_y + 1), the Kraskov et al. (2004) estimator 1, on features and target scaled to unit (population)
    // standard deviation. With a noise seed it first adds the tie-breaking noise of scikit-learn,
    // 1e-10 max(1, mean |v|) N(0, 1) per column and for the target (its random_state); without one the data are used
    // as they are, so the estimate is deterministic. Rows run in blocks (see FeatureScoringTensorOps.RowBlocks) and
    // the k-th neighbour distance is a partial selection, so only a block x n slice of the pair distances is held: the
    // full [n, n] grids it replaced were 3.2 GB each at 20000 rows.
    internal static Vector<T> MutualInfoRegression<T>(
        Matrix<T> data, Vector<T> target, int neighbors = MutualInfoNeighbors, int? noiseSeed = null)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = data.Rows;
        int p = data.Columns;
        if (n <= neighbors || p == 0)
            return new Vector<T>(p);

        var digammaTable = DigammaTable<T>(n);
        T constant = ops.Add(
            ops.Subtract(ops.FromDouble(HarmonicNumber(n - 1)), ops.FromDouble(EulerGamma)),
            ops.Subtract(ops.FromDouble(HarmonicNumber(neighbors - 1)), ops.FromDouble(EulerGamma)));

        // the feature noise is drawn before the target noise, as scikit-learn does
        var random = noiseSeed.HasValue ? RandomHelper.CreateSeededRandom(noiseSeed.Value) : null;
        var x = FilterStatisticTensorOps.AsTensor(data);
        var columns = new Tensor<T>[p];
        for (int j = 0; j < p; j++)
            columns[j] = WithTieNoise(ScaleToUnitStd(engine.Reshape(
                engine.TensorIndexSelect(x, new Tensor<int>(new[] { j }, new[] { 1 }), 1), new[] { n })), random);
        var y = WithTieNoise(ScaleToUnitStd(FilterStatisticTensorOps.AsTensor(target)), random);

        T infinity = ops.FromDouble(double.PositiveInfinity);
        T rowShare = ops.FromDouble(1.0 / n);
        var meanPsi = new Tensor<T>[p];
        for (int j = 0; j < p; j++)
            meanPsi[j] = new Tensor<T>(new[] { 1 });

        foreach (var block in FeatureScoringTensorOps.RowBlocks(Enumerable.Range(0, n).ToArray(), n))
        {
            int start = block[0];
            int count = block.Length;
            var notSelf = PreprocessingTensorOps.OneMinus(FeatureScoringTensorOps.SelfMask<T>(block, n));
            var dy = FeatureScoringTensorOps.KeepOrFill(AbsoluteDifferences(y, start, count), notSelf, infinity);
            for (int j = 0; j < p; j++)
            {
                var dx = FeatureScoringTensorOps.KeepOrFill(AbsoluteDifferences(columns[j], start, count), notSelf, infinity);

                // distance to the k-th nearest neighbour in the joint space (max norm), self excluded: the largest of
                // the k smallest distances, from a top-k of the negated distances
                var joint = engine.TensorMax(dx, dy);
                var nearest = engine.TensorTopK(engine.TensorNegate(joint), neighbors, 1, out _);
                var radius = engine.ReduceMax(engine.TensorNegate(nearest), new[] { 1 }, true);
                var radiusGrid = engine.TensorBroadcastTo(radius, new[] { count, n });

                // neighbours strictly inside that radius in each marginal space, counted in int (a 0/1 sum in T stops
                // counting at 2048 for Half)
                var nx = engine.ReduceSum(engine.TensorLessThan(dx, radiusGrid).Cast<int>(), new[] { 1 }, false);
                var ny = engine.ReduceSum(engine.TensorLessThan(dy, radiusGrid).Cast<int>(), new[] { 1 }, false);
                var psi = engine.TensorAdd(engine.TensorGather(digammaTable, nx, 0), engine.TensorGather(digammaTable, ny, 0));

                // the row means, every term divided by n before the sum (a sum of n digammas overflows Half)
                meanPsi[j] = engine.TensorAdd(meanPsi[j],
                    engine.ReduceSum(engine.TensorMultiplyScalar(psi, rowShare), new[] { 0 }, true));
            }
        }

        // MI = psi(n) + psi(k) - (mean psi(n_x + 1) + mean psi(n_y + 1)), clipped at 0 as scikit-learn does
        var mi = engine.TensorAddScalar(engine.TensorNegate(engine.TensorConcatenate(meanPsi, 0)), constant);
        return engine.TensorMax(mi, ops.Zero).ToVector();
    }

    // The tie-breaking noise of scikit-learn, v + 1e-10 max(1, mean |v|) N(0, 1), drawn with the next seed of the
    // random source; the values unchanged without one. (1e-10 is below the resolution of Half and float at unit
    // scale, so the noise only moves double values.)
    private static Tensor<T> WithTieNoise<T>(Tensor<T> values, Random? random)
    {
        if (random is null)
            return values;

        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        T meanMagnitude = engine.TensorMean(engine.TensorAbs(values));
        T amplitude = ops.Multiply(ops.FromDouble(TieNoiseScale),
            ops.GreaterThan(meanMagnitude, ops.One) ? meanMagnitude : ops.One);
        var noise = engine.TensorRandnLike(values, random.Next());
        return engine.TensorAdd(values, engine.TensorMultiplyScalar(noise, amplitude));
    }

    // The entries of a vector divided by its population standard deviation (sklearn.preprocessing.scale without
    // centring); a constant vector stays as it is.
    private static Tensor<T> ScaleToUnitStd<T>(Tensor<T> values)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        var centered = engine.TensorSubtractScalar(values, engine.TensorMean(values));
        T std = ops.Sqrt(ops.Divide(engine.TensorSumOfSquares(centered), ops.FromDouble(values.Length)));
        return ops.GreaterThan(std, ops.Zero) ? engine.TensorDivideScalar(values, std) : values;
    }

    /// <summary>
    /// <c>psi(m + 1) = H(m) - gamma</c> for <c>m = 0 .. n - 1</c> ([n]): the digamma function at the integers 1..n
    /// from the harmonic numbers. The table depends on <paramref name="n"/> alone (no data), so it is summed in
    /// double and returned in T: in Half the integers past 2048 are not exact and a running sum of 1 / m stops
    /// growing once 1 / m is below half its spacing (near 6.8 by m = 512), so psi(3001) would read 6.2, not 8.0.
    /// </summary>
    internal static Tensor<T> DigammaTable<T>(int n)
    {
        var engine = AiDotNetEngine.Current;
        if (n <= 0)
            return new Tensor<T>(new[] { 0 });

        var harmonic = engine.TensorCumSum(engine.TensorReciprocal(engine.TensorArange<double>(1, n, 1)), 0);
        return engine.TensorAddScalar(
            engine.TensorConcatenate(new[] { new Tensor<double>(new[] { 1 }), harmonic }, 0), -EulerGamma).Cast<T>();
    }

    // |v[i] - v[j]| for the rows i in [start, start + count) and every j ([n] to [count, n]).
    private static Tensor<T> AbsoluteDifferences<T>(Tensor<T> values, int start, int count)
    {
        var engine = AiDotNetEngine.Current;
        int n = values.Length;
        return engine.TensorAbs(engine.TensorSubtract(
            FilterStatisticTensorOps.ColumnBroadcast(engine.TensorNarrow(values, 0, start, count), n),
            FilterStatisticTensorOps.RowBroadcast(values, count)));
    }

    // H(m) = 1 + 1/2 + ... + 1/m, H(0) = 0; a scalar constant of the estimator, not data.
    private static double HarmonicNumber(int m)
    {
        double sum = 0;
        for (int i = 1; i <= m; i++)
            sum += 1.0 / i;
        return sum;
    }

    // Every feature scored 0 with p-value 1 (the statistic is undefined for this sample).
    private static (Vector<T> Scores, Vector<T>? PValues) Uninformative<T>(int p)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        return (new Vector<T>(p), FilterStatisticTensorOps.Filled(new[] { p }, ops.One).ToVector());
    }

    /// <summary>
    /// Benjamini-Hochberg adjusted p-values: <c>min(1, min_{r' &gt;= r} p_(r') n / r')</c> for the p-value of rank r
    /// (statsmodels <c>multipletests(method="fdr_bh")</c>). Comparing them with alpha (<c>&lt;=</c>) is the selection
    /// rule of scikit-learn's <c>SelectFdr</c>.
    /// </summary>
    public static Vector<T> BenjaminiHochbergCorrection<T>(Vector<T> pValues)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = pValues.Length;
        if (n == 0)
            return new Vector<T>(0);

        var p = FilterStatisticTensorOps.AsTensor(pValues);
        var order = PreprocessingTensorOps.StableOrder(p, descending: false);
        var orderTensor = new Tensor<int>(order, new[] { n });
        var sorted = engine.TensorGather(p, orderTensor, 0);

        // p_(r) n / r, then the running minimum from the largest rank down
        // the factors n / r depend on the ranks alone, so they are formed in double (capped at T's largest value)
        // and applied in one multiply: in Half the ranks past 2048 are not exact, and p_(r) / r underflows to 0 for a
        // small p-value at a high rank (1e-5 / 3000) before the product with n can bring it back
        var factors = engine.TensorClampMax(
            engine.TensorMultiplyScalar(engine.TensorReciprocal(engine.TensorArange<double>(1, n + 1, 1)), (double)n),
            ops.ToDouble(ops.MaxValue)).Cast<T>();
        var scaled = engine.TensorMultiply(sorted, factors);
        var runningMin = engine.TensorFlip(engine.TensorCumMin(engine.TensorFlip(scaled, new[] { 0 }), 0), new[] { 0 });
        var adjustedSorted = engine.TensorMin(runningMin, ops.One);

        // back to the original feature order
        var inverse = engine.TensorArgsort(orderTensor, 0, false);
        return engine.TensorGather(adjustedSorted, inverse, 0).ToVector();
    }

    /// <summary>
    /// Bonferroni adjusted p-values: <c>min(1, p n)</c>. The factor n is capped at T's largest value: 65536 tests read
    /// as infinity in Half, which made a zero p-value 0 * infinity = NaN and rejected a perfectly predictive feature
    /// (any p-value above 1 / 65504 is adjusted to 1 either way).
    /// </summary>
    public static Vector<T> BonferroniCorrection<T>(Vector<T> pValues)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        T factor = ops.FromDouble(Math.Min(pValues.Length, ops.ToDouble(ops.MaxValue)));
        return engine.TensorMin(
            engine.TensorMultiplyScalar(FilterStatisticTensorOps.AsTensor(pValues), factor), ops.One).ToVector();
    }
}
