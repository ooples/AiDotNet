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
    // Below this, a centred sum of squares is treated as zero (a constant feature or target).
    private const double ZeroVariance = 1e-10;

    // scikit-learn's mutual_info_regression default neighbour count.
    private const int MutualInfoNeighbors = 3;

    // Euler-Mascheroni constant: digamma(m) = H(m - 1) - gamma for a positive integer m.
    private const double EulerGamma = 0.57721566490153286060651209;

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
        if (dfBetween <= 0 || dfWithin <= 0)
            return Uninformative<T>(p);

        var (ssBetween, ssWithin) = FilterStatisticTensorOps.AnovaSums(FilterStatisticTensorOps.AsTensor(data), oneHot);
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

        var (sxx, sxy, syy) = FilterStatisticTensorOps.CenteredCrossSums(
            FilterStatisticTensorOps.AsTensor(data), FilterStatisticTensorOps.AsTensor(target));
        if (!ops.GreaterThan(syy, ops.FromDouble(ZeroVariance)))
            return Uninformative<T>(p);

        // r^2 = Sxy^2 / (Sxx Syy); a constant feature (Sxx = 0) has an undefined correlation, scored 0 with p = 1
        var r2 = FilterStatisticTensorOps.GuardedDivide(
            engine.TensorSquare(sxy), engine.TensorMultiplyScalar(sxx, syy), ZeroVariance, ops.Zero);
        var informative = engine.TensorGreaterThan(sxx, ops.FromDouble(ZeroVariance));

        // F = r^2 / (1 - r^2) (n - 2); a perfect fit (r^2 within numpy.isclose of 1) is the largest finite value
        var oneMinusR2 = FilterStatisticTensorOps.OneMinus(r2);
        var perfect = engine.TensorLessThan(engine.TensorAbs(oneMinusR2), ops.FromDouble(1e-8 + 1e-5));
        var f = engine.TensorMultiplyScalar(
            FilterStatisticTensorOps.GuardedDivide(r2, oneMinusR2, 0.0, ops.Zero), ops.FromDouble(df));
        f = engine.TensorWhere(perfect, FilterStatisticTensorOps.FilledLike(f, ops.MaxValue), f);
        f = engine.TensorWhere(informative, f, FilterStatisticTensorOps.FilledLike(f, ops.Zero));

        var pValues = FilterStatisticTensorOps.FSurvival(f, 1, df);
        pValues = engine.TensorWhere(perfect, FilterStatisticTensorOps.FilledLike(pValues, ops.Zero), pValues);
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
        var classProbability = engine.TensorDivideScalar(engine.ReduceSum(oneHot, new[] { 0 }, false), ops.FromDouble(n));
        var featureTotal = engine.ReduceSum(x, new[] { 0 }, false);
        var expected = engine.TensorOuter(classProbability, featureTotal);

        // a feature that is zero everywhere has no expected mass: scored 0 with p = 1
        var terms = FilterStatisticTensorOps.GuardedDivide(
            engine.TensorSquare(engine.TensorSubtract(observed, expected)), expected, 0.0, ops.Zero);
        var chi2 = engine.ReduceSum(terms, new[] { 0 }, false);
        var pValues = FilterStatisticTensorOps.ChiSquareSurvival(chi2, k - 1);
        var informative = engine.TensorGreaterThan(featureTotal, ops.Zero);
        pValues = engine.TensorWhere(informative, pValues, FilterStatisticTensorOps.FilledLike(pValues, ops.One));
        return (chi2.ToVector(), pValues.ToVector());
    }

    // sklearn.feature_selection.mutual_info_regression (n_neighbors = 3) without its 1e-10 tie-breaking noise:
    // MI = psi(n) + psi(k) - mean psi(n_x + 1) - mean psi(n_y + 1), the Kraskov et al. (2004) estimator 1,
    // on features and target scaled to unit (population) standard deviation.
    internal static Vector<T> MutualInfoRegression<T>(Matrix<T> data, Vector<T> target, int neighbors = MutualInfoNeighbors)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = data.Rows;
        int p = data.Columns;
        if (n <= neighbors)
            return new Vector<T>(p);

        // digamma at the integers 1..n from the harmonic numbers: psi(m) = H(m - 1) - gamma
        var harmonic = engine.TensorCumSum(engine.TensorReciprocal(engine.TensorArange<T>(1, n, 1)), 0);
        var digammaTable = engine.TensorAddScalar(
            engine.TensorConcatenate(new[] { new Tensor<T>(new[] { 1 }), harmonic }, 0), ops.FromDouble(-EulerGamma));
        T constant = ops.Add(
            ops.Subtract(ops.FromDouble(HarmonicNumber(n - 1)), ops.FromDouble(EulerGamma)),
            ops.Subtract(ops.FromDouble(HarmonicNumber(neighbors - 1)), ops.FromDouble(EulerGamma)));

        var y = ScaleToUnitStd(FilterStatisticTensorOps.AsTensor(target));
        var self = engine.TensorEye<T>(n);
        var infinity = FilterStatisticTensorOps.Filled(new[] { n, n }, ops.FromDouble(double.PositiveInfinity));
        var dy = engine.TensorWhere(self, infinity, AbsoluteDifferences(y));

        var x = FilterStatisticTensorOps.AsTensor(data);
        var scores = new List<Tensor<T>>(p);
        for (int j = 0; j < p; j++)
        {
            var column = engine.Reshape(engine.TensorIndexSelect(x, new Tensor<int>(new[] { j }, new[] { 1 }), 1), new[] { n });
            var dx = engine.TensorWhere(self, infinity, AbsoluteDifferences(ScaleToUnitStd(column)));

            // distance to the k-th nearest neighbour in the joint space (max norm), self excluded
            var joint = engine.TensorMax(dx, dy);
            var radius = engine.TensorIndexSelect(
                engine.TensorSort(joint, 1, false).Item1, new Tensor<int>(new[] { neighbors - 1 }, new[] { 1 }), 1);
            var radiusGrid = engine.TensorBroadcastTo(radius, new[] { n, n });

            // neighbours strictly inside that radius in each marginal space
            var nx = engine.ReduceSum(engine.TensorLessThan(dx, radiusGrid), new[] { 1 }, false);
            var ny = engine.ReduceSum(engine.TensorLessThan(dy, radiusGrid), new[] { 1 }, false);
            var psiX = engine.TensorGather(digammaTable, engine.TensorRound(nx).Cast<int>(), 0);
            var psiY = engine.TensorGather(digammaTable, engine.TensorRound(ny).Cast<int>(), 0);
            scores.Add(engine.TensorAdd(
                engine.ReduceMean(psiX, new[] { 0 }, true), engine.ReduceMean(psiY, new[] { 0 }, true)));
        }

        if (p == 0)
            return new Vector<T>(0);

        // MI = psi(n) + psi(k) - (mean psi(n_x + 1) + mean psi(n_y + 1)), clipped at 0 as scikit-learn does
        var mi = engine.TensorAddScalar(engine.TensorNegate(engine.TensorConcatenate(scores.ToArray(), 0)), constant);
        return engine.TensorMax(mi, ops.Zero).ToVector();
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

    // |v[i] - v[j]| for every pair ([n] to [n, n]).
    private static Tensor<T> AbsoluteDifferences<T>(Tensor<T> values)
    {
        var engine = AiDotNetEngine.Current;
        int n = values.Length;
        return engine.TensorAbs(engine.TensorSubtract(
            FilterStatisticTensorOps.ColumnBroadcast(values, n), FilterStatisticTensorOps.RowBroadcast(values, n)));
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
        var order = FilterStatisticTensorOps.StableOrder(p, descending: false);
        var orderTensor = new Tensor<int>(order, new[] { n });
        var sorted = engine.TensorGather(p, orderTensor, 0);

        // p_(r) n / r, then the running minimum from the largest rank down
        var scaled = engine.TensorMultiplyScalar(
            engine.TensorDivide(sorted, engine.TensorArange<T>(1, n + 1, 1)), ops.FromDouble(n));
        var runningMin = engine.TensorFlip(engine.TensorCumMin(engine.TensorFlip(scaled, new[] { 0 }), 0), new[] { 0 });
        var adjustedSorted = engine.TensorMin(runningMin, ops.One);

        // back to the original feature order
        var inverse = engine.TensorArgsort(orderTensor, 0, false);
        return engine.TensorGather(adjustedSorted, inverse, 0).ToVector();
    }

    /// <summary>Bonferroni adjusted p-values: <c>min(1, p n)</c>.</summary>
    public static Vector<T> BonferroniCorrection<T>(Vector<T> pValues)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        return engine.TensorMin(
            engine.TensorMultiplyScalar(FilterStatisticTensorOps.AsTensor(pValues), ops.FromDouble(pValues.Length)),
            ops.One).ToVector();
    }
}
