using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Wrapper;

/// <summary>
/// Whole-tensor default scorers shared by the wrapper feature selectors, generic in <typeparamref name="T"/>.
/// The selectors' scorer delegates return <see cref="double"/>, so each scorer converts its final value once.
/// </summary>
internal static class WrapperScoringOps
{
    private const double Epsilon = 1e-10;

    /// <summary>The given columns of <paramref name="data"/> as an [n, k] tensor.</summary>
    public static Tensor<T> Columns<T>(Matrix<T> data, int[] features) =>
        Tensor<T>.FromMatrix(PreprocessingTensorOps.SelectColumns(data, features));

    /// <summary>
    /// |Pearson r| of each given column with <paramref name="target"/>, as [k]; 0 when the column's or the target's
    /// centred sum of squares is not above 1e-10.
    /// </summary>
    public static Tensor<T> AbsTargetCorrelations<T>(Matrix<T> data, Vector<T> target, int[] features) =>
        AiDotNetEngine.Current.TensorAbs(
            PreprocessingTensorOps.TargetCorrelations(Columns(data, features), Tensor<T>.FromVector(target), Epsilon));

    /// <summary>|Pearson r| of column <paramref name="feature"/> with the target (see <see cref="AbsTargetCorrelations{T}"/>).</summary>
    public static double AbsTargetCorrelation<T>(Matrix<T> data, Vector<T> target, int feature) =>
        ToDouble(AbsTargetCorrelations(data, target, new[] { feature }).GetFlat(0));

    /// <summary>
    /// |Pearson r| of column <paramref name="feature"/> with <paramref name="target"/>; 0 when
    /// <c>sqrt(sxx * syy)</c> is not above 1e-10.
    /// </summary>
    public static double AbsTargetCorrelationWithDenominatorGuard<T>(Matrix<T> data, Vector<T> target, int feature) =>
        Math.Abs(ToDouble(PreprocessingTensorOps.CorrelationsWithDenominatorGuard(
            Columns(data, new[] { feature }), Tensor<T>.FromVector(target), Epsilon).GetFlat(0)));

    /// <summary>Mean of <see cref="AbsTargetCorrelations{T}"/> over the given (non-empty) columns.</summary>
    public static double MeanAbsTargetCorrelation<T>(Matrix<T> data, Vector<T> target, int[] features) =>
        ToDouble(AiDotNetEngine.Current.TensorMean(AbsTargetCorrelations(data, target, features)));

    /// <summary>
    /// Pearson r between columns <paramref name="first"/> and <paramref name="second"/>; 0 when either centred sum
    /// of squares is not above 1e-10.
    /// </summary>
    public static double ColumnCorrelation<T>(Matrix<T> data, int first, int second) =>
        ToDouble(PreprocessingTensorOps.TargetCorrelations(
            Columns(data, new[] { first }), Tensor<T>.FromVector(data.GetColumn(second)), Epsilon).GetFlat(0));

    /// <summary>
    /// Pearson r between columns <paramref name="first"/> and <paramref name="second"/>; 0 when
    /// <c>sqrt(sxx * syy)</c> is not above 1e-10.
    /// </summary>
    public static double ColumnCorrelationWithDenominatorGuard<T>(Matrix<T> data, int first, int second) =>
        ToDouble(PreprocessingTensorOps.CorrelationsWithDenominatorGuard(
            Columns(data, new[] { first }), Tensor<T>.FromVector(data.GetColumn(second)), Epsilon).GetFlat(0));

    /// <summary>
    /// Mean |Pearson r| over the pairs (i &lt; j) of the given columns, a pair scoring 0 when either centred sum of
    /// squares is not above 1e-10; 0 when there are fewer than two columns.
    /// </summary>
    public static double MeanAbsPairCorrelation<T>(Matrix<T> data, int[] features)
    {
        int k = features.Length;
        if (k < 2)
            return 0;

        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        var centered = SelectorTensorOps.CenterColumns(Columns(data, features));
        var sumOfSquares = SelectorTensorOps.ColumnSumOfSquares(centered);
        var varying = engine.TensorGreaterThan(sumOfSquares, ops.FromDouble(Epsilon));
        var safe = engine.TensorWhere(varying, sumOfSquares, PreprocessingTensorOps.Filled(new[] { k }, ops.One));
        var products = engine.TensorMatMul(engine.TensorTranspose(centered), centered);
        var correlation = engine.TensorDivide(products, engine.TensorSqrt(engine.TensorOuter(safe, safe)));
        var counted = engine.TensorMultiply(engine.TensorOuter(varying, varying), SelectorTensorOps.StrictUpperMask<T>(k));
        T total = engine.TensorSum(engine.TensorMultiply(counted, engine.TensorAbs(correlation)));
        return ToDouble(total) / (k * (k - 1) / 2);
    }

    /// <summary>
    /// Ordinary least-squares slopes of <paramref name="y"/> ([m]) on <paramref name="x"/> ([m, k]) with an
    /// intercept, like scikit-learn's <c>LinearRegression</c>: the minimum-norm solution on the centred data
    /// through the SVD, ignoring singular values not above 1e-10 times the largest. Returns the slopes [k] and, in
    /// <paramref name="xMean"/> ([1, k]) and <paramref name="yMean"/>, the means that give the intercept.
    /// </summary>
    public static Tensor<T> LeastSquaresSlopes<T>(Tensor<T> x, Tensor<T> y, out Tensor<T> xMean, out T yMean)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int m = x.Shape[0];
        int k = x.Shape[1];
        xMean = engine.ReduceMean(x, new[] { 0 }, true);
        yMean = engine.TensorMean(y);
        var xc = engine.TensorSubtract(x, xMean);
        var yc = engine.Reshape(engine.TensorSubtractScalar(y, yMean), new[] { m, 1 });

        var (u, s, vh) = LinalgDispatch<T>.Svd(xc, false);
        int rank = s.Length;
        var kept = engine.TensorGreaterThan(s, ops.Multiply(engine.TensorMaxValue(s), ops.FromDouble(Epsilon)));
        var safe = engine.TensorWhere(kept, s, PreprocessingTensorOps.Filled(new[] { rank }, ops.One));
        var inverse = engine.TensorMultiply(kept, engine.TensorReciprocal(safe));
        var projections = engine.Reshape(engine.TensorMatMul(engine.TensorTranspose(u), yc), new[] { rank });
        var scaled = engine.Reshape(engine.TensorMultiply(inverse, projections), new[] { rank, 1 });
        return engine.Reshape(engine.TensorMatMul(engine.TensorTranspose(vh), scaled), new[] { k });
    }

    /// <summary>
    /// Predictions <c>yMean + (x - xMean) slopes</c> ([m]) of a <see cref="LeastSquaresSlopes{T}"/> fit.
    /// </summary>
    public static Tensor<T> Predict<T>(Tensor<T> x, Tensor<T> slopes, Tensor<T> xMean, T yMean)
    {
        var engine = AiDotNetEngine.Current;
        int m = x.Shape[0];
        int k = x.Shape[1];
        var fitted = engine.TensorMatMul(engine.TensorSubtract(x, xMean), engine.Reshape(slopes, new[] { k, 1 }));
        return engine.TensorAddScalar(engine.Reshape(fitted, new[] { m }), yMean);
    }

    /// <summary>
    /// Coefficient of determination <c>1 - SS_res / SS_tot</c> of <paramref name="predictions"/> for
    /// <paramref name="y"/> (both [m]); 0 when <c>SS_tot</c> is not above 1e-10.
    /// </summary>
    public static double RSquared<T>(Tensor<T> y, Tensor<T> predictions)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        T ssRes = engine.TensorSumOfSquares(engine.TensorSubtract(y, predictions));
        T ssTot = engine.TensorSumOfSquares(engine.TensorSubtractScalar(y, engine.TensorMean(y)));
        return ops.GreaterThan(ssTot, ops.FromDouble(Epsilon))
            ? ToDouble(ops.Subtract(ops.One, ops.Divide(ssRes, ssTot)))
            : 0;
    }

    /// <summary>
    /// In-sample R^2 of the least-squares fit (with intercept) of <paramref name="target"/> on the given columns,
    /// scikit-learn's <c>LinearRegression().fit(X, y).score(X, y)</c>.
    /// </summary>
    public static double LinearRegressionRSquared<T>(Matrix<T> data, Vector<T> target, int[] features)
    {
        var x = Columns(data, features);
        var y = Tensor<T>.FromVector(target);
        var slopes = LeastSquaresSlopes(x, y, out var xMean, out T yMean);
        return RSquared(y, Predict(x, slopes, xMean, yMean));
    }

    /// <summary>
    /// Converts per-feature values to the <see cref="double"/> array a selector's double-valued search heuristics use.
    /// </summary>
    public static double[] ToDoubleArray<T>(Tensor<T> values) => values.Cast<double>().ToArray();

    /// <summary>Converts a scorer result to the <see cref="double"/> the scorer delegates return.</summary>
    public static double ToDouble<T>(T value) => MathHelper.GetNumericOperations<T>().ToDouble(value);
}
