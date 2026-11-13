using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Wrapper;

/// <summary>
/// Whole-tensor default scorers shared by the wrapper feature selectors, generic in <typeparamref name="T"/>.
/// Scores are returned in <typeparamref name="T"/>; the scalar helpers below let the selectors' per-candidate search
/// loops compare and combine scores without element-wise numeric-operation calls.
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
    public static T AbsTargetCorrelation<T>(Matrix<T> data, Vector<T> target, int feature) =>
        AbsTargetCorrelations(data, target, new[] { feature }).GetFlat(0);

    /// <summary>
    /// |Pearson r| of column <paramref name="feature"/> with <paramref name="target"/>; 0 when
    /// <c>sqrt(sxx * syy)</c> is not above 1e-10.
    /// </summary>
    public static T AbsTargetCorrelationWithDenominatorGuard<T>(Matrix<T> data, Vector<T> target, int feature) =>
        AiDotNetEngine.Current.TensorAbs(PreprocessingTensorOps.CorrelationsWithDenominatorGuard(
            Columns(data, new[] { feature }), Tensor<T>.FromVector(target), Epsilon)).GetFlat(0);

    /// <summary>Mean of <see cref="AbsTargetCorrelations{T}"/> over the given (non-empty) columns.</summary>
    public static T MeanAbsTargetCorrelation<T>(Matrix<T> data, Vector<T> target, int[] features) =>
        AiDotNetEngine.Current.TensorMean(AbsTargetCorrelations(data, target, features));

    /// <summary>
    /// Pearson r between columns <paramref name="first"/> and <paramref name="second"/>; 0 when either centred sum
    /// of squares is not above 1e-10.
    /// </summary>
    public static T ColumnCorrelation<T>(Matrix<T> data, int first, int second) =>
        PreprocessingTensorOps.TargetCorrelations(
            Columns(data, new[] { first }), Tensor<T>.FromVector(data.GetColumn(second)), Epsilon).GetFlat(0);

    /// <summary>
    /// Pearson r between columns <paramref name="first"/> and <paramref name="second"/>; 0 when
    /// <c>sqrt(sxx * syy)</c> is not above 1e-10.
    /// </summary>
    public static T ColumnCorrelationWithDenominatorGuard<T>(Matrix<T> data, int first, int second) =>
        PreprocessingTensorOps.CorrelationsWithDenominatorGuard(
            Columns(data, new[] { first }), Tensor<T>.FromVector(data.GetColumn(second)), Epsilon).GetFlat(0);

    /// <summary>
    /// Mean |Pearson r| over the pairs (i &lt; j) of the given columns, a pair scoring 0 when either centred sum of
    /// squares is not above 1e-10; 0 when there are fewer than two columns.
    /// </summary>
    public static T MeanAbsPairCorrelation<T>(Matrix<T> data, int[] features)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int k = features.Length;
        if (k < 2)
            return ops.Zero;

        var engine = AiDotNetEngine.Current;
        var centered = FilterStatisticTensorOps.CenterColumns(Columns(data, features));
        var sumOfSquares = SelectorTensorOps.ColumnSumOfSquares(centered);
        var varying = engine.TensorGreaterThan(sumOfSquares, PreprocessingTensorOps.Tolerance<T>(Epsilon));
        var safe = engine.TensorWhere(varying, sumOfSquares, PreprocessingTensorOps.Filled(new[] { k }, ops.One));
        var products = engine.TensorMatMul(engine.TensorTranspose(centered), centered);
        // sqrt(a) sqrt(b) rather than sqrt(a b): the product of two sums of squares overflows Half long before either
        var root = engine.TensorSqrt(safe);
        var correlation = engine.TensorDivide(products, engine.TensorOuter(root, root));
        var counted = engine.TensorMultiply(engine.TensorOuter(varying, varying), SelectorTensorOps.StrictUpperMask<T>(k));
        T total = engine.TensorSum(engine.TensorMultiply(counted, engine.TensorAbs(correlation)));
        return ops.Divide(total, ops.FromDouble(k * (k - 1) / 2));
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
        var kept = engine.TensorGreaterThan(s, ops.Multiply(engine.TensorMaxValue(s), PreprocessingTensorOps.Tolerance<T>(Epsilon)));
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
    public static T RSquared<T>(Tensor<T> y, Tensor<T> predictions)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        T ssRes = engine.TensorSumOfSquares(engine.TensorSubtract(y, predictions));
        T ssTot = engine.TensorSumOfSquares(engine.TensorSubtractScalar(y, engine.TensorMean(y)));
        return ops.GreaterThan(ssTot, PreprocessingTensorOps.Tolerance<T>(Epsilon))
            ? ops.Subtract(ops.One, ops.Divide(ssRes, ssTot))
            : ops.Zero;
    }

    /// <summary>
    /// <paramref name="x"/> ([m] or [m, k]) without its rows <paramref name="start"/> .. <paramref name="end"/> - 1.
    /// </summary>
    public static Tensor<T> ExcludeRows<T>(Tensor<T> x, int start, int end)
    {
        var engine = AiDotNetEngine.Current;
        int m = x.Shape[0];
        var parts = new List<Tensor<T>>();
        if (start > 0)
            parts.Add(RowBlock(x, 0, start));
        if (end < m)
            parts.Add(RowBlock(x, end, m - end));
        return parts.Count == 1 ? parts[0] : engine.TensorConcatenate(parts.ToArray(), 0);
    }

    /// <summary>Rows <paramref name="start"/> .. <paramref name="start"/> + <paramref name="count"/> - 1 of an [m] or [m, k] tensor.</summary>
    public static Tensor<T> RowBlock<T>(Tensor<T> x, int start, int count) =>
        x.Rank == 1
            ? AiDotNetEngine.Current.TensorSlice(x, new[] { start }, new[] { count })
            : AiDotNetEngine.Current.TensorSlice(x, new[] { start, 0 }, new[] { count, x.Shape[1] });

    /// <summary>
    /// In-sample R^2 of the least-squares fit (with intercept) of <paramref name="target"/> on the given columns,
    /// scikit-learn's <c>LinearRegression().fit(X, y).score(X, y)</c>.
    /// </summary>
    public static T LinearRegressionRSquared<T>(Matrix<T> data, Vector<T> target, int[] features)
    {
        var x = Columns(data, features);
        var y = Tensor<T>.FromVector(target);
        var slopes = LeastSquaresSlopes(x, y, out var xMean, out T yMean);
        return RSquared(y, Predict(x, slopes, xMean, yMean));
    }

    /// <summary>A double constant as <typeparamref name="T"/>.</summary>
    public static T Scalar<T>(double value) => MathHelper.GetNumericOperations<T>().FromDouble(value);

    /// <summary><c>a + b</c>.</summary>
    public static T Add<T>(T a, T b) => MathHelper.GetNumericOperations<T>().Add(a, b);

    /// <summary><c>a - b</c>.</summary>
    public static T Subtract<T>(T a, T b) => MathHelper.GetNumericOperations<T>().Subtract(a, b);

    /// <summary><c>a * b</c>.</summary>
    public static T Multiply<T>(T a, T b) => MathHelper.GetNumericOperations<T>().Multiply(a, b);

    /// <summary><c>a / b</c>.</summary>
    public static T Divide<T>(T a, T b) => MathHelper.GetNumericOperations<T>().Divide(a, b);

    /// <summary><c>e^x</c>.</summary>
    public static T Exp<T>(T x) => MathHelper.GetNumericOperations<T>().Exp(x);

    /// <summary>Whether <c>a &gt; b</c>.</summary>
    public static bool IsAbove<T>(T a, T b) => MathHelper.GetNumericOperations<T>().GreaterThan(a, b);

    /// <summary>Whether <c>a &gt;= b</c>.</summary>
    public static bool IsAtLeast<T>(T a, T b) => MathHelper.GetNumericOperations<T>().GreaterThanOrEquals(a, b);

    /// <summary>Whether <c>a &lt; b</c>.</summary>
    public static bool IsBelow<T>(T a, T b) => MathHelper.GetNumericOperations<T>().LessThan(a, b);

    /// <summary>The larger of <paramref name="a"/> and <paramref name="b"/> (<paramref name="a"/> on ties).</summary>
    public static T Max<T>(T a, T b) => IsAbove(b, a) ? b : a;

    /// <summary>
    /// The next <paramref name="count"/> values of <paramref name="random"/>.NextDouble(), in draw order. The
    /// metaheuristic selectors draw their random numbers through this, in the order master's scalar loops drew
    /// them, so a seeded run keeps reproducing the same search (a bulk engine draw seeded per vector would
    /// consume the generator differently).
    /// </summary>
    public static double[] NextDoubles(Random random, int count)
    {
        var draws = new double[count];
        for (int i = 0; i < count; i++)
            draws[i] = random.NextDouble();
        return draws;
    }

    /// <summary>
    /// The next <paramref name="count"/> draws of <paramref name="random"/>.NextDouble() (in draw order, see
    /// <see cref="NextDoubles"/>) as a 1-D tensor in T: the generator yields doubles, converted once as a whole.
    /// </summary>
    public static Tensor<T> Draws<T>(Random random, int count) => FromDraws<T>(NextDoubles(random, count));

    /// <summary>Uniform draws already taken from a generator, as a 1-D tensor in T (converted once as a whole).</summary>
    public static Tensor<T> FromDraws<T>(double[] draws) =>
        new Tensor<double>((double[])draws.Clone(), new[] { draws.Length }).Cast<T>();

    /// <summary>
    /// Every <paramref name="stride"/>-th entry of a 1-D tensor starting at <paramref name="offset"/>
    /// (entries <c>offset, offset + stride, ...</c>), for draws interleaved per dimension.
    /// </summary>
    public static Tensor<T> Strided<T>(Tensor<T> values, int stride, int offset)
    {
        var engine = AiDotNetEngine.Current;
        int count = values.Length / stride;
        return engine.Reshape(engine.TensorSliceAxis(engine.Reshape(values, new[] { count, stride }), 1, offset), new[] { count });
    }

    /// <summary><c>scale x + offset</c> element-wise.</summary>
    public static Tensor<T> Affine<T>(Tensor<T> x, T scale, T offset)
    {
        var engine = AiDotNetEngine.Current;
        return engine.TensorAddScalar(engine.TensorMultiplyScalar(x, scale), offset);
    }

    /// <summary>A 1-D tensor holding <paramref name="values"/>.</summary>
    public static Tensor<T> FromList<T>(IReadOnlyList<T> values) =>
        new Tensor<T>(values.ToArray(), new[] { values.Count });
    /// <summary>
    /// A [length] vector holding <paramref name="values"/> at their keys and <paramref name="fill"/> elsewhere.
    /// </summary>
    public static Vector<T> ScoresVector<T>(int length, T fill, IReadOnlyDictionary<int, T> values)
    {
        var result = PreprocessingTensorOps.Filled(new[] { length }, fill);
        if (values.Count > 0)
        {
            var keys = values.Keys.ToArray();
            var entries = new Tensor<T>(keys.Select(key => values[key]).ToArray(), new[] { keys.Length });
            result = AiDotNetEngine.Current.TensorIndexCopy(result, 0, new Tensor<int>(keys, new[] { keys.Length }), entries);
        }

        return result.ToVector();
    }

    /// <summary>
    /// A [length] vector holding <paramref name="values"/> ([keys.Length]) at <paramref name="keys"/> and
    /// <paramref name="fill"/> elsewhere.
    /// </summary>
    public static Vector<T> ScoresVector<T>(int length, T fill, int[] keys, Tensor<T> values)
    {
        var result = PreprocessingTensorOps.Filled(new[] { length }, fill);
        if (keys.Length > 0)
        {
            result = AiDotNetEngine.Current.TensorIndexCopy(result, 0,
                new Tensor<int>((int[])keys.Clone(), new[] { keys.Length }), values);
        }

        return result.ToVector();
    }

    /// <summary>The entries of a 1-D tensor at <paramref name="positions"/>, as [positions.Length].</summary>
    public static Tensor<T> Gather<T>(Tensor<T> values, int[] positions) =>
        AiDotNetEngine.Current.TensorIndexSelect(values, new Tensor<int>((int[])positions.Clone(), new[] { positions.Length }), 0);

    /// <summary>
    /// Mean |Pearson r| over the pairs (i &lt; j) of the given columns, a pair scoring 0 when <c>sqrt(sxx * syy)</c> is
    /// not above 1e-10; 0 when there are fewer than two columns.
    /// </summary>
    public static T MeanAbsPairCorrelationWithDenominatorGuard<T>(Matrix<T> data, int[] features)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int k = features.Length;
        if (k < 2)
            return ops.Zero;

        var engine = AiDotNetEngine.Current;
        var centered = FilterStatisticTensorOps.CenterColumns(Columns(data, features));
        var sumOfSquares = SelectorTensorOps.ColumnSumOfSquares(centered);
        // sqrt(a) sqrt(b) rather than sqrt(a b): the product of two sums of squares overflows Half long before either
        var root = engine.TensorSqrt(sumOfSquares);
        var denominator = engine.TensorOuter(root, root);
        var valid = engine.TensorGreaterThan(denominator, PreprocessingTensorOps.Tolerance<T>(Epsilon));
        var safe = engine.TensorWhere(valid, denominator, PreprocessingTensorOps.Filled(new[] { k, k }, ops.One));
        var correlation = engine.TensorDivide(engine.TensorMatMul(engine.TensorTranspose(centered), centered), safe);
        var counted = engine.TensorMultiply(valid, SelectorTensorOps.StrictUpperMask<T>(k));
        T total = engine.TensorSum(engine.TensorMultiply(counted, engine.TensorAbs(correlation)));
        return ops.Divide(total, ops.FromDouble(k * (k - 1) / 2));
    }

    /// <summary>A [length] 0/1 tensor with ones at <paramref name="features"/> (counted once per occurrence).</summary>
    public static Tensor<T> Membership<T>(int[] features, int length)
    {
        if (features.Length == 0)
            return PreprocessingTensorOps.Filled(new[] { length }, MathHelper.GetNumericOperations<T>().Zero);

        var engine = AiDotNetEngine.Current;
        var oneHot = engine.TensorOneHot<T>(new Tensor<int>((int[])features.Clone(), new[] { features.Length }), length);
        return engine.ReduceSum(oneHot, new[] { 0 }, false);
    }

    /// <summary>A [rows, length] 0/1 tensor whose row r marks <paramref name="subsets"/>[r].</summary>
    public static Tensor<T> MembershipRows<T>(IReadOnlyList<int[]> subsets, int length) =>
        AiDotNetEngine.Current.TensorStack(subsets.Select(subset => Membership<T>(subset, length)).ToArray(), 0);

    /// <summary>The positions of the true entries of a mask, ascending.</summary>
    public static int[] Positions(bool[] mask) => Enumerable.Range(0, mask.Length).Where(j => mask[j]).ToArray();

    /// <summary>
    /// Per feature, the sum of the scores of the subsets ([rows] <paramref name="scores"/>) that contain it, as
    /// [length]; a subset adds nothing to the features it leaves out.
    /// </summary>
    public static Tensor<T> ScoreSums<T>(IReadOnlyList<int[]> subsets, Tensor<T> scores, int length)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        if (subsets.Count == 0)
            return PreprocessingTensorOps.Filled(new[] { length }, ops.Zero);

        var membership = MembershipRows<T>(subsets, length);
        var shape = new[] { subsets.Count, length };
        var contributions = engine.TensorWhere(engine.TensorGreaterThan(membership, ops.Zero),
            engine.TensorMultiply(membership, engine.TensorBroadcastTo(engine.Reshape(scores, new[] { subsets.Count, 1 }), shape)),
            PreprocessingTensorOps.Filled(shape, ops.Zero));
        return engine.ReduceSum(contributions, new[] { 0 }, false);
    }
}
