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
}
