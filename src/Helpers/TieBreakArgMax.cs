using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Helpers;

/// <summary>
/// The hand-written arg-max scan that <see cref="TieBreakArgMax"/> reproduces.
/// </summary>
internal enum ArgMaxScan
{
    /// <summary>
    /// <c>best = x[0]; for c &gt;= 1: if (NumOps.Compare(x[c], best) &gt; 0)</c>. <c>Compare</c> is
    /// <c>CompareTo</c>, which orders NaN below every number: the result is the first index of the largest
    /// non-NaN value, or 0 when the whole row is NaN.
    /// </summary>
    CompareFromFirst,

    /// <summary>
    /// <c>best = x[0]; for c &gt;= 1: if (NumOps.GreaterThan(x[c], best))</c>. <c>GreaterThan</c> is false
    /// against NaN, so a NaN in column 0 is never replaced (result 0) and NaN elsewhere is skipped.
    /// </summary>
    GreaterThanFromFirst,

    /// <summary>
    /// <c>best = NumOps.MinValue; idx = 0; for c &gt;= 0: if (NumOps.GreaterThan(x[c], best))</c>. Only entries
    /// strictly above <c>MinValue</c> (so never NaN or -infinity) can win; when none does the result is 0.
    /// </summary>
    GreaterThanFromMinValue,
}

/// <summary>
/// Row-wise arg-max and stable ordering with the exact tie and NaN rules of hand-written scans, built from
/// whole-tensor engine operations.
/// </summary>
/// <remarks>
/// <see cref="IEngine.TensorArgMax{T}(Tensor{T}, int)"/> and <see cref="IEngine.TensorArgsort{T}(Tensor{T}, int, bool)"/>
/// do not document which index wins a tie or how NaN is treated. The scans they replace return the FIRST index
/// of the maximum, so these helpers build a composite (rank, index) key that is unique in every row and only
/// then hand it to the engine.
/// </remarks>
internal static class TieBreakArgMax
{
    /// <summary>
    /// Returns, for each row of a [rows, cols] tensor, the index the given scan would pick.
    /// </summary>
    /// <param name="scores">Scores of shape [rows, cols].</param>
    /// <param name="scan">The scan whose tie and NaN behaviour to reproduce.</param>
    /// <returns>Column indices of shape [rows].</returns>
    public static Tensor<int> FirstMaxIndexPerRow<T>(Tensor<T> scores, ArgMaxScan scan = ArgMaxScan.CompareFromFirst)
    {
        if (scores is null)
        {
            throw new ArgumentNullException(nameof(scores));
        }

        if (scores.Shape.Length != 2)
        {
            throw new ArgumentException("Scores must be a [rows, cols] tensor.", nameof(scores));
        }

        int rows = scores.Shape[0];
        int cols = scores.Shape[1];
        if (cols == 0)
        {
            throw new ArgumentException("Scores must have at least one column.", nameof(scores));
        }

        if (rows == 0)
        {
            return new Tensor<int>(new[] { 0 });
        }

        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        var zeros = engine.TensorZerosLike(scores);
        var ones = engine.TensorOnesLike(scores);
        var minValue = engine.TensorFullLike(scores, ops.MinValue);

        // eligible = entries that can win; rank = 2 for an eligible row maximum, 1 for another eligible entry
        // that still beats NaN under CompareTo, 0 otherwise. key = rank * cols + (cols - 1 - column).
        Tensor<T> eligible;
        Tensor<T> fallbackRank;
        if (scan == ArgMaxScan.GreaterThanFromMinValue)
        {
            eligible = engine.TensorGreaterThan(scores, ops.MinValue);
            fallbackRank = zeros;
        }
        else
        {
            eligible = engine.TensorWhere(engine.TensorIsNan(scores), zeros, ones);
            fallbackRank = eligible;
        }

        var clean = engine.TensorWhere(eligible, scores, minValue);
        var rowMax = engine.TensorBroadcastTo(engine.ReduceMax(clean, new[] { 1 }, true), new[] { rows, cols });
        var isMax = engine.TensorMultiply(engine.TensorEquals(clean, rowMax), eligible);

        var rank = engine.TensorAdd(engine.TensorMultiplyScalar(isMax, ops.FromDouble(2.0)), fallbackRank);
        var reversedColumn = engine.TensorArange<T>(cols - 1, -1, -1).Reshape(1, cols);
        var key = engine.TensorAdd(engine.TensorMultiplyScalar(rank, ops.FromDouble(cols)), reversedColumn);
        var best = engine.TensorArgMax(key, 1).Reshape(rows);

        if (scan != ArgMaxScan.GreaterThanFromFirst)
        {
            return best;
        }

        var firstColumnNaN = engine.TensorIsNan(engine.TensorNarrow(scores, 1, 0, 1).Reshape(rows));
        return engine.TensorWhere(firstColumnNaN, new Tensor<int>(new[] { rows }), best);
    }

    /// <summary>
    /// Returns, for each row of a [rows, cols] tensor, the column order of a STABLE sort by value, descending,
    /// under <c>NumOps.Compare</c> ordering (NaN ranks below every number).
    /// </summary>
    /// <remarks>
    /// The engine sort is not documented as stable, so each element's final position is computed directly as
    /// its rank: the number of row entries that sort before it (strictly greater, or equal and earlier). The
    /// ranks form a permutation without ties, so arg-sorting them is unambiguous. Cost is O(rows * cols^2).
    /// </remarks>
    /// <param name="values">Values of shape [rows, cols].</param>
    /// <returns>Column indices of shape [rows, cols]; row r lists its columns from largest to smallest.</returns>
    public static Tensor<int> StableDescendingOrder<T>(Tensor<T> values)
    {
        if (values is null)
        {
            throw new ArgumentNullException(nameof(values));
        }

        if (values.Shape.Length != 2)
        {
            throw new ArgumentException("Values must be a [rows, cols] tensor.", nameof(values));
        }

        int rows = values.Shape[0];
        int cols = values.Shape[1];
        if (rows == 0 || cols == 0)
        {
            return new Tensor<int>(new[] { rows, cols });
        }

        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        var pairShape = new[] { rows, cols, cols };

        var isNaN = engine.TensorIsNan(values);
        var valid = engine.TensorWhere(isNaN, engine.TensorZerosLike(values), engine.TensorOnesLike(values));
        var clean = engine.TensorWhere(isNaN, engine.TensorFullLike(values, ops.MinValue), values);

        // Axis 1 is the element being ranked (j), axis 2 the element compared against it (l).
        var cleanJ = engine.TensorBroadcastTo(clean.Reshape(rows, cols, 1), pairShape);
        var cleanL = engine.TensorBroadcastTo(clean.Reshape(rows, 1, cols), pairShape);
        var validJ = engine.TensorBroadcastTo(valid.Reshape(rows, cols, 1), pairShape);
        var validL = engine.TensorBroadcastTo(valid.Reshape(rows, 1, cols), pairShape);
        var invalidJ = engine.ScalarMinusTensor(ops.One, validJ);
        var invalidL = engine.ScalarMinusTensor(ops.One, validL);
        var bothValid = engine.TensorMultiply(validL, validJ);

        // l sorts before j when it is larger (a number beats NaN), or equal and at a lower column.
        var lGreater = engine.TensorAdd(
            engine.TensorMultiply(validL, invalidJ),
            engine.TensorMultiply(bothValid, engine.TensorGreaterThan(cleanL, cleanJ)));
        var lEqual = engine.TensorAdd(
            engine.TensorMultiply(bothValid, engine.TensorEquals(cleanL, cleanJ)),
            engine.TensorMultiply(invalidL, invalidJ));

        var column = engine.TensorArange<T>(0, cols);
        var columnJ = engine.TensorBroadcastTo(column.Reshape(1, cols, 1), pairShape);
        var columnL = engine.TensorBroadcastTo(column.Reshape(1, 1, cols), pairShape);
        var lEarlier = engine.TensorLessThan(columnL, columnJ);

        var before = engine.TensorAdd(lGreater, engine.TensorMultiply(lEqual, lEarlier));
        var rank = engine.ReduceSum(before, new[] { 2 }, false);
        return engine.TensorArgsort(rank, 1, false);
    }

    /// <summary>
    /// Returns the first index holding the maximum of a vector, with the same rules as
    /// <see cref="FirstMaxIndexPerRow{T}(Tensor{T}, ArgMaxScan)"/>.
    /// </summary>
    public static int FirstMaxIndex<T>(Tensor<T> values, ArgMaxScan scan = ArgMaxScan.CompareFromFirst)
    {
        if (values is null)
        {
            throw new ArgumentNullException(nameof(values));
        }

        return FirstMaxIndexPerRow(values.Reshape(1, values.Length), scan).GetFlat(0);
    }
}
