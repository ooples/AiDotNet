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

        // The rank (0..3) is exact in every T; the key is built in long because rank * cols + column would
        // lose integers in T (Half above 2048, float above 2^24) and tie distinct columns.
        var rank = engine.TensorAdd(engine.TensorMultiplyScalar(isMax, ops.FromDouble(2.0)), fallbackRank).Cast<long>();
        var reversedColumn = engine.TensorBroadcastTo(
            engine.TensorArange<long>(cols - 1, -1, -1).Reshape(1, cols), new[] { rows, cols });
        var key = engine.TensorAdd(engine.TensorMultiplyScalar(rank, (long)cols), reversedColumn);
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
    /// under <c>NumOps.Compare</c> ordering (NaN ranks below every number, so it comes last).
    /// </summary>
    /// <param name="values">Values of shape [rows, cols].</param>
    /// <returns>Column indices of shape [rows, cols]; row r lists its columns from largest to smallest.</returns>
    public static Tensor<int> StableDescendingOrder<T>(Tensor<T> values) => StableOrder(values, descending: true);

    /// <summary>
    /// Returns, for each row of a [rows, cols] tensor, the column order of a STABLE sort by value under
    /// <c>NumOps.Compare</c> ordering (NaN ranks below every number).
    /// </summary>
    /// <remarks>
    /// The engine sort is not documented as stable, so each element's final position is computed directly as
    /// its rank: the number of row entries that sort before it (strictly before in value order, or equal and
    /// at a lower column). The ranks form a permutation without ties, so arg-sorting them is unambiguous.
    /// Cost is O(rows * cols^2); use it for short rows (neighbour lists, candidate sets).
    /// </remarks>
    /// <param name="values">Values of shape [rows, cols].</param>
    /// <param name="descending">True for largest first, false for smallest first.</param>
    /// <returns>Column indices of shape [rows, cols].</returns>
    public static Tensor<int> StableOrder<T>(Tensor<T> values, bool descending)
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

        // l sorts strictly before j: descending puts larger numbers first and NaN last, ascending puts NaN
        // first and smaller numbers first.
        var lStrictlyBefore = descending
            ? engine.TensorAdd(
                engine.TensorMultiply(validL, invalidJ),
                engine.TensorMultiply(bothValid, engine.TensorGreaterThan(cleanL, cleanJ)))
            : engine.TensorAdd(
                engine.TensorMultiply(invalidL, validJ),
                engine.TensorMultiply(bothValid, engine.TensorLessThan(cleanL, cleanJ)));
        var lEqual = engine.TensorAdd(
            engine.TensorMultiply(bothValid, engine.TensorEquals(cleanL, cleanJ)),
            engine.TensorMultiply(invalidL, invalidJ));

        // Columns, the tie-break and the rank are int: in T, Half loses column indices and stops counting at
        // 2048, which ties distinct elements.
        var column = engine.TensorArange<int>(0, cols);
        var columnJ = engine.TensorBroadcastTo(column.Reshape(1, cols, 1), pairShape);
        var columnL = engine.TensorBroadcastTo(column.Reshape(1, 1, cols), pairShape);
        var lEarlier = engine.TensorLessThan(columnL, columnJ);

        var before = engine.TensorAdd(lStrictlyBefore.Cast<int>(), engine.TensorMultiply(lEqual.Cast<int>(), lEarlier));
        var rank = engine.ReduceSum(before, new[] { 2 }, false);
        return engine.TensorArgsort(rank, 1, false);
    }

    /// <summary>
    /// Returns, for each row of a [rows, cols] tensor, the first <paramref name="k"/> columns of a STABLE
    /// ascending sort under <c>NumOps.Compare</c> ordering, smallest first: NaN ranks below every number
    /// (including negative infinity), so NaN entries come first, then the numbers, ties in column order. This is
    /// the order <c>Array.Sort</c> with <c>NumOps.Compare</c> gives, made stable.
    /// </summary>
    public static Tensor<int> StableSmallestK<T>(Tensor<T> values, int k)
    {
        if (values is null)
        {
            throw new ArgumentNullException(nameof(values));
        }

        int rows = values.Shape[0];
        int cols = values.Shape[1];
        if (k < 0 || k > cols)
        {
            throw new ArgumentOutOfRangeException(nameof(k));
        }

        if (rows == 0 || k == 0)
        {
            return new Tensor<int>(new[] { rows, k });
        }

        return AiDotNetEngine.Current.TensorNarrow(StableOrderRows(values, descending: false), 1, 0, k);
    }

    /// <summary>
    /// Returns the order of a STABLE descending sort of a vector of any length under <c>NumOps.Compare</c>
    /// ordering (NaN ranks below every number, so it comes last), in O(n log n). This is the order
    /// <c>OrderByDescending</c> gives on the values.
    /// </summary>
    /// <param name="values">Values of any shape; they are ordered as one flat vector.</param>
    /// <returns>Flat indices of shape [n], largest value first.</returns>
    public static Tensor<int> StableDescendingOrder1D<T>(Tensor<T> values)
    {
        if (values is null)
        {
            throw new ArgumentNullException(nameof(values));
        }

        int n = values.Length;
        if (n == 0)
        {
            return new Tensor<int>(new[] { 0 });
        }

        return StableOrderRows(values.Reshape(1, n), descending: true).Reshape(n);
    }

    /// <summary>
    /// Stable per-row sort order of a [rows, cols] tensor under <c>NumOps.Compare</c> ordering, in
    /// O(rows * cols * log cols). NaN ranks below every number: it comes first ascending and last descending.
    /// </summary>
    /// <remarks>
    /// NaN is first replaced by the row's smallest number, which merges it into the lowest value group. The engine
    /// sort then puts equal values next to each other in some order; each element's dense group rank, a NaN tier
    /// inside the group and its column form the unique 64-bit key <c>rank * 2c + tier * c + column</c>, so
    /// arg-sorting the key is unambiguous. The tier is the NaN flag descending (NaN after the numbers it joined) and
    /// its complement ascending (NaN before them).
    /// </remarks>
    private static Tensor<int> StableOrderRows<T>(Tensor<T> values, bool descending)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int rows = values.Shape[0];
        int cols = values.Shape[1];
        var shape = new[] { rows, cols };

        var isNaN = engine.TensorIsNan(values);
        var numbersOnly = engine.TensorWhere(isNaN, engine.TensorFullLike(values, ops.MaxValue), values);
        var rowMin = engine.TensorNarrow(engine.TensorSort(numbersOnly, 1, false).Values, 1, 0, 1);
        var clean = engine.TensorWhere(isNaN, engine.TensorBroadcastTo(rowMin, shape), values);

        // Dense rank of every value within its row, counted from the first in sort order
        var sorted = engine.TensorSort(clean, 1, descending);
        var startsGroup = cols == 1
            ? new Tensor<int>(new[] { rows, 1 })
            : engine.TensorConcatenate(
                new[]
                {
                    new Tensor<int>(new[] { rows, 1 }),
                    engine.TensorNotEquals(
                        engine.TensorNarrow(sorted.Values, 1, 1, cols - 1),
                        engine.TensorNarrow(sorted.Values, 1, 0, cols - 1)).Cast<int>(),
                },
                1);
        var sortedRank = engine.TensorCumSum(startsGroup, 1);

        // Back to the original columns: the sort indices are a permutation, so their argsort is unambiguous
        var inverse = engine.TensorArgsort(sorted.Indices, 1, false);
        var rank = engine.TensorTakeAlongDim(sortedRank, inverse, 1);

        long width = cols;
        var nanFlag = engine.TensorWhere(isNaN, Tensor<long>.CreateDefault(shape, 1L), new Tensor<long>(shape));
        var tier = descending ? nanFlag : engine.ScalarMinusTensor(1L, nanFlag);
        var columns = engine.TensorBroadcastTo(engine.TensorArange<int>(0, cols).Reshape(1, cols), shape).Cast<long>();
        var key = engine.TensorAdd(
            engine.TensorAdd(
                engine.TensorMultiplyScalar(rank.Cast<long>(), 2L * width),
                engine.TensorMultiplyScalar(tier, width)),
            columns);
        return engine.TensorArgsort(key, 1, false);
    }

    /// <summary>
    /// Returns the indices of the <paramref name="topK"/> largest scores, largest first, ties in index order
    /// and NaN last: the indices <c>OrderByDescending(score).Take(topK)</c> yields.
    /// </summary>
    public static int[] StableTopKDescending<T>(T[] scores, int topK)
    {
        if (scores is null)
        {
            throw new ArgumentNullException(nameof(scores));
        }

        int take = Math.Max(0, Math.Min(topK, scores.Length));
        if (take == 0)
        {
            return new int[0];
        }

        var order = StableDescendingOrder1D(new Tensor<T>(scores, new[] { scores.Length })).ToArray();
        var top = new int[take];
        Array.Copy(order, top, take);
        return top;
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
