using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection;

/// <summary>
/// Whole-array engine building blocks shared by the anomaly detectors (row broadcasting, column
/// statistics, sliding windows, row selection). Everything stays generic in <typeparamref name="T"/>.
/// </summary>
internal static class AnomalyTensorOps
{
    // Upper bound on the elements of one block of row pairs materialised by the pairwise helpers.
    private const int PairBlockElements = 1 << 22;

    private static IEngine Engine => AiDotNetEngine.Current;

    /// <summary>Copies a matrix into a [rows, columns] tensor.</summary>
    internal static Tensor<T> ToTensor<T>(Matrix<T> matrix) => Tensor<T>.FromMatrix(matrix);

    /// <summary>Repeats a length-n vector tensor as the n x width matrix whose row i is filled with v[i].</summary>
    internal static Tensor<T> BroadcastColumn<T>(Tensor<T> values, int width)
    {
        int n = values.Length;
        return Engine.TensorBroadcastTo(Engine.Reshape(values, new[] { n, 1 }), new[] { n, width });
    }

    /// <summary>Repeats a length-d vector tensor as the rows x d matrix whose every row is v.</summary>
    internal static Tensor<T> BroadcastRow<T>(Tensor<T> values, int rows)
    {
        int d = values.Length;
        return Engine.TensorBroadcastTo(Engine.Reshape(values, new[] { 1, d }), new[] { rows, d });
    }

    /// <summary>m - row, row subtracted from every row of m.</summary>
    internal static Tensor<T> SubtractRow<T>(Tensor<T> m, Tensor<T> row)
        => Engine.TensorSubtract(m, BroadcastRow(row, m.Shape[0]));

    /// <summary>m / row, every row of m divided element-wise by row.</summary>
    internal static Tensor<T> DivideRow<T>(Tensor<T> m, Tensor<T> row)
        => Engine.TensorDivide(m, BroadcastRow(row, m.Shape[0]));

    /// <summary>m * row, every row of m multiplied element-wise by row.</summary>
    internal static Tensor<T> MultiplyRow<T>(Tensor<T> m, Tensor<T> row)
        => Engine.TensorMultiply(m, BroadcastRow(row, m.Shape[0]));

    /// <summary>Row i of m divided by values[i].</summary>
    internal static Tensor<T> DivideColumn<T>(Tensor<T> m, Tensor<T> values)
        => Engine.TensorDivide(m, BroadcastColumn(values, m.Shape[1]));

    /// <summary>Row i of m multiplied by values[i].</summary>
    internal static Tensor<T> MultiplyColumn<T>(Tensor<T> m, Tensor<T> values)
        => Engine.TensorMultiply(m, BroadcastColumn(values, m.Shape[1]));

    /// <summary>
    /// The trailing windows of a 1-D series: row i holds series[i - window + 1 .. i], zero-padded where
    /// the window starts before the series. <paramref name="mask"/> is 1 at real entries, 0 at padding.
    /// </summary>
    internal static Tensor<T> TrailingWindows<T>(Tensor<T> series, int window, out Tensor<T> mask)
        => SlidingWindows(series, window - 1, 0, out mask);

    /// <summary>
    /// The sliding windows of a 1-D series of length n: row i holds series[i - before .. i + after],
    /// zero-padded where the window runs past either end. <paramref name="mask"/> ([n, before + after + 1])
    /// is 1 at real entries and 0 at padding, so its row sums are the truncated window lengths.
    /// </summary>
    internal static Tensor<T> SlidingWindows<T>(Tensor<T> series, int before, int after, out Tensor<T> mask)
    {
        var engine = Engine;
        int n = series.Length;
        int width = before + after + 1;
        var ones = Tensor<T>.CreateOnes(new[] { n });
        if (width == 1)
        {
            mask = engine.Reshape(ones, new[] { n, 1 });
            return engine.Reshape(series, new[] { n, 1 });
        }

        var parts = new List<Tensor<T>>(3);
        var maskParts = new List<Tensor<T>>(3);
        if (before > 0)
        {
            parts.Add(Tensor<T>.CreateZeros(new[] { before }));
            maskParts.Add(Tensor<T>.CreateZeros(new[] { before }));
        }

        parts.Add(series);
        maskParts.Add(ones);
        if (after > 0)
        {
            parts.Add(Tensor<T>.CreateZeros(new[] { after }));
            maskParts.Add(Tensor<T>.CreateZeros(new[] { after }));
        }

        mask = engine.TensorUnfold(engine.TensorConcatenate(maskParts.ToArray(), 0), 0, width, 1);
        return engine.TensorUnfold(engine.TensorConcatenate(parts.ToArray(), 0), 0, width, 1);
    }

    /// <summary>
    /// Exact squared Euclidean distances between every row of x ([n, d]) and every row of y ([m, d]),
    /// summed from the explicit differences (no ||x||^2 + ||y||^2 - 2xy cancellation). Row pairs are
    /// materialised in bounded blocks.
    /// </summary>
    internal static Tensor<T> PairwiseSquaredDistances<T>(Tensor<T> x, Tensor<T> y)
    {
        var engine = Engine;
        int n = x.Shape[0];
        int m = y.Shape[0];
        int d = x.Shape[1];
        if (n == 0 || m == 0 || d == 0)
        {
            return Tensor<T>.CreateZeros(new[] { n, m });
        }

        long perRow = (long)m * d;
        int rowsPerBlock = (int)Math.Max(1, Math.Min(n, PairBlockElements / perRow));
        if (rowsPerBlock >= n)
        {
            return engine.Reshape(RowPairSquaredDistances(x, y, n), new[] { n, m });
        }

        var result = Tensor<T>.CreateZeros(new[] { n, m });
        for (int start = 0; start < n; start += rowsPerBlock)
        {
            int count = Math.Min(rowsPerBlock, n - start);
            var block = engine.TensorSlice(x, new[] { start, 0 }, new[] { count, d });
            var values = engine.Reshape(RowPairSquaredDistances(block, y, count), new[] { count, m });
            result = engine.TensorSetSlice(result, values, new[] { start, 0 });
        }

        return result;
    }

    /// <summary>Exact Euclidean distances between every row of x and every row of y.</summary>
    internal static Tensor<T> PairwiseDistances<T>(Tensor<T> x, Tensor<T> y)
        => Engine.TensorSqrt(PairwiseSquaredDistances(x, y));

    /// <summary>Exact Euclidean distances between every pair of rows of a matrix.</summary>
    internal static Tensor<T> PairwiseDistances<T>(Matrix<T> x)
    {
        var t = ToTensor(x);
        return PairwiseDistances(t, t);
    }

    /// <summary>Exact squared Euclidean distances from one point to every row of data.</summary>
    internal static Tensor<T> SquaredDistancesToPoint<T>(Tensor<T> data, Tensor<T> point)
    {
        var engine = Engine;
        return engine.ReduceSum(engine.TensorSquare(SubtractRow(data, point)), new[] { 1 }, false);
    }

    // The [rows * m] flattened squared distances of every (block row, y row) pair.
    private static Tensor<T> RowPairSquaredDistances<T>(Tensor<T> block, Tensor<T> y, int rows)
    {
        var engine = Engine;
        int m = y.Shape[0];
        var a = engine.TensorRepeatInterleave(block, m, 0);
        var b = engine.TensorTile(y, new[] { rows, 1 });
        return engine.ReduceSum(engine.TensorSquare(engine.TensorSubtract(a, b)), new[] { 1 }, false);
    }

    /// <summary>Per-column mean and population (divide by n) standard deviation of a matrix.</summary>
    internal static (Vector<T> Means, Vector<T> Stds) ColumnMeanStd<T>(Matrix<T> x)
    {
        var (std, mean) = Engine.TensorStdMean(ToTensor(x), new[] { 0 }, 0, false);
        return (mean.ToVector(), std.ToVector());
    }

    /// <summary>
    /// Per row, the largest |x_ij - mean_j| / std_j over the columns whose std is non-zero, or 0 when
    /// there is none. A NaN ratio never wins the comparison.
    /// </summary>
    internal static Vector<T> MaxStandardizedDeviation<T>(Matrix<T> x, Vector<T> means, Vector<T> stds)
        => MaxScaledDeviation(x, means, stds, false, MathHelper.GetNumericOperations<T>().One);

    /// <summary>
    /// As <see cref="MaxStandardizedDeviation{T}(Matrix{T}, Vector{T}, Vector{T})"/> with the deviation
    /// multiplied by <paramref name="factor"/> first (when <paramref name="scaled"/>):
    /// |factor * (x_ij - center_j)| / scale_j.
    /// </summary>
    internal static Vector<T> MaxScaledDeviation<T>(Matrix<T> x, Vector<T> means, Vector<T> stds, bool scaled, T factor)
    {
        var engine = Engine;
        var numOps = MathHelper.GetNumericOperations<T>();
        var stdTensor = stds.AsTensor();
        var deviation = SubtractRow(ToTensor(x), means.AsTensor());
        if (scaled)
        {
            deviation = engine.TensorMultiplyScalar(deviation, factor);
        }

        var ratio = DivideRow(engine.TensorAbs(deviation), stdTensor);

        // Zero-std columns are skipped: x/0 is +inf or NaN, times 0 is NaN, and NaN entries become 0.
        ratio = MultiplyRow(ratio, engine.TensorNotEquals(stdTensor, numOps.Zero));
        ratio = engine.TensorMaskedFill(ratio, engine.TensorIsNan(ratio), numOps.Zero);
        return engine.ReduceMax(ratio, new[] { 1 }, false).ToVector();
    }

    /// <summary>A 1-D pattern repeated end to end and cut to length n (element i is pattern[i % period]).</summary>
    internal static Tensor<T> TileToLength<T>(Tensor<T> pattern, int n)
    {
        var engine = Engine;
        int period = pattern.Length;
        int repeats = (n + period - 1) / period;
        var tiled = repeats > 1 ? engine.TensorTile(pattern, new[] { repeats }) : pattern;
        return tiled.Length == n ? tiled : engine.TensorSlice(tiled, new[] { 0 }, new[] { n });
    }

    /// <summary>
    /// Per-phase means of a 1-D series: element s is the mean of series[i] over the i with i % period == s
    /// (0 for a phase with no samples).
    /// </summary>
    internal static Tensor<T> PhaseMeans<T>(Tensor<T> series, int period)
    {
        var engine = Engine;
        int n = series.Length;
        int rows = (n + period - 1) / period;
        int padding = rows * period - n;
        var values = series;
        var mask = Tensor<T>.CreateOnes(new[] { n });
        if (padding > 0)
        {
            values = engine.TensorConcatenate(new[] { series, Tensor<T>.CreateZeros(new[] { padding }) }, 0);
            mask = engine.TensorConcatenate(new[] { mask, Tensor<T>.CreateZeros(new[] { padding }) }, 0);
        }

        var sums = engine.ReduceSum(engine.Reshape(values, new[] { rows, period }), new[] { 0 }, false);
        var counts = engine.ReduceSum(engine.Reshape(mask, new[] { rows, period }), new[] { 0 }, false);
        var numOps = MathHelper.GetNumericOperations<T>();
        return engine.TensorDivide(sums, engine.TensorMax(counts, numOps.One));
    }

    /// <summary>The columns of a matrix as separate 1-D tensors.</summary>
    internal static Tensor<T>[] Columns<T>(Matrix<T> x)
    {
        var data = ToTensor(x);
        var columns = new Tensor<T>[x.Columns];
        for (int j = 0; j < columns.Length; j++)
        {
            columns[j] = Engine.TensorSliceAxis(data, 1, j);
        }

        return columns;
    }

    /// <summary>0, 1, ..., n - 1 as an int tensor.</summary>
    internal static Tensor<int> Range(int n) => Engine.TensorArange<int>(0, n, 1);

    /// <summary>The entries of an int tensor where a 0/1 mask equals <paramref name="flag"/> (1 or 0), in order.</summary>
    internal static Tensor<int> SelectWhere<T>(Tensor<int> items, Tensor<T> mask01, T flag)
        => Engine.TensorMaskedSelect(items, Engine.TensorEqScalar(mask01, flag));

    /// <summary>
    /// Assembles per-row values delivered in pieces: the row ids of all pieces together are a permutation
    /// of 0..n-1, and the result holds, at each row id, the value delivered for it.
    /// </summary>
    internal static Tensor<T> AssembleByRow<T>(List<Tensor<int>> rowPieces, List<Tensor<T>> valuePieces)
    {
        var engine = Engine;
        var rows = rowPieces.Count == 1 ? rowPieces[0] : engine.TensorConcatenate(rowPieces.ToArray(), 0);
        var values = valuePieces.Count == 1 ? valuePieces[0] : engine.TensorConcatenate(valuePieces.ToArray(), 0);
        return engine.TensorGather(values, engine.TensorArgsort(rows, 0, false), 0);
    }

    /// <summary>The projections rows · direction of every row of a [n, d] tensor onto a length-d vector.</summary>
    internal static Tensor<T> Project<T>(Tensor<T> rows, Vector<T> direction)
    {
        var engine = Engine;
        int n = rows.Shape[0];
        var column = Tensor<T>.FromVector(direction, new[] { direction.Length, 1 });
        return engine.Reshape(engine.TensorMatMul(rows, column), new[] { n });
    }

    /// <summary>Inlier/outlier labels: -1 where score &gt; threshold, 1 otherwise (NaN scores are inliers).</summary>
    internal static Vector<T> LabelsFromThreshold<T>(Vector<T> scores, T threshold)
    {
        if (scores.Length == 0)
        {
            return new Vector<T>(0);
        }

        var engine = Engine;
        var numOps = MathHelper.GetNumericOperations<T>();
        var above = engine.TensorGreaterThan(scores.AsTensor(), threshold);
        return engine.TensorAddScalar(engine.TensorMultiplyScalar(above, numOps.FromDouble(-2)), numOps.One).ToVector();
    }

    /// <summary>
    /// The number of real entries in each window of <see cref="SlidingWindows"/>: for row i of a length-n
    /// series, min(n, i + after + 1) - max(0, i - before).
    /// </summary>
    internal static int[] SlidingWindowLengths(int n, int before, int after)
    {
        var lengths = new int[n];
        for (int i = 0; i < n; i++)
        {
            lengths[i] = Math.Min(n, i + after + 1) - Math.Max(0, i - before);
        }

        return lengths;
    }

    /// <summary>
    /// The median of the real entries of each row of a padded window matrix ([n, w], mask 1 at real
    /// entries); row i has lengths[i] real entries. Even counts average the two middle values.
    /// </summary>
    internal static Tensor<T> MaskedRowMedians<T>(Tensor<T> windows, Tensor<T> mask, int[] lengths)
    {
        var engine = Engine;
        var numOps = MathHelper.GetNumericOperations<T>();
        int n = windows.Shape[0];

        // Padding sorts after every real value, so the real entries occupy the first lengths[i] slots.
        var padded = engine.TensorMaskedFill(windows, engine.TensorEqScalar(mask, numOps.Zero), numOps.MaxValue);
        var (sorted, _) = engine.TensorSort(padded, 1, false);

        var lower = new int[n];
        var upper = new int[n];
        for (int i = 0; i < n; i++)
        {
            lower[i] = (lengths[i] - 1) / 2;
            upper[i] = lengths[i] / 2;
        }

        var low = engine.TensorTakeAlongDim(sorted, Tensor<int>.FromVector(Vector<int>.Wrap(lower), new[] { n, 1 }), 1);
        var high = engine.TensorTakeAlongDim(sorted, Tensor<int>.FromVector(Vector<int>.Wrap(upper), new[] { n, 1 }), 1);

        // For an odd count both picks are the middle value and (v + v) / 2 == v exactly.
        var median = engine.TensorMultiplyScalar(engine.TensorAdd(low, high), numOps.FromDouble(0.5));
        return engine.Reshape(median, new[] { n });
    }

    /// <summary>
    /// Per-column medians of a [n, d] tensor (n &gt; 0); an even n averages the two middle values.
    /// </summary>
    internal static Tensor<T> ColumnMedians<T>(Tensor<T> data)
    {
        var engine = Engine;
        var numOps = MathHelper.GetNumericOperations<T>();
        int n = data.Shape[0];
        var (sorted, _) = engine.TensorSort(data, 0, false);
        var low = engine.TensorSliceAxis(sorted, 0, (n - 1) / 2);
        var high = engine.TensorSliceAxis(sorted, 0, n / 2);

        // For an odd n both picks are the middle value and (v + v) * 0.5 == v exactly.
        return engine.TensorMultiplyScalar(engine.TensorAdd(low, high), numOps.FromDouble(0.5));
    }

    /// <summary>
    /// The given percentile (0-100) of every column of an already column-sorted [n, d] tensor, by linear
    /// interpolation between the order statistics around index p / 100 * (n - 1).
    /// </summary>
    internal static Tensor<T> SortedColumnPercentile<T>(Tensor<T> sortedColumns, double percentile)
    {
        var engine = Engine;
        var numOps = MathHelper.GetNumericOperations<T>();
        int n = sortedColumns.Shape[0];
        double index = (percentile / 100.0) * (n - 1);
        int lower = (int)Math.Floor(index);
        int upper = (int)Math.Ceiling(index);

        if (lower == upper || upper >= n)
        {
            return engine.TensorSliceAxis(sortedColumns, 0, Math.Min(lower, n - 1));
        }

        T fraction = numOps.FromDouble(index - lower);
        var low = engine.TensorSliceAxis(sortedColumns, 0, lower);
        var high = engine.TensorSliceAxis(sortedColumns, 0, upper);
        return engine.TensorAdd(low, engine.TensorMultiplyScalar(engine.TensorSubtract(high, low), fraction));
    }

    /// <summary>
    /// The quantile q of every column of an already column-sorted [n, d] tensor, as
    /// StatisticsHelper.CalculateQuantile computes it: position = (n - 1) * q,
    /// sorted[i] * (1 - f) + sorted[i + 1] * f with i = floor(position), f = position - i.
    /// </summary>
    internal static Tensor<T> SortedColumnQuantile<T>(Tensor<T> sortedColumns, double quantile)
    {
        var engine = Engine;
        var numOps = MathHelper.GetNumericOperations<T>();
        int n = sortedColumns.Shape[0];
        T position = numOps.Multiply(numOps.FromDouble(n - 1), numOps.FromDouble(quantile));
        int index = Convert.ToInt32(numOps.Floor(position));
        T fraction = numOps.Subtract(position, numOps.FromDouble(index));

        var low = engine.TensorSliceAxis(sortedColumns, 0, index);
        if (index + 1 >= n)
        {
            return low;
        }

        var high = engine.TensorSliceAxis(sortedColumns, 0, index + 1);
        return engine.TensorAdd(
            engine.TensorMultiplyScalar(low, numOps.Subtract(numOps.One, fraction)),
            engine.TensorMultiplyScalar(high, fraction));
    }

    /// <summary>
    /// Per row, the largest distance outside the per-column band [low_j, high_j], divided by scale_j:
    /// (low_j - x_ij) / scale_j below the band, (x_ij - high_j) / scale_j above it, 0 inside (and for NaN).
    /// Columns whose scale is zero are skipped.
    /// </summary>
    internal static Vector<T> MaxOutsideBand<T>(Matrix<T> x, Tensor<T> low, Tensor<T> high, Tensor<T> scale)
    {
        var engine = Engine;
        var numOps = MathHelper.GetNumericOperations<T>();
        var data = ToTensor(x);
        int rows = x.Rows;
        var below = engine.TensorSubtract(BroadcastRow(low, rows), data);
        var above = SubtractRow(data, high);
        var outside = engine.TensorMax(engine.TensorMax(below, above), numOps.Zero);
        var score = DivideRow(outside, scale);

        // Zero-scale columns are skipped: their ratio is +inf or NaN, times 0 is NaN, and NaN becomes 0.
        score = MultiplyRow(score, engine.TensorNotEquals(scale, numOps.Zero));
        score = engine.TensorMaskedFill(score, engine.TensorIsNan(score), numOps.Zero);
        return engine.ReduceMax(score, new[] { 1 }, false).ToVector();
    }

    /// <summary>Per-column median absolute deviation from the given per-column centers.</summary>
    internal static Tensor<T> ColumnMads<T>(Tensor<T> data, Tensor<T> centers)
        => ColumnMedians(Engine.TensorAbs(SubtractRow(data, centers)));

    /// <summary>The scatter matrix centered^T centered of already centered rows, divided by divisor.</summary>
    internal static Tensor<T> Covariance<T>(Tensor<T> centered, T divisor)
    {
        var engine = Engine;
        return engine.TensorDivideScalar(engine.TensorMatMul(engine.TensorTranspose(centered), centered), divisor);
    }

    /// <summary>
    /// Squared Mahalanobis distance of every already centered row: d_i' * inverse * d_i.
    /// </summary>
    internal static Tensor<T> MahalanobisSquared<T>(Tensor<T> centered, Tensor<T> inverse)
    {
        var engine = Engine;
        var projected = engine.TensorMatMul(centered, engine.TensorTranspose(inverse));
        return engine.ReduceSum(engine.TensorMultiply(centered, projected), new[] { 1 }, false);
    }

    /// <summary>The p x p identity matrix times a scalar.</summary>
    internal static Tensor<T> ScaledIdentity<T>(int p, T scale)
        => Engine.TensorMultiplyScalar(Engine.TensorEye<T>(p), scale);

    /// <summary>
    /// The smallest and largest value of a vector, NaN entries ignored; (MaxValue, MinValue) when there is
    /// no non-NaN entry, as a running min/max started from those bounds gives.
    /// </summary>
    internal static (T Min, T Max) MinMaxIgnoringNaN<T>(Vector<T> values)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        if (values.Length == 0)
        {
            return (numOps.MaxValue, numOps.MinValue);
        }

        var engine = Engine;
        var t = values.AsTensor();
        var nan = engine.TensorIsNan(t);
        return (engine.TensorMinValue(engine.TensorMaskedFill(t, nan, numOps.MaxValue)),
            engine.TensorMaxValue(engine.TensorMaskedFill(t, nan, numOps.MinValue)));
    }

    /// <summary>
    /// The element-wise maximum over a list of equal-length vectors, NaN entries ignored; MinValue where
    /// every entry is NaN or the list is empty.
    /// </summary>
    internal static Vector<T> ElementwiseMaxIgnoringNaN<T>(IReadOnlyList<Vector<T>> vectors, int length)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        if (vectors.Count == 0)
        {
            return Engine.Fill(length, numOps.MinValue);
        }

        var engine = Engine;
        var stacked = engine.TensorStack(vectors.Select(v => v.AsTensor()).ToArray(), 0);
        var comparable = engine.TensorMaskedFill(stacked, engine.TensorIsNan(stacked), numOps.MinValue);
        return engine.ReduceMax(comparable, new[] { 0 }, false).ToVector();
    }

    /// <summary>
    /// Min-max normalization against a training range, clamped to [0, 1]; every score maps to 0.5 when the
    /// range is not above 1e-10.
    /// </summary>
    internal static Vector<T> NormalizeToTrainingRange<T>(Vector<T> scores, T trainMin, T trainMax)
    {
        var engine = Engine;
        var numOps = MathHelper.GetNumericOperations<T>();
        T range = numOps.Subtract(trainMax, trainMin);
        if (!numOps.GreaterThan(range, numOps.FromDouble(1e-10)))
        {
            return engine.Fill(scores.Length, numOps.FromDouble(0.5));
        }

        if (scores.Length == 0)
        {
            return new Vector<T>(0);
        }

        var normalized = engine.TensorDivideScalar(engine.TensorSubtractScalar(scores.AsTensor(), trainMin), range);
        return engine.TensorClamp(normalized, numOps.Zero, numOps.One).ToVector();
    }

    /// <summary>The element-wise mean over a non-empty list of equal-length vectors (summed in list order).</summary>
    internal static Vector<T> ElementwiseMean<T>(IReadOnlyList<Vector<T>> vectors)
    {
        var engine = Engine;
        var numOps = MathHelper.GetNumericOperations<T>();
        var stacked = engine.TensorStack(vectors.Select(v => v.AsTensor()).ToArray(), 0);
        return engine.TensorDivideScalar(engine.ReduceSum(stacked, new[] { 0 }, false), numOps.FromDouble(vectors.Count)).ToVector();
    }

    /// <summary>
    /// Pearson correlation of each row of a with the same row of b ([m, r] each): cov / sqrt(varA * varB),
    /// or 0 when that denominator is not above 1e-10 or r &lt; 2.
    /// </summary>
    internal static Tensor<T> RowPearson<T>(Tensor<T> a, Tensor<T> b)
    {
        var engine = Engine;
        var numOps = MathHelper.GetNumericOperations<T>();
        int m = a.Shape[0];
        int r = a.Shape[1];
        if (r < 2)
        {
            return Tensor<T>.CreateZeros(new[] { m });
        }

        var da = engine.TensorSubtract(a, BroadcastColumn(engine.ReduceMean(a, new[] { 1 }, false), r));
        var db = engine.TensorSubtract(b, BroadcastColumn(engine.ReduceMean(b, new[] { 1 }, false), r));
        var cov = engine.ReduceSum(engine.TensorMultiply(da, db), new[] { 1 }, false);
        var varA = engine.ReduceSum(engine.TensorSquare(da), new[] { 1 }, false);
        var varB = engine.ReduceSum(engine.TensorSquare(db), new[] { 1 }, false);
        var denom = engine.TensorSqrt(engine.TensorMultiply(varA, varB));
        var usable = engine.TensorGreaterThan(denom, numOps.FromDouble(1e-10));
        return engine.TensorWhere(usable, engine.TensorDivide(cov, denom), Tensor<T>.CreateZeros(new[] { m }));
    }

    /// <summary>
    /// For each row of values ([m, e]) a 0/1 mask of the <paramref name="count"/> largest entries, ties
    /// going to the lower column index (the order a stable descending sort gives).
    /// </summary>
    internal static Tensor<T> TopCountMask<T>(Tensor<T> values, int count)
    {
        var engine = Engine;
        var numOps = MathHelper.GetNumericOperations<T>();
        int m = values.Shape[0];
        int e = values.Shape[1];
        var shape = new[] { m, e, e };

        // own[i, a, b] = values[i, a]; other[i, a, b] = values[i, b]
        var own = engine.TensorBroadcastTo(engine.Reshape(values, new[] { m, e, 1 }), shape);
        var other = engine.TensorBroadcastTo(engine.Reshape(values, new[] { m, 1, e }), shape);

        // rank(a) = #{b : v_b > v_a} + #{b < a : v_b == v_a}
        var earlier = engine.TensorBroadcastTo(
            engine.Reshape(engine.TensorTril(Tensor<T>.CreateOnes(new[] { e, e }), -1), new[] { 1, e, e }), shape);
        var beats = engine.TensorAdd(
            engine.TensorGreaterThan(other, own),
            engine.TensorMultiply(engine.TensorEquals(other, own), earlier));
        var rank = engine.ReduceSum(beats, new[] { 2 }, false);
        return engine.TensorLessThan(rank, numOps.FromDouble(count));
    }

    /// <summary>
    /// The k nearest reference rows of every query row: their Euclidean distances ([m, k], ascending) and,
    /// in <paramref name="indices"/>, their row indices. With <paramref name="excludeSelf"/> (query and
    /// reference are the same rows) each row's own entry is pushed past every real distance first.
    /// </summary>
    internal static Tensor<T> NearestNeighbors<T>(Tensor<T> query, Tensor<T> reference, int k, bool excludeSelf,
        out Tensor<int> indices)
        => NearestNeighbors(query, reference, k, excludeSelf, false, out indices);

    /// <summary>
    /// As <see cref="NearestNeighbors{T}(Tensor{T}, Tensor{T}, int, bool, out Tensor{int})"/>, returning
    /// squared distances when <paramref name="squared"/> is set (the neighbour order is the same).
    /// </summary>
    internal static Tensor<T> NearestNeighbors<T>(Tensor<T> query, Tensor<T> reference, int k, bool excludeSelf,
        bool squared, out Tensor<int> indices)
    {
        var engine = Engine;
        var numOps = MathHelper.GetNumericOperations<T>();
        int m = query.Shape[0];
        var distances = squared ? PairwiseSquaredDistances(query, reference) : PairwiseDistances(query, reference);
        if (excludeSelf)
        {
            var self = engine.TensorEqScalar(engine.TensorEye<T>(m), numOps.One);
            distances = engine.TensorMaskedFill(distances, self, numOps.MaxValue);
        }

        var (sorted, order) = engine.TensorSort(distances, 1, false);
        indices = engine.TensorSlice(order, new[] { 0, 0 }, new[] { m, k });
        return engine.TensorSlice(sorted, new[] { 0, 0 }, new[] { m, k });
    }

    /// <summary>values[indices[i, j]] for every entry of a [m, k] index tensor, as a [m, k] tensor.</summary>
    internal static Tensor<T> GatherByIndex<T>(Tensor<T> values, Tensor<int> indices)
    {
        var engine = Engine;
        int m = indices.Shape[0];
        int k = indices.Shape[1];
        if (m == 0 || k == 0)
        {
            return Tensor<T>.CreateZeros(new[] { m, k });
        }

        var flat = engine.Reshape(indices, new[] { m * k });
        return engine.Reshape(engine.TensorGather(values, flat, 0), new[] { m, k });
    }

    /// <summary>The rows of a [n, k] int tensor as separate arrays.</summary>
    internal static int[][] ToJagged(Tensor<int> indices)
    {
        int n = indices.Shape[0];
        int k = indices.Shape[1];
        var flat = indices.ToArray();
        var rows = new int[n][];
        for (int i = 0; i < n; i++)
        {
            rows[i] = new int[k];
            Array.Copy(flat, i * k, rows[i], 0, k);
        }

        return rows;
    }

    /// <summary>Equal-length int rows as one [n, k] int tensor.</summary>
    internal static Tensor<int> FromJagged(int[][] rows, int k)
    {
        int n = rows.Length;
        var flat = new int[n * k];
        for (int i = 0; i < n; i++)
        {
            Array.Copy(rows[i], 0, flat, i * k, k);
        }

        return Tensor<int>.FromVector(Vector<int>.Wrap(flat), new[] { n, k });
    }

    /// <summary>The rows of a [n, k] tensor as separate vectors.</summary>
    internal static Vector<T>[] RowVectors<T>(Tensor<T> m)
    {
        int n = m.Shape[0];
        var rows = new Vector<T>[n];
        for (int i = 0; i < n; i++)
        {
            rows[i] = Engine.TensorSliceAxis(m, 0, i).ToVector();
        }

        return rows;
    }

    /// <summary>Equal-length vectors as the rows of one [n, k] tensor.</summary>
    internal static Tensor<T> StackRows<T>(Vector<T>[] rows)
        => Engine.TensorStack(rows.Select(r => r.AsTensor()).ToArray(), 0);

    /// <summary>The positions of the non-zero entries of a 1-D tensor, ascending.</summary>
    internal static int[] NonzeroIndices<T>(Tensor<T> values)
    {
        if (values.Length == 0)
        {
            return Array.Empty<int>();
        }

        return Engine.TensorNonzero(values).ToArray();
    }

    /// <summary>The rows of m at the given indices, in order.</summary>
    internal static Tensor<T> SelectRows<T>(Tensor<T> m, int[] rows)
    {
        if (rows.Length == 0)
        {
            return Tensor<T>.CreateZeros(new[] { 0, m.Shape[1] });
        }

        return Engine.TensorIndexSelect(m, Vector<int>.Wrap(rows).AsTensor(), 0);
    }

    /// <summary>The columns of m at the given indices, in order.</summary>
    internal static Tensor<T> SelectColumns<T>(Tensor<T> m, int[] columns)
    {
        if (columns.Length == 0)
        {
            return Tensor<T>.CreateZeros(new[] { m.Shape[0], 0 });
        }

        return Engine.TensorIndexSelect(m, Vector<int>.Wrap(columns).AsTensor(), 1);
    }

    /// <summary>The elements of v at the given indices, in order.</summary>
    internal static Vector<T> SelectElements<T>(Vector<T> v, int[] indices)
    {
        if (indices.Length == 0)
        {
            return new Vector<T>(0);
        }

        return Engine.TensorGather(v.AsTensor(), Vector<int>.Wrap(indices).AsTensor(), 0).ToVector();
    }
}
