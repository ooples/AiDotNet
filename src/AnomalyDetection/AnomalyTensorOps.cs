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
