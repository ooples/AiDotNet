using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection;

/// <summary>
/// Whole-array engine building blocks shared by the anomaly detectors (row broadcasting, column
/// statistics, sliding windows, row selection). Everything stays generic in <typeparamref name="T"/>.
/// </summary>
internal static class AnomalyTensorOps
{
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
    {
        var engine = Engine;
        int n = series.Length;
        var ones = Tensor<T>.CreateOnes(new[] { n });
        if (window <= 1)
        {
            mask = engine.Reshape(ones, new[] { n, 1 });
            return engine.Reshape(series, new[] { n, 1 });
        }

        var padding = Tensor<T>.CreateZeros(new[] { window - 1 });
        mask = engine.TensorUnfold(engine.TensorConcatenate(new[] { padding, ones }, 0), 0, window, 1);
        return engine.TensorUnfold(engine.TensorConcatenate(new[] { padding, series }, 0), 0, window, 1);
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
