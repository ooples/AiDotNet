using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Augmentation.Tabular;

/// <summary>
/// Whole-array engine helpers shared by the SMOTE-family tabular augmenters.
/// </summary>
/// <typeparam name="T">The numeric type of the data.</typeparam>
internal static class TabularEngineOps<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Upper bound on the elements of one [block, rows, columns] difference tile in
    /// <see cref="PairwiseDistances"/>, so peak memory stays proportional to the [rows, rows] result.
    /// </summary>
    private const int PairwiseTileElements = 1 << 22;

    /// <summary>
    /// Euclidean distance between every pair of rows, as a [rows, rows] tensor of doubles.
    /// </summary>
    /// <remarks>
    /// Feature values are converted to double before subtraction and the distances stay in double, so
    /// squaring cannot overflow an integral or float <typeparamref name="T"/> and neighbor ranking sees
    /// the same distances as the double computation. Rows are processed in bounded tiles.
    /// </remarks>
    internal static Tensor<double> PairwiseDistances(Matrix<T> data)
    {
        var engine = AiDotNetEngine.Current;
        int rows = data.Rows;
        int cols = data.Columns;
        if (rows == 0)
        {
            return new Tensor<double>([0, 0]);
        }

        var values = Tensor<T>.FromMatrix(data).Cast<double>();
        var right = engine.Reshape(values, [1, rows, cols]);
        long perRow = Math.Max(1L, (long)rows * cols);
        int block = (int)Math.Max(1L, Math.Min(rows, PairwiseTileElements / perRow));
        var tiles = new List<Tensor<double>>();
        for (int start = 0; start < rows; start += block)
        {
            int count = Math.Min(block, rows - start);
            var left = engine.Reshape(engine.TensorNarrow(values, 0, start, count), [count, 1, cols]);
            int[] tileShape = [count, rows, cols];
            var diff = engine.TensorSubtract(
                engine.TensorBroadcastTo(left, tileShape), engine.TensorBroadcastTo(right, tileShape));
            tiles.Add(engine.ReduceSum(engine.TensorMultiply(diff, diff), [2], false));
        }

        var squared = tiles.Count == 1 ? tiles[0] : engine.TensorConcatenate(tiles.ToArray(), 0);
        return engine.TensorSqrt(squared);
    }

    /// <summary>
    /// Synthetic rows from[i] + gaps[i] * (to[i] - from[i]), followed by zero rows up to
    /// <paramref name="totalRows"/>; <paramref name="gaps"/> are the raw generator draws.
    /// </summary>
    internal static Matrix<T> InterpolateRows(Matrix<T> data, int[] from, int[] to, double[] gaps, int totalRows)
    {
        var engine = AiDotNetEngine.Current;
        int cols = data.Columns;
        int count = from.Length;
        if (count == 0)
        {
            return new Matrix<T>(totalRows, cols);
        }

        // Evaluated in double and converted to T once at the end, as a per-feature double computation
        // would be: the gaps lie in [0, 1), so converting them to an integral T first zeroes every step.
        var values = Tensor<T>.FromMatrix(data).Cast<double>();
        var start = engine.TensorIndexSelect(values, new Tensor<int>(from, [count]), 0);
        var end = engine.TensorIndexSelect(values, new Tensor<int>(to, [count]), 0);
        var step = engine.TensorMultiply(
            engine.TensorBroadcastTo(new Tensor<double>(gaps, [count, 1]), [count, cols]),
            engine.TensorSubtract(end, start));
        var synthetic = engine.TensorAdd(start, step).Cast<T>();
        if (count < totalRows)
        {
            synthetic = engine.TensorConcatenate([synthetic, new Tensor<T>([totalRows - count, cols])], 0);
        }

        return synthetic.ToMatrix();
    }

    /// <summary>
    /// Positions of <paramref name="values"/> in stable sorted order (ties keep their original order),
    /// ascending or descending.
    /// </summary>
    /// <summary>
    /// Positions of <paramref name="values"/> in stable sorted order (ties keep their original order),
    /// ascending or descending.
    /// </summary>
    /// <remarks>
    /// The engine sort is not guaranteed stable, so position j gets the unique key
    /// (number of values sorting strictly before v[j]) * count + j, found with one sort and a binary
    /// search; sorting those keys gives the stable order in O(count log count).
    /// </remarks>
    internal static int[] StableOrder<TValue>(Tensor<TValue> values, bool descending)
    {
        int count = values.Length;
        if (count == 0)
        {
            return [];
        }

        var engine = AiDotNetEngine.Current;
        var flat = engine.Reshape(values.Contiguous(), [count]);
        var sorted = engine.TensorSort(flat, 0, false).Values;

        // ascending: #{m : v[m] < v[j]}; descending: #{m : v[m] > v[j]} = count - #{m : v[m] <= v[j]}
        var before = descending
            ? engine.ScalarMinusTensor(count, engine.TensorSearchSorted(sorted, flat, true))
            : engine.TensorSearchSorted(sorted, flat, false);
        var position = engine.TensorArange<long>(0, count, 1);
        var key = engine.TensorAdd(engine.TensorMultiplyScalar(before.Cast<long>(), (long)count), position);
        return engine.TensorArgsort(key, 0, false).ToArray();
    }
    /// tensor, excluding the sample itself, closest first (ties by row order).
    /// </summary>
    internal static int[] NearestNeighbors(Tensor<double> distances, int sampleIdx, int k)
    {
        var engine = AiDotNetEngine.Current;
        int rows = distances.Shape[0];
        var others = Enumerable.Range(0, rows).Where(i => i != sampleIdx).ToArray();
        if (others.Length == 0 || k <= 0)
        {
            return [];
        }

        var row = engine.TensorSliceAxis(distances, 0, sampleIdx).Contiguous();
        var candidates = engine.TensorIndexSelect(row, new Tensor<int>(others, [others.Length]), 0);
        var order = StableOrder(candidates, descending: false);
        int take = Math.Min(k, order.Length);
        var nearest = new int[take];
        for (int i = 0; i < take; i++)
        {
            nearest[i] = others[order[i]];
        }

        return nearest;
    }

    /// <summary>
    /// Stacks the rows of two matrices (row counts may be zero).
    /// </summary>
    internal static Matrix<T> ConcatenateRows(Matrix<T> first, Matrix<T> second)
    {
        if (second.Rows == 0)
        {
            return first.Clone();
        }

        if (first.Rows == 0)
        {
            return second.Clone();
        }

        return AiDotNetEngine.Current.TensorConcatenate(
            [Tensor<T>.FromMatrix(first), Tensor<T>.FromMatrix(second)], 0).ToMatrix();
    }

    /// <summary>
    /// First <paramref name="firstCount"/> entries of <paramref name="first"/> followed by the
    /// first <paramref name="secondCount"/> entries of <paramref name="second"/>.
    /// </summary>
    internal static Vector<T> ConcatenateLabels(Vector<T> first, int firstCount, Vector<T> second, int secondCount)
    {
        var engine = AiDotNetEngine.Current;
        var parts = new List<Tensor<T>>();
        if (firstCount > 0)
        {
            parts.Add(engine.TensorSlice(first.AsTensor(), [0], [firstCount]));
        }

        if (secondCount > 0)
        {
            parts.Add(engine.TensorSlice(second.AsTensor(), [0], [secondCount]));
        }

        return parts.Count == 0
            ? new Vector<T>(0)
            : engine.TensorConcatenate(parts.ToArray(), 0).ToVector();
    }

    /// <summary>
    /// A vector of <paramref name="length"/> copies of <paramref name="value"/>.
    /// </summary>
    internal static Vector<T> Filled(int length, T value)
    {
        var filled = new Tensor<T>([length]);
        AiDotNetEngine.Current.TensorFill(filled, value);
        return filled.ToVector();
    }

    /// <summary>
    /// Rows of <paramref name="data"/> and entries of <paramref name="labels"/> at the given indices.
    /// </summary>
    internal static (Matrix<T> Data, Vector<T> Labels) SelectRows(Matrix<T> data, Vector<T> labels, int[] indices)
    {
        if (indices.Length == 0)
        {
            return (new Matrix<T>(0, data.Columns), new Vector<T>(0));
        }

        var engine = AiDotNetEngine.Current;
        var order = new Tensor<int>(indices, [indices.Length]);
        var selectedData = engine.TensorIndexSelect(Tensor<T>.FromMatrix(data), order, 0).ToMatrix();
        var selectedLabels = engine.TensorIndexSelect(labels.AsTensor(), order, 0).ToVector();
        return (selectedData, selectedLabels);
    }

    /// <summary>
    /// Ascending indices of the labels equal to <paramref name="target"/>.
    /// </summary>
    internal static int[] IndicesEqualTo(Vector<T> labels, T target)
    {
        if (labels.Length == 0)
        {
            return [];
        }

        var engine = AiDotNetEngine.Current;
        var matches = engine.TensorEquals(labels.AsTensor(), target);
        return engine.TensorNonzero(matches).ToArray();
    }

    /// <summary>
    /// Distinct labels in first-occurrence order, and for each label its index in that list.
    /// </summary>
    internal static int[] ClassIndices(Vector<T> labels, out List<T> classLabels, out int[] classCounts)
    {
        int n = labels.Length;
        if (n == 0)
        {
            classLabels = new List<T>();
            classCounts = [];
            return [];
        }

        var engine = AiDotNetEngine.Current;
        var values = labels.AsTensor();
        var distinct = engine.TensorUnique(values, false);
        int classes = distinct.Length;
        classLabels = new List<T>(distinct.ToArray());

        // membership[i, c] = 1 when label i equals distinct class c (exactly one per row)
        int[] shape = [n, classes];
        var membership = engine.TensorEquals(
            engine.TensorBroadcastTo(engine.Reshape(values, [n, 1]), shape).Contiguous(),
            engine.TensorBroadcastTo(engine.Reshape(distinct, [1, classes]), shape).Contiguous());
        classCounts = engine.ReduceSum(membership.Cast<int>(), [0], false).ToArray();
        return engine.TensorArgMax(membership, 1).ToArray();
    }

    /// <summary>
    /// Index of the first class label equal to <paramref name="label"/>, or -1 when absent.
    /// </summary>
    internal static int IndexOf(List<T> classLabels, T label)
    {
        if (classLabels.Count == 0)
        {
            return -1;
        }

        var engine = AiDotNetEngine.Current;
        var matches = engine.TensorEquals(new Tensor<T>(classLabels.ToArray(), [classLabels.Count]), label);
        var hits = engine.TensorNonzero(matches).ToArray();
        return hits.Length == 0 ? -1 : hits[0];
    }
}
