namespace AiDotNet.CausalDiscovery;

/// <summary>
/// Engine-op helpers shared by the causal discovery algorithms: dense [rows x cols] weight matrices built
/// from per-edge statistics with one scatter, and scalar tests on tensors without per-element access.
/// </summary>
internal static class CausalTensorHelpers
{
    /// <summary>
    /// A zero matrix holding <c>weights[p]</c> at the row-major position <c>p</c> of each entry.
    /// </summary>
    internal static Matrix<T> FromStatistics<T>(int rows, int cols, Dictionary<int, double> weights)
    {
        var flat = new Tensor<T>(new[] { rows * cols });
        if (weights.Count > 0)
        {
            var positions = new List<int>(weights.Keys);
            var values = new List<double>(positions.Count);
            values.AddRange(positions.Select(p => weights[p]));
            flat = AiDotNetEngine.Current.TensorScatter(flat,
                new Tensor<int>(positions.ToArray(), new[] { positions.Count }),
                Vector<double>.FromList(values).AsTensor().Cast<T>(), 0);
        }

        return flat.Reshape(new[] { rows, cols }).ToMatrix();
    }

    /// <summary>
    /// A zero matrix holding <c>weights[p]</c> at the row-major position <c>p</c> of each entry.
    /// </summary>
    internal static Matrix<T> FromValues<T>(int rows, int cols, Dictionary<int, T> weights)
    {
        var flat = new Tensor<T>(new[] { rows * cols });
        if (weights.Count > 0)
        {
            var positions = new List<int>(weights.Keys);
            var values = new List<T>(positions.Count);
            values.AddRange(positions.Select(p => weights[p]));
            flat = AiDotNetEngine.Current.TensorScatter(flat,
                new Tensor<int>(positions.ToArray(), new[] { positions.Count }),
                Vector<T>.FromList(values).AsTensor(), 0);
        }

        return flat.Reshape(new[] { rows, cols }).ToMatrix();
    }

    /// <summary>
    /// Overwrites <paramref name="destination"/> ([rows x cols]) with <c>weights[p]</c> at each row-major position <c>p</c>.
    /// </summary>
    internal static Tensor<T> OverwriteStatistics<T>(Tensor<T> destination, Dictionary<int, double> weights)
    {
        if (weights.Count == 0)
            return destination;

        int rows = destination.Shape[0], cols = destination.Shape[1];
        var positions = new List<int>(weights.Keys);
        var values = new List<double>(positions.Count);
        values.AddRange(positions.Select(p => weights[p]));
        return AiDotNetEngine.Current.TensorScatter(destination.Reshape(new[] { rows * cols }),
            new Tensor<int>(positions.ToArray(), new[] { positions.Count }),
            Vector<double>.FromList(values).AsTensor().Cast<T>(), 0).Reshape(new[] { rows, cols });
    }

    /// <summary>
    /// Indices that sort the 1-D <paramref name="keys"/> (which must not contain NaN), with equal keys kept
    /// in ascending index order - the order a stable sort, or a comparison that breaks ties by index, gives.
    /// </summary>
    internal static int[] StableArgsort<T>(Tensor<T> keys, bool descending)
    {
        var engine = AiDotNetEngine.Current;
        int k = keys.Length;
        if (k == 0)
            return Array.Empty<int>();

        int[] order = engine.TensorArgsort(keys, -1, descending).ToArray();
        if (k == 1)
            return order;

        // Any sort leaves equal keys adjacent; re-order each run of equal keys by index.
        var sorted = engine.TensorIndexSelect(keys, new Tensor<int>((int[])order.Clone(), new[] { k }), 0);
        var equalToNext = engine.TensorEquals(
            engine.TensorSlice(sorted, new[] { 1 }, new[] { k - 1 }), engine.TensorSlice(sorted, new[] { 0 }, new[] { k - 1 }));
        int[] ties = engine.TensorNonzero(equalToNext).ToArray();
        for (int t = 0; t < ties.Length; t++)
        {
            int start = ties[t], end = start + 1;
            while (t + 1 < ties.Length && ties[t + 1] == end)
            {
                t++;
                end++;
            }

            Array.Sort(order, start, end - start + 1);
        }

        return order;
    }

    /// <summary>
    /// True when any entry of <paramref name="values"/> is below <paramref name="bound"/> (NaN never is).
    /// </summary>
    internal static bool AnyBelow<T>(Tensor<T> values, T bound)
    {
        var engine = AiDotNetEngine.Current;
        return engine.TensorCountNonzero(engine.TensorLessThan(values, bound)) > 0;
    }

    /// <summary>
    /// True when any entry of <paramref name="values"/> is above <paramref name="bound"/> (NaN never is).
    /// </summary>
    internal static bool AnyAbove<T>(Tensor<T> values, T bound)
    {
        var engine = AiDotNetEngine.Current;
        return engine.TensorCountNonzero(engine.TensorGreaterThan(values, bound)) > 0;
    }

    /// <summary>
    /// Keeps <paramref name="source"/> ([rows x cols]) at the given row-major positions and zeroes the rest.
    /// </summary>
    internal static Matrix<T> KeepPositions<T>(Tensor<T> source, IEnumerable<int> positions)
    {
        var engine = AiDotNetEngine.Current;
        int rows = source.Shape[0], cols = source.Shape[1];
        var flatSource = source.Reshape(new[] { rows * cols });
        var flat = new Tensor<T>(new[] { rows * cols });
        var distinct = new SortedSet<int>(positions);
        if (distinct.Count > 0)
        {
            var index = new Tensor<int>(distinct.ToArray(), new[] { distinct.Count });
            flat = engine.TensorScatter(flat, index, engine.TensorIndexSelect(flatSource, index, 0), 0);
        }

        return flat.Reshape(new[] { rows, cols }).ToMatrix();
    }
}
