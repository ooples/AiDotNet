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
