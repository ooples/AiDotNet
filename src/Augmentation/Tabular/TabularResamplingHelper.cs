using AiDotNet.CausalDiscovery;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Augmentation.Tabular;

/// <summary>
/// Engine-op building blocks shared by the tabular resamplers (Tomek links, NearMiss, random undersampling).
/// </summary>
/// <typeparam name="T">The numeric type of the data.</typeparam>
internal static class TabularResamplingHelper<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Row indices whose label equals <paramref name="label"/>, in ascending order.
    /// </summary>
    public static int[] IndicesOfLabel(Vector<T> labels, T label)
    {
        var engine = AiDotNetEngine.Current;
        if (labels.Length == 0)
        {
            return Array.Empty<int>();
        }

        return engine.TensorNonzero(engine.TensorEquals(labels.AsTensor(), label)).ToArray();
    }

    /// <summary>
    /// The distinct labels other than <paramref name="excluded"/>, in ascending order (the order np.unique gives).
    /// </summary>
    public static T[] OtherLabels(Vector<T> labels, T excluded)
    {
        if (labels.Length == 0)
        {
            return Array.Empty<T>();
        }

        var engine = AiDotNetEngine.Current;
        var distinct = engine.TensorUnique(labels.AsTensor(), sorted: true);
        int[] keep = engine.TensorNonzero(engine.TensorNotEquals(distinct, excluded)).ToArray();
        return engine.TensorIndexSelect(distinct, new Tensor<int>(keep, new[] { keep.Length }), 0).ToArray();
    }

    /// <summary>
    /// The mean of the <paramref name="k"/> smallest (or largest) entries of <paramref name="values"/>.
    /// </summary>
    public static T MeanOfExtremes(Tensor<T> values, int k, bool largest)
    {
        var engine = AiDotNetEngine.Current;
        var (extremes, _) = engine.TopK(values, k, 0, largest);
        return engine.TensorMean(extremes);
    }

    /// <summary>
    /// Euclidean distances from <paramref name="point"/> (length d) to every row of <paramref name="points"/> [n, d].
    /// </summary>
    /// <remarks>
    /// Each row's differences are divided by that row's largest |difference| m before squaring, and the result is
    /// m * sqrt(sum((diff / m)^2)) (the hypot scaling). The squares then lie in [0, 1] and their sum in [1, d], so
    /// a Half distance above 256 stays finite instead of overflowing past 65504, and near rows keep their
    /// precision because every row is scaled by its own m.
    /// </remarks>
    public static Tensor<T> DistancesTo(Tensor<T> points, Tensor<T> point)
    {
        var engine = AiDotNetEngine.Current;
        int n = points.Shape[0];
        int d = points.Shape[1];
        var broadcast = engine.TensorBroadcastTo(engine.Reshape(point, new[] { 1, d }), new[] { n, d });
        var diff = engine.TensorSubtract(points, broadcast);

        var rowScale = engine.ReduceMax(engine.TensorAbs(diff), new[] { 1 }, keepDims: false);
        var zeros = new Tensor<T>(new[] { n });
        var positive = engine.TensorGreaterThan(rowScale, NumOps.Zero);
        var safeScale = engine.TensorWhere(positive, rowScale, engine.TensorAddScalar(zeros, NumOps.One));
        var scaled = engine.TensorDivide(
            diff, engine.TensorBroadcastTo(engine.Reshape(safeScale, new[] { n, 1 }), new[] { n, d }));
        var squared = engine.ReduceSum(engine.TensorMultiply(scaled, scaled), new[] { 1 }, keepDims: false);
        return engine.TensorMultiply(rowScale, engine.TensorSqrt(squared));
    }

    /// <summary>
    /// The rows of <paramref name="data"/> at <paramref name="indices"/> as a [count, d] tensor.
    /// </summary>
    public static Tensor<T> Rows(Matrix<T> data, int[] indices)
    {
        var source = Tensor<T>.FromMatrix(data);
        return AiDotNetEngine.Current.TensorIndexSelect(source, new Tensor<int>((int[])indices.Clone(), new[] { indices.Length }), 0);
    }

    /// <summary>
    /// The nearest other row of every row (Euclidean; the lowest index wins ties), or -1 when there is no other row.
    /// </summary>
    public static int[] NearestNeighbors(Matrix<T> data)
    {
        var engine = AiDotNetEngine.Current;
        int rows = data.Rows;
        var neighbors = new int[rows];
        if (rows < 2)
        {
            for (int i = 0; i < rows; i++)
            {
                neighbors[i] = -1;
            }

            return neighbors;
        }

        // The row itself is removed from the candidates rather than masked with the largest value of T: an
        // infinite distance (a difference past Half's range) or a tie at that value let the row pick itself.
        var points = Tensor<T>.FromMatrix(data);
        var candidates = new int[rows - 1];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0, c = 0; j < rows; j++)
            {
                if (j != i)
                {
                    candidates[c++] = j;
                }
            }

            var distances = DistancesTo(points, data.GetRow(i).AsTensor());
            var others = engine.TensorIndexSelect(distances, new Tensor<int>((int[])candidates.Clone(), new[] { rows - 1 }), 0);
            neighbors[i] = candidates[engine.TensorArgMin(others, 0).Data.Span[0]];
        }

        return neighbors;
    }

    /// <summary>
    /// Indices that sort <paramref name="keys"/>, equal keys in ascending index order (a stable sort).
    /// </summary>
    public static int[] StableOrder(Tensor<T> keys, bool descending)
    {
        return CausalTensorHelpers.StableArgsort(keys, descending);
    }

    /// <summary>
    /// The rows and labels at <paramref name="indices"/>, in that order.
    /// </summary>
    public static (Matrix<T> Data, Vector<T> Labels) Select(Matrix<T> data, Vector<T> labels, int[] indices)
    {
        var engine = AiDotNetEngine.Current;
        var index = new Tensor<int>((int[])indices.Clone(), new[] { indices.Length });
        var selectedData = engine.TensorIndexSelect(Tensor<T>.FromMatrix(data), index, 0).ToMatrix();
        var selectedLabels = engine.TensorIndexSelect(labels.AsTensor(), index, 0).ToVector();
        return (selectedData, selectedLabels);
    }
}
