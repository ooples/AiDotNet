using AiDotNet.LinearAlgebra;
using AiDotNet.Preprocessing.Scalers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.ImbalancedLearning;

/// <summary>
/// Whole-array engine helpers shared by the resamplers: class counts and members, Euclidean distances from one
/// sample to a set of candidates, nearest-neighbour selection, interpolation and the row gathers that build the
/// resampled data.
/// </summary>
internal static class ResamplingTensorOps
{
    /// <summary>
    /// The number of samples of each class, keyed in the order the classes first appear in <paramref name="y"/>.
    /// </summary>
    public static NumericDictionary<T, int> ClassCounts<T>(Vector<T> y)
    {
        var counts = new NumericDictionary<T, int>();
        int n = y.Length;
        if (n == 0)
            return counts;

        var engine = AiDotNetEngine.Current;
        var labels = Tensor<T>.FromVector(y);
        var classes = PreprocessingTensorOps.SortedUnique(labels);
        int c = classes.Length;
        var shape = new[] { c, n };
        var membership = engine.TensorEquals(
            engine.TensorBroadcastTo(engine.Reshape(classes, new[] { c, 1 }), shape).Contiguous(),
            engine.TensorBroadcastTo(engine.Reshape(labels, new[] { 1, n }), shape).Contiguous());
        // Summed as int: a sum in T stops counting at 2048 for Half
        var sizes = engine.ReduceSum(membership.Cast<int>(), new[] { 1 }, false).AsSpan().ToArray();
        var first = PreprocessingTensorOps.FirstNonZeroPerRow(membership);
        var values = classes.AsSpan().ToArray();
        foreach (int k in Enumerable.Range(0, c).Where(k => sizes[k] > 0).OrderBy(k => first[k]))
            counts[values[k]] = sizes[k];
        return counts;
    }

    /// <summary>The positions of the samples labelled <paramref name="label"/>, in ascending order.</summary>
    public static List<int> ClassIndices<T>(Vector<T> y, T label)
    {
        if (y.Length == 0)
            return new List<int>();
        return new List<int>(PreprocessingTensorOps.MaskPositions(
            AiDotNetEngine.Current.TensorEquals(Tensor<T>.FromVector(y), label)));
    }

    /// <summary>
    /// Euclidean distances [m] from row <paramref name="sampleIndex"/> of <paramref name="data"/> ([n, p]) to the
    /// <paramref name="candidates"/> rows, in candidate order.
    /// </summary>
    public static Tensor<T> Distances<T>(Tensor<T> data, int sampleIndex, int[] candidates)
    {
        int m = candidates.Length;
        if (m == 0)
            return new Tensor<T>(new[] { 0 });

        var engine = AiDotNetEngine.Current;
        var rows = engine.TensorIndexSelect(data, new Tensor<int>((int[])candidates.Clone(), new[] { m }), 0);
        // A gathered copy, not a slice view: Tensor.Cast reads the backing storage from its start
        var sample = engine.TensorIndexSelect(data, new Tensor<int>(new[] { sampleIndex }, new[] { 1 }), 0);

        // Half is reduced in float (see ScalerTensorOps.ReducesInFloat): the squared gap of coordinates 256 apart
        // overflows Half, which made every distant candidate tie at infinity.
        return ScalerTensorOps.ReducesInFloat<T>()
            ? EuclideanDistances(rows.Cast<float>(), sample.Cast<float>()).Cast<T>()
            : EuclideanDistances(rows, sample);
    }

    /// <summary>The Euclidean distance [m] of every row of <paramref name="rows"/> ([m, p]) from <paramref name="sample"/> ([1, p]).</summary>
    private static Tensor<TW> EuclideanDistances<TW>(Tensor<TW> rows, Tensor<TW> sample)
    {
        var engine = AiDotNetEngine.Current;
        var diff = engine.TensorSubtract(rows, engine.TensorBroadcastTo(sample, rows.Shape.ToArray()).Contiguous());
        return engine.TensorSqrt(engine.ReduceSum(engine.TensorMultiply(diff, diff), new[] { 1 }, false));
    }

    /// <summary>
    /// The (at most) <paramref name="k"/> candidates nearest to row <paramref name="sampleIndex"/>, excluding the
    /// row itself, nearest first. Equal distances keep candidate order and NaN distances come first, like the
    /// stable LINQ <c>OrderBy</c> on the distance as a double.
    /// </summary>
    public static List<int> Nearest<T>(Tensor<T> data, int sampleIndex, IEnumerable<int> candidateIndices, int k)
    {
        var candidates = candidateIndices.Where(c => c != sampleIndex).ToArray();
        int take = Math.Min(k, candidates.Length);
        if (take <= 0)
            return new List<int>();

        var ops = MathHelper.GetNumericOperations<T>();
        var distances = ScalerTensorOps.FillNan(Distances(data, sampleIndex, candidates), ops.MinValue);
        return PreprocessingTensorOps.StableOrder(distances, descending: false)
            .Take(take)
            .Select(i => candidates[i])
            .ToList();
    }

    /// <summary>
    /// The nearest other row to row <paramref name="sampleIndex"/>: the lowest-index row at the smallest distance
    /// below <c>MaxValue</c> (NaN distances never qualify), or -1 when there is none.
    /// </summary>
    public static int NearestOther<T>(Tensor<T> data, int sampleIndex)
    {
        var candidates = Enumerable.Range(0, data.Shape[0]).Where(i => i != sampleIndex).ToArray();
        if (candidates.Length == 0)
            return -1;

        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        var distances = ScalerTensorOps.FillNan(Distances(data, sampleIndex, candidates), ops.MaxValue);
        T nearest = engine.TensorMinValue(distances);
        if (!PreprocessingTensorOps.IsBelow(nearest, ops.MaxValue))
            return -1;
        return candidates[PreprocessingTensorOps.MaskPositions(engine.TensorEquals(distances, nearest))[0]];
    }

    /// <summary>The synthetic sample <c>sample1 + t (sample2 - sample1)</c>.</summary>
    public static Vector<T> Interpolate<T>(Vector<T> sample1, Vector<T> sample2, T t)
    {
        var engine = AiDotNetEngine.Current;
        var start = Tensor<T>.FromVector(sample1);
        var diff = engine.TensorSubtract(Tensor<T>.FromVector(sample2), start);
        return engine.TensorAdd(start, engine.TensorMultiplyScalar(diff, t)).ToVector();
    }

    /// <summary>The rows (and labels) at <paramref name="rows"/>, in that order.</summary>
    public static (Matrix<T> X, Vector<T> Y) KeepRows<T>(Matrix<T> x, Vector<T> y, int[] rows) =>
        (PreprocessingTensorOps.SelectRows(x, rows), PreprocessingTensorOps.SelectElements(y, rows));

    /// <summary>The rows not in <paramref name="removed"/>, in ascending order.</summary>
    public static int[] RowsExcept(int count, ICollection<int> removed) =>
        Enumerable.Range(0, count).Where(i => !removed.Contains(i)).ToArray();

    /// <summary>
    /// The original data followed by the synthetic samples (as rows) and their labels.
    /// </summary>
    public static (Matrix<T> X, Vector<T> Y) Append<T>(Matrix<T> x, Vector<T> y, List<Vector<T>> samples, List<T> labels)
    {
        var engine = AiDotNetEngine.Current;
        var data = Tensor<T>.FromMatrix(x);
        var target = Tensor<T>.FromVector(y);
        if (samples.Count == 0)
            return (data.ToMatrix(), target.ToVector());

        var synthetic = engine.TensorStack(samples.Select(s => Tensor<T>.FromVector(s)).ToArray(), 0);
        var resampledX = engine.TensorConcatenate(new[] { data, synthetic }, 0).ToMatrix();
        var syntheticLabels = new Tensor<T>(labels.ToArray(), new[] { labels.Count });
        var resampledY = engine.TensorConcatenate(new[] { target, syntheticLabels }, 0).ToVector();
        return (resampledX, resampledY);
    }
}
