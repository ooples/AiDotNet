using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection;

/// <summary>
/// Whole-array engine helpers shared by the feature selectors (column statistics, rank transforms and
/// class-wise moments), generic in <typeparamref name="T"/> and free of per-element loops.
/// </summary>
internal static class FeatureSelectionTensorOps
{
    /// <summary>Column medians of an [n, p] tensor (mean of the two middle values for even n), as [p].</summary>
    public static Tensor<T> ColumnMedian<T>(Tensor<T> x, bool keepDims = false) =>
        AiDotNetEngine.Current.TensorQuantile(x, 0.5, 0, keepDims, QuantileInterpolation.Linear);

    /// <summary>Column median absolute deviation (unscaled, scipy <c>median_abs_deviation(scale=1)</c>), as [p].</summary>
    public static Tensor<T> ColumnMedianAbsoluteDeviation<T>(Tensor<T> x)
    {
        var engine = AiDotNetEngine.Current;
        var deviations = engine.TensorAbs(engine.TensorSubtract(x, ColumnMedian(x, keepDims: true)));
        return ColumnMedian(deviations);
    }

    /// <summary>
    /// Power spectrum <c>|DFT_k|^2</c>, k = 0 .. w/2, of every column of a [w, m] tensor by the direct DFT
    /// (angle <c>2 pi k i / w</c>), as [w/2 + 1, m].
    /// </summary>
    public static Tensor<T> DftPower<T>(Tensor<T> x)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int w = x.Shape[0];
        int bins = w / 2 + 1;
        var angle = engine.TensorMultiplyScalar(
            engine.TensorOuter(engine.TensorArange<T>(0, bins, 1), engine.TensorArange<T>(0, w, 1)),
            ops.FromDouble(2 * Math.PI / w));
        var re = engine.TensorMatMul(engine.TensorCos(angle), x);
        var im = engine.TensorMatMul(engine.TensorSin(angle), x);
        return engine.TensorAdd(engine.TensorMultiply(re, re), engine.TensorMultiply(im, im));
    }

    /// <summary>
    /// Shannon entropy in bits of each column of a non-negative [k, m] tensor after normalising the column to sum
    /// to 1, counting only probabilities above <paramref name="minProbability"/>; a column whose total is not above
    /// <paramref name="minTotal"/> scores 0. Returns [m].
    /// </summary>
    public static Tensor<T> ColumnEntropyBits<T>(Tensor<T> weights, double minTotal, double minProbability)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        var total = engine.ReduceSum(weights, new[] { 0 }, true);
        var probability = engine.TensorDivide(weights, total);
        var counted = engine.TensorGreaterThan(probability, ops.FromDouble(minProbability));
        var zeros = PreprocessingTensorOps.Filled(weights.Shape.ToArray(), ops.Zero);
        var terms = engine.TensorWhere(counted, engine.TensorMultiply(probability, engine.TensorLog2(probability)), zeros);
        var entropy = engine.TensorNegate(engine.ReduceSum(terms, new[] { 0 }, false));
        var valid = engine.TensorGreaterThan(engine.Reshape(total, new[] { total.Length }), ops.FromDouble(minTotal));
        return engine.TensorWhere(valid, entropy, PreprocessingTensorOps.Filled(new[] { total.Length }, ops.Zero));
    }

    /// <summary>
    /// Squared DFT magnitude <c>|sum_i x_i e^(-2 pi f i)|^2</c> of every column of an [n, m] tensor at each
    /// frequency of <paramref name="frequencies"/> (cycles per sample, shape [f]), as [f, m].
    /// </summary>
    public static Tensor<T> FrequencyPower<T>(Tensor<T> x, Tensor<T> frequencies)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = x.Shape[0];
        var angle = engine.TensorMultiplyScalar(
            engine.TensorOuter(frequencies, engine.TensorArange<T>(0, n, 1)), ops.FromDouble(2 * Math.PI));
        var re = engine.TensorMatMul(engine.TensorCos(angle), x);
        var im = engine.TensorMatMul(engine.TensorSin(angle), x);
        return engine.TensorAdd(engine.TensorMultiply(re, re), engine.TensorMultiply(im, im));
    }

    /// <summary>Column means subtracted from an [n, m] tensor.</summary>
    public static Tensor<T> CenterColumns<T>(Tensor<T> x)
    {
        var engine = AiDotNetEngine.Current;
        return engine.TensorSubtract(x, engine.ReduceMean(x, new[] { 0 }, true));
    }

    /// <summary>Column sums of squares of an [n, m] tensor, as [m].</summary>
    public static Tensor<T> ColumnSumOfSquares<T>(Tensor<T> x)
    {
        var engine = AiDotNetEngine.Current;
        return engine.ReduceSum(engine.TensorMultiply(x, x), new[] { 0 }, false);
    }

    /// <summary>Column sample variances (divisor n - 1) of an [n, m] tensor, as [m].</summary>
    public static Tensor<T> ColumnSampleVariance<T>(Tensor<T> x)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        return AiDotNetEngine.Current.TensorDivideScalar(
            ColumnSumOfSquares(CenterColumns(x)), ops.FromDouble(x.Shape[0] - 1));
    }

    /// <summary>Column population variances (divisor n) of an [n, m] tensor, as [m].</summary>
    public static Tensor<T> ColumnVariance<T>(Tensor<T> x)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        return AiDotNetEngine.Current.TensorDivideScalar(
            ColumnSumOfSquares(CenterColumns(x)), ops.FromDouble(x.Shape[0]));
    }

    /// <summary>
    /// Lag products <c>sum_{i &lt; n - lag} x[i, j] * x[i + lag, j]</c> of every column of an [n, m] tensor, as [m].
    /// </summary>
    public static Tensor<T> LagProducts<T>(Tensor<T> x, int lag)
    {
        var engine = AiDotNetEngine.Current;
        int n = x.Shape[0];
        int m = x.Shape[1];
        var head = engine.TensorSlice(x, new[] { 0, 0 }, new[] { n - lag, m });
        var tail = engine.TensorSlice(x, new[] { lag, 0 }, new[] { n - lag, m });
        return engine.ReduceSum(engine.TensorMultiply(head, tail), new[] { 0 }, false);
    }

    /// <summary>Entries of <paramref name="values"/> where <paramref name="condition"/> is non-zero, 0 elsewhere.</summary>
    public static Tensor<T> ZeroUnless<T>(Tensor<T> condition, Tensor<T> values)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        return AiDotNetEngine.Current.TensorWhere(condition, values,
            PreprocessingTensorOps.Filled(values.Shape.ToArray(), ops.Zero));
    }

    /// <summary>A [p, p] 0/1 mask of the strictly upper-triangular entries (column &gt; row).</summary>
    public static Tensor<T> StrictUpperMask<T>(int p)
    {
        var engine = AiDotNetEngine.Current;
        var index = engine.TensorArange<T>(0, p, 1);
        var rows = engine.TensorBroadcastTo(engine.Reshape(index, new[] { p, 1 }), new[] { p, p });
        var columns = engine.TensorBroadcastTo(engine.Reshape(index, new[] { 1, p }), new[] { p, p });
        return engine.TensorGreaterThan(columns, rows);
    }

    /// <summary>
    /// Pearson correlation matrix of the columns of an [n, p] tensor using sample covariances and sample
    /// standard deviations offset by <paramref name="stdEpsilon"/>, with the diagonal set to 1. Returns [p, p].
    /// </summary>
    public static Tensor<T> SampleCorrelationMatrix<T>(Tensor<T> x, double stdEpsilon)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = x.Shape[0];
        int p = x.Shape[1];
        var centered = CenterColumns(x);
        var covariance = engine.TensorDivideScalar(
            engine.TensorMatMul(engine.TensorTranspose(centered), centered), ops.FromDouble(n - 1));
        var std = engine.TensorAddScalar(
            engine.TensorSqrt(engine.TensorDivideScalar(ColumnSumOfSquares(centered), ops.FromDouble(n - 1))),
            ops.FromDouble(stdEpsilon));
        var correlation = engine.TensorDivide(covariance, engine.TensorOuter(std, std));
        return engine.TensorWhere(engine.TensorEye<T>(p), PreprocessingTensorOps.Filled(new[] { p, p }, ops.One), correlation);
    }

    /// <summary>
    /// Greedy redundancy pruning: walking the features in order, each kept feature removes every later feature
    /// whose entry in the [p, p] 0/1 <paramref name="redundant"/> matrix is set. Returns the kept indices, ascending.
    /// </summary>
    public static int[] KeepFirstOfRedundantPairs<T>(Tensor<T> redundant)
    {
        int p = redundant.Shape[0];
        var pairs = PreprocessingTensorOps.MaskPositions(
            AiDotNetEngine.Current.TensorMultiply(redundant, StrictUpperMask<T>(p)));
        var removed = new bool[p];
        foreach (int position in pairs)
        {
            int first = position / p;
            int second = position % p;
            if (!removed[first])
                removed[second] = true;
        }

        return Enumerable.Range(0, p).Where(j => !removed[j]).ToArray();
    }

    /// <summary>
    /// One Haar analysis step on every column of an [len, m] tensor (len &gt;= 2; a trailing odd row is dropped):
    /// approximation <c>(x[2i] + x[2i+1]) / sqrt(2)</c> and detail <c>(x[2i] - x[2i+1]) / sqrt(2)</c>, each [len/2, m].
    /// </summary>
    public static (Tensor<T> Approximation, Tensor<T> Detail) HaarStep<T>(Tensor<T> x)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int half = x.Shape[0] / 2;
        int m = x.Shape[1];
        var pairs = engine.Reshape(engine.TensorSlice(x, new[] { 0, 0 }, new[] { 2 * half, m }), new[] { half, 2, m });
        var even = engine.TensorSliceAxis(pairs, 1, 0);
        var odd = engine.TensorSliceAxis(pairs, 1, 1);
        T scale = ops.FromDouble(Math.Sqrt(2));
        return (engine.TensorDivideScalar(engine.TensorAdd(even, odd), scale),
            engine.TensorDivideScalar(engine.TensorSubtract(even, odd), scale));
    }
}
