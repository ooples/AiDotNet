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
}
