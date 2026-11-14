using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Critic score statistics shared by <see cref="WGAN{T}"/> and <see cref="WGANGP{T}"/>.
/// </summary>
internal static class CriticScoreMetrics
{
    /// <summary>
    /// Adds the mean, min, max and sample standard deviation of the first critic output of each sample.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Matches the LINQ statistics on the scores: no samples throws, the minimum is NaN when any score is NaN,
    /// and the maximum ignores NaN unless every score is NaN.
    /// </para>
    /// <para>
    /// Critic scores are unbounded, so the mean and deviation are computed on the scores divided by their
    /// largest finite magnitude and scaled back: in Half a raw sum or square overflows at 65504.
    /// </para>
    /// </remarks>
    public static void Add<T>(Dictionary<string, double> metrics, Tensor<T> criticScores, int sampleSize)
    {
        if (sampleSize <= 0)
        {
            throw new InvalidOperationException("Sequence contains no elements");
        }

        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        var scores = engine.TensorNarrow(engine.TensorNarrow(criticScores, 0, 0, sampleSize), 1, 0, 1).Reshape(sampleSize);
        var zeros = engine.TensorZerosLike(scores);
        var isNaN = engine.TensorIsNan(scores);
        int nanCount = engine.TensorCountNonzero(engine.TensorWhere(isNaN, engine.TensorOnesLike(scores), zeros));

        T scale = engine.TensorMaxValue(engine.TensorWhere(engine.TensorIsFinite(scores), engine.TensorAbs(scores), zeros));
        if (!ops.GreaterThan(scale, ops.Zero))
        {
            scale = ops.One;
        }

        double scaleValue = ops.ToDouble(scale);
        var scaled = engine.TensorDivideScalar(scores, scale);
        T scaledMean = ops.Divide(engine.TensorSum(scaled), ops.FromDouble(sampleSize));
        metrics["AverageCriticScore"] = ops.ToDouble(scaledMean) * scaleValue;

        if (nanCount > 0)
        {
            metrics["MinCriticScore"] = double.NaN;
        }
        else
        {
            metrics["MinCriticScore"] = ops.ToDouble(engine.TensorMinValue(scores));
        }

        if (nanCount == sampleSize)
        {
            metrics["MaxCriticScore"] = double.NaN;
        }
        else
        {
            T smallest = engine.TensorMinValue(engine.TensorWhere(isNaN, engine.TensorFullLike(scores, ops.MaxValue), scores));
            var numbers = engine.TensorWhere(isNaN, engine.TensorFullLike(scores, smallest), scores);
            metrics["MaxCriticScore"] = ops.ToDouble(engine.TensorMaxValue(numbers));
        }

        // Sample standard deviation (n - 1), 0 below two samples
        double standardDeviation = 0.0;
        if (sampleSize >= 2)
        {
            var deviation = engine.TensorAddScalar(scaled, ops.Negate(scaledMean));
            T sumOfSquares = engine.TensorSum(engine.TensorMultiply(deviation, deviation));
            T variance = ops.Divide(sumOfSquares, ops.FromDouble(sampleSize - 1));
            standardDeviation = ops.ToDouble(ops.Sqrt(variance)) * scaleValue;
        }

        metrics["CriticScoreStdDev"] = standardDeviation;
    }
}
