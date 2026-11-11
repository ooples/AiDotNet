namespace AiDotNet.Data.Transforms.Numeric;

/// <summary>
/// Applies Z-score normalization: (x - mean) / std, computing mean and std from a reference dataset.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Unlike <see cref="NormalizeTransform{T}"/> which requires pre-computed mean/std,
/// this transform computes statistics from a reference dataset during construction.
/// </para>
/// <para><b>For Beginners:</b> Z-score normalization converts your data so that
/// the average becomes 0 and the spread becomes 1. This is computed from a reference
/// dataset (typically the training set) and then applied to all data.
/// </para>
/// </remarks>
public class StandardScaleTransform<T> : ITransform<T[], T[]>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly double[] _mean;
    private readonly double[] _std;

    /// <summary>
    /// Creates a standard scaler from a reference dataset.
    /// </summary>
    /// <param name="referenceData">The reference data to compute statistics from. Each row is a sample.</param>
    public StandardScaleTransform(T[][] referenceData)
    {
        if (referenceData is null)
        {
            throw new ArgumentNullException(nameof(referenceData));
        }

        if (referenceData.Length == 0)
        {
            throw new ArgumentException("Reference data cannot be empty.", nameof(referenceData));
        }

        int featureCount = referenceData[0].Length;
        for (int i = 0; i < referenceData.Length; i++)
        {
            if (referenceData[i] is null)
            {
                throw new ArgumentException($"Reference data row {i} is null.", nameof(referenceData));
            }

            if (referenceData[i].Length != featureCount)
            {
                throw new ArgumentException(
                    $"Reference data row {i} has {referenceData[i].Length} elements, expected {featureCount}.",
                    nameof(referenceData));
            }
        }

        int rows = referenceData.Length;
        if (featureCount == 0)
        {
            _mean = new double[0];
            _std = new double[0];
        }
        else
        {
            var engine = AiDotNetEngine.Current;
            var data = TransformTensors.RowsAsDouble(referenceData, featureCount);
            var shape = new[] { rows, featureCount };

            // Compute mean (column sums / rows)
            var mean = engine.TensorDivideScalar(engine.ReduceSum(data, new[] { 0 }, keepDims: true), (double)rows);

            // Compute (population) standard deviation; near-zero deviations fall back to 1
            var centered = engine.TensorSubtract(data, engine.TensorBroadcastTo(mean, shape));
            var variance = engine.TensorDivideScalar(
                engine.ReduceSum(engine.TensorMultiply(centered, centered), new[] { 0 }, keepDims: false), (double)rows);
            var std = engine.TensorSqrt(variance);
            var tiny = engine.TensorEqScalar(engine.TensorLessThan(std, 1e-12), 1.0);
            var ones = new Tensor<double>(new[] { featureCount });
            ones.Fill(1.0);

            _mean = mean.AsSpan().ToArray();
            _std = engine.TensorWhere(tiny, ones, std).AsSpan().ToArray();
        }
    }

    /// <summary>
    /// Creates a standard scaler from pre-computed statistics.
    /// </summary>
    /// <param name="mean">The mean for each feature.</param>
    /// <param name="std">The standard deviation for each feature.</param>
    public StandardScaleTransform(double[] mean, double[] std)
    {
        if (mean is null)
        {
            throw new ArgumentNullException(nameof(mean));
        }

        if (std is null)
        {
            throw new ArgumentNullException(nameof(std));
        }

        if (mean.Length != std.Length)
        {
            throw new ArgumentException(
                $"Mean length ({mean.Length}) must match std length ({std.Length}).",
                nameof(std));
        }

        _mean = (double[])mean.Clone();
        _std = (double[])std.Clone();
    }

    /// <inheritdoc/>
    public T[] Apply(T[] input)
    {
        if (input is null)
        {
            throw new ArgumentNullException(nameof(input));
        }

        int len = Math.Min(input.Length, _mean.Length);
        var result = new T[input.Length];

        if (len > 0)
        {
            // (x - mean) / std over the first len features, computed in double and cast back once
            var engine = AiDotNetEngine.Current;
            var shape = new[] { len };
            var values = new Tensor<double>(TransformTensors.PrefixAsDouble(input, len), shape);
            var normalized = engine.TensorDivide(
                engine.TensorSubtract(values, new Tensor<double>(TransformTensors.Prefix(_mean, len), shape)),
                new Tensor<double>(TransformTensors.Prefix(_std, len), shape));
            normalized.Cast<T>().AsSpan().CopyTo(result.AsSpan(0, len));
        }

        // Features beyond the fitted statistics pass through unchanged
        Array.Copy(input, len, result, len, input.Length - len);

        return result;
    }
}
