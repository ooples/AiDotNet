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
        _mean = new double[featureCount];
        _std = new double[featureCount];

        // Compute mean
        for (int j = 0; j < featureCount; j++)
        {
            double sum = 0;
            for (int i = 0; i < referenceData.Length; i++)
            {
                sum += NumOps.ToDouble(referenceData[i][j]);
            }

            _mean[j] = sum / referenceData.Length;
        }

        // Compute standard deviation
        for (int j = 0; j < featureCount; j++)
        {
            double sumSqDiff = 0;
            for (int i = 0; i < referenceData.Length; i++)
            {
                double diff = NumOps.ToDouble(referenceData[i][j]) - _mean[j];
                sumSqDiff += diff * diff;
            }

            _std[j] = Math.Sqrt(sumSqDiff / referenceData.Length);
            if (_std[j] < 1e-12)
            {
                _std[j] = 1.0;
            }
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

        for (int i = 0; i < len; i++)
        {
            double val = NumOps.ToDouble(input[i]);
            double normalized = (val - _mean[i]) / _std[i];
            result[i] = NumOps.FromDouble(normalized);
        }

        for (int i = len; i < input.Length; i++)
        {
            result[i] = input[i];
        }

        return result;
    }
}
