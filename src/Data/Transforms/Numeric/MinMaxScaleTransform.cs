namespace AiDotNet.Data.Transforms.Numeric;

/// <summary>
/// Scales values to a target range using min-max normalization.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// The formula is: x_scaled = (x - min) / (max - min) * (targetMax - targetMin) + targetMin
/// </para>
/// <para><b>For Beginners:</b> Min-max scaling squeezes your data into a specific range
/// (default [0, 1]). This is commonly used for pixel values or when your algorithm
/// requires bounded inputs.
/// </para>
/// </remarks>
public class MinMaxScaleTransform<T> : ITransform<T[], T[]>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly double[] _min;
    private readonly double[] _max;
    private readonly double _targetMin;
    private readonly double _targetMax;

    /// <summary>
    /// Creates a min-max scaler from a reference dataset.
    /// </summary>
    /// <param name="referenceData">The reference data to compute min/max from.</param>
    /// <param name="targetMin">The minimum value of the target range. Default is 0.</param>
    /// <param name="targetMax">The maximum value of the target range. Default is 1.</param>
    public MinMaxScaleTransform(T[][] referenceData, double targetMin = 0.0, double targetMax = 1.0)
    {
        if (referenceData is null)
        {
            throw new ArgumentNullException(nameof(referenceData));
        }

        if (referenceData.Length == 0)
        {
            throw new ArgumentException("Reference data cannot be empty.", nameof(referenceData));
        }

        if (targetMin >= targetMax)
        {
            throw new ArgumentException("Target min must be less than target max.");
        }

        _targetMin = targetMin;
        _targetMax = targetMax;

        int featureCount = referenceData[0].Length;
        _min = new double[featureCount];
        _max = new double[featureCount];

        for (int j = 0; j < featureCount; j++)
        {
            _min[j] = double.MaxValue;
            _max[j] = double.MinValue;
        }

        for (int i = 0; i < referenceData.Length; i++)
        {
            for (int j = 0; j < featureCount; j++)
            {
                double val = NumOps.ToDouble(referenceData[i][j]);
                if (val < _min[j])
                {
                    _min[j] = val;
                }

                if (val > _max[j])
                {
                    _max[j] = val;
                }
            }
        }
    }

    /// <summary>
    /// Creates a min-max scaler from pre-computed statistics.
    /// </summary>
    /// <param name="min">The minimum for each feature.</param>
    /// <param name="max">The maximum for each feature.</param>
    /// <param name="targetMin">The minimum value of the target range. Default is 0.</param>
    /// <param name="targetMax">The maximum value of the target range. Default is 1.</param>
    public MinMaxScaleTransform(double[] min, double[] max, double targetMin = 0.0, double targetMax = 1.0)
    {
        if (min is null)
        {
            throw new ArgumentNullException(nameof(min));
        }

        if (max is null)
        {
            throw new ArgumentNullException(nameof(max));
        }

        if (min.Length != max.Length)
        {
            throw new ArgumentException(
                $"Min length ({min.Length}) must match max length ({max.Length}).",
                nameof(max));
        }

        if (targetMin >= targetMax)
        {
            throw new ArgumentException("Target min must be less than target max.");
        }

        _min = (double[])min.Clone();
        _max = (double[])max.Clone();
        _targetMin = targetMin;
        _targetMax = targetMax;
    }

    /// <inheritdoc/>
    public T[] Apply(T[] input)
    {
        if (input is null)
        {
            throw new ArgumentNullException(nameof(input));
        }

        int len = Math.Min(input.Length, _min.Length);
        var result = new T[input.Length];
        double targetRange = _targetMax - _targetMin;

        for (int i = 0; i < len; i++)
        {
            double val = NumOps.ToDouble(input[i]);
            double range = _max[i] - _min[i];

            double scaled;
            if (Math.Abs(range) < 1e-12)
            {
                scaled = _targetMin;
            }
            else
            {
                scaled = ((val - _min[i]) / range) * targetRange + _targetMin;
            }

            result[i] = NumOps.FromDouble(scaled);
        }

        for (int i = len; i < input.Length; i++)
        {
            result[i] = input[i];
        }

        return result;
    }
}
