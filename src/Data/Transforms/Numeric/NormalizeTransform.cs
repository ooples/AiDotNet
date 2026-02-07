namespace AiDotNet.Data.Transforms.Numeric;

/// <summary>
/// Normalizes an array of values using mean and standard deviation: (x - mean) / std.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// This is the standard normalization used in image preprocessing and feature scaling.
/// Each element is independently normalized using the provided per-channel or global mean/std.
/// </para>
/// <para><b>For Beginners:</b> Normalization makes your data have zero mean and unit variance,
/// which helps neural networks train faster and more stably.
/// </para>
/// </remarks>
public class NormalizeTransform<T> : ITransform<T[], T[]>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly T[] _mean;
    private readonly T[] _std;

    /// <summary>
    /// Creates a normalize transform with per-element mean and standard deviation.
    /// </summary>
    /// <param name="mean">The mean values for each element/channel.</param>
    /// <param name="std">The standard deviation values for each element/channel.</param>
    public NormalizeTransform(T[] mean, T[] std)
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

        _mean = mean;
        _std = std;
    }

    /// <summary>
    /// Creates a normalize transform with a single global mean and standard deviation.
    /// </summary>
    /// <param name="mean">The global mean value.</param>
    /// <param name="std">The global standard deviation value.</param>
    /// <param name="length">The expected length of input arrays.</param>
    public NormalizeTransform(T mean, T std, int length)
    {
        if (length <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(length), "Length must be positive.");
        }

        _mean = new T[length];
        _std = new T[length];
        for (int i = 0; i < length; i++)
        {
            _mean[i] = mean;
            _std[i] = std;
        }
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
            T diff = NumOps.Subtract(input[i], _mean[i]);
            double stdVal = NumOps.ToDouble(_std[i]);
            if (Math.Abs(stdVal) < 1e-12)
            {
                result[i] = diff;
            }
            else
            {
                result[i] = NumOps.Divide(diff, _std[i]);
            }
        }

        // Copy remaining elements unchanged if input is longer than mean/std
        for (int i = len; i < input.Length; i++)
        {
            result[i] = input[i];
        }

        return result;
    }
}
