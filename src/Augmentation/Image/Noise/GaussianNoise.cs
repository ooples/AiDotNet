using AiDotNet.Augmentation.Base;
using AiDotNet.Augmentation.Data;

namespace AiDotNet.Augmentation.Image.Noise;

/// <summary>
/// Adds Gaussian noise to an image.
/// </summary>
/// <remarks>
/// <para>
/// Gaussian noise adds random values drawn from a normal (Gaussian) distribution to each pixel.
/// This simulates sensor noise in cameras and helps the model become robust to noisy inputs.
/// </para>
/// <para><b>For Beginners:</b> Think of this like the "grain" you see in photos taken in low light.
/// Adding random noise to training images teaches your model to focus on the real features
/// rather than memorizing exact pixel values.
/// </para>
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>When training data is too clean (synthetic or studio images)</item>
/// <item>When deployed images may have sensor noise</item>
/// <item>As a regularization technique to prevent overfitting</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class GaussianNoise<T> : AugmentationBase<T, ImageTensor<T>>
{
    /// <summary>
    /// Gets the mean of the Gaussian distribution.
    /// </summary>
    public double Mean { get; }

    /// <summary>
    /// Gets the minimum standard deviation of the noise.
    /// </summary>
    public double MinStd { get; }

    /// <summary>
    /// Gets the maximum standard deviation of the noise.
    /// </summary>
    public double MaxStd { get; }

    /// <summary>
    /// Gets the minimum valid pixel value (for clamping).
    /// </summary>
    public double MinValue { get; }

    /// <summary>
    /// Gets the maximum valid pixel value (for clamping).
    /// </summary>
    public double MaxValue { get; }

    /// <summary>
    /// Creates a new Gaussian noise augmentation.
    /// </summary>
    /// <param name="mean">
    /// The mean of the Gaussian distribution.
    /// Industry standard default is 0.0 (no bias).
    /// </param>
    /// <param name="minStd">
    /// The minimum standard deviation of the noise.
    /// Industry standard default is 0.01.
    /// </param>
    /// <param name="maxStd">
    /// The maximum standard deviation of the noise.
    /// Industry standard default is 0.05.
    /// </param>
    /// <param name="probability">
    /// The probability of applying this augmentation (0.0 to 1.0).
    /// Industry standard default is 0.5.
    /// </param>
    /// <param name="minValue">
    /// The minimum valid pixel value for clamping.
    /// Default is 0.0 for normalized images.
    /// </param>
    /// <param name="maxValue">
    /// The maximum valid pixel value for clamping.
    /// Default is 1.0 for normalized images.
    /// </param>
    public GaussianNoise(
        double mean = 0.0,
        double minStd = 0.01,
        double maxStd = 0.05,
        double probability = 0.5,
        double minValue = 0.0,
        double maxValue = 1.0)
        : base(probability)
    {
        if (minStd < 0)
            throw new ArgumentOutOfRangeException(nameof(minStd), "Standard deviation must be non-negative");
        if (maxStd < 0)
            throw new ArgumentOutOfRangeException(nameof(maxStd), "Standard deviation must be non-negative");
        if (minStd > maxStd)
            throw new ArgumentException("minStd must be <= maxStd");

        Mean = mean;
        MinStd = minStd;
        MaxStd = maxStd;
        MinValue = minValue;
        MaxValue = maxValue;
    }

    /// <summary>
    /// Applies Gaussian noise to the image.
    /// </summary>
    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();
        int height = result.Height;
        int width = result.Width;
        int channels = result.Channels;

        // Sample standard deviation for this augmentation
        double std = context.GetRandomDouble(MinStd, MaxStd);

        // Add Gaussian noise to each pixel
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                for (int c = 0; c < channels; c++)
                {
                    double value = Convert.ToDouble(data.GetPixel(y, x, c));

                    // Generate Gaussian noise using Box-Muller transform
                    double noise = SampleGaussian(context, Mean, std);
                    double newValue = value + noise;

                    // Clamp to valid range
                    newValue = Math.Max(MinValue, Math.Min(MaxValue, newValue));

                    result.SetPixel(y, x, c, (T)Convert.ChangeType(newValue, typeof(T)));
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Samples from a Gaussian distribution using Box-Muller transform.
    /// </summary>
    private static double SampleGaussian(AugmentationContext<T> context, double mean, double std)
    {
        // Box-Muller transform
        double u1 = context.GetRandomDouble(0.0001, 0.9999); // Avoid exact 0 or 1
        double u2 = context.GetRandomDouble(0.0, 1.0);

        double z0 = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        return mean + std * z0;
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var parameters = base.GetParameters();
        parameters["mean"] = Mean;
        parameters["min_std"] = MinStd;
        parameters["max_std"] = MaxStd;
        parameters["min_value"] = MinValue;
        parameters["max_value"] = MaxValue;
        return parameters;
    }
}
