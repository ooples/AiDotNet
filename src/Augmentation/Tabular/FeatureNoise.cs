using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Augmentation.Tabular;

/// <summary>
/// Adds Gaussian noise to numerical features in tabular data.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This augmentation adds small random variations to your data,
/// similar to measurement noise in real-world data. This helps models become robust to
/// small fluctuations and prevents overfitting to exact values.</para>
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>Numerical features that have natural variation</item>
/// <item>Small datasets where regularization is needed</item>
/// <item>When you want to simulate measurement uncertainty</item>
/// </list>
/// </para>
/// <para><b>When NOT to use:</b>
/// <list type="bullet">
/// <item>Categorical features (use other augmentations)</item>
/// <item>Features with strict constraints (e.g., binary flags)</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class FeatureNoise<T> : TabularAugmenterBase<T>
{
    /// <summary>
    /// Gets the standard deviation of the noise.
    /// </summary>
    /// <remarks>
    /// <para>Default: 0.01 (1% of unit variance)</para>
    /// <para>Higher values add more noise. Typical range: 0.001 to 0.1</para>
    /// </remarks>
    public double NoiseStdDev { get; }

    /// <summary>
    /// Gets or sets the indices of features to apply noise to.
    /// </summary>
    /// <remarks>
    /// <para>If null, applies to all features.</para>
    /// </remarks>
    public int[]? FeatureIndices { get; set; }

    /// <summary>
    /// Creates a new feature noise augmentation.
    /// </summary>
    /// <param name="noiseStdDev">Standard deviation of Gaussian noise (default: 0.01).</param>
    /// <param name="probability">Probability of applying this augmentation (default: 0.5).</param>
    /// <param name="featureIndices">Optional array of feature indices to apply noise to.</param>
    public FeatureNoise(
        double noiseStdDev = 0.01,
        double probability = 0.5,
        int[]? featureIndices = null) : base(probability)
    {
        if (noiseStdDev < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(noiseStdDev), "Noise standard deviation must be non-negative.");
        }

        NoiseStdDev = noiseStdDev;
        FeatureIndices = featureIndices;
    }

    /// <inheritdoc />
    protected override Matrix<T> ApplyAugmentation(Matrix<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();
        int rows = GetSampleCount(data);
        int cols = GetFeatureCount(data);

        // Determine which features to augment
        var featuresToAugment = FeatureIndices ?? Enumerable.Range(0, cols).ToArray();

        for (int i = 0; i < rows; i++)
        {
            foreach (int j in featuresToAugment.Where(f => f >= 0 && f < cols))
            {
                double originalValue = Convert.ToDouble(result[i, j]);
                double noise = context.SampleGaussian(0, NoiseStdDev);
                double newValue = originalValue + noise;
                result[i, j] = (T)Convert.ChangeType(newValue, typeof(T));
            }
        }

        return result;
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var parameters = base.GetParameters();
        parameters["noiseStdDev"] = NoiseStdDev;
        if (FeatureIndices is not null)
        {
            parameters["featureIndices"] = FeatureIndices;
        }
        return parameters;
    }
}
