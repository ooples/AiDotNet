namespace AiDotNet.Models;

/// <summary>
/// Options for regression models.
/// </summary>
public class RegressionOptions<T>
{
    /// <summary>
    /// The percentage of the data to use for training. The rest will be used for testing.
    /// </summary>
    public double TrainingPctSize { get; set; } = 75;

    /// <summary>
    /// The normalization to use on the data before training and testing.
    /// </summary>
    public INormalizer<T>? Normalization { get; set; }

    /// <summary>
    /// The outlier removal to use on the data before training and testing.
    /// </summary>
    public IOutlierRemoval? OutlierRemoval { get; set; }
}