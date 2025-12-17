namespace AiDotNet.Models;

/// <summary>
/// Contains metrics for certified accuracy evaluation.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
public class CertifiedAccuracyMetrics<T>
{
    /// <summary>
    /// Gets or sets the standard accuracy on clean examples.
    /// </summary>
    public double CleanAccuracy { get; set; }

    /// <summary>
    /// Gets or sets the certified accuracy at the specified radius.
    /// </summary>
    public double CertifiedAccuracy { get; set; }

    /// <summary>
    /// Gets or sets the certification radius used.
    /// </summary>
    public T CertificationRadius { get; set; } = default!;

    /// <summary>
    /// Gets or sets the average certified radius across all examples.
    /// </summary>
    public T AverageCertifiedRadius { get; set; } = default!;

    /// <summary>
    /// Gets or sets the percentage of examples that could be certified.
    /// </summary>
    public double CertificationRate { get; set; }

    /// <summary>
    /// Gets or sets the median certified radius.
    /// </summary>
    public T MedianCertifiedRadius { get; set; } = default!;

    /// <summary>
    /// Gets or sets additional certification metrics.
    /// </summary>
    public Dictionary<string, double> AdditionalMetrics { get; set; } = new();
}
