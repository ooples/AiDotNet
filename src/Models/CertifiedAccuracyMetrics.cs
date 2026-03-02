namespace AiDotNet.Models;

#pragma warning disable CS8618 // Generic T properties use default(T) - always used with value types
/// <summary>
/// Contains metrics for certified accuracy evaluation.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
public class CertifiedAccuracyMetrics<T>
{
    [System.Diagnostics.CodeAnalysis.SetsRequiredMembers]
    public CertifiedAccuracyMetrics() { }

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
    public required T CertificationRadius { get; set; }

    /// <summary>
    /// Gets or sets the average certified radius across all examples.
    /// </summary>
    public required T AverageCertifiedRadius { get; set; }

    /// <summary>
    /// Gets or sets the percentage of examples that could be certified.
    /// </summary>
    public double CertificationRate { get; set; }

    /// <summary>
    /// Gets or sets the median certified radius.
    /// </summary>
    public required T MedianCertifiedRadius { get; set; }

    /// <summary>
    /// Gets or sets additional certification metrics.
    /// </summary>
    public Dictionary<string, double> AdditionalMetrics { get; set; } = new();
}
