namespace AiDotNet.Enums;

/// <summary>
/// Specifies how much detail should be included in benchmark reports.
/// </summary>
public enum BenchmarkReportDetailLevel
{
    /// <summary>
    /// Include only high-level summary metrics.
    /// </summary>
    Summary,

    /// <summary>
    /// Include additional structured breakdowns when available (for example, category-level metrics).
    /// </summary>
    Detailed
}

