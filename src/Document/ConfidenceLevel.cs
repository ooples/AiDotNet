namespace AiDotNet.Document;

/// <summary>
/// Confidence levels for document AI predictions.
/// </summary>
public enum ConfidenceLevel
{
    /// <summary>
    /// Very low confidence (below 25%).
    /// </summary>
    VeryLow,

    /// <summary>
    /// Low confidence (25-50%).
    /// </summary>
    Low,

    /// <summary>
    /// Medium confidence (50-75%).
    /// </summary>
    Medium,

    /// <summary>
    /// High confidence (75-90%).
    /// </summary>
    High,

    /// <summary>
    /// Very high confidence (above 90%).
    /// </summary>
    VeryHigh
}
