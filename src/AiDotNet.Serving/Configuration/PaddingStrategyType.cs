namespace AiDotNet.Serving.Configuration;

/// <summary>
/// Padding strategies for variable-length sequences.
/// </summary>
public enum PaddingStrategyType
{
    /// <summary>Pad to the minimum required length for each batch.</summary>
    Minimal,

    /// <summary>Pad to predefined bucket sizes for better batching efficiency.</summary>
    Bucket,

    /// <summary>Pad all sequences to a fixed size.</summary>
    Fixed
}

