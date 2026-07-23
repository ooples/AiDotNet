namespace AiDotNet.Models.Options;

/// <summary>
/// Represents a contiguous parameter index range.
/// </summary>
public class ParameterIndexRange
{
    /// <summary>
    /// Gets or sets the start index (0-based).
    /// </summary>
    public int Start { get; set; }

    /// <summary>
    /// Gets or sets the length of the range.
    /// </summary>
    public int Length { get; set; }
}

