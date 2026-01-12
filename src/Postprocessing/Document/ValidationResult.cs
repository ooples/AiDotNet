namespace AiDotNet.Postprocessing.Document;

/// <summary>
/// Result of document schema validation.
/// </summary>
public class ValidationResult
{
    /// <summary>
    /// Whether the data is valid.
    /// </summary>
    public bool IsValid { get; set; }

    /// <summary>
    /// Validation errors.
    /// </summary>
    public IList<string> Errors { get; set; } = new List<string>();

    /// <summary>
    /// Validation warnings.
    /// </summary>
    public IList<string> Warnings { get; set; } = new List<string>();
}
