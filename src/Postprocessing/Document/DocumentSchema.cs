namespace AiDotNet.Postprocessing.Document;

/// <summary>
/// Schema for document validation.
/// </summary>
public class DocumentSchema
{
    /// <summary>
    /// Required field names.
    /// </summary>
    public IList<string> RequiredFields { get; set; } = new List<string>();

    /// <summary>
    /// Expected types for fields.
    /// </summary>
    public Dictionary<string, DataType> FieldTypes { get; set; } = new();

    /// <summary>
    /// Regex patterns for field validation.
    /// </summary>
    public Dictionary<string, string> FieldPatterns { get; set; } = new();
}
