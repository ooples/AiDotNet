namespace AiDotNet.Postprocessing.Document;

/// <summary>
/// Rule for extracting a field from document text.
/// </summary>
public class FieldExtractionRule
{
    /// <summary>
    /// Name of the field to extract.
    /// </summary>
    public string FieldName { get; set; } = "";

    /// <summary>
    /// Regex pattern for extraction.
    /// </summary>
    public string? Pattern { get; set; }

    /// <summary>
    /// Labels that might precede the value.
    /// </summary>
    public IList<string>? Labels { get; set; }

    /// <summary>
    /// Expected data type.
    /// </summary>
    public DataType DataType { get; set; } = DataType.String;

    /// <summary>
    /// Default value if extraction fails.
    /// </summary>
    public object? DefaultValue { get; set; }
}
