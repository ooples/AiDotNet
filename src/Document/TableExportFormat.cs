namespace AiDotNet.Document;

/// <summary>
/// Export formats for extracted table data.
/// </summary>
public enum TableExportFormat
{
    /// <summary>
    /// Comma-separated values format.
    /// </summary>
    CSV,

    /// <summary>
    /// JSON format.
    /// </summary>
    JSON,

    /// <summary>
    /// HTML table format.
    /// </summary>
    HTML,

    /// <summary>
    /// Markdown table format.
    /// </summary>
    Markdown,

    /// <summary>
    /// Excel-compatible XML format.
    /// </summary>
    Excel
}
