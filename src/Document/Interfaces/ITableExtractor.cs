namespace AiDotNet.Document.Interfaces;

/// <summary>
/// Interface for table detection and structure recognition models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Table extraction models detect tables in documents and extract their structure
/// (rows, columns, cells) along with the content in each cell.
/// </para>
/// <para>
/// <b>For Beginners:</b> Documents often contain tables with important data. Table extraction
/// helps computers understand where tables are, how they're structured, and what data they contain.
/// This is useful for extracting financial data, product catalogs, or any tabular information.
///
/// Example usage:
/// <code>
/// var tables = tableExtractor.DetectTables(documentImage);
/// foreach (var table in tables)
/// {
///     var structure = tableExtractor.RecognizeStructure(table.Image);
///     Console.WriteLine($"Table with {structure.NumRows} rows and {structure.NumColumns} columns");
/// }
/// </code>
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("TableExtractor")]
public interface ITableExtractor<T> : IDocumentModel<T>
{
    /// <summary>
    /// Detects tables in a document image.
    /// </summary>
    /// <param name="documentImage">The document image tensor.</param>
    /// <returns>List of detected table regions with bounding boxes.</returns>
    IEnumerable<TableRegion<T>> DetectTables(Tensor<T> documentImage);

    /// <summary>
    /// Recognizes the structure of a table (rows, columns, cells).
    /// </summary>
    /// <param name="tableImage">Cropped table image tensor (from DetectTables).</param>
    /// <returns>Table structure with cell positions and spans.</returns>
    TableStructureResult<T> RecognizeStructure(Tensor<T> tableImage);

    /// <summary>
    /// Extracts table content as structured data.
    /// </summary>
    /// <param name="documentImage">The document image tensor.</param>
    /// <returns>Tables as lists of rows, where each row is a list of cell contents.</returns>
    /// <remarks>
    /// <para>
    /// This is a convenience method that combines table detection, structure recognition,
    /// and OCR to return the final table content.
    /// </para>
    /// </remarks>
    IEnumerable<List<List<string>>> ExtractTableContent(Tensor<T> documentImage);

    /// <summary>
    /// Exports detected tables to a specific format.
    /// </summary>
    /// <param name="documentImage">The document image tensor.</param>
    /// <param name="format">Output format (CSV, JSON, HTML, Markdown, Excel).</param>
    /// <returns>Serialized table data in the specified format.</returns>
    string ExportTables(Tensor<T> documentImage, TableExportFormat format);

    /// <summary>
    /// Gets whether this model supports bordered tables.
    /// </summary>
    bool SupportsBorderedTables { get; }

    /// <summary>
    /// Gets whether this model supports borderless tables.
    /// </summary>
    bool SupportsBorderlessTables { get; }

    /// <summary>
    /// Gets whether this model can detect merged cells (row/column spans).
    /// </summary>
    bool SupportsMergedCells { get; }
}
