namespace AiDotNet.Document;

/// <summary>
/// Represents the result of table structure recognition.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Tables have structure (rows, columns, cells) and content.
/// This result class describes the table's layout including merged cells,
/// headers, and the data in each cell.
/// </para>
/// </remarks>
public class TableStructureResult<T>
{
    /// <summary>
    /// Gets the number of rows in the table.
    /// </summary>
    public int NumRows { get; init; }

    /// <summary>
    /// Gets the number of columns in the table.
    /// </summary>
    public int NumColumns { get; init; }

    /// <summary>
    /// Gets all cells in the table.
    /// </summary>
    public IReadOnlyList<TableCell<T>> Cells { get; init; } = [];

    /// <summary>
    /// Gets the header row indices (often row 0).
    /// </summary>
    public IReadOnlyList<int> HeaderRows { get; init; } = [];

    /// <summary>
    /// Gets whether the table has detected borders.
    /// </summary>
    public bool HasBorders { get; init; }

    /// <summary>
    /// Gets the overall confidence for the structure detection.
    /// </summary>
    public T Confidence { get; init; } = default!;

    /// <summary>
    /// Gets the table content as a 2D list of strings.
    /// </summary>
    /// <returns>Rows containing columns of cell text.</returns>
    public List<List<string>> ToStringGrid()
    {
        var grid = new List<List<string>>();
        for (int row = 0; row < NumRows; row++)
        {
            var rowData = new List<string>();
            for (int col = 0; col < NumColumns; col++)
            {
                var cell = Cells.FirstOrDefault(c => c.Row == row && c.Column == col);
                rowData.Add(cell?.Text ?? string.Empty);
            }
            grid.Add(rowData);
        }
        return grid;
    }
}

/// <summary>
/// Represents a single cell in a table.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class TableCell<T>
{
    /// <summary>
    /// Gets the row index (0-based).
    /// </summary>
    public int Row { get; init; }

    /// <summary>
    /// Gets the column index (0-based).
    /// </summary>
    public int Column { get; init; }

    /// <summary>
    /// Gets the row span (number of rows this cell covers).
    /// </summary>
    public int RowSpan { get; init; } = 1;

    /// <summary>
    /// Gets the column span (number of columns this cell covers).
    /// </summary>
    public int ColSpan { get; init; } = 1;

    /// <summary>
    /// Gets the text content of the cell.
    /// </summary>
    public string Text { get; init; } = string.Empty;

    /// <summary>
    /// Gets the bounding box of the cell.
    /// </summary>
    public Vector<T> BoundingBox { get; init; } = Vector<T>.Empty();

    /// <summary>
    /// Gets whether this is a header cell.
    /// </summary>
    public bool IsHeader { get; init; }

    /// <summary>
    /// Gets the confidence for this cell's detection.
    /// </summary>
    public T Confidence { get; init; } = default!;
}

/// <summary>
/// Represents a detected table region in a document.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class TableRegion<T>
{
    /// <summary>
    /// Gets the bounding box of the table region.
    /// </summary>
    public Vector<T> BoundingBox { get; init; } = Vector<T>.Empty();

    /// <summary>
    /// Gets the cropped table image.
    /// </summary>
    public Tensor<T>? Image { get; init; }

    /// <summary>
    /// Gets the detection confidence.
    /// </summary>
    public T Confidence { get; init; } = default!;

    /// <summary>
    /// Gets the page index where the table was found.
    /// </summary>
    public int PageIndex { get; init; }

    /// <summary>
    /// Gets the table index on the page.
    /// </summary>
    public int TableIndex { get; init; }
}
