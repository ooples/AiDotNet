using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Models;

namespace AiDotNet.RetrievalAugmentedGeneration.ChunkingStrategies;

/// <summary>
/// Specialized splitter that correctly parses and chunks tabular data from documents.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <remarks>
/// Handles various table formats (Markdown, CSV, HTML tables) and ensures table integrity
/// by keeping related rows together and preserving column headers.
/// </remarks>
public class TableAwareTextSplitter<T> : ChunkingStrategyBase<T>
{
    private readonly int _maxRowsPerChunk;
    private readonly bool _includeHeadersInEachChunk;

    /// <summary>
    /// Initializes a new instance of the <see cref="TableAwareTextSplitter{T}"/> class.
    /// </summary>
    /// <param name="maxRowsPerChunk">Maximum number of table rows per chunk.</param>
    /// <param name="includeHeadersInEachChunk">Whether to include table headers in each chunk.</param>
    /// <param name="numericOperations">The numeric operations provider.</param>
    public TableAwareTextSplitter(
        int maxRowsPerChunk,
        bool includeHeadersInEachChunk,
        INumericOperations<T> numericOperations)
        : base(numericOperations)
    {
        if (maxRowsPerChunk <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxRowsPerChunk), "Max rows per chunk must be positive");
            
        _maxRowsPerChunk = maxRowsPerChunk;
        _includeHeadersInEachChunk = includeHeadersInEachChunk;
    }

    /// <summary>
    /// Splits text while preserving table structure.
    /// </summary>
    public override IEnumerable<Document<T>> Chunk(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            return Enumerable.Empty<Document<T>>();

        var chunks = new List<Document<T>>();
        var lines = text.Split(new[] { "\r\n", "\r", "\n" }, StringSplitOptions.None);
        
        var i = 0;
        while (i < lines.Length)
        {
            // Check if current position is start of a table
            if (IsTableStart(lines, i))
            {
                var tableChunks = ProcessTable(lines, ref i);
                chunks.AddRange(tableChunks);
            }
            else
            {
                // Regular text line
                chunks.Add(new Document<T>
                {
                    Id = Guid.NewGuid().ToString(),
                    Content = lines[i],
                    Metadata = new Dictionary<string, object>
                    {
                        ["type"] = "text",
                        ["chunkIndex"] = chunks.Count
                    }
                });
                i++;
            }
        }

        return chunks;
    }

    private bool IsTableStart(string[] lines, int index)
    {
        if (index >= lines.Length)
            return false;

        var line = lines[index].Trim();
        
        // Markdown table (starts with |)
        if (line.StartsWith("|"))
            return true;

        // CSV (contains commas)
        if (line.Contains(",") && index + 1 < lines.Length && lines[index + 1].Contains(","))
            return true;

        // HTML table
        if (line.StartsWith("<table", StringComparison.OrdinalIgnoreCase))
            return true;

        return false;
    }

    private List<Document<T>> ProcessTable(string[] lines, ref int index)
    {
        var chunks = new List<Document<T>>();
        var tableStart = index;
        var headerRows = new List<string>();
        var dataRows = new List<string>();

        // Extract table header
        while (index < lines.Length && IsTableRow(lines[index]))
        {
            if (headerRows.Count < 2) // Typically header + separator
            {
                headerRows.Add(lines[index]);
            }
            else
            {
                dataRows.Add(lines[index]);
            }
            index++;

            // Create chunk when we hit max rows
            if (dataRows.Count >= _maxRowsPerChunk)
            {
                var chunkContent = new List<string>();
                if (_includeHeadersInEachChunk)
                {
                    chunkContent.AddRange(headerRows);
                }
                chunkContent.AddRange(dataRows);

                chunks.Add(new Document<T>
                {
                    Id = Guid.NewGuid().ToString(),
                    Content = string.Join(Environment.NewLine, chunkContent),
                    Metadata = new Dictionary<string, object>
                    {
                        ["type"] = "table",
                        ["chunkIndex"] = chunks.Count,
                        ["tableStart"] = tableStart
                    }
                });

                dataRows.Clear();
            }
        }

        // Add remaining rows
        if (dataRows.Count > 0)
        {
            var chunkContent = new List<string>();
            if (_includeHeadersInEachChunk)
            {
                chunkContent.AddRange(headerRows);
            }
            chunkContent.AddRange(dataRows);

            chunks.Add(new Document<T>
            {
                Id = Guid.NewGuid().ToString(),
                Content = string.Join(Environment.NewLine, chunkContent),
                Metadata = new Dictionary<string, object>
                {
                    ["type"] = "table",
                    ["chunkIndex"] = chunks.Count,
                    ["tableStart"] = tableStart
                }
            });
        }

        return chunks;
    }

    private bool IsTableRow(string line)
    {
        var trimmed = line.Trim();
        return trimmed.StartsWith("|") || 
               trimmed.Contains(",") || 
               trimmed.StartsWith("<tr", StringComparison.OrdinalIgnoreCase);
    }
}
