using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.RetrievalAugmentedGeneration.ChunkingStrategies;

/// <summary>
/// Specialized splitter that correctly parses and chunks tabular data from documents.
/// </summary>
/// <remarks>
/// Handles various table formats (Markdown, CSV, HTML tables) and ensures table integrity
/// by keeping related rows together and preserving column headers.
/// </remarks>
public class TableAwareTextSplitter : ChunkingStrategyBase
{
    private readonly int _maxRowsPerChunk;
    private readonly bool _includeHeadersInEachChunk;

    /// <summary>
    /// Initializes a new instance of the <see cref="TableAwareTextSplitter"/> class.
    /// </summary>
    /// <param name="chunkSize">The maximum size of each chunk.</param>
    /// <param name="chunkOverlap">The overlap between consecutive chunks.</param>
    /// <param name="maxRowsPerChunk">Maximum number of table rows per chunk.</param>
    /// <param name="includeHeadersInEachChunk">Whether to include table headers in each chunk.</param>
    public TableAwareTextSplitter(
        int chunkSize = 2000,
        int chunkOverlap = 200,
        int maxRowsPerChunk = 50,
        bool includeHeadersInEachChunk = true)
        : base(chunkSize, chunkOverlap)
    {
        if (maxRowsPerChunk <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxRowsPerChunk), "Max rows per chunk must be positive");

        _maxRowsPerChunk = maxRowsPerChunk;
        _includeHeadersInEachChunk = includeHeadersInEachChunk;
    }

    /// <summary>
    /// Splits text while preserving table structure.
    /// </summary>
    protected override IEnumerable<(string Chunk, int StartPosition, int EndPosition)> ChunkCore(string text)
    {
        var chunks = new List<(string Chunk, int StartPosition, int EndPosition)>();
        var lines = text.Split(new[] { "\r\n", "\r", "\n" }, StringSplitOptions.None);

        var position = 0;
        var i = 0;
        while (i < lines.Length)
        {
            // Check if current position is start of a table
            if (IsTableStart(lines, i))
            {
                var tableChunks = ProcessTable(lines, ref i, ref position);
                chunks.AddRange(tableChunks);
            }
            else
            {
                // Regular text line - add as single chunk
                var lineLength = lines[i].Length + Environment.NewLine.Length;
                chunks.Add((lines[i], position, position + lines[i].Length));
                position += lineLength;
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

    private List<(string Chunk, int StartPosition, int EndPosition)> ProcessTable(string[] lines, ref int index, ref int position)
    {
        var chunks = new List<(string Chunk, int StartPosition, int EndPosition)>();
        var startPosition = position;
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

            position += lines[index].Length + Environment.NewLine.Length;
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

                var chunkText = string.Join(Environment.NewLine, chunkContent);
                var chunkStart = startPosition;
                var chunkEnd = position;

                chunks.Add((chunkText, chunkStart, chunkEnd));

                dataRows.Clear();
                startPosition = position;
            }
        }

        // Add remaining rows
        if (dataRows.Count > 0 || headerRows.Count > 0)
        {
            var chunkContent = new List<string>();
            if (_includeHeadersInEachChunk && headerRows.Count > 0)
            {
                chunkContent.AddRange(headerRows);
            }
            chunkContent.AddRange(dataRows);

            if (chunkContent.Count > 0)
            {
                var chunkText = string.Join(Environment.NewLine, chunkContent);
                chunks.Add((chunkText, startPosition, position));
            }
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

