using AiDotNet.Interfaces;

namespace AiDotNet.RetrievalAugmentedGeneration.ChunkingStrategies;

/// <summary>
/// Splits structured documents based on header tags (H1, H2, H3, etc.).
/// </summary>
/// <remarks>
/// Ideal for Markdown and HTML documents where headers provide natural semantic boundaries.
/// Preserves document structure and hierarchy.
/// </remarks>
public class HeaderBasedTextSplitter : ChunkingStrategyBase
{
    private readonly bool _combineSmallChunks;
    private readonly int _minChunkSize;

    /// <summary>
    /// Initializes a new instance of the <see cref="HeaderBasedTextSplitter"/> class.
    /// </summary>
    /// <param name="chunkSize">Maximum size of each chunk in characters.</param>
    /// <param name="chunkOverlap">The number of characters that should overlap between consecutive chunks.</param>
    /// <param name="minChunkSize">Minimum size for chunk combination.</param>
    /// <param name="combineSmallChunks">Whether to combine small chunks.</param>
    public HeaderBasedTextSplitter(
        int chunkSize,
        int chunkOverlap = 0,
        int minChunkSize = 100,
        bool combineSmallChunks = true)
        : base(chunkSize, chunkOverlap)
    {
        if (minChunkSize < 0)
            throw new ArgumentOutOfRangeException(nameof(minChunkSize), "Min chunk size cannot be negative");

        if (minChunkSize > chunkSize)
            throw new ArgumentOutOfRangeException(nameof(minChunkSize), "Min chunk size cannot exceed max chunk size");

        _minChunkSize = minChunkSize;
        _combineSmallChunks = combineSmallChunks;
    }

    /// <summary>
    /// Core chunking logic that splits text based on header hierarchy.
    /// </summary>
    protected override IEnumerable<(string Chunk, int StartPosition, int EndPosition)> ChunkCore(string text)
    {
        var chunks = new List<(string, int, int)>();
        var lines = text.Split(new[] { "\r\n", "\r", "\n" }, StringSplitOptions.None);
        var currentChunk = new List<string>();
        var previousChunkLines = new List<string>();
        var chunkStart = 0;
        var position = 0;

        foreach (var line in lines)
        {
            var actualLineEndingLength = GetActualLineEndingLength(text, position, line.Length);
            var lineLength = line.Length + actualLineEndingLength;

            // Check if line is a header (Markdown ## or HTML <h>)
            if (IsHeader(line))
            {
                // Save current chunk if not empty (always preserve to prevent data loss)
                if (currentChunk.Count > 0)
                {
                    var content = string.Join(Environment.NewLine, currentChunk);
                    chunks.Add((content, chunkStart, position));

                    // Store lines for overlap
                    previousChunkLines = GetOverlapLines(currentChunk);
                    currentChunk.Clear();
                }

                chunkStart = position;

                // Add overlap from previous chunk (after setting chunkStart)
                if (ChunkOverlap > 0 && previousChunkLines.Count > 0)
                {
                    currentChunk.AddRange(previousChunkLines);
                    // Adjust start to account for overlapped content
                    var overlapLength = string.Join(Environment.NewLine, previousChunkLines).Length;
                    chunkStart = Math.Max(0, position - overlapLength);
                }

                currentChunk.Add(line);
            }
            else
            {
                currentChunk.Add(line);

                // Split if chunk gets too large
                var currentSize = string.Join(Environment.NewLine, currentChunk).Length;
                if (currentSize >= ChunkSize)
                {
                    var content = string.Join(Environment.NewLine, currentChunk);
                    chunks.Add((content, chunkStart, position + lineLength));

                    // Store lines for overlap
                    previousChunkLines = GetOverlapLines(currentChunk);
                    currentChunk.Clear();

                    chunkStart = position + lineLength;

                    // Add overlap from previous chunk (after setting chunkStart)
                    if (ChunkOverlap > 0 && previousChunkLines.Count > 0)
                    {
                        currentChunk.AddRange(previousChunkLines);
                        // Adjust start to account for overlapped content
                        var overlapLength = string.Join(Environment.NewLine, previousChunkLines).Length;
                        chunkStart = Math.Max(0, (position + lineLength) - overlapLength);
                    }
                }
            }

            position += lineLength;
        }

        // Add remaining content
        if (currentChunk.Count > 0)
        {
            var content = string.Join(Environment.NewLine, currentChunk);
            chunks.Add((content, chunkStart, position));
        }

        // Combine small chunks if enabled
        if (_combineSmallChunks && chunks.Count > 1)
        {
            chunks = CombineSmallChunks(chunks);
        }

        return chunks;
    }

    private int GetActualLineEndingLength(string text, int position, int lineLength)
    {
        if (position + lineLength >= text.Length)
            return 0;

        var afterLine = text.Substring(position + lineLength);
        if (afterLine.StartsWith("\r\n"))
            return 2;
        if (afterLine.StartsWith("\n") || afterLine.StartsWith("\r"))
            return 1;

        return 0;
    }

    private List<string> GetOverlapLines(List<string> lines)
    {
        if (ChunkOverlap <= 0 || lines.Count == 0)
            return new List<string>();

        var overlapLines = new List<string>();
        var overlapSize = 0;

        for (int i = lines.Count - 1; i >= 0 && overlapSize < ChunkOverlap; i--)
        {
            overlapLines.Insert(0, lines[i]);
            overlapSize += lines[i].Length + Environment.NewLine.Length;
        }

        return overlapLines;
    }

    private bool IsHeader(string line)
    {
        if (string.IsNullOrWhiteSpace(line))
            return false;

        var trimmed = line.TrimStart();

        // Markdown headers (# ## ### etc.)
        if (trimmed.StartsWith("#"))
            return true;

        // HTML headers (<h1> <h2> etc.)
        if (trimmed.StartsWith("<h", StringComparison.OrdinalIgnoreCase) &&
            trimmed.Length > 2 &&
            char.IsDigit(trimmed[2]))
            return true;

        return false;
    }

    private List<(string, int, int)> CombineSmallChunks(List<(string content, int start, int end)> chunks)
    {
        var result = new List<(string, int, int)>();
        var i = 0;

        while (i < chunks.Count)
        {
            var current = chunks[i];

            // If this chunk is large enough, add it as-is
            if (current.content.Length >= _minChunkSize || i == chunks.Count - 1)
            {
                result.Add(current);
                i++;
                continue;
            }

            // Try to combine with next chunk
            var combined = current.content;
            var combinedStart = current.start;
            var combinedEnd = current.end;
            var j = i + 1;

            while (j < chunks.Count && combined.Length < _minChunkSize)
            {
                var next = chunks[j];

                // Check if adding next chunk would exceed max chunk size
                var nextLength = Environment.NewLine.Length + next.content.Length;
                if (combined.Length + nextLength >= ChunkSize)
                    break;

                combined += Environment.NewLine + next.content;
                combinedEnd = next.end;
                j++;
            }

            result.Add((combined, combinedStart, combinedEnd));
            i = j;
        }

        return result;
    }
}
