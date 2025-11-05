using AiDotNet.RetrievalAugmentedGeneration.Interfaces;

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
        var chunkStart = 0;
        var position = 0;

        foreach (var line in lines)
        {
            var lineLength = line.Length + Environment.NewLine.Length;

            // Check if line is a header (Markdown ## or HTML <h>)
            if (IsHeader(line))
            {
                // Save current chunk if not empty
                if (currentChunk.Count > 0)
                {
                    var content = string.Join(Environment.NewLine, currentChunk);
                    if (content.Length >= _minChunkSize || !_combineSmallChunks)
                    {
                        chunks.Add((content, chunkStart, position));
                    }
                    currentChunk.Clear();
                }

                chunkStart = position;
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
                    currentChunk.Clear();
                    chunkStart = position + lineLength;
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

        return chunks;
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
}
