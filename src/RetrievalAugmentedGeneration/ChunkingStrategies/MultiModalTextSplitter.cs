using AiDotNet.Interfaces;

namespace AiDotNet.RetrievalAugmentedGeneration.ChunkingStrategies;

/// <summary>
/// Multi-modal splitter for documents containing both text and images.
/// </summary>
/// <remarks>
/// Creates chunks that keep text and related images together, preserving the relationship
/// between visual and textual content for better context preservation.
/// </remarks>
public class MultiModalTextSplitter : ChunkingStrategyBase
{
    private readonly bool _preserveImageContext;
    private readonly int _contextWindowSize;

    /// <summary>
    /// Initializes a new instance of the <see cref="MultiModalTextSplitter"/> class.
    /// </summary>
    /// <param name="chunkSize">Maximum size of text portion in each chunk.</param>
    /// <param name="chunkOverlap">The number of characters that should overlap between consecutive chunks.</param>
    /// <param name="contextWindowSize">Number of characters before/after image to include.</param>
    /// <param name="preserveImageContext">Whether to keep surrounding text with images.</param>
    public MultiModalTextSplitter(
        int chunkSize,
        int chunkOverlap = 0,
        int contextWindowSize = 200,
        bool preserveImageContext = true)
        : base(chunkSize, chunkOverlap)
    {
        if (contextWindowSize < 0)
            throw new ArgumentOutOfRangeException(nameof(contextWindowSize), "Context window size cannot be negative");

        // Ensure minimum context window to prevent division issues
        if (contextWindowSize > 0 && contextWindowSize < 50)
            throw new ArgumentOutOfRangeException(nameof(contextWindowSize), "Context window size must be at least 50 characters when enabled");

        _contextWindowSize = contextWindowSize;
        _preserveImageContext = preserveImageContext;
    }

    /// <summary>
    /// Core chunking logic that splits text while preserving text-image relationships.
    /// </summary>
    protected override IEnumerable<(string Chunk, int StartPosition, int EndPosition)> ChunkCore(string text)
    {
        var chunks = new List<(string, int, int)>();
        var lines = text.Split(new[] { "\r\n", "\r", "\n" }, StringSplitOptions.None);

        var currentChunk = new List<string>();
        var chunkStart = 0;
        var position = 0;

        for (var i = 0; i < lines.Length; i++)
        {
            var line = lines[i];
            var lineLength = line.Length + Environment.NewLine.Length;

            // Detect image references (Markdown ![alt](url) or HTML <img>)
            if (IsImageReference(line))
            {
                if (_preserveImageContext)
                {
                    // Include context before image
                    var contextStart = Math.Max(0, i - (_contextWindowSize / 50)); // Approximate lines
                    for (var j = contextStart; j < i; j++)
                    {
                        if (!currentChunk.Contains(lines[j]))
                        {
                            currentChunk.Add(lines[j]);
                        }
                    }
                }

                currentChunk.Add(line);

                if (_preserveImageContext && _contextWindowSize > 0)
                {
                    // Include context after image (minimum 1 line if context window is set)
                    var linesToInclude = Math.Max(1, _contextWindowSize / 50);
                    var contextEnd = Math.Min(lines.Length, i + linesToInclude);
                    for (var j = i + 1; j < contextEnd; j++)
                    {
                        currentChunk.Add(lines[j]);
                    }
                    i = contextEnd - 1; // Skip ahead
                }

                // Create chunk with image
                var content = string.Join(Environment.NewLine, currentChunk);
                chunks.Add((content, chunkStart, position + lineLength));

                currentChunk.Clear();
                chunkStart = position + lineLength;
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

                    // Apply overlap: keep last N lines for next chunk
                    if (ChunkOverlap > 0)
                    {
                        var overlapLines = new List<string>();
                        var overlapSize = 0;
                        for (int k = currentChunk.Count - 1; k >= 0 && overlapSize < ChunkOverlap; k--)
                        {
                            var overlapLine = currentChunk[k];
                            overlapLines.Insert(0, overlapLine);
                            overlapSize += overlapLine.Length + Environment.NewLine.Length;
                        }
                        currentChunk.Clear();
                        currentChunk.AddRange(overlapLines);
                        // Adjust chunk start to account for overlap
                        chunkStart = position + lineLength - overlapSize;
                    }
                    else
                    {
                        currentChunk.Clear();
                        chunkStart = position + lineLength;
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

        return chunks;
    }

    private bool IsImageReference(string line)
    {
        var trimmed = line.Trim();

        // Markdown image: ![alt](url)
        if (trimmed.Contains("![") && trimmed.Contains("]("))
            return true;

        // HTML image: <img src="...">
        if (trimmed.IndexOf("<img", StringComparison.OrdinalIgnoreCase) >= 0)
            return true;

        return false;
    }
}
