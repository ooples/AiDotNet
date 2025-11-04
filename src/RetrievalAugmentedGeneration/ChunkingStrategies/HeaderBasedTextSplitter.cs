using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Models;

namespace AiDotNet.RetrievalAugmentedGeneration.ChunkingStrategies;

/// <summary>
/// Splits structured documents based on header tags (H1, H2, H3, etc.).
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <remarks>
/// Ideal for Markdown and HTML documents where headers provide natural semantic boundaries.
/// Preserves document structure and hierarchy.
/// </remarks>
public class HeaderBasedTextSplitter : ChunkingStrategyBase
{
    private readonly int _maxChunkSize;
    private readonly bool _combineSmallChunks;
    private readonly int _minChunkSize;

    /// <summary>
    /// Initializes a new instance of the <see cref="HeaderBasedTextSplitter{T}"/> class.
    /// </summary>
    /// <param name="maxChunkSize">Maximum size of each chunk in characters.</param>
    /// <param name="minChunkSize">Minimum size for chunk combination.</param>
    /// <param name="combineSmallChunks">Whether to combine small chunks.</param>
    /// <param name="numericOperations">The numeric operations provider.</param>
    public HeaderBasedTextSplitter(
        int maxChunkSize,
        int minChunkSize,
        bool combineSmallChunks,
        INumericOperations<T> numericOperations)
        : base(numericOperations)
    {
        if (maxChunkSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxChunkSize), "Max chunk size must be positive");
            
        if (minChunkSize < 0)
            throw new ArgumentOutOfRangeException(nameof(minChunkSize), "Min chunk size cannot be negative");
            
        _maxChunkSize = maxChunkSize;
        _minChunkSize = minChunkSize;
        _combineSmallChunks = combineSmallChunks;
    }

    /// <summary>
    /// Splits text based on header hierarchy.
    /// </summary>
    public override IEnumerable<Document<T>> Chunk(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            return Enumerable.Empty<Document<T>>();

        var chunks = new List<Document<T>>();
        var lines = text.Split(new[] { "\r\n", "\r", "\n" }, StringSplitOptions.None);
        var currentChunk = new List<string>();
        var currentHeader = string.Empty;

        foreach (var line in lines)
        {
            // Check if line is a header (Markdown ## or HTML <h>)
            if (IsHeader(line))
            {
                // Save current chunk if not empty
                if (currentChunk.Count > 0)
                {
                    var content = string.Join(Environment.NewLine, currentChunk);
                    if (content.Length >= _minChunkSize || !_combineSmallChunks)
                    {
                        chunks.Add(new Document<T>
                        {
                            Id = Guid.NewGuid().ToString(),
                            Content = content,
                            Metadata = new Dictionary<string, object>
                            {
                                ["header"] = currentHeader,
                                ["chunkIndex"] = chunks.Count
                            }
                        });
                        currentChunk.Clear();
                    }
                }

                currentHeader = line.Trim();
                currentChunk.Add(line);
            }
            else
            {
                currentChunk.Add(line);

                // Split if chunk gets too large
                var currentSize = string.Join(Environment.NewLine, currentChunk).Length;
                if (currentSize >= _maxChunkSize)
                {
                    chunks.Add(new Document<T>
                    {
                        Id = Guid.NewGuid().ToString(),
                        Content = string.Join(Environment.NewLine, currentChunk),
                        Metadata = new Dictionary<string, object>
                        {
                            ["header"] = currentHeader,
                            ["chunkIndex"] = chunks.Count
                        }
                    });
                    currentChunk.Clear();
                }
            }
        }

        // Add remaining content
        if (currentChunk.Count > 0)
        {
            chunks.Add(new Document<T>
            {
                Id = Guid.NewGuid().ToString(),
                Content = string.Join(Environment.NewLine, currentChunk),
                Metadata = new Dictionary<string, object>
                {
                    ["header"] = currentHeader,
                    ["chunkIndex"] = chunks.Count
                }
            });
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

