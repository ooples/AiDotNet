using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Models;

namespace AiDotNet.RetrievalAugmentedGeneration.ChunkingStrategies;

/// <summary>
/// Multi-modal splitter for documents containing both text and images.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <remarks>
/// Creates chunks that keep text and related images together, preserving the relationship
/// between visual and textual content for better context preservation.
/// </remarks>
public class MultiModalTextSplitter : ChunkingStrategyBase
{
    private readonly int _maxChunkSize;
    private readonly bool _preserveImageContext;
    private readonly int _contextWindowSize;

    /// <summary>
    /// Initializes a new instance of the <see cref="MultiModalTextSplitter{T}"/> class.
    /// </summary>
    /// <param name="maxChunkSize">Maximum size of text portion in each chunk.</param>
    /// <param name="contextWindowSize">Number of characters before/after image to include.</param>
    /// <param name="preserveImageContext">Whether to keep surrounding text with images.</param>
    /// <param name="numericOperations">The numeric operations provider.</param>
    public MultiModalTextSplitter(
        int maxChunkSize,
        int contextWindowSize,
        bool preserveImageContext,
        INumericOperations<T> numericOperations)
        : base(numericOperations)
    {
        if (maxChunkSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxChunkSize), "Max chunk size must be positive");
            
        if (contextWindowSize < 0)
            throw new ArgumentOutOfRangeException(nameof(contextWindowSize), "Context window size cannot be negative");
            
        _maxChunkSize = maxChunkSize;
        _contextWindowSize = contextWindowSize;
        _preserveImageContext = preserveImageContext;
    }

    /// <summary>
    /// Splits text while preserving text-image relationships.
    /// </summary>
    public override IEnumerable<Document<T>> Chunk(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            return Enumerable.Empty<Document<T>>();

        var chunks = new List<Document<T>>();
        var lines = text.Split(new[] { "\r\n", "\r", "\n" }, StringSplitOptions.None);
        
        var currentChunk = new List<string>();
        var imageReferences = new List<string>();

        for (var i = 0; i < lines.Length; i++)
        {
            var line = lines[i];

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

                imageReferences.Add(ExtractImageReference(line));
                currentChunk.Add(line);

                if (_preserveImageContext)
                {
                    // Include context after image
                    var contextEnd = Math.Min(lines.Length, i + (_contextWindowSize / 50));
                    for (var j = i + 1; j < contextEnd; j++)
                    {
                        currentChunk.Add(lines[j]);
                    }
                    i = contextEnd - 1; // Skip ahead
                }

                // Create chunk with image
                chunks.Add(new Document<T>
                {
                    Id = Guid.NewGuid().ToString(),
                    Content = string.Join(Environment.NewLine, currentChunk),
                    Metadata = new Dictionary<string, object>
                    {
                        ["type"] = "multimodal",
                        ["images"] = imageReferences.ToArray(),
                        ["chunkIndex"] = chunks.Count
                    }
                });

                currentChunk.Clear();
                imageReferences.Clear();
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
                            ["type"] = "text",
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
                    ["type"] = "text",
                    ["chunkIndex"] = chunks.Count
                }
            });
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
        if (trimmed.Contains("<img", StringComparison.OrdinalIgnoreCase))
            return true;

        return false;
    }

    private string ExtractImageReference(string line)
    {
        // Simple extraction - in production would use regex
        var trimmed = line.Trim();
        
        // Extract from Markdown ![alt](url)
        var start = trimmed.IndexOf("](");
        if (start >= 0)
        {
            var end = trimmed.IndexOf(")", start + 2);
            if (end >= 0)
            {
                return trimmed.Substring(start + 2, end - start - 2);
            }
        }

        // Extract from HTML <img src="url">
        start = trimmed.IndexOf("src=\"", StringComparison.OrdinalIgnoreCase);
        if (start >= 0)
        {
            var end = trimmed.IndexOf("\"", start + 5);
            if (end >= 0)
            {
                return trimmed.Substring(start + 5, end - start - 5);
            }
        }

        return line;
    }
}

