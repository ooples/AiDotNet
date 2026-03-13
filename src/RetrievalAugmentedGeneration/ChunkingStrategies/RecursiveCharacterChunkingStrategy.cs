using AiDotNet.Interfaces;

namespace AiDotNet.RetrievalAugmentedGeneration.ChunkingStrategies;

/// <summary>
/// Recursively splits text using a hierarchy of separators to preserve document structure.
/// </summary>
/// <remarks>
/// <para>
/// This advanced chunking strategy tries to split text using the most semantically meaningful
/// separators first (e.g., double newlines for paragraphs), falling back to less meaningful
/// separators (single newlines, spaces) only when necessary. This preserves the natural
/// structure of documents and keeps related content together.
/// </para>
/// <para><b>For Beginners:</b> This is a smart splitter that keeps related text together.
/// 
/// Think of it like organizing a document by trying the best splits first:
/// 
/// Priority 1: Split by double newlines (paragraphs)
///   "Paragraph 1\n\nParagraph 2\n\nParagraph 3"
///   → Keeps each paragraph whole
/// 
/// Priority 2: If paragraphs are too big, split by single newlines (sentences/lines)
///   "Long paragraph with\nmultiple lines\nthat need splitting"
///   → Splits at line breaks
/// 
/// Priority 3: If lines are too big, split by periods (sentences)
///   "First sentence. Second sentence. Third sentence."
///   → Splits at sentences
/// 
/// Priority 4: If sentences are too big, split by spaces (words)
///   "This is a very long sentence without periods"
///   → Splits at words
/// 
/// Priority 5: Last resort, split by characters
///   "ReallyLongWordWithNoSpaces"
///   → Splits anywhere
/// 
/// Why this is better than simple splitting:
/// - Keeps paragraphs together when possible (best semantic unity)
/// - Falls back gracefully when content is too large
/// - Preserves natural document structure
/// - Works well with various document formats (code, articles, books)
/// 
/// Example with chunkSize=100, overlap=20:
/// 
/// Input: "First paragraph.\n\nSecond paragraph that is very long and needs to be split into multiple chunks.\n\nThird paragraph."
/// 
/// 1. Try splitting by "\n\n" → Second paragraph too large
/// 2. Split second paragraph by " " → Gets multiple chunks
/// 3. Add overlap between chunks
/// 
/// Result:
/// - Chunk 1: "First paragraph."
/// - Chunk 2: "Second paragraph that is very long and" (overlap from chunk 1)
/// - Chunk 3: "very long and needs to be split into" (overlap from chunk 2)
/// - Chunk 4: "split into multiple chunks."
/// - Chunk 5: "Third paragraph."
/// </para>
/// </remarks>
public class RecursiveCharacterChunkingStrategy : ChunkingStrategyBase
{
    private readonly string[] _separators;

    /// <summary>
    /// Initializes a new instance of the RecursiveCharacterChunkingStrategy class.
    /// </summary>
    /// <param name="chunkSize">Maximum size for each chunk in characters (default: 1000).</param>
    /// <param name="chunkOverlap">Number of characters to overlap between chunks (default: 200).</param>
    /// <param name="separators">Ordered list of separators to try (default: paragraph, newline, period, space, character).</param>
    public RecursiveCharacterChunkingStrategy(
        int chunkSize = 1000,
        int chunkOverlap = 200,
        string[]? separators = null)
        : base(chunkSize, chunkOverlap)
    {
        _separators = separators ?? new[] { "\n\n", "\n", ". ", " ", "" };
    }

    /// <summary>
    /// Recursively splits text using the separator hierarchy.
    /// </summary>
    /// <param name="text">The validated text to split.</param>
    /// <returns>Chunks with position information.</returns>
    protected override IEnumerable<(string Chunk, int StartPosition, int EndPosition)> ChunkCore(string text)
    {
        var chunks = SplitTextRecursively(text, _separators);
        var results = new List<(string, int, int)>();
        var position = 0;

        foreach (var chunk in chunks)
        {
            var endPos = position + chunk.Length;
            results.Add((chunk, position, endPos));
            position = endPos;
        }

        return results;
    }

    /// <summary>
    /// Recursively splits text, trying each separator in order.
    /// </summary>
    /// <param name="text">The text to split.</param>
    /// <param name="separators">The ordered list of separators to try.</param>
    /// <returns>A list of text chunks.</returns>
    private List<string> SplitTextRecursively(string text, string[] separators)
    {
        var finalChunks = new List<string>();

        // Base case: if text is small enough, return it as a single chunk
        if (text.Length <= ChunkSize)
        {
            if (!string.IsNullOrWhiteSpace(text))
            {
                finalChunks.Add(text.Trim());
            }
            return finalChunks;
        }

        // Get the current separator
        var separator = separators[0];
        var nextSeparators = separators.Length > 1 ? separators.Skip(1).ToArray() : Array.Empty<string>();

        // Split by the current separator
        string[] splits;
        if (!string.IsNullOrEmpty(separator))
        {
            splits = text.Split(new[] { separator }, StringSplitOptions.None);
        }
        else if (nextSeparators.Length == 0)
        {
            // Final fallback: split by character
            splits = text.Select(c => c.ToString()).ToArray();
        }
        else
        {
            splits = new[] { text };
        }

        var currentChunk = new System.Text.StringBuilder();

        foreach (var split in splits)
        {
            // If this split would make the chunk too large
            if (currentChunk.Length + split.Length + separator.Length > ChunkSize)
            {
                // If we have accumulated content, save it
                if (currentChunk.Length > 0)
                {
                    finalChunks.Add(currentChunk.ToString().Trim());
                    currentChunk.Clear();

                    // Add overlap from the end of the last chunk
                    if (finalChunks.Count > 0 && ChunkOverlap > 0)
                    {
                        var lastChunk = finalChunks[finalChunks.Count - 1];
                        var overlapStart = Math.Max(0, lastChunk.Length - ChunkOverlap);
                        var overlap = lastChunk.Substring(overlapStart);
                        currentChunk.Append(overlap);
                    }
                }

                // If the split itself is too large, recursively split it with next separator
                if (split.Length > ChunkSize)
                {
                    var subChunks = SplitTextRecursively(split, nextSeparators);
                    // SubChunks are already appropriately sized from recursive splitting,
                    // add them directly to finalChunks instead of re-merging
                    foreach (var subChunk in subChunks)
                    {
                        if (!string.IsNullOrWhiteSpace(subChunk))
                        {
                            finalChunks.Add(subChunk.Trim());
                        }
                    }
                }
                else
                {
                    if (currentChunk.Length > 0)
                    {
                        currentChunk.Append(separator);
                    }
                    currentChunk.Append(split);
                }
            }
            else
            {
                if (currentChunk.Length > 0 && !string.IsNullOrEmpty(separator))
                {
                    currentChunk.Append(separator);
                }
                currentChunk.Append(split);
            }
        }

        // Add any remaining content
        if (currentChunk.Length > 0)
        {
            finalChunks.Add(currentChunk.ToString().Trim());
        }

        return finalChunks;
    }
}
