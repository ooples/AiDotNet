using AiDotNet.Interfaces;

namespace AiDotNet.RetrievalAugmentedGeneration.ChunkingStrategies;

/// <summary>
/// Provides a base implementation for text chunking strategies with common functionality.
/// </summary>
/// <remarks>
/// <para>
/// This abstract class implements the IChunkingStrategy interface and provides common functionality
/// for text splitting strategies. It handles validation and provides utility methods for chunk overlap
/// while allowing derived classes to focus on implementing the core chunking algorithm.
/// </para>
/// <para><b>For Beginners:</b> This is the foundation that all text splitting methods build upon.
/// 
/// Think of it like a template for dividing text:
/// - It handles common tasks (checking inputs, managing overlap)
/// - Specific chunking methods (fixed-size, sentence-based) just fill in how they split text
/// - This ensures all chunking strategies work consistently
/// </para>
/// </remarks>
public abstract class ChunkingStrategyBase : IChunkingStrategy
{
    private readonly int _chunkSize;
    private readonly int _chunkOverlap;

    /// <summary>
    /// Gets the target size for each chunk in characters.
    /// </summary>
    public int ChunkSize => _chunkSize;

    /// <summary>
    /// Gets the number of characters that should overlap between consecutive chunks.
    /// </summary>
    public int ChunkOverlap => _chunkOverlap;

    /// <summary>
    /// Initializes a new instance of the ChunkingStrategyBase class.
    /// </summary>
    /// <param name="chunkSize">The target size for each chunk in characters.</param>
    /// <param name="chunkOverlap">The number of characters that should overlap between consecutive chunks.</param>
    protected ChunkingStrategyBase(int chunkSize, int chunkOverlap)
    {
        if (chunkSize <= 0)
            throw new ArgumentException("ChunkSize must be greater than zero", nameof(chunkSize));

        if (chunkOverlap < 0)
            throw new ArgumentException("ChunkOverlap cannot be negative", nameof(chunkOverlap));

        if (chunkOverlap >= chunkSize)
            throw new ArgumentException("ChunkOverlap must be less than ChunkSize", nameof(chunkOverlap));

        _chunkSize = chunkSize;
        _chunkOverlap = chunkOverlap;
    }

    /// <summary>
    /// Splits a text string into chunks according to the strategy's rules.
    /// </summary>
    /// <param name="text">The text to split into chunks.</param>
    /// <returns>A collection of text chunks, ordered as they appear in the original text.</returns>
    public IEnumerable<string> Chunk(string text)
    {
        ValidateText(text);
        return ChunkCore(text).Select(tuple => tuple.Chunk);
    }

    /// <summary>
    /// Splits a text string into chunks and returns them with position metadata.
    /// </summary>
    /// <param name="text">The text to split into chunks.</param>
    /// <returns>A collection of tuples containing each chunk, its start position, and end position in the original text.</returns>
    public IEnumerable<(string Chunk, int StartPosition, int EndPosition)> ChunkWithPositions(string text)
    {
        ValidateText(text);
        return ChunkCore(text);
    }

    /// <summary>
    /// Core chunking logic to be implemented by derived classes.
    /// </summary>
    /// <param name="text">The validated text to split into chunks.</param>
    /// <returns>A collection of tuples containing each chunk and its position information.</returns>
    /// <remarks>
    /// <para><b>For Implementers:</b> This is where you implement your specific chunking algorithm.
    /// 
    /// You don't need to:
    /// - Validate the text (already done)
    /// - Handle null/empty text (already validated)
    /// 
    /// You should:
    /// - Split the text according to your strategy (fixed-size, sentence-based, etc.)
    /// - Respect ChunkSize and ChunkOverlap settings
    /// - Return chunks with accurate start/end positions
    /// - Ensure chunks appear in order
    /// 
    /// Use the helper method CreateOverlappingChunks() for simple fixed-size chunking.
    /// </para>
    /// </remarks>
    protected abstract IEnumerable<(string Chunk, int StartPosition, int EndPosition)> ChunkCore(string text);

    /// <summary>
    /// Validates the input text.
    /// </summary>
    /// <param name="text">The text to validate.</param>
    /// <remarks>
    /// <para><b>For Implementers:</b> Override this if you need custom text validation.
    /// For example, checking encoding or specific character requirements.
    /// </para>
    /// </remarks>
    protected virtual void ValidateText(string text)
    {
        if (text == null)
            throw new ArgumentNullException(nameof(text));

        if (string.IsNullOrEmpty(text))
            throw new ArgumentException("Text cannot be empty", nameof(text));
    }

    /// <summary>
    /// Creates overlapping chunks from text using simple character-based splitting.
    /// </summary>
    /// <param name="text">The text to chunk.</param>
    /// <returns>A collection of overlapping chunks with position information.</returns>
    /// <remarks>
    /// <para>
    /// This helper method implements basic fixed-size chunking with overlap.
    /// It's useful for derived classes that need simple character-based splitting
    /// or as a fallback when more sophisticated splitting isn't possible.
    /// </para>
    /// <para><b>For Implementers:</b> Use this for simple fixed-size chunking.
    /// 
    /// This method:
    /// - Splits text every ChunkSize characters
    /// - Overlaps chunks by ChunkOverlap characters
    /// - Tracks exact positions in original text
    /// - Handles edge cases (last chunk, small text)
    /// 
    /// Override ChunkCore if you need:
    /// - Sentence-aware splitting
    /// - Paragraph-aware splitting
    /// - Semantic chunking
    /// - Any non-character-based logic
    /// </para>
    /// </remarks>
    protected IEnumerable<(string Chunk, int StartPosition, int EndPosition)> CreateOverlappingChunks(string text)
    {
        var chunks = new List<(string, int, int)>();
        var textLength = text.Length;
        var position = 0;

        while (position < textLength)
        {
            var remainingLength = textLength - position;
            var currentChunkSize = Math.Min(_chunkSize, remainingLength);

            var chunk = text.Substring(position, currentChunkSize);
            var endPosition = position + currentChunkSize;

            chunks.Add((chunk, position, endPosition));

            // Move to next chunk position with overlap
            position += _chunkSize - _chunkOverlap;

            // If next position would go past the end and we've already captured the last chunk, stop
            if (position >= textLength && chunks.Count > 0 && chunks[chunks.Count - 1].Item3 == textLength)
                break;
        }

        return chunks;
    }

    /// <summary>
    /// Splits text on sentence boundaries while respecting chunk size limits.
    /// </summary>
    /// <param name="text">The text to split.</param>
    /// <param name="sentenceEndings">Characters that indicate sentence endings.</param>
    /// <returns>A collection of chunks with position information.</returns>
    /// <remarks>
    /// <para>
    /// This helper method splits text at sentence boundaries when possible, falling back
    /// to character-based splitting if sentences are too long. It respects the configured
    /// chunk size and overlap settings.
    /// </para>
    /// <para><b>For Implementers:</b> Use this for sentence-aware chunking.
    /// 
    /// This method:
    /// - Tries to break at sentence boundaries (., !, ?)
    /// - Falls back to character splitting for very long sentences
    /// - Respects ChunkSize limits
    /// - Maintains ChunkOverlap between chunks
    /// 
    /// Good for: General text, articles, documentation
    /// Not ideal for: Code, structured data, poetry
    /// </para>
    /// </remarks>
    protected IEnumerable<(string Chunk, int StartPosition, int EndPosition)> SplitOnSentences(
        string text,
        char[]? sentenceEndings = null)
    {
        var endings = sentenceEndings ?? new[] { '.', '!', '?' };

        var chunks = new List<(string, int, int)>();
        var currentChunk = new System.Text.StringBuilder();
        var chunkStart = 0;
        var position = 0;

        while (position < text.Length)
        {
            currentChunk.Append(text[position]);
            position++;

            // Check if we're at a sentence boundary
            var isSentenceEnd = position < text.Length &&
                               endings.Contains(text[position - 1]) &&
                               char.IsWhiteSpace(text[position]);

            // Create chunk if we hit sentence end and chunk is large enough, or if chunk is at max size
            if ((isSentenceEnd && currentChunk.Length >= _chunkSize / 2) ||
                currentChunk.Length >= _chunkSize)
            {
                var chunkText = currentChunk.ToString();
                if (!string.IsNullOrWhiteSpace(chunkText))
                {
                    chunks.Add((chunkText, chunkStart, chunkStart + chunkText.Length));
                }

                // Start new chunk with overlap
                if (position < text.Length)
                {
                    // Ensure overlap doesn't cause next chunk to exceed chunkSize
                    // Use the larger of (length - overlap) and (length - chunkSize/2) to prevent oversized chunks
                    var overlapStart = Math.Max(currentChunk.Length - _chunkOverlap,
                                                currentChunk.Length - _chunkSize / 2);
                    overlapStart = Math.Max(0, overlapStart);
                    currentChunk = new System.Text.StringBuilder(currentChunk.ToString().Substring(overlapStart));
                    chunkStart = chunkStart + overlapStart;
                }
                else
                {
                    currentChunk.Clear();
                }
            }
        }

        // Add final chunk if there's content
        if (currentChunk.Length > 0)
        {
            var chunkText = currentChunk.ToString();
            if (!string.IsNullOrWhiteSpace(chunkText))
            {
                chunks.Add((chunkText, chunkStart, chunkStart + chunkText.Length));
            }
        }

        return chunks;
    }
}
