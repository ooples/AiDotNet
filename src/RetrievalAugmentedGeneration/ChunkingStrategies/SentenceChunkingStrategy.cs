using AiDotNet.Interfaces;

namespace AiDotNet.RetrievalAugmentedGeneration.ChunkingStrategies;

/// <summary>
/// Splits text into chunks based on sentence boundaries to preserve semantic coherence.
/// </summary>
/// <remarks>
/// <para>
/// This chunking strategy splits text at sentence boundaries (periods, question marks,
/// exclamation points) and combines sentences until reaching the target chunk size.
/// This approach preserves complete thoughts and improves retrieval quality compared
/// to arbitrary character-based splitting.
/// </para>
/// <para><b>For Beginners:</b> This keeps complete sentences together in each chunk.
/// 
/// Think of it like organizing a book:
/// - Bad way: Cut every 500 characters, even mid-sentence
///   "The cat sat on the m|at. The dog ran ar|ound the yard."
/// - Good way: Keep sentences whole
///   Chunk 1: "The cat sat on the mat. The dog ran around the yard."
///   Chunk 2: "The bird flew over the fence. The fish swam in the pond."
/// 
/// Why this matters:
/// - Retrieval works better when searching complete thoughts
/// - Generators get more coherent context
/// - No weird sentence fragments that confuse the model
/// 
/// Parameters:
/// - targetChunkSize: Aim for this many characters per chunk
/// - maxChunkSize: Never exceed this size (may break sentences if needed)
/// - overlapSentences: Number of sentences to repeat between chunks for context
/// 
/// Example with targetChunkSize=100, overlapSentences=1:
/// "First sentence. Second sentence. Third sentence. Fourth sentence."
/// 
/// Chunk 1: "First sentence. Second sentence. Third sentence."
/// Chunk 2: "Third sentence. Fourth sentence." (overlap: "Third sentence")
/// </para>
/// </remarks>
public class SentenceChunkingStrategy : ChunkingStrategyBase
{
    private readonly int _targetChunkSize;
    private readonly int _maxChunkSize;
    private readonly int _overlapSentences;
    private static readonly char[] SentenceEnders = { '.', '!', '?' };

    /// <summary>
    /// Initializes a new instance of the SentenceChunkingStrategy class.
    /// </summary>
    /// <param name="targetChunkSize">Target size for each chunk in characters (default: 500).</param>
    /// <param name="maxChunkSize">Maximum allowed chunk size in characters (default: 1000).</param>
    /// <param name="overlapSentences">Number of sentences to overlap between chunks (default: 1).</param>
    public SentenceChunkingStrategy(int targetChunkSize = 500, int maxChunkSize = 1000, int overlapSentences = 1)
        : base(maxChunkSize, 0)
    {
        if (maxChunkSize < targetChunkSize)
        {
            throw new ArgumentException("Maximum chunk size must be greater than or equal to target chunk size.", nameof(maxChunkSize));
        }

        if (overlapSentences < 0)
        {
            throw new ArgumentException("Overlap sentences cannot be negative.", nameof(overlapSentences));
        }

        _targetChunkSize = targetChunkSize;
        _maxChunkSize = maxChunkSize;
        _overlapSentences = overlapSentences;
    }

    /// <summary>
    /// Splits text into chunks at sentence boundaries with accurate position tracking.
    /// </summary>
    /// <param name="text">The validated text to split.</param>
    /// <returns>Chunks with position information.</returns>
    protected override IEnumerable<(string Chunk, int StartPosition, int EndPosition)> ChunkCore(string text)
    {
        // Split text into sentences with their positions
        var sentencesWithPos = SplitIntoSentencesWithPositions(text);

        if (sentencesWithPos.Count == 0)
            yield break;

        var currentSentences = new List<(string, int, int)>();
        var currentLength = 0;

        for (int i = 0; i < sentencesWithPos.Count; i++)
        {
            var (sentence, sentStart, sentEnd) = sentencesWithPos[i];
            var sentenceLength = sentence.Length;

            // If adding this sentence would exceed maxChunkSize and we have content, create a chunk
            if (currentLength + sentenceLength > _maxChunkSize && currentSentences.Count > 0)
            {
                // Create chunk from accumulated sentences
                var chunkStart = currentSentences[0].Item2;
                var chunkEnd = currentSentences[currentSentences.Count - 1].Item3;
                var chunkText = text.Substring(chunkStart, chunkEnd - chunkStart);
                yield return (chunkText, chunkStart, chunkEnd);

                // Keep the last N sentences for overlap (include separator lengths)
                var overlapStart = Math.Max(0, currentSentences.Count - _overlapSentences);
                currentSentences = currentSentences.GetRange(overlapStart, currentSentences.Count - overlapStart);
                currentLength = currentSentences.Count > 0
                    ? currentSentences[currentSentences.Count - 1].Item3 - currentSentences[0].Item2
                    : 0;
            }

            // Handle sentences that exceed maxChunkSize on their own
            if (sentenceLength > _maxChunkSize)
            {
                // If we have accumulated content, save it first
                if (currentSentences.Count > 0)
                {
                    var chunkStart = currentSentences[0].Item2;
                    var chunkEnd = currentSentences[currentSentences.Count - 1].Item3;
                    var chunkText = text.Substring(chunkStart, chunkEnd - chunkStart);
                    yield return (chunkText, chunkStart, chunkEnd);
                    currentSentences.Clear();
                    currentLength = 0;
                }

                // Split the oversized sentence into smaller pieces
                for (int pos = 0; pos < sentence.Length; pos += _maxChunkSize)
                {
                    var pieceLength = Math.Min(_maxChunkSize, sentence.Length - pos);
                    var pieceStart = sentStart + pos;
                    var pieceEnd = pieceStart + pieceLength;
                    var pieceText = text.Substring(pieceStart, pieceLength);
                    yield return (pieceText, pieceStart, pieceEnd);
                }

                continue;
            }

            currentSentences.Add((sentence, sentStart, sentEnd));
            currentLength = currentSentences[currentSentences.Count - 1].Item3 - currentSentences[0].Item2;

            // If we've reached target size, create a chunk
            if (currentLength >= _targetChunkSize)
            {
                var chunkStart = currentSentences[0].Item2;
                var chunkEnd = currentSentences[currentSentences.Count - 1].Item3;
                var chunkText = text.Substring(chunkStart, chunkEnd - chunkStart);
                yield return (chunkText, chunkStart, chunkEnd);

                // Keep the last N sentences for overlap
                var overlapStart = Math.Max(0, currentSentences.Count - _overlapSentences);
                currentSentences = currentSentences.GetRange(overlapStart, currentSentences.Count - overlapStart);
                currentLength = currentSentences.Sum(s => s.Item1.Length);
            }
        }

        // Add remaining sentences as final chunk
        if (currentSentences.Count > 0)
        {
            var chunkStart = currentSentences[0].Item2;
            var chunkEnd = currentSentences[currentSentences.Count - 1].Item3;
            var chunkText = text.Substring(chunkStart, chunkEnd - chunkStart);
            yield return (chunkText, chunkStart, chunkEnd);
        }
    }

    /// <summary>
    /// Splits text into individual sentences with their positions in the original text.
    /// </summary>
    /// <param name="text">The text to split.</param>
    /// <returns>A list of tuples containing (sentence, start position, end position).</returns>
    private List<(string, int, int)> SplitIntoSentencesWithPositions(string text)
    {
        var results = new List<(string, int, int)>();
        var currentStart = 0;
        var currentLength = 0;

        for (int i = 0; i < text.Length; i++)
        {
            currentLength++;
            var ch = text[i];

            // Check for sentence endings
            var isSentenceEnd = (ch == '.' || ch == '!' || ch == '?');

            if (isSentenceEnd)
            {
                // Look ahead to see if this is really a sentence end
                var nextIdx = i + 1;
                while (nextIdx < text.Length && char.IsWhiteSpace(text[nextIdx]))
                    nextIdx++;

                // If next character is uppercase or we're at end, it's a sentence boundary
                var isRealEnd = nextIdx >= text.Length || char.IsUpper(text[nextIdx]);

                if (isRealEnd || i == text.Length - 1)
                {
                    var sentence = text.Substring(currentStart, currentLength).Trim();
                    if (!string.IsNullOrWhiteSpace(sentence))
                    {
                        // Calculate actual positions of trimmed sentence
                        var trimmedStart = currentStart;
                        while (trimmedStart < text.Length && char.IsWhiteSpace(text[trimmedStart]))
                            trimmedStart++;

                        var trimmedEnd = currentStart + currentLength;
                        while (trimmedEnd > trimmedStart && char.IsWhiteSpace(text[trimmedEnd - 1]))
                            trimmedEnd--;

                        results.Add((sentence, trimmedStart, trimmedEnd));
                    }

                    currentStart = i + 1;
                    currentLength = 0;
                }
            }
        }

        // Handle any remaining text as final sentence
        if (currentLength > 0)
        {
            var sentence = text.Substring(currentStart, currentLength).Trim();
            if (!string.IsNullOrWhiteSpace(sentence))
            {
                var trimmedStart = currentStart;
                while (trimmedStart < text.Length && char.IsWhiteSpace(text[trimmedStart]))
                    trimmedStart++;

                var trimmedEnd = currentStart + currentLength;
                while (trimmedEnd > trimmedStart && char.IsWhiteSpace(text[trimmedEnd - 1]))
                    trimmedEnd--;

                results.Add((sentence, trimmedStart, trimmedEnd));
            }
        }

        return results;
    }
    private List<string> SplitIntoSentences(string text)
    {
        return Helpers.TextProcessingHelper.SplitIntoSentences(text);
    }
}
