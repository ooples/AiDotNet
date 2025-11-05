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
    /// Splits text into chunks at sentence boundaries.
    /// </summary>
    /// <param name="text">The validated text to split.</param>
    /// <returns>Chunks with position information.</returns>
    protected override IEnumerable<(string Chunk, int StartPosition, int EndPosition)> ChunkCore(string text)
    {
        var sentences = SplitIntoSentences(text);
        var chunks = new List<string>();
        var currentChunk = new List<string>();
        var currentLength = 0;

        for (int i = 0; i < sentences.Count; i++)
        {
            var sentence = sentences[i];
            var sentenceLength = sentence.Length;

            // If adding this sentence would exceed maxChunkSize and we have content, create a chunk
            if (currentLength + sentenceLength > _maxChunkSize && currentChunk.Count > 0)
            {
                chunks.Add(string.Join(" ", currentChunk));
                
                // Keep the last N sentences for overlap
                var overlapStart = Math.Max(0, currentChunk.Count - _overlapSentences);
                currentChunk = currentChunk.GetRange(overlapStart, currentChunk.Count - overlapStart);
                currentLength = currentChunk.Sum(s => s.Length + 1) - 1; // +1 for space, -1 to remove last space
            }
            
            // Handle sentences that exceed maxChunkSize on their own
            if (sentenceLength > _maxChunkSize)
            {
                // If we have accumulated content, save it first
                if (currentChunk.Count > 0)
                {
                    chunks.Add(string.Join(" ", currentChunk));
                    currentChunk.Clear();
                    currentLength = 0;
                }
                
                // Split the oversized sentence into smaller pieces
                for (int pos = 0; pos < sentence.Length; pos += _maxChunkSize)
                {
                    var pieceLength = Math.Min(_maxChunkSize, sentence.Length - pos);
                    chunks.Add(sentence.Substring(pos, pieceLength));
                }
                
                continue;
            }

            currentChunk.Add(sentence);
            currentLength += sentenceLength + (currentChunk.Count > 1 ? 1 : 0); // Add space if not first sentence

            // If we've reached target size, create a chunk
            if (currentLength >= _targetChunkSize)
            {
                chunks.Add(string.Join(" ", currentChunk));
                
                // Keep the last N sentences for overlap
                var overlapStart = Math.Max(0, currentChunk.Count - _overlapSentences);
                currentChunk = currentChunk.GetRange(overlapStart, currentChunk.Count - overlapStart);
                currentLength = currentChunk.Sum(s => s.Length + 1) - 1;
            }
        }

        // Add remaining sentences as final chunk
        if (currentChunk.Count > 0)
        {
            chunks.Add(string.Join(" ", currentChunk));
        }

        // Convert to tuples with positions (track actual positions in original text)
        var results = new List<(string, int, int)>();
        var searchPos = 0;
        
        foreach (var chunk in chunks)
        {
            // Find where this chunk appears in the original text
            var startPos = text.IndexOf(chunk, searchPos, StringComparison.Ordinal);
            if (startPos == -1)
            {
                // Fallback: if exact match not found (shouldn't happen), use sequential position
                startPos = searchPos;
            }
            
            var endPos = startPos + chunk.Length;
            results.Add((chunk, startPos, endPos));
            
            // Move search position forward, accounting for overlap
            searchPos = startPos + 1;
        }

        return results;
    }

    /// <summary>
    /// Splits text into individual sentences.
    /// </summary>
    /// <param name="text">The text to split.</param>
    /// <returns>A list of sentences.</returns>
    private List<string> SplitIntoSentences(string text)
    {
        return Helpers.TextProcessingHelper.SplitIntoSentences(text);
    }
}
