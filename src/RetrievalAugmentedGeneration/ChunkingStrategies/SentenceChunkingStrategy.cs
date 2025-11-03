using AiDotNet.RetrievalAugmentedGeneration.Interfaces;

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
    {
        if (targetChunkSize <= 0)
        {
            throw new ArgumentException("Target chunk size must be positive.", nameof(targetChunkSize));
        }

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
    /// <returns>An enumerable of text chunks.</returns>
    protected override IEnumerable<string> ChunkCore(string text)
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

        return chunks;
    }

    /// <summary>
    /// Splits text into individual sentences.
    /// </summary>
    /// <param name="text">The text to split.</param>
    /// <returns>A list of sentences.</returns>
    private List<string> SplitIntoSentences(string text)
    {
        var sentences = new List<string>();
        var currentSentence = new System.Text.StringBuilder();

        for (int i = 0; i < text.Length; i++)
        {
            currentSentence.Append(text[i]);

            // Check if this is a sentence ender
            if (Array.IndexOf(SentenceEnders, text[i]) >= 0)
            {
                // Look ahead to see if there's whitespace (actual sentence end)
                if (i + 1 < text.Length && char.IsWhiteSpace(text[i + 1]))
                {
                    var sentence = currentSentence.ToString().Trim();
                    if (!string.IsNullOrWhiteSpace(sentence))
                    {
                        sentences.Add(sentence);
                    }
                    currentSentence.Clear();
                }
                // Also check for end of text
                else if (i + 1 == text.Length)
                {
                    var sentence = currentSentence.ToString().Trim();
                    if (!string.IsNullOrWhiteSpace(sentence))
                    {
                        sentences.Add(sentence);
                    }
                    currentSentence.Clear();
                }
            }
        }

        // Add any remaining text as a final sentence
        if (currentSentence.Length > 0)
        {
            var sentence = currentSentence.ToString().Trim();
            if (!string.IsNullOrWhiteSpace(sentence))
            {
                sentences.Add(sentence);
            }
        }

        return sentences;
    }
}
