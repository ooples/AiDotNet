

namespace AiDotNet.RetrievalAugmentedGeneration.ChunkingStrategies;

/// <summary>
/// A simple fixed-size text chunking strategy that splits text at character boundaries.
/// </summary>
/// <remarks>
/// <para>
/// This implementation splits text into fixed-size chunks with optional overlap.
/// It's the simplest chunking strategy and works well for general text when semantic
/// boundaries (sentences, paragraphs) are not critical. The chunking is purely character-based
/// and does not consider word or sentence boundaries.
/// </para>
/// <para><b>For Beginners:</b> This splits text into equal-sized pieces like slicing bread.
/// 
/// Think of it like cutting a loaf of bread into equal slices:
/// - Each chunk is the same size (ChunkSize characters)
/// - Slices overlap a bit (ChunkOverlap characters)
/// - Cuts happen at any point, even mid-word
/// 
/// How it works:
/// - ChunkSize = 500 means each piece is about 500 characters
/// - ChunkOverlap = 50 means pieces overlap by 50 characters
/// - No special handling for sentences or words
/// 
/// Good for:
/// - Simple text processing
/// - When you don't care about word boundaries
/// - Consistent chunk sizes
/// - Fast processing
/// 
/// Not ideal for:
/// - Preserving sentence meaning (might cut mid-sentence)
/// - Code or structured data
/// - When semantic coherence is critical
/// 
/// For example, with ChunkSize=20, ChunkOverlap=5:
/// Input: "The quick brown fox jumps over the lazy dog"
/// Chunk 1: "The quick brown fox " (0-20)
/// Chunk 2: "n fox jumps over the" (15-35) - overlaps with Chunk 1
/// Chunk 3: " the lazy dog" (30-43) - overlaps with Chunk 2
/// </para>
/// </remarks>
public class FixedSizeChunkingStrategy : ChunkingStrategyBase
{
    /// <summary>
    /// Initializes a new instance of the FixedSizeChunkingStrategy class.
    /// </summary>
    /// <param name="chunkSize">The target size for each chunk in characters (default: 500).</param>
    /// <param name="chunkOverlap">The number of characters that should overlap between consecutive chunks (default: 50).</param>
    public FixedSizeChunkingStrategy(int chunkSize = 500, int chunkOverlap = 50)
        : base(chunkSize, chunkOverlap)
    {
    }

    /// <summary>
    /// Core chunking logic using simple fixed-size character splitting.
    /// </summary>
    /// <param name="text">The validated text to split into chunks.</param>
    /// <returns>A collection of tuples containing each chunk and its position information.</returns>
    protected override IEnumerable<(string Chunk, int StartPosition, int EndPosition)> ChunkCore(string text)
    {
        // Use the base class helper for simple character-based chunking
        return CreateOverlappingChunks(text);
    }
}
