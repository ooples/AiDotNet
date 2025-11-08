namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for text chunking strategies that split documents into smaller segments.
/// </summary>
/// <remarks>
/// <para>
/// A chunking strategy determines how to divide large text documents into smaller, manageable pieces.
/// This is essential for RAG systems because embedding models have maximum token limits, and smaller
/// chunks enable more precise retrieval. Different strategies balance between preserving context
/// and creating appropriately-sized segments.
/// </para>
/// <para><b>For Beginners:</b> A chunking strategy is like deciding how to slice a pizza.
/// 
/// Think of different ways to divide a long document:
/// - Fixed-size chunks: Cut every 500 words (like equal pizza slices)
/// - Sentence-based: Keep sentences together (like cutting between toppings)
/// - Paragraph-based: Keep paragraphs intact (like cutting by sections)
/// - Semantic: Group related content (like separating different flavor sections)
/// 
/// Why chunk documents?
/// - Long documents don't fit in the AI model (like a pizza too big for one plate)
/// - Smaller chunks make search more precise (finding exactly the relevant part)
/// - You can retrieve just the relevant sections, not entire documents
/// 
/// For example, searching a 100-page manual:
/// - Without chunking: Return the entire manual (overwhelming)
/// - With chunking: Return just the 2 paragraphs that answer your question (perfect!)
/// </para>
/// </remarks>
public interface IChunkingStrategy
{
    /// <summary>
    /// Gets the target size for each chunk in characters.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The chunk size determines the approximate length of each text segment.
    /// Actual chunks may be slightly larger or smaller depending on the strategy's
    /// rules for preserving sentence or paragraph boundaries.
    /// </para>
    /// <para><b>For Beginners:</b> This is how long each piece should be.
    /// 
    /// Common sizes:
    /// - Small (200-500 chars): Very focused, many chunks, precise retrieval
    /// - Medium (500-1000 chars): Balanced, typical choice
    /// - Large (1000-2000 chars): More context, fewer chunks
    /// 
    /// (Note: 500 characters ≈ 75-100 words ≈ 1-2 paragraphs)
    /// </para>
    /// </remarks>
    int ChunkSize { get; }

    /// <summary>
    /// Gets the number of characters that should overlap between consecutive chunks.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Overlap helps preserve context across chunk boundaries. When chunks overlap,
    /// information that spans the boundary appears in both chunks, preventing loss
    /// of meaning. Typical overlap is 10-20% of chunk size.
    /// </para>
    /// <para><b>For Beginners:</b> This is how much chunks should overlap with each other.
    /// 
    /// Why overlap?
    /// Imagine splitting this sentence at the wor|d boundary:
    /// - Chunk 1 ends with: "...at the wor"
    /// - Chunk 2 starts with: "d boundary..."
    /// - Neither chunk contains the complete word!
    /// 
    /// With overlap:
    /// - Chunk 1 ends with: "...at the word boundary. The next..."
    /// - Chunk 2 starts with: "...word boundary. The next sentence..."
    /// - Both chunks have complete context
    /// 
    /// Typical overlap: 50-100 characters (10-20% of chunk size)
    /// </para>
    /// </remarks>
    int ChunkOverlap { get; }

    /// <summary>
    /// Splits a text string into chunks according to the strategy's rules.
    /// </summary>
    /// <param name="text">The text to split into chunks.</param>
    /// <returns>A collection of text chunks, ordered as they appear in the original text.</returns>
    /// <remarks>
    /// <para>
    /// This method divides the input text into smaller segments according to the strategy's
    /// chunking rules. Chunks are returned in the order they appear in the source text.
    /// Each chunk should be self-contained enough to be meaningful when retrieved independently.
    /// </para>
    /// <para><b>For Beginners:</b> This breaks up a long text into smaller pieces.
    /// 
    /// For example:
    /// - Input: A 5000-word article about climate change
    /// - ChunkSize: 500 characters
    /// - ChunkOverlap: 50 characters
    /// - Output: ~25 chunks, each containing a portion of the article
    /// 
    /// Each chunk is like a mini-document that can be searched independently.
    /// When you search for "renewable energy", only the chunks about renewable energy
    /// get returned, not the entire 5000-word article.
    /// </para>
    /// </remarks>
    IEnumerable<string> Chunk(string text);

    /// <summary>
    /// Splits a text string into chunks and returns them with position metadata.
    /// </summary>
    /// <param name="text">The text to split into chunks.</param>
    /// <returns>A collection of tuples containing each chunk, its start position, and end position in the original text.</returns>
    /// <remarks>
    /// <para>
    /// This method provides additional metadata about where each chunk appears in the
    /// original text. This is useful for citation extraction, highlighting, or reconstructing
    /// the original document structure. Positions are character offsets from the start of the text.
    /// </para>
    /// <para><b>For Beginners:</b> This breaks up text and tells you where each piece came from.
    /// 
    /// For example:
    /// - Input: "The quick brown fox jumps over the lazy dog"
    /// - ChunkSize: 15, ChunkOverlap: 5
    /// - Output:
    ///   ("The quick brown", start: 0, end: 15)
    ///   ("brown fox jumps", start: 10, end: 25)
    ///   ("jumps over the", start: 20, end: 34)
    /// 
    /// Why is this useful?
    /// When the AI uses a chunk in its answer, you can show the user:
    /// "This information came from characters 20-34 of the original document"
    /// 
    /// This enables precise citations and highlighting.
    /// </para>
    /// </remarks>
    IEnumerable<(string Chunk, int StartPosition, int EndPosition)> ChunkWithPositions(string text);
}
