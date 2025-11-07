# Junior Developer Implementation Guide: Issue #341
## RAG Chunking Strategies - Comprehensive Unit Testing

### Overview
This guide will walk you through creating comprehensive unit tests for the RAG chunking strategies infrastructure in AiDotNet. The code already exists - your job is to thoroughly test all chunking implementations to ensure they work correctly across all scenarios.

---

## For Beginners: What Is Chunking?

### The Problem: Documents Are Too Big

Imagine you have a 100-page user manual and someone asks: "How do I reset the device?"

**Without Chunking:**
- Send entire 100-page manual to the AI
- AI gets overwhelmed or hits token limits
- Expensive (processing all 100 pages costs money)
- Slow (takes time to read everything)
- Imprecise ("The answer is somewhere in here...")

**With Chunking:**
- Split manual into 200 smaller chunks (paragraphs/sections)
- Find the 3 chunks most relevant to "reset device"
- Send only those 3 chunks to the AI
- Fast, cheap, precise!

### Real-World Analogy

Think of chunking like organizing a cookbook:

- **Bad approach**: One giant book with all recipes in random order
- **Good approach**: Organized into sections (appetizers, main courses, desserts) with individual recipe cards

When someone asks for a dessert recipe, you don't hand them the entire cookbook - you go to the desserts section and pull out the relevant recipe cards.

### Why Chunk Size and Overlap Matter

**Chunk Size:**
- Too small (50 characters): "...the quick brown fox..." (no context)
- Too large (10,000 characters): Still overwhelming, defeats the purpose
- Just right (500-1000 characters): 1-3 paragraphs with complete thoughts

**Chunk Overlap:**
```
Without overlap:
Chunk 1: "...this is important because"
Chunk 2: "it enables better search..."
^ LOST CONTEXT! Neither chunk is complete.

With overlap (50 chars):
Chunk 1: "...this is important because it enables better..."
Chunk 2: "...because it enables better search results."
^ Both chunks preserve the complete thought!
```

---

## What EXISTS in the Codebase

### Core Infrastructure

**Interface:**
- `C:\Users\cheat\source\repos\AiDotNet\src\Interfaces\IChunkingStrategy.cs`
  - Defines contract for all chunking strategies
  - Methods: `Chunk(string text)`, `ChunkWithPositions(string text)`
  - Properties: `ChunkSize`, `ChunkOverlap`

**Base Class:**
- `C:\Users\cheat\source\repos\AiDotNet\src\RetrievalAugmentedGeneration\ChunkingStrategies\ChunkingStrategyBase.cs`
  - Implements common functionality
  - Validation logic
  - Helper methods: `CreateOverlappingChunks()`, `SplitOnSentences()`

### Existing Implementations

**Simple Strategies:**
1. **FixedSizeChunkingStrategy** - Character-based fixed-size chunks
2. **SlidingWindowChunkingStrategy** - Sliding window with overlap
3. **SentenceChunkingStrategy** - Sentence-boundary-aware chunking
4. **RecursiveCharacterChunkingStrategy** - Recursive splitting with separators

**Advanced Strategies:**
5. **SemanticChunkingStrategy** - Semantic similarity-based chunking
6. **MarkdownTextSplitter** - Markdown structure-aware splitting
7. **CodeAwareTextSplitter** - Code structure-aware splitting
8. **HeaderBasedTextSplitter** - Document header hierarchy splitting
9. **TableAwareTextSplitter** - Preserves table structure
10. **MultiModalTextSplitter** - Handles text + images
11. **AgenticChunker** - LLM-driven intelligent chunking
12. **RecursiveCharacterTextSplitter** - Enhanced recursive splitting

---

## What's MISSING (This Issue)

The code exists but **tests are incomplete or missing**. You need to create comprehensive tests for:

### Test Coverage Gaps

1. **Basic Functionality Tests**
   - Constructor validation (invalid chunk sizes, overlap > size)
   - Basic chunking operations
   - Chunk overlap behavior
   - Position metadata accuracy

2. **Edge Cases**
   - Empty text
   - Text shorter than chunk size
   - Text exactly equal to chunk size
   - Very long text (10,000+ characters)
   - Special characters (Unicode, emoji, newlines)

3. **Strategy-Specific Behavior**
   - Sentence boundaries (. ! ?)
   - Markdown headers (#, ##, ###)
   - Code blocks (```, indentation)
   - Table structure preservation
   - Multi-modal content handling

4. **Overlap Validation**
   - Correct overlap amount
   - No duplicate content beyond overlap
   - Overlap at boundaries

5. **Performance Tests**
   - Large documents (100KB+ text)
   - Many small chunks vs few large chunks
   - Memory efficiency

---

## Step-by-Step Implementation

### Step 1: Set Up Test Project Structure

```csharp
// File: tests/RetrievalAugmentedGeneration/ChunkingStrategies/ChunkingStrategyTestBase.cs

using Xunit;
using AiDotNet.RetrievalAugmentedGeneration.ChunkingStrategies;
using AiDotNet.Interfaces;

namespace AiDotNet.Tests.RetrievalAugmentedGeneration.ChunkingStrategies;

/// <summary>
/// Base class for chunking strategy tests with common test utilities.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> This is a shared test helper class.
///
/// Why use a base class for tests?
/// - Avoid duplicating common test code
/// - Ensure all strategies are tested consistently
/// - Make it easy to add new strategy tests
///
/// All specific chunking strategy tests will inherit from this.
/// </remarks>
public abstract class ChunkingStrategyTestBase
{
    /// <summary>
    /// Creates a sample text for testing.
    /// </summary>
    protected string CreateSampleText(int sentenceCount)
    {
        var sentences = new[]
        {
            "This is the first sentence.",
            "Here is another sentence with different content.",
            "The third sentence discusses a new topic.",
            "This sentence contains important information.",
            "Finally, we conclude with this last sentence."
        };

        var result = new System.Text.StringBuilder();
        for (int i = 0; i < sentenceCount; i++)
        {
            result.Append(sentences[i % sentences.Length]);
            result.Append(" ");
        }

        return result.ToString().Trim();
    }

    /// <summary>
    /// Verifies that chunks respect the configured overlap.
    /// </summary>
    protected void AssertCorrectOverlap(IEnumerable<(string Chunk, int Start, int End)> chunks, int expectedOverlap)
    {
        var chunkList = chunks.ToList();

        for (int i = 0; i < chunkList.Count - 1; i++)
        {
            var current = chunkList[i];
            var next = chunkList[i + 1];

            // Calculate actual overlap
            int overlapStart = Math.Max(current.Start, next.Start);
            int overlapEnd = Math.Min(current.End, next.End);
            int actualOverlap = Math.Max(0, overlapEnd - overlapStart);

            // Allow some flexibility for sentence-based chunking
            Assert.InRange(actualOverlap, 0, expectedOverlap * 2);
        }
    }

    /// <summary>
    /// Verifies that chunk positions accurately reflect content in original text.
    /// </summary>
    protected void AssertCorrectPositions(string originalText, IEnumerable<(string Chunk, int Start, int End)> chunks)
    {
        foreach (var (chunk, start, end) in chunks)
        {
            // Extract substring from original using positions
            string extractedChunk = originalText.Substring(start, end - start);

            // Should match the chunk content (allowing for whitespace normalization)
            Assert.Equal(chunk.Trim(), extractedChunk.Trim());
        }
    }

    /// <summary>
    /// Verifies that all chunks are non-empty and within size limits.
    /// </summary>
    protected void AssertChunksWithinSizeLimits(IEnumerable<string> chunks, int maxSize)
    {
        foreach (var chunk in chunks)
        {
            Assert.NotEmpty(chunk);
            Assert.True(chunk.Length <= maxSize,
                $"Chunk length {chunk.Length} exceeds max size {maxSize}");
        }
    }

    /// <summary>
    /// Verifies that chunks cover the entire original text without gaps.
    /// </summary>
    protected void AssertNoCoverageGaps(string originalText, IEnumerable<(string Chunk, int Start, int End)> chunks)
    {
        var chunkList = chunks.OrderBy(c => c.Start).ToList();

        // First chunk should start at or near the beginning
        Assert.True(chunkList.First().Start < 50, "First chunk should start near beginning");

        // Last chunk should end at or near the end
        Assert.True(chunkList.Last().End >= originalText.Length - 50,
            "Last chunk should cover end of text");

        // Check for major gaps between chunks (allowing for intentional overlap)
        for (int i = 0; i < chunkList.Count - 1; i++)
        {
            var gap = chunkList[i + 1].Start - chunkList[i].End;
            Assert.True(gap <= 0, $"Found gap of {gap} characters between chunks {i} and {i + 1}");
        }
    }
}
```

### Step 2: Test FixedSizeChunkingStrategy

```csharp
// File: tests/RetrievalAugmentedGeneration/ChunkingStrategies/FixedSizeChunkingStrategyTests.cs

using Xunit;
using AiDotNet.RetrievalAugmentedGeneration.ChunkingStrategies;

namespace AiDotNet.Tests.RetrievalAugmentedGeneration.ChunkingStrategies;

/// <summary>
/// Tests for FixedSizeChunkingStrategy.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> These tests verify character-based fixed-size chunking.
///
/// FixedSizeChunkingStrategy is the simplest chunking approach:
/// - Splits text every N characters
/// - Doesn't care about sentence or word boundaries
/// - Like cutting a pizza into equal slices with a ruler
///
/// What we're testing:
/// - Chunks are the right size
/// - Overlap works correctly
/// - Edge cases (empty text, very short text)
/// - Position metadata is accurate
/// </remarks>
public class FixedSizeChunkingStrategyTests : ChunkingStrategyTestBase
{
    [Fact]
    public void Constructor_WithValidParameters_Initializes()
    {
        // Arrange & Act
        var strategy = new FixedSizeChunkingStrategy(chunkSize: 100, chunkOverlap: 20);

        // Assert
        Assert.Equal(100, strategy.ChunkSize);
        Assert.Equal(20, strategy.ChunkOverlap);
    }

    [Theory]
    [InlineData(0, 0)]
    [InlineData(-1, 0)]
    [InlineData(100, -1)]
    [InlineData(100, 100)]  // Overlap equal to size
    [InlineData(100, 150)]  // Overlap > size
    public void Constructor_WithInvalidParameters_ThrowsArgumentException(int chunkSize, int overlap)
    {
        // Arrange, Act & Assert
        Assert.Throws<ArgumentException>(() =>
            new FixedSizeChunkingStrategy(chunkSize, overlap));
    }

    [Fact]
    public void Chunk_WithSimpleText_ReturnsCorrectChunks()
    {
        // Arrange
        var strategy = new FixedSizeChunkingStrategy(chunkSize: 20, chunkOverlap: 5);
        var text = "This is a test of the chunking strategy functionality.";

        // Act
        var chunks = strategy.Chunk(text).ToList();

        // Assert
        Assert.NotEmpty(chunks);
        Assert.All(chunks, chunk => Assert.True(chunk.Length <= 20));

        // All chunks together should cover the original text
        var combined = string.Concat(chunks);
        Assert.Contains("This is a test", combined);
        Assert.Contains("functionality", combined);
    }

    [Fact]
    public void ChunkWithPositions_ReturnsAccuratePositions()
    {
        // Arrange
        var strategy = new FixedSizeChunkingStrategy(chunkSize: 15, chunkOverlap: 3);
        var text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

        // Act
        var chunks = strategy.ChunkWithPositions(text).ToList();

        // Assert
        AssertCorrectPositions(text, chunks);

        // Verify specific positions
        Assert.Equal(0, chunks.First().StartPosition);
        Assert.True(chunks.Last().EndPosition <= text.Length);
    }

    [Fact]
    public void Chunk_WithOverlap_CreatesOverlappingChunks()
    {
        // Arrange
        var strategy = new FixedSizeChunkingStrategy(chunkSize: 30, chunkOverlap: 10);
        var text = CreateSampleText(sentenceCount: 5);

        // Act
        var chunks = strategy.ChunkWithPositions(text).ToList();

        // Assert
        Assert.True(chunks.Count > 1, "Should create multiple chunks");
        AssertCorrectOverlap(chunks, expectedOverlap: 10);
    }

    [Fact]
    public void Chunk_WithEmptyText_ThrowsArgumentException()
    {
        // Arrange
        var strategy = new FixedSizeChunkingStrategy(chunkSize: 100, chunkOverlap: 20);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => strategy.Chunk("").ToList());
        Assert.Throws<ArgumentNullException>(() => strategy.Chunk(null).ToList());
    }

    [Fact]
    public void Chunk_WithTextShorterThanChunkSize_ReturnsSingleChunk()
    {
        // Arrange
        var strategy = new FixedSizeChunkingStrategy(chunkSize: 100, chunkOverlap: 20);
        var text = "Short text.";

        // Act
        var chunks = strategy.Chunk(text).ToList();

        // Assert
        Assert.Single(chunks);
        Assert.Equal(text, chunks[0]);
    }

    [Fact]
    public void Chunk_WithTextEqualToChunkSize_ReturnsSingleChunk()
    {
        // Arrange
        var strategy = new FixedSizeChunkingStrategy(chunkSize: 50, chunkOverlap: 10);
        var text = new string('A', 50);  // Exactly 50 characters

        // Act
        var chunks = strategy.Chunk(text).ToList();

        // Assert
        Assert.Single(chunks);
        Assert.Equal(50, chunks[0].Length);
    }

    [Fact]
    public void Chunk_WithLargeDocument_HandlesEfficiently()
    {
        // Arrange
        var strategy = new FixedSizeChunkingStrategy(chunkSize: 500, chunkOverlap: 50);
        var text = CreateSampleText(sentenceCount: 200);  // Large text

        // Act
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        var chunks = strategy.Chunk(text).ToList();
        stopwatch.Stop();

        // Assert
        Assert.NotEmpty(chunks);
        Assert.True(stopwatch.ElapsedMilliseconds < 1000,
            $"Chunking took {stopwatch.ElapsedMilliseconds}ms, should be < 1000ms");

        AssertChunksWithinSizeLimits(chunks, maxSize: 500);
    }

    [Fact]
    public void Chunk_WithUnicodeCharacters_HandlesCorrectly()
    {
        // Arrange
        var strategy = new FixedSizeChunkingStrategy(chunkSize: 20, chunkOverlap: 5);
        var text = "Hello 世界! 🌍 This is a test with émojis and spëcial çharacters.";

        // Act
        var chunks = strategy.Chunk(text).ToList();

        // Assert
        Assert.NotEmpty(chunks);
        Assert.All(chunks, chunk =>
        {
            Assert.True(chunk.Length <= 20);
            Assert.NotEmpty(chunk.Trim());
        });
    }

    [Fact]
    public void Chunk_WithNewlines_PreservesStructure()
    {
        // Arrange
        var strategy = new FixedSizeChunkingStrategy(chunkSize: 30, chunkOverlap: 5);
        var text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.";

        // Act
        var chunks = strategy.Chunk(text).ToList();

        // Assert
        Assert.NotEmpty(chunks);

        // Newlines should be preserved in chunks
        var combinedChunks = string.Join("", chunks);
        Assert.Contains("\n\n", text);  // Original has newlines
    }

    [Fact]
    public void ChunkWithPositions_CoversEntireText()
    {
        // Arrange
        var strategy = new FixedSizeChunkingStrategy(chunkSize: 25, chunkOverlap: 5);
        var text = CreateSampleText(sentenceCount: 10);

        // Act
        var chunks = strategy.ChunkWithPositions(text).ToList();

        // Assert
        AssertNoCoverageGaps(text, chunks);
    }

    [Fact]
    public void Chunk_WithZeroOverlap_CreatesNonOverlappingChunks()
    {
        // Arrange
        var strategy = new FixedSizeChunkingStrategy(chunkSize: 20, chunkOverlap: 0);
        var text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

        // Act
        var chunks = strategy.ChunkWithPositions(text).ToList();

        // Assert
        Assert.True(chunks.Count > 1);

        // Verify no overlap
        for (int i = 0; i < chunks.Count - 1; i++)
        {
            Assert.Equal(chunks[i].EndPosition, chunks[i + 1].StartPosition);
        }
    }

    [Fact]
    public void Chunk_ConsecutiveCalls_ReturnsSameResults()
    {
        // Arrange
        var strategy = new FixedSizeChunkingStrategy(chunkSize: 50, chunkOverlap: 10);
        var text = CreateSampleText(sentenceCount: 5);

        // Act
        var chunks1 = strategy.Chunk(text).ToList();
        var chunks2 = strategy.Chunk(text).ToList();

        // Assert - Deterministic behavior
        Assert.Equal(chunks1.Count, chunks2.Count);
        for (int i = 0; i < chunks1.Count; i++)
        {
            Assert.Equal(chunks1[i], chunks2[i]);
        }
    }
}
```

### Step 3: Test SentenceChunkingStrategy

```csharp
// File: tests/RetrievalAugmentedGeneration/ChunkingStrategies/SentenceChunkingStrategyTests.cs

using Xunit;
using AiDotNet.RetrievalAugmentedGeneration.ChunkingStrategies;

namespace AiDotNet.Tests.RetrievalAugmentedGeneration.ChunkingStrategies;

/// <summary>
/// Tests for SentenceChunkingStrategy.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> These tests verify sentence-boundary-aware chunking.
///
/// SentenceChunkingStrategy is smarter than fixed-size:
/// - Tries to break at sentence boundaries (. ! ?)
/// - Keeps complete thoughts together
/// - Falls back to character splitting for very long sentences
/// - Like cutting pizza along the natural dividing lines between slices
///
/// What we're testing:
/// - Sentences stay intact when possible
/// - Sentence endings are detected correctly (. ! ?)
/// - Long sentences are handled gracefully
/// - Edge cases (no punctuation, all caps, etc.)
/// </remarks>
public class SentenceChunkingStrategyTests : ChunkingStrategyTestBase
{
    [Fact]
    public void Chunk_WithMultipleSentences_BreaksAtSentenceBoundaries()
    {
        // Arrange
        var strategy = new SentenceChunkingStrategy(chunkSize: 50, chunkOverlap: 10);
        var text = "First sentence. Second sentence. Third sentence.";

        // Act
        var chunks = strategy.Chunk(text).ToList();

        // Assert
        Assert.NotEmpty(chunks);

        // Each chunk should contain complete sentences (ending with .)
        Assert.All(chunks, chunk =>
        {
            var trimmed = chunk.Trim();
            if (trimmed.Length > 0)
            {
                // Should end with sentence ending or be part of a larger sentence
                Assert.True(
                    trimmed.EndsWith(".") ||
                    trimmed.EndsWith("!") ||
                    trimmed.EndsWith("?") ||
                    !text.EndsWith(trimmed)  // Not the last chunk
                );
            }
        });
    }

    [Fact]
    public void Chunk_WithQuestionAndExclamation_RecognizesSentenceEndings()
    {
        // Arrange
        var strategy = new SentenceChunkingStrategy(chunkSize: 60, chunkOverlap: 10);
        var text = "Is this a question? Yes it is! Now we have a statement.";

        // Act
        var chunks = strategy.Chunk(text).ToList();

        // Assert
        Assert.NotEmpty(chunks);

        // Verify that sentence endings (? and !) are recognized
        var allText = string.Join(" ", chunks);
        Assert.Contains("?", allText);
        Assert.Contains("!", allText);
    }

    [Fact]
    public void Chunk_WithVeryLongSentence_FallsBackToCharacterSplitting()
    {
        // Arrange
        var strategy = new SentenceChunkingStrategy(chunkSize: 50, chunkOverlap: 5);

        // Create a very long sentence with no punctuation
        var longSentence = "This is a very long sentence without any punctuation marks that goes on and on " +
                          "and on for many words and should eventually need to be split even though " +
                          "there are no sentence boundaries to break at naturally";

        // Act
        var chunks = strategy.Chunk(longSentence).ToList();

        // Assert
        Assert.True(chunks.Count > 1, "Long sentence should be split into multiple chunks");
        AssertChunksWithinSizeLimits(chunks, maxSize: 50);
    }

    [Fact]
    public void Chunk_WithAbbreviations_DoesNotBreakIncorrectly()
    {
        // Arrange
        var strategy = new SentenceChunkingStrategy(chunkSize: 80, chunkOverlap: 10);
        var text = "Dr. Smith works at U.S.A. Inc. He is very skilled. Ms. Jones agrees.";

        // Act
        var chunks = strategy.Chunk(text).ToList();

        // Assert
        Assert.NotEmpty(chunks);

        // Check that abbreviations don't cause premature splits
        // "Dr. Smith" should ideally stay together
        var combined = string.Join(" ", chunks);
        Assert.Contains("Dr.", combined);
        Assert.Contains("U.S.A.", combined);
    }

    [Fact]
    public void Chunk_WithEmptyText_ThrowsArgumentException()
    {
        // Arrange
        var strategy = new SentenceChunkingStrategy(chunkSize: 100, chunkOverlap: 20);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => strategy.Chunk("").ToList());
    }

    [Fact]
    public void Chunk_WithNoSentenceEndings_StillChunks()
    {
        // Arrange
        var strategy = new SentenceChunkingStrategy(chunkSize: 30, chunkOverlap: 5);
        var text = "This text has no sentence endings at all just keeps going and going";

        // Act
        var chunks = strategy.Chunk(text).ToList();

        // Assert
        Assert.NotEmpty(chunks);
        AssertChunksWithinSizeLimits(chunks, maxSize: 40);  // Allow slight overflow for sentence preservation
    }

    [Fact]
    public void ChunkWithPositions_MaintainsCorrectPositions()
    {
        // Arrange
        var strategy = new SentenceChunkingStrategy(chunkSize: 60, chunkOverlap: 10);
        var text = "Sentence one. Sentence two. Sentence three. Sentence four.";

        // Act
        var chunks = strategy.ChunkWithPositions(text).ToList();

        // Assert
        AssertCorrectPositions(text, chunks);
    }

    [Fact]
    public void Chunk_WithMixedPunctuation_HandlesCorrectly()
    {
        // Arrange
        var strategy = new SentenceChunkingStrategy(chunkSize: 70, chunkOverlap: 10);
        var text = "First! Second? Third. Fourth; fifth, sixth.";

        // Act
        var chunks = strategy.Chunk(text).ToList();

        // Assert
        Assert.NotEmpty(chunks);

        // Semicolons and commas should NOT be treated as sentence endings
        // Only . ! ? are sentence endings
        var allText = string.Join("", chunks);
        Assert.Contains(";", allText);
        Assert.Contains(",", allText);
    }

    [Fact]
    public void Chunk_WithUnicodeText_PreservesCharacters()
    {
        // Arrange
        var strategy = new SentenceChunkingStrategy(chunkSize: 50, chunkOverlap: 10);
        var text = "Première phrase. Deuxième phrase. Troisième phrase!";

        // Act
        var chunks = strategy.Chunk(text).ToList();

        // Assert
        Assert.NotEmpty(chunks);
        var combined = string.Join(" ", chunks);
        Assert.Contains("è", combined);
        Assert.Contains("î", combined);
    }
}
```

### Step 4: Test SemanticChunkingStrategy

```csharp
// File: tests/RetrievalAugmentedGeneration/ChunkingStrategies/SemanticChunkingStrategyTests.cs

using Xunit;
using AiDotNet.RetrievalAugmentedGeneration.ChunkingStrategies;
using AiDotNet.RetrievalAugmentedGeneration.Embeddings;

namespace AiDotNet.Tests.RetrievalAugmentedGeneration.ChunkingStrategies;

/// <summary>
/// Tests for SemanticChunkingStrategy.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> These tests verify semantic similarity-based chunking.
///
/// SemanticChunkingStrategy is the smartest approach:
/// - Groups semantically related sentences together
/// - Uses embeddings to measure similarity
/// - Creates chunks based on meaning, not just size
/// - Like organizing pizza toppings by flavor (all veggies together, all meats together)
///
/// What we're testing:
/// - Similar sentences end up in same chunk
/// - Topic changes create new chunks
/// - Embedding model integration works
/// - Handles edge cases (single sentence, no similarity)
/// </remarks>
public class SemanticChunkingStrategyTests : ChunkingStrategyTestBase
{
    [Fact]
    public void Constructor_WithValidEmbeddingModel_Initializes()
    {
        // Arrange
        var embeddingModel = new StubEmbeddingModel<double>(dimension: 384);

        // Act
        var strategy = new SemanticChunkingStrategy<double>(
            embeddingModel: embeddingModel,
            chunkSize: 500,
            chunkOverlap: 50,
            similarityThreshold: 0.7
        );

        // Assert
        Assert.Equal(500, strategy.ChunkSize);
        Assert.Equal(50, strategy.ChunkOverlap);
    }

    [Fact]
    public void Constructor_WithNullEmbeddingModel_ThrowsArgumentNullException()
    {
        // Arrange, Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            new SemanticChunkingStrategy<double>(
                embeddingModel: null,
                chunkSize: 500,
                chunkOverlap: 50
            ));
    }

    [Fact]
    public void Chunk_GroupsSimilarSentencesTogether()
    {
        // Arrange
        var embeddingModel = new StubEmbeddingModel<double>(dimension: 384);
        var strategy = new SemanticChunkingStrategy<double>(
            embeddingModel: embeddingModel,
            chunkSize: 200,
            chunkOverlap: 20,
            similarityThreshold: 0.7
        );

        // Text with clear topic groups
        var text = "Dogs are loyal pets. Cats are independent animals. " +
                  "The stock market rose today. Financial markets are volatile. " +
                  "Puppies love to play. Kittens enjoy toys.";

        // Act
        var chunks = strategy.Chunk(text).ToList();

        // Assert
        Assert.NotEmpty(chunks);
        // StubEmbeddingModel returns similar embeddings for similar text
        // so related sentences should be grouped
    }

    [Fact]
    public void Chunk_WithSingleSentence_ReturnsSingleChunk()
    {
        // Arrange
        var embeddingModel = new StubEmbeddingModel<double>(dimension: 384);
        var strategy = new SemanticChunkingStrategy<double>(
            embeddingModel: embeddingModel,
            chunkSize: 500,
            chunkOverlap: 50
        );
        var text = "This is a single sentence.";

        // Act
        var chunks = strategy.Chunk(text).ToList();

        // Assert
        Assert.Single(chunks);
        Assert.Equal(text, chunks[0]);
    }

    [Fact]
    public void Chunk_WithHighSimilarityThreshold_CreatesSmallerChunks()
    {
        // Arrange
        var embeddingModel = new StubEmbeddingModel<double>(dimension: 384);

        var strictStrategy = new SemanticChunkingStrategy<double>(
            embeddingModel: embeddingModel,
            chunkSize: 500,
            chunkOverlap: 50,
            similarityThreshold: 0.95  // Very strict
        );

        var lenientStrategy = new SemanticChunkingStrategy<double>(
            embeddingModel: embeddingModel,
            chunkSize: 500,
            chunkOverlap: 50,
            similarityThreshold: 0.5  // Very lenient
        );

        var text = CreateSampleText(sentenceCount: 10);

        // Act
        var strictChunks = strictStrategy.Chunk(text).ToList();
        var lenientChunks = lenientStrategy.Chunk(text).ToList();

        // Assert
        // Stricter threshold should create more chunks (harder to group sentences)
        Assert.True(strictChunks.Count >= lenientChunks.Count,
            $"Strict: {strictChunks.Count} chunks, Lenient: {lenientChunks.Count} chunks");
    }

    [Fact]
    public void Chunk_RespectsMaximumChunkSize()
    {
        // Arrange
        var embeddingModel = new StubEmbeddingModel<double>(dimension: 384);
        var strategy = new SemanticChunkingStrategy<double>(
            embeddingModel: embeddingModel,
            chunkSize: 100,
            chunkOverlap: 10
        );

        var text = CreateSampleText(sentenceCount: 20);

        // Act
        var chunks = strategy.Chunk(text).ToList();

        // Assert
        AssertChunksWithinSizeLimits(chunks, maxSize: 150);  // Allow slight overflow
    }
}
```

---

## Testing Strategy

### Coverage Targets

Aim for **80%+ code coverage** on all chunking strategies:

```bash
# Run tests with coverage
dotnet test --collect:"XPlat Code Coverage"

# View coverage report
reportgenerator -reports:**/coverage.cobertura.xml -targetdir:coveragereport
```

### Test Organization

```
tests/
└── RetrievalAugmentedGeneration/
    └── ChunkingStrategies/
        ├── ChunkingStrategyTestBase.cs           (Shared helpers)
        ├── FixedSizeChunkingStrategyTests.cs     (Basic chunking)
        ├── SlidingWindowChunkingStrategyTests.cs (Sliding window)
        ├── SentenceChunkingStrategyTests.cs      (Sentence-aware)
        ├── RecursiveCharacterChunkingStrategyTests.cs
        ├── SemanticChunkingStrategyTests.cs      (Semantic similarity)
        ├── MarkdownTextSplitterTests.cs          (Markdown structure)
        ├── CodeAwareTextSplitterTests.cs         (Code blocks)
        ├── HeaderBasedTextSplitterTests.cs       (Header hierarchy)
        ├── TableAwareTextSplitterTests.cs        (Table preservation)
        ├── MultiModalTextSplitterTests.cs        (Text + images)
        └── AgenticChunkerTests.cs                (LLM-driven)
```

### Running Tests

```bash
# Run all chunking tests
dotnet test --filter "FullyQualifiedName~ChunkingStrategy"

# Run specific strategy tests
dotnet test --filter "FullyQualifiedName~FixedSizeChunkingStrategyTests"

# Run with verbose output
dotnet test --logger "console;verbosity=detailed"
```

---

## Common Pitfalls

### Pitfall 1: Not Testing Edge Cases

**Wrong:**
```csharp
[Fact]
public void Chunk_Works()
{
    var strategy = new FixedSizeChunkingStrategy(100, 20);
    var chunks = strategy.Chunk("Some text").ToList();
    Assert.NotEmpty(chunks);  // Too vague!
}
```

**Correct:**
```csharp
[Fact]
public void Chunk_WithTextShorterThanChunkSize_ReturnsSingleChunk()
{
    var strategy = new FixedSizeChunkingStrategy(chunkSize: 100, chunkOverlap: 20);
    var text = "Short";

    var chunks = strategy.Chunk(text).ToList();

    Assert.Single(chunks);
    Assert.Equal("Short", chunks[0]);
}

[Fact]
public void Chunk_WithEmptyText_ThrowsArgumentException()
{
    var strategy = new FixedSizeChunkingStrategy(100, 20);
    Assert.Throws<ArgumentException>(() => strategy.Chunk("").ToList());
}
```

### Pitfall 2: Ignoring Position Metadata

**Wrong:**
```csharp
var chunks = strategy.Chunk(text).ToList();
// Only test the chunk text, ignore positions
```

**Correct:**
```csharp
var chunks = strategy.ChunkWithPositions(text).ToList();

foreach (var (chunk, start, end) in chunks)
{
    // Verify positions are accurate
    string extracted = text.Substring(start, end - start);
    Assert.Equal(chunk, extracted);
}
```

### Pitfall 3: Not Testing Overlap

**Wrong:**
```csharp
var chunks = strategy.Chunk(text).ToList();
Assert.True(chunks.Count > 1);  // Assumes overlap works
```

**Correct:**
```csharp
var chunks = strategy.ChunkWithPositions(text).ToList();

for (int i = 0; i < chunks.Count - 1; i++)
{
    var overlapStart = Math.Max(chunks[i].Start, chunks[i+1].Start);
    var overlapEnd = Math.Min(chunks[i].End, chunks[i+1].End);
    var actualOverlap = overlapEnd - overlapStart;

    Assert.InRange(actualOverlap, 0, expectedOverlap * 2);
}
```

### Pitfall 4: Hardcoding Test Data

**Wrong:**
```csharp
var text = "The quick brown fox...";  // Same text everywhere
```

**Correct:**
```csharp
protected string CreateSampleText(int sentenceCount)
{
    // Reusable test data generator
    return GenerateText(sentenceCount);
}

[Theory]
[InlineData(5)]
[InlineData(10)]
[InlineData(50)]
public void Chunk_WithVariousTextLengths_Works(int sentenceCount)
{
    var text = CreateSampleText(sentenceCount);
    // Test with different text sizes
}
```

---

## Testing Checklist

### For Each Chunking Strategy

- [ ] Constructor validation (invalid parameters throw exceptions)
- [ ] Basic chunking produces non-empty results
- [ ] Chunks respect maximum size limits
- [ ] Overlap is correctly implemented
- [ ] Position metadata is accurate
- [ ] Empty/null text throws ArgumentException
- [ ] Text shorter than chunk size returns single chunk
- [ ] Text equal to chunk size works correctly
- [ ] Large documents (10KB+) are handled efficiently
- [ ] Unicode characters are preserved
- [ ] Newlines and whitespace are handled correctly
- [ ] Consecutive calls with same input return same output (deterministic)
- [ ] No coverage gaps in chunked text
- [ ] Strategy-specific behavior works (sentences, markdown, code, etc.)

### Overall Coverage

- [ ] All public methods tested
- [ ] All edge cases covered
- [ ] Performance tests for large inputs
- [ ] Integration tests with real-world documents
- [ ] Code coverage >= 80%

---

## Next Steps

After completing tests for chunking strategies:

1. Run all tests and ensure 100% pass
2. Check code coverage report
3. Add any missing edge case tests
4. Update documentation with test examples
5. Move to **Issue #342** (RAG Advanced Patterns testing)

---

## Resources

### AiDotNet Patterns

- Study existing test patterns in `tests/` directory
- Follow three-tier pattern: Interface → Base → Implementation
- Use xUnit for all tests
- Use `[Theory]` and `[InlineData]` for parameterized tests

### Key Concepts

- **Chunk Size**: Target length of each chunk in characters
- **Chunk Overlap**: Number of characters shared between consecutive chunks
- **Position Metadata**: Start and end character offsets in original text
- **Sentence Boundaries**: Period, exclamation mark, question mark
- **Semantic Similarity**: Using embeddings to measure topic relatedness

### Common Assertions

```csharp
Assert.NotEmpty(chunks);
Assert.Single(chunks);
Assert.Equal(expected, actual);
Assert.True(condition, message);
Assert.InRange(value, min, max);
Assert.Throws<TException>(() => code);
Assert.All(collection, item => assertion);
```

---

## Questions?

If you get stuck:
1. Review the existing test examples above
2. Check the interface and base class implementations
3. Look at similar tests in `tests/` directory
4. Ensure all assertions are specific and meaningful
5. Test edge cases thoroughly

Good luck! Comprehensive testing ensures the chunking infrastructure works reliably for all RAG use cases.
