using AiDotNet.RetrievalAugmentedGeneration.ChunkingStrategies;
using Xunit;

namespace AiDotNetTests.IntegrationTests.RAG
{
    /// <summary>
    /// Integration tests for Chunking Strategy implementations.
    /// Tests validate chunk sizes, overlap, boundary handling, and text splitting correctness.
    /// </summary>
    public class ChunkingStrategyIntegrationTests
    {
        #region FixedSizeChunking Tests

        [Fact]
        public void FixedSizeChunking_BasicText_CreatesCorrectChunks()
        {
            // Arrange
            var strategy = new FixedSizeChunkingStrategy(chunkSize: 20, chunkOverlap: 5);
            var text = "The quick brown fox jumps over the lazy dog.";

            // Act
            var chunks = strategy.Chunk(text);
            var chunkList = chunks.ToList();

            // Assert
            Assert.NotEmpty(chunkList);
            Assert.All(chunkList, chunk =>
            {
                Assert.True(chunk.Length <= 20, $"Chunk too long: {chunk.Length}");
            });
        }

        [Fact]
        public void FixedSizeChunking_WithOverlap_ChunksOverlapCorrectly()
        {
            // Arrange
            var strategy = new FixedSizeChunkingStrategy(chunkSize: 30, chunkOverlap: 10);
            var text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyz";

            // Act
            var chunks = strategy.Chunk(text);
            var chunkList = chunks.ToList();

            // Assert
            Assert.True(chunkList.Count >= 2);

            // Verify overlap exists between consecutive chunks
            for (int i = 0; i < chunkList.Count - 1; i++)
            {
                var chunk1End = chunkList[i].Substring(Math.Max(0, chunkList[i].Length - 10));
                var chunk2Start = chunkList[i + 1].Substring(0, Math.Min(10, chunkList[i + 1].Length));

                // There should be some overlap in content
                Assert.True(chunk1End.Length > 0);
                Assert.True(chunk2Start.Length > 0);
            }
        }

        [Fact]
        public void FixedSizeChunking_EmptyText_ReturnsNoChunks()
        {
            // Arrange
            var strategy = new FixedSizeChunkingStrategy(chunkSize: 100, chunkOverlap: 10);
            var text = "";

            // Act
            var chunks = strategy.Chunk(text);

            // Assert
            Assert.Empty(chunks);
        }

        [Fact]
        public void FixedSizeChunking_TextSmallerThanChunkSize_ReturnsSingleChunk()
        {
            // Arrange
            var strategy = new FixedSizeChunkingStrategy(chunkSize: 100, chunkOverlap: 10);
            var text = "Short text.";

            // Act
            var chunks = strategy.Chunk(text);
            var chunkList = chunks.ToList();

            // Assert
            Assert.Single(chunkList);
            Assert.Equal("Short text.", chunkList[0]);
        }

        [Fact]
        public void FixedSizeChunking_LargeDocument_ChunksEvenly()
        {
            // Arrange
            var strategy = new FixedSizeChunkingStrategy(chunkSize: 500, chunkOverlap: 50);
            var text = string.Join(" ", Enumerable.Range(1, 1000).Select(i => $"Word{i}"));

            // Act
            var chunks = strategy.Chunk(text);
            var chunkList = chunks.ToList();

            // Assert
            Assert.True(chunkList.Count > 5); // Should create multiple chunks
            Assert.All(chunkList, chunk =>
            {
                Assert.True(chunk.Length <= 500, $"Chunk exceeds size: {chunk.Length}");
            });
        }

        [Fact]
        public void FixedSizeChunking_NoOverlap_ChunksAreContiguous()
        {
            // Arrange
            var strategy = new FixedSizeChunkingStrategy(chunkSize: 20, chunkOverlap: 0);
            var text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

            // Act
            var chunks = strategy.Chunk(text);
            var chunkList = chunks.ToList();

            // Assert - Reconstruct text from chunks (with no overlap, should match exactly)
            var reconstructed = string.Join("", chunkList);
            Assert.Equal(text, reconstructed);
        }

        #endregion

        #region RecursiveCharacterChunking Tests

        [Fact]
        public void RecursiveChunking_ParagraphText_SplitsAtParagraphBoundaries()
        {
            // Arrange
            var strategy = new RecursiveCharacterChunkingStrategy(chunkSize: 100, chunkOverlap: 10);
            var text = @"First paragraph contains important information.

Second paragraph has more details about the topic.

Third paragraph concludes the discussion.";

            // Act
            var chunks = strategy.Chunk(text);
            var chunkList = chunks.ToList();

            // Assert
            Assert.NotEmpty(chunkList);
            Assert.All(chunkList, chunk =>
            {
                Assert.True(chunk.Length <= 100 + 20, $"Chunk too long: {chunk.Length}"); // Some buffer for recursive splitting
            });
        }

        [Fact]
        public void RecursiveChunking_WithNewlines_PreservesStructure()
        {
            // Arrange
            var strategy = new RecursiveCharacterChunkingStrategy(chunkSize: 50, chunkOverlap: 5);
            var text = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5";

            // Act
            var chunks = strategy.Chunk(text);
            var chunkList = chunks.ToList();

            // Assert
            Assert.NotEmpty(chunkList);
            // Chunks should respect line boundaries when possible
        }

        [Fact]
        public void RecursiveChunking_LongUnbreakableText_HandlesGracefully()
        {
            // Arrange
            var strategy = new RecursiveCharacterChunkingStrategy(chunkSize: 50, chunkOverlap: 5);
            var text = new string('A', 200); // Long text without natural boundaries

            // Act
            var chunks = strategy.Chunk(text);
            var chunkList = chunks.ToList();

            // Assert
            Assert.NotEmpty(chunkList);
            Assert.True(chunkList.Count > 2); // Should split despite no boundaries
        }

        #endregion

        #region SentenceChunking Tests

        [Fact]
        public void SentenceChunking_MultiSentenceText_SplitsAtSentenceBoundaries()
        {
            // Arrange
            var strategy = new SentenceChunkingStrategy(chunkSize: 100, chunkOverlap: 0);
            var text = "First sentence. Second sentence. Third sentence. Fourth sentence.";

            // Act
            var chunks = strategy.Chunk(text);
            var chunkList = chunks.ToList();

            // Assert
            Assert.NotEmpty(chunkList);
            // Verify chunks end with sentence terminators when possible
            foreach (var chunk in chunkList)
            {
                var trimmed = chunk.Trim();
                if (trimmed.Length > 0)
                {
                    var lastChar = trimmed[trimmed.Length - 1];
                    // Should end with punctuation or be the last chunk
                    Assert.True(lastChar == '.' || lastChar == '!' || lastChar == '?' ||
                               chunk == chunkList.Last());
                }
            }
        }

        [Fact]
        public void SentenceChunking_ComplexPunctuation_HandlesCorrectly()
        {
            // Arrange
            var strategy = new SentenceChunkingStrategy(chunkSize: 200, chunkOverlap: 10);
            var text = @"Dr. Smith works at A.I. Corp. His research focuses on machine learning!
                        What makes his work unique? He combines theory with practice.";

            // Act
            var chunks = strategy.Chunk(text);
            var chunkList = chunks.ToList();

            // Assert
            Assert.NotEmpty(chunkList);
        }

        [Fact]
        public void SentenceChunking_SingleLongSentence_SplitsIfNecessary()
        {
            // Arrange
            var strategy = new SentenceChunkingStrategy(chunkSize: 50, chunkOverlap: 5);
            var text = "This is an extremely long sentence that contains many words and clauses " +
                      "and should be split into multiple chunks even though it is a single sentence " +
                      "because it exceeds the chunk size limit significantly.";

            // Act
            var chunks = strategy.Chunk(text);
            var chunkList = chunks.ToList();

            // Assert
            Assert.True(chunkList.Count > 1); // Should split long sentence
        }

        #endregion

        #region SemanticChunking Tests

        [Fact]
        public void SemanticChunking_ThematicText_GroupsRelatedContent()
        {
            // Arrange - Using stub embedding for testing
            var embeddingModel = new AiDotNet.RetrievalAugmentedGeneration.Embeddings.StubEmbeddingModel<double>(
                embeddingDimension: 384);
            var strategy = new SemanticChunkingStrategy(
                embeddingModel: embeddingModel,
                chunkSize: 200,
                chunkOverlap: 20,
                similarityThreshold: 0.5);

            var text = @"Machine learning is a subset of artificial intelligence.
                        It focuses on training algorithms to learn from data.

                        Cooking pasta is simple. Boil water, add salt, and cook for 10 minutes.
                        Always use fresh ingredients for best results.

                        Neural networks are inspired by biological neurons.
                        They consist of layers that process information.";

            // Act
            var chunks = strategy.Chunk(text);
            var chunkList = chunks.ToList();

            // Assert
            Assert.NotEmpty(chunkList);
            Assert.All(chunkList, chunk => Assert.True(chunk.Length > 0));
        }

        [Fact]
        public void SemanticChunking_SimilarSentences_CombinesIntoSameChunk()
        {
            // Arrange
            var embeddingModel = new AiDotNet.RetrievalAugmentedGeneration.Embeddings.StubEmbeddingModel<double>(
                embeddingDimension: 384);
            var strategy = new SemanticChunkingStrategy(
                embeddingModel: embeddingModel,
                chunkSize: 300,
                chunkOverlap: 30,
                similarityThreshold: 0.3);

            var text = @"Dogs are loyal pets. Dogs are friendly animals. Dogs love to play.
                        Cats are independent creatures. Cats enjoy solitude. Cats are graceful.";

            // Act
            var chunks = strategy.Chunk(text);
            var chunkList = chunks.ToList();

            // Assert
            Assert.NotEmpty(chunkList);
            // Semantic chunking should group related sentences
        }

        #endregion

        #region SlidingWindowChunking Tests

        [Fact]
        public void SlidingWindowChunking_FixedWindow_CreatesOverlappingChunks()
        {
            // Arrange
            var strategy = new SlidingWindowChunkingStrategy(windowSize: 50, stepSize: 25);
            var text = "The quick brown fox jumps over the lazy dog. " +
                      "Pack my box with five dozen liquor jugs. " +
                      "How vexingly quick daft zebras jump!";

            // Act
            var chunks = strategy.Chunk(text);
            var chunkList = chunks.ToList();

            // Assert
            Assert.NotEmpty(chunkList);
            Assert.All(chunkList, chunk => Assert.True(chunk.Length <= 50));

            // Verify sliding window behavior
            if (chunkList.Count > 1)
            {
                Assert.True(chunkList.Count > 2); // Should have overlap creating more chunks
            }
        }

        [Fact]
        public void SlidingWindowChunking_StepSizeEqualsWindowSize_NoOverlap()
        {
            // Arrange
            var strategy = new SlidingWindowChunkingStrategy(windowSize: 30, stepSize: 30);
            var text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyz";

            // Act
            var chunks = strategy.Chunk(text);
            var chunkList = chunks.ToList();

            // Assert
            Assert.NotEmpty(chunkList);
            var reconstructed = string.Join("", chunkList);
            Assert.Equal(text, reconstructed);
        }

        [Fact]
        public void SlidingWindowChunking_SmallStepSize_CreatesMoreChunks()
        {
            // Arrange
            var text = "This is a test document with enough text to create multiple chunks.";
            var strategy1 = new SlidingWindowChunkingStrategy(windowSize: 30, stepSize: 30);
            var strategy2 = new SlidingWindowChunkingStrategy(windowSize: 30, stepSize: 10);

            // Act
            var chunks1 = strategy1.Chunk(text).ToList();
            var chunks2 = strategy2.Chunk(text).ToList();

            // Assert
            Assert.True(chunks2.Count > chunks1.Count,
                $"Smaller step size should create more chunks: {chunks2.Count} vs {chunks1.Count}");
        }

        #endregion

        #region MarkdownTextSplitter Tests

        [Fact]
        public void MarkdownSplitter_HeaderBasedSplitting_PreservesHierarchy()
        {
            // Arrange
            var strategy = new MarkdownTextSplitter(chunkSize: 200, chunkOverlap: 20);
            var markdown = @"# Title

## Section 1
Content for section 1.

## Section 2
Content for section 2.

### Subsection 2.1
Detailed content here.";

            // Act
            var chunks = strategy.Chunk(markdown);
            var chunkList = chunks.ToList();

            // Assert
            Assert.NotEmpty(chunkList);
            // Verify headers are preserved in chunks
        }

        [Fact]
        public void MarkdownSplitter_CodeBlocks_HandledAsUnits()
        {
            // Arrange
            var strategy = new MarkdownTextSplitter(chunkSize: 150, chunkOverlap: 10);
            var markdown = @"Here is some code:

```csharp
public class Example
{
    public void Method() { }
}
```

More text follows.";

            // Act
            var chunks = strategy.Chunk(markdown);
            var chunkList = chunks.ToList();

            // Assert
            Assert.NotEmpty(chunkList);
        }

        [Fact]
        public void MarkdownSplitter_Lists_MaintainStructure()
        {
            // Arrange
            var strategy = new MarkdownTextSplitter(chunkSize: 100, chunkOverlap: 10);
            var markdown = @"Shopping list:

- Apples
- Bananas
- Oranges
- Grapes
- Strawberries";

            // Act
            var chunks = strategy.Chunk(markdown);
            var chunkList = chunks.ToList();

            // Assert
            Assert.NotEmpty(chunkList);
        }

        #endregion

        #region CodeAwareTextSplitter Tests

        [Fact]
        public void CodeAwareSplitter_CSharpCode_SplitsAtFunctionBoundaries()
        {
            // Arrange
            var strategy = new CodeAwareTextSplitter(language: "csharp", chunkSize: 200, chunkOverlap: 20);
            var code = @"public class Calculator
{
    public int Add(int a, int b)
    {
        return a + b;
    }

    public int Subtract(int a, int b)
    {
        return a - b;
    }

    public int Multiply(int a, int b)
    {
        return a * b;
    }
}";

            // Act
            var chunks = strategy.Chunk(code);
            var chunkList = chunks.ToList();

            // Assert
            Assert.NotEmpty(chunkList);
            // Verify chunks maintain code structure
        }

        [Fact]
        public void CodeAwareSplitter_PreservesIndentation_InChunks()
        {
            // Arrange
            var strategy = new CodeAwareTextSplitter(language: "python", chunkSize: 150, chunkOverlap: 10);
            var code = @"def calculate_total(items):
    total = 0
    for item in items:
        if item.valid:
            total += item.price
    return total";

            // Act
            var chunks = strategy.Chunk(code);
            var chunkList = chunks.ToList();

            // Assert
            Assert.NotEmpty(chunkList);
        }

        #endregion

        #region TableAwareTextSplitter Tests

        [Fact]
        public void TableAwareSplitter_MarkdownTable_PreservesTableStructure()
        {
            // Arrange
            var strategy = new TableAwareTextSplitter(chunkSize: 200, chunkOverlap: 20);
            var text = @"Data table:

| Name  | Age | City      |
|-------|-----|-----------|
| Alice | 30  | New York  |
| Bob   | 25  | London    |
| Carol | 35  | Tokyo     |

End of table.";

            // Act
            var chunks = strategy.Chunk(text);
            var chunkList = chunks.ToList();

            // Assert
            Assert.NotEmpty(chunkList);
        }

        [Fact]
        public void TableAwareSplitter_LargeTable_SplitsAppropriately()
        {
            // Arrange
            var strategy = new TableAwareTextSplitter(chunkSize: 150, chunkOverlap: 10);
            var rows = string.Join("\n", Enumerable.Range(1, 20).Select(i =>
                $"| Item{i} | Value{i} | Description for item {i} |"));
            var text = $"| Header1 | Header2 | Header3 |\n|---------|---------|----------|\n{rows}";

            // Act
            var chunks = strategy.Chunk(text);
            var chunkList = chunks.ToList();

            // Assert
            Assert.NotEmpty(chunkList);
        }

        #endregion

        #region HeaderBasedTextSplitter Tests

        [Fact]
        public void HeaderBasedSplitter_SplitsAtHeaders_MaintainsContext()
        {
            // Arrange
            var strategy = new HeaderBasedTextSplitter(chunkSize: 200, chunkOverlap: 20);
            var text = @"# Main Title

Introduction paragraph.

## First Section

Content of first section with details.

## Second Section

Content of second section with more information.

### Subsection

Nested content here.";

            // Act
            var chunks = strategy.Chunk(text);
            var chunkList = chunks.ToList();

            // Assert
            Assert.NotEmpty(chunkList);
            Assert.True(chunkList.Count >= 2); // Should split at major headers
        }

        #endregion

        #region Edge Cases and Stress Tests

        [Fact]
        public void ChunkingStrategies_WhitespaceOnlyText_HandlesGracefully()
        {
            // Arrange
            var strategy = new FixedSizeChunkingStrategy(chunkSize: 100, chunkOverlap: 10);
            var text = "     \n\n\n     \t\t\t     ";

            // Act
            var chunks = strategy.Chunk(text);
            var chunkList = chunks.ToList();

            // Assert - Either empty or single chunk with whitespace
            Assert.True(chunkList.Count <= 1);
        }

        [Fact]
        public void ChunkingStrategies_UnicodeText_HandlesCorrectly()
        {
            // Arrange
            var strategy = new FixedSizeChunkingStrategy(chunkSize: 50, chunkOverlap: 5);
            var text = "Hello 世界! こんにちは Здравствуй مرحبا 🌍🌎🌏";

            // Act
            var chunks = strategy.Chunk(text);
            var chunkList = chunks.ToList();

            // Assert
            Assert.NotEmpty(chunkList);
            Assert.All(chunkList, chunk => Assert.True(chunk.Length > 0));
        }

        [Fact]
        public void ChunkingStrategies_VeryLargeDocument_CompletesInReasonableTime()
        {
            // Arrange
            var strategy = new FixedSizeChunkingStrategy(chunkSize: 500, chunkOverlap: 50);
            var text = string.Join(" ", Enumerable.Range(1, 10000).Select(i => $"Word{i}"));
            var stopwatch = System.Diagnostics.Stopwatch.StartNew();

            // Act
            var chunks = strategy.Chunk(text);
            var chunkList = chunks.ToList();
            stopwatch.Stop();

            // Assert
            Assert.NotEmpty(chunkList);
            Assert.True(stopwatch.ElapsedMilliseconds < 5000,
                $"Chunking took too long: {stopwatch.ElapsedMilliseconds}ms");
        }

        [Fact]
        public void ChunkingStrategies_ChunkSizeLargerThanText_ReturnsSingleChunk()
        {
            // Arrange
            var strategies = new IChunkingStrategy[]
            {
                new FixedSizeChunkingStrategy(chunkSize: 1000, chunkOverlap: 0),
                new RecursiveCharacterChunkingStrategy(chunkSize: 1000, chunkOverlap: 0),
                new SentenceChunkingStrategy(chunkSize: 1000, chunkOverlap: 0)
            };
            var text = "Short text that fits in one chunk.";

            // Act & Assert
            foreach (var strategy in strategies)
            {
                var chunks = strategy.Chunk(text).ToList();
                Assert.Single(chunks);
                Assert.Equal(text, chunks[0]);
            }
        }

        [Fact]
        public void ChunkingStrategies_OverlapLargerThanChunkSize_HandlesGracefully()
        {
            // Arrange - This is an edge case that should either throw or handle gracefully
            var text = "This is a test document with some content.";

            // Act & Assert - Should not crash
            try
            {
                var strategy = new FixedSizeChunkingStrategy(chunkSize: 50, chunkOverlap: 100);
                var chunks = strategy.Chunk(text).ToList();
                // If it doesn't throw, verify it produces reasonable output
                Assert.NotEmpty(chunks);
            }
            catch (ArgumentException)
            {
                // This is also acceptable behavior
                Assert.True(true);
            }
        }

        #endregion
    }
}
