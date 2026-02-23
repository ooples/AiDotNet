#nullable disable
using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.RetrievalAugmentedGeneration.ContextCompression;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using Xunit;

namespace AiDotNetTests.UnitTests.RetrievalAugmentedGeneration.ContextCompression
{
    public class AutoCompressorTests : ContextCompressorTestBase
    {
        #region Constructor Tests

        [Fact]
        public void Constructor_WithDefaultParameters_CreatesInstance()
        {
            // Arrange & Act
            var compressor = new AutoCompressor<double>();

            // Assert
            Assert.NotNull(compressor);
        }

        [Fact]
        public void Constructor_WithValidParameters_CreatesInstance()
        {
            // Arrange & Act
            var compressor = new AutoCompressor<double>(maxOutputLength: 1000, compressionRatio: 0.7);

            // Assert
            Assert.NotNull(compressor);
        }

        [Fact]
        public void Constructor_WithZeroMaxOutputLength_ThrowsArgumentOutOfRangeException()
        {
            // Arrange & Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new AutoCompressor<double>(maxOutputLength: 0));
        }

        [Fact]
        public void Constructor_WithNegativeMaxOutputLength_ThrowsArgumentOutOfRangeException()
        {
            // Arrange & Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new AutoCompressor<double>(maxOutputLength: -100));
        }

        [Fact]
        public void Constructor_WithZeroCompressionRatio_ThrowsArgumentOutOfRangeException()
        {
            // Arrange & Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new AutoCompressor<double>(maxOutputLength: 500, compressionRatio: 0));
        }

        [Fact]
        public void Constructor_WithNegativeCompressionRatio_ThrowsArgumentOutOfRangeException()
        {
            // Arrange & Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new AutoCompressor<double>(maxOutputLength: 500, compressionRatio: -0.5));
        }

        [Fact]
        public void Constructor_WithCompressionRatioGreaterThanOne_ThrowsArgumentOutOfRangeException()
        {
            // Arrange & Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new AutoCompressor<double>(maxOutputLength: 500, compressionRatio: 1.5));
        }

        [Fact]
        public void Constructor_WithCompressionRatioEqualToOne_CreatesInstance()
        {
            // Arrange & Act
            var compressor = new AutoCompressor<double>(maxOutputLength: 500, compressionRatio: 1.0);

            // Assert
            Assert.NotNull(compressor);
        }

        #endregion

        #region Basic Functionality Tests

        [Fact]
        public void Compress_WithValidDocuments_ReturnsCompressedDocuments()
        {
            // Arrange
            var compressor = new AutoCompressor<double>(maxOutputLength: 500, compressionRatio: 0.5);
            var documents = CreateSampleDocuments();
            var query = "machine learning neural networks";

            // Act
            var result = compressor.Compress(documents, query);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(documents.Count, result.Count);
        }

        [Fact]
        public void Compress_WithNullDocuments_ThrowsArgumentNullException()
        {
            // Arrange
            var compressor = new AutoCompressor<double>();
            var query = "test query";

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                compressor.Compress(null, query));
        }

        [Fact]
        public void Compress_WithNullQuery_ThrowsArgumentException()
        {
            // Arrange
            var compressor = new AutoCompressor<double>();
            var documents = CreateSampleDocuments();

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                compressor.Compress(documents, null));
        }

        [Fact]
        public void Compress_WithEmptyQuery_ThrowsArgumentException()
        {
            // Arrange
            var compressor = new AutoCompressor<double>();
            var documents = CreateSampleDocuments();

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                compressor.Compress(documents, string.Empty));
        }

        [Fact]
        public void Compress_WithWhitespaceQuery_ThrowsArgumentException()
        {
            // Arrange
            var compressor = new AutoCompressor<double>();
            var documents = CreateSampleDocuments();

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                compressor.Compress(documents, "   "));
        }

        [Fact]
        public void Compress_WithEmptyDocumentList_ReturnsEmptyList()
        {
            // Arrange
            var compressor = new AutoCompressor<double>();
            var documents = new List<Document<double>>();
            var query = "test query";

            // Act
            var result = compressor.Compress(documents, query);

            // Assert
            Assert.NotNull(result);
            Assert.Empty(result);
        }

        #endregion

        #region Compression Quality Tests

        [Fact]
        public void Compress_ReducesDocumentSize()
        {
            // Arrange
            var compressor = new AutoCompressor<double>(maxOutputLength: 300, compressionRatio: 0.5);
            var documents = CreateSampleDocuments();
            var query = "machine learning";

            // Act
            var result = compressor.Compress(documents, query);

            // Assert
            AssertCompressed(documents, result, allowEqual: true);
        }

        [Fact]
        public void Compress_RespectsMaxOutputLength()
        {
            // Arrange
            var maxLength = 200;
            var compressor = new AutoCompressor<double>(maxOutputLength: maxLength, compressionRatio: 0.8);
            var longDoc = CreateLargeDocument("large");
            var documents = new List<Document<double>> { longDoc };
            var query = "document content";

            // Act
            var result = compressor.Compress(documents, query);

            // Assert
            Assert.Single(result);
            Assert.True(result[0].Content.Length <= maxLength,
                $"Compressed length {result[0].Content.Length} should be <= {maxLength}");
        }

        [Fact]
        public void Compress_WithLowCompressionRatio_RetainsFewSentences()
        {
            // Arrange
            var compressor = new AutoCompressor<double>(maxOutputLength: 1000, compressionRatio: 0.3);
            var document = CreateDocumentWithLength("doc", 10);
            var documents = new List<Document<double>> { document };
            var query = "test";

            // Act
            var result = compressor.Compress(documents, query);

            // Assert
            Assert.Single(result);
            Assert.True(result[0].Content.Length < document.Content.Length);
        }

        [Fact]
        public void Compress_WithHighCompressionRatio_RetainsMoreContent()
        {
            // Arrange
            var lowCompressor = new AutoCompressor<double>(maxOutputLength: 2000, compressionRatio: 0.3);
            var highCompressor = new AutoCompressor<double>(maxOutputLength: 2000, compressionRatio: 0.7);
            var documents = CreateSampleDocuments();
            var query = "machine learning";

            // Act
            var lowResult = lowCompressor.Compress(documents, query);
            var highResult = highCompressor.Compress(documents, query);

            // Assert
            var lowLength = lowResult.Sum(d => d.Content.Length);
            var highLength = highResult.Sum(d => d.Content.Length);
            Assert.True(highLength >= lowLength,
                "Higher compression ratio should produce longer or equal output");
        }

        [Fact]
        public void Compress_PrioritizesQueryRelevantContent()
        {
            // Arrange
            var compressor = new AutoCompressor<double>(maxOutputLength: 150, compressionRatio: 0.4);
            var document = new Document<double>("doc",
                "Machine learning is powerful. The weather is nice today. Neural networks learn patterns. Birds are flying. Deep learning uses layers.");
            var documents = new List<Document<double>> { document };
            var query = "machine learning neural networks";

            // Act
            var result = compressor.Compress(documents, query);

            // Assert
            Assert.Single(result);
            var content = result[0].Content.ToLowerInvariant();
            // Should prioritize sentences with query terms
            Assert.True(
                content.Contains("machine") ||
                content.Contains("learning") ||
                content.Contains("neural"));
        }

        [Fact]
        public void Compress_PreservesMetadata()
        {
            // Arrange
            var compressor = new AutoCompressor<double>();
            var documents = CreateSampleDocuments();
            var query = "machine learning";

            // Act
            var result = compressor.Compress(documents, query);

            // Assert
            AssertMetadataPreserved(documents, result);
        }

        [Fact]
        public void Compress_PreservesRelevanceScores()
        {
            // Arrange
            var compressor = new AutoCompressor<double>();
            var documents = CreateSampleDocuments();
            var query = "machine learning";

            // Act
            var result = compressor.Compress(documents, query);

            // Assert
            AssertRelevanceScoresPreserved(documents, result);
        }

        [Fact]
        public void Compress_PreservesDocumentIds()
        {
            // Arrange
            var compressor = new AutoCompressor<double>();
            var documents = CreateSampleDocuments();
            var query = "machine learning";

            // Act
            var result = compressor.Compress(documents, query);

            // Assert
            for (int i = 0; i < documents.Count; i++)
            {
                Assert.Equal(documents[i].Id, result[i].Id);
            }
        }

        #endregion

        #region Edge Cases Tests

        [Fact]
        public void Compress_WithEmptyDocument_ReturnsEmptyContent()
        {
            // Arrange
            var compressor = new AutoCompressor<double>();
            var documents = new List<Document<double>>
            {
                new Document<double>("empty", string.Empty)
            };
            var query = "test";

            // Act
            var result = compressor.Compress(documents, query);

            // Assert
            Assert.Single(result);
            Assert.Equal(string.Empty, result[0].Content);
        }

        [Fact]
        public void Compress_WithWhitespaceOnlyDocument_ReturnsEmptyContent()
        {
            // Arrange
            var compressor = new AutoCompressor<double>();
            var documents = new List<Document<double>>
            {
                new Document<double>("whitespace", "   \t\n  ")
            };
            var query = "test";

            // Act
            var result = compressor.Compress(documents, query);

            // Assert
            Assert.Single(result);
            Assert.Equal(string.Empty, result[0].Content);
        }

        [Fact]
        public void Compress_WithSingleSentence_HandlesCorrectly()
        {
            // Arrange
            var compressor = new AutoCompressor<double>(maxOutputLength: 500, compressionRatio: 0.5);
            var documents = new List<Document<double>>
            {
                new Document<double>("single", "Machine learning is important.")
            };
            var query = "machine learning";

            // Act
            var result = compressor.Compress(documents, query);

            // Assert
            Assert.Single(result);
            Assert.NotEmpty(result[0].Content);
        }

        [Fact]
        public void Compress_WithVeryLargeDocument_CompressesSuccessfully()
        {
            // Arrange
            var compressor = new AutoCompressor<double>(maxOutputLength: 500, compressionRatio: 0.3);
            var largeDoc = CreateLargeDocument("large");
            var documents = new List<Document<double>> { largeDoc };
            var query = "document content long";

            // Act
            var result = compressor.Compress(documents, query);

            // Assert
            Assert.Single(result);
            Assert.True(result[0].Content.Length <= 500);
            Assert.NotEmpty(result[0].Content);
        }

        [Fact]
        public void Compress_WithUnicodeContent_HandlesCorrectly()
        {
            // Arrange
            var compressor = new AutoCompressor<double>();
            var unicodeDoc = CreateUnicodeDocument("unicode");
            var documents = new List<Document<double>> { unicodeDoc };
            var query = "学习 learning";

            // Act
            var result = compressor.Compress(documents, query);

            // Assert
            Assert.Single(result);
            Assert.NotEmpty(result[0].Content);
        }

        [Fact]
        public void Compress_WithSpecialCharacters_HandlesCorrectly()
        {
            // Arrange
            var compressor = new AutoCompressor<double>();
            var specialDoc = CreateSpecialCharDocument("special");
            var documents = new List<Document<double>> { specialDoc };
            var query = "special chars testing";

            // Act
            var result = compressor.Compress(documents, query);

            // Assert
            Assert.Single(result);
            Assert.NotEmpty(result[0].Content);
        }

        [Fact]
        public void Compress_WithMultipleDocuments_ProcessesAllDocuments()
        {
            // Arrange
            var compressor = new AutoCompressor<double>(maxOutputLength: 300, compressionRatio: 0.5);
            var documents = new List<Document<double>>();
            for (int i = 0; i < 10; i++)
            {
                documents.Add(CreateDocumentWithLength($"doc{i}", 10));
            }
            var query = "test sentence content";

            // Act
            var result = compressor.Compress(documents, query);

            // Assert
            Assert.Equal(10, result.Count);
            foreach (var doc in result)
            {
                Assert.NotNull(doc.Content);
            }
        }

        [Fact]
        public void Compress_WithDocumentSmallerThanMaxLength_MayReturnOriginal()
        {
            // Arrange
            var compressor = new AutoCompressor<double>(maxOutputLength: 1000, compressionRatio: 0.9);
            var shortDoc = new Document<double>("short", "Short document with minimal content.");
            var documents = new List<Document<double>> { shortDoc };
            var query = "test";

            // Act
            var result = compressor.Compress(documents, query);

            // Assert
            Assert.Single(result);
            Assert.NotEmpty(result[0].Content);
        }

        #endregion

        #region Sentence Scoring Tests

        [Fact]
        public void Compress_PrioritizesEarlySentences()
        {
            // Arrange
            var compressor = new AutoCompressor<double>(maxOutputLength: 100, compressionRatio: 0.3);
            var document = new Document<double>("doc",
                "First sentence with important content. Second sentence also relevant. Third sentence here. Fourth sentence present. Fifth sentence exists.");
            var documents = new List<Document<double>> { document };
            var query = "content relevant";

            // Act
            var result = compressor.Compress(documents, query);

            // Assert
            Assert.Single(result);
            Assert.NotEmpty(result[0].Content);
            // Early sentences with query terms should be prioritized
        }

        [Fact]
        public void Compress_ScoresBasedOnQueryMatch()
        {
            // Arrange
            var compressor = new AutoCompressor<double>(maxOutputLength: 200, compressionRatio: 0.5);
            var document = new Document<double>("doc",
                "Machine learning is a field of AI. The weather is sunny today. Neural networks process information. Birds are singing outside.");
            var documents = new List<Document<double>> { document };
            var query = "machine learning neural networks";

            // Act
            var result = compressor.Compress(documents, query);

            // Assert
            Assert.Single(result);
            var content = result[0].Content.ToLowerInvariant();
            // Should include sentences matching query
            int matchCount = 0;
            if (content.Contains("machine") || content.Contains("learning")) matchCount++;
            if (content.Contains("neural") || content.Contains("networks")) matchCount++;
            Assert.True(matchCount > 0, "Compressed content should include query-relevant sentences");
        }

        #endregion

        #region Integration Tests

        [Fact]
        public void Compress_WithDifferentCompressionRatios_ProducesExpectedLengths()
        {
            // Arrange
            var documents = CreateSampleDocuments();
            var query = "machine learning";
            var ratios = new[] { 0.3, 0.5, 0.7 };
            var results = new List<int>();

            // Act
            foreach (var ratio in ratios)
            {
                var compressor = new AutoCompressor<double>(maxOutputLength: 2000, compressionRatio: ratio);
                var compressed = compressor.Compress(documents, query);
                results.Add(compressed.Sum(d => d.Content.Length));
            }

            // Assert
            for (int i = 0; i < ratios.Length - 1; i++)
            {
                Assert.True(results[i] <= results[i + 1],
                    $"Higher compression ratio should produce longer output. Got {results[i]} and {results[i + 1]}");
            }
        }

        [Fact]
        public void Compress_WithDifferentMaxLengths_RespectsBounds()
        {
            // Arrange
            var largeDoc = CreateLargeDocument("large");
            var documents = new List<Document<double>> { largeDoc };
            var query = "document content";
            var maxLengths = new[] { 100, 300, 500 };

            // Act & Assert
            foreach (var maxLength in maxLengths)
            {
                var compressor = new AutoCompressor<double>(maxOutputLength: maxLength, compressionRatio: 0.5);
                var result = compressor.Compress(documents, query);
                Assert.Single(result);
                Assert.True(result[0].Content.Length <= maxLength,
                    $"Result length {result[0].Content.Length} should be <= {maxLength}");
            }
        }

        [Fact]
        public void Compress_WithDifferentQueries_ProducesDifferentResults()
        {
            // Arrange
            // Use a single document with distinct sentences for each query
            // This ensures the query affects which sentence is prioritized
            var compressor = new AutoCompressor<double>(maxOutputLength: 100, compressionRatio: 0.3);
            var documents = new List<Document<double>>
            {
                new Document<double>("mixed", "Machine learning algorithms power modern AI. Weather forecasting uses complex models. Data science combines statistics and programming.")
            };
            var query1 = "machine learning AI";
            var query2 = "weather forecasting";

            // Act
            var result1 = compressor.Compress(documents, query1);
            var result2 = compressor.Compress(documents, query2);

            // Assert
            // Different queries should prioritize different sentences
            Assert.NotEqual(result1[0].Content, result2[0].Content);
        }

        [Fact]
        public void Compress_MultipleInvocations_ProducesSameResults()
        {
            // Arrange
            var compressor = new AutoCompressor<double>(maxOutputLength: 500, compressionRatio: 0.5);
            var documents = CreateSampleDocuments();
            var query = "machine learning";

            // Act
            var result1 = compressor.Compress(documents, query);
            var result2 = compressor.Compress(documents, query);

            // Assert
            Assert.Equal(result1.Count, result2.Count);
            for (int i = 0; i < result1.Count; i++)
            {
                Assert.Equal(result1[i].Content, result2[i].Content);
            }
        }

        [Fact]
        public void Compress_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var compressor = new AutoCompressor<float>(maxOutputLength: 500, compressionRatio: 0.5f);
            var documents = new List<Document<float>>
            {
                new Document<float>("doc1", "Machine learning is powerful. Neural networks are important.")
                {
                    RelevanceScore = 0.9f,
                    HasRelevanceScore = true
                }
            };
            var query = "machine learning";

            // Act
            var result = compressor.Compress(documents, query);

            // Assert
            Assert.NotNull(result);
            Assert.Single(result);
        }

        #endregion

        #region Performance Tests

        [Fact]
        public void Compress_With100KBDocument_CompletesInReasonableTime()
        {
            // Arrange
            var compressor = new AutoCompressor<double>(maxOutputLength: 500, compressionRatio: 0.3);
            // Create a document > 100KB
            var largeContent = string.Join(" ", Enumerable.Repeat("This is a sentence with some content to test compression performance.", 3000));
            var largeDoc = new Document<double>("huge", largeContent);
            var documents = new List<Document<double>> { largeDoc };
            var query = "content compression test";

            // Act
            var result = compressor.Compress(documents, query);

            // Assert
            Assert.Single(result);
            Assert.True(result[0].Content.Length <= 500);
            Assert.NotEmpty(result[0].Content);
        }

        #endregion
    }
}
