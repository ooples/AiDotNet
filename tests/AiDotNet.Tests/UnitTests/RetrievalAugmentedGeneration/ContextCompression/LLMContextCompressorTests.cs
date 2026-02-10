#nullable disable
using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.RetrievalAugmentedGeneration.ContextCompression;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using Xunit;

namespace AiDotNetTests.UnitTests.RetrievalAugmentedGeneration.ContextCompression
{
    public class LLMContextCompressorTests : ContextCompressorTestBase
    {
        #region Constructor Tests

        [Fact]
        public void Constructor_WithValidCompressionRatio_CreatesInstance()
        {
            // Arrange & Act
            var compressor = new LLMContextCompressor<double>(compressionRatio: 0.5);

            // Assert
            Assert.NotNull(compressor);
        }

        [Fact]
        public void Constructor_WithCompressionRatioZero_CreatesInstance()
        {
            // Arrange & Act
            // Zero compression ratio is valid - it means keeping 0% of content (maximum compression)
            var compressor = new LLMContextCompressor<double>(compressionRatio: 0);

            // Assert
            Assert.NotNull(compressor);
        }

        [Fact]
        public void Constructor_WithCompressionRatioNegative_ThrowsArgumentOutOfRangeException()
        {
            // Arrange & Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new LLMContextCompressor<double>(compressionRatio: -0.5));
        }

        [Fact]
        public void Constructor_WithCompressionRatioGreaterThanOne_ThrowsArgumentOutOfRangeException()
        {
            // Arrange & Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new LLMContextCompressor<double>(compressionRatio: 1.5));
        }

        [Fact]
        public void Constructor_WithCompressionRatioOne_CreatesInstance()
        {
            // Arrange & Act
            var compressor = new LLMContextCompressor<double>(compressionRatio: 1.0);

            // Assert
            Assert.NotNull(compressor);
        }

        #endregion

        #region Basic Functionality Tests

        [Fact]
        public void Compress_WithValidDocuments_ReturnsCompressedDocuments()
        {
            // Arrange
            var compressor = new LLMContextCompressor<double>(compressionRatio: 0.5);
            var documents = CreateSampleDocuments();
            var query = "machine learning neural networks";

            // Act
            var result = compressor.Compress(documents, query);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(documents.Count, result.Count);
            AssertCompressed(documents, result);
        }

        [Fact]
        public void Compress_WithNullDocuments_ThrowsArgumentNullException()
        {
            // Arrange
            var compressor = new LLMContextCompressor<double>(compressionRatio: 0.5);
            var query = "test query";

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                compressor.Compress(null, query));
        }

        [Fact]
        public void Compress_WithNullQuery_ThrowsArgumentException()
        {
            // Arrange
            var compressor = new LLMContextCompressor<double>(compressionRatio: 0.5);
            var documents = CreateSampleDocuments();

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                compressor.Compress(documents, null));
        }

        [Fact]
        public void Compress_WithEmptyQuery_ThrowsArgumentException()
        {
            // Arrange
            var compressor = new LLMContextCompressor<double>(compressionRatio: 0.5);
            var documents = CreateSampleDocuments();

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                compressor.Compress(documents, string.Empty));
        }

        [Fact]
        public void Compress_WithWhitespaceQuery_ThrowsArgumentException()
        {
            // Arrange
            var compressor = new LLMContextCompressor<double>(compressionRatio: 0.5);
            var documents = CreateSampleDocuments();

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                compressor.Compress(documents, "   "));
        }

        [Fact]
        public void Compress_WithEmptyDocumentList_ReturnsEmptyList()
        {
            // Arrange
            var compressor = new LLMContextCompressor<double>(compressionRatio: 0.5);
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
        public void Compress_PreservesRelevantInformation()
        {
            // Arrange
            var compressor = new LLMContextCompressor<double>(compressionRatio: 0.5);
            var documents = CreateSampleDocuments();
            var query = "machine learning neural networks";

            // Act
            var result = compressor.Compress(documents, query);

            // Assert
            // Check that compressed content contains query terms
            var compressedContent = string.Join(" ", result.Select(d => d.Content)).ToLowerInvariant();
            Assert.Contains("machine learning", compressedContent);
        }

        [Fact]
        public void Compress_WithLowCompressionRatio_RetainsFewSentences()
        {
            // Arrange
            var compressor = new LLMContextCompressor<double>(compressionRatio: 0.3);
            var document = new Document<double>("doc1",
                "First sentence about machine learning. Second sentence about neural networks. Third sentence about data science. Fourth sentence about Python programming.");
            var documents = new List<Document<double>> { document };
            var query = "machine learning";

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
            var lowCompressor = new LLMContextCompressor<double>(compressionRatio: 0.3);
            var highCompressor = new LLMContextCompressor<double>(compressionRatio: 0.7);
            var documents = CreateSampleDocuments();
            var query = "machine learning";

            // Act
            var lowResult = lowCompressor.Compress(documents, query);
            var highResult = highCompressor.Compress(documents, query);

            // Assert
            var lowLength = lowResult.Sum(d => d.Content.Length);
            var highLength = highResult.Sum(d => d.Content.Length);
            Assert.True(highLength > lowLength);
        }

        [Fact]
        public void Compress_PreservesMetadata()
        {
            // Arrange
            var compressor = new LLMContextCompressor<double>(compressionRatio: 0.5);
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
            var compressor = new LLMContextCompressor<double>(compressionRatio: 0.5);
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
            var compressor = new LLMContextCompressor<double>(compressionRatio: 0.5);
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
        public void Compress_WithEmptyDocument_ReturnsEmptyDocument()
        {
            // Arrange
            var compressor = new LLMContextCompressor<double>(compressionRatio: 0.5);
            var documents = new List<Document<double>>
            {
                new Document<double>("empty", string.Empty)
            };
            var query = "test query";

            // Act
            var result = compressor.Compress(documents, query);

            // Assert
            Assert.Single(result);
            Assert.Equal(string.Empty, result[0].Content);
        }

        [Fact]
        public void Compress_WithSingleSentenceDocument_RetainsSentence()
        {
            // Arrange
            var compressor = new LLMContextCompressor<double>(compressionRatio: 0.5);
            var documents = new List<Document<double>>
            {
                new Document<double>("single", "This is a single sentence about machine learning.")
            };
            var query = "machine learning";

            // Act
            var result = compressor.Compress(documents, query);

            // Assert
            Assert.Single(result);
            Assert.NotEmpty(result[0].Content);
        }

        [Fact]
        public void Compress_WithDocumentSmallerThanTarget_ReturnsOriginal()
        {
            // Arrange
            var compressor = new LLMContextCompressor<double>(compressionRatio: 0.9);
            var documents = new List<Document<double>>
            {
                new Document<double>("small", "Short text.")
            };
            var query = "test";

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
            var compressor = new LLMContextCompressor<double>(compressionRatio: 0.3);
            var largeDoc = CreateLargeDocument("large");
            var documents = new List<Document<double>> { largeDoc };
            var query = "document content";

            // Act
            var result = compressor.Compress(documents, query);

            // Assert
            Assert.Single(result);
            Assert.True(result[0].Content.Length < largeDoc.Content.Length);
            Assert.True(result[0].Content.Length > 0);
        }

        [Fact]
        public void Compress_WithUnicodeContent_HandlesCorrectly()
        {
            // Arrange
            var compressor = new LLMContextCompressor<double>(compressionRatio: 0.5);
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
            var compressor = new LLMContextCompressor<double>(compressionRatio: 0.5);
            var specialDoc = CreateSpecialCharDocument("special");
            var documents = new List<Document<double>> { specialDoc };
            var query = "special chars";

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
            var compressor = new LLMContextCompressor<double>(compressionRatio: 0.5);
            var documents = new List<Document<double>>();
            for (int i = 0; i < 10; i++)
            {
                documents.Add(CreateDocumentWithLength($"doc{i}", 5));
            }
            var query = "test content";

            // Act
            var result = compressor.Compress(documents, query);

            // Assert
            Assert.Equal(10, result.Count);
            foreach (var doc in result)
            {
                Assert.NotNull(doc.Content);
            }
        }

        #endregion

        #region CompressText Method Tests

        [Fact]
        public void CompressText_WithValidInput_CompressesText()
        {
            // Arrange
            var compressor = new LLMContextCompressor<double>(compressionRatio: 0.5);
            var text = "Machine learning is powerful. Neural networks are important. Python is useful. Data science is growing.";
            var query = "machine learning neural";

            // Act
            var result = compressor.CompressText(query, text);

            // Assert
            Assert.NotNull(result);
            Assert.True(result.Length < text.Length);
        }

        [Fact]
        public void CompressText_WithEmptyText_ReturnsEmpty()
        {
            // Arrange
            var compressor = new LLMContextCompressor<double>(compressionRatio: 0.5);
            var query = "test";

            // Act
            var result = compressor.CompressText(query, string.Empty);

            // Assert
            Assert.Equal(string.Empty, result);
        }

        [Fact]
        public void CompressText_WithNullText_ReturnsNull()
        {
            // Arrange
            var compressor = new LLMContextCompressor<double>(compressionRatio: 0.5);
            var query = "test";

            // Act
            var result = compressor.CompressText(query, null);

            // Assert
            Assert.Null(result);
        }

        [Fact]
        public void CompressText_SelectsMostRelevantSentences()
        {
            // Arrange
            var compressor = new LLMContextCompressor<double>(compressionRatio: 0.4);
            var text = "Machine learning is a subset of AI. The weather is nice today. Neural networks learn patterns. Birds are flying outside. Deep learning uses multiple layers.";
            var query = "machine learning neural networks";

            // Act
            var result = compressor.CompressText(query, text);

            // Assert
            Assert.NotNull(result);
            // Should prefer sentences with query terms
            var lowerResult = result.ToLowerInvariant();
            Assert.True(lowerResult.Contains("machine") || lowerResult.Contains("learning") || lowerResult.Contains("neural"));
        }

        #endregion

        #region Integration Tests

        [Fact]
        public void Compress_WithDifferentCompressionRatios_ProducesExpectedRatios()
        {
            // Arrange
            var documents = CreateSampleDocuments();
            var query = "machine learning";
            var ratios = new[] { 0.3, 0.5, 0.7 };
            var results = new List<double>();

            // Act
            foreach (var ratio in ratios)
            {
                var compressor = new LLMContextCompressor<double>(compressionRatio: ratio);
                var compressed = compressor.Compress(documents, query);
                var actualRatio = CalculateCompressionRatio(documents, compressed);
                results.Add(actualRatio);
            }

            // Assert
            for (int i = 0; i < ratios.Length - 1; i++)
            {
                Assert.True(results[i] <= results[i + 1],
                    $"Higher compression ratio should produce longer output. Got {results[i]} and {results[i + 1]}");
            }
        }

        [Fact]
        public void Compress_WithDifferentQueries_ProducesDifferentResults()
        {
            // Arrange
            // Use a single document with distinct sentences for each query
            // This ensures the query affects which sentence is prioritized
            var compressor = new LLMContextCompressor<double>(compressionRatio: 0.3); // Keep ~1 of 3 sentences
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
            var compressor = new LLMContextCompressor<double>(compressionRatio: 0.5);
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

        #endregion
    }
}
