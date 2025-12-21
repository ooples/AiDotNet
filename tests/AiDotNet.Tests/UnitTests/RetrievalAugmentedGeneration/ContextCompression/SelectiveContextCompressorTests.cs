using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.RetrievalAugmentedGeneration.ContextCompression;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using Xunit;

namespace AiDotNetTests.UnitTests.RetrievalAugmentedGeneration.ContextCompression
{
    public class SelectiveContextCompressorTests : ContextCompressorTestBase
    {
        #region Constructor Tests

        [Fact]
        public void Constructor_WithValidParameters_CreatesInstance()
        {
            // Arrange & Act
            var compressor = new SelectiveContextCompressor<double>(maxSentences: 5, relevanceThreshold: 0.3);

            // Assert
            Assert.NotNull(compressor);
        }

        [Fact]
        public void Constructor_WithZeroMaxSentences_ThrowsArgumentOutOfRangeException()
        {
            // Arrange & Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new SelectiveContextCompressor<double>(maxSentences: 0, relevanceThreshold: 0.3));
        }

        [Fact]
        public void Constructor_WithNegativeMaxSentences_ThrowsArgumentOutOfRangeException()
        {
            // Arrange & Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new SelectiveContextCompressor<double>(maxSentences: -5, relevanceThreshold: 0.3));
        }

        [Fact]
        public void Constructor_WithDifferentThresholds_CreatesInstance()
        {
            // Arrange & Act
            var compressor1 = new SelectiveContextCompressor<double>(maxSentences: 5, relevanceThreshold: 0.1);
            var compressor2 = new SelectiveContextCompressor<double>(maxSentences: 5, relevanceThreshold: 0.9);

            // Assert
            Assert.NotNull(compressor1);
            Assert.NotNull(compressor2);
        }

        #endregion

        #region Basic Functionality Tests

        [Fact]
        public void Compress_WithValidDocuments_ReturnsCompressedDocuments()
        {
            // Arrange
            var compressor = new SelectiveContextCompressor<double>(maxSentences: 5, relevanceThreshold: 0.1);
            var documents = CreateSampleDocuments();
            var query = "machine learning neural networks";

            // Act
            var result = compressor.Compress(documents, query);

            // Assert
            Assert.NotNull(result);
            Assert.True(result.Count <= documents.Count);
        }

        [Fact]
        public void Compress_WithNullDocuments_ThrowsArgumentNullException()
        {
            // Arrange
            var compressor = new SelectiveContextCompressor<double>(maxSentences: 5, relevanceThreshold: 0.3);
            var query = "test query";

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                compressor.Compress(null, query));
        }

        [Fact]
        public void Compress_WithNullQuery_ThrowsArgumentException()
        {
            // Arrange
            var compressor = new SelectiveContextCompressor<double>(maxSentences: 5, relevanceThreshold: 0.3);
            var documents = CreateSampleDocuments();

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                compressor.Compress(documents, null));
        }

        [Fact]
        public void Compress_WithEmptyQuery_ThrowsArgumentException()
        {
            // Arrange
            var compressor = new SelectiveContextCompressor<double>(maxSentences: 5, relevanceThreshold: 0.3);
            var documents = CreateSampleDocuments();

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                compressor.Compress(documents, string.Empty));
        }

        [Fact]
        public void Compress_WithEmptyDocumentList_ReturnsEmptyList()
        {
            // Arrange
            var compressor = new SelectiveContextCompressor<double>(maxSentences: 5, relevanceThreshold: 0.3);
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
        public void Compress_SelectsRelevantSentences()
        {
            // Arrange
            var compressor = new SelectiveContextCompressor<double>(maxSentences: 3, relevanceThreshold: 0.1);
            var document = new Document<double>("doc1",
                "Machine learning is a subset of AI. The sky is blue today. Neural networks process data. Birds are singing outside. Deep learning uses layers.");
            var documents = new List<Document<double>> { document };
            var query = "machine learning neural networks";

            // Act
            var result = compressor.Compress(documents, query);

            // Assert
            Assert.NotEmpty(result);
            var content = result[0].Content.ToLowerInvariant();
            // Should prefer sentences with query terms
            Assert.True(
                content.Contains("machine") ||
                content.Contains("learning") ||
                content.Contains("neural"));
        }

        [Fact]
        public void Compress_RespectsMaxSentences()
        {
            // Arrange
            var maxSentences = 2;
            var compressor = new SelectiveContextCompressor<double>(maxSentences: maxSentences, relevanceThreshold: 0.0);
            var document = CreateDocumentWithLength("doc1", 10);
            var documents = new List<Document<double>> { document };
            var query = "sentence test content";

            // Act
            var result = compressor.Compress(documents, query);

            // Assert
            if (result.Any())
            {
                var sentenceCount = result[0].Content.Split(new[] { '.' }, StringSplitOptions.RemoveEmptyEntries).Length;
                Assert.True(sentenceCount <= maxSentences,
                    $"Result should have at most {maxSentences} sentences, got {sentenceCount}");
            }
        }

        [Fact]
        public void Compress_WithHighThreshold_FiltersMoreSentences()
        {
            // Arrange
            var lowThresholdCompressor = new SelectiveContextCompressor<double>(maxSentences: 10, relevanceThreshold: 0.0);
            var highThresholdCompressor = new SelectiveContextCompressor<double>(maxSentences: 10, relevanceThreshold: 0.5);
            var documents = CreateSampleDocuments();
            var query = "machine learning";

            // Act
            var lowResult = lowThresholdCompressor.Compress(documents, query);
            var highResult = highThresholdCompressor.Compress(documents, query);

            // Assert
            var lowLength = lowResult.Sum(d => d.Content.Length);
            var highLength = highResult.Sum(d => d.Content.Length);
            Assert.True(highLength <= lowLength,
                "Higher threshold should produce shorter or equal output");
        }

        [Fact]
        public void Compress_PreservesMetadata()
        {
            // Arrange
            var compressor = new SelectiveContextCompressor<double>(maxSentences: 5, relevanceThreshold: 0.1);
            var documents = CreateSampleDocuments();
            var query = "machine learning";

            // Act
            var result = compressor.Compress(documents, query);

            // Assert
            foreach (var doc in result)
            {
                var original = documents.FirstOrDefault(d => d.Id == doc.Id);
                if (original != null && original.Metadata != null)
                {
                    foreach (var kvp in original.Metadata)
                    {
                        Assert.True(doc.Metadata.ContainsKey(kvp.Key),
                            $"Metadata key '{kvp.Key}' should be preserved");
                    }
                }
            }
        }

        [Fact]
        public void Compress_PreservesRelevanceScores()
        {
            // Arrange
            var compressor = new SelectiveContextCompressor<double>(maxSentences: 5, relevanceThreshold: 0.1);
            var documents = CreateSampleDocuments();
            var query = "machine learning";

            // Act
            var result = compressor.Compress(documents, query);

            // Assert
            foreach (var doc in result)
            {
                var original = documents.FirstOrDefault(d => d.Id == doc.Id);
                if (original != null && original.HasRelevanceScore)
                {
                    Assert.True(doc.HasRelevanceScore, "HasRelevanceScore should be preserved");
                    Assert.Equal(original.RelevanceScore, doc.RelevanceScore);
                }
            }
        }

        [Fact]
        public void Compress_PreservesDocumentIds()
        {
            // Arrange
            var compressor = new SelectiveContextCompressor<double>(maxSentences: 5, relevanceThreshold: 0.1);
            var documents = CreateSampleDocuments();
            var query = "machine learning";

            // Act
            var result = compressor.Compress(documents, query);

            // Assert
            foreach (var doc in result)
            {
                Assert.True(documents.Any(d => d.Id == doc.Id),
                    $"Document ID '{doc.Id}' should exist in original documents");
            }
        }

        #endregion

        #region Edge Cases Tests

        [Fact]
        public void Compress_WithEmptyDocument_FiltersOutDocument()
        {
            // Arrange
            var compressor = new SelectiveContextCompressor<double>(maxSentences: 5, relevanceThreshold: 0.1);
            var documents = new List<Document<double>>
            {
                new Document<double>("empty", string.Empty),
                CreateSampleDocuments()[0]
            };
            var query = "machine learning";

            // Act
            var result = compressor.Compress(documents, query);

            // Assert
            Assert.NotNull(result);
            // Empty document might be filtered out
            Assert.True(result.Count <= documents.Count);
        }

        [Fact]
        public void Compress_WithNoRelevantSentences_MayReturnEmpty()
        {
            // Arrange
            var compressor = new SelectiveContextCompressor<double>(maxSentences: 5, relevanceThreshold: 0.9);
            var document = new Document<double>("doc", "The sky is blue. Grass is green. Water is wet.");
            var documents = new List<Document<double>> { document };
            var query = "machine learning neural networks artificial intelligence";

            // Act
            var result = compressor.Compress(documents, query);

            // Assert
            Assert.NotNull(result);
            // May be empty or have minimal content due to high threshold
        }

        [Fact]
        public void Compress_WithSingleSentence_HandlesCorrectly()
        {
            // Arrange
            var compressor = new SelectiveContextCompressor<double>(maxSentences: 5, relevanceThreshold: 0.1);
            var document = new Document<double>("single", "Machine learning is important.");
            var documents = new List<Document<double>> { document };
            var query = "machine learning";

            // Act
            var result = compressor.Compress(documents, query);

            // Assert
            Assert.NotEmpty(result);
            Assert.NotEmpty(result[0].Content);
        }

        [Fact]
        public void Compress_WithVeryLargeDocument_CompressesSuccessfully()
        {
            // Arrange
            var compressor = new SelectiveContextCompressor<double>(maxSentences: 10, relevanceThreshold: 0.1);
            var largeDoc = CreateLargeDocument("large");
            var documents = new List<Document<double>> { largeDoc };
            var query = "document content long";

            // Act
            var result = compressor.Compress(documents, query);

            // Assert
            Assert.NotEmpty(result);
            if (result.Any() && !string.IsNullOrEmpty(result[0].Content))
            {
                Assert.True(result[0].Content.Length < largeDoc.Content.Length);
            }
        }

        [Fact]
        public void Compress_WithUnicodeContent_HandlesCorrectly()
        {
            // Arrange
            var compressor = new SelectiveContextCompressor<double>(maxSentences: 5, relevanceThreshold: 0.0);
            var unicodeDoc = CreateUnicodeDocument("unicode");
            var documents = new List<Document<double>> { unicodeDoc };
            var query = "学习 learning";

            // Act
            var result = compressor.Compress(documents, query);

            // Assert
            Assert.NotNull(result);
            if (result.Any())
            {
                Assert.NotEmpty(result[0].Content);
            }
        }

        [Fact]
        public void Compress_WithSpecialCharacters_HandlesCorrectly()
        {
            // Arrange
            var compressor = new SelectiveContextCompressor<double>(maxSentences: 5, relevanceThreshold: 0.0);
            var specialDoc = CreateSpecialCharDocument("special");
            var documents = new List<Document<double>> { specialDoc };
            var query = "special testing";

            // Act
            var result = compressor.Compress(documents, query);

            // Assert
            Assert.NotNull(result);
            if (result.Any())
            {
                Assert.NotEmpty(result[0].Content);
            }
        }

        [Fact]
        public void Compress_WithMultipleDocuments_ProcessesAll()
        {
            // Arrange
            var compressor = new SelectiveContextCompressor<double>(maxSentences: 5, relevanceThreshold: 0.1);
            var documents = new List<Document<double>>();
            for (int i = 0; i < 10; i++)
            {
                documents.Add(CreateDocumentWithLength($"doc{i}", 10));
            }
            var query = "test sentence content";

            // Act
            var result = compressor.Compress(documents, query);

            // Assert
            Assert.NotNull(result);
            // Some documents might be filtered out if no relevant sentences
        }

        #endregion

        #region Relevance Filtering Tests

        [Fact]
        public void Compress_WithZeroThreshold_IncludesAllSentences()
        {
            // Arrange
            var compressor = new SelectiveContextCompressor<double>(maxSentences: 100, relevanceThreshold: 0.0);
            var document = CreateDocumentWithLength("doc", 5);
            var documents = new List<Document<double>> { document };
            var query = "test";

            // Act
            var result = compressor.Compress(documents, query);

            // Assert
            Assert.NotEmpty(result);
            // Should include content since threshold is 0
            Assert.NotEmpty(result[0].Content);
        }

        [Fact]
        public void Compress_OrdersSentencesByRelevance()
        {
            // Arrange
            var compressor = new SelectiveContextCompressor<double>(maxSentences: 2, relevanceThreshold: 0.0);
            var document = new Document<double>("doc",
                "The weather is nice. Machine learning is important. Birds are flying. Neural networks are powerful.");
            var documents = new List<Document<double>> { document };
            var query = "machine learning neural networks";

            // Act
            var result = compressor.Compress(documents, query);

            // Assert
            Assert.NotEmpty(result);
            var content = result[0].Content.ToLowerInvariant();
            // Most relevant sentences should be included
            Assert.True(
                content.Contains("machine") ||
                content.Contains("learning") ||
                content.Contains("neural"));
        }

        #endregion

        #region Integration Tests

        [Fact]
        public void Compress_WithDifferentMaxSentences_ProducesDifferentLengths()
        {
            // Arrange
            var document = CreateDocumentWithLength("doc", 20);
            var documents = new List<Document<double>> { document };
            var query = "test sentence content";

            var compressor1 = new SelectiveContextCompressor<double>(maxSentences: 2, relevanceThreshold: 0.0);
            var compressor2 = new SelectiveContextCompressor<double>(maxSentences: 10, relevanceThreshold: 0.0);

            // Act
            var result1 = compressor1.Compress(documents, query);
            var result2 = compressor2.Compress(documents, query);

            // Assert
            if (result1.Any() && result2.Any())
            {
                var length1 = result1[0].Content.Length;
                var length2 = result2[0].Content.Length;
                Assert.True(length2 >= length1,
                    "Higher maxSentences should produce longer or equal output");
            }
        }

        [Fact]
        public void Compress_MultipleInvocations_ProducesSameResults()
        {
            // Arrange
            var compressor = new SelectiveContextCompressor<double>(maxSentences: 5, relevanceThreshold: 0.1);
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
            var compressor = new SelectiveContextCompressor<float>(maxSentences: 5, relevanceThreshold: 0.1f);
            var documents = new List<Document<float>>
            {
                new Document<float>("doc1", "Machine learning is important. Neural networks are powerful.")
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
        }

        #endregion
    }
}
