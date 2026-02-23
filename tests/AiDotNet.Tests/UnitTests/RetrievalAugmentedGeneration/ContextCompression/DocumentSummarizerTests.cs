#nullable disable
using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Helpers;
using AiDotNet.RetrievalAugmentedGeneration.ContextCompression;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using Xunit;

namespace AiDotNetTests.UnitTests.RetrievalAugmentedGeneration.ContextCompression
{
    public class DocumentSummarizerTests : ContextCompressorTestBase
    {
        #region Constructor Tests

        [Fact]
        public void Constructor_WithValidParameters_CreatesInstance()
        {
            // Arrange & Act
            var summarizer = new DocumentSummarizer<double>(NumOps, maxSummaryLength: 500);

            // Assert
            Assert.NotNull(summarizer);
        }

        [Fact]
        public void Constructor_WithNullNumericOperations_ThrowsArgumentNullException()
        {
            // Arrange & Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new DocumentSummarizer<double>(null, maxSummaryLength: 500));
        }

        [Fact]
        public void Constructor_WithZeroMaxSummaryLength_ThrowsArgumentOutOfRangeException()
        {
            // Arrange & Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new DocumentSummarizer<double>(NumOps, maxSummaryLength: 0));
        }

        [Fact]
        public void Constructor_WithNegativeMaxSummaryLength_ThrowsArgumentOutOfRangeException()
        {
            // Arrange & Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new DocumentSummarizer<double>(NumOps, maxSummaryLength: -100));
        }

        [Fact]
        public void Constructor_WithCustomMaxSummaryLength_CreatesInstance()
        {
            // Arrange & Act
            var summarizer = new DocumentSummarizer<double>(NumOps, maxSummaryLength: 1000);

            // Assert
            Assert.NotNull(summarizer);
        }

        #endregion

        #region Basic Functionality Tests

        [Fact]
        public void Compress_WithValidDocuments_ReturnsSummarizedDocuments()
        {
            // Arrange
            var summarizer = new DocumentSummarizer<double>(NumOps, maxSummaryLength: 200);
            var documents = CreateSampleDocuments();
            var query = "machine learning";

            // Act
            var result = summarizer.Compress(documents, query);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(documents.Count, result.Count);
        }

        [Fact]
        public void Compress_WithNullDocuments_ThrowsArgumentNullException()
        {
            // Arrange
            var summarizer = new DocumentSummarizer<double>(NumOps, maxSummaryLength: 500);
            var query = "test query";

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                summarizer.Compress(null, query));
        }

        [Fact]
        public void Compress_WithNullQuery_ThrowsArgumentException()
        {
            // Arrange
            var summarizer = new DocumentSummarizer<double>(NumOps, maxSummaryLength: 500);
            var documents = CreateSampleDocuments();

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                summarizer.Compress(documents, null));
        }

        [Fact]
        public void Compress_WithEmptyQuery_ThrowsArgumentException()
        {
            // Arrange
            var summarizer = new DocumentSummarizer<double>(NumOps, maxSummaryLength: 500);
            var documents = CreateSampleDocuments();

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                summarizer.Compress(documents, string.Empty));
        }

        [Fact]
        public void Compress_WithEmptyDocumentList_ReturnsEmptyList()
        {
            // Arrange
            var summarizer = new DocumentSummarizer<double>(NumOps, maxSummaryLength: 500);
            var documents = new List<Document<double>>();
            var query = "test query";

            // Act
            var result = summarizer.Compress(documents, query);

            // Assert
            Assert.NotNull(result);
            Assert.Empty(result);
        }

        #endregion

        #region Summarization Quality Tests

        [Fact]
        public void Compress_LongDocument_RespectsSummaryLength()
        {
            // Arrange
            var maxLength = 200;
            var summarizer = new DocumentSummarizer<double>(NumOps, maxSummaryLength: maxLength);
            var longText = string.Join(" ", Enumerable.Repeat("This is a sentence about machine learning and artificial intelligence.", 50));
            var documents = new List<Document<double>>
            {
                new Document<double>("long", longText)
            };
            var query = "machine learning";

            // Act
            var result = summarizer.Compress(documents, query);

            // Assert
            Assert.Single(result);
            Assert.True(result[0].Content.Length <= maxLength,
                $"Summary length {result[0].Content.Length} should be <= {maxLength}");
        }

        [Fact]
        public void Compress_ShortDocument_ReturnsOriginal()
        {
            // Arrange
            var summarizer = new DocumentSummarizer<double>(NumOps, maxSummaryLength: 500);
            var shortText = "This is a short document.";
            var documents = new List<Document<double>>
            {
                new Document<double>("short", shortText)
            };
            var query = "short";

            // Act
            var result = summarizer.Compress(documents, query);

            // Assert
            Assert.Single(result);
            Assert.Equal(shortText, result[0].Content);
        }

        [Fact]
        public void Compress_WithQueryTerms_PrioritizesRelevantSentences()
        {
            // Arrange
            var summarizer = new DocumentSummarizer<double>(NumOps, maxSummaryLength: 100);
            var text = "Machine learning is important. The sky is blue. Neural networks are powerful. Birds can fly. Deep learning is advanced.";
            var documents = new List<Document<double>>
            {
                new Document<double>("doc", text)
            };
            var query = "machine learning neural networks";

            // Act
            var result = summarizer.Compress(documents, query);

            // Assert
            Assert.Single(result);
            var summary = result[0].Content.ToLowerInvariant();
            // Should contain at least one query term
            Assert.True(
                summary.Contains("machine") ||
                summary.Contains("learning") ||
                summary.Contains("neural"),
                "Summary should prioritize sentences with query terms");
        }

        [Fact]
        public void Compress_PreservesMetadata()
        {
            // Arrange
            var summarizer = new DocumentSummarizer<double>(NumOps, maxSummaryLength: 500);
            var documents = CreateSampleDocuments();
            var query = "machine learning";

            // Act
            var result = summarizer.Compress(documents, query);

            // Assert
            AssertMetadataPreserved(documents, result);
        }

        [Fact]
        public void Compress_PreservesRelevanceScores()
        {
            // Arrange
            var summarizer = new DocumentSummarizer<double>(NumOps, maxSummaryLength: 500);
            var documents = CreateSampleDocuments();
            var query = "machine learning";

            // Act
            var result = summarizer.Compress(documents, query);

            // Assert
            AssertRelevanceScoresPreserved(documents, result);
        }

        [Fact]
        public void Compress_PreservesDocumentIds()
        {
            // Arrange
            var summarizer = new DocumentSummarizer<double>(NumOps, maxSummaryLength: 500);
            var documents = CreateSampleDocuments();
            var query = "machine learning";

            // Act
            var result = summarizer.Compress(documents, query);

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
            var summarizer = new DocumentSummarizer<double>(NumOps, maxSummaryLength: 500);
            var documents = new List<Document<double>>
            {
                new Document<double>("empty", string.Empty)
            };
            var query = "test";

            // Act
            var result = summarizer.Compress(documents, query);

            // Assert
            Assert.Single(result);
            Assert.Equal(string.Empty, result[0].Content);
        }

        [Fact]
        public void Compress_WithVeryLongSingleSentence_TruncatesCorrectly()
        {
            // Arrange
            var maxLength = 100;
            var summarizer = new DocumentSummarizer<double>(NumOps, maxSummaryLength: maxLength);
            var longSentence = new string('a', 500);
            var documents = new List<Document<double>>
            {
                new Document<double>("long", longSentence)
            };
            var query = "test";

            // Act
            var result = summarizer.Compress(documents, query);

            // Assert
            Assert.Single(result);
            Assert.True(result[0].Content.Length <= maxLength,
                $"Result length {result[0].Content.Length} should be <= {maxLength}");
        }

        [Fact]
        public void Compress_WithVeryLargeDocument_SummarizesSuccessfully()
        {
            // Arrange
            var summarizer = new DocumentSummarizer<double>(NumOps, maxSummaryLength: 500);
            var largeDoc = CreateLargeDocument("large");
            var documents = new List<Document<double>> { largeDoc };
            var query = "document content";

            // Act
            var result = summarizer.Compress(documents, query);

            // Assert
            Assert.Single(result);
            Assert.True(result[0].Content.Length <= 500);
            Assert.NotEmpty(result[0].Content);
        }

        [Fact]
        public void Compress_WithUnicodeContent_HandlesCorrectly()
        {
            // Arrange
            var summarizer = new DocumentSummarizer<double>(NumOps, maxSummaryLength: 500);
            var unicodeDoc = CreateUnicodeDocument("unicode");
            var documents = new List<Document<double>> { unicodeDoc };
            var query = "学习";

            // Act
            var result = summarizer.Compress(documents, query);

            // Assert
            Assert.Single(result);
            Assert.NotEmpty(result[0].Content);
        }

        [Fact]
        public void Compress_WithSpecialCharacters_HandlesCorrectly()
        {
            // Arrange
            var summarizer = new DocumentSummarizer<double>(NumOps, maxSummaryLength: 500);
            var specialDoc = CreateSpecialCharDocument("special");
            var documents = new List<Document<double>> { specialDoc };
            var query = "special";

            // Act
            var result = summarizer.Compress(documents, query);

            // Assert
            Assert.Single(result);
            Assert.NotEmpty(result[0].Content);
        }

        [Fact]
        public void Compress_WithMultipleDocuments_ProcessesAllDocuments()
        {
            // Arrange
            var summarizer = new DocumentSummarizer<double>(NumOps, maxSummaryLength: 200);
            var documents = new List<Document<double>>();
            for (int i = 0; i < 10; i++)
            {
                documents.Add(CreateDocumentWithLength($"doc{i}", 20));
            }
            var query = "test content";

            // Act
            var result = summarizer.Compress(documents, query);

            // Assert
            Assert.Equal(10, result.Count);
            foreach (var doc in result)
            {
                Assert.NotNull(doc.Content);
            }
        }

        #endregion

        #region SummarizeText Method Tests

        [Fact]
        public void SummarizeText_WithValidInput_SummarizesText()
        {
            // Arrange
            var summarizer = new DocumentSummarizer<double>(NumOps, maxSummaryLength: 100);
            var text = string.Join(" ", Enumerable.Repeat("This is a sentence about machine learning.", 20));

            // Act
            var result = summarizer.SummarizeText(text);

            // Assert
            Assert.NotNull(result);
            Assert.True(result.Length <= 100);
        }

        [Fact]
        public void SummarizeText_WithEmptyText_ReturnsEmpty()
        {
            // Arrange
            var summarizer = new DocumentSummarizer<double>(NumOps, maxSummaryLength: 500);

            // Act
            var result = summarizer.SummarizeText(string.Empty);

            // Assert
            Assert.Equal(string.Empty, result);
        }

        [Fact]
        public void SummarizeText_WithNullText_ReturnsNull()
        {
            // Arrange
            var summarizer = new DocumentSummarizer<double>(NumOps, maxSummaryLength: 500);

            // Act
            var result = summarizer.SummarizeText(null);

            // Assert
            Assert.Null(result);
        }

        [Fact]
        public void SummarizeText_WithQueryTerms_PrioritizesRelevantContent()
        {
            // Arrange
            var summarizer = new DocumentSummarizer<double>(NumOps, maxSummaryLength: 150);
            var text = "Machine learning is powerful. The weather is nice. Neural networks are important. Birds fly high. Deep learning is advanced.";
            var queryTerms = new List<string> { "machine", "learning", "neural" };

            // Act
            var result = summarizer.SummarizeText(text, queryTerms);

            // Assert
            Assert.NotNull(result);
            var lowerResult = result.ToLowerInvariant();
            Assert.True(
                lowerResult.Contains("machine") ||
                lowerResult.Contains("learning") ||
                lowerResult.Contains("neural"));
        }

        [Fact]
        public void SummarizeText_TextShorterThanMaxLength_ReturnsOriginal()
        {
            // Arrange
            var summarizer = new DocumentSummarizer<double>(NumOps, maxSummaryLength: 500);
            var text = "Short text.";

            // Act
            var result = summarizer.SummarizeText(text);

            // Assert
            Assert.Equal(text, result);
        }

        #endregion

        #region Summarize Method Tests

        [Fact]
        public void Summarize_WithValidDocuments_ReturnsSummarizedDocuments()
        {
            // Arrange
            var summarizer = new DocumentSummarizer<double>(NumOps, maxSummaryLength: 200);
            var documents = CreateSampleDocuments();

            // Act
            var result = summarizer.Summarize(documents);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(documents.Count, result.Count);
        }

        [Fact]
        public void Summarize_WithNullDocuments_ThrowsArgumentNullException()
        {
            // Arrange
            var summarizer = new DocumentSummarizer<double>(NumOps, maxSummaryLength: 500);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                summarizer.Summarize(null));
        }

        [Fact]
        public void Summarize_PreservesMetadata()
        {
            // Arrange
            var summarizer = new DocumentSummarizer<double>(NumOps, maxSummaryLength: 500);
            var documents = CreateSampleDocuments();

            // Act
            var result = summarizer.Summarize(documents);

            // Assert
            AssertMetadataPreserved(documents, result);
        }

        #endregion

        #region Integration Tests

        [Fact]
        public void Compress_WithDifferentMaxLengths_ProducesDifferentResults()
        {
            // Arrange
            var longText = string.Join(" ", Enumerable.Repeat("This is a sentence about machine learning.", 50));
            var documents = new List<Document<double>>
            {
                new Document<double>("doc", longText)
            };
            var query = "machine learning";

            var summarizer1 = new DocumentSummarizer<double>(NumOps, maxSummaryLength: 100);
            var summarizer2 = new DocumentSummarizer<double>(NumOps, maxSummaryLength: 300);

            // Act
            var result1 = summarizer1.Compress(documents, query);
            var result2 = summarizer2.Compress(documents, query);

            // Assert
            Assert.True(result1[0].Content.Length <= 100);
            Assert.True(result2[0].Content.Length <= 300);
            Assert.True(result2[0].Content.Length >= result1[0].Content.Length);
        }

        [Fact]
        public void Compress_MultipleInvocations_ProducesSameResults()
        {
            // Arrange
            var summarizer = new DocumentSummarizer<double>(NumOps, maxSummaryLength: 500);
            var documents = CreateSampleDocuments();
            var query = "machine learning";

            // Act
            var result1 = summarizer.Compress(documents, query);
            var result2 = summarizer.Compress(documents, query);

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
