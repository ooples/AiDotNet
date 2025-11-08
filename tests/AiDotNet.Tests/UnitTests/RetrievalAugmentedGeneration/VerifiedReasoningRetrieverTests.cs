using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.RetrievalAugmentedGeneration.AdvancedPatterns;
using AiDotNet.RetrievalAugmentedGeneration.Generators;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using AiDotNet.RetrievalAugmentedGeneration.Retrievers;
using Xunit;

namespace AiDotNetTests.UnitTests.RetrievalAugmentedGeneration
{
    public class VerifiedReasoningRetrieverTests
    {
        // Mock retriever for testing
        private class MockRetriever : RetrieverBase<double>
        {
            private readonly List<Document<double>> _documents;

            public MockRetriever(List<Document<double>> documents) : base(defaultTopK: 5)
            {
                _documents = documents ?? new List<Document<double>>();
            }

            protected override IEnumerable<Document<double>> RetrieveCore(
                string query,
                int topK,
                Dictionary<string, object> metadataFilters)
            {
                return _documents.Take(topK);
            }
        }

        private MockRetriever CreateMockRetriever()
        {
            var docs = new List<Document<double>>
            {
                new Document<double>
                {
                    Id = "doc1",
                    Content = "Gene therapy involves modifying genes to treat diseases.",
                    RelevanceScore = 0.9,
                    HasRelevanceScore = true
                },
                new Document<double>
                {
                    Id = "doc2",
                    Content = "Safety concerns include immune responses and off-target effects.",
                    RelevanceScore = 0.85,
                    HasRelevanceScore = true
                },
                new Document<double>
                {
                    Id = "doc3",
                    Content = "Clinical trials follow strict protocols to ensure patient safety.",
                    RelevanceScore = 0.8,
                    HasRelevanceScore = true
                }
            };
            return new MockRetriever(docs);
        }

        [Fact]
        public void Constructor_WithNullGenerator_ThrowsArgumentNullException()
        {
            // Arrange
            var mockRetriever = CreateMockRetriever();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new VerifiedReasoningRetriever<double>(null!, mockRetriever));
        }

        [Fact]
        public void Constructor_WithNullRetriever_ThrowsArgumentNullException()
        {
            // Arrange
            var generator = new StubGenerator<double>();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new VerifiedReasoningRetriever<double>(generator, null!));
        }

        [Fact]
        public void Constructor_WithValidArguments_InitializesCorrectly()
        {
            // Arrange
            var generator = new StubGenerator<double>();
            var mockRetriever = CreateMockRetriever();

            // Act
            var verifiedRetriever = new VerifiedReasoningRetriever<double>(
                generator,
                mockRetriever,
                verificationThreshold: 0.7,
                maxRefinementAttempts: 2);

            // Assert
            Assert.NotNull(verifiedRetriever);
        }

        [Fact]
        public void Constructor_WithInvalidThreshold_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            var generator = new StubGenerator<double>();
            var mockRetriever = CreateMockRetriever();

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new VerifiedReasoningRetriever<double>(
                    generator,
                    mockRetriever,
                    verificationThreshold: 1.5));

            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new VerifiedReasoningRetriever<double>(
                    generator,
                    mockRetriever,
                    verificationThreshold: -0.1));
        }

        [Fact]
        public void Constructor_WithNegativeRefinementAttempts_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            var generator = new StubGenerator<double>();
            var mockRetriever = CreateMockRetriever();

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new VerifiedReasoningRetriever<double>(
                    generator,
                    mockRetriever,
                    maxRefinementAttempts: -1));
        }

        [Fact]
        public void RetrieveWithVerification_WithNullQuery_ThrowsArgumentException()
        {
            // Arrange
            var generator = new StubGenerator<double>();
            var mockRetriever = CreateMockRetriever();
            var verifiedRetriever = new VerifiedReasoningRetriever<double>(generator, mockRetriever);

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                verifiedRetriever.RetrieveWithVerification(null!, 10));
        }

        [Fact]
        public void RetrieveWithVerification_WithEmptyQuery_ThrowsArgumentException()
        {
            // Arrange
            var generator = new StubGenerator<double>();
            var mockRetriever = CreateMockRetriever();
            var verifiedRetriever = new VerifiedReasoningRetriever<double>(generator, mockRetriever);

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                verifiedRetriever.RetrieveWithVerification("   ", 10));
        }

        [Fact]
        public void RetrieveWithVerification_WithNegativeTopK_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            var generator = new StubGenerator<double>();
            var mockRetriever = CreateMockRetriever();
            var verifiedRetriever = new VerifiedReasoningRetriever<double>(generator, mockRetriever);

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                verifiedRetriever.RetrieveWithVerification("test query", -1));
        }

        [Fact]
        public void RetrieveWithVerification_WithValidQuery_ReturnsResult()
        {
            // Arrange
            var generator = new StubGenerator<double>();
            var mockRetriever = CreateMockRetriever();
            var verifiedRetriever = new VerifiedReasoningRetriever<double>(
                generator,
                mockRetriever,
                verificationThreshold: 0.5);

            // Act
            var result = verifiedRetriever.RetrieveWithVerification(
                "What are the safety considerations for gene therapy?",
                topK: 10);

            // Assert
            Assert.NotNull(result);
            Assert.NotNull(result.Documents);
            Assert.NotNull(result.VerifiedSteps);
            var docList = result.Documents.ToList();
            Assert.NotEmpty(docList);
            Assert.True(docList.Count <= 10);
        }

        [Fact]
        public void RetrieveWithVerification_ReturnsVerifiedSteps()
        {
            // Arrange
            var generator = new StubGenerator<double>();
            var mockRetriever = CreateMockRetriever();
            var verifiedRetriever = new VerifiedReasoningRetriever<double>(generator, mockRetriever);

            // Act
            var result = verifiedRetriever.RetrieveWithVerification(
                "What are gene therapy safety concerns?",
                topK: 5);

            // Assert
            Assert.NotNull(result.VerifiedSteps);
            Assert.NotEmpty(result.VerifiedSteps);

            foreach (var step in result.VerifiedSteps)
            {
                Assert.NotNull(step.Statement);
                Assert.NotEmpty(step.Statement);
                Assert.True(step.VerificationScore >= 0 && step.VerificationScore <= 1);
                Assert.NotNull(step.CritiqueFeedback);
            }
        }

        [Fact]
        public void RetrieveWithVerification_CalculatesAverageVerificationScore()
        {
            // Arrange
            var generator = new StubGenerator<double>();
            var mockRetriever = CreateMockRetriever();
            var verifiedRetriever = new VerifiedReasoningRetriever<double>(generator, mockRetriever);

            // Act
            var result = verifiedRetriever.RetrieveWithVerification(
                "Gene therapy safety",
                topK: 5);

            // Assert
            Assert.True(result.AverageVerificationScore >= 0);
            Assert.True(result.AverageVerificationScore <= 1);
        }

        [Fact]
        public void RetrieveWithVerification_TracksRefinementAttempts()
        {
            // Arrange
            var generator = new StubGenerator<double>();
            var mockRetriever = CreateMockRetriever();
            var verifiedRetriever = new VerifiedReasoningRetriever<double>(
                generator,
                mockRetriever,
                verificationThreshold: 0.9, // High threshold to trigger refinement
                maxRefinementAttempts: 2);

            // Act
            var result = verifiedRetriever.RetrieveWithVerification(
                "Complex medical query",
                topK: 5);

            // Assert
            Assert.NotNull(result.VerifiedSteps);
            foreach (var step in result.VerifiedSteps)
            {
                Assert.True(step.RefinementAttempts >= 0);
                Assert.True(step.RefinementAttempts <= 2);
            }
        }

        [Fact]
        public void RetrieveWithVerification_WithMetadataFilters_ReturnsResults()
        {
            // Arrange
            var generator = new StubGenerator<double>();
            var mockRetriever = CreateMockRetriever();
            var verifiedRetriever = new VerifiedReasoningRetriever<double>(generator, mockRetriever);
            var filters = new Dictionary<string, object> { { "domain", "medical" } };

            // Act
            var result = verifiedRetriever.RetrieveWithVerification(
                "Gene therapy safety",
                topK: 5,
                filters);

            // Assert
            Assert.NotNull(result);
            Assert.NotNull(result.Documents);
        }

        [Fact]
        public void VerifiedReasoningStep_HasAllRequiredProperties()
        {
            // Arrange
            var step = new VerifiedReasoningRetriever<double>.VerifiedReasoningStep
            {
                Statement = "Test statement",
                SupportingDocuments = new List<Document<double>>(),
                VerificationScore = 0.85,
                IsVerified = true,
                CritiqueFeedback = "Good reasoning",
                RefinementAttempts = 0,
                OriginalStatement = "Original test statement"
            };

            // Assert
            Assert.Equal("Test statement", step.Statement);
            Assert.NotNull(step.SupportingDocuments);
            Assert.Equal(0.85, step.VerificationScore);
            Assert.True(step.IsVerified);
            Assert.Equal("Good reasoning", step.CritiqueFeedback);
            Assert.Equal(0, step.RefinementAttempts);
            Assert.Equal("Original test statement", step.OriginalStatement);
        }

        [Fact]
        public void VerifiedReasoningResult_HasAllRequiredProperties()
        {
            // Arrange
            var result = new VerifiedReasoningRetriever<double>.VerifiedReasoningResult
            {
                Documents = new List<Document<double>>(),
                VerifiedSteps = new List<VerifiedReasoningRetriever<double>.VerifiedReasoningStep>(),
                AverageVerificationScore = 0.75,
                RefinedStepsCount = 2
            };

            // Assert
            Assert.NotNull(result.Documents);
            Assert.NotNull(result.VerifiedSteps);
            Assert.Equal(0.75, result.AverageVerificationScore);
            Assert.Equal(2, result.RefinedStepsCount);
        }
    }
}
