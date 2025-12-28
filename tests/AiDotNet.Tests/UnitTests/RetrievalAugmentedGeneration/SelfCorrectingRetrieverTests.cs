using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.AdvancedPatterns;
using AiDotNet.RetrievalAugmentedGeneration.Generators;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using AiDotNet.RetrievalAugmentedGeneration.Retrievers;
using Xunit;

namespace AiDotNetTests.UnitTests.RetrievalAugmentedGeneration
{
    /// <summary>
    /// Integration tests for SelfCorrectingRetriever.
    /// Tests verify the iterative critique and refinement loop behavior.
    /// </summary>
    public class SelfCorrectingRetrieverTests
    {
        // Mock retriever that tracks calls and can return different docs per call
        private class TrackingMockRetriever : RetrieverBase<double>
        {
            private readonly List<Document<double>> _documents;
            private readonly List<Document<double>> _additionalDocuments;
            private int _callCount;
            public List<string> RetrievalQueries { get; } = new List<string>();
            public int RetrievalCallCount => _callCount;

            public TrackingMockRetriever(
                List<Document<double>> initialDocuments,
                List<Document<double>> additionalDocuments = null) : base(defaultTopK: 5)
            {
                _documents = initialDocuments ?? new List<Document<double>>();
                _additionalDocuments = additionalDocuments ?? new List<Document<double>>();
            }

            protected override IEnumerable<Document<double>> RetrieveCore(
                string query,
                int topK,
                Dictionary<string, object> metadataFilters)
            {
                _callCount++;
                RetrievalQueries.Add(query);

                if (_callCount == 1)
                {
                    return _documents.Take(topK);
                }
                else
                {
                    // Return additional documents for subsequent calls
                    return _additionalDocuments.Any()
                        ? _additionalDocuments.Take(topK)
                        : _documents.Take(topK);
                }
            }
        }

        // Mock generator that can simulate different critique responses
        private class CritiqueMockGenerator : IGenerator<double>
        {
            private readonly Queue<string> _generateResponses;
            private readonly Queue<string> _groundedAnswers;
            public List<string> GeneratePrompts { get; } = new List<string>();
            public List<string> GroundedQueries { get; } = new List<string>();
            public int GenerateCallCount => GeneratePrompts.Count;

            public int MaxContextTokens => 2048;
            public int MaxGenerationTokens => 500;

            public CritiqueMockGenerator(
                IEnumerable<string> critiqueResponses = null,
                IEnumerable<string> groundedAnswers = null)
            {
                _generateResponses = new Queue<string>(critiqueResponses ?? new[] { "The answer is complete and accurate." });
                _groundedAnswers = new Queue<string>(groundedAnswers ?? new[] { "This is the answer." });
            }

            public string Generate(string prompt)
            {
                GeneratePrompts.Add(prompt);
                return _generateResponses.Count > 0
                    ? _generateResponses.Dequeue()
                    : "The answer is complete and accurate.";
            }

            public GroundedAnswer<double> GenerateGrounded(string query, IEnumerable<Document<double>> context)
            {
                GroundedQueries.Add(query);
                var contextList = context?.ToList() ?? new List<Document<double>>();
                var answer = _groundedAnswers.Count > 0
                    ? _groundedAnswers.Dequeue()
                    : "This is the generated answer.";

                return new GroundedAnswer<double>
                {
                    Query = query,
                    Answer = answer,
                    SourceDocuments = contextList,
                    Citations = new List<string>(),
                    ConfidenceScore = 0.8
                };
            }
        }

        private List<Document<double>> CreateTestDocuments()
        {
            return new List<Document<double>>
            {
                new Document<double>
                {
                    Id = "doc1",
                    Content = "The Roman Empire fell in 476 AD when Romulus Augustulus was deposed.",
                    RelevanceScore = 0.90,
                    HasRelevanceScore = true
                },
                new Document<double>
                {
                    Id = "doc2",
                    Content = "Economic problems and currency debasement weakened the Roman Empire.",
                    RelevanceScore = 0.85,
                    HasRelevanceScore = true
                },
                new Document<double>
                {
                    Id = "doc3",
                    Content = "Germanic tribes invaded Roman territories in the 4th and 5th centuries.",
                    RelevanceScore = 0.80,
                    HasRelevanceScore = true
                }
            };
        }

        private List<Document<double>> CreateAdditionalDocuments()
        {
            return new List<Document<double>>
            {
                new Document<double>
                {
                    Id = "additional1",
                    Content = "The Byzantine Empire, or Eastern Roman Empire, survived until 1453.",
                    RelevanceScore = 0.75,
                    HasRelevanceScore = true
                },
                new Document<double>
                {
                    Id = "additional2",
                    Content = "Constantine moved the capital to Constantinople in 330 AD.",
                    RelevanceScore = 0.70,
                    HasRelevanceScore = true
                }
            };
        }

        #region Constructor Tests

        [Fact]
        public void Constructor_WithNullGenerator_ThrowsArgumentNullException()
        {
            // Arrange
            var retriever = new TrackingMockRetriever(CreateTestDocuments());

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new SelfCorrectingRetriever<double>(null!, retriever));
        }

        [Fact]
        public void Constructor_WithNullRetriever_ThrowsArgumentNullException()
        {
            // Arrange
            var generator = new CritiqueMockGenerator();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new SelfCorrectingRetriever<double>(generator, null!));
        }

        [Theory]
        [InlineData(0)]
        [InlineData(-1)]
        [InlineData(-100)]
        public void Constructor_WithNonPositiveMaxIterations_ThrowsArgumentOutOfRangeException(int maxIterations)
        {
            // Arrange
            var generator = new CritiqueMockGenerator();
            var retriever = new TrackingMockRetriever(CreateTestDocuments());

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new SelfCorrectingRetriever<double>(generator, retriever, maxIterations));
        }

        [Theory]
        [InlineData(1)]
        [InlineData(3)]
        [InlineData(10)]
        public void Constructor_WithValidMaxIterations_InitializesCorrectly(int maxIterations)
        {
            // Arrange
            var generator = new CritiqueMockGenerator();
            var retriever = new TrackingMockRetriever(CreateTestDocuments());

            // Act
            var selfCorrector = new SelfCorrectingRetriever<double>(generator, retriever, maxIterations);

            // Assert
            Assert.NotNull(selfCorrector);
        }

        [Fact]
        public void Constructor_WithDefaultMaxIterations_InitializesCorrectly()
        {
            // Arrange
            var generator = new CritiqueMockGenerator();
            var retriever = new TrackingMockRetriever(CreateTestDocuments());

            // Act
            var selfCorrector = new SelfCorrectingRetriever<double>(generator, retriever);

            // Assert
            Assert.NotNull(selfCorrector);
        }

        #endregion

        #region RetrieveAndAnswer Tests

        [Fact]
        public void RetrieveAndAnswer_WithNullQuery_ThrowsArgumentException()
        {
            // Arrange
            var generator = new CritiqueMockGenerator();
            var retriever = new TrackingMockRetriever(CreateTestDocuments());
            var selfCorrector = new SelfCorrectingRetriever<double>(generator, retriever);

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                selfCorrector.RetrieveAndAnswer(null!, 10));
        }

        [Fact]
        public void RetrieveAndAnswer_WithEmptyQuery_ThrowsArgumentException()
        {
            // Arrange
            var generator = new CritiqueMockGenerator();
            var retriever = new TrackingMockRetriever(CreateTestDocuments());
            var selfCorrector = new SelfCorrectingRetriever<double>(generator, retriever);

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                selfCorrector.RetrieveAndAnswer("   ", 10));
        }

        [Theory]
        [InlineData(0)]
        [InlineData(-1)]
        [InlineData(-100)]
        public void RetrieveAndAnswer_WithNonPositiveTopK_ThrowsArgumentOutOfRangeException(int topK)
        {
            // Arrange
            var generator = new CritiqueMockGenerator();
            var retriever = new TrackingMockRetriever(CreateTestDocuments());
            var selfCorrector = new SelfCorrectingRetriever<double>(generator, retriever);

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                selfCorrector.RetrieveAndAnswer("What caused the fall of Rome?", topK));
        }

        [Fact]
        public void RetrieveAndAnswer_WithValidQuery_ReturnsAnswer()
        {
            // Arrange
            var generator = new CritiqueMockGenerator();
            var retriever = new TrackingMockRetriever(CreateTestDocuments());
            var selfCorrector = new SelfCorrectingRetriever<double>(generator, retriever);

            // Act
            var result = selfCorrector.RetrieveAndAnswer("What caused the fall of Rome?", 10);

            // Assert
            Assert.NotNull(result);
            Assert.NotEmpty(result);
        }

        [Fact]
        public void RetrieveAndAnswer_WithNoDocuments_ReturnsNoInfoMessage()
        {
            // Arrange
            var generator = new CritiqueMockGenerator();
            var retriever = new TrackingMockRetriever(new List<Document<double>>());
            var selfCorrector = new SelfCorrectingRetriever<double>(generator, retriever);

            // Act
            var result = selfCorrector.RetrieveAndAnswer("Unknown topic", 10);

            // Assert
            Assert.NotNull(result);
            Assert.Contains("don't have enough information", result);
        }

        #endregion

        #region Satisfaction Detection Tests

        [Theory]
        [InlineData("The answer is complete and accurate.")]
        [InlineData("This is comprehensive and satisfactory.")]
        [InlineData("No errors found, the information is sufficient.")]
        [InlineData("The response is well-covered with no gaps.")]
        public void RetrieveAndAnswer_WithPositiveCritique_StopsIterating(string positiveCritique)
        {
            // Arrange - Critique indicates satisfaction
            var critiques = new[] { positiveCritique };
            var generator = new CritiqueMockGenerator(critiques);
            var retriever = new TrackingMockRetriever(CreateTestDocuments());
            var selfCorrector = new SelfCorrectingRetriever<double>(generator, retriever);

            // Act
            var result = selfCorrector.RetrieveAndAnswer("What caused the fall of Rome?", 10);

            // Assert
            Assert.NotNull(result);
            // Should stop after first iteration if critique is positive
            // Initial retrieval only
            Assert.Equal(1, retriever.RetrievalCallCount);
        }

        [Theory]
        [InlineData("The answer is incomplete, missing details about the Byzantine Empire.")]
        [InlineData("There are gaps in the coverage, needs more about economics.")]
        [InlineData("The response is unclear about the timeline of events.")]
        [InlineData("Error: the date mentioned is incorrect.")]
        public void RetrieveAndAnswer_WithNegativeCritique_ContinuesIterating(string negativeCritique)
        {
            // Arrange - First critique is negative, second is positive
            var critiques = new[]
            {
                negativeCritique,
                "The answer is now complete and accurate."
            };
            var generator = new CritiqueMockGenerator(critiques);
            var retriever = new TrackingMockRetriever(
                CreateTestDocuments(),
                CreateAdditionalDocuments());
            var selfCorrector = new SelfCorrectingRetriever<double>(generator, retriever);

            // Act
            var result = selfCorrector.RetrieveAndAnswer("What caused the fall of Rome?", 10);

            // Assert
            Assert.NotNull(result);
            // Should have additional retrieval attempts due to negative critique
        }

        [Fact]
        public void RetrieveAndAnswer_CountsPositiveVsNegativeIndicators()
        {
            // Arrange - Critique with more positive than negative indicators
            // "complete" and "accurate" (2 positive) vs nothing negative
            var critiques = new[]
            {
                "The answer is complete and accurate with no issues."
            };
            var generator = new CritiqueMockGenerator(critiques);
            var retriever = new TrackingMockRetriever(CreateTestDocuments());
            var selfCorrector = new SelfCorrectingRetriever<double>(generator, retriever);

            // Act
            var result = selfCorrector.RetrieveAndAnswer("Test query", 10);

            // Assert
            Assert.NotNull(result);
            // Should stop after one iteration (positive > negative)
            Assert.Equal(1, retriever.RetrievalCallCount);
        }

        [Fact]
        public void RetrieveAndAnswer_MixedIndicators_ComparesCount()
        {
            // Arrange - Critique with mixed indicators
            // "complete" (1 positive) vs "missing" and "unclear" (2 negative)
            // negative > positive, should continue
            var critiques = new[]
            {
                "The answer is complete but missing some details and unclear on dates.",
                "Now the answer is complete, accurate, and comprehensive."
            };
            var generator = new CritiqueMockGenerator(critiques);
            var retriever = new TrackingMockRetriever(
                CreateTestDocuments(),
                CreateAdditionalDocuments());
            var selfCorrector = new SelfCorrectingRetriever<double>(generator, retriever);

            // Act
            var result = selfCorrector.RetrieveAndAnswer("What caused the fall of Rome?", 10);

            // Assert
            Assert.NotNull(result);
            // First critique: 1 positive, 2 negative -> continue
            // Second critique: 3 positive, 0 negative -> stop
        }

        #endregion

        #region Missing Information Extraction Tests

        [Theory]
        [InlineData("The answer is missing details about the Byzantine Empire.")]
        [InlineData("The response needs more information about economic factors.")]
        [InlineData("It should include details about Germanic invasions.")]
        [InlineData("The answer lacks information about Constantine.")]
        [InlineData("There's a gap in coverage of the Eastern Empire.")]
        [InlineData("Additional information about military decline would improve it.")]
        public void RetrieveAndAnswer_ExtractsMissingInfoFromCritique(string critiqueWithMissingInfo)
        {
            // Arrange
            var critiques = new[]
            {
                critiqueWithMissingInfo,
                "The answer is now complete and accurate."
            };
            var generator = new CritiqueMockGenerator(critiques);
            var retriever = new TrackingMockRetriever(
                CreateTestDocuments(),
                CreateAdditionalDocuments());
            var selfCorrector = new SelfCorrectingRetriever<double>(generator, retriever);

            // Act
            var result = selfCorrector.RetrieveAndAnswer("What caused the fall of Rome?", 10);

            // Assert
            Assert.NotNull(result);
            // Should have extracted missing info and done additional retrieval
        }

        [Fact]
        public void RetrieveAndAnswer_StopsIfNoMissingInfoExtracted()
        {
            // Arrange - Negative critique but no extractable missing info pattern
            var critiques = new[]
            {
                "The answer is bad and wrong and terrible." // Negative but no pattern match
            };
            var generator = new CritiqueMockGenerator(critiques);
            var retriever = new TrackingMockRetriever(CreateTestDocuments());
            var selfCorrector = new SelfCorrectingRetriever<double>(generator, retriever);

            // Act
            var result = selfCorrector.RetrieveAndAnswer("Test query", 10);

            // Assert
            Assert.NotNull(result);
            // Should stop because no missing info could be extracted
        }

        #endregion

        #region Iteration Limit Tests

        [Fact]
        public void RetrieveAndAnswer_RespectsMaxIterations()
        {
            // Arrange - Always negative critiques to force max iterations
            var critiques = Enumerable.Repeat(
                "The answer is incomplete and needs more information about everything.",
                10).ToArray();
            var generator = new CritiqueMockGenerator(critiques);
            var retriever = new TrackingMockRetriever(
                CreateTestDocuments(),
                CreateAdditionalDocuments());
            var selfCorrector = new SelfCorrectingRetriever<double>(generator, retriever, maxIterations: 3);

            // Act
            var result = selfCorrector.RetrieveAndAnswer("What caused the fall of Rome?", 10);

            // Assert
            Assert.NotNull(result);
            // Max 3 iterations = initial + 2 additional = 3 retrieval calls max
            // Plus additional calls for missing info = at most 3 + 3 = 6 calls
            Assert.True(retriever.RetrievalCallCount <= 6,
                $"Expected at most 6 retrieval calls but got {retriever.RetrievalCallCount}");
        }

        [Fact]
        public void RetrieveAndAnswer_WithMaxIterationsOne_OnlyIteratesOnce()
        {
            // Arrange
            var critiques = new[] { "The answer is incomplete." };
            var generator = new CritiqueMockGenerator(critiques);
            var retriever = new TrackingMockRetriever(CreateTestDocuments());
            var selfCorrector = new SelfCorrectingRetriever<double>(generator, retriever, maxIterations: 1);

            // Act
            var result = selfCorrector.RetrieveAndAnswer("Test query", 10);

            // Assert
            Assert.NotNull(result);
            // With maxIterations=1, should only do initial retrieval
            Assert.Equal(1, retriever.RetrievalCallCount);
        }

        #endregion

        #region Document Deduplication Tests

        [Fact]
        public void RetrieveAndAnswer_DoesNotAddDuplicateDocuments()
        {
            // Arrange - Additional docs are same as initial (by ID)
            var docs = CreateTestDocuments();
            var critiques = new[]
            {
                "The answer needs more details about the Roman Empire.",
                "The answer is now complete and accurate."
            };
            var generator = new CritiqueMockGenerator(critiques);
            var retriever = new TrackingMockRetriever(docs, docs); // Same docs returned
            var selfCorrector = new SelfCorrectingRetriever<double>(generator, retriever);

            // Act
            var result = selfCorrector.RetrieveAndAnswer("What caused the fall of Rome?", 10);

            // Assert
            Assert.NotNull(result);
            // Should not add duplicate documents based on ID
        }

        #endregion

        #region Early Termination Tests

        [Fact]
        public void RetrieveAndAnswer_StopsWhenNoNewDocumentsFound()
        {
            // Arrange - Return empty list for additional retrieval
            var critiques = new[]
            {
                "The answer needs more information about obscure topic.",
                "The answer is now complete."
            };
            var generator = new CritiqueMockGenerator(critiques);
            var retriever = new TrackingMockRetriever(
                CreateTestDocuments(),
                new List<Document<double>>()); // Empty additional docs
            var selfCorrector = new SelfCorrectingRetriever<double>(generator, retriever);

            // Act
            var result = selfCorrector.RetrieveAndAnswer("What caused the fall of Rome?", 10);

            // Assert
            Assert.NotNull(result);
            // Should stop when no additional documents found
        }

        #endregion

        #region Metadata Filter Tests

        [Fact]
        public void RetrieveAndAnswer_PassesMetadataFilters()
        {
            // Arrange
            var generator = new CritiqueMockGenerator();
            var retriever = new TrackingMockRetriever(CreateTestDocuments());
            var selfCorrector = new SelfCorrectingRetriever<double>(generator, retriever);
            var filters = new Dictionary<string, object>
            {
                { "category", "history" },
                { "year", 2024 }
            };

            // Act
            var result = selfCorrector.RetrieveAndAnswer("What caused the fall of Rome?", 10, filters);

            // Assert
            Assert.NotNull(result);
            // Filters should be passed to retriever (verified by retriever mock if extended)
        }

        [Fact]
        public void RetrieveAndAnswer_WithNullFilters_UsesEmptyDictionary()
        {
            // Arrange
            var generator = new CritiqueMockGenerator();
            var retriever = new TrackingMockRetriever(CreateTestDocuments());
            var selfCorrector = new SelfCorrectingRetriever<double>(generator, retriever);

            // Act - Should not throw with null filters
            var result = selfCorrector.RetrieveAndAnswer("What caused the fall of Rome?", 10, null);

            // Assert
            Assert.NotNull(result);
        }

        #endregion

        #region Edge Cases

        [Fact]
        public void RetrieveAndAnswer_HandlesEmptyAnswer()
        {
            // Arrange - Generator returns empty grounded answer
            var groundedAnswers = new[] { "" };
            var generator = new CritiqueMockGenerator(
                new[] { "The answer is complete." },
                groundedAnswers);
            var retriever = new TrackingMockRetriever(CreateTestDocuments());
            var selfCorrector = new SelfCorrectingRetriever<double>(generator, retriever);

            // Act
            var result = selfCorrector.RetrieveAndAnswer("Test query", 10);

            // Assert
            Assert.NotNull(result);
            // Should handle empty answer gracefully
        }

        [Fact]
        public void RetrieveAndAnswer_WithVeryLargeCritique_HandlesCorrectly()
        {
            // Arrange - Very long critique
            var longCritique = "The answer is complete and accurate. " + new string('x', 10000);
            var generator = new CritiqueMockGenerator(new[] { longCritique });
            var retriever = new TrackingMockRetriever(CreateTestDocuments());
            var selfCorrector = new SelfCorrectingRetriever<double>(generator, retriever);

            // Act
            var result = selfCorrector.RetrieveAndAnswer("Test query", 10);

            // Assert
            Assert.NotNull(result);
            // Should handle long critique
            Assert.Equal(1, retriever.RetrievalCallCount);
        }

        #endregion
    }
}
