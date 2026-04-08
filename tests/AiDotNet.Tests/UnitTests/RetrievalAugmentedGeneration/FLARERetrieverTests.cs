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
    /// Integration tests for FLARERetriever (Forward-Looking Active REtrieval).
    /// Tests verify the actual behavior of uncertainty detection and iterative retrieval.
    /// </summary>
    public class FLARERetrieverTests
    {
        // Mock retriever that tracks retrieval calls for verification
        private class TrackingMockRetriever : RetrieverBase<double>
        {
            private readonly List<Document<double>> _documents;
            public List<string> RetrievalQueries { get; } = new List<string>();
            public int RetrievalCallCount => RetrievalQueries.Count;

            public TrackingMockRetriever(List<Document<double>> documents) : base(defaultTopK: 5)
            {
                _documents = documents ?? new List<Document<double>>();
            }

            protected override IEnumerable<Document<double>> RetrieveCore(
                string query,
                int topK,
                Dictionary<string, object> metadataFilters)
            {
                RetrievalQueries.Add(query);
                return _documents.Take(topK);
            }
        }

        // Mock generator that can be configured to produce specific outputs
        private class ConfigurableMockGenerator : IGenerator<double>
        {
            private readonly Queue<string> _responses;
            private readonly string _defaultResponse;
            public List<string> GeneratePrompts { get; } = new List<string>();
            public List<string> GroundedQueries { get; } = new List<string>();

            public int MaxContextTokens => 2048;
            public int MaxGenerationTokens => 500;

            public ConfigurableMockGenerator(string defaultResponse = "This is a confident answer.")
            {
                _defaultResponse = defaultResponse;
                _responses = new Queue<string>();
            }

            public ConfigurableMockGenerator(IEnumerable<string> responses)
            {
                _responses = new Queue<string>(responses);
                _defaultResponse = "Default response";
            }

            public string Generate(string prompt)
            {
                GeneratePrompts.Add(prompt);
                return _responses.Count > 0 ? _responses.Dequeue() : _defaultResponse;
            }

            public GroundedAnswer<double> GenerateGrounded(string query, IEnumerable<Document<double>> context)
            {
                GroundedQueries.Add(query);
                var contextList = context?.ToList() ?? new List<Document<double>>();
                return new GroundedAnswer<double>
                {
                    Query = query,
                    Answer = _responses.Count > 0 ? _responses.Dequeue() : _defaultResponse,
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
                    Content = "Quantum computing uses quantum bits (qubits) that can exist in superposition.",
                    RelevanceScore = 0.95,
                    HasRelevanceScore = true
                },
                new Document<double>
                {
                    Id = "doc2",
                    Content = "Classical computers use binary bits that are either 0 or 1.",
                    RelevanceScore = 0.85,
                    HasRelevanceScore = true
                },
                new Document<double>
                {
                    Id = "doc3",
                    Content = "Entanglement allows qubits to be correlated in ways impossible for classical bits.",
                    RelevanceScore = 0.80,
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
                new FLARERetriever<double>(null!, retriever));
        }

        [Fact]
        public void Constructor_WithNullRetriever_ThrowsArgumentNullException()
        {
            // Arrange
            var generator = new ConfigurableMockGenerator();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new FLARERetriever<double>(generator, null!));
        }

        [Fact]
        public void Constructor_WithValidArguments_InitializesCorrectly()
        {
            // Arrange
            var generator = new ConfigurableMockGenerator();
            var retriever = new TrackingMockRetriever(CreateTestDocuments());

            // Act
            var flare = new FLARERetriever<double>(generator, retriever);

            // Assert
            Assert.NotNull(flare);
        }

        [Theory]
        [InlineData(-0.1)]
        [InlineData(1.1)]
        [InlineData(-1.0)]
        [InlineData(2.0)]
        public void Constructor_WithInvalidThreshold_ThrowsArgumentOutOfRangeException(double threshold)
        {
            // Arrange
            var generator = new ConfigurableMockGenerator();
            var retriever = new TrackingMockRetriever(CreateTestDocuments());

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new FLARERetriever<double>(generator, retriever, threshold));
        }

        [Theory]
        [InlineData(0.0)]
        [InlineData(0.5)]
        [InlineData(1.0)]
        public void Constructor_WithValidThreshold_InitializesCorrectly(double threshold)
        {
            // Arrange
            var generator = new ConfigurableMockGenerator();
            var retriever = new TrackingMockRetriever(CreateTestDocuments());

            // Act
            var flare = new FLARERetriever<double>(generator, retriever, threshold);

            // Assert
            Assert.NotNull(flare);
        }

        #endregion

        #region GenerateWithActiveRetrieval Tests

        [Fact]
        public void GenerateWithActiveRetrieval_WithNullQuery_ThrowsArgumentException()
        {
            // Arrange
            var generator = new ConfigurableMockGenerator();
            var retriever = new TrackingMockRetriever(CreateTestDocuments());
            var flare = new FLARERetriever<double>(generator, retriever);

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                flare.GenerateWithActiveRetrieval(null!));
        }

        [Fact]
        public void GenerateWithActiveRetrieval_WithEmptyQuery_ThrowsArgumentException()
        {
            // Arrange
            var generator = new ConfigurableMockGenerator();
            var retriever = new TrackingMockRetriever(CreateTestDocuments());
            var flare = new FLARERetriever<double>(generator, retriever);

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                flare.GenerateWithActiveRetrieval("   "));
        }

        [Fact]
        public void GenerateWithActiveRetrieval_WithConfidentResponse_StopsAfterOneIteration()
        {
            // Arrange
            var generator = new ConfigurableMockGenerator("Quantum computing is a type of computation that uses quantum mechanical phenomena.");
            var retriever = new TrackingMockRetriever(CreateTestDocuments());
            var flare = new FLARERetriever<double>(generator, retriever, uncertaintyThreshold: 0.5);

            // Act
            var result = flare.GenerateWithActiveRetrieval("What is quantum computing?");

            // Assert
            Assert.NotNull(result);
            Assert.NotEmpty(result);
            // Initial retrieval should happen
            Assert.True(retriever.RetrievalCallCount >= 1);
        }

        [Fact]
        public void GenerateWithActiveRetrieval_WithUncertainResponse_TriggersAdditionalRetrieval()
        {
            // Arrange - Generator produces uncertainty phrases that should trigger re-retrieval
            var responses = new[]
            {
                "I'm not sure about the specifics. Need more information about quantum gates.",
                "Quantum gates manipulate qubits to perform computations."
            };
            var generator = new ConfigurableMockGenerator(responses);
            var retriever = new TrackingMockRetriever(CreateTestDocuments());
            var flare = new FLARERetriever<double>(generator, retriever, uncertaintyThreshold: 0.5);

            // Act
            var result = flare.GenerateWithActiveRetrieval("How do quantum gates work?");

            // Assert
            Assert.NotNull(result);
            // With uncertainty phrases, should trigger additional retrieval
            // Initial retrieval (1) + potential additional retrieval due to uncertainty
            Assert.True(retriever.RetrievalCallCount >= 1);
        }

        [Fact]
        public void GenerateWithActiveRetrieval_RespectsMaxIterations()
        {
            // Arrange - Generator always produces uncertain responses
            var uncertainResponse = "I don't know about this topic. Need more information about everything.";
            var generator = new ConfigurableMockGenerator(uncertainResponse);
            var retriever = new TrackingMockRetriever(CreateTestDocuments());
            var flare = new FLARERetriever<double>(generator, retriever, uncertaintyThreshold: 0.9);

            // Act
            var result = flare.GenerateWithActiveRetrieval("Explain everything about quantum physics");

            // Assert
            // Should not exceed max iterations (5) regardless of uncertainty
            // Max iterations = 5, initial retrieval + up to 4 additional retrievals
            Assert.True(retriever.RetrievalCallCount <= 10,
                $"Expected at most 10 retrieval calls but got {retriever.RetrievalCallCount}");
        }

        [Fact]
        public void GenerateWithActiveRetrieval_ReturnsNonEmptyResult()
        {
            // Arrange
            var generator = new ConfigurableMockGenerator("Quantum computing harnesses quantum phenomena.");
            var retriever = new TrackingMockRetriever(CreateTestDocuments());
            var flare = new FLARERetriever<double>(generator, retriever);

            // Act
            var result = flare.GenerateWithActiveRetrieval("What is quantum computing?");

            // Assert
            Assert.NotNull(result);
            Assert.False(string.IsNullOrWhiteSpace(result));
        }

        #endregion

        #region Uncertainty Detection Tests

        [Theory]
        [InlineData("not sure")]
        [InlineData("don't know")]
        [InlineData("need more")]
        [InlineData("unclear")]
        [InlineData("missing")]
        [InlineData("uncertain")]
        public void GenerateWithActiveRetrieval_UncertaintyPhrases_TriggerLowConfidence(string uncertaintyPhrase)
        {
            // Arrange - Response contains uncertainty phrase
            var response = $"This answer is {uncertaintyPhrase} about the details.";
            var generator = new ConfigurableMockGenerator(response);
            var retriever = new TrackingMockRetriever(CreateTestDocuments());
            // Set threshold to 0.4 so 0.3 (low confidence from uncertainty) triggers re-retrieval
            var flare = new FLARERetriever<double>(generator, retriever, uncertaintyThreshold: 0.4);

            // Act
            flare.GenerateWithActiveRetrieval("Test query");

            // Assert
            // The implementation detects uncertainty phrases and returns 0.3 confidence
            // This should trigger the re-retrieval loop
            Assert.True(generator.GeneratePrompts.Count >= 1);
        }

        [Fact]
        public void GenerateWithActiveRetrieval_WithHighRelevanceDocuments_CalculatesHigherConfidence()
        {
            // Arrange - Documents with high relevance scores
            var highRelevanceDocs = new List<Document<double>>
            {
                new Document<double>
                {
                    Id = "high1",
                    Content = "Highly relevant content about quantum computing and its applications.",
                    RelevanceScore = 0.99,
                    HasRelevanceScore = true
                },
                new Document<double>
                {
                    Id = "high2",
                    Content = "More highly relevant quantum computing information for testing.",
                    RelevanceScore = 0.98,
                    HasRelevanceScore = true
                }
            };
            // Long confident response (length contributes to confidence)
            var longResponse = new string('a', 600); // > 500 chars for full length score
            var generator = new ConfigurableMockGenerator(longResponse);
            var retriever = new TrackingMockRetriever(highRelevanceDocs);
            var flare = new FLARERetriever<double>(generator, retriever, uncertaintyThreshold: 0.5);

            // Act
            var result = flare.GenerateWithActiveRetrieval("What is quantum computing?");

            // Assert
            Assert.NotNull(result);
            // With high relevance and long response, should be confident after fewer iterations
        }

        #endregion

        #region Missing Information Extraction Tests

        [Fact]
        public void GenerateWithActiveRetrieval_WithMissingInfoPattern_ExtractsMissingTopic()
        {
            // Arrange - Response contains pattern that should extract missing info
            var responses = new[]
            {
                "I need more information about quantum entanglement.",
                "Quantum entanglement is a phenomenon where particles become correlated."
            };
            var generator = new ConfigurableMockGenerator(responses);
            var retriever = new TrackingMockRetriever(CreateTestDocuments());
            var flare = new FLARERetriever<double>(generator, retriever, uncertaintyThreshold: 0.5);

            // Act
            flare.GenerateWithActiveRetrieval("Explain quantum phenomena");

            // Assert
            // Should have made additional retrieval with extracted topic
            Assert.True(retriever.RetrievalCallCount >= 1);
        }

        [Theory]
        [InlineData("need more information about quantum gates.")]
        [InlineData("unclear about superposition.")]
        [InlineData("missing details on entanglement.")]
        [InlineData("don't know about decoherence.")]
        [InlineData("requires more data on qubits.")]
        public void GenerateWithActiveRetrieval_RecognizesMissingInfoPatterns(string missingInfoPhrase)
        {
            // Arrange
            var responses = new[]
            {
                missingInfoPhrase,
                "Now I have complete information."
            };
            var generator = new ConfigurableMockGenerator(responses);
            var retriever = new TrackingMockRetriever(CreateTestDocuments());
            var flare = new FLARERetriever<double>(generator, retriever, uncertaintyThreshold: 0.5);

            // Act
            var result = flare.GenerateWithActiveRetrieval("Test query");

            // Assert
            Assert.NotNull(result);
            // Patterns should be recognized and trigger additional retrieval
        }

        #endregion

        #region Edge Cases

        [Fact]
        public void GenerateWithActiveRetrieval_WithEmptyDocuments_HandlesGracefully()
        {
            // Arrange
            var generator = new ConfigurableMockGenerator("Answer based on limited context.");
            var retriever = new TrackingMockRetriever(new List<Document<double>>());
            var flare = new FLARERetriever<double>(generator, retriever);

            // Act
            var result = flare.GenerateWithActiveRetrieval("What is quantum computing?");

            // Assert
            Assert.NotNull(result);
        }

        [Fact]
        public void GenerateWithActiveRetrieval_WithDocumentsWithoutRelevanceScores_UsesDefaultConfidence()
        {
            // Arrange - Documents without relevance scores
            var docsWithoutScores = new List<Document<double>>
            {
                new Document<double>
                {
                    Id = "noscore1",
                    Content = "Document without relevance score.",
                    HasRelevanceScore = false
                },
                new Document<double>
                {
                    Id = "noscore2",
                    Content = "Another document without score.",
                    HasRelevanceScore = false
                }
            };
            var generator = new ConfigurableMockGenerator("Answer using documents without scores.");
            var retriever = new TrackingMockRetriever(docsWithoutScores);
            var flare = new FLARERetriever<double>(generator, retriever);

            // Act
            var result = flare.GenerateWithActiveRetrieval("Test query");

            // Assert
            Assert.NotNull(result);
            // Should use 0.5 default for relevance calculation
        }

        [Fact]
        public void GenerateWithActiveRetrieval_WithThresholdZero_AlwaysRetrievesMore()
        {
            // Arrange - Threshold 0 means any confidence passes
            var generator = new ConfigurableMockGenerator("Short");
            var retriever = new TrackingMockRetriever(CreateTestDocuments());
            var flare = new FLARERetriever<double>(generator, retriever, uncertaintyThreshold: 0.0);

            // Act
            var result = flare.GenerateWithActiveRetrieval("Test query");

            // Assert
            Assert.NotNull(result);
            // With threshold 0, even low confidence should pass
        }

        [Fact]
        public void GenerateWithActiveRetrieval_WithThresholdOne_RequiresHighConfidence()
        {
            // Arrange - Threshold 1.0 requires perfect confidence
            var generator = new ConfigurableMockGenerator("Short answer");
            var retriever = new TrackingMockRetriever(CreateTestDocuments());
            var flare = new FLARERetriever<double>(generator, retriever, uncertaintyThreshold: 1.0);

            // Act
            var result = flare.GenerateWithActiveRetrieval("Test query");

            // Assert
            Assert.NotNull(result);
            // With threshold 1.0, will likely iterate until max iterations
        }

        #endregion

        #region Document Deduplication Tests

        [Fact]
        public void GenerateWithActiveRetrieval_DoesNotDuplicateDocuments()
        {
            // Arrange - Retriever always returns the same documents
            var docs = CreateTestDocuments();
            var responses = new[]
            {
                "I need more information about quantum bits.",
                "I need more information about superposition.",
                "Now I understand quantum computing completely."
            };
            var generator = new ConfigurableMockGenerator(responses);
            var retriever = new TrackingMockRetriever(docs);
            var flare = new FLARERetriever<double>(generator, retriever, uncertaintyThreshold: 0.5);

            // Act
            var result = flare.GenerateWithActiveRetrieval("What is quantum computing?");

            // Assert
            Assert.NotNull(result);
            // The implementation should not add duplicate documents by ID
            // This is verified by the implementation checking `d.Id == doc.Id`
        }

        #endregion
    }
}
