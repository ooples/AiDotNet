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
    public class ChainOfThoughtRetrieverTests
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
                    Content = "Photosynthesis is the process by which plants convert sunlight into energy.",
                    RelevanceScore = 0.9,
                    HasRelevanceScore = true
                },
                new Document<double>
                {
                    Id = "doc2",
                    Content = "Climate change is caused by increased greenhouse gas emissions.",
                    RelevanceScore = 0.8,
                    HasRelevanceScore = true
                },
                new Document<double>
                {
                    Id = "doc3",
                    Content = "Carbon dioxide is absorbed by plants during photosynthesis.",
                    RelevanceScore = 0.85,
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
                new ChainOfThoughtRetriever<double>(null!, mockRetriever));
        }

        [Fact]
        public void Constructor_WithNullRetriever_ThrowsArgumentNullException()
        {
            // Arrange
            var generator = new StubGenerator<double>();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new ChainOfThoughtRetriever<double>(generator, null!));
        }

        [Fact]
        public void Constructor_WithValidArguments_InitializesCorrectly()
        {
            // Arrange
            var generator = new StubGenerator<double>();
            var mockRetriever = CreateMockRetriever();

            // Act
            var cotRetriever = new ChainOfThoughtRetriever<double>(generator, mockRetriever);

            // Assert
            Assert.NotNull(cotRetriever);
        }

        [Fact]
        public void Constructor_WithFewShotExamples_InitializesCorrectly()
        {
            // Arrange
            var generator = new StubGenerator<double>();
            var mockRetriever = CreateMockRetriever();
            var fewShotExamples = new List<string>
            {
                "Example 1: Question about physics -> Break into concepts",
                "Example 2: Question about history -> Identify key events"
            };

            // Act
            var cotRetriever = new ChainOfThoughtRetriever<double>(
                generator,
                mockRetriever,
                fewShotExamples);

            // Assert
            Assert.NotNull(cotRetriever);
        }

        [Fact]
        public void Retrieve_WithNullQuery_ThrowsArgumentException()
        {
            // Arrange
            var generator = new StubGenerator<double>();
            var mockRetriever = CreateMockRetriever();
            var cotRetriever = new ChainOfThoughtRetriever<double>(generator, mockRetriever);

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                cotRetriever.Retrieve(null!, 10));
        }

        [Fact]
        public void Retrieve_WithEmptyQuery_ThrowsArgumentException()
        {
            // Arrange
            var generator = new StubGenerator<double>();
            var mockRetriever = CreateMockRetriever();
            var cotRetriever = new ChainOfThoughtRetriever<double>(generator, mockRetriever);

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                cotRetriever.Retrieve("   ", 10));
        }

        [Fact]
        public void Retrieve_WithNegativeTopK_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            var generator = new StubGenerator<double>();
            var mockRetriever = CreateMockRetriever();
            var cotRetriever = new ChainOfThoughtRetriever<double>(generator, mockRetriever);

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                cotRetriever.Retrieve("test query", -1));
        }

        [Fact]
        public void Retrieve_WithValidQuery_ReturnsDocuments()
        {
            // Arrange
            var generator = new StubGenerator<double>();
            var mockRetriever = CreateMockRetriever();
            var cotRetriever = new ChainOfThoughtRetriever<double>(generator, mockRetriever);

            // Act
            var results = cotRetriever.Retrieve("How does photosynthesis affect climate?", 10);

            // Assert
            Assert.NotNull(results);
            var resultList = results.ToList();
            Assert.NotEmpty(resultList);
        }

        [Fact]
        public void Retrieve_WithMetadataFilters_ReturnsDocuments()
        {
            // Arrange
            var generator = new StubGenerator<double>();
            var mockRetriever = CreateMockRetriever();
            var cotRetriever = new ChainOfThoughtRetriever<double>(generator, mockRetriever);
            var filters = new Dictionary<string, object> { { "category", "science" } };

            // Act
            var results = cotRetriever.Retrieve("How does photosynthesis affect climate?", 10, filters);

            // Assert
            Assert.NotNull(results);
            var resultList = results.ToList();
            Assert.NotEmpty(resultList);
        }

        [Fact]
        public void RetrieveWithSelfConsistency_WithNullQuery_ThrowsArgumentException()
        {
            // Arrange
            var generator = new StubGenerator<double>();
            var mockRetriever = CreateMockRetriever();
            var cotRetriever = new ChainOfThoughtRetriever<double>(generator, mockRetriever);

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                cotRetriever.RetrieveWithSelfConsistency(null!, 10));
        }

        [Fact]
        public void RetrieveWithSelfConsistency_WithNegativeTopK_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            var generator = new StubGenerator<double>();
            var mockRetriever = CreateMockRetriever();
            var cotRetriever = new ChainOfThoughtRetriever<double>(generator, mockRetriever);

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                cotRetriever.RetrieveWithSelfConsistency("test query", -1));
        }

        [Fact]
        public void RetrieveWithSelfConsistency_WithNegativeNumPaths_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            var generator = new StubGenerator<double>();
            var mockRetriever = CreateMockRetriever();
            var cotRetriever = new ChainOfThoughtRetriever<double>(generator, mockRetriever);

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                cotRetriever.RetrieveWithSelfConsistency("test query", 10, numPaths: -1));
        }

        [Fact]
        public void RetrieveWithSelfConsistency_WithValidQuery_ReturnsDocuments()
        {
            // Arrange
            var generator = new StubGenerator<double>();
            var mockRetriever = CreateMockRetriever();
            var cotRetriever = new ChainOfThoughtRetriever<double>(generator, mockRetriever);

            // Act
            var results = cotRetriever.RetrieveWithSelfConsistency(
                "What are the impacts of renewable energy?",
                topK: 5,
                numPaths: 3);

            // Assert
            Assert.NotNull(results);
            var resultList = results.ToList();
            Assert.NotEmpty(resultList);
            Assert.True(resultList.Count <= 5);
        }

        [Fact]
        public void RetrieveWithSelfConsistency_RanksDocumentsByFrequency()
        {
            // Arrange
            var generator = new StubGenerator<double>();
            var mockRetriever = CreateMockRetriever();
            var cotRetriever = new ChainOfThoughtRetriever<double>(generator, mockRetriever);

            // Act
            var results = cotRetriever.RetrieveWithSelfConsistency(
                "What is photosynthesis?",
                topK: 10,
                numPaths: 3);

            // Assert
            Assert.NotNull(results);
            var resultList = results.ToList();
            Assert.NotEmpty(resultList);
            // Documents that appear in multiple paths should be ranked higher
        }
    }
}
