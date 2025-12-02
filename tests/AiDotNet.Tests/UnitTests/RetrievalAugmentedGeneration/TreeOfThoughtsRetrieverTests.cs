using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.RetrievalAugmentedGeneration.AdvancedPatterns;
using AiDotNet.RetrievalAugmentedGeneration.Generators;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using AiDotNet.RetrievalAugmentedGeneration.Retrievers;
using Xunit;

namespace AiDotNetTests.UnitTests.RetrievalAugmentedGeneration
{
    public class TreeOfThoughtsRetrieverTests
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
                    Content = "Quantum computers use qubits to perform calculations.",
                    RelevanceScore = 0.9,
                    HasRelevanceScore = true
                },
                new Document<double>
                {
                    Id = "doc2",
                    Content = "Quantum computing applications include cryptography and optimization.",
                    RelevanceScore = 0.85,
                    HasRelevanceScore = true
                },
                new Document<double>
                {
                    Id = "doc3",
                    Content = "Quantum supremacy was demonstrated by Google in 2019.",
                    RelevanceScore = 0.8,
                    HasRelevanceScore = true
                },
                new Document<double>
                {
                    Id = "doc4",
                    Content = "Quantum algorithms like Shor's algorithm can factor large numbers.",
                    RelevanceScore = 0.75,
                    HasRelevanceScore = true
                },
                new Document<double>
                {
                    Id = "doc5",
                    Content = "Quantum error correction is essential for reliable quantum computing.",
                    RelevanceScore = 0.7,
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
                new TreeOfThoughtsRetriever<double>(null!, mockRetriever));
        }

        [Fact]
        public void Constructor_WithNullRetriever_ThrowsArgumentNullException()
        {
            // Arrange
            var generator = new StubGenerator<double>();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new TreeOfThoughtsRetriever<double>(generator, null!));
        }

        [Fact]
        public void Constructor_WithValidArguments_InitializesCorrectly()
        {
            // Arrange
            var generator = new StubGenerator<double>();
            var mockRetriever = CreateMockRetriever();

            // Act
            var totRetriever = new TreeOfThoughtsRetriever<double>(
                generator,
                mockRetriever,
                maxDepth: 3,
                branchingFactor: 3);

            // Assert
            Assert.NotNull(totRetriever);
        }

        [Fact]
        public void Constructor_WithInvalidMaxDepth_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            var generator = new StubGenerator<double>();
            var mockRetriever = CreateMockRetriever();

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new TreeOfThoughtsRetriever<double>(generator, mockRetriever, maxDepth: 0));

            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new TreeOfThoughtsRetriever<double>(generator, mockRetriever, maxDepth: 15));
        }

        [Fact]
        public void Constructor_WithInvalidBranchingFactor_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            var generator = new StubGenerator<double>();
            var mockRetriever = CreateMockRetriever();

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new TreeOfThoughtsRetriever<double>(generator, mockRetriever, branchingFactor: 0));

            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new TreeOfThoughtsRetriever<double>(generator, mockRetriever, branchingFactor: 15));
        }

        [Fact]
        public void Retrieve_WithNullQuery_ThrowsArgumentException()
        {
            // Arrange
            var generator = new StubGenerator<double>();
            var mockRetriever = CreateMockRetriever();
            var totRetriever = new TreeOfThoughtsRetriever<double>(generator, mockRetriever);

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                totRetriever.Retrieve(null!, 10));
        }

        [Fact]
        public void Retrieve_WithEmptyQuery_ThrowsArgumentException()
        {
            // Arrange
            var generator = new StubGenerator<double>();
            var mockRetriever = CreateMockRetriever();
            var totRetriever = new TreeOfThoughtsRetriever<double>(generator, mockRetriever);

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                totRetriever.Retrieve("   ", 10));
        }

        [Fact]
        public void Retrieve_WithNegativeTopK_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            var generator = new StubGenerator<double>();
            var mockRetriever = CreateMockRetriever();
            var totRetriever = new TreeOfThoughtsRetriever<double>(generator, mockRetriever);

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                totRetriever.Retrieve("test query", -1));
        }

        [Fact]
        public void Retrieve_WithBreadthFirstStrategy_ReturnsDocuments()
        {
            // Arrange
            var generator = new StubGenerator<double>();
            var mockRetriever = CreateMockRetriever();
            var totRetriever = new TreeOfThoughtsRetriever<double>(
                generator,
                mockRetriever,
                maxDepth: 2,
                branchingFactor: 2, searchStrategy: TreeSearchStrategy.BreadthFirst);

            // Act
            var results = totRetriever.Retrieve(
                "What are quantum computing applications?",
                topK: 5);

            // Assert
            Assert.NotNull(results);
            var resultList = results.ToList();
            Assert.NotEmpty(resultList);
            Assert.True(resultList.Count <= 5);
        }

        [Fact]
        public void Retrieve_WithDepthFirstStrategy_ReturnsDocuments()
        {
            // Arrange
            var generator = new StubGenerator<double>();
            var mockRetriever = CreateMockRetriever();
            var totRetriever = new TreeOfThoughtsRetriever<double>(
                generator,
                mockRetriever,
                maxDepth: 2,
                branchingFactor: 2, searchStrategy: TreeSearchStrategy.DepthFirst);

            // Act
            var results = totRetriever.Retrieve(
                "What are quantum computing applications?",
                topK: 5);

            // Assert
            Assert.NotNull(results);
            var resultList = results.ToList();
            Assert.NotEmpty(resultList);
            Assert.True(resultList.Count <= 5);
        }

        [Fact]
        public void Retrieve_WithBestFirstStrategy_ReturnsDocuments()
        {
            // Arrange
            var generator = new StubGenerator<double>();
            var mockRetriever = CreateMockRetriever();
            var totRetriever = new TreeOfThoughtsRetriever<double>(
                generator,
                mockRetriever,
                maxDepth: 2,
                branchingFactor: 2, searchStrategy: TreeSearchStrategy.BestFirst);

            // Act
            var results = totRetriever.Retrieve(
                "What are quantum computing applications?",
                topK: 5);

            // Assert
            Assert.NotNull(results);
            var resultList = results.ToList();
            Assert.NotEmpty(resultList);
            Assert.True(resultList.Count <= 5);
        }

        [Fact]
        public void Retrieve_WithMetadataFilters_ReturnsDocuments()
        {
            // Arrange
            var generator = new StubGenerator<double>();
            var mockRetriever = CreateMockRetriever();
            var totRetriever = new TreeOfThoughtsRetriever<double>(generator, mockRetriever, searchStrategy: TreeSearchStrategy.BestFirst);
            var filters = new Dictionary<string, object> { { "domain", "quantum" } };

            // Act
            var results = totRetriever.Retrieve(
                "Quantum computing applications",
                topK: 5, metadataFilters: filters);

            // Assert
            Assert.NotNull(results);
            var resultList = results.ToList();
            Assert.NotEmpty(resultList);
        }

        [Fact]
        public void Retrieve_WithDifferentDepths_ReturnsAppropriateResults()
        {
            // Arrange
            var generator = new StubGenerator<double>();
            var mockRetriever = CreateMockRetriever();

            var shallowRetriever = new TreeOfThoughtsRetriever<double>(
                generator, mockRetriever, maxDepth: 1, branchingFactor: 2);

            var deepRetriever = new TreeOfThoughtsRetriever<double>(
                generator, mockRetriever, maxDepth: 3, branchingFactor: 2);

            // Act
            var shallowResults = shallowRetriever.Retrieve("test query", 10);
            var deepResults = deepRetriever.Retrieve("test query", 10);

            // Assert
            Assert.NotNull(shallowResults);
            Assert.NotNull(deepResults);
            // Deep retriever may explore more paths and find more diverse documents
        }
    }
}
