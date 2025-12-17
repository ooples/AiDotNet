// Nullable disabled: This test file intentionally passes null values to test argument validation
#nullable disable

using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using AiDotNet.RetrievalAugmentedGeneration.Retrievers;
using Xunit;

namespace AiDotNetTests.UnitTests.RetrievalAugmentedGeneration.Retrievers
{
    /// <summary>
    /// Unit tests for advanced retriever classes (MultiVector, ParentDocument, ColBERT, Graph)
    /// </summary>

    #region MultiVectorRetriever Tests

    public class MultiVectorRetrieverTests
    {
        private readonly IDocumentStore<double> _documentStore;
        private const int VectorDimension = 128;

        public MultiVectorRetrieverTests()
        {
            _documentStore = TestHelpers.CreateDocumentStore<double>(VectorDimension);
        }

        [Fact]
        public void Constructor_WithNullDocumentStore_ThrowsArgumentNullException()
        {
            // Arrange & Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new MultiVectorRetriever<double>(null, 3, "max"));
        }

        [Fact]
        public void Constructor_WithZeroVectorsPerDocument_ThrowsArgumentOutOfRangeException()
        {
            // Arrange & Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new MultiVectorRetriever<double>(_documentStore, 0, "max"));
        }

        [Fact]
        public void Constructor_WithNullAggregationMethod_ThrowsArgumentNullException()
        {
            // Arrange & Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new MultiVectorRetriever<double>(_documentStore, 3, null));
        }

        [Fact]
        public void Constructor_WithValidParameters_CreatesInstance()
        {
            // Arrange & Act
            var retriever = new MultiVectorRetriever<double>(_documentStore, 3, "max");

            // Assert
            Assert.NotNull(retriever);
        }

        [Fact]
        public void Constructor_WithDifferentAggregationMethods_CreatesInstance()
        {
            // Arrange & Act
            var maxRetriever = new MultiVectorRetriever<double>(_documentStore, 3, "max");
            var meanRetriever = new MultiVectorRetriever<double>(_documentStore, 3, "mean");
            var weightedRetriever = new MultiVectorRetriever<double>(_documentStore, 3, "weighted");

            // Assert
            Assert.NotNull(maxRetriever);
            Assert.NotNull(meanRetriever);
            Assert.NotNull(weightedRetriever);
        }

        [Fact]
        public void Retrieve_WithNullQuery_ThrowsArgumentException()
        {
            // Arrange
            var retriever = new MultiVectorRetriever<double>(_documentStore, 3, "max");

            // Act & Assert
            Assert.Throws<ArgumentException>(() => retriever.Retrieve(null));
        }

        [Fact]
        public void Retrieve_WithEmptyQuery_ThrowsArgumentException()
        {
            // Arrange
            var retriever = new MultiVectorRetriever<double>(_documentStore, 3, "max");

            // Act & Assert
            Assert.Throws<ArgumentException>(() => retriever.Retrieve(""));
        }

        [Fact]
        public void Retrieve_WithZeroTopK_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            var retriever = new MultiVectorRetriever<double>(_documentStore, 3, "max");

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() => retriever.Retrieve("test", 0));
        }
    }

    #endregion

    #region ParentDocumentRetriever Tests

    public class ParentDocumentRetrieverTests
    {
        private readonly IDocumentStore<double> _documentStore;
        private readonly IEmbeddingModel<double> _embeddingModel;
        private const int VectorDimension = 128;

        public ParentDocumentRetrieverTests()
        {
            _documentStore = TestHelpers.CreateDocumentStore<double>(VectorDimension);
            _embeddingModel = TestHelpers.CreateEmbeddingModel<double>(VectorDimension);
        }

        [Fact]
        public void Constructor_WithNullDocumentStore_ThrowsArgumentNullException()
        {
            // Arrange & Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new ParentDocumentRetriever<double>(null, _embeddingModel, 256, 2048, true));
        }

        [Fact]
        public void Constructor_WithNullEmbeddingModel_ThrowsArgumentNullException()
        {
            // Arrange & Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new ParentDocumentRetriever<double>(_documentStore, null, 256, 2048, true));
        }

        [Fact]
        public void Constructor_WithZeroChunkSize_ThrowsArgumentOutOfRangeException()
        {
            // Arrange & Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new ParentDocumentRetriever<double>(_documentStore, _embeddingModel, 0, 2048, true));
        }

        [Fact]
        public void Constructor_WithZeroParentSize_ThrowsArgumentOutOfRangeException()
        {
            // Arrange & Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new ParentDocumentRetriever<double>(_documentStore, _embeddingModel, 256, 0, true));
        }

        [Fact]
        public void Constructor_WithParentSizeLessThanChunkSize_ThrowsArgumentException()
        {
            // Arrange & Act & Assert
            Assert.Throws<ArgumentException>(() =>
                new ParentDocumentRetriever<double>(_documentStore, _embeddingModel, 2048, 256, true));
        }

        [Fact]
        public void Constructor_WithValidParameters_CreatesInstance()
        {
            // Arrange & Act
            var retriever = new ParentDocumentRetriever<double>(_documentStore, _embeddingModel, 256, 2048, true);

            // Assert
            Assert.NotNull(retriever);
        }

        [Fact]
        public void Constructor_WithIncludeNeighboringChunks_CreatesInstance()
        {
            // Arrange & Act
            var retrieverWith = new ParentDocumentRetriever<double>(_documentStore, _embeddingModel, 256, 2048, true);
            var retrieverWithout = new ParentDocumentRetriever<double>(_documentStore, _embeddingModel, 256, 2048, false);

            // Assert
            Assert.NotNull(retrieverWith);
            Assert.NotNull(retrieverWithout);
        }

        [Fact]
        public void Retrieve_WithNullQuery_ThrowsArgumentException()
        {
            // Arrange
            var retriever = new ParentDocumentRetriever<double>(_documentStore, _embeddingModel, 256, 2048, true);

            // Act & Assert
            Assert.Throws<ArgumentException>(() => retriever.Retrieve(null));
        }

        [Fact]
        public void Retrieve_WithEmptyQuery_ThrowsArgumentException()
        {
            // Arrange
            var retriever = new ParentDocumentRetriever<double>(_documentStore, _embeddingModel, 256, 2048, true);

            // Act & Assert
            Assert.Throws<ArgumentException>(() => retriever.Retrieve(""));
        }

        [Fact]
        public void Retrieve_WithZeroTopK_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            var retriever = new ParentDocumentRetriever<double>(_documentStore, _embeddingModel, 256, 2048, true);

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() => retriever.Retrieve("test", 0));
        }

        [Fact]
        public void Retrieve_WithEmptyDocumentStore_ReturnsEmptyResults()
        {
            // Arrange
            var retriever = new ParentDocumentRetriever<double>(_documentStore, _embeddingModel, 256, 2048, true);

            // Act
            var results = retriever.Retrieve("test query");

            // Assert
            Assert.Empty(results);
        }

        [Fact]
        public void Retrieve_WithValidQuery_WorksCorrectly()
        {
            // Arrange
            AddChunkedDocuments();
            var retriever = new ParentDocumentRetriever<double>(_documentStore, _embeddingModel, 256, 2048, false);

            // Act
            var results = retriever.Retrieve("machine learning", topK: 3).ToList();

            // Assert
            // Should work even with limited chunks
            Assert.NotNull(results);
            Assert.NotEmpty(results);
        }

        private void AddChunkedDocuments()
        {
            var chunks = new List<Document<double>>
            {
                new Document<double>("chunk1", "machine learning algorithms")
                {
                    Metadata = new Dictionary<string, object>
                    {
                        ["parent_id"] = "parent1",
                        ["chunk_index"] = 0
                    }
                },
                new Document<double>("chunk2", "deep neural networks")
                {
                    Metadata = new Dictionary<string, object>
                    {
                        ["parent_id"] = "parent1",
                        ["chunk_index"] = 1
                    }
                }
            };

            var vectorDocs = chunks.Select(doc =>
            {
                var embedding = _embeddingModel.Embed(doc.Content);
                return new VectorDocument<double>
                {
                    Document = doc,
                    Embedding = embedding
                };
            });

            _documentStore.AddBatch(vectorDocs);

            // Also add parent document
            var parentDoc = new Document<double>("parent1", "machine learning algorithms and deep neural networks combined content")
            {
                Metadata = new Dictionary<string, object>()
            };
            var parent = new VectorDocument<double>
            {
                Document = parentDoc,
                Embedding = _embeddingModel.Embed("machine learning algorithms and deep neural networks combined content")
            };
            _documentStore.Add(parent);
        }
    }

    #endregion

    #region ColBERTRetriever Tests

    public class ColBERTRetrieverTests
    {
        private readonly IDocumentStore<double> _documentStore;
        private const int VectorDimension = 128;

        public ColBERTRetrieverTests()
        {
            _documentStore = TestHelpers.CreateDocumentStore<double>(VectorDimension);
        }

        [Fact]
        public void Constructor_WithNullDocumentStore_ThrowsArgumentNullException()
        {
            // Arrange & Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new ColBERTRetriever<double>(null, "model.onnx", 512, 32));
        }

        [Fact]
        public void Constructor_WithNullModelPath_ThrowsArgumentNullException()
        {
            // Arrange & Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new ColBERTRetriever<double>(_documentStore, null, 512, 32));
        }

        [Fact]
        public void Constructor_WithZeroMaxDocLength_ThrowsArgumentOutOfRangeException()
        {
            // Arrange & Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new ColBERTRetriever<double>(_documentStore, "model.onnx", 0, 32));
        }

        [Fact]
        public void Constructor_WithZeroMaxQueryLength_ThrowsArgumentOutOfRangeException()
        {
            // Arrange & Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new ColBERTRetriever<double>(_documentStore, "model.onnx", 512, 0));
        }

        [Fact]
        public void Constructor_WithValidParameters_CreatesInstance()
        {
            // Arrange & Act
            var retriever = new ColBERTRetriever<double>(_documentStore, "model.onnx", 512, 32);

            // Assert
            Assert.NotNull(retriever);
        }

        [Fact]
        public void Retrieve_WithNullQuery_ThrowsArgumentException()
        {
            // Arrange
            var retriever = new ColBERTRetriever<double>(_documentStore, "model.onnx", 512, 32);

            // Act & Assert
            Assert.Throws<ArgumentException>(() => retriever.Retrieve(null));
        }

        [Fact]
        public void Retrieve_WithEmptyQuery_ThrowsArgumentException()
        {
            // Arrange
            var retriever = new ColBERTRetriever<double>(_documentStore, "model.onnx", 512, 32);

            // Act & Assert
            Assert.Throws<ArgumentException>(() => retriever.Retrieve(""));
        }

        [Fact]
        public void Retrieve_WithZeroTopK_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            var retriever = new ColBERTRetriever<double>(_documentStore, "model.onnx", 512, 32);

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() => retriever.Retrieve("test", 0));
        }

        [Fact]
        public void Retrieve_WithEmptyDocumentStore_ReturnsEmptyResults()
        {
            // Arrange
            var retriever = new ColBERTRetriever<double>(_documentStore, "model.onnx", 512, 32);

            // Act
            var results = retriever.Retrieve("test query");

            // Assert
            Assert.Empty(results);
        }
    }

    #endregion

    #region GraphRetriever Tests

    public class GraphRetrieverTests
    {
        private readonly IDocumentStore<double> _documentStore;
        private readonly IEmbeddingModel<double> _embeddingModel;
        private const int VectorDimension = 128;

        public GraphRetrieverTests()
        {
            _documentStore = TestHelpers.CreateDocumentStore<double>(VectorDimension);
            _embeddingModel = TestHelpers.CreateEmbeddingModel<double>(VectorDimension);
        }

        [Fact]
        public void Constructor_WithNullDocumentStore_ThrowsArgumentNullException()
        {
            // Arrange & Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new GraphRetriever<double>(null, _embeddingModel));
        }

        [Fact]
        public void Constructor_WithNullEmbeddingModel_ThrowsArgumentNullException()
        {
            // Arrange & Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new GraphRetriever<double>(_documentStore, null));
        }

        [Fact]
        public void Constructor_WithValidParameters_CreatesInstance()
        {
            // Arrange & Act
            var retriever = new GraphRetriever<double>(_documentStore, _embeddingModel);

            // Assert
            Assert.NotNull(retriever);
        }

        [Fact]
        public void Constructor_WithEntityExtractionEnabled_CreatesInstance()
        {
            // Arrange & Act
            var retrieverEnabled = new GraphRetriever<double>(_documentStore, _embeddingModel, enableAdvancedEntityExtraction: true);
            var retrieverDisabled = new GraphRetriever<double>(_documentStore, _embeddingModel, enableAdvancedEntityExtraction: false);

            // Assert
            Assert.NotNull(retrieverEnabled);
            Assert.NotNull(retrieverDisabled);
        }

        [Fact]
        public void Retrieve_WithNullQuery_ThrowsArgumentException()
        {
            // Arrange
            var retriever = new GraphRetriever<double>(_documentStore, _embeddingModel);

            // Act & Assert
            Assert.Throws<ArgumentException>(() => retriever.Retrieve(null));
        }

        [Fact]
        public void Retrieve_WithEmptyQuery_ThrowsArgumentException()
        {
            // Arrange
            var retriever = new GraphRetriever<double>(_documentStore, _embeddingModel);

            // Act & Assert
            Assert.Throws<ArgumentException>(() => retriever.Retrieve(""));
        }

        [Fact]
        public void Retrieve_WithZeroTopK_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            var retriever = new GraphRetriever<double>(_documentStore, _embeddingModel);

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() => retriever.Retrieve("test", 0));
        }

        [Fact]
        public void Retrieve_WithEmptyDocumentStore_ReturnsEmptyResults()
        {
            // Arrange
            var retriever = new GraphRetriever<double>(_documentStore, _embeddingModel);

            // Act
            var results = retriever.Retrieve("test query");

            // Assert
            Assert.Empty(results);
        }

        [Fact]
        public void Retrieve_WithValidQuery_ReturnsDocuments()
        {
            // Arrange
            AddSampleDocuments();
            var retriever = new GraphRetriever<double>(_documentStore, _embeddingModel);

            // Act
            var results = retriever.Retrieve("machine learning", topK: 3).ToList();

            // Assert
            Assert.NotEmpty(results);
            Assert.All(results, doc => Assert.True(doc.HasRelevanceScore));
        }

        [Fact]
        public void Retrieve_WithEntityQuery_WorksCorrectly()
        {
            // Arrange
            AddSampleDocuments();
            var retriever = new GraphRetriever<double>(_documentStore, _embeddingModel, enableAdvancedEntityExtraction: true);

            // Act - Query with entities (capitalized words)
            var results = retriever.Retrieve("Albert Einstein physics", topK: 5).ToList();

            // Assert
            Assert.NotNull(results);
            // Results may be empty if query entities don't match document content,
            // but the retrieval should complete without errors
        }

        [Fact]
        public void Retrieve_WithMetadataFilter_ReturnsOnlyMatchingDocuments()
        {
            // Arrange
            AddSampleDocuments();
            var retriever = new GraphRetriever<double>(_documentStore, _embeddingModel);
            var filters = new Dictionary<string, object> { ["category"] = "AI" };

            // Act
            var results = retriever.Retrieve("machine", 10, filters).ToList();

            // Assert
            Assert.All(results, doc =>
            {
                Assert.True(doc.Metadata.TryGetValue("category", out var category));
                Assert.Equal("AI", category);
            });
        }

        [Fact]
        public void Retrieve_WithQuotedEntities_WorksCorrectly()
        {
            // Arrange
            AddSampleDocuments();
            var retriever = new GraphRetriever<double>(_documentStore, _embeddingModel);

            // Act - Query with quoted entities
            var results = retriever.Retrieve("\"machine learning\" algorithms", topK: 5).ToList();

            // Assert
            Assert.NotEmpty(results);
        }

        private void AddSampleDocuments()
        {
            var documents = TestHelpers.CreateSampleDocuments<double>();
            var vectorDocuments = documents.Select(doc =>
            {
                var embedding = _embeddingModel.Embed(doc.Content);
                return new VectorDocument<double>
                {
                    Document = doc,
                    Embedding = embedding
                };
            });

            _documentStore.AddBatch(vectorDocuments);
        }
    }

    #endregion
}
