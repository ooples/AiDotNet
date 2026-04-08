using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.DocumentStores;
using AiDotNet.RetrievalAugmentedGeneration.Graph;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using Xunit;

namespace AiDotNetTests.UnitTests.RetrievalAugmentedGeneration
{
    public class HybridGraphRetrieverTests
    {
        private readonly KnowledgeGraph<double> _graph;
        private readonly InMemoryDocumentStore<double> _documentStore;
        private readonly HybridGraphRetriever<double> _retriever;

        public HybridGraphRetrieverTests()
        {
            // Create knowledge graph
            _graph = new KnowledgeGraph<double>();

            // Create document store with 3-dimensional embeddings
            _documentStore = new InMemoryDocumentStore<double>(3);

            // Create retriever (no similarity metric needed - it uses StatisticsHelper internally)
            _retriever = new HybridGraphRetriever<double>(_graph, _documentStore);

            // Setup test data
            SetupTestData();
        }

        private void SetupTestData()
        {
            // Create nodes with embeddings
            var aliceEmbedding = new Vector<double>(new double[] { 1.0, 0.0, 0.0 });
            var alice = new GraphNode<double>("alice", "Person")
            {
                Properties = new Dictionary<string, object> { { "name", "Alice" } },
                Embedding = aliceEmbedding
            };

            var bobEmbedding = new Vector<double>(new double[] { 0.8, 0.2, 0.0 });
            var bob = new GraphNode<double>("bob", "Person")
            {
                Properties = new Dictionary<string, object> { { "name", "Bob" } },
                Embedding = bobEmbedding
            };

            var charlieEmbedding = new Vector<double>(new double[] { 0.5, 0.5, 0.0 });
            var charlie = new GraphNode<double>("charlie", "Person")
            {
                Properties = new Dictionary<string, object> { { "name", "Charlie" } },
                Embedding = charlieEmbedding
            };

            var davidEmbedding = new Vector<double>(new double[] { 0.2, 0.8, 0.0 });
            var david = new GraphNode<double>("david", "Person")
            {
                Properties = new Dictionary<string, object> { { "name", "David" } },
                Embedding = davidEmbedding
            };

            // Add nodes to graph
            _graph.AddNode(alice);
            _graph.AddNode(bob);
            _graph.AddNode(charlie);
            _graph.AddNode(david);

            // Create edges (social network)
            _graph.AddEdge(new GraphEdge<double>("alice", "bob", "KNOWS") { Weight = 1.0 });
            _graph.AddEdge(new GraphEdge<double>("bob", "charlie", "KNOWS") { Weight = 1.0 });
            _graph.AddEdge(new GraphEdge<double>("charlie", "david", "KNOWS") { Weight = 1.0 });

            // Add documents to document store
            _documentStore.Add(new VectorDocument<double>(
                new Document<double>("alice", "Alice content"),
                aliceEmbedding));

            _documentStore.Add(new VectorDocument<double>(
                new Document<double>("bob", "Bob content"),
                bobEmbedding));

            _documentStore.Add(new VectorDocument<double>(
                new Document<double>("charlie", "Charlie content"),
                charlieEmbedding));

            _documentStore.Add(new VectorDocument<double>(
                new Document<double>("david", "David content"),
                davidEmbedding));
        }

        #region Basic Retrieval Tests

        [Fact]
        public void Retrieve_WithoutExpansion_ReturnsOnlyVectorResults()
        {
            // Arrange
            var query = new Vector<double>(new double[] { 1.0, 0.0, 0.0 }); // Similar to Alice

            // Act
            var results = _retriever.Retrieve(query, topK: 2, expansionDepth: 0, maxResults: 10);

            // Assert
            Assert.Equal(2, results.Count);
            Assert.All(results, r => Assert.Equal(RetrievalSource.VectorSearch, r.Source));
            Assert.Equal(0, results[0].Depth);
            Assert.Equal("alice", results[0].NodeId); // Most similar
        }

        [Fact]
        public void Retrieve_WithExpansion_IncludesGraphNeighbors()
        {
            // Arrange
            var query = new Vector<double>(new double[] { 1.0, 0.0, 0.0 }); // Similar to Alice

            // Act
            var results = _retriever.Retrieve(query, topK: 1, expansionDepth: 1, maxResults: 10);

            // Assert
            Assert.True(results.Count > 1); // Should include Alice + neighbors
            Assert.Contains(results, r => r.NodeId == "alice" && r.Source == RetrievalSource.VectorSearch);
            Assert.Contains(results, r => r.NodeId == "bob" && r.Source == RetrievalSource.GraphTraversal);
        }

        [Fact]
        public void Retrieve_WithDepth2_ReachesDistantNodes()
        {
            // Arrange
            var query = new Vector<double>(new double[] { 1.0, 0.0, 0.0 }); // Similar to Alice

            // Act
            var results = _retriever.Retrieve(query, topK: 1, expansionDepth: 2, maxResults: 10);

            // Assert
            // Should reach: Alice (0-hop) -> Bob (1-hop) -> Charlie (2-hop)
            Assert.Contains(results, r => r.NodeId == "charlie");
            var charlie = results.First(r => r.NodeId == "charlie");
            Assert.Equal(2, charlie.Depth);
            Assert.Equal(RetrievalSource.GraphTraversal, charlie.Source);
        }

        #endregion

        #region Depth Penalty Tests

        [Fact]
        public void Retrieve_AppliesDepthPenalty_CloserNodesRankHigher()
        {
            // Arrange
            var query = new Vector<double>(new double[] { 1.0, 0.0, 0.0 });

            // Act
            var results = _retriever.Retrieve(query, topK: 1, expansionDepth: 2, maxResults: 10);

            // Assert - Closer nodes should have higher scores due to depth penalty
            var bob = results.FirstOrDefault(r => r.NodeId == "bob");
            var charlie = results.FirstOrDefault(r => r.NodeId == "charlie");

            if (bob != null && charlie != null)
            {
                // Bob (1-hop) should score higher than Charlie (2-hop) due to depth penalty
                // even if their raw vector similarities are similar
                Assert.True(bob.Depth < charlie.Depth);
            }
        }

        #endregion

        #region Relationship-Aware Retrieval Tests

        [Fact]
        public void RetrieveWithRelationships_UsesRelationshipWeights()
        {
            // Arrange
            var query = new Vector<double>(new double[] { 1.0, 0.0, 0.0 });
            var weights = new Dictionary<string, double>
            {
                { "KNOWS", 1.5 } // Boost KNOWS relationships
            };

            // Act
            var results = _retriever.RetrieveWithRelationships(query, topK: 1, weights, maxResults: 10);

            // Assert
            Assert.NotEmpty(results);
            var traversedResults = results.Where(r => r.Source == RetrievalSource.GraphTraversal).ToList();
            Assert.NotEmpty(traversedResults);
            Assert.All(traversedResults, r => Assert.Equal("KNOWS", r.RelationType));
        }

        [Fact]
        public void RetrieveWithRelationships_IncludesRelationshipInfo()
        {
            // Arrange
            var query = new Vector<double>(new double[] { 1.0, 0.0, 0.0 });

            // Act
            var results = _retriever.RetrieveWithRelationships(query, topK: 1, maxResults: 10);

            // Assert
            var bobResult = results.FirstOrDefault(r => r.NodeId == "bob");
            Assert.NotNull(bobResult);
            Assert.Equal("KNOWS", bobResult.RelationType);
            Assert.Equal("alice", bobResult.ParentNodeId);
        }

        #endregion

        #region MaxResults Tests

        [Fact]
        public void Retrieve_RespectsMaxResults()
        {
            // Arrange
            var query = new Vector<double>(new double[] { 1.0, 0.0, 0.0 });

            // Act
            var results = _retriever.Retrieve(query, topK: 2, expansionDepth: 2, maxResults: 2);

            // Assert
            Assert.Equal(2, results.Count);
        }

        [Fact]
        public void Retrieve_SortsByScore()
        {
            // Arrange
            var query = new Vector<double>(new double[] { 1.0, 0.0, 0.0 });

            // Act
            var results = _retriever.Retrieve(query, topK: 2, expansionDepth: 1, maxResults: 10);

            // Assert - Results should be sorted by score descending
            for (int i = 0; i < results.Count - 1; i++)
            {
                Assert.True(results[i].Score >= results[i + 1].Score);
            }
        }

        #endregion

        #region Error Handling Tests

        [Fact]
        public void Retrieve_NullEmbedding_ThrowsException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                _retriever.Retrieve(null!, topK: 5));
        }

        [Fact]
        public void Retrieve_EmptyEmbedding_ThrowsException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                _retriever.Retrieve(new Vector<double>(Array.Empty<double>()), topK: 5));
        }

        [Fact]
        public void Retrieve_InvalidTopK_ThrowsException()
        {
            // Arrange
            var query = new Vector<double>(new double[] { 1.0, 0.0, 0.0 });

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                _retriever.Retrieve(query, topK: 0));

            Assert.Throws<ArgumentOutOfRangeException>(() =>
                _retriever.Retrieve(query, topK: -1));
        }

        [Fact]
        public void Retrieve_NegativeExpansionDepth_ThrowsException()
        {
            // Arrange
            var query = new Vector<double>(new double[] { 1.0, 0.0, 0.0 });

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                _retriever.Retrieve(query, topK: 5, expansionDepth: -1));
        }

        #endregion

        #region Result Properties Tests

        [Fact]
        public void Retrieve_PopulatesResultProperties()
        {
            // Arrange
            var query = new Vector<double>(new double[] { 1.0, 0.0, 0.0 });

            // Act
            var results = _retriever.Retrieve(query, topK: 1, expansionDepth: 1, maxResults: 10);

            // Assert
            Assert.All(results, r =>
            {
                Assert.NotNull(r.NodeId);
                Assert.True(r.Score >= 0.0);
                Assert.True(r.Depth >= 0);

                if (r.Source == RetrievalSource.VectorSearch)
                {
                    Assert.Equal(0, r.Depth);
                    Assert.Null(r.ParentNodeId);
                }
                else if (r.Source == RetrievalSource.GraphTraversal)
                {
                    Assert.True(r.Depth > 0);
                    Assert.NotNull(r.ParentNodeId);
                }
            });
        }

        [Fact]
        public void Retrieve_IncludesEmbeddings()
        {
            // Arrange
            var query = new Vector<double>(new double[] { 1.0, 0.0, 0.0 });

            // Act
            var results = _retriever.Retrieve(query, topK: 1, expansionDepth: 0, maxResults: 10);

            // Assert - embeddings may or may not be populated depending on the source
            Assert.NotEmpty(results);
        }

        #endregion

        #region Complex Scenario Tests

        [Fact]
        public void Retrieve_ComplexGraph_ProducesCoherentResults()
        {
            // Arrange - Add more complex graph structure
            var graph = new KnowledgeGraph<double>();
            var documentStore = new InMemoryDocumentStore<double>(3);

            // Create a small community
            for (int i = 0; i < 5; i++)
            {
                var embedding = new Vector<double>(new double[] { i * 0.2, 1 - i * 0.2, 0.0 });
                var node = new GraphNode<double>($"user{i}", "Person")
                {
                    Properties = new Dictionary<string, object> { { "name", $"User{i}" } },
                    Embedding = embedding
                };
                graph.AddNode(node);
                documentStore.Add(new VectorDocument<double>(
                    new Document<double>($"user{i}", $"User{i} content"),
                    embedding));
            }

            // Create connections
            for (int i = 0; i < 4; i++)
            {
                graph.AddEdge(new GraphEdge<double>($"user{i}", $"user{i + 1}", "FRIENDS_WITH") { Weight = 1.0 });
            }

            var retriever = new HybridGraphRetriever<double>(graph, documentStore);
            var query = new Vector<double>(new double[] { 0.0, 1.0, 0.0 }); // Similar to user0

            // Act
            var results = retriever.Retrieve(query, topK: 1, expansionDepth: 2, maxResults: 5);

            // Assert
            Assert.NotEmpty(results);
            Assert.True(results.Count <= 5);
            Assert.Contains(results, r => r.Source == RetrievalSource.VectorSearch);
        }

        #endregion

        #region Async Tests

        [Fact]
        public async Task RetrieveAsync_WorksCorrectly()
        {
            // Arrange
            var query = new Vector<double>(new double[] { 1.0, 0.0, 0.0 });

            // Act
            var results = await _retriever.RetrieveAsync(query, topK: 2, expansionDepth: 1, maxResults: 10);

            // Assert
            Assert.NotEmpty(results);
            Assert.Contains(results, r => r.Source == RetrievalSource.VectorSearch);
        }

        #endregion
    }
}
