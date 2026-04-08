using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.RetrievalAugmentedGeneration.Graph;
using Xunit;

namespace AiDotNetTests.UnitTests.RetrievalAugmentedGeneration
{
    public class FileGraphStoreTests : IDisposable
    {
        private readonly string _testDirectory;

        public FileGraphStoreTests()
        {
            _testDirectory = Path.Combine(Path.GetTempPath(), "filegraph_tests_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(_testDirectory);
        }

        public void Dispose()
        {
            if (Directory.Exists(_testDirectory))
                Directory.Delete(_testDirectory, true);
        }

        private string GetTestStoragePath()
        {
            return Path.Combine(_testDirectory, Guid.NewGuid().ToString("N"));
        }

        private GraphNode<double> CreateTestNode(string id, string label, Dictionary<string, object>? properties = null)
        {
            var node = new GraphNode<double>(id, label);
            if (properties != null)
            {
                foreach (var kvp in properties)
                {
                    node.SetProperty(kvp.Key, kvp.Value);
                }
            }
            return node;
        }

        private GraphEdge<double> CreateTestEdge(string sourceId, string relationType, string targetId, double weight = 1.0)
        {
            return new GraphEdge<double>(sourceId, targetId, relationType, weight);
        }

        #region Constructor Tests

        [Fact]
        public void Constructor_WithValidPath_CreatesStoreAndDirectory()
        {
            // Arrange
            var storagePath = GetTestStoragePath();

            // Act
            using var store = new FileGraphStore<double>(storagePath);

            // Assert
            Assert.True(Directory.Exists(storagePath));
            Assert.Equal(0, store.NodeCount);
            Assert.Equal(0, store.EdgeCount);
        }

        [Fact]
        public void Constructor_WithNullPath_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() => new FileGraphStore<double>(null!));
        }

        [Fact]
        public void Constructor_WithEmptyPath_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() => new FileGraphStore<double>(""));
        }

        [Fact]
        public void Constructor_CreatesRequiredFiles()
        {
            // Arrange
            var storagePath = GetTestStoragePath();

            // Act - use explicit using block to ensure dispose before assertions
            using (var store = new FileGraphStore<double>(storagePath))
            {
                var node = CreateTestNode("node1", "PERSON");
                store.AddNode(node);
            } // Dispose called here - flushes to disk

            // Assert
            Assert.True(File.Exists(Path.Combine(storagePath, "node_index.db")));
            Assert.True(File.Exists(Path.Combine(storagePath, "nodes.dat")));
        }

        #endregion

        #region AddNode Tests

        [Fact]
        public void AddNode_WithValidNode_IncreasesNodeCount()
        {
            // Arrange
            var storagePath = GetTestStoragePath();
            using var store = new FileGraphStore<double>(storagePath);
            var node = CreateTestNode("node1", "PERSON");

            // Act
            store.AddNode(node);

            // Assert
            Assert.Equal(1, store.NodeCount);
        }

        [Fact]
        public void AddNode_WithNullNode_ThrowsArgumentNullException()
        {
            // Arrange
            var storagePath = GetTestStoragePath();
            using var store = new FileGraphStore<double>(storagePath);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => store.AddNode(null!));
        }

        [Fact]
        public void AddNode_WithProperties_PersistsProperties()
        {
            // Arrange
            var storagePath = GetTestStoragePath();
            using var store = new FileGraphStore<double>(storagePath);
            var properties = new Dictionary<string, object>
            {
                { "name", "Alice" },
                { "age", 30 },
                { "active", true }
            };
            var node = CreateTestNode("node1", "PERSON", properties);

            // Act
            store.AddNode(node);
            var retrieved = store.GetNode("node1");

            // Assert
            Assert.NotNull(retrieved);
            Assert.Equal("Alice", retrieved.GetProperty<string>("name"));
            Assert.Equal(30, retrieved.GetProperty<int>("age"));
            Assert.True(retrieved.GetProperty<bool>("active"));
        }

        #endregion

        #region AddEdge Tests

        [Fact]
        public void AddEdge_WithValidEdge_IncreasesEdgeCount()
        {
            // Arrange
            var storagePath = GetTestStoragePath();
            using var store = new FileGraphStore<double>(storagePath);
            var node1 = CreateTestNode("node1", "PERSON");
            var node2 = CreateTestNode("node2", "COMPANY");
            store.AddNode(node1);
            store.AddNode(node2);
            var edge = CreateTestEdge("node1", "WORKS_AT", "node2");

            // Act
            store.AddEdge(edge);

            // Assert
            Assert.Equal(1, store.EdgeCount);
        }

        [Fact]
        public void AddEdge_WithNullEdge_ThrowsArgumentNullException()
        {
            // Arrange
            var storagePath = GetTestStoragePath();
            using var store = new FileGraphStore<double>(storagePath);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => store.AddEdge(null!));
        }

        [Fact]
        public void AddEdge_WithNonexistentSourceNode_ThrowsInvalidOperationException()
        {
            // Arrange
            var storagePath = GetTestStoragePath();
            using var store = new FileGraphStore<double>(storagePath);
            var node = CreateTestNode("node1", "PERSON");
            store.AddNode(node);
            var edge = CreateTestEdge("nonexistent", "KNOWS", "node1");

            // Act & Assert
            var exception = Assert.Throws<InvalidOperationException>(() => store.AddEdge(edge));
            Assert.Contains("Source node 'nonexistent' does not exist", exception.Message);
        }

        #endregion

        #region GetNode Tests

        [Fact]
        public void GetNode_WithExistingId_ReturnsNode()
        {
            // Arrange
            var storagePath = GetTestStoragePath();
            using var store = new FileGraphStore<double>(storagePath);
            var node = CreateTestNode("node1", "PERSON");
            store.AddNode(node);

            // Act
            var retrieved = store.GetNode("node1");

            // Assert
            Assert.NotNull(retrieved);
            Assert.Equal("node1", retrieved.Id);
            Assert.Equal("PERSON", retrieved.Label);
        }

        [Fact]
        public void GetNode_WithNonexistentId_ReturnsNull()
        {
            // Arrange
            var storagePath = GetTestStoragePath();
            using var store = new FileGraphStore<double>(storagePath);

            // Act
            var retrieved = store.GetNode("nonexistent");

            // Assert
            Assert.Null(retrieved);
        }

        #endregion

        #region Persistence Tests

        [Fact]
        public void Persistence_NodesAndEdges_SurviveRestart()
        {
            // Arrange
            var storagePath = GetTestStoragePath();

            // Create and populate graph
            using (var store = new FileGraphStore<double>(storagePath))
            {
                var alice = CreateTestNode("alice", "PERSON", new Dictionary<string, object> { { "name", "Alice" } });
                var bob = CreateTestNode("bob", "PERSON", new Dictionary<string, object> { { "name", "Bob" } });
                var acme = CreateTestNode("acme", "COMPANY", new Dictionary<string, object> { { "name", "Acme Corp" } });

                store.AddNode(alice);
                store.AddNode(bob);
                store.AddNode(acme);

                var edge1 = CreateTestEdge("alice", "KNOWS", "bob", 0.9);
                var edge2 = CreateTestEdge("alice", "WORKS_AT", "acme", 1.0);
                store.AddEdge(edge1);
                store.AddEdge(edge2);

                // Dispose to flush
            }

            // Act - Reload from disk
            using (var reloadedStore = new FileGraphStore<double>(storagePath))
            {
                // Assert - Verify nodes
                Assert.Equal(3, reloadedStore.NodeCount);
                Assert.Equal(2, reloadedStore.EdgeCount);

                var alice = reloadedStore.GetNode("alice");
                Assert.NotNull(alice);
                Assert.Equal("Alice", alice.GetProperty<string>("name"));

                var bob = reloadedStore.GetNode("bob");
                Assert.NotNull(bob);
                Assert.Equal("Bob", bob.GetProperty<string>("name"));

                // Verify edges
                var aliceOutgoing = reloadedStore.GetOutgoingEdges("alice").ToList();
                Assert.Equal(2, aliceOutgoing.Count);
                Assert.Contains(aliceOutgoing, e => e.TargetId == "bob");
                Assert.Contains(aliceOutgoing, e => e.TargetId == "acme");
            }
        }

        [Fact]
        public void Persistence_LabelIndices_RebuildCorrectly()
        {
            // Arrange
            var storagePath = GetTestStoragePath();

            using (var store = new FileGraphStore<double>(storagePath))
            {
                store.AddNode(CreateTestNode("person1", "PERSON"));
                store.AddNode(CreateTestNode("person2", "PERSON"));
                store.AddNode(CreateTestNode("company1", "COMPANY"));
            }

            // Act - Reload
            using (var reloadedStore = new FileGraphStore<double>(storagePath))
            {
                // Assert
                var persons = reloadedStore.GetNodesByLabel("PERSON").ToList();
                var companies = reloadedStore.GetNodesByLabel("COMPANY").ToList();

                Assert.Equal(2, persons.Count);
                Assert.Single(companies);
            }
        }

        [Fact]
        public void Persistence_EdgeIndices_RebuildCorrectly()
        {
            // Arrange
            var storagePath = GetTestStoragePath();

            using (var store = new FileGraphStore<double>(storagePath))
            {
                store.AddNode(CreateTestNode("node1", "PERSON"));
                store.AddNode(CreateTestNode("node2", "PERSON"));
                store.AddNode(CreateTestNode("node3", "COMPANY"));

                store.AddEdge(CreateTestEdge("node1", "KNOWS", "node2"));
                store.AddEdge(CreateTestEdge("node1", "WORKS_AT", "node3"));
                store.AddEdge(CreateTestEdge("node2", "WORKS_AT", "node3"));
            }

            // Act - Reload
            using (var reloadedStore = new FileGraphStore<double>(storagePath))
            {
                // Assert - Check outgoing edges
                var node1Outgoing = reloadedStore.GetOutgoingEdges("node1").ToList();
                Assert.Equal(2, node1Outgoing.Count);

                // Assert - Check incoming edges
                var node3Incoming = reloadedStore.GetIncomingEdges("node3").ToList();
                Assert.Equal(2, node3Incoming.Count);
            }
        }

        #endregion

        #region RemoveNode Tests

        [Fact]
        public void RemoveNode_WithExistingNode_RemovesNodeAndReturnsTrue()
        {
            // Arrange
            var storagePath = GetTestStoragePath();
            using var store = new FileGraphStore<double>(storagePath);
            var node = CreateTestNode("node1", "PERSON");
            store.AddNode(node);

            // Act
            var result = store.RemoveNode("node1");

            // Assert
            Assert.True(result);
            Assert.Equal(0, store.NodeCount);
            Assert.Null(store.GetNode("node1"));
        }

        [Fact]
        public void RemoveNode_RemovesAllConnectedEdges()
        {
            // Arrange
            var storagePath = GetTestStoragePath();
            using var store = new FileGraphStore<double>(storagePath);
            var node1 = CreateTestNode("node1", "PERSON");
            var node2 = CreateTestNode("node2", "PERSON");
            var node3 = CreateTestNode("node3", "COMPANY");
            store.AddNode(node1);
            store.AddNode(node2);
            store.AddNode(node3);

            var edge1 = CreateTestEdge("node1", "KNOWS", "node2");
            var edge2 = CreateTestEdge("node2", "WORKS_AT", "node3");
            var edge3 = CreateTestEdge("node1", "WORKS_AT", "node3");
            store.AddEdge(edge1);
            store.AddEdge(edge2);
            store.AddEdge(edge3);

            // Act
            store.RemoveNode("node1");

            // Assert
            Assert.Equal(2, store.NodeCount);
            Assert.Equal(1, store.EdgeCount);
            Assert.Null(store.GetEdge(edge1.Id));
            Assert.Null(store.GetEdge(edge3.Id));
            Assert.NotNull(store.GetEdge(edge2.Id));
        }

        [Fact]
        public void RemoveNode_Persists_AfterReload()
        {
            // Arrange
            var storagePath = GetTestStoragePath();

            using (var store = new FileGraphStore<double>(storagePath))
            {
                store.AddNode(CreateTestNode("node1", "PERSON"));
                store.AddNode(CreateTestNode("node2", "PERSON"));
                store.RemoveNode("node1");
            }

            // Act - Reload
            using (var reloadedStore = new FileGraphStore<double>(storagePath))
            {
                // Assert
                Assert.Equal(1, reloadedStore.NodeCount);
                Assert.Null(reloadedStore.GetNode("node1"));
                Assert.NotNull(reloadedStore.GetNode("node2"));
            }
        }

        #endregion

        #region Clear Tests

        [Fact]
        public void Clear_RemovesAllNodesAndEdges()
        {
            // Arrange
            var storagePath = GetTestStoragePath();
            using var store = new FileGraphStore<double>(storagePath);
            var node1 = CreateTestNode("node1", "PERSON");
            var node2 = CreateTestNode("node2", "COMPANY");
            store.AddNode(node1);
            store.AddNode(node2);
            var edge = CreateTestEdge("node1", "WORKS_AT", "node2");
            store.AddEdge(edge);

            // Act
            store.Clear();

            // Assert
            Assert.Equal(0, store.NodeCount);
            Assert.Equal(0, store.EdgeCount);
            Assert.Empty(store.GetAllNodes());
            Assert.Empty(store.GetAllEdges());
        }

        [Fact]
        public void Clear_DeletesDataFiles()
        {
            // Arrange
            var storagePath = GetTestStoragePath();
            using var store = new FileGraphStore<double>(storagePath);
            store.AddNode(CreateTestNode("node1", "PERSON"));
            store.Dispose();

            var nodesFile = Path.Combine(storagePath, "nodes.dat");
            Assert.True(File.Exists(nodesFile));

            // Act
            using var store2 = new FileGraphStore<double>(storagePath);
            store2.Clear();

            // Assert
            Assert.False(File.Exists(nodesFile));
        }

        #endregion

        #region Integration Tests

        [Fact]
        public void ComplexGraph_WithMultipleOperations_MaintainsConsistency()
        {
            // Arrange
            var storagePath = GetTestStoragePath();

            using (var store = new FileGraphStore<double>(storagePath))
            {
                // Create a small social network
                var alice = CreateTestNode("alice", "PERSON", new Dictionary<string, object> { { "name", "Alice" } });
                var bob = CreateTestNode("bob", "PERSON", new Dictionary<string, object> { { "name", "Bob" } });
                var charlie = CreateTestNode("charlie", "PERSON", new Dictionary<string, object> { { "name", "Charlie" } });
                var acme = CreateTestNode("acme", "COMPANY", new Dictionary<string, object> { { "name", "Acme Corp" } });

                store.AddNode(alice);
                store.AddNode(bob);
                store.AddNode(charlie);
                store.AddNode(acme);

                var edge1 = CreateTestEdge("alice", "KNOWS", "bob", 0.9);
                var edge2 = CreateTestEdge("bob", "KNOWS", "charlie", 0.8);
                var edge3 = CreateTestEdge("alice", "WORKS_AT", "acme", 1.0);
                var edge4 = CreateTestEdge("bob", "WORKS_AT", "acme", 1.0);

                store.AddEdge(edge1);
                store.AddEdge(edge2);
                store.AddEdge(edge3);
                store.AddEdge(edge4);

                // Verify initial state
                Assert.Equal(4, store.NodeCount);
                Assert.Equal(4, store.EdgeCount);
            }

            // Reload and modify
            using (var store = new FileGraphStore<double>(storagePath))
            {
                // Remove Bob - this removes his 3 edges: edge1 (alice->bob), edge2 (bob->charlie), edge4 (bob->acme)
                // Only edge3 (alice->acme) remains
                store.RemoveNode("bob");

                Assert.Equal(3, store.NodeCount);
                Assert.Equal(1, store.EdgeCount);
            }

            // Reload again and verify persistence
            using (var store = new FileGraphStore<double>(storagePath))
            {
                Assert.Equal(3, store.NodeCount);
                Assert.Equal(1, store.EdgeCount);
                Assert.Null(store.GetNode("bob"));
                Assert.Single(store.GetIncomingEdges("acme"));
            }
        }

        [Fact]
        public void LargeGraph_WithHundredsOfNodes_PerformsCorrectly()
        {
            // Arrange
            var storagePath = GetTestStoragePath();
            const int nodeCount = 500;

            using (var store = new FileGraphStore<double>(storagePath))
            {
                // Add many nodes
                for (int i = 0; i < nodeCount; i++)
                {
                    var node = CreateTestNode($"node_{i:D4}", "PERSON", new Dictionary<string, object>
                    {
                        { "name", $"Person {i}" },
                        { "index", i }
                    });
                    store.AddNode(node);
                }

                // Add some edges
                for (int i = 0; i < nodeCount - 1; i += 2)
                {
                    store.AddEdge(CreateTestEdge($"node_{i:D4}", "KNOWS", $"node_{(i + 1):D4}"));
                }

                Assert.Equal(nodeCount, store.NodeCount);
                Assert.Equal(nodeCount / 2, store.EdgeCount);
            }

            // Reload and verify
            using (var reloadedStore = new FileGraphStore<double>(storagePath))
            {
                Assert.Equal(nodeCount, reloadedStore.NodeCount);
                Assert.Equal(nodeCount / 2, reloadedStore.EdgeCount);

                // Verify random samples
                var node100 = reloadedStore.GetNode("node_0100");
                Assert.NotNull(node100);
                Assert.Equal("Person 100", node100.GetProperty<string>("name"));

                var node250 = reloadedStore.GetNode("node_0250");
                Assert.NotNull(node250);
                Assert.Equal(250, node250.GetProperty<int>("index"));
            }
        }

        #endregion

        #region KnowledgeGraph Integration Test

        [Fact]
        public void KnowledgeGraph_WithFileGraphStore_PersistsCorrectly()
        {
            // Arrange
            var storagePath = GetTestStoragePath();

            // Create graph with file storage
            using (var fileStore = new FileGraphStore<double>(storagePath))
            {
                var graph = new KnowledgeGraph<double>(fileStore);
                var alice = CreateTestNode("alice", "PERSON", new Dictionary<string, object> { { "name", "Alice" } });
                var bob = CreateTestNode("bob", "PERSON", new Dictionary<string, object> { { "name", "Bob" } });

                graph.AddNode(alice);
                graph.AddNode(bob);
                graph.AddEdge(CreateTestEdge("alice", "KNOWS", "bob"));

                Assert.Equal(2, graph.NodeCount);
                Assert.Equal(1, graph.EdgeCount);
            }

            // Reload with new KnowledgeGraph instance
            using (var fileStore = new FileGraphStore<double>(storagePath))
            {
                var graph = new KnowledgeGraph<double>(fileStore);
                // Assert - Data persisted
                Assert.Equal(2, graph.NodeCount);
                Assert.Equal(1, graph.EdgeCount);

                var alice = graph.GetNode("alice");
                Assert.NotNull(alice);
                Assert.Equal("Alice", alice.GetProperty<string>("name"));

                // Test graph algorithms still work
                var path = graph.FindShortestPath("alice", "bob");
                Assert.Equal(2, path.Count);
                Assert.Equal("alice", path[0]);
                Assert.Equal("bob", path[1]);
            }
        }

        #endregion
    }
}
