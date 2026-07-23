using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Graph;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNetTests.UnitTests.RetrievalAugmentedGeneration
{
    public class MemoryGraphStoreTests
    {
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

        [Fact(Timeout = 60000)]
        public async Task Constructor_InitializesEmptyStore()
        {
            // Arrange & Act
            var store = new MemoryGraphStore<double>();

            // Assert
            Assert.Equal(0, store.NodeCount);
            Assert.Equal(0, store.EdgeCount);
        }

        #endregion

        #region AddNode Tests

        [Fact(Timeout = 60000)]
        public async Task AddNode_WithValidNode_IncreasesNodeCount()
        {
            // Arrange
            var store = new MemoryGraphStore<double>();
            var node = CreateTestNode("node1", "PERSON");

            // Act
            store.AddNode(node);

            // Assert
            Assert.Equal(1, store.NodeCount);
        }

        [Fact(Timeout = 60000)]
        public async Task AddNode_WithNullNode_ThrowsArgumentNullException()
        {
            // Arrange
            var store = new MemoryGraphStore<double>();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => store.AddNode(null!));
        }

        [Fact(Timeout = 60000)]
        public async Task AddNode_WithDuplicateId_UpdatesExistingNode()
        {
            // Arrange
            var store = new MemoryGraphStore<double>();
            var node1 = CreateTestNode("node1", "PERSON", new Dictionary<string, object> { { "name", "Alice" } });
            var node2 = CreateTestNode("node1", "PERSON", new Dictionary<string, object> { { "name", "Alice Updated" } });

            // Act
            store.AddNode(node1);
            store.AddNode(node2);

            // Assert
            Assert.Equal(1, store.NodeCount);
            var retrieved = store.GetNode("node1");
            Assert.NotNull(retrieved);
            Assert.Equal("Alice Updated", retrieved.GetProperty<string>("name"));
        }

        [Fact(Timeout = 60000)]
        public async Task AddNode_WithMultipleLabels_IndexesCorrectly()
        {
            // Arrange
            var store = new MemoryGraphStore<double>();
            var person1 = CreateTestNode("person1", "PERSON");
            var person2 = CreateTestNode("person2", "PERSON");
            var company = CreateTestNode("company1", "COMPANY");

            // Act
            store.AddNode(person1);
            store.AddNode(person2);
            store.AddNode(company);

            // Assert
            Assert.Equal(3, store.NodeCount);
            Assert.Equal(2, store.GetNodesByLabel("PERSON").Count());
            Assert.Single(store.GetNodesByLabel("COMPANY"));
        }

        #endregion

        #region AddEdge Tests

        [Fact(Timeout = 60000)]
        public async Task AddEdge_WithValidEdge_IncreasesEdgeCount()
        {
            // Arrange
            var store = new MemoryGraphStore<double>();
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

        [Fact(Timeout = 60000)]
        public async Task AddEdge_WithNullEdge_ThrowsArgumentNullException()
        {
            // Arrange
            var store = new MemoryGraphStore<double>();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => store.AddEdge(null!));
        }

        [Fact(Timeout = 60000)]
        public async Task AddEdge_WithNonexistentSourceNode_ThrowsInvalidOperationException()
        {
            // Arrange
            var store = new MemoryGraphStore<double>();
            var node = CreateTestNode("node1", "PERSON");
            store.AddNode(node);
            var edge = CreateTestEdge("nonexistent", "KNOWS", "node1");

            // Act & Assert
            var exception = Assert.Throws<InvalidOperationException>(() => store.AddEdge(edge));
            Assert.Contains("Source node 'nonexistent' does not exist", exception.Message);
        }

        [Fact(Timeout = 60000)]
        public async Task AddEdge_WithNonexistentTargetNode_ThrowsInvalidOperationException()
        {
            // Arrange
            var store = new MemoryGraphStore<double>();
            var node = CreateTestNode("node1", "PERSON");
            store.AddNode(node);
            var edge = CreateTestEdge("node1", "KNOWS", "nonexistent");

            // Act & Assert
            var exception = Assert.Throws<InvalidOperationException>(() => store.AddEdge(edge));
            Assert.Contains("Target node 'nonexistent' does not exist", exception.Message);
        }

        #endregion

        #region GetNode Tests

        [Fact(Timeout = 60000)]
        public async Task GetNode_WithExistingId_ReturnsNode()
        {
            // Arrange
            var store = new MemoryGraphStore<double>();
            var node = CreateTestNode("node1", "PERSON");
            store.AddNode(node);

            // Act
            var retrieved = store.GetNode("node1");

            // Assert
            Assert.NotNull(retrieved);
            Assert.Equal("node1", retrieved.Id);
            Assert.Equal("PERSON", retrieved.Label);
        }

        [Fact(Timeout = 60000)]
        public async Task GetNode_WithNonexistentId_ReturnsNull()
        {
            // Arrange
            var store = new MemoryGraphStore<double>();

            // Act
            var retrieved = store.GetNode("nonexistent");

            // Assert
            Assert.Null(retrieved);
        }

        #endregion

        #region GetEdge Tests

        [Fact(Timeout = 60000)]
        public async Task GetEdge_WithExistingId_ReturnsEdge()
        {
            // Arrange
            var store = new MemoryGraphStore<double>();
            var node1 = CreateTestNode("node1", "PERSON");
            var node2 = CreateTestNode("node2", "COMPANY");
            store.AddNode(node1);
            store.AddNode(node2);
            var edge = CreateTestEdge("node1", "WORKS_AT", "node2");
            store.AddEdge(edge);

            // Act
            var retrieved = store.GetEdge(edge.Id);

            // Assert
            Assert.NotNull(retrieved);
            Assert.Equal(edge.Id, retrieved.Id);
            Assert.Equal("node1", retrieved.SourceId);
            Assert.Equal("node2", retrieved.TargetId);
        }

        [Fact(Timeout = 60000)]
        public async Task GetEdge_WithNonexistentId_ReturnsNull()
        {
            // Arrange
            var store = new MemoryGraphStore<double>();

            // Act
            var retrieved = store.GetEdge("nonexistent");

            // Assert
            Assert.Null(retrieved);
        }

        #endregion

        #region RemoveNode Tests

        [Fact(Timeout = 60000)]
        public async Task RemoveNode_WithExistingNode_RemovesNodeAndReturnsTrue()
        {
            // Arrange
            var store = new MemoryGraphStore<double>();
            var node = CreateTestNode("node1", "PERSON");
            store.AddNode(node);

            // Act
            var result = store.RemoveNode("node1");

            // Assert
            Assert.True(result);
            Assert.Equal(0, store.NodeCount);
            Assert.Null(store.GetNode("node1"));
        }

        [Fact(Timeout = 60000)]
        public async Task RemoveNode_WithNonexistentNode_ReturnsFalse()
        {
            // Arrange
            var store = new MemoryGraphStore<double>();

            // Act
            var result = store.RemoveNode("nonexistent");

            // Assert
            Assert.False(result);
        }

        [Fact(Timeout = 60000)]
        public async Task RemoveNode_RemovesAllConnectedEdges()
        {
            // Arrange
            var store = new MemoryGraphStore<double>();
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
            Assert.Equal(1, store.EdgeCount); // Only edge2 remains
            Assert.Null(store.GetEdge(edge1.Id));
            Assert.Null(store.GetEdge(edge3.Id));
            Assert.NotNull(store.GetEdge(edge2.Id));
        }

        [Fact(Timeout = 60000)]
        public async Task RemoveNode_RemovesFromLabelIndex()
        {
            // Arrange
            var store = new MemoryGraphStore<double>();
            var person1 = CreateTestNode("person1", "PERSON");
            var person2 = CreateTestNode("person2", "PERSON");
            store.AddNode(person1);
            store.AddNode(person2);

            // Act
            store.RemoveNode("person1");

            // Assert
            var personsRemaining = store.GetNodesByLabel("PERSON").ToList();
            Assert.Single(personsRemaining);
            Assert.Equal("person2", personsRemaining[0].Id);
        }

        #endregion

        #region RemoveEdge Tests

        [Fact(Timeout = 60000)]
        public async Task RemoveEdge_WithExistingEdge_RemovesEdgeAndReturnsTrue()
        {
            // Arrange
            var store = new MemoryGraphStore<double>();
            var node1 = CreateTestNode("node1", "PERSON");
            var node2 = CreateTestNode("node2", "COMPANY");
            store.AddNode(node1);
            store.AddNode(node2);
            var edge = CreateTestEdge("node1", "WORKS_AT", "node2");
            store.AddEdge(edge);

            // Act
            var result = store.RemoveEdge(edge.Id);

            // Assert
            Assert.True(result);
            Assert.Equal(0, store.EdgeCount);
            Assert.Null(store.GetEdge(edge.Id));
        }

        [Fact(Timeout = 60000)]
        public async Task RemoveEdge_WithNonexistentEdge_ReturnsFalse()
        {
            // Arrange
            var store = new MemoryGraphStore<double>();

            // Act
            var result = store.RemoveEdge("nonexistent");

            // Assert
            Assert.False(result);
        }

        [Fact(Timeout = 60000)]
        public async Task RemoveEdge_DoesNotRemoveNodes()
        {
            // Arrange
            var store = new MemoryGraphStore<double>();
            var node1 = CreateTestNode("node1", "PERSON");
            var node2 = CreateTestNode("node2", "COMPANY");
            store.AddNode(node1);
            store.AddNode(node2);
            var edge = CreateTestEdge("node1", "WORKS_AT", "node2");
            store.AddEdge(edge);

            // Act
            store.RemoveEdge(edge.Id);

            // Assert
            Assert.Equal(2, store.NodeCount);
            Assert.NotNull(store.GetNode("node1"));
            Assert.NotNull(store.GetNode("node2"));
        }

        #endregion

        #region GetOutgoingEdges Tests

        [Fact(Timeout = 60000)]
        public async Task GetOutgoingEdges_WithExistingEdges_ReturnsCorrectEdges()
        {
            // Arrange
            var store = new MemoryGraphStore<double>();
            var node1 = CreateTestNode("node1", "PERSON");
            var node2 = CreateTestNode("node2", "PERSON");
            var node3 = CreateTestNode("node3", "COMPANY");
            store.AddNode(node1);
            store.AddNode(node2);
            store.AddNode(node3);

            var edge1 = CreateTestEdge("node1", "KNOWS", "node2");
            var edge2 = CreateTestEdge("node1", "WORKS_AT", "node3");
            var edge3 = CreateTestEdge("node2", "WORKS_AT", "node3");
            store.AddEdge(edge1);
            store.AddEdge(edge2);
            store.AddEdge(edge3);

            // Act
            var outgoing = store.GetOutgoingEdges("node1").ToList();

            // Assert
            Assert.Equal(2, outgoing.Count);
            Assert.Contains(outgoing, e => e.Id == edge1.Id);
            Assert.Contains(outgoing, e => e.Id == edge2.Id);
        }

        [Fact(Timeout = 60000)]
        public async Task GetOutgoingEdges_WithNoEdges_ReturnsEmpty()
        {
            // Arrange
            var store = new MemoryGraphStore<double>();
            var node = CreateTestNode("node1", "PERSON");
            store.AddNode(node);

            // Act
            var outgoing = store.GetOutgoingEdges("node1");

            // Assert
            Assert.Empty(outgoing);
        }

        [Fact(Timeout = 60000)]
        public async Task GetOutgoingEdges_WithNonexistentNode_ReturnsEmpty()
        {
            // Arrange
            var store = new MemoryGraphStore<double>();

            // Act
            var outgoing = store.GetOutgoingEdges("nonexistent");

            // Assert
            Assert.Empty(outgoing);
        }

        #endregion

        #region GetIncomingEdges Tests

        [Fact(Timeout = 60000)]
        public async Task GetIncomingEdges_WithExistingEdges_ReturnsCorrectEdges()
        {
            // Arrange
            var store = new MemoryGraphStore<double>();
            var node1 = CreateTestNode("node1", "PERSON");
            var node2 = CreateTestNode("node2", "PERSON");
            var node3 = CreateTestNode("node3", "COMPANY");
            store.AddNode(node1);
            store.AddNode(node2);
            store.AddNode(node3);

            var edge1 = CreateTestEdge("node1", "WORKS_AT", "node3");
            var edge2 = CreateTestEdge("node2", "WORKS_AT", "node3");
            var edge3 = CreateTestEdge("node1", "KNOWS", "node2");
            store.AddEdge(edge1);
            store.AddEdge(edge2);
            store.AddEdge(edge3);

            // Act
            var incoming = store.GetIncomingEdges("node3").ToList();

            // Assert
            Assert.Equal(2, incoming.Count);
            Assert.Contains(incoming, e => e.Id == edge1.Id);
            Assert.Contains(incoming, e => e.Id == edge2.Id);
        }

        [Fact(Timeout = 60000)]
        public async Task GetIncomingEdges_WithNoEdges_ReturnsEmpty()
        {
            // Arrange
            var store = new MemoryGraphStore<double>();
            var node = CreateTestNode("node1", "PERSON");
            store.AddNode(node);

            // Act
            var incoming = store.GetIncomingEdges("node1");

            // Assert
            Assert.Empty(incoming);
        }

        [Fact(Timeout = 60000)]
        public async Task GetIncomingEdges_WithNonexistentNode_ReturnsEmpty()
        {
            // Arrange
            var store = new MemoryGraphStore<double>();

            // Act
            var incoming = store.GetIncomingEdges("nonexistent");

            // Assert
            Assert.Empty(incoming);
        }

        #endregion

        #region GetNodesByLabel Tests

        [Fact(Timeout = 60000)]
        public async Task GetNodesByLabel_WithMatchingNodes_ReturnsCorrectNodes()
        {
            // Arrange
            var store = new MemoryGraphStore<double>();
            var person1 = CreateTestNode("person1", "PERSON");
            var person2 = CreateTestNode("person2", "PERSON");
            var company = CreateTestNode("company1", "COMPANY");
            store.AddNode(person1);
            store.AddNode(person2);
            store.AddNode(company);

            // Act
            var persons = store.GetNodesByLabel("PERSON").ToList();

            // Assert
            Assert.Equal(2, persons.Count);
            Assert.Contains(persons, n => n.Id == "person1");
            Assert.Contains(persons, n => n.Id == "person2");
        }

        [Fact(Timeout = 60000)]
        public async Task GetNodesByLabel_WithNoMatches_ReturnsEmpty()
        {
            // Arrange
            var store = new MemoryGraphStore<double>();
            var person = CreateTestNode("person1", "PERSON");
            store.AddNode(person);

            // Act
            var companies = store.GetNodesByLabel("COMPANY");

            // Assert
            Assert.Empty(companies);
        }

        [Fact(Timeout = 60000)]
        public async Task GetNodesByLabel_WithNonexistentLabel_ReturnsEmpty()
        {
            // Arrange
            var store = new MemoryGraphStore<double>();

            // Act
            var nodes = store.GetNodesByLabel("NONEXISTENT");

            // Assert
            Assert.Empty(nodes);
        }

        #endregion

        #region GetAllNodes Tests

        [Fact(Timeout = 60000)]
        public async Task GetAllNodes_WithMultipleNodes_ReturnsAllNodes()
        {
            // Arrange
            var store = new MemoryGraphStore<double>();
            var node1 = CreateTestNode("node1", "PERSON");
            var node2 = CreateTestNode("node2", "COMPANY");
            var node3 = CreateTestNode("node3", "LOCATION");
            store.AddNode(node1);
            store.AddNode(node2);
            store.AddNode(node3);

            // Act
            var allNodes = store.GetAllNodes().ToList();

            // Assert
            Assert.Equal(3, allNodes.Count);
            Assert.Contains(allNodes, n => n.Id == "node1");
            Assert.Contains(allNodes, n => n.Id == "node2");
            Assert.Contains(allNodes, n => n.Id == "node3");
        }

        [Fact(Timeout = 60000)]
        public async Task GetAllNodes_WithEmptyStore_ReturnsEmpty()
        {
            // Arrange
            var store = new MemoryGraphStore<double>();

            // Act
            var allNodes = store.GetAllNodes();

            // Assert
            Assert.Empty(allNodes);
        }

        #endregion

        #region GetAllEdges Tests

        [Fact(Timeout = 60000)]
        public async Task GetAllEdges_WithMultipleEdges_ReturnsAllEdges()
        {
            // Arrange
            var store = new MemoryGraphStore<double>();
            var node1 = CreateTestNode("node1", "PERSON");
            var node2 = CreateTestNode("node2", "PERSON");
            var node3 = CreateTestNode("node3", "COMPANY");
            store.AddNode(node1);
            store.AddNode(node2);
            store.AddNode(node3);

            var edge1 = CreateTestEdge("node1", "KNOWS", "node2");
            var edge2 = CreateTestEdge("node1", "WORKS_AT", "node3");
            var edge3 = CreateTestEdge("node2", "WORKS_AT", "node3");
            store.AddEdge(edge1);
            store.AddEdge(edge2);
            store.AddEdge(edge3);

            // Act
            var allEdges = store.GetAllEdges().ToList();

            // Assert
            Assert.Equal(3, allEdges.Count);
            Assert.Contains(allEdges, e => e.Id == edge1.Id);
            Assert.Contains(allEdges, e => e.Id == edge2.Id);
            Assert.Contains(allEdges, e => e.Id == edge3.Id);
        }

        [Fact(Timeout = 60000)]
        public async Task GetAllEdges_WithEmptyStore_ReturnsEmpty()
        {
            // Arrange
            var store = new MemoryGraphStore<double>();

            // Act
            var allEdges = store.GetAllEdges();

            // Assert
            Assert.Empty(allEdges);
        }

        #endregion

        #region Clear Tests

        [Fact(Timeout = 60000)]
        public async Task Clear_RemovesAllNodesAndEdges()
        {
            // Arrange
            var store = new MemoryGraphStore<double>();
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

        [Fact(Timeout = 60000)]
        public async Task Clear_OnEmptyStore_DoesNotThrow()
        {
            // Arrange
            var store = new MemoryGraphStore<double>();

            // Act & Assert
            store.Clear(); // Should not throw
            Assert.Equal(0, store.NodeCount);
            Assert.Equal(0, store.EdgeCount);
        }

        #endregion

        #region Integration Tests

        [Fact(Timeout = 60000)]
        public async Task ComplexGraph_WithMultipleOperations_MaintainsConsistency()
        {
            // Arrange
            var store = new MemoryGraphStore<double>();

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

            // Act & Assert - Verify initial state
            Assert.Equal(4, store.NodeCount);
            Assert.Equal(4, store.EdgeCount);
            Assert.Equal(3, store.GetNodesByLabel("PERSON").Count());
            Assert.Single(store.GetNodesByLabel("COMPANY"));

            // Act - Remove Bob
            store.RemoveNode("bob");

            // Assert - Bob and his edges are gone
            // Bob has 3 edges: edge1 (incoming from alice), edge2 (outgoing to charlie), edge4 (outgoing to acme)
            // Only edge3 (alice -> acme) remains, so EdgeCount = 1
            Assert.Equal(3, store.NodeCount);
            Assert.Equal(1, store.EdgeCount);
            Assert.Null(store.GetNode("bob"));
            Assert.Null(store.GetEdge(edge1.Id));
            Assert.Null(store.GetEdge(edge2.Id));
            Assert.Null(store.GetEdge(edge4.Id));

            // Assert - Alice's edge to Acme still exists
            Assert.NotNull(store.GetEdge(edge3.Id));
            Assert.Single(store.GetOutgoingEdges("alice"));
            Assert.Single(store.GetIncomingEdges("acme"));
        }

        #endregion
    }
}
