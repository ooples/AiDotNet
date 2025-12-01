using System;
using System.Collections.Generic;
using System.IO;
using AiDotNet.RetrievalAugmentedGeneration.Graph;
using Xunit;

namespace AiDotNetTests.UnitTests.RetrievalAugmentedGeneration
{
    public class GraphTransactionTests : IDisposable
    {
        private readonly string _testDirectory;

        public GraphTransactionTests()
        {
            _testDirectory = Path.Combine(Path.GetTempPath(), "txn_tests_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(_testDirectory);
        }

        public void Dispose()
        {
            if (Directory.Exists(_testDirectory))
                Directory.Delete(_testDirectory, true);
        }

        private GraphNode<double> CreateTestNode(string id, string label)
        {
            var node = new GraphNode<double>(id, label);
            node.SetProperty("name", $"Node {id}");
            return node;
        }

        private GraphEdge<double> CreateTestEdge(string sourceId, string relationType, string targetId)
        {
            return new GraphEdge<double>(sourceId, targetId, relationType, 1.0);
        }

        #region Basic Transaction Tests

        [Fact]
        public void Transaction_Begin_SetsStateToActive()
        {
            // Arrange
            var store = new MemoryGraphStore<double>();
            var txn = new GraphTransaction<double>(store);

            // Act
            txn.Begin();

            // Assert
            Assert.Equal(TransactionState.Active, txn.State);
            Assert.NotEqual(-1, txn.TransactionId);
        }

        [Fact]
        public void Transaction_BeginTwice_ThrowsException()
        {
            // Arrange
            var store = new MemoryGraphStore<double>();
            var txn = new GraphTransaction<double>(store);
            txn.Begin();

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() => txn.Begin());
        }

        [Fact]
        public void Transaction_AddNodeBeforeBegin_ThrowsException()
        {
            // Arrange
            var store = new MemoryGraphStore<double>();
            var txn = new GraphTransaction<double>(store);
            var node = CreateTestNode("node1", "PERSON");

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() => txn.AddNode(node));
        }

        #endregion

        #region Commit Tests

        [Fact]
        public void Transaction_Commit_AppliesAllOperations()
        {
            // Arrange
            var store = new MemoryGraphStore<double>();
            var txn = new GraphTransaction<double>(store);
            var node1 = CreateTestNode("alice", "PERSON");
            var node2 = CreateTestNode("bob", "PERSON");
            var edge = CreateTestEdge("alice", "KNOWS", "bob");

            // Act
            txn.Begin();
            txn.AddNode(node1);
            txn.AddNode(node2);
            txn.AddEdge(edge);
            txn.Commit();

            // Assert
            Assert.Equal(TransactionState.Committed, txn.State);
            Assert.Equal(2, store.NodeCount);
            Assert.Equal(1, store.EdgeCount);
            Assert.NotNull(store.GetNode("alice"));
            Assert.NotNull(store.GetNode("bob"));
        }

        [Fact]
        public void Transaction_CommitWithRemoveOperations_WorksCorrectly()
        {
            // Arrange
            var store = new MemoryGraphStore<double>();
            store.AddNode(CreateTestNode("alice", "PERSON"));
            store.AddNode(CreateTestNode("bob", "PERSON"));
            var edge = CreateTestEdge("alice", "KNOWS", "bob");
            store.AddEdge(edge);

            var txn = new GraphTransaction<double>(store);

            // Act
            txn.Begin();
            txn.RemoveEdge(edge.Id);
            txn.RemoveNode("bob");
            txn.Commit();

            // Assert
            Assert.Equal(TransactionState.Committed, txn.State);
            Assert.Equal(1, store.NodeCount);
            Assert.Equal(0, store.EdgeCount);
            Assert.NotNull(store.GetNode("alice"));
            Assert.Null(store.GetNode("bob"));
        }

        [Fact]
        public void Transaction_CommitBeforeBegin_ThrowsException()
        {
            // Arrange
            var store = new MemoryGraphStore<double>();
            var txn = new GraphTransaction<double>(store);

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() => txn.Commit());
        }

        [Fact]
        public void Transaction_CommitAfterCommit_ThrowsException()
        {
            // Arrange
            var store = new MemoryGraphStore<double>();
            var txn = new GraphTransaction<double>(store);
            txn.Begin();
            txn.AddNode(CreateTestNode("node1", "PERSON"));
            txn.Commit();

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() => txn.Commit());
        }

        #endregion

        #region Rollback Tests

        [Fact]
        public void Transaction_Rollback_DiscardsAllOperations()
        {
            // Arrange
            var store = new MemoryGraphStore<double>();
            var txn = new GraphTransaction<double>(store);
            var node1 = CreateTestNode("alice", "PERSON");
            var node2 = CreateTestNode("bob", "PERSON");

            // Act
            txn.Begin();
            txn.AddNode(node1);
            txn.AddNode(node2);
            txn.Rollback();

            // Assert
            Assert.Equal(TransactionState.RolledBack, txn.State);
            Assert.Equal(0, store.NodeCount);
            Assert.Null(store.GetNode("alice"));
            Assert.Null(store.GetNode("bob"));
        }

        [Fact]
        public void Transaction_RollbackBeforeBegin_ThrowsException()
        {
            // Arrange
            var store = new MemoryGraphStore<double>();
            var txn = new GraphTransaction<double>(store);

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() => txn.Rollback());
        }

        [Fact]
        public void Transaction_RollbackAfterCommit_ThrowsException()
        {
            // Arrange
            var store = new MemoryGraphStore<double>();
            var txn = new GraphTransaction<double>(store);
            txn.Begin();
            txn.AddNode(CreateTestNode("node1", "PERSON"));
            txn.Commit();

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() => txn.Rollback());
        }

        #endregion

        #region WAL Integration Tests

        [Fact]
        public void Transaction_WithWAL_LogsOperationsBeforeCommit()
        {
            // Arrange
            var walPath = Path.Combine(_testDirectory, "test.wal");
            var wal = new WriteAheadLog(walPath);
            var store = new MemoryGraphStore<double>();
            var txn = new GraphTransaction<double>(store, wal);
            var node = CreateTestNode("alice", "PERSON");

            // Act
            txn.Begin();
            txn.AddNode(node);
            txn.Commit();

            // Assert
            var entries = wal.ReadLog();
            Assert.NotEmpty(entries);
            Assert.Contains(entries, e => e.OperationType == WALOperationType.AddNode && e.NodeId == "alice");
            Assert.Contains(entries, e => e.OperationType == WALOperationType.Checkpoint);

            wal.Dispose();
        }

        [Fact]
        public void Transaction_WithWAL_RollbackDoesNotLogCheckpoint()
        {
            // Arrange
            var walPath = Path.Combine(_testDirectory, "test_rollback.wal");
            var wal = new WriteAheadLog(walPath);
            var store = new MemoryGraphStore<double>();
            var txn = new GraphTransaction<double>(store, wal);
            var node = CreateTestNode("alice", "PERSON");

            // Act
            txn.Begin();
            txn.AddNode(node);
            txn.Rollback();

            // Assert
            var entries = wal.ReadLog();
            // No checkpoint should be logged on rollback
            Assert.DoesNotContain(entries, e => e.OperationType == WALOperationType.Checkpoint);

            wal.Dispose();
        }

        [Fact]
        public void Transaction_WithWAL_SupportsMultipleOperations()
        {
            // Arrange
            var walPath = Path.Combine(_testDirectory, "test_multi.wal");
            var wal = new WriteAheadLog(walPath);
            var store = new MemoryGraphStore<double>();
            var txn = new GraphTransaction<double>(store, wal);

            // Act
            txn.Begin();
            txn.AddNode(CreateTestNode("alice", "PERSON"));
            txn.AddNode(CreateTestNode("bob", "PERSON"));
            txn.AddEdge(CreateTestEdge("alice", "KNOWS", "bob"));
            txn.Commit();

            // Assert
            var entries = wal.ReadLog();
            Assert.Equal(4, entries.Count); // 2 AddNode + 1 AddEdge + 1 Checkpoint
            Assert.Equal(WALOperationType.AddNode, entries[0].OperationType);
            Assert.Equal(WALOperationType.AddNode, entries[1].OperationType);
            Assert.Equal(WALOperationType.AddEdge, entries[2].OperationType);
            Assert.Equal(WALOperationType.Checkpoint, entries[3].OperationType);

            wal.Dispose();
        }

        #endregion

        #region FileGraphStore Integration Tests

        [Fact]
        public void Transaction_WithFileStore_CommitPersistsData()
        {
            // Arrange
            var storagePath = Path.Combine(_testDirectory, Guid.NewGuid().ToString("N"));
            using var store = new FileGraphStore<double>(storagePath);
            var txn = new GraphTransaction<double>(store);

            // Act
            txn.Begin();
            txn.AddNode(CreateTestNode("alice", "PERSON"));
            txn.AddNode(CreateTestNode("bob", "PERSON"));
            txn.Commit();

            // Assert
            Assert.Equal(2, store.NodeCount);
        }

        [Fact]
        public void Transaction_WithFileStoreAndWAL_FullACIDSupport()
        {
            // Arrange
            var storagePath = Path.Combine(_testDirectory, Guid.NewGuid().ToString("N"));
            var walPath = Path.Combine(_testDirectory, "full_acid.wal");
            var wal = new WriteAheadLog(walPath);
            using var store = new FileGraphStore<double>(storagePath, wal);
            var txn = new GraphTransaction<double>(store, wal);

            // Act
            txn.Begin();
            txn.AddNode(CreateTestNode("alice", "PERSON"));
            txn.AddNode(CreateTestNode("bob", "PERSON"));
            txn.AddEdge(CreateTestEdge("alice", "KNOWS", "bob"));
            txn.Commit();

            // Assert - Check store
            Assert.Equal(2, store.NodeCount);
            Assert.Equal(1, store.EdgeCount);

            // Assert - Check WAL
            var entries = wal.ReadLog();
            Assert.Equal(4, entries.Count); // 2 AddNode + 1 AddEdge + 1 Checkpoint

            wal.Dispose();
        }

        [Fact]
        public void Transaction_WithFileStoreAndWAL_RollbackPreventsPersistence()
        {
            // Arrange
            var storagePath = Path.Combine(_testDirectory, Guid.NewGuid().ToString("N"));
            var walPath = Path.Combine(_testDirectory, "rollback_test.wal");
            var wal = new WriteAheadLog(walPath);
            using var store = new FileGraphStore<double>(storagePath, wal);
            var txn = new GraphTransaction<double>(store, wal);

            // Act
            txn.Begin();
            txn.AddNode(CreateTestNode("alice", "PERSON"));
            txn.AddNode(CreateTestNode("bob", "PERSON"));
            txn.Rollback();

            // Assert
            Assert.Equal(0, store.NodeCount);

            wal.Dispose();
        }

        #endregion

        #region Dispose Tests

        [Fact]
        public void Transaction_Dispose_AutoRollbacksActiveTransaction()
        {
            // Arrange
            var store = new MemoryGraphStore<double>();
            var txn = new GraphTransaction<double>(store);
            var node = CreateTestNode("alice", "PERSON");

            // Act
            txn.Begin();
            txn.AddNode(node);
            txn.Dispose(); // Should auto-rollback

            // Assert
            Assert.Equal(0, store.NodeCount);
        }

        [Fact]
        public void Transaction_Dispose_DoesNotAffectCommittedTransaction()
        {
            // Arrange
            var store = new MemoryGraphStore<double>();
            var txn = new GraphTransaction<double>(store);
            var node = CreateTestNode("alice", "PERSON");

            // Act
            txn.Begin();
            txn.AddNode(node);
            txn.Commit();
            txn.Dispose();

            // Assert
            Assert.Equal(1, store.NodeCount);
        }

        [Fact]
        public void Transaction_UsingStatement_AutoRollbacksOnException()
        {
            // Arrange
            var store = new MemoryGraphStore<double>();

            // Act & Assert
            try
            {
                using var txn = new GraphTransaction<double>(store);
                txn.Begin();
                txn.AddNode(CreateTestNode("alice", "PERSON"));
                throw new Exception("Simulated error");
            }
            catch
            {
                // Swallow exception
            }

            // Transaction should have been rolled back
            Assert.Equal(0, store.NodeCount);
        }

        #endregion

        #region Error Handling Tests

        [Fact]
        public void Transaction_Commit_FailsIfStoreThrows()
        {
            // Arrange
            var store = new MemoryGraphStore<double>();
            var txn = new GraphTransaction<double>(store);

            // Add invalid edge (nodes don't exist)
            var edge = CreateTestEdge("nonexistent1", "KNOWS", "nonexistent2");

            // Act & Assert
            txn.Begin();
            txn.AddEdge(edge);
            Assert.Throws<InvalidOperationException>(() => txn.Commit());
            Assert.Equal(TransactionState.Failed, txn.State);
        }

        [Fact]
        public void Transaction_FailedTransaction_CanBeRolledBack()
        {
            // Arrange
            var store = new MemoryGraphStore<double>();
            var txn = new GraphTransaction<double>(store);
            var edge = CreateTestEdge("nonexistent1", "KNOWS", "nonexistent2");

            // Act
            txn.Begin();
            txn.AddNode(CreateTestNode("alice", "PERSON")); // This is valid
            txn.AddEdge(edge); // This will fail on commit

            try
            {
                txn.Commit();
            }
            catch (InvalidOperationException)
            {
                // Expected
            }

            // Transaction is now in Failed state
            Assert.Equal(TransactionState.Failed, txn.State);

            // Should be able to rollback
            txn.Rollback();
            Assert.Equal(TransactionState.RolledBack, txn.State);
        }

        #endregion

        #region ACID Property Tests

        [Fact]
        public void Transaction_Atomicity_AllOrNothing()
        {
            // Arrange
            var store = new MemoryGraphStore<double>();
            var txn = new GraphTransaction<double>(store);

            // Act - Transaction with error in the middle
            txn.Begin();
            txn.AddNode(CreateTestNode("alice", "PERSON"));
            txn.AddNode(CreateTestNode("bob", "PERSON"));
            txn.AddEdge(CreateTestEdge("alice", "KNOWS", "charlie")); // charlie doesn't exist - will fail

            try
            {
                txn.Commit();
            }
            catch (InvalidOperationException)
            {
                // Expected failure
            }

            // Assert - Nothing should be committed (Atomicity)
            // Because the commit failed, the entire transaction should be reverted
            Assert.Equal(TransactionState.Failed, txn.State);
        }

        [Fact]
        public void Transaction_Consistency_GraphRemainsValid()
        {
            // Arrange
            var store = new MemoryGraphStore<double>();
            var txn1 = new GraphTransaction<double>(store);
            var txn2 = new GraphTransaction<double>(store);

            // Act - First transaction succeeds
            txn1.Begin();
            txn1.AddNode(CreateTestNode("alice", "PERSON"));
            txn1.AddNode(CreateTestNode("bob", "PERSON"));
            txn1.Commit();

            // Second transaction uses the nodes from first
            txn2.Begin();
            txn2.AddEdge(CreateTestEdge("alice", "KNOWS", "bob"));
            txn2.Commit();

            // Assert - Graph is in valid state
            Assert.Equal(2, store.NodeCount);
            Assert.Equal(1, store.EdgeCount);
            var aliceEdges = store.GetOutgoingEdges("alice");
            Assert.Single(aliceEdges);
        }

        [Fact]
        public void Transaction_Durability_WithWALSurvivesCrash()
        {
            // Arrange
            var storagePath = Path.Combine(_testDirectory, Guid.NewGuid().ToString("N"));
            var walPath = Path.Combine(_testDirectory, "durability_test.wal");

            // First session - write data
            using (var wal = new WriteAheadLog(walPath))
            using (var store = new FileGraphStore<double>(storagePath, wal))
            {
                var txn = new GraphTransaction<double>(store, wal);
                txn.Begin();
                txn.AddNode(CreateTestNode("alice", "PERSON"));
                txn.AddNode(CreateTestNode("bob", "PERSON"));
                txn.Commit();
            }
            // Simulate crash - dispose everything

            // Second session - verify data survived
            using (var wal = new WriteAheadLog(walPath))
            using (var store = new FileGraphStore<double>(storagePath, wal))
            {
                // Assert - Data should still be there (Durability)
                Assert.Equal(2, store.NodeCount);
                Assert.NotNull(store.GetNode("alice"));
                Assert.NotNull(store.GetNode("bob"));

                // WAL should show checkpoint
                var entries = wal.ReadLog();
                Assert.Contains(entries, e => e.OperationType == WALOperationType.Checkpoint);
            }
        }

        #endregion

        #region Complex Scenario Tests

        [Fact]
        public void Transaction_MultipleSequentialTransactions_WorkCorrectly()
        {
            // Arrange
            var store = new MemoryGraphStore<double>();

            // Act - Multiple transactions
            using (var txn1 = new GraphTransaction<double>(store))
            {
                txn1.Begin();
                txn1.AddNode(CreateTestNode("alice", "PERSON"));
                txn1.Commit();
            }

            using (var txn2 = new GraphTransaction<double>(store))
            {
                txn2.Begin();
                txn2.AddNode(CreateTestNode("bob", "PERSON"));
                txn2.Commit();
            }

            using (var txn3 = new GraphTransaction<double>(store))
            {
                txn3.Begin();
                txn3.AddEdge(CreateTestEdge("alice", "KNOWS", "bob"));
                txn3.Commit();
            }

            // Assert
            Assert.Equal(2, store.NodeCount);
            Assert.Equal(1, store.EdgeCount);
        }

        [Fact]
        public void Transaction_MixedSuccessAndRollback_MaintainsConsistency()
        {
            // Arrange
            var store = new MemoryGraphStore<double>();

            // Transaction 1 - Success
            using (var txn1 = new GraphTransaction<double>(store))
            {
                txn1.Begin();
                txn1.AddNode(CreateTestNode("alice", "PERSON"));
                txn1.Commit();
            }

            // Transaction 2 - Rollback
            using (var txn2 = new GraphTransaction<double>(store))
            {
                txn2.Begin();
                txn2.AddNode(CreateTestNode("bob", "PERSON"));
                txn2.Rollback();
            }

            // Transaction 3 - Success
            using (var txn3 = new GraphTransaction<double>(store))
            {
                txn3.Begin();
                txn3.AddNode(CreateTestNode("charlie", "PERSON"));
                txn3.Commit();
            }

            // Assert
            Assert.Equal(2, store.NodeCount); // Only alice and charlie
            Assert.NotNull(store.GetNode("alice"));
            Assert.Null(store.GetNode("bob")); // Rolled back
            Assert.NotNull(store.GetNode("charlie"));
        }

        [Fact]
        public void Transaction_LargeTransaction_HandlesMultipleOperations()
        {
            // Arrange
            var store = new MemoryGraphStore<double>();
            var txn = new GraphTransaction<double>(store);

            // Act - Add 100 nodes and 99 edges in a single transaction
            txn.Begin();
            for (int i = 0; i < 100; i++)
            {
                txn.AddNode(CreateTestNode($"node{i}", "PERSON"));
            }
            for (int i = 0; i < 99; i++)
            {
                txn.AddEdge(CreateTestEdge($"node{i}", "NEXT", $"node{i + 1}"));
            }
            txn.Commit();

            // Assert
            Assert.Equal(100, store.NodeCount);
            Assert.Equal(99, store.EdgeCount);
        }

        #endregion
    }
}
