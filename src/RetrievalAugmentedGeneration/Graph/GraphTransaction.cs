using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using AiDotNet.Validation;

namespace AiDotNet.RetrievalAugmentedGeneration.Graph;

/// <summary>
/// Transaction coordinator for managing transactions on graph stores with best-effort rollback.
/// </summary>
/// <typeparam name="T">The numeric type used for vector operations.</typeparam>
/// <remarks>
/// <para>
/// This class provides transaction management with the following guarantees:
/// - <b>Atomicity (Best-Effort)</b>: If an operation fails during commit, compensating rollback
///   is attempted in reverse order. However, if an undo operation fails, it is swallowed and
///   rollback continues with remaining operations. Full atomicity is not guaranteed.
/// - <b>Consistency</b>: Graph validation rules are enforced during operations.
/// - <b>Isolation</b>: Single-threaded; no concurrent transaction support.
/// - <b>Durability</b>: When a WAL is provided, operations are logged before execution.
///   Without a WAL, durability is not guaranteed.
/// </para>
/// <para>
/// <b>Important:</b> This is a lightweight transaction implementation suitable for single-process
/// use cases. For full ACID compliance with crash recovery, ensure a WriteAheadLog is configured.
/// </para>
/// <para><b>For Beginners:</b> Transactions ensure your changes are safe.
///
/// Think of a bank transfer:
/// - Debit $100 from Alice
/// - Credit $100 to Bob
///
/// Without transactions:
/// - If crash happens after debit but before credit, $100 disappears!
///
/// With transactions:
/// - Begin transaction
/// - Debit Alice
/// - Credit Bob
/// - Commit (both succeed) OR Rollback (both undone)
/// - Money never disappears!
///
/// In graphs:
/// ```csharp
/// var txn = new GraphTransaction(store, wal);
/// txn.Begin();
/// try
/// {
///     txn.AddNode(node1);
///     txn.AddEdge(edge1);
///     txn.Commit(); // Both saved
/// }
/// catch
/// {
///     txn.Rollback(); // Both undone
/// }
/// ```
///
/// This ensures your graph is never in a broken state!
/// </para>
/// </remarks>
public class GraphTransaction<T> : IDisposable
{
    private readonly IGraphStore<T> _store;
    private readonly WriteAheadLog? _wal;
    private readonly List<TransactionOperation<T>> _operations;
    private TransactionState _state;
    private long _transactionId;
    private bool _disposed;

    /// <summary>
    /// Gets the current state of the transaction.
    /// </summary>
    public TransactionState State => _state;

    /// <summary>
    /// Gets the transaction ID.
    /// </summary>
    public long TransactionId => _transactionId;

    /// <summary>
    /// Initializes a new instance of the <see cref="GraphTransaction{T}"/> class.
    /// </summary>
    /// <param name="store">The graph store to operate on.</param>
    /// <param name="wal">Optional Write-Ahead Log for durability.</param>
    public GraphTransaction(IGraphStore<T> store, WriteAheadLog? wal = null)
    {
        Guard.NotNull(store);
        _store = store;
        _wal = wal;
        _operations = new List<TransactionOperation<T>>();
        _state = TransactionState.NotStarted;
        _transactionId = -1;
    }

    /// <summary>
    /// Begins a new transaction.
    /// </summary>
    /// <exception cref="InvalidOperationException">Thrown if transaction already started.</exception>
    public void Begin()
    {
        if (_state != TransactionState.NotStarted)
            throw new InvalidOperationException($"Transaction already in state: {_state}");

        _state = TransactionState.Active;
        _transactionId = DateTime.UtcNow.Ticks; // Simple ID generation
        _operations.Clear();
    }

    /// <summary>
    /// Adds a node within the transaction.
    /// </summary>
    /// <param name="node">The node to add.</param>
    public void AddNode(GraphNode<T> node)
    {
        EnsureActive();

        _operations.Add(new TransactionOperation<T>
        {
            Type = GraphOperationType.AddNode,
            Node = node
        });
    }

    /// <summary>
    /// Adds an edge within the transaction.
    /// </summary>
    /// <param name="edge">The edge to add.</param>
    public void AddEdge(GraphEdge<T> edge)
    {
        EnsureActive();

        _operations.Add(new TransactionOperation<T>
        {
            Type = GraphOperationType.AddEdge,
            Edge = edge
        });
    }

    /// <summary>
    /// Removes a node within the transaction.
    /// </summary>
    /// <param name="nodeId">The ID of the node to remove.</param>
    public void RemoveNode(string nodeId)
    {
        EnsureActive();

        // Capture original node data for potential undo
        var originalNode = _store.GetNode(nodeId);

        _operations.Add(new TransactionOperation<T>
        {
            Type = GraphOperationType.RemoveNode,
            NodeId = nodeId,
            Node = originalNode // Store for undo
        });
    }

    /// <summary>
    /// Removes an edge within the transaction.
    /// </summary>
    /// <param name="edgeId">The ID of the edge to remove.</param>
    public void RemoveEdge(string edgeId)
    {
        EnsureActive();

        // Capture original edge data for potential undo
        var originalEdge = _store.GetEdge(edgeId);

        _operations.Add(new TransactionOperation<T>
        {
            Type = GraphOperationType.RemoveEdge,
            EdgeId = edgeId,
            Edge = originalEdge // Store for undo
        });
    }

    /// <summary>
    /// Commits the transaction, applying all operations with best-effort atomicity.
    /// </summary>
    /// <exception cref="InvalidOperationException">Thrown if transaction not active.</exception>
    /// <remarks>
    /// <para>
    /// If an operation fails mid-way, compensating rollback logic is executed
    /// to undo already-applied operations in reverse order. However, this is
    /// <b>best-effort</b>: if an undo operation throws an exception, it is
    /// caught and swallowed, and rollback continues with remaining operations.
    /// </para>
    /// <para>
    /// This means that after a failed commit, the graph may be left in a
    /// partially modified state if undo operations also fail. For production
    /// use cases requiring strict atomicity, consider using a database-backed
    /// graph store with native transaction support.
    /// </para>
    /// </remarks>
    public void Commit()
    {
        EnsureActive();

        var appliedOperations = new List<TransactionOperation<T>>();

        try
        {
            // NOTE: We do NOT log operations here if the store is a FileGraphStore,
            // because FileGraphStore already logs to WAL internally when operations are applied.
            // Only MemoryGraphStore needs transaction-level WAL logging.
            bool storeHasOwnWal = _store is FileGraphStore<T>;

            // Log to WAL first (durability) - only for stores without their own WAL
            if (_wal != null && !storeHasOwnWal)
            {
                foreach (var op in _operations)
                {
                    LogOperation(op);
                }
            }

            // Apply all operations, tracking which ones succeed
            foreach (var op in _operations)
            {
                ApplyOperation(op);
                appliedOperations.Add(op);
            }

            // Checkpoint if using WAL
            _wal?.LogCheckpoint();

            _state = TransactionState.Committed;
        }
        catch (Exception)
        {
            // Compensating rollback: undo applied operations in reverse order
            for (int i = appliedOperations.Count - 1; i >= 0; i--)
            {
                try
                {
                    UndoOperation(appliedOperations[i]);
                }
                catch
                {
                    // Best-effort rollback - continue with remaining undo operations
                }
            }

            _state = TransactionState.Failed;
            throw;
        }
    }

    /// <summary>
    /// Rolls back the transaction, discarding all operations.
    /// </summary>
    public void Rollback()
    {
        if (_state != TransactionState.Active && _state != TransactionState.Failed)
            throw new InvalidOperationException($"Cannot rollback transaction in state: {_state}");

        // Simply discard operations (they were never applied)
        _operations.Clear();
        _state = TransactionState.RolledBack;
    }

    /// <summary>
    /// Ensures the transaction is in active state.
    /// </summary>
    private void EnsureActive()
    {
        if (_state != TransactionState.Active)
            throw new InvalidOperationException($"Transaction not active. Current state: {_state}");
    }

    /// <summary>
    /// Logs an operation to the WAL.
    /// </summary>
    private void LogOperation(TransactionOperation<T> op)
    {
        if (_wal == null)
            return;

        switch (op.Type)
        {
            case GraphOperationType.AddNode:
                _wal.LogAddNode(op.Node!);
                break;
            case GraphOperationType.AddEdge:
                _wal.LogAddEdge(op.Edge!);
                break;
            case GraphOperationType.RemoveNode:
                _wal.LogRemoveNode(op.NodeId!);
                break;
            case GraphOperationType.RemoveEdge:
                _wal.LogRemoveEdge(op.EdgeId!);
                break;
        }
    }

    /// <summary>
    /// Applies an operation to the graph store.
    /// </summary>
    private void ApplyOperation(TransactionOperation<T> op)
    {
        switch (op.Type)
        {
            case GraphOperationType.AddNode:
                if (op.Node != null)
                    _store.AddNode(op.Node);
                break;
            case GraphOperationType.AddEdge:
                if (op.Edge != null)
                    _store.AddEdge(op.Edge);
                break;
            case GraphOperationType.RemoveNode:
                if (op.NodeId != null)
                    _store.RemoveNode(op.NodeId);
                break;
            case GraphOperationType.RemoveEdge:
                if (op.EdgeId != null)
                    _store.RemoveEdge(op.EdgeId);
                break;
        }
    }

    /// <summary>
    /// Undoes an already-applied operation (compensating action).
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method performs the reverse of each operation type:
    /// - AddNode → RemoveNode (using the stored node's ID)
    /// - AddEdge → RemoveEdge (using the stored edge's ID)
    /// - RemoveNode → AddNode (re-adds the captured original node)
    /// - RemoveEdge → AddEdge (re-adds the captured original edge)
    /// </para>
    /// <para>
    /// The original node/edge data is captured during the RecordRemoveNode/RecordRemoveEdge
    /// operations, allowing complete restoration during undo.
    /// </para>
    /// </remarks>
    private void UndoOperation(TransactionOperation<T> op)
    {
        switch (op.Type)
        {
            case GraphOperationType.AddNode:
                // Undo add by removing the node
                if (op.Node != null)
                    _store.RemoveNode(op.Node.Id);
                break;
            case GraphOperationType.AddEdge:
                // Undo add by removing the edge
                if (op.Edge != null)
                    _store.RemoveEdge(op.Edge.Id);
                break;
            case GraphOperationType.RemoveNode:
                // Undo remove by re-adding the node (if we have the original data)
                if (op.Node != null)
                    _store.AddNode(op.Node);
                break;
            case GraphOperationType.RemoveEdge:
                // Undo remove by re-adding the edge (if we have the original data)
                if (op.Edge != null)
                    _store.AddEdge(op.Edge);
                break;
        }
    }

    /// <summary>
    /// Disposes the transaction, rolling back if still active.
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
            return;

        // Auto-rollback if still active
        if (_state == TransactionState.Active)
        {
            try
            {
                Rollback();
            }
            catch (InvalidOperationException ex)
            {
                // Rollback errors during dispose are suppressed per IDisposable contract,
                // but traced for diagnostics. Transaction may already be in invalid state.
                Debug.WriteLine($"GraphTransaction.Dispose: Rollback failed with InvalidOperationException: {ex.Message}");
            }
            catch (IOException ex)
            {
                // I/O errors during dispose are suppressed per IDisposable contract,
                // but traced for diagnostics. Underlying store may be unavailable.
                Debug.WriteLine($"GraphTransaction.Dispose: Rollback failed with IOException: {ex.Message}");
            }
        }

        _disposed = true;
    }
}

/// <summary>
/// Represents a single operation within a transaction.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
internal class TransactionOperation<T>
{
    public GraphOperationType Type { get; set; }
    public GraphNode<T>? Node { get; set; }
    public GraphEdge<T>? Edge { get; set; }
    public string? NodeId { get; set; }
    public string? EdgeId { get; set; }
}

/// <summary>
/// Types of operations supported in graph transactions.
/// </summary>
internal enum GraphOperationType
{
    AddNode,
    AddEdge,
    RemoveNode,
    RemoveEdge
}

/// <summary>
/// Represents the state of a transaction.
/// </summary>
public enum TransactionState
{
    /// <summary>
    /// Transaction has not been started.
    /// </summary>
    NotStarted,

    /// <summary>
    /// Transaction is active and accepting operations.
    /// </summary>
    Active,

    /// <summary>
    /// Transaction has been committed successfully.
    /// </summary>
    Committed,

    /// <summary>
    /// Transaction has been rolled back.
    /// </summary>
    RolledBack,

    /// <summary>
    /// Transaction failed during commit.
    /// </summary>
    Failed
}
