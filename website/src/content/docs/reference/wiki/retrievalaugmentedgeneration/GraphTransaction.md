---
title: "GraphTransaction<T>"
description: "Transaction coordinator for managing transactions on graph stores with best-effort rollback."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.Graph`

Transaction coordinator for managing transactions on graph stores with best-effort rollback.

## For Beginners

Transactions ensure your changes are safe.

Think of a bank transfer:

- Debit $100 from Alice
- Credit $100 to Bob

Without transactions:

- If crash happens after debit but before credit, $100 disappears!

With transactions:

- Begin transaction
- Debit Alice
- Credit Bob
- Commit (both succeed) OR Rollback (both undone)
- Money never disappears!

In graphs:
```cs
var txn = new GraphTransaction(store, wal);
txn.Begin();
try
{
txn.AddNode(node1);
txn.AddEdge(edge1);
txn.Commit(); // Both saved
}
catch
{
txn.Rollback(); // Both undone
}
```

This ensures your graph is never in a broken state!

## How It Works

This class provides transaction management with the following guarantees:

- **Atomicity (Best-Effort)**: If an operation fails during commit, compensating rollback

is attempted in reverse order. However, if an undo operation fails, it is swallowed and
rollback continues with remaining operations. Full atomicity is not guaranteed.

- **Consistency**: Graph validation rules are enforced during operations.
- **Isolation**: Single-threaded; no concurrent transaction support.
- **Durability**: When a WAL is provided, operations are logged before execution.

Without a WAL, durability is not guaranteed.

**Important:** This is a lightweight transaction implementation suitable for single-process
use cases. For full ACID compliance with crash recovery, ensure a WriteAheadLog is configured.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GraphTransaction(IGraphStore<>,WriteAheadLog)` | Initializes a new instance of the `GraphTransaction` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `State` | Gets the current state of the transaction. |
| `TransactionId` | Gets the transaction ID. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddEdge(GraphEdge<>)` | Adds an edge within the transaction. |
| `AddNode(GraphNode<>)` | Adds a node within the transaction. |
| `ApplyOperation(TransactionOperation<>)` | Applies an operation to the graph store. |
| `Begin` | Begins a new transaction. |
| `Commit` | Commits the transaction, applying all operations with best-effort atomicity. |
| `Dispose` | Disposes the transaction, rolling back if still active. |
| `EnsureActive` | Ensures the transaction is in active state. |
| `LogOperation(TransactionOperation<>)` | Logs an operation to the WAL. |
| `RemoveEdge(String)` | Removes an edge within the transaction. |
| `RemoveNode(String)` | Removes a node within the transaction. |
| `Rollback` | Rolls back the transaction, discarding all operations. |
| `UndoOperation(TransactionOperation<>)` | Undoes an already-applied operation (compensating action). |

