---
title: "WriteAheadLog"
description: "Write-Ahead Log (WAL) for ensuring ACID properties and crash recovery."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.Graph`

Write-Ahead Log (WAL) for ensuring ACID properties and crash recovery.

## For Beginners

Think of WAL like a ship's log or diary.

Before making any change to your graph:

1. Write what you're about to do in the log (WAL)
2. Make sure the log is saved to disk
3. Then make the actual change

If the system crashes:

- The log shows what was happening
- You can replay the log to restore the graph
- No data is lost!

This is how databases ensure "durability" - the D in ACID.

Real-world analogy:

- Bank transaction: First log "transfer $100", then move the money
- If crash happens after logging but before transfer, replay the log on restart
- Money isn't lost!

## How It Works

A Write-Ahead Log records all changes before they're applied to the main data files.
This ensures data integrity and enables recovery after crashes.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `WriteAheadLog(String)` | Initializes a new instance of the `WriteAheadLog` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CurrentTransactionId` | Gets the current transaction ID. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Dispose` | Disposes the WAL, ensuring all entries are flushed. |
| `LogAddEdge(GraphEdge<>)` | Logs an edge addition operation. |
| `LogAddNode(GraphNode<>)` | Logs a node addition operation. |
| `LogCheckpoint` | Logs a checkpoint (all data successfully persisted to disk). |
| `LogRemoveEdge(String)` | Logs an edge removal operation. |
| `LogRemoveNode(String)` | Logs a node removal operation. |
| `ReadLog` | Reads all WAL entries from the log file. |
| `RestoreLastTransactionId` | Restores the last transaction ID from an existing WAL file. |
| `Truncate` | Truncates the WAL after a successful checkpoint. |
| `WriteEntry(WALEntry)` | Writes a WAL entry to the log file. |

