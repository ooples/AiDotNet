using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Newtonsoft.Json;
using AiDotNet.Validation;

namespace AiDotNet.RetrievalAugmentedGeneration.Graph;

/// <summary>
/// Write-Ahead Log (WAL) for ensuring ACID properties and crash recovery.
/// </summary>
/// <remarks>
/// <para>
/// A Write-Ahead Log records all changes before they're applied to the main data files.
/// This ensures data integrity and enables recovery after crashes.
/// </para>
/// <para><b>For Beginners:</b> Think of WAL like a ship's log or diary.
///
/// Before making any change to your graph:
/// 1. Write what you're about to do in the log (WAL)
/// 2. Make sure the log is saved to disk
/// 3. Then make the actual change
///
/// If the system crashes:
/// - The log shows what was happening
/// - You can replay the log to restore the graph
/// - No data is lost!
///
/// This is how databases ensure "durability" - the D in ACID.
///
/// Real-world analogy:
/// - Bank transaction: First log "transfer $100", then move the money
/// - If crash happens after logging but before transfer, replay the log on restart
/// - Money isn't lost!
/// </para>
/// </remarks>
public class WriteAheadLog : IDisposable
{
    private readonly string _walFilePath;
    private StreamWriter? _writer;
    private long _currentTransactionId;
    private readonly object _lock = new object();
    private bool _disposed;

    /// <summary>
    /// Gets the current transaction ID.
    /// </summary>
    public long CurrentTransactionId => _currentTransactionId;

    /// <summary>
    /// Initializes a new instance of the <see cref="WriteAheadLog"/> class.
    /// </summary>
    /// <param name="walFilePath">Path to the WAL file.</param>
    public WriteAheadLog(string walFilePath)
    {
        Guard.NotNull(walFilePath);
        _walFilePath = walFilePath;

        // Ensure directory exists
        var directory = Path.GetDirectoryName(walFilePath);
        if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
            Directory.CreateDirectory(directory);

        // Restore transaction ID from existing log to prevent duplicates after restart
        _currentTransactionId = RestoreLastTransactionId();

        // Open WAL file for append
        _writer = new StreamWriter(_walFilePath, append: true, Encoding.UTF8)
        {
            AutoFlush = true // Critical: flush immediately for durability
        };
    }

    /// <summary>
    /// Restores the last transaction ID from an existing WAL file.
    /// </summary>
    /// <returns>The maximum transaction ID found in the log, or 0 if no log exists.</returns>
    private long RestoreLastTransactionId()
    {
        if (!File.Exists(_walFilePath))
            return 0;

        long maxId = 0;
        try
        {
            foreach (var line in File.ReadLines(_walFilePath))
            {
                try
                {
                    var entry = JsonConvert.DeserializeObject<WALEntry>(line);
                    if (entry != null && entry.TransactionId > maxId)
                        maxId = entry.TransactionId;
                }
                catch (JsonException)
                {
                    // Skip malformed lines
                }
            }
        }
        catch (IOException)
        {
            // If we can't read the file, start from 0
            return 0;
        }
        return maxId;
    }

    /// <summary>
    /// Logs a node addition operation.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="node">The node being added.</param>
    /// <returns>The transaction ID for this operation.</returns>
    public long LogAddNode<T>(GraphNode<T> node)
    {
        lock (_lock)
        {
            var txnId = ++_currentTransactionId;
            var entry = new WALEntry
            {
                TransactionId = txnId,
                Timestamp = DateTime.UtcNow,
                OperationType = WALOperationType.AddNode,
                NodeId = node.Id,
                Data = JsonConvert.SerializeObject(node)
            };

            WriteEntry(entry);
            return txnId;
        }
    }

    /// <summary>
    /// Logs an edge addition operation.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="edge">The edge being added.</param>
    /// <returns>The transaction ID for this operation.</returns>
    public long LogAddEdge<T>(GraphEdge<T> edge)
    {
        lock (_lock)
        {
            var txnId = ++_currentTransactionId;
            var entry = new WALEntry
            {
                TransactionId = txnId,
                Timestamp = DateTime.UtcNow,
                OperationType = WALOperationType.AddEdge,
                EdgeId = edge.Id,
                Data = JsonConvert.SerializeObject(edge)
            };

            WriteEntry(entry);
            return txnId;
        }
    }

    /// <summary>
    /// Logs a node removal operation.
    /// </summary>
    /// <param name="nodeId">The ID of the node being removed.</param>
    /// <returns>The transaction ID for this operation.</returns>
    public long LogRemoveNode(string nodeId)
    {
        lock (_lock)
        {
            var txnId = ++_currentTransactionId;
            var entry = new WALEntry
            {
                TransactionId = txnId,
                Timestamp = DateTime.UtcNow,
                OperationType = WALOperationType.RemoveNode,
                NodeId = nodeId
            };

            WriteEntry(entry);
            return txnId;
        }
    }

    /// <summary>
    /// Logs an edge removal operation.
    /// </summary>
    /// <param name="edgeId">The ID of the edge being removed.</param>
    /// <returns>The transaction ID for this operation.</returns>
    public long LogRemoveEdge(string edgeId)
    {
        lock (_lock)
        {
            var txnId = ++_currentTransactionId;
            var entry = new WALEntry
            {
                TransactionId = txnId,
                Timestamp = DateTime.UtcNow,
                OperationType = WALOperationType.RemoveEdge,
                EdgeId = edgeId
            };

            WriteEntry(entry);
            return txnId;
        }
    }

    /// <summary>
    /// Logs a checkpoint (all data successfully persisted to disk).
    /// </summary>
    /// <returns>The transaction ID for this checkpoint.</returns>
    public long LogCheckpoint()
    {
        lock (_lock)
        {
            var txnId = ++_currentTransactionId;
            var entry = new WALEntry
            {
                TransactionId = txnId,
                Timestamp = DateTime.UtcNow,
                OperationType = WALOperationType.Checkpoint
            };

            WriteEntry(entry);
            return txnId;
        }
    }

    /// <summary>
    /// Reads all WAL entries from the log file.
    /// </summary>
    /// <returns>List of WAL entries in order.</returns>
    public List<WALEntry> ReadLog()
    {
        var entries = new List<WALEntry>();

        if (!File.Exists(_walFilePath))
            return entries;

        lock (_lock)
        {
            // Flush writer to ensure all entries are on disk
            _writer?.Flush();

            // Use FileShare.ReadWrite to allow reading while writer is still open (Windows compatibility)
            using var fileStream = new FileStream(_walFilePath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
            using var reader = new StreamReader(fileStream, Encoding.UTF8);
            string? line;
            while ((line = reader.ReadLine()) != null)
            {
                try
                {
                    var entry = JsonConvert.DeserializeObject<WALEntry>(line);
                    if (entry != null)
                        entries.Add(entry);
                }
                catch (JsonSerializationException)
                {
                    // Skip corrupted entries
                }
            }
        }

        return entries;
    }

    /// <summary>
    /// Truncates the WAL after a successful checkpoint.
    /// </summary>
    /// <remarks>
    /// This removes old entries that have been successfully applied,
    /// keeping the WAL file from growing indefinitely.
    /// </remarks>
    public void Truncate()
    {
        lock (_lock)
        {
            _writer?.Close();
            _writer?.Dispose();

            if (File.Exists(_walFilePath))
                File.Delete(_walFilePath);

            _writer = new StreamWriter(_walFilePath, append: true, Encoding.UTF8)
            {
                AutoFlush = true
            };

            _currentTransactionId = 0;
        }
    }

    /// <summary>
    /// Writes a WAL entry to the log file.
    /// </summary>
    private void WriteEntry(WALEntry entry)
    {
        var json = JsonConvert.SerializeObject(entry);
        _writer?.WriteLine(json);
        // AutoFlush ensures it's written to disk immediately
    }

    /// <summary>
    /// Disposes the WAL, ensuring all entries are flushed.
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
            return;

        lock (_lock)
        {
            _writer?.Flush();
            _writer?.Close();
            _writer?.Dispose();
            _disposed = true;
        }
    }
}

/// <summary>
/// Represents a single entry in the Write-Ahead Log.
/// </summary>
public class WALEntry
{
    /// <summary>
    /// Gets or sets the transaction ID.
    /// </summary>
    public long TransactionId { get; set; }

    /// <summary>
    /// Gets or sets the timestamp of the operation.
    /// </summary>
    public DateTime Timestamp { get; set; }

    /// <summary>
    /// Gets or sets the type of operation.
    /// </summary>
    public WALOperationType OperationType { get; set; }

    /// <summary>
    /// Gets or sets the node ID (for node operations).
    /// </summary>
    public string? NodeId { get; set; }

    /// <summary>
    /// Gets or sets the edge ID (for edge operations).
    /// </summary>
    public string? EdgeId { get; set; }

    /// <summary>
    /// Gets or sets the serialized data for the operation.
    /// </summary>
    public string? Data { get; set; }
}

/// <summary>
/// Types of operations that can be logged in the WAL.
/// </summary>
public enum WALOperationType
{
    /// <summary>
    /// Add a node to the graph.
    /// </summary>
    AddNode,

    /// <summary>
    /// Add an edge to the graph.
    /// </summary>
    AddEdge,

    /// <summary>
    /// Remove a node from the graph.
    /// </summary>
    RemoveNode,

    /// <summary>
    /// Remove an edge from the graph.
    /// </summary>
    RemoveEdge,

    /// <summary>
    /// Checkpoint - all operations up to this point are persisted.
    /// </summary>
    Checkpoint,

    /// <summary>
    /// Begin a transaction.
    /// </summary>
    BeginTransaction,

    /// <summary>
    /// Commit a transaction.
    /// </summary>
    CommitTransaction,

    /// <summary>
    /// Rollback a transaction.
    /// </summary>
    RollbackTransaction
}
