using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using AiDotNet.Attributes;
using AiDotNet.Data;
using AiDotNet.Enums;
using AiDotNet.Validation;

namespace AiDotNet.RetrievalAugmentedGeneration.Graph;

/// <summary>
/// Simple file-based index for mapping string keys to file offsets.
/// </summary>
/// <remarks>
/// <para>
/// This class provides a persistent index structure that maps string keys (e.g., node IDs)
/// to byte offsets in data files. The index is stored on disk and reloaded on restart,
/// enabling fast lookups without scanning entire data files.
/// </para>
/// <para><b>For Beginners:</b> Think of this like a book's index at the back.
///
/// Without an index:
/// - To find "photosynthesis", you'd read every page from start to finish
/// - Very slow for large books (or large data files)
///
/// With an index:
/// - Look up "photosynthesis" → find it's on page 157
/// - Jump directly to page 157
/// - Much faster!
///
/// This class does the same for graph data:
/// - Key: "node_alice_001"
/// - Value: byte offset 45678 in nodes.dat file
/// - We can jump directly to byte 45678 to read Alice's data
///
/// The index itself is stored in a file so it survives application restarts.
/// </para>
/// <para>
/// <b>Implementation Note:</b> This is a simplified index using a sorted dictionary.
/// For production systems with millions of entries, consider implementing a true
/// B-Tree structure with splitting/merging nodes, or use an embedded database like
/// SQLite or LevelDB.
/// </para>
/// </remarks>
[ComponentType(ComponentType.DocumentStore)]
[PipelineStage(PipelineStage.Indexing)]
public class BTreeIndex : IDisposable
{
    private readonly string _indexFilePath;
    private readonly SortedDictionary<string, long> _index;
    private bool _isDirty;
    private bool _disposed;

    /// <summary>
    /// Gets the number of entries in the index.
    /// </summary>
    public int Count => _index.Count;

    /// <summary>
    /// Initializes a new instance of the <see cref="BTreeIndex"/> class.
    /// </summary>
    /// <param name="indexFilePath">The path to the index file on disk.</param>
    public BTreeIndex(string indexFilePath)
    {
        Guard.NotNull(indexFilePath);
        _indexFilePath = indexFilePath;
        _index = new SortedDictionary<string, long>();
        _isDirty = false;

        LoadFromDisk();
    }

    /// <summary>
    /// Adds or updates a key-offset mapping in the index.
    /// </summary>
    /// <param name="key">The key to index (e.g., node ID).</param>
    /// <param name="offset">The byte offset in the data file.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This adds an entry to the index.
    ///
    /// Example:
    /// - index.Add("alice", 1024)
    /// - This means: "Alice's data starts at byte 1024 in the file"
    ///
    /// If "alice" already exists, it updates to the new offset.
    /// The index is marked as "dirty" and will be saved to disk later.
    /// </para>
    /// </remarks>
    public void Add(string key, long offset)
    {
        if (string.IsNullOrWhiteSpace(key))
            throw new ArgumentException("Key cannot be null or whitespace", nameof(key));
        if (offset < 0)
            throw new ArgumentException("Offset cannot be negative", nameof(offset));

        _index[key] = offset;
        _isDirty = true;
    }

    /// <summary>
    /// Retrieves the file offset associated with a key.
    /// </summary>
    /// <param name="key">The key to look up.</param>
    /// <returns>The byte offset if found; otherwise, -1.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This looks up where data is stored.
    ///
    /// Example:
    /// - var offset = index.Get("alice")
    /// - Returns: 1024 (meaning Alice's data is at byte 1024)
    /// - Or returns -1 if "alice" is not in the index
    /// </para>
    /// </remarks>
    public long Get(string key)
    {
        if (string.IsNullOrWhiteSpace(key))
            return -1;

        return _index.TryGetValue(key, out var offset) ? offset : -1;
    }

    /// <summary>
    /// Checks if a key exists in the index.
    /// </summary>
    /// <param name="key">The key to check.</param>
    /// <returns>True if the key exists; otherwise, false.</returns>
    public bool Contains(string key)
    {
        return !string.IsNullOrWhiteSpace(key) && _index.ContainsKey(key);
    }

    /// <summary>
    /// Removes a key from the index.
    /// </summary>
    /// <param name="key">The key to remove.</param>
    /// <returns>True if the key was found and removed; otherwise, false.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This removes an entry from the index.
    ///
    /// Example:
    /// - index.Remove("alice")
    /// - Now we can no longer look up where Alice's data is
    /// - The actual data file is NOT modified - just the index
    ///
    /// Note: For production systems, you'd typically mark entries as deleted
    /// rather than removing them, to avoid fragmenting the data file.
    /// </para>
    /// </remarks>
    public bool Remove(string key)
    {
        if (string.IsNullOrWhiteSpace(key))
            return false;

        var removed = _index.Remove(key);
        if (removed)
            _isDirty = true;

        return removed;
    }

    /// <summary>
    /// Gets all keys in the index.
    /// </summary>
    /// <returns>Collection of all keys.</returns>
    public IEnumerable<string> GetAllKeys()
    {
        return _index.Keys.ToList();
    }

    /// <summary>
    /// Removes all entries from the index.
    /// </summary>
    public void Clear()
    {
        _index.Clear();
        _isDirty = true;
    }

    /// <summary>
    /// Saves the index to disk if it has been modified.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method writes the index to disk in a simple text format:
    /// - Each line: "key|offset"
    /// - Example: "alice|1024"
    ///
    /// The file is only written if changes have been made (_isDirty flag).
    /// This is called automatically by Dispose() to ensure data isn't lost.
    /// </para>
    /// <para><b>For Beginners:</b> This saves the index to a file.
    ///
    /// Think of it like saving your work:
    /// - You've been adding entries to the index (in memory)
    /// - Flush() writes everything to disk
    /// - Now the index survives if the program crashes or restarts
    ///
    /// Format example (nodes_index.db):
    /// ```
    /// alice|1024
    /// bob|2048
    /// charlie|3072
    /// ```
    /// </para>
    /// </remarks>
    public void Flush()
    {
        if (!_isDirty)
            return;

        try
        {
            // Ensure directory exists
            var directory = Path.GetDirectoryName(_indexFilePath);
            if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
                Directory.CreateDirectory(directory);

            // Write index to temporary file first (atomic write)
            var tempPath = _indexFilePath + ".tmp";
            using (var writer = new StreamWriter(tempPath, false, Encoding.UTF8))
            {
                foreach (var kvp in _index)
                {
                    writer.WriteLine($"{kvp.Key}|{kvp.Value}");
                }
            }

            // Replace old index file with new one atomically. Both Move
            // and Replace can fail with sharing violations on Windows when
            // Defender or the indexer is mid-scan; RobustFileOps retries
            // with linear backoff on transient IOException.
            if (File.Exists(_indexFilePath))
            {
                var backupPath = _indexFilePath + ".bak";
                RobustFileOps.ReplaceWithRetry(tempPath, _indexFilePath, backupPath);

                // The replace has already succeeded — the new index is
                // durable at _indexFilePath. Backup cleanup is best-effort:
                // if the .bak file hits the same antivirus / indexer lock
                // this PR addresses elsewhere, swallow the failure rather
                // than throw from Flush() and leave _isDirty set when the
                // actual flush was successful.
                try
                {
                    if (File.Exists(backupPath))
                        File.Delete(backupPath);
                }
                catch (IOException) { /* leave stale .bak; next flush reuses it */ }
                catch (UnauthorizedAccessException) { /* same */ }
            }
            else
            {
                RobustFileOps.MoveWithRetry(tempPath, _indexFilePath);
            }

            _isDirty = false;
        }
        catch (IOException ex)
        {
            throw new IOException($"Failed to flush index to disk: {_indexFilePath}", ex);
        }
        catch (UnauthorizedAccessException ex)
        {
            throw new IOException($"Failed to flush index to disk: {_indexFilePath}", ex);
        }
    }

    /// <summary>
    /// Loads the index from disk if it exists.
    /// </summary>
    private void LoadFromDisk()
    {
        if (!File.Exists(_indexFilePath))
            return;

        try
        {
            using var reader = new StreamReader(_indexFilePath, Encoding.UTF8);
            string? line;
            while ((line = reader.ReadLine()) != null)
            {
                var parts = line.Split('|');
                if (parts.Length != 2)
                    continue;

                var key = parts[0];
                if (long.TryParse(parts[1], out var offset))
                {
                    _index[key] = offset;
                }
            }

            _isDirty = false;
        }
        catch (IOException ex)
        {
            throw new IOException($"Failed to load index from disk: {_indexFilePath}", ex);
        }
        catch (UnauthorizedAccessException ex)
        {
            throw new IOException($"Failed to load index from disk: {_indexFilePath}", ex);
        }
    }

    /// <summary>
    /// Disposes the index, ensuring all changes are saved to disk.
    /// </summary>
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Releases resources used by the index.
    /// </summary>
    /// <param name="disposing">True if called from Dispose(), false if called from finalizer.</param>
    protected virtual void Dispose(bool disposing)
    {
        if (_disposed)
            return;

        try
        {
            if (disposing)
            {
                // Flush managed resources
                Flush();
            }
        }
        finally
        {
            // Ensure _disposed is set even if Flush throws
            _disposed = true;
        }
    }
}
