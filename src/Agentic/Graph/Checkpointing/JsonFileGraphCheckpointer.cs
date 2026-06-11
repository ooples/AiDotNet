using System.IO;
using Newtonsoft.Json;

namespace AiDotNet.Agentic.Graph.Checkpointing;

/// <summary>
/// A durable <see cref="IGraphCheckpointer{TState}"/> that persists all threads' checkpoints to a single
/// JSON file, so runs survive process restarts. Zero extra dependencies (uses Newtonsoft.Json).
/// </summary>
/// <typeparam name="TState">The graph's state type (must be JSON-serializable).</typeparam>
/// <remarks>
/// <para>
/// Simple and self-contained: every save reads, mutates, and rewrites the file under an in-process lock,
/// so it is correct for single-process durability but not tuned for high-throughput or multi-process
/// concurrent writers. For those, use a database-backed checkpointer (e.g., SQLite in
/// <c>AiDotNet.Storage.Sqlite</c>).
/// </para>
/// <para><b>For Beginners:</b> Saves your graph's progress to a file on disk, so if the app restarts you
/// can resume right where it left off.
/// </para>
/// </remarks>
public sealed class JsonFileGraphCheckpointer<TState> : IGraphCheckpointer<TState>
{
    private readonly object _gate = new();
    private readonly string _path;

    /// <summary>
    /// Initializes the checkpointer, creating the containing directory if needed.
    /// </summary>
    /// <param name="filePath">The JSON file used to persist checkpoints.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="filePath"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">Thrown when <paramref name="filePath"/> is empty/whitespace.</exception>
    public JsonFileGraphCheckpointer(string filePath)
    {
        Guard.NotNullOrWhiteSpace(filePath);
        _path = filePath;
        var dir = Path.GetDirectoryName(filePath);
        if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
        {
            Directory.CreateDirectory(dir);
        }
    }

    /// <inheritdoc/>
    public Task SaveAsync(GraphCheckpoint<TState> checkpoint, CancellationToken cancellationToken = default)
    {
        Guard.NotNull(checkpoint);
        lock (_gate)
        {
            var data = Load();
            if (!data.TryGetValue(checkpoint.ThreadId, out var list))
            {
                list = new List<GraphCheckpoint<TState>>();
                data[checkpoint.ThreadId] = list;
            }

            list.Add(checkpoint);
            Persist(data);
        }

        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    public Task<GraphCheckpoint<TState>?> GetLatestAsync(string threadId, CancellationToken cancellationToken = default)
    {
        Guard.NotNull(threadId);
        lock (_gate)
        {
            var data = Load();
            if (data.TryGetValue(threadId, out var list) && list.Count > 0)
            {
                return Task.FromResult<GraphCheckpoint<TState>?>(list[list.Count - 1]);
            }
        }

        return Task.FromResult<GraphCheckpoint<TState>?>(null);
    }

    /// <inheritdoc/>
    public Task<GraphCheckpoint<TState>?> GetAsync(string threadId, string checkpointId, CancellationToken cancellationToken = default)
    {
        Guard.NotNull(threadId);
        Guard.NotNull(checkpointId);
        lock (_gate)
        {
            var data = Load();
            if (data.TryGetValue(threadId, out var list))
            {
                foreach (var cp in list)
                {
                    if (cp.CheckpointId == checkpointId)
                    {
                        return Task.FromResult<GraphCheckpoint<TState>?>(cp);
                    }
                }
            }
        }

        return Task.FromResult<GraphCheckpoint<TState>?>(null);
    }

    /// <inheritdoc/>
    public Task<IReadOnlyList<GraphCheckpoint<TState>>> GetHistoryAsync(string threadId, CancellationToken cancellationToken = default)
    {
        Guard.NotNull(threadId);
        lock (_gate)
        {
            var data = Load();
            IReadOnlyList<GraphCheckpoint<TState>> history = data.TryGetValue(threadId, out var list)
                ? list.ToList()
                : new List<GraphCheckpoint<TState>>();
            return Task.FromResult(history);
        }
    }

    private Dictionary<string, List<GraphCheckpoint<TState>>> Load()
    {
        if (!File.Exists(_path))
        {
            return new Dictionary<string, List<GraphCheckpoint<TState>>>(StringComparer.Ordinal);
        }

        var json = File.ReadAllText(_path);
        if (string.IsNullOrWhiteSpace(json))
        {
            return new Dictionary<string, List<GraphCheckpoint<TState>>>(StringComparer.Ordinal);
        }

        try
        {
            var data = JsonConvert.DeserializeObject<Dictionary<string, List<GraphCheckpoint<TState>>>>(json);
            return data ?? new Dictionary<string, List<GraphCheckpoint<TState>>>(StringComparer.Ordinal);
        }
        catch (JsonException ex)
        {
            // Surface corruption explicitly rather than silently returning empty — silently dropping the
            // history would let the next Save overwrite (and permanently lose) recoverable checkpoint data.
            throw new InvalidOperationException(
                $"The checkpoint file at '{_path}' is corrupt and could not be parsed. " +
                "Move or delete it to start a fresh checkpoint store.", ex);
        }
    }

    private void Persist(Dictionary<string, List<GraphCheckpoint<TState>>> data)
    {
        var json = JsonConvert.SerializeObject(data, Formatting.None);

        // Atomic write: serialize to a temp file in the same directory, flush it to disk, then atomically
        // replace the target. A crash mid-write therefore leaves the previous (valid) file intact rather than
        // a half-written, unparseable one.
        var directory = Path.GetDirectoryName(_path);
        var tempPath = Path.Combine(
            string.IsNullOrEmpty(directory) ? "." : directory,
            $".{Path.GetFileName(_path)}.{Guid.NewGuid():N}.tmp");

        try
        {
            using (var stream = new FileStream(tempPath, FileMode.Create, FileAccess.Write, FileShare.None))
            using (var writer = new StreamWriter(stream))
            {
                writer.Write(json);
                writer.Flush();
                stream.Flush(flushToDisk: true);
            }

#if NET5_0_OR_GREATER
            File.Move(tempPath, _path, overwrite: true);
#else
            if (File.Exists(_path))
            {
                // File.Replace is atomic when the destination exists; it also clears the temp file.
                File.Replace(tempPath, _path, destinationBackupFileName: null);
            }
            else
            {
                File.Move(tempPath, _path);
            }
#endif
        }
        finally
        {
            if (File.Exists(tempPath))
            {
                try
                {
                    File.Delete(tempPath);
                }
                catch (IOException)
                {
                    // Best-effort cleanup of the temp file; a leftover .tmp is harmless.
                }
            }
        }
    }
}
