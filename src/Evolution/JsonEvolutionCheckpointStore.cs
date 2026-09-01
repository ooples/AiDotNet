using System.Text;
using AiDotNet.Interfaces;
using AiDotNet.Validation;
using Newtonsoft.Json;

namespace AiDotNet.Evolution;

/// <summary>
/// Persists one evolution run to an atomic JSON file, retaining the immediately previous valid snapshot.
/// </summary>
public sealed class JsonEvolutionCheckpointStore : IEvolutionCheckpointStore
{
    private readonly object _gate = new();
    private readonly string _path;
    private readonly string _previousPath;
    private readonly long _maxCheckpointBytes;

    /// <summary>Initializes a file-backed store.</summary>
    /// <param name="filePath">The primary checkpoint JSON path.</param>
    /// <param name="maxCheckpointBytes">Maximum encoded JSON size accepted for one checkpoint.</param>
    public JsonEvolutionCheckpointStore(string filePath, long maxCheckpointBytes = 64L * 1024L * 1024L)
    {
        Guard.NotNullOrWhiteSpace(filePath);
        if (maxCheckpointBytes <= 0) throw new ArgumentOutOfRangeException(nameof(maxCheckpointBytes));
        _path = Path.GetFullPath(filePath);
        _previousPath = _path + ".previous";
        _maxCheckpointBytes = maxCheckpointBytes;
        string? directory = Path.GetDirectoryName(_path);
        if (!string.IsNullOrEmpty(directory)) Directory.CreateDirectory(directory);
    }

    /// <inheritdoc/>
    public Task SaveAsync(EvolutionCheckpoint checkpoint, CancellationToken cancellationToken = default)
    {
        Guard.NotNull(checkpoint);
        cancellationToken.ThrowIfCancellationRequested();
        checkpoint.Validate();
        lock (_gate)
        {
            cancellationToken.ThrowIfCancellationRequested();
            EvolutionCheckpoint? existing = TryLoad(_path, checkpoint.RunId);
            if (existing is null) existing = TryLoad(_previousPath, checkpoint.RunId);
            if (existing is not null)
            {
                ValidateSuccessor(existing, checkpoint);
                if (checkpoint.Sequence == existing.Sequence) return Task.CompletedTask;
            }
            Persist(checkpoint, cancellationToken);
        }
        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    public Task<EvolutionCheckpoint?> LoadLatestAsync(string runId, CancellationToken cancellationToken = default)
    {
        Guard.NotNullOrWhiteSpace(runId);
        cancellationToken.ThrowIfCancellationRequested();
        lock (_gate)
        {
            cancellationToken.ThrowIfCancellationRequested();
            try
            {
                EvolutionCheckpoint? primary = TryLoad(_path, runId);
                if (primary is not null) return Task.FromResult<EvolutionCheckpoint?>(primary);
            }
            catch (InvalidDataException) when (File.Exists(_previousPath))
            {
                EvolutionCheckpoint? previous = TryLoad(_previousPath, runId);
                return Task.FromResult<EvolutionCheckpoint?>(previous);
            }
            EvolutionCheckpoint? fallback = TryLoad(_previousPath, runId);
            return Task.FromResult<EvolutionCheckpoint?>(fallback);
        }
    }

    private void Persist(EvolutionCheckpoint checkpoint, CancellationToken cancellationToken)
    {
        string directory = Path.GetDirectoryName(_path) ?? ".";
        string tempPath = Path.Combine(directory, $".{Path.GetFileName(_path)}.{Guid.NewGuid():N}.tmp");
        string json = JsonConvert.SerializeObject(CheckpointDocument.From(checkpoint), Formatting.Indented);
        byte[] payload = new UTF8Encoding(encoderShouldEmitUTF8Identifier: false).GetBytes(json);
        if (payload.LongLength > _maxCheckpointBytes)
            throw new InvalidDataException($"The evolution checkpoint exceeds the {_maxCheckpointBytes}-byte limit.");
        try
        {
            using (var stream = new FileStream(tempPath, FileMode.CreateNew, FileAccess.Write, FileShare.None))
            {
                cancellationToken.ThrowIfCancellationRequested();
                stream.Write(payload, 0, payload.Length);
                stream.Flush(flushToDisk: true);
            }

            cancellationToken.ThrowIfCancellationRequested();
            if (File.Exists(_path))
            {
                File.Replace(tempPath, _path, _previousPath, ignoreMetadataErrors: true);
            }
            else
            {
                File.Move(tempPath, _path);
            }
        }
        finally
        {
            if (File.Exists(tempPath))
            {
                try { File.Delete(tempPath); }
                catch (IOException) { }
            }
        }
    }

    private EvolutionCheckpoint? TryLoad(string path, string runId)
    {
        if (!File.Exists(path)) return null;
        try
        {
            if (new FileInfo(path).Length > _maxCheckpointBytes)
                throw new InvalidDataException($"The evolution checkpoint exceeds the {_maxCheckpointBytes}-byte limit.");
            string json;
            using (var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                if (stream.Length > _maxCheckpointBytes)
                    throw new InvalidDataException($"The evolution checkpoint exceeds the {_maxCheckpointBytes}-byte limit.");
                using var reader = new StreamReader(stream, Encoding.UTF8, detectEncodingFromByteOrderMarks: true);
                json = reader.ReadToEnd();
            }
            if (Encoding.UTF8.GetByteCount(json) > _maxCheckpointBytes)
                throw new InvalidDataException($"The evolution checkpoint exceeds the {_maxCheckpointBytes}-byte limit.");
            CheckpointDocument? document = JsonConvert.DeserializeObject<CheckpointDocument>(json);
            if (document is null) throw new InvalidDataException("The evolution checkpoint document is empty.");
            document.Validate();
            EvolutionCheckpoint checkpoint = document.ToCheckpoint();
            checkpoint.Validate();
            if (!string.Equals(checkpoint.RunId, runId, StringComparison.Ordinal))
                throw new InvalidDataException("The evolution checkpoint belongs to a different run.");
            return checkpoint;
        }
        catch (JsonException exception)
        {
            throw new InvalidDataException("The evolution checkpoint JSON is invalid.", exception);
        }
    }

    private static void ValidateSuccessor(EvolutionCheckpoint existing, EvolutionCheckpoint checkpoint)
    {
        if (!string.Equals(existing.CompatibilityHash, checkpoint.CompatibilityHash, StringComparison.Ordinal))
            throw new InvalidOperationException("A checkpoint run cannot change compatibility identity.");
        if (checkpoint.Sequence < existing.Sequence)
            throw new InvalidOperationException("A checkpoint store cannot move a run backwards.");
        if (checkpoint.Sequence == existing.Sequence &&
            !string.Equals(existing.Checksum, checkpoint.Checksum, StringComparison.Ordinal))
        {
            throw new InvalidOperationException("A checkpoint sequence cannot identify two different states.");
        }
    }

    private sealed class CheckpointDocument
    {
        public int SchemaVersion { get; set; }
        public string RunId { get; set; } = string.Empty;
        public long Sequence { get; set; }
        public string CompatibilityHash { get; set; } = string.Empty;
        public string Payload { get; set; } = string.Empty;
        public string Checksum { get; set; } = string.Empty;
        public string DocumentChecksum { get; set; } = string.Empty;

        public static CheckpointDocument From(EvolutionCheckpoint checkpoint)
        {
            var document = new CheckpointDocument
            {
                SchemaVersion = checkpoint.SchemaVersion,
                RunId = checkpoint.RunId,
                Sequence = checkpoint.Sequence,
                CompatibilityHash = checkpoint.CompatibilityHash,
                Payload = checkpoint.Payload,
                Checksum = checkpoint.Checksum
            };
            document.DocumentChecksum = document.ComputeDocumentChecksum();
            return document;
        }

        public void Validate()
        {
            if (!string.Equals(DocumentChecksum, ComputeDocumentChecksum(), StringComparison.Ordinal))
                throw new InvalidDataException("The evolution checkpoint document checksum validation failed.");
        }

        private string ComputeDocumentChecksum() => EvolutionHash.Combine(new[]
        {
            SchemaVersion.ToString(System.Globalization.CultureInfo.InvariantCulture),
            RunId,
            Sequence.ToString(System.Globalization.CultureInfo.InvariantCulture),
            CompatibilityHash,
            Payload,
            Checksum
        });

        public EvolutionCheckpoint ToCheckpoint() => new(RunId, Sequence, CompatibilityHash, Payload, Checksum, SchemaVersion);
    }
}
