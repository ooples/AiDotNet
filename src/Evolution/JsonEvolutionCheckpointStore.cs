using System.Text;
using AiDotNet.Interfaces;
using AiDotNet.Validation;
using Newtonsoft.Json;

namespace AiDotNet.Evolution;

/// <summary>
/// Persists one evolution run to an atomic JSON file, retaining the immediately previous valid snapshot.
/// </summary>
/// <remarks>
/// <para>
/// Each save serializes the checkpoint with Newtonsoft.Json into a temporary file in the target directory,
/// flushes it to disk, and then swaps it into place with <c>File.Replace</c>, which also moves the prior file to
/// <c>&lt;path&gt;.previous</c>. A crash at any point therefore leaves either the old snapshot or the new one
/// intact, never a torn file. Every document carries a checksum over its own fields in addition to the payload
/// checksum inside <see cref="EvolutionCheckpoint"/>; loads verify both, reject files larger than the configured
/// byte limit, and fall back to the previous snapshot when the primary file is corrupt. Saves also enforce that a
/// run never changes its compatibility hash or moves its sequence backwards, and a save whose sequence and
/// checksum match the stored snapshot is a no-op. All operations are serialized through one lock, so a single
/// instance may be shared by concurrent callers; the asynchronous signatures complete synchronously.
/// </para>
/// <para><b>For Beginners:</b> A checkpoint is a save file for an evolution run: if the process stops, the run can
/// resume from its last committed state instead of starting over. This store writes that save file as JSON on disk
/// in a crash-safe way (write a temporary copy first, then swap it in) and keeps the previous save as a backup.
/// Create one with a path such as <c>new JsonEvolutionCheckpointStore("runs/search-01.json")</c>, hand it to the
/// engine, and on restart <see cref="LoadLatestAsync"/> returns the newest valid snapshot for that run ID, or
/// <c>null</c> when the run has never been saved. Use <c>InMemoryEvolutionCheckpointStore</c> instead for tests or
/// short runs that need no durability.</para>
/// </remarks>
public sealed class JsonEvolutionCheckpointStore : IEvolutionCheckpointStore
{
    private readonly object _gate = new();
    private readonly string _path;
    private readonly string _previousPath;
    private readonly long _maxCheckpointBytes;

    /// <summary>Initializes a file-backed store.</summary>
    /// <param name="filePath">The primary checkpoint JSON path.</param>
    /// <param name="maxCheckpointBytes">Maximum encoded JSON size accepted for one checkpoint.</param>
    /// <exception cref="ArgumentNullException"><paramref name="filePath"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException"><paramref name="filePath"/> is empty or white space.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="maxCheckpointBytes"/> is not positive.</exception>
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

    /// <summary>Creates a store at the deterministic checkpoint path an output directory implies for one run.</summary>
    /// <param name="outputDirectory">The non-blank output directory, matching <c>EvolutionEngineOptions.OutputDirectory</c>.</param>
    /// <param name="runId">The non-blank run identifier, matching <c>EvolutionEngineOptions.RunId</c>.</param>
    /// <param name="maxCheckpointBytes">Maximum encoded JSON size accepted for one checkpoint.</param>
    /// <returns>A store writing to <see cref="EvolutionOutputLayout.CheckpointPath"/> with the same atomic pattern.</returns>
    /// <exception cref="ArgumentNullException">An argument is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">An argument is empty or white space, or the directory is not a valid path.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="maxCheckpointBytes"/> is not positive.</exception>
    /// <remarks>
    /// Two runs under one output directory get separate checkpoint files because the file name is derived from the
    /// run identifier, and the same run resumed later finds its own file without being told the path.
    /// </remarks>
    public static JsonEvolutionCheckpointStore ForOutputDirectory(string outputDirectory, string runId,
        long maxCheckpointBytes = 64L * 1024L * 1024L) =>
        new(new EvolutionOutputLayout(outputDirectory, runId).CheckpointPath, maxCheckpointBytes);

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

    /// <summary>Serialization shape of one on-disk checkpoint, with a checksum over its own fields.</summary>
    private sealed class CheckpointDocument
    {
        /// <summary>Gets or sets the checkpoint schema version.</summary>
        public int SchemaVersion { get; set; }
        /// <summary>Gets or sets the run identifier.</summary>
        public string RunId { get; set; } = string.Empty;
        /// <summary>Gets or sets the committed-state sequence.</summary>
        public long Sequence { get; set; }
        /// <summary>Gets or sets the resume compatibility hash.</summary>
        public string CompatibilityHash { get; set; } = string.Empty;
        /// <summary>Gets or sets the engine-owned serialized payload.</summary>
        public string Payload { get; set; } = string.Empty;
        /// <summary>Gets or sets the payload checksum.</summary>
        public string Checksum { get; set; } = string.Empty;
        /// <summary>Gets or sets the best elite quality recorded on the envelope.</summary>
        public double? Quality { get; set; }
        /// <summary>Gets or sets the direction a larger <see cref="Quality"/> is better in.</summary>
        public AiDotNet.Enums.EvolutionOptimizationDirection QualityDirection { get; set; }
        /// <summary>Gets or sets the checksum over every other field of this document.</summary>
        public string DocumentChecksum { get; set; } = string.Empty;

        /// <summary>Creates a document from a validated checkpoint and stamps its document checksum.</summary>
        public static CheckpointDocument From(EvolutionCheckpoint checkpoint)
        {
            var document = new CheckpointDocument
            {
                SchemaVersion = checkpoint.SchemaVersion,
                RunId = checkpoint.RunId,
                Sequence = checkpoint.Sequence,
                CompatibilityHash = checkpoint.CompatibilityHash,
                Payload = checkpoint.Payload,
                Checksum = checkpoint.Checksum,
                Quality = checkpoint.Quality,
                QualityDirection = checkpoint.QualityDirection
            };
            document.DocumentChecksum = document.ComputeDocumentChecksum();
            return document;
        }

        /// <summary>Verifies the document checksum against the current field values.</summary>
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
            Checksum,
            Quality?.ToString("R", System.Globalization.CultureInfo.InvariantCulture) ?? "none",
            ((int)QualityDirection).ToString(System.Globalization.CultureInfo.InvariantCulture)
        });

        /// <summary>Rebuilds the checkpoint carried by this document without recomputing its payload checksum.</summary>
        public EvolutionCheckpoint ToCheckpoint()
        {
            try
            {
                return new EvolutionCheckpoint(RunId, Sequence, CompatibilityHash, Payload, Checksum, SchemaVersion,
                    Quality, QualityDirection);
            }
            catch (ArgumentOutOfRangeException exception)
            {
                throw new InvalidDataException("The evolution checkpoint envelope carries an invalid value.", exception);
            }
        }
    }
}
