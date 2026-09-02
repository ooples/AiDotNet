using System.Globalization;
using System.Text;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Validation;
using Newtonsoft.Json;

namespace AiDotNet.Evolution;

/// <summary>Keeps a run's checkpoints as numbered, checksummed snapshots in one directory, with retention.</summary>
/// <remarks>
/// <para>
/// Each save writes a new immutable file named <c>checkpoint-&lt;sequence&gt;.json</c> using the same atomic pattern as
/// <see cref="JsonEvolutionCheckpointStore"/>: serialize with Newtonsoft.Json into a temporary file in the same
/// directory, flush it to disk, then move it into place. Because file names never repeat within a run, a crash can only
/// leave a complete new snapshot or none at all, and every earlier snapshot stays exactly as it was written. Every
/// document carries a checksum over its own envelope fields in addition to the payload checksum inside
/// <see cref="EvolutionCheckpoint"/>, so a truncated or edited file is detected rather than deserialized.
/// </para>
/// <para>
/// <see cref="LoadLatestAsync"/> walks the snapshots newest-first and returns the newest one that loads, verifies both
/// checksums, and belongs to the requested run, skipping the rest. That is a full journal rather than the one-deep
/// <c>.previous</c> fallback of the single-file store, so losing the last two snapshots to a bad disk still resumes a
/// run. <see cref="ListCheckpoints"/> reports the same walk as envelope metadata only, including the snapshots that
/// failed, so a damaged file is visible instead of silently missing. After each successful save, retention deletes the
/// snapshots outside both configured quotas; it never deletes the newest valid snapshot, the best-quality one, or any
/// file it could not read. Both the listing and retention verify each remaining snapshot, so their cost is one read per
/// retained file and retention is what keeps that number small.
/// </para>
/// <para><b>For Beginners:</b> This is where a long evolutionary search saves its progress so it can be resumed later.
/// Unlike a single save file that is overwritten each time, this store keeps a numbered history in a folder, so you can
/// go back to an earlier point and not only to the very last one. Point it at a folder for one run, hand it to the
/// engine together with a genome codec, and set <c>EvolutionEngineOptions.Resume</c> on the next run to continue from
/// the newest good save. If a save file is damaged, the store quietly moves on to the previous one instead of failing,
/// and <see cref="ListCheckpoints"/> shows you which files exist and how good each of them was.</para>
/// </remarks>
public sealed class DirectoryEvolutionCheckpointStore : IEvolutionCheckpointStore
{
    /// <summary>The prefix every snapshot file name starts with.</summary>
    public const string FileNamePrefix = "checkpoint-";

    /// <summary>The extension every snapshot file name ends with.</summary>
    public const string FileNameExtension = ".json";

    private const int SequenceDigits = 12;

    private readonly object _gate = new();
    private readonly string _directory;
    private readonly long _maxCheckpointBytes;
    private readonly EvolutionCheckpointRetentionOptions _retention;

    /// <summary>Initializes a directory-backed store and creates the directory when it does not exist.</summary>
    /// <param name="directory">The directory this run's snapshots live in.</param>
    /// <param name="retention">How many snapshots to keep; <c>null</c> uses the defaults of five newest plus one best.</param>
    /// <param name="maxCheckpointBytes">Maximum encoded JSON size accepted for one snapshot.</param>
    /// <exception cref="ArgumentNullException"><paramref name="directory"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException"><paramref name="directory"/> is empty or white space.</exception>
    /// <exception cref="ArgumentOutOfRangeException">
    /// <paramref name="maxCheckpointBytes"/> is not positive, or the retention quotas are out of range.
    /// </exception>
    public DirectoryEvolutionCheckpointStore(string directory,
        EvolutionCheckpointRetentionOptions? retention = null,
        long maxCheckpointBytes = 64L * 1024L * 1024L)
    {
        Guard.NotNullOrWhiteSpace(directory);
        if (maxCheckpointBytes <= 0) throw new ArgumentOutOfRangeException(nameof(maxCheckpointBytes));
        _directory = Path.GetFullPath(directory.Trim());
        _maxCheckpointBytes = maxCheckpointBytes;
        _retention = (retention ?? new EvolutionCheckpointRetentionOptions()).SnapshotAndValidate();
        Directory.CreateDirectory(_directory);
        DirectoryPath = _directory;
    }

    /// <summary>Creates a store in the per-run subdirectory an output directory implies.</summary>
    /// <param name="outputDirectory">The non-blank output directory, matching <c>EvolutionEngineOptions.OutputDirectory</c>.</param>
    /// <param name="runId">The non-blank run identifier, matching <c>EvolutionEngineOptions.RunId</c>.</param>
    /// <param name="retention">How many snapshots to keep; <c>null</c> uses the defaults.</param>
    /// <param name="maxCheckpointBytes">Maximum encoded JSON size accepted for one snapshot.</param>
    /// <returns>A store writing under <see cref="EvolutionOutputLayout.CheckpointsDirectory"/> in a folder named for the run.</returns>
    /// <exception cref="ArgumentNullException">An argument is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">An argument is empty or white space, or the directory is not a valid path.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="maxCheckpointBytes"/> or a retention quota is out of range.</exception>
    public static DirectoryEvolutionCheckpointStore ForOutputDirectory(string outputDirectory, string runId,
        EvolutionCheckpointRetentionOptions? retention = null, long maxCheckpointBytes = 64L * 1024L * 1024L)
    {
        var layout = new EvolutionOutputLayout(outputDirectory, runId);
        return new DirectoryEvolutionCheckpointStore(Path.Combine(layout.CheckpointsDirectory, layout.Stem),
            retention, maxCheckpointBytes);
    }

    /// <summary>Gets the resolved absolute directory snapshots are written to.</summary>
    public string DirectoryPath { get; }

    /// <inheritdoc/>
    public Task SaveAsync(EvolutionCheckpoint checkpoint, CancellationToken cancellationToken = default)
    {
        Guard.NotNull(checkpoint);
        cancellationToken.ThrowIfCancellationRequested();
        checkpoint.Validate();
        lock (_gate)
        {
            cancellationToken.ThrowIfCancellationRequested();
            EvolutionCheckpoint? existing = LoadNewestValid(checkpoint.RunId);
            if (existing is not null)
            {
                ValidateSuccessor(existing, checkpoint);
                if (checkpoint.Sequence == existing.Sequence) return Task.CompletedTask;
            }
            Persist(checkpoint, cancellationToken);
            ApplyRetention(checkpoint.RunId);
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
            return Task.FromResult(LoadNewestValid(runId));
        }
    }

    /// <summary>Lists every snapshot in the directory, newest first, reporting envelope metadata rather than payloads.</summary>
    /// <param name="runId">
    /// The run whose snapshots are listed. A snapshot belonging to a different run is reported with
    /// <see cref="EvolutionCheckpointDescriptor.IsValid"/> clear, exactly like an unreadable one.
    /// </param>
    /// <returns>One descriptor per snapshot file, ordered by descending sequence.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="runId"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException"><paramref name="runId"/> is empty or white space.</exception>
    public IReadOnlyList<EvolutionCheckpointDescriptor> ListCheckpoints(string runId)
    {
        Guard.NotNullOrWhiteSpace(runId);
        lock (_gate)
        {
            var descriptors = new List<EvolutionCheckpointDescriptor>();
            foreach (SnapshotFile snapshot in EnumerateSnapshots())
            {
                long size;
                try
                {
                    size = new FileInfo(snapshot.Path).Length;
                }
                catch (IOException)
                {
                    size = 0;
                }
                EvolutionCheckpoint? checkpoint = TryLoad(snapshot.Path, runId);
                descriptors.Add(checkpoint is null
                    ? new EvolutionCheckpointDescriptor(snapshot.Sequence, snapshot.FileName, size, isValid: false)
                    : new EvolutionCheckpointDescriptor(snapshot.Sequence, snapshot.FileName, size, isValid: true,
                        checkpoint.RunId, checkpoint.CompatibilityHash, checkpoint.Quality, checkpoint.QualityDirection));
            }
            return descriptors;
        }
    }

    private EvolutionCheckpoint? LoadNewestValid(string runId)
    {
        foreach (SnapshotFile snapshot in EnumerateSnapshots())
        {
            EvolutionCheckpoint? checkpoint = TryLoad(snapshot.Path, runId);
            if (checkpoint is not null) return checkpoint;
        }
        return null;
    }

    /// <summary>Enumerates the snapshot files in the directory ordered by descending sequence.</summary>
    private List<SnapshotFile> EnumerateSnapshots()
    {
        var snapshots = new List<SnapshotFile>();
        if (!Directory.Exists(_directory)) return snapshots;
        foreach (string path in Directory.GetFiles(_directory, FileNamePrefix + "*" + FileNameExtension))
        {
            string fileName = Path.GetFileName(path);
            string digits = fileName.Substring(FileNamePrefix.Length,
                fileName.Length - FileNamePrefix.Length - FileNameExtension.Length);
            if (digits.Length == 0 || !long.TryParse(digits, NumberStyles.None, CultureInfo.InvariantCulture,
                out long sequence) || sequence < 0)
            {
                continue;
            }
            snapshots.Add(new SnapshotFile(sequence, fileName, path));
        }
        snapshots.Sort(static (first, second) =>
        {
            int bySequence = second.Sequence.CompareTo(first.Sequence);
            return bySequence != 0 ? bySequence : string.CompareOrdinal(first.FileName, second.FileName);
        });
        return snapshots;
    }

    /// <summary>Deletes the snapshots that fall outside both retention quotas.</summary>
    /// <remarks>
    /// The protected set is built before anything is deleted and always contains the newest valid snapshot and the
    /// single best-quality one, so no quota combination can remove either. Unreadable files are protected too: they
    /// cost almost nothing and are the evidence that something went wrong.
    /// </remarks>
    private void ApplyRetention(string runId)
    {
        List<SnapshotFile> snapshots = EnumerateSnapshots();
        var loaded = new List<KeyValuePair<SnapshotFile, EvolutionCheckpoint>>();
        var protectedNames = new HashSet<string>(StringComparer.Ordinal);
        foreach (SnapshotFile snapshot in snapshots)
        {
            EvolutionCheckpoint? checkpoint = TryLoad(snapshot.Path, runId);
            if (checkpoint is null)
            {
                protectedNames.Add(snapshot.FileName);
                continue;
            }
            loaded.Add(new KeyValuePair<SnapshotFile, EvolutionCheckpoint>(snapshot, checkpoint));
        }
        if (loaded.Count == 0) return;

        // Newest valid first, so the head of the list is both the resume target and the first keep-last slot.
        for (int index = 0; index < loaded.Count && index < _retention.KeepLast; index++)
            protectedNames.Add(loaded[index].Key.FileName);
        protectedNames.Add(loaded[0].Key.FileName);

        List<KeyValuePair<SnapshotFile, EvolutionCheckpoint>> byQuality = loaded
            .OrderByDescending(item => item.Value.Quality.HasValue)
            .ThenByDescending(item => Rank(item.Value))
            .ThenByDescending(item => item.Key.Sequence)
            .ToList();
        int bestQuota = _retention.KeepBest < 1 ? 1 : _retention.KeepBest;
        for (int index = 0; index < byQuality.Count && index < bestQuota; index++)
            protectedNames.Add(byQuality[index].Key.FileName);

        foreach (KeyValuePair<SnapshotFile, EvolutionCheckpoint> item in loaded)
        {
            if (protectedNames.Contains(item.Key.FileName)) continue;
            try
            {
                File.Delete(item.Key.Path);
            }
            catch (IOException)
            {
                // A snapshot held open by a reader is simply retained until the next save.
            }
            catch (UnauthorizedAccessException)
            {
                // Same reasoning: retention is best-effort housekeeping and must never fail a save.
            }
        }
    }

    /// <summary>Orients a checkpoint's quality so that a larger value always means better.</summary>
    private static double Rank(EvolutionCheckpoint checkpoint)
    {
        if (!checkpoint.Quality.HasValue) return double.NegativeInfinity;
        return checkpoint.QualityDirection == EvolutionOptimizationDirection.Maximize
            ? checkpoint.Quality.Value
            : -checkpoint.Quality.Value;
    }

    private void Persist(EvolutionCheckpoint checkpoint, CancellationToken cancellationToken)
    {
        string targetPath = Path.Combine(_directory, FileNameFor(checkpoint.Sequence));
        string tempPath = Path.Combine(_directory,
            $".{FileNameFor(checkpoint.Sequence)}.{Guid.NewGuid():N}.tmp");
        string json = JsonConvert.SerializeObject(SnapshotDocument.From(checkpoint), Formatting.Indented);
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
            if (File.Exists(targetPath))
            {
                File.Replace(tempPath, targetPath, destinationBackupFileName: null, ignoreMetadataErrors: true);
            }
            else
            {
                File.Move(tempPath, targetPath);
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

    /// <summary>Builds the fixed-width file name one sequence is stored under.</summary>
    /// <param name="sequence">The non-negative committed-state sequence.</param>
    /// <returns>The file name, zero-padded so an ordinal name sort matches the numeric order.</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="sequence"/> is negative.</exception>
    public static string FileNameFor(long sequence)
    {
        if (sequence < 0) throw new ArgumentOutOfRangeException(nameof(sequence));
        return FileNamePrefix + sequence.ToString("D" + SequenceDigits.ToString(CultureInfo.InvariantCulture),
            CultureInfo.InvariantCulture) + FileNameExtension;
    }

    /// <summary>Loads one snapshot, returning <c>null</c> for anything unreadable, corrupt, or foreign.</summary>
    private EvolutionCheckpoint? TryLoad(string path, string runId)
    {
        try
        {
            if (!File.Exists(path)) return null;
            if (new FileInfo(path).Length > _maxCheckpointBytes) return null;
            string json;
            using (var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                if (stream.Length > _maxCheckpointBytes) return null;
                using var reader = new StreamReader(stream, Encoding.UTF8, detectEncodingFromByteOrderMarks: true);
                json = reader.ReadToEnd();
            }
            if (Encoding.UTF8.GetByteCount(json) > _maxCheckpointBytes) return null;
            SnapshotDocument? document = JsonConvert.DeserializeObject<SnapshotDocument>(json);
            if (document is null || !document.IsIntact()) return null;
            EvolutionCheckpoint checkpoint = document.ToCheckpoint();
            checkpoint.Validate();
            return string.Equals(checkpoint.RunId, runId, StringComparison.Ordinal) ? checkpoint : null;
        }
        catch (JsonException)
        {
            return null;
        }
        catch (InvalidDataException)
        {
            return null;
        }
        catch (ArgumentException)
        {
            return null;
        }
        catch (IOException)
        {
            return null;
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

    /// <summary>One snapshot file discovered in the directory.</summary>
    private sealed class SnapshotFile
    {
        public SnapshotFile(long sequence, string fileName, string path)
        {
            Sequence = sequence;
            FileName = fileName;
            Path = path;
        }

        public long Sequence { get; }
        public string FileName { get; }
        public string Path { get; }
    }

    /// <summary>Serialization shape of one on-disk snapshot, with a checksum over its own fields.</summary>
    private sealed class SnapshotDocument
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
        public EvolutionOptimizationDirection QualityDirection { get; set; }
        /// <summary>Gets or sets the checksum over every other field of this document.</summary>
        public string DocumentChecksum { get; set; } = string.Empty;

        public static SnapshotDocument From(EvolutionCheckpoint checkpoint)
        {
            var document = new SnapshotDocument
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

        public bool IsIntact() =>
            string.Equals(DocumentChecksum, ComputeDocumentChecksum(), StringComparison.Ordinal);

        public EvolutionCheckpoint ToCheckpoint() => new(RunId, Sequence, CompatibilityHash, Payload, Checksum,
            SchemaVersion, Quality, QualityDirection);

        private string ComputeDocumentChecksum() => EvolutionHash.Combine(new[]
        {
            SchemaVersion.ToString(CultureInfo.InvariantCulture),
            RunId,
            Sequence.ToString(CultureInfo.InvariantCulture),
            CompatibilityHash,
            Payload,
            Checksum,
            Quality?.ToString("R", CultureInfo.InvariantCulture) ?? "none",
            ((int)QualityDirection).ToString(CultureInfo.InvariantCulture)
        });
    }
}
