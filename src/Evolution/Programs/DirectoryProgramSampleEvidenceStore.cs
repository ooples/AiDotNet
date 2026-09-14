using System.IO;
using System.Text;
using System.Text.Json;
using AiDotNet.Interfaces;
using JsonException = System.Text.Json.JsonException;
using JsonSerializer = System.Text.Json.JsonSerializer;

namespace AiDotNet.Evolution.Programs;

/// <summary>Atomic, content-addressed raw scalar observations for persistent program-fitness reuse.</summary>
/// <remarks>
/// Supply genuine observations through the capture callback, not synthetic copies of an aggregate.
/// Verification reopens the artifact, validates its digest and exact candidate/origin/result binding, and
/// recomputes mean and sample standard error. Only ProgramMeasurementStatistics.Version is supported;
/// confidence intervals and other aggregation policies require a different provider. Descriptors/objectives
/// are bound metadata, not independently measured by scalar samples. Current correctness remains outside reuse.
/// The private directory must be protected from hostile writers and aliases. Hashes are not signatures or
/// proof of physical acquisition, independence, hardware identity or stationarity. Cooperating writers use
/// an exclusive file handle; contention declines persistence through IOException, without retrying forever.
/// Each artifact is bounded; no eviction, overwrite, external service, evaluator or background task runs here.
/// The capture callback must be thread-safe, bounded and separately metered; logical facade counters are not I/O costs.
/// </remarks>
public sealed class DirectoryProgramSampleEvidenceStore : IProgramMeasurementEvidenceStore
{
    /// <summary>Maximum encoded bytes per evidence artifact, including bound metadata.</summary>
    public const int MaximumArtifactBytes = 2 * 1024 * 1024;
    private static readonly UTF8Encoding Utf8 = new(false, true);
    private readonly Func<ProgramGenome, EvolutionTaskResult, EvolutionEvaluationContext, CancellationToken,
        ValueTask<IReadOnlyList<ProgramMeasurementObservation>?>> _capture;
    private int _cleanupFailed;

    /// <summary>Creates a private absolute store with an explicit acquisition-source fingerprint and capacity.</summary>
    public DirectoryProgramSampleEvidenceStore(string directory, string captureVersion,
        Func<ProgramGenome, EvolutionTaskResult, EvolutionEvaluationContext, CancellationToken,
            ValueTask<IReadOnlyList<ProgramMeasurementObservation>?>> capture, int maximumEntries = 4096)
    {
        if (directory is null) throw new ArgumentNullException(nameof(directory));
        VersionPinnedProgramFitnessEvaluator.ValidateIdentity(captureVersion, nameof(captureVersion));
        _capture = capture ?? throw new ArgumentNullException(nameof(capture));
        if (!Path.IsPathRooted(directory) || (Path.DirectorySeparatorChar == '\\' &&
            !directory.StartsWith("\\\\", StringComparison.Ordinal) &&
            (directory.Length < 3 || !char.IsLetter(directory[0]) || directory[1] != ':' || directory[2] is not ('\\' or '/'))))
            throw new ArgumentException("Use a fully qualified private evidence directory.", nameof(directory));
        string full = Path.GetFullPath(directory).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        string root = (Path.GetPathRoot(Path.GetFullPath(directory)) ?? string.Empty).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        if (string.Equals(full, root, StringComparison.OrdinalIgnoreCase)) throw new ArgumentException("A filesystem root cannot be an evidence store.", nameof(directory));
        if (maximumEntries < 1 || maximumEntries > 1_000_000) throw new ArgumentOutOfRangeException(nameof(maximumEntries));
        DirectoryPath = full; MaximumEntries = maximumEntries;
        VersionHash = EvolutionHash.Combine(new[] { "directory-program-samples-v1", ProgramMeasurementStatistics.Version, captureVersion });
        Directory.CreateDirectory(full);
    }

    /// <inheritdoc/>
    public string VersionHash { get; }
    /// <summary>Gets the lexical private root; this is not a symlink or security attestation.</summary>
    public string DirectoryPath { get; }
    /// <summary>Gets the cooperating-writer artifact count limit; retained evidence is never evicted.</summary>
    public int MaximumEntries { get; }
    /// <summary>Gets whether temporary cleanup failed; the original publication outcome is preserved.</summary>
    public bool HasTemporaryCleanupFailure => Volatile.Read(ref _cleanupFailed) != 0;

    /// <inheritdoc/>
    public async ValueTask<string?> RetainAsync(ProgramGenome candidate, EvolutionTaskResult measurement,
        EvolutionEvaluationContext context, CancellationToken cancellationToken = default)
    {
        ValidateArguments(candidate, measurement, context); cancellationToken.ThrowIfCancellationRequested();
        if (measurement.MeasurementOrigin?.Kind != EvolutionMeasurementOriginKind.Measured) return null;
        var captured = await _capture(candidate, measurement, context, cancellationToken).ConfigureAwait(false);
        cancellationToken.ThrowIfCancellationRequested();
        if (captured is null) return null;
        var samples = captured.Take(EvolutionMeasurementOrigin.MaximumSampleIds + 1).ToArray();
        if (!Matches(measurement, samples)) return null;
        string binding = Binding(candidate, measurement);
        string json = JsonSerializer.Serialize(new { SchemaVersion = 1, Binding = binding, Observations = samples });
        if (Utf8.GetByteCount(json) > MaximumArtifactBytes) return null;
        string digest = EvolutionHash.Compute(json);
        byte[] bytes = Utf8.GetBytes(json);
        // The persistent lock marker is not evidence and is never deleted; the OS releases its handle on crash.
        using var gate = new FileStream(FileName(".writer.lock"), FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None);
        cancellationToken.ThrowIfCancellationRequested();
        byte[]? existing = Read(digest, cancellationToken);
        if (existing is not null)
        {
            if (!bytes.SequenceEqual(existing)) throw new InvalidDataException("Existing raw evidence conflicts with its digest.");
            return digest;
        }
        if (Directory.EnumerateFiles(DirectoryPath, "*.json", SearchOption.TopDirectoryOnly).Take(MaximumEntries).Count() >= MaximumEntries) return null;
        string temporary = FileName(".raw-" + Guid.NewGuid().ToString("N") + ".tmp");
        try
        {
            using (var file = new FileStream(temporary, FileMode.CreateNew, FileAccess.Write, FileShare.None))
            {
                file.Write(bytes, 0, bytes.Length); file.Flush(true);
            }
            cancellationToken.ThrowIfCancellationRequested();
            File.Move(temporary, FileName(digest + ".json"));
            return digest;
        }
        finally
        {
            try { if (File.Exists(temporary)) File.Delete(temporary); }
            catch (IOException) { Interlocked.Exchange(ref _cleanupFailed, 1); }
            catch (UnauthorizedAccessException) { Interlocked.Exchange(ref _cleanupFailed, 1); }
        }
    }

    /// <inheritdoc/>
    public ValueTask<bool> VerifyAsync(ProgramGenome candidate, EvolutionTaskResult measurement, string evidenceSha256,
        EvolutionEvaluationContext context, CancellationToken cancellationToken = default)
    {
        ValidateArguments(candidate, measurement, context); cancellationToken.ThrowIfCancellationRequested();
        if (evidenceSha256 is null || evidenceSha256.Length != 64 || evidenceSha256.Any(c => c is not (>= '0' and <= '9') and not (>= 'a' and <= 'f')))
            return new(false);
        byte[]? bytes = Read(evidenceSha256, cancellationToken);
        if (bytes is null) return new(false);
        try
        {
            string json = Utf8.GetString(bytes);
            if (EvolutionHash.Compute(json) != evidenceSha256) return new(false);
            using var document = JsonDocument.Parse(json, new JsonDocumentOptions { MaxDepth = 8 });
            var root = document.RootElement;
            Fields(root, "SchemaVersion", "Binding", "Observations");
            if (root.GetProperty("SchemaVersion").GetInt32() != 1 || root.GetProperty("Binding").GetString() != Binding(candidate, measurement)) return new(false);
            var samples = new List<ProgramMeasurementObservation>();
            foreach (var element in root.GetProperty("Observations").EnumerateArray())
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (samples.Count >= EvolutionMeasurementOrigin.MaximumSampleIds) return new(false);
                Fields(element, "SampleId", "Value");
                string? sampleId = element.GetProperty("SampleId").GetString();
                if (sampleId is null) return new(false);
                samples.Add(new ProgramMeasurementObservation(sampleId, element.GetProperty("Value").GetDouble()));
            }
            return new(Matches(measurement, samples));
        }
        catch (Exception error) when (error is JsonException or ArgumentException or InvalidOperationException or FormatException or OverflowException or InvalidDataException)
        { return new(false); }
    }

    private static bool Matches(EvolutionTaskResult measurement, IReadOnlyList<ProgramMeasurementObservation> samples)
    {
        var origin = measurement.MeasurementOrigin;
        if (measurement.Status != EvolutionEvaluationStatus.Completed || measurement.Quality is not { } quality || measurement.ConstraintViolations.Any(value => value > 0) ||
            origin is null || origin.StatisticsVersion != ProgramMeasurementStatistics.Version || !origin.StandardError.HasValue ||
            origin.ConfidenceLevel.HasValue || samples.Any(sample => sample is null) ||
            !origin.SampleIds.SequenceEqual(samples.Select(sample => sample.SampleId), StringComparer.Ordinal)) return false;
        try
        {
            var stats = ProgramMeasurementStatistics.Calculate(samples);
            return Close(quality, stats.Mean) && Close(origin.StandardError.Value, stats.StandardError);
        }
        catch (ArgumentException) { return false; }
    }

    private static bool Close(double left, double right) => Math.Abs(left - right) <= 1e-12 * Math.Max(1, Math.Max(Math.Abs(left), Math.Abs(right)));

    private string Binding(ProgramGenome candidate, EvolutionTaskResult measurement) => JsonSerializer.Serialize(new
    {
        Provider = VersionHash,
        candidate.Id,
        PayloadSha256 = EvolutionHash.Compute(new ProgramGenomeCodec().Serialize(candidate)),
        measurement.Status,
        measurement.Quality,
        measurement.Direction,
        Descriptors = measurement.Descriptors.OrderBy(pair => pair.Key, StringComparer.Ordinal).ToArray(),
        measurement.Objectives,
        measurement.ConstraintViolations,
        Metrics = measurement.Metrics.OrderBy(pair => pair.Key, StringComparer.Ordinal).ToArray(),
        Origin = measurement.MeasurementOrigin?.AsReused(EvolutionMeasurementOriginKind.PersistentReuse).ToJson()
        // Deliberately exclude current operation cost, diagnostics and artifacts; original cost remains in Origin.
    });

    private byte[]? Read(string digest, CancellationToken cancellationToken)
    {
        try
        {
            using var file = new FileStream(FileName(digest + ".json"), FileMode.Open, FileAccess.Read, FileShare.Read | FileShare.Delete);
            if (file.Length > MaximumArtifactBytes) throw new InvalidDataException("Raw evidence exceeds its byte limit.");
            using var buffer = new MemoryStream();
            byte[] chunk = new byte[8192];
            int count;
            while ((count = file.Read(chunk, 0, chunk.Length)) > 0)
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (buffer.Length + count > MaximumArtifactBytes) throw new InvalidDataException("Raw evidence exceeds its byte limit.");
                buffer.Write(chunk, 0, count);
            }
            cancellationToken.ThrowIfCancellationRequested();
            return buffer.ToArray();
        }
        catch (FileNotFoundException) { return null; }
    }

    private string FileName(string leaf) => DirectoryPath + Path.DirectorySeparatorChar + leaf; // Only validated digest or generated constant/GUID leaves.

    private static void Fields(JsonElement element, params string[] names)
    {
        var remaining = new HashSet<string>(names, StringComparer.Ordinal);
        foreach (var field in element.EnumerateObject()) if (!remaining.Remove(field.Name)) throw new InvalidDataException("Unknown or duplicate raw evidence field.");
        if (remaining.Count != 0) throw new InvalidDataException("Missing raw evidence field.");
    }

    private static void ValidateArguments(ProgramGenome candidate, EvolutionTaskResult measurement, EvolutionEvaluationContext context)
    {
        if (candidate is null) throw new ArgumentNullException(nameof(candidate));
        if (measurement is null) throw new ArgumentNullException(nameof(measurement));
        if (context is null) throw new ArgumentNullException(nameof(context));
    }
}
