using AiDotNet.Validation;

namespace AiDotNet.Evolution;

/// <summary>A versioned, checksummed, opaque engine snapshot.</summary>
public sealed class EvolutionCheckpoint
{
    /// <summary>The current public checkpoint schema.</summary>
    public const int CurrentSchemaVersion = 1;

    /// <summary>Initializes a checkpoint and computes its payload checksum.</summary>
    /// <param name="runId">The stable run identifier.</param>
    /// <param name="sequence">The monotonically increasing committed-state sequence.</param>
    /// <param name="compatibilityHash">Hash of every component that affects resume semantics.</param>
    /// <param name="payload">The engine-owned serialized state.</param>
    /// <param name="schemaVersion">The checkpoint schema version.</param>
    public EvolutionCheckpoint(string runId, long sequence, string compatibilityHash, string payload,
        int schemaVersion = CurrentSchemaVersion)
        : this(runId, sequence, compatibilityHash, payload, EvolutionHash.Compute(payload), schemaVersion)
    {
    }

    internal EvolutionCheckpoint(string runId, long sequence, string compatibilityHash, string payload,
        string checksum, int schemaVersion)
    {
        Guard.NotNullOrWhiteSpace(runId);
        Guard.NotNullOrWhiteSpace(compatibilityHash);
        Guard.NotNull(payload);
        Guard.NotNullOrWhiteSpace(checksum);
        if (sequence < 0) throw new ArgumentOutOfRangeException(nameof(sequence));
        if (schemaVersion <= 0) throw new ArgumentOutOfRangeException(nameof(schemaVersion));
        RunId = runId.Trim();
        Sequence = sequence;
        CompatibilityHash = compatibilityHash.Trim();
        Payload = payload;
        Checksum = checksum.Trim();
        SchemaVersion = schemaVersion;
    }

    /// <summary>Gets the schema version.</summary>
    public int SchemaVersion { get; }

    /// <summary>Gets the run identifier.</summary>
    public string RunId { get; }

    /// <summary>Gets the committed-state sequence.</summary>
    public long Sequence { get; }

    /// <summary>Gets the resume compatibility hash.</summary>
    public string CompatibilityHash { get; }

    /// <summary>Gets the engine-owned serialized payload.</summary>
    public string Payload { get; }

    /// <summary>Gets the lowercase SHA-256 checksum of <see cref="Payload"/>.</summary>
    public string Checksum { get; }

    /// <summary>Verifies the schema and payload checksum.</summary>
    /// <exception cref="InvalidDataException">The schema or checksum is invalid.</exception>
    public void Validate()
    {
        if (SchemaVersion != CurrentSchemaVersion)
            throw new InvalidDataException($"Unsupported evolution checkpoint schema {SchemaVersion}.");
        string actual = EvolutionHash.Compute(Payload);
        if (!string.Equals(actual, Checksum, StringComparison.Ordinal))
            throw new InvalidDataException("Evolution checkpoint checksum validation failed.");
    }

    internal EvolutionCheckpoint Clone() => new(RunId, Sequence, CompatibilityHash, Payload, Checksum, SchemaVersion);
}
