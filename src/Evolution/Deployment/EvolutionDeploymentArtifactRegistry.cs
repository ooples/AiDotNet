using System.Text.Json;

namespace AiDotNet.Evolution.Deployment;

/// <summary>Write-once deployment payloads/evidence and one compare-and-swap deployment slot per private directory.</summary>
/// <remarks>
/// Selection refreshes small slot/quarantine records, not model bytes. Writes use flushed files and non-overwriting
/// publication; slot replacement is atomic for cooperating processes. This implementation does not attest directory
/// durability across power loss. Applications must explicitly authorize best-effort persistence before promotion.
/// Protect the directory and its parents from untrusted writers. Retention/quotas are application-owned.
/// </remarks>
public sealed class EvolutionDeploymentArtifactRegistry
{
    private readonly string _directory;
    private const int MetadataLimit = 16 * 1024;
    private const int EvidenceLimit = 4 * 1024 * 1024;

    /// <summary>Creates a private registry with one application deployment slot.</summary>
    public EvolutionDeploymentArtifactRegistry(string directory)
    {
        if (string.IsNullOrWhiteSpace(directory) || !Path.IsPathRooted(directory))
            throw new ArgumentException("An absolute private registry directory is required.", nameof(directory));
        _directory = Path.GetFullPath(directory);
        Directory.CreateDirectory(_directory);
        DeploymentEncoding.RefuseLink(_directory);
    }

    /// <summary>Stores exact candidate bytes without promoting them. Oversize payloads are never truncated.</summary>
    public string Stage(EvolutionDeployableArtifact artifact)
    {
        if (artifact is null) throw new ArgumentNullException(nameof(artifact));
        Put(artifact.PayloadHash, ".payload", artifact.CopyPayload(), EvolutionDeployableArtifact.MaximumPayloadBytes);
        Put(artifact.Id, ".artifact", JsonSerializer.SerializeToUtf8Bytes(ArtifactDocument.From(artifact)), MetadataLimit);
        return artifact.Id;
    }

    /// <summary>Loads a canonical artifact for the exact envelope without executing or activating it.</summary>
    public EvolutionDeployableArtifact Load(string artifactId, EvolutionDeploymentEnvelope envelope)
    {
        if (envelope is null) throw new ArgumentNullException(nameof(envelope));
        byte[] manifest = ReadObject(artifactId, ".artifact", MetadataLimit);
        var document = DeploymentEncoding.Parse<ArtifactDocument>(manifest);
        if (document.SchemaVersion != 1) throw new InvalidDataException("Unsupported deployment artifact schema.");
        var actualEnvelope = new EvolutionDeploymentEnvelope(document.RuntimeHash, document.DeviceHash, document.CompilerHash,
            document.DatasetHash, document.WorkloadHash, document.ValidationProtocolHash);
        if (actualEnvelope.Key != envelope.Key) throw new InvalidDataException("Artifact is outside the requested applicability envelope.");
        byte[] payload = ReadObject(document.PayloadHash, ".payload", EvolutionDeployableArtifact.MaximumPayloadBytes);
        if (DeploymentEncoding.Hash(payload) != document.PayloadHash) throw new InvalidDataException("Deployment payload hash mismatch.");
        var artifact = new EvolutionDeployableArtifact(document.Kind, document.Format, document.TypeIdentity, actualEnvelope, payload);
        if (artifact.Id != artifactId || !manifest.SequenceEqual(JsonSerializer.SerializeToUtf8Bytes(ArtifactDocument.From(artifact))))
            throw new InvalidDataException("Deployment artifact identity or canonical manifest differs.");
        return artifact;
    }

    /// <summary>Retains bounded raw validation/regression evidence and returns its exact byte digest.</summary>
    public string RetainEvidence(byte[] evidence)
    {
        if (evidence is null || evidence.Length == 0 || evidence.Length > EvidenceLimit)
            throw new ArgumentException("Evidence must be nonempty and bounded.", nameof(evidence));
        byte[] copy = (byte[])evidence.Clone();
        string id = DeploymentEncoding.Hash(copy);
        Put(id, ".evidence", copy, EvidenceLimit);
        return id;
    }

    /// <summary>Reads hash-verified raw evidence without interpreting it.</summary>
    public byte[] ReadEvidence(string evidenceId)
    {
        byte[] bytes = ReadObject(evidenceId, ".evidence", EvidenceLimit);
        if (DeploymentEncoding.Hash(bytes) != evidenceId) throw new InvalidDataException("Deployment evidence hash mismatch.");
        return bytes;
    }

    /// <summary>Checks persistent quarantine; malformed tombstones also deny admission.</summary>
    public bool IsQuarantined(string artifactId)
    {
        CheckRoot();
        DeploymentEncoding.RequireHash(artifactId);
        // GetAttributes propagates access failures instead of File.Exists silently treating them as admission.
        try { _ = File.GetAttributes(Path.Combine(_directory, artifactId + ".quarantine")); return true; }
        catch (FileNotFoundException) { return false; }
    }

    internal Slot ReadSlot()
    {
        CheckRoot();
        byte[] bytes;
        try { bytes = DeploymentEncoding.Read(Path.Combine(_directory, "active.json"), MetadataLimit); }
        catch (FileNotFoundException) { return new Slot(null, null, null, null, null, true); }
        var document = DeploymentEncoding.Parse<SlotDocument>(bytes);
        if (document.SchemaVersion != 1) throw new InvalidDataException("Unsupported deployment slot schema.");
        if (document.ActiveId is not null) DeploymentEncoding.RequireHash(document.ActiveId);
        if (document.PriorId is not null) DeploymentEncoding.RequireHash(document.PriorId);
        DeploymentEncoding.RequireHash(document.EvidenceId);
        DeploymentEncoding.RequireHash(document.ValidationEvidenceId);
        DeploymentEncoding.RequireHash(document.EnvelopeKey);
        if (!Guid.TryParseExact(document.Nonce, "N", out _) || !bytes.SequenceEqual(JsonSerializer.SerializeToUtf8Bytes(document)))
            throw new InvalidDataException("Noncanonical deployment slot.");
        return new Slot(DeploymentEncoding.Hash(bytes), document.ActiveId, document.PriorId, document.EnvelopeKey,
            document.ValidationEvidenceId, document.CandidateEvidenceSide);
    }

    internal bool TryPromote(string? expectedRevision, EvolutionDeployableArtifact candidate,
        EvolutionDeployableArtifact incumbent, string evidenceId)
    {
        Stage(candidate); Stage(incumbent); ReadEvidence(evidenceId);
        using var lease = Lease();
        Slot current = ReadSlot();
        if (current.Revision != expectedRevision || IsQuarantined(candidate.Id) || IsQuarantined(incumbent.Id)) return false;
        if (current.ActiveId is not null && current.EnvelopeKey == candidate.Envelope.Key &&
            !IsQuarantined(current.ActiveId) && current.ActiveId != incumbent.Id) return false;
        WriteSlot(candidate.Id, incumbent.Id, candidate.Envelope.Key, evidenceId, evidenceId, true);
        return true;
    }

    internal bool TryQuarantine(string? expectedRevision, string expectedActiveId, string evidenceId,
        EvolutionDeploymentEnvelope envelope, out EvolutionDeployableArtifact? rollback)
    {
        rollback = null;
        ReadEvidence(evidenceId);
        using var lease = Lease();
        Slot current = ReadSlot();
        if (current.Revision != expectedRevision || current.ActiveId != expectedActiveId || current.EnvelopeKey != envelope.Key) return false;
        // Never replace the first regression receipt, even if a retry observes another regression.
        string marker = Path.Combine(_directory, expectedActiveId + ".quarantine");
        if (!IsQuarantined(expectedActiveId)) PublishNew(marker, JsonSerializer.SerializeToUtf8Bytes(new { SchemaVersion = 1, EvidenceId = evidenceId }));
        if (current.PriorId is not null && !IsQuarantined(current.PriorId))
        {
            try { rollback = Load(current.PriorId, envelope); }
            catch (Exception error) when (error is IOException or InvalidDataException or ArgumentException or JsonException or UnauthorizedAccessException)
            { /* An invalid prior never prevents quarantine of a measured regression. */ }
        }
        // Failure here leaves the tombstone visible, so a reader cannot reload the regressing active pointer.
        WriteSlot(rollback?.Id, null, envelope.Key, evidenceId, current.ValidationEvidenceId!, false);
        return true;
    }

    private void WriteSlot(string? active, string? prior, string envelopeKey, string evidenceId, string validationEvidenceId, bool candidateSide)
    {
        string pending = Path.Combine(_directory, Guid.NewGuid().ToString("N") + ".pending");
        DeploymentEncoding.WriteNew(pending, JsonSerializer.SerializeToUtf8Bytes(new SlotDocument
        { ActiveId = active, PriorId = prior, EnvelopeKey = envelopeKey, EvidenceId = evidenceId,
            ValidationEvidenceId = validationEvidenceId, CandidateEvidenceSide = candidateSide, Nonce = Guid.NewGuid().ToString("N") }));
        string destination = Path.Combine(_directory, "active.json");
        if (File.Exists(destination))
        {
            DeploymentEncoding.RefuseLink(destination);
            File.Replace(pending, destination, null);
        }
        else File.Move(pending, destination);
    }
    private FileStream Lease()
    {
        CheckRoot();
        string path = Path.Combine(_directory, "writer.lock");
        if (File.Exists(path)) DeploymentEncoding.RefuseLink(path);
        return new FileStream(path, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None);
    }
    private void CheckRoot() => DeploymentEncoding.RefuseLink(_directory);
    private byte[] ReadObject(string id, string extension, int limit)
    {
        CheckRoot(); DeploymentEncoding.RequireHash(id);
        return DeploymentEncoding.Read(Path.Combine(_directory, id + extension), limit);
    }
    private void Put(string id, string extension, byte[] bytes, int limit)
    {
        CheckRoot(); DeploymentEncoding.RequireHash(id);
        if (bytes.Length > limit) throw new InvalidDataException("Deployment object exceeds its byte bound.");
        string target = Path.Combine(_directory, id + extension);
        if (!File.Exists(target))
        {
            try { PublishNew(target, bytes); }
            catch (IOException) when (File.Exists(target)) { /* A cooperating identical writer may win. */ }
        }
        if (!DeploymentEncoding.Read(target, limit).SequenceEqual(bytes)) throw new InvalidDataException("Existing deployment object differs; refusing replacement.");
    }
    private void PublishNew(string target, byte[] bytes)
    {
        string pending = Path.Combine(_directory, Guid.NewGuid().ToString("N") + ".pending");
        DeploymentEncoding.WriteNew(pending, bytes);
        File.Move(pending, target);
    }

    internal sealed class Slot
    {
        internal Slot(string? revision, string? active, string? prior, string? envelope, string? validation, bool candidateSide)
        { Revision = revision; ActiveId = active; PriorId = prior; EnvelopeKey = envelope; ValidationEvidenceId = validation; CandidateEvidenceSide = candidateSide; }
        internal string? Revision { get; }
        internal string? ActiveId { get; }
        internal string? PriorId { get; }
        internal string? EnvelopeKey { get; }
        internal string? ValidationEvidenceId { get; }
        internal bool CandidateEvidenceSide { get; }
    }
    private sealed class SlotDocument
    {
        public SlotDocument() { }
        public int SchemaVersion { get; set; } = 1;
        public string? ActiveId { get; set; }
        public string? PriorId { get; set; }
        public string EnvelopeKey { get; set; } = string.Empty;
        public string EvidenceId { get; set; } = string.Empty;
        public string ValidationEvidenceId { get; set; } = string.Empty;
        public bool CandidateEvidenceSide { get; set; }
        public string Nonce { get; set; } = string.Empty;
    }
    private sealed class ArtifactDocument
    {
        public ArtifactDocument() { }
        public int SchemaVersion { get; set; } = 1;
        public EvolutionDeploymentArtifactKind Kind { get; set; }
        public string Format { get; set; } = string.Empty;
        public string TypeIdentity { get; set; } = string.Empty;
        public string PayloadHash { get; set; } = string.Empty;
        public string RuntimeHash { get; set; } = string.Empty;
        public string DeviceHash { get; set; } = string.Empty;
        public string CompilerHash { get; set; } = string.Empty;
        public string DatasetHash { get; set; } = string.Empty;
        public string WorkloadHash { get; set; } = string.Empty;
        public string ValidationProtocolHash { get; set; } = string.Empty;
        internal static ArtifactDocument From(EvolutionDeployableArtifact artifact) => new()
        {
            Kind = artifact.Kind, Format = artifact.Format, TypeIdentity = artifact.TypeIdentity, PayloadHash = artifact.PayloadHash,
            RuntimeHash = artifact.Envelope.RuntimeHash, DeviceHash = artifact.Envelope.DeviceHash, CompilerHash = artifact.Envelope.CompilerHash,
            DatasetHash = artifact.Envelope.DatasetHash, WorkloadHash = artifact.Envelope.WorkloadHash,
            ValidationProtocolHash = artifact.Envelope.ValidationProtocolHash
        };
    }
}
