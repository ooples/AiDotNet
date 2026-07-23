namespace AiDotNet.Serving.Persistence.Entities;

/// <summary>
/// Persisted encrypted artifact key material (protected at rest) for key release.
/// </summary>
public sealed class ProtectedArtifactEntity
{
    public string ArtifactName { get; set; } = string.Empty;

    public string EncryptedPath { get; set; } = string.Empty;

    public string KeyId { get; set; } = string.Empty;

    public string Algorithm { get; set; } = string.Empty;

    public byte[] ProtectedKey { get; set; } = Array.Empty<byte>();

    public byte[] ProtectedNonce { get; set; } = Array.Empty<byte>();

    public DateTimeOffset CreatedAt { get; set; } = DateTimeOffset.UtcNow;
}

