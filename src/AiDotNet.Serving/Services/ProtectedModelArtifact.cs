using AiDotNet.Validation;

namespace AiDotNet.Serving.Services;

/// <summary>
/// Represents a protected (encrypted) model artifact along with server-held key material.
/// </summary>
public sealed class ProtectedModelArtifact
{
    public ProtectedModelArtifact(string modelName, string encryptedPath, string keyId, byte[] key, byte[] nonce, string algorithm)
    {
        ModelName = modelName;
        EncryptedPath = encryptedPath;
        KeyId = keyId;
        Guard.NotNull(key);
        Guard.NotNull(nonce);
        Key = key.ToArray();
        Nonce = nonce.ToArray();
        Algorithm = algorithm;
    }

    public string ModelName { get; }
    public string EncryptedPath { get; }
    public string KeyId { get; }
    internal byte[] Key { get; }
    internal byte[] Nonce { get; }
    public string Algorithm { get; }
}

