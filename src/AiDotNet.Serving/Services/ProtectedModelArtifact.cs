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
        Key = (key ?? throw new ArgumentNullException(nameof(key))).ToArray();
        Nonce = (nonce ?? throw new ArgumentNullException(nameof(nonce))).ToArray();
        Algorithm = algorithm;
    }

    public string ModelName { get; }
    public string EncryptedPath { get; }
    public string KeyId { get; }
    internal byte[] Key { get; }
    internal byte[] Nonce { get; }
    public string Algorithm { get; }
}

