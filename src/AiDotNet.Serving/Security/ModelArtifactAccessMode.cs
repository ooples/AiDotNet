namespace AiDotNet.Serving.Security;

/// <summary>
/// Model artifact access modes for distributed deployments.
/// </summary>
public enum ModelArtifactAccessMode
{
    /// <summary>
    /// Option A: model never leaves the server; clients use the REST API for inference.
    /// </summary>
    ServerOnly,

    /// <summary>
    /// Option B: clients can download the model artifact (plaintext).
    /// </summary>
    DirectDownload,

    /// <summary>
    /// Option C: clients download an encrypted artifact and must pass attestation to obtain the decryption key.
    /// </summary>
    EncryptedWithAttestedKeyRelease
}

