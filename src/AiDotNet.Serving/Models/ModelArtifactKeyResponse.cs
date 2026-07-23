namespace AiDotNet.Serving.Models;

/// <summary>
/// Response containing information required to decrypt an encrypted model artifact.
/// </summary>
public class ModelArtifactKeyResponse
{
    /// <summary>
    /// Gets or sets the key identifier.
    /// </summary>
    public string KeyId { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the encryption algorithm name.
    /// </summary>
    public string Algorithm { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the base64-encoded symmetric key.
    /// </summary>
    public string KeyBase64 { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the base64-encoded nonce.
    /// </summary>
    public string NonceBase64 { get; set; } = string.Empty;
}

