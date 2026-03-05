namespace AiDotNet.Enums;

/// <summary>
/// Specifies the encryption scheme applied to the model payload within an AIMF envelope.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> When a model is saved with encryption enabled, the payload (model weights)
/// is encrypted using a license key. The header remains plaintext so tools can inspect model metadata
/// (type, shapes, whether it's encrypted) without needing the key. Only the actual model data
/// requires a valid license key to decrypt and load.
/// </remarks>
public enum PayloadEncryptionScheme
{
    /// <summary>
    /// Payload is stored as plaintext (no encryption).
    /// </summary>
    None = 0,

    /// <summary>
    /// Payload is encrypted using AES-256-GCM with a key derived from a license key via PBKDF2-SHA256.
    /// Requires .NET Core 3.0 or later (not available on .NET Framework 4.7.1).
    /// </summary>
    AesGcm256 = 1,

    /// <summary>
    /// Payload is encrypted using AES-256-GCM with enhanced key derivation that incorporates a
    /// build-time signing key and optional server-side decryption token.
    /// Files encrypted with this scheme can only be decrypted by official builds.
    /// </summary>
    AesGcm256Signed = 2
}
