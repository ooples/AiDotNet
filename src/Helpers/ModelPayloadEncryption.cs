using System.Security.Cryptography;
using System.Text;

namespace AiDotNet.Helpers;

/// <summary>
/// Contains the result of encrypting a model payload with AES-256-GCM.
/// </summary>
public sealed class EncryptedPayload
{
    /// <summary>Gets the random salt used for key derivation.</summary>
    public byte[] Salt { get; }

    /// <summary>Gets the 12-byte nonce (IV) used for AES-GCM encryption.</summary>
    public byte[] Nonce { get; }

    /// <summary>Gets the 16-byte GCM authentication tag.</summary>
    public byte[] Tag { get; }

    /// <summary>Gets the encrypted ciphertext.</summary>
    public byte[] Ciphertext { get; }

    public EncryptedPayload(byte[] salt, byte[] nonce, byte[] tag, byte[] ciphertext)
    {
        Salt = salt;
        Nonce = nonce;
        Tag = tag;
        Ciphertext = ciphertext;
    }
}

/// <summary>
/// Provides AES-256-GCM encryption and decryption for AIMF model payloads,
/// with key derivation from license keys via PBKDF2-SHA256.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> This helper encrypts model weights so they can only be loaded
/// with a valid license key. The license key is stretched into a 256-bit AES key using
/// PBKDF2 with a random salt, then used with AES-GCM which provides both confidentiality
/// and authentication (tamper detection).
///
/// The salt, nonce, and authentication tag are stored in the AIMF header alongside the
/// encrypted payload. These values are not secret — they are required for decryption
/// but do not reveal the key or plaintext.
///
/// <b>Note:</b> AES-GCM requires .NET Core 3.0 or later. On .NET Framework 4.7.1,
/// both Encrypt and Decrypt throw <see cref="PlatformNotSupportedException"/>.
/// </remarks>
public static class ModelPayloadEncryption
{
    private const int SaltSize = 16;
    private const int NonceSize = 12;
    private const int TagSize = 16;
    private const int KeySize = 32; // 256 bits
    private const int Pbkdf2Iterations = 210_000;

    /// <summary>
    /// Encrypts a model payload using AES-256-GCM with a key derived from a license key.
    /// </summary>
    /// <param name="payload">The plaintext model data to encrypt.</param>
    /// <param name="licenseKey">The license key used to derive the encryption key.</param>
    /// <param name="aadText">
    /// Additional authenticated data (e.g., TypeName + shapes) that is authenticated
    /// but not encrypted. Prevents payload swapping between different model files.
    /// </param>
    /// <returns>An <see cref="EncryptedPayload"/> containing salt, nonce, tag, and ciphertext.</returns>
    /// <exception cref="ArgumentNullException">Thrown when payload is null.</exception>
    /// <exception cref="ArgumentException">Thrown when licenseKey is null or whitespace.</exception>
    /// <exception cref="PlatformNotSupportedException">Thrown on .NET Framework 4.7.1.</exception>
    public static EncryptedPayload Encrypt(byte[] payload, string licenseKey, string aadText)
    {
        if (payload is null)
        {
            throw new ArgumentNullException(nameof(payload));
        }

        if (string.IsNullOrWhiteSpace(licenseKey))
        {
            throw new ArgumentException("License key cannot be null or empty.", nameof(licenseKey));
        }

#if NET471
        throw new PlatformNotSupportedException(
            "AIMF payload encryption requires .NET Core 3.0 or later.");
#else
        var salt = RandomNumberGenerator.GetBytes(SaltSize);
        var nonce = RandomNumberGenerator.GetBytes(NonceSize);
        var tag = new byte[TagSize];
        var ciphertext = new byte[payload.Length];
        byte[] key = Array.Empty<byte>();

        try
        {
            key = DeriveKey(licenseKey, salt);
            var aad = string.IsNullOrEmpty(aadText)
                ? Array.Empty<byte>()
                : Encoding.UTF8.GetBytes(aadText);

            using var aesGcm = new AesGcm(key, TagSize);
            aesGcm.Encrypt(nonce, payload, ciphertext, tag, aad);

            return new EncryptedPayload(salt, nonce, tag, ciphertext);
        }
        finally
        {
            CryptographicOperations.ZeroMemory(key);
        }
#endif
    }

    /// <summary>
    /// Decrypts an AES-256-GCM encrypted model payload using a license key.
    /// </summary>
    /// <param name="ciphertext">The encrypted payload data.</param>
    /// <param name="licenseKey">The license key used to derive the decryption key.</param>
    /// <param name="salt">The salt used during encryption (from the AIMF header).</param>
    /// <param name="nonce">The nonce used during encryption (from the AIMF header).</param>
    /// <param name="tag">The GCM authentication tag (from the AIMF header).</param>
    /// <param name="aadText">
    /// The same additional authenticated data used during encryption.
    /// Must match exactly or decryption will fail with an authentication error.
    /// </param>
    /// <returns>The decrypted plaintext model data.</returns>
    /// <exception cref="ArgumentNullException">Thrown when ciphertext, salt, nonce, or tag is null.</exception>
    /// <exception cref="ArgumentException">Thrown when licenseKey is null or whitespace.</exception>
    /// <exception cref="CryptographicException">Thrown when the key is wrong or data has been tampered with.</exception>
    /// <exception cref="PlatformNotSupportedException">Thrown on .NET Framework 4.7.1.</exception>
    public static byte[] Decrypt(byte[] ciphertext, string licenseKey, byte[] salt, byte[] nonce, byte[] tag, string aadText)
    {
        if (ciphertext is null)
        {
            throw new ArgumentNullException(nameof(ciphertext));
        }

        if (string.IsNullOrWhiteSpace(licenseKey))
        {
            throw new ArgumentException("License key cannot be null or empty.", nameof(licenseKey));
        }

        if (salt is null)
        {
            throw new ArgumentNullException(nameof(salt));
        }

        if (nonce is null)
        {
            throw new ArgumentNullException(nameof(nonce));
        }

        if (tag is null)
        {
            throw new ArgumentNullException(nameof(tag));
        }

#if NET471
        throw new PlatformNotSupportedException(
            "AIMF payload encryption requires .NET Core 3.0 or later.");
#else
        var plaintext = new byte[ciphertext.Length];
        byte[] key = Array.Empty<byte>();

        try
        {
            key = DeriveKey(licenseKey, salt);
            var aad = string.IsNullOrEmpty(aadText)
                ? Array.Empty<byte>()
                : Encoding.UTF8.GetBytes(aadText);

            using var aesGcm = new AesGcm(key, TagSize);
            aesGcm.Decrypt(nonce, ciphertext, tag, plaintext, aad);

            return plaintext;
        }
        finally
        {
            CryptographicOperations.ZeroMemory(key);
        }
#endif
    }

    /// <summary>
    /// Builds the AAD (Additional Authenticated Data) string from model metadata.
    /// This prevents an attacker from swapping encrypted payloads between different model files.
    /// </summary>
    /// <param name="typeName">The model type name from the AIMF header.</param>
    /// <param name="inputShape">The input shape from the AIMF header.</param>
    /// <param name="outputShape">The output shape from the AIMF header.</param>
    /// <returns>A deterministic string representation of the model metadata for use as AAD.</returns>
    public static string BuildAad(string typeName, int[] inputShape, int[] outputShape)
    {
        var inputStr = inputShape is null || inputShape.Length == 0
            ? ""
            : string.Join(",", inputShape);
        var outputStr = outputShape is null || outputShape.Length == 0
            ? ""
            : string.Join(",", outputShape);
        return $"{typeName ?? string.Empty}|{inputStr}|{outputStr}";
    }

    /// <summary>
    /// Encrypts a model payload using AES-256-GCM with enhanced key derivation that
    /// incorporates the build-time signing key and an optional server-side decryption token.
    /// </summary>
    /// <param name="payload">The plaintext model data to encrypt.</param>
    /// <param name="licenseKey">The license key used to derive the base encryption key.</param>
    /// <param name="aadText">Additional authenticated data for tamper detection.</param>
    /// <param name="decryptionToken">Optional server-side escrow decryption token (Layer 2).</param>
    /// <returns>An <see cref="EncryptedPayload"/> containing salt, nonce, tag, and ciphertext.</returns>
    public static EncryptedPayload EncryptSigned(byte[] payload, string licenseKey, string aadText, byte[]? decryptionToken = null)
    {
        if (payload is null)
        {
            throw new ArgumentNullException(nameof(payload));
        }

        if (string.IsNullOrWhiteSpace(licenseKey))
        {
            throw new ArgumentException("License key cannot be null or empty.", nameof(licenseKey));
        }

#if NET471
        throw new PlatformNotSupportedException(
            "AIMF payload encryption requires .NET Core 3.0 or later.");
#else
        // Integrity check (Layer 3)
        if (!AssemblyIntegrityChecker.VerifyIntegrity())
        {
            throw new CryptographicException(
                "Assembly integrity check failed. The library may have been tampered with.");
        }

        var salt = RandomNumberGenerator.GetBytes(SaltSize);
        var nonce = RandomNumberGenerator.GetBytes(NonceSize);
        var tag = new byte[TagSize];
        var ciphertext = new byte[payload.Length];
        byte[] key = Array.Empty<byte>();

        try
        {
            key = DeriveSignedKey(licenseKey, salt, decryptionToken);
            var aad = string.IsNullOrEmpty(aadText)
                ? Array.Empty<byte>()
                : Encoding.UTF8.GetBytes(aadText);

            using var aesGcm = new AesGcm(key, TagSize);
            aesGcm.Encrypt(nonce, payload, ciphertext, tag, aad);

            return new EncryptedPayload(salt, nonce, tag, ciphertext);
        }
        finally
        {
            CryptographicOperations.ZeroMemory(key);
        }
#endif
    }

    /// <summary>
    /// Decrypts an AES-256-GCM encrypted model payload using enhanced key derivation
    /// that incorporates the build-time signing key and an optional server-side decryption token.
    /// </summary>
    public static byte[] DecryptSigned(
        byte[] ciphertext, string licenseKey, byte[] salt, byte[] nonce, byte[] tag,
        string aadText, byte[]? decryptionToken = null)
    {
        if (ciphertext is null)
        {
            throw new ArgumentNullException(nameof(ciphertext));
        }

        if (string.IsNullOrWhiteSpace(licenseKey))
        {
            throw new ArgumentException("License key cannot be null or empty.", nameof(licenseKey));
        }

        if (salt is null)
        {
            throw new ArgumentNullException(nameof(salt));
        }

        if (nonce is null)
        {
            throw new ArgumentNullException(nameof(nonce));
        }

        if (tag is null)
        {
            throw new ArgumentNullException(nameof(tag));
        }

#if NET471
        throw new PlatformNotSupportedException(
            "AIMF payload encryption requires .NET Core 3.0 or later.");
#else
        // Integrity check (Layer 3)
        if (!AssemblyIntegrityChecker.VerifyIntegrity())
        {
            throw new CryptographicException(
                "Assembly integrity check failed. The library may have been tampered with.");
        }

        var plaintext = new byte[ciphertext.Length];
        byte[] key = Array.Empty<byte>();

        try
        {
            key = DeriveSignedKey(licenseKey, salt, decryptionToken);
            var aad = string.IsNullOrEmpty(aadText)
                ? Array.Empty<byte>()
                : Encoding.UTF8.GetBytes(aadText);

            using var aesGcm = new AesGcm(key, TagSize);
            aesGcm.Decrypt(nonce, ciphertext, tag, plaintext, aad);

            return plaintext;
        }
        finally
        {
            CryptographicOperations.ZeroMemory(key);
        }
#endif
    }

#if !NET471
    /// <summary>
    /// Derives a 256-bit AES key from a license key string and salt using PBKDF2-SHA256.
    /// </summary>
    private static byte[] DeriveKey(string licenseKey, byte[] salt)
    {
        var keyBytes = Encoding.UTF8.GetBytes(licenseKey);
        try
        {
            return Rfc2898DeriveBytes.Pbkdf2(
                password: keyBytes,
                salt: salt,
                iterations: Pbkdf2Iterations,
                hashAlgorithm: HashAlgorithmName.SHA256,
                outputLength: KeySize);
        }
        finally
        {
            CryptographicOperations.ZeroMemory(keyBytes);
        }
    }

    /// <summary>
    /// Derives a 256-bit AES key using enhanced derivation that incorporates the build-time
    /// signing key (Layer 1) and optional decryption token (Layer 2).
    /// </summary>
    /// <remarks>
    /// Key derivation: HMAC-SHA256(PBKDF2(licenseKey, salt), buildKey + decryptionToken)
    /// This ensures that fork builds (no build key) derive a different final key and
    /// cannot decrypt models encrypted by official builds.
    /// </remarks>
    private static byte[] DeriveSignedKey(string licenseKey, byte[] salt, byte[]? decryptionToken)
    {
        byte[] baseKey = Array.Empty<byte>();
        byte[] finalKey = Array.Empty<byte>();

        try
        {
            // Step 1: Standard PBKDF2 derivation
            baseKey = DeriveKey(licenseKey, salt);

            // Step 2: Incorporate build key and decryption token via HMAC
            var buildKey = BuildKeyProvider.GetBuildKey();
            var tokenBytes = decryptionToken ?? Array.Empty<byte>();

            // Combine buildKey + decryptionToken as the HMAC message
            var hmacMessage = new byte[buildKey.Length + tokenBytes.Length];
            if (buildKey.Length > 0)
            {
                Buffer.BlockCopy(buildKey, 0, hmacMessage, 0, buildKey.Length);
            }

            if (tokenBytes.Length > 0)
            {
                Buffer.BlockCopy(tokenBytes, 0, hmacMessage, buildKey.Length, tokenBytes.Length);
            }

            using var hmac = new HMACSHA256(baseKey);
            finalKey = hmac.ComputeHash(hmacMessage);

            // Return a copy since we need to zero baseKey
            var result = new byte[KeySize];
            Buffer.BlockCopy(finalKey, 0, result, 0, KeySize);
            return result;
        }
        finally
        {
            CryptographicOperations.ZeroMemory(baseKey);
            CryptographicOperations.ZeroMemory(finalKey);
        }
    }
#endif
}
