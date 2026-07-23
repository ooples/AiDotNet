using System.Security.Cryptography;

namespace AiDotNet.FederatedLearning.TEE;

/// <summary>
/// Internal helper providing authenticated AES-256 encryption for TEE data sealing.
/// </summary>
/// <remarks>
/// On .NET Core 3.0+ / .NET 5+, uses AES-256-GCM (hardware-accelerated AEAD).
/// On .NET Framework 4.7.1, falls back to AES-256-CBC + HMAC-SHA256 (encrypt-then-MAC).
/// Both produce the same wire format: iv/nonce || tag/mac || ciphertext.
/// </remarks>
internal static class TeeAesHelper
{
    /// <summary>
    /// Encrypts plaintext with AES-256 authenticated encryption.
    /// </summary>
    /// <param name="key">256-bit (32-byte) encryption key.</param>
    /// <param name="plaintext">Data to encrypt.</param>
    /// <returns>Encrypted data: iv || tag || ciphertext.</returns>
    public static byte[] Encrypt(byte[] key, byte[] plaintext)
    {
        if (key is null || key.Length != 32)
        {
            throw new ArgumentException("Key must be 32 bytes (AES-256).", nameof(key));
        }

        if (plaintext is null || plaintext.Length == 0)
        {
            throw new ArgumentException("Plaintext must not be null or empty.", nameof(plaintext));
        }

#if NETCOREAPP3_0_OR_GREATER || NET5_0_OR_GREATER
        return EncryptGcm(key, plaintext);
#else
        return EncryptCbcHmac(key, plaintext);
#endif
    }

    /// <summary>
    /// Decrypts data encrypted by <see cref="Encrypt"/>.
    /// </summary>
    /// <param name="key">256-bit (32-byte) encryption key.</param>
    /// <param name="encryptedData">iv || tag || ciphertext.</param>
    /// <returns>Decrypted plaintext.</returns>
    public static byte[] Decrypt(byte[] key, byte[] encryptedData)
    {
        if (key is null || key.Length != 32)
        {
            throw new ArgumentException("Key must be 32 bytes (AES-256).", nameof(key));
        }

#if NETCOREAPP3_0_OR_GREATER || NET5_0_OR_GREATER
        return DecryptGcm(key, encryptedData);
#else
        return DecryptCbcHmac(key, encryptedData);
#endif
    }

#if NETCOREAPP3_0_OR_GREATER || NET5_0_OR_GREATER
    private static byte[] EncryptGcm(byte[] key, byte[] plaintext)
    {
        const int nonceLength = 12;
        const int tagLength = 16;

        var nonce = new byte[nonceLength];
        using (var rng = RandomNumberGenerator.Create())
        {
            rng.GetBytes(nonce);
        }

        var ciphertext = new byte[plaintext.Length];
        var tag = new byte[tagLength];

        using (var aes = new AesGcm(key, tagLength))
        {
            aes.Encrypt(nonce, plaintext, ciphertext, tag);
        }

        // nonce(12) || tag(16) || ciphertext
        var result = new byte[nonceLength + tagLength + ciphertext.Length];
        Buffer.BlockCopy(nonce, 0, result, 0, nonceLength);
        Buffer.BlockCopy(tag, 0, result, nonceLength, tagLength);
        Buffer.BlockCopy(ciphertext, 0, result, nonceLength + tagLength, ciphertext.Length);

        return result;
    }

    private static byte[] DecryptGcm(byte[] key, byte[] encryptedData)
    {
        const int nonceLength = 12;
        const int tagLength = 16;
        int minLength = nonceLength + tagLength + 1;

        if (encryptedData is null || encryptedData.Length < minLength)
        {
            throw new ArgumentException($"Encrypted data must be at least {minLength} bytes.", nameof(encryptedData));
        }

        var nonce = new byte[nonceLength];
        var tag = new byte[tagLength];
        int ciphertextLength = encryptedData.Length - nonceLength - tagLength;
        var ciphertext = new byte[ciphertextLength];

        Buffer.BlockCopy(encryptedData, 0, nonce, 0, nonceLength);
        Buffer.BlockCopy(encryptedData, nonceLength, tag, 0, tagLength);
        Buffer.BlockCopy(encryptedData, nonceLength + tagLength, ciphertext, 0, ciphertextLength);

        var plaintext = new byte[ciphertextLength];

        using (var aes = new AesGcm(key, tagLength))
        {
            aes.Decrypt(nonce, ciphertext, tag, plaintext);
        }

        return plaintext;
    }
#else
    // .NET Framework fallback: AES-256-CBC + HMAC-SHA256 (encrypt-then-MAC)
    // Wire format: iv(16) || mac(32) || ciphertext (PKCS7 padded)
    // To maintain compatible header layout: we use iv(16) as "nonce" slot,
    // mac(32) as "tag" slot, then ciphertext.

    private static byte[] EncryptCbcHmac(byte[] key, byte[] plaintext)
    {
        // Split 32-byte key into 16 bytes for AES, 16 bytes for HMAC derivation
        var encKey = new byte[32]; // Full key for AES-256
        Buffer.BlockCopy(key, 0, encKey, 0, 32);

        // Derive a separate MAC key using SHA-256(key || "MAC")
        byte[] macKeyMaterial;
        using (var sha = SHA256.Create())
        {
            var macInput = new byte[key.Length + 3];
            Buffer.BlockCopy(key, 0, macInput, 0, key.Length);
            macInput[key.Length] = (byte)'M';
            macInput[key.Length + 1] = (byte)'A';
            macInput[key.Length + 2] = (byte)'C';
            macKeyMaterial = sha.ComputeHash(macInput);
        }

        byte[] iv;
        byte[] ciphertext;

        using (var aes = Aes.Create())
        {
            aes.KeySize = 256;
            aes.Key = encKey;
            aes.Mode = CipherMode.CBC;
            aes.Padding = PaddingMode.PKCS7;
            aes.GenerateIV();
            iv = aes.IV;

            using (var encryptor = aes.CreateEncryptor())
            {
                ciphertext = encryptor.TransformFinalBlock(plaintext, 0, plaintext.Length);
            }
        }

        // Compute MAC over iv || ciphertext
        byte[] mac;
        using (var hmac = new HMACSHA256(macKeyMaterial))
        {
            var macData = new byte[iv.Length + ciphertext.Length];
            Buffer.BlockCopy(iv, 0, macData, 0, iv.Length);
            Buffer.BlockCopy(ciphertext, 0, macData, iv.Length, ciphertext.Length);
            mac = hmac.ComputeHash(macData);
        }

        // iv(16) || mac(32) || ciphertext
        var result = new byte[iv.Length + mac.Length + ciphertext.Length];
        Buffer.BlockCopy(iv, 0, result, 0, iv.Length);
        Buffer.BlockCopy(mac, 0, result, iv.Length, mac.Length);
        Buffer.BlockCopy(ciphertext, 0, result, iv.Length + mac.Length, ciphertext.Length);

        Array.Clear(encKey, 0, encKey.Length);
        Array.Clear(macKeyMaterial, 0, macKeyMaterial.Length);

        return result;
    }

    private static byte[] DecryptCbcHmac(byte[] key, byte[] encryptedData)
    {
        const int ivLength = 16;
        const int macLength = 32;
        int minLength = ivLength + macLength + 1;

        if (encryptedData is null || encryptedData.Length < minLength)
        {
            throw new ArgumentException($"Encrypted data must be at least {minLength} bytes.", nameof(encryptedData));
        }

        var encKey = new byte[32];
        Buffer.BlockCopy(key, 0, encKey, 0, 32);

        byte[] macKeyMaterial;
        using (var sha = SHA256.Create())
        {
            var macInput = new byte[key.Length + 3];
            Buffer.BlockCopy(key, 0, macInput, 0, key.Length);
            macInput[key.Length] = (byte)'M';
            macInput[key.Length + 1] = (byte)'A';
            macInput[key.Length + 2] = (byte)'C';
            macKeyMaterial = sha.ComputeHash(macInput);
        }

        var iv = new byte[ivLength];
        var storedMac = new byte[macLength];
        int ciphertextLength = encryptedData.Length - ivLength - macLength;
        var ciphertext = new byte[ciphertextLength];

        Buffer.BlockCopy(encryptedData, 0, iv, 0, ivLength);
        Buffer.BlockCopy(encryptedData, ivLength, storedMac, 0, macLength);
        Buffer.BlockCopy(encryptedData, ivLength + macLength, ciphertext, 0, ciphertextLength);

        // Verify MAC first (encrypt-then-MAC)
        byte[] computedMac;
        using (var hmac = new HMACSHA256(macKeyMaterial))
        {
            var macData = new byte[iv.Length + ciphertext.Length];
            Buffer.BlockCopy(iv, 0, macData, 0, iv.Length);
            Buffer.BlockCopy(ciphertext, 0, macData, iv.Length, ciphertext.Length);
            computedMac = hmac.ComputeHash(macData);
        }

        if (!ConstantTimeEquals(storedMac, computedMac))
        {
            throw new CryptographicException("MAC verification failed. Data may have been tampered with.");
        }

        byte[] plaintext;
        using (var aes = Aes.Create())
        {
            aes.KeySize = 256;
            aes.Key = encKey;
            aes.IV = iv;
            aes.Mode = CipherMode.CBC;
            aes.Padding = PaddingMode.PKCS7;

            using (var decryptor = aes.CreateDecryptor())
            {
                plaintext = decryptor.TransformFinalBlock(ciphertext, 0, ciphertext.Length);
            }
        }

        Array.Clear(encKey, 0, encKey.Length);
        Array.Clear(macKeyMaterial, 0, macKeyMaterial.Length);

        return plaintext;
    }

    private static bool ConstantTimeEquals(byte[] a, byte[] b)
    {
        if (a.Length != b.Length) return false;

        int diff = 0;
        for (int i = 0; i < a.Length; i++)
        {
            diff |= a[i] ^ b[i];
        }

        return diff == 0;
    }
#endif
}
