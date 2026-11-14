using System.Security.Cryptography;

namespace AiDotNet.FederatedLearning.Cryptography;

/// <summary>
/// Authenticated encryption (<c>AE.enc</c> / <c>AE.dec</c> of Bonawitz et al. 2017, section 3.6) for the secret shares
/// clients exchange through the server: AES-256-CTR encrypt-then-MAC with HMAC-SHA256.
/// </summary>
/// <remarks>
/// <para>
/// The paper's prototype uses AES-GCM. <c>AesGcm</c> does not exist on .NET Framework or .NET Standard 2.0, so this
/// uses the generic composition of a CPA-secure cipher with a strong MAC (encrypt-then-MAC), which gives
/// IND-CCA2 / INT-CTXT authenticated encryption (Bellare and Namprempre, 2000) on every target. The 64-byte key is
/// split into a 32-byte AES key and a 32-byte HMAC key. Wire format: <c>IV (16) || ciphertext || tag (32)</c>, the tag
/// covering <c>IV || ciphertext</c>.
/// </para>
/// <para><b>For Beginners:</b> A client's secret shares travel through the server, which must neither read nor
/// alter them. Encryption hides them; the tag lets the recipient detect any tampering.</para>
/// </remarks>
internal static class AuthenticatedShareCipher
{
    internal const int KeyLength = 64;
    private const int CipherKeyLength = 32;
    private const int TagLength = 32;

    /// <summary>Encrypts <paramref name="plaintext"/> under the 64-byte <paramref name="key"/> with a random IV.</summary>
    internal static byte[] Encrypt(byte[] key, byte[] plaintext)
    {
        ValidateKey(key);
        if (plaintext is null)
        {
            throw new ArgumentNullException(nameof(plaintext));
        }

        SplitKey(key, out var cipherKey, out var macKey);
        try
        {
            var iv = new byte[AesCtr.BlockSize];
            using (var rng = RandomNumberGenerator.Create())
            {
                rng.GetBytes(iv);
            }

            var output = new byte[AesCtr.BlockSize + plaintext.Length + TagLength];
            Buffer.BlockCopy(iv, 0, output, 0, AesCtr.BlockSize);
            var body = (byte[])plaintext.Clone();
            AesCtr.Transform(cipherKey, iv, body);
            Buffer.BlockCopy(body, 0, output, AesCtr.BlockSize, body.Length);

            using var hmac = new HMACSHA256(macKey);
            var tag = hmac.ComputeHash(output, 0, AesCtr.BlockSize + body.Length);
            Buffer.BlockCopy(tag, 0, output, AesCtr.BlockSize + body.Length, TagLength);
            return output;
        }
        finally
        {
            Array.Clear(cipherKey, 0, cipherKey.Length);
            Array.Clear(macKey, 0, macKey.Length);
        }
    }

    /// <summary>
    /// Verifies the tag and decrypts. Returns <c>false</c> (and no plaintext) when the ciphertext is malformed or was
    /// not produced under <paramref name="key"/>; the caller treats that as a protocol abort.
    /// </summary>
    internal static bool TryDecrypt(byte[] key, byte[] ciphertext, out byte[] plaintext)
    {
        ValidateKey(key);
        plaintext = Array.Empty<byte>();
        if (ciphertext is null || ciphertext.Length < AesCtr.BlockSize + TagLength)
        {
            return false;
        }

        SplitKey(key, out var cipherKey, out var macKey);
        try
        {
            int bodyLength = ciphertext.Length - AesCtr.BlockSize - TagLength;
            byte[] expectedTag;
            using (var hmac = new HMACSHA256(macKey))
            {
                expectedTag = hmac.ComputeHash(ciphertext, 0, AesCtr.BlockSize + bodyLength);
            }

            if (!FixedTimeEquals(expectedTag, 0, ciphertext, AesCtr.BlockSize + bodyLength, TagLength))
            {
                return false;
            }

            var iv = new byte[AesCtr.BlockSize];
            Buffer.BlockCopy(ciphertext, 0, iv, 0, AesCtr.BlockSize);
            var body = new byte[bodyLength];
            Buffer.BlockCopy(ciphertext, AesCtr.BlockSize, body, 0, bodyLength);
            AesCtr.Transform(cipherKey, iv, body);
            plaintext = body;
            return true;
        }
        finally
        {
            Array.Clear(cipherKey, 0, cipherKey.Length);
            Array.Clear(macKey, 0, macKey.Length);
        }
    }

    private static void ValidateKey(byte[] key)
    {
        if (key is null)
        {
            throw new ArgumentNullException(nameof(key));
        }

        if (key.Length != KeyLength)
        {
            throw new ArgumentException($"The share cipher key must be {KeyLength} bytes.", nameof(key));
        }
    }

    private static void SplitKey(byte[] key, out byte[] cipherKey, out byte[] macKey)
    {
        cipherKey = new byte[CipherKeyLength];
        macKey = new byte[KeyLength - CipherKeyLength];
        Buffer.BlockCopy(key, 0, cipherKey, 0, CipherKeyLength);
        Buffer.BlockCopy(key, CipherKeyLength, macKey, 0, macKey.Length);
    }

    // CryptographicOperations.FixedTimeEquals is .NET Core 2.1+; this is the same constant-time comparison.
    private static bool FixedTimeEquals(byte[] left, int leftOffset, byte[] right, int rightOffset, int length)
    {
        int difference = 0;
        for (int i = 0; i < length; i++)
        {
            difference |= left[leftOffset + i] ^ right[rightOffset + i];
        }

        return difference == 0;
    }
}
