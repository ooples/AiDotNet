using System.Security.Cryptography;

namespace AiDotNet.FederatedLearning.Cryptography;

/// <summary>
/// HKDF (HMAC-based Key Derivation Function) using HMAC-SHA256.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> HKDF turns an input secret into one or more strong cryptographic keys.
/// We use it to derive pairwise mask seeds from a shared secret.
/// </remarks>
internal static class HkdfSha256
{
    public static byte[] DeriveKey(byte[] inputKeyMaterial, byte[] salt, byte[] info, int length)
    {
        if (inputKeyMaterial == null || inputKeyMaterial.Length == 0)
        {
            throw new ArgumentException("Input key material cannot be null or empty.", nameof(inputKeyMaterial));
        }

        if (length <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(length), "Length must be positive.");
        }

        salt ??= Array.Empty<byte>();
        info ??= Array.Empty<byte>();

        var prk = Extract(salt, inputKeyMaterial);
        try
        {
            return Expand(prk, info, length);
        }
        finally
        {
            Array.Clear(prk, 0, prk.Length);
        }
    }

    private static byte[] Extract(byte[] salt, byte[] inputKeyMaterial)
    {
        using var hmac = new HMACSHA256(salt.Length == 0 ? new byte[32] : salt);
        return hmac.ComputeHash(inputKeyMaterial);
    }

    private static byte[] Expand(byte[] prk, byte[] info, int length)
    {
        const int hashLen = 32;
        int n = (int)Math.Ceiling(length / (double)hashLen);
        if (n > 255)
        {
            throw new ArgumentException("Cannot derive more than 255 blocks of SHA-256 output.", nameof(length));
        }

        using var hmac = new HMACSHA256(prk);
        var okm = new byte[length];
        var previous = Array.Empty<byte>();
        int offset = 0;

        for (int blockIndex = 1; blockIndex <= n; blockIndex++)
        {
            var input = new byte[previous.Length + info.Length + 1];
            Buffer.BlockCopy(previous, 0, input, 0, previous.Length);
            Buffer.BlockCopy(info, 0, input, previous.Length, info.Length);
            input[input.Length - 1] = (byte)blockIndex;

            var t = hmac.ComputeHash(input);
            Array.Clear(input, 0, input.Length);

            int toCopy = Math.Min(hashLen, length - offset);
            Buffer.BlockCopy(t, 0, okm, offset, toCopy);
            offset += toCopy;

            previous = t;
        }

        Array.Clear(previous, 0, previous.Length);
        return okm;
    }
}

