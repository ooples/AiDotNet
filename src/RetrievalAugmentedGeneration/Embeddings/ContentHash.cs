using System.Security.Cryptography;
using System.Text;

namespace AiDotNet.RetrievalAugmentedGeneration.Embeddings;

/// <summary>
/// Provides stable, content-based hashing utilities used for caching and de-duplicating text.
/// </summary>
/// <remarks>
/// <para>
/// The hashes produced here are deterministic across processes and platforms (unlike
/// <see cref="object.GetHashCode"/>, which is randomized per process on modern .NET runtimes).
/// This makes them suitable as cache keys and for ingestion de-duplication where the same
/// content must always map to the same key.
/// </para>
/// <para><b>For Beginners:</b> A content hash is a short "fingerprint" of a piece of text.
///
/// - The same text always produces the same fingerprint.
/// - Different text (almost) always produces a different fingerprint.
/// - It lets us recognize "we have already seen this exact text" without storing the whole text.
///
/// We use it so that repeated pieces of text are only embedded once and then reused.
/// </para>
/// </remarks>
public static class ContentHash
{
    /// <summary>
    /// Computes a stable SHA-256 hash of the supplied text and returns it as a lowercase hex string.
    /// </summary>
    /// <param name="text">The text to hash.</param>
    /// <returns>A 64-character lowercase hexadecimal string representing the SHA-256 digest of the UTF-8 bytes of <paramref name="text"/>.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="text"/> is <c>null</c>.</exception>
    public static string ComputeHash(string text)
    {
        if (text == null)
            throw new ArgumentNullException(nameof(text));

        var bytes = Encoding.UTF8.GetBytes(text);

        // SHA256 is available on net471, net8.0 and net10.0.
        using var sha256 = SHA256.Create();
        var hashBytes = sha256.ComputeHash(bytes);

        return ToHex(hashBytes);
    }

    /// <summary>
    /// Converts a byte array to a lowercase hexadecimal string.
    /// </summary>
    /// <param name="bytes">The bytes to convert.</param>
    /// <returns>A lowercase hexadecimal representation of <paramref name="bytes"/>.</returns>
    private static string ToHex(byte[] bytes)
    {
        // Manual hex conversion keeps behavior identical across all target frameworks
        // (Convert.ToHexString is not available on net471).
        const string hexChars = "0123456789abcdef";
        var chars = new char[bytes.Length * 2];
        for (int i = 0; i < bytes.Length; i++)
        {
            var b = bytes[i];
            chars[i * 2] = hexChars[b >> 4];
            chars[(i * 2) + 1] = hexChars[b & 0x0F];
        }

        return new string(chars);
    }
}
