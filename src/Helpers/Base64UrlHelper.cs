namespace AiDotNet.Helpers;

/// <summary>
/// Minimal Base64URL (RFC 4648 §5) encode/decode helper used by the asymmetric license
/// token path. Base64URL uses '-'/'_' instead of '+'/'/' and omits padding.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> License tokens are pieces of text that must survive being placed in
/// URLs, environment variables, and files without special characters causing trouble. Base64URL
/// is a text encoding that avoids the characters (<c>+</c>, <c>/</c>, <c>=</c>) that cause issues
/// in those places. This helper converts between raw bytes and that safe text form.
/// </remarks>
internal static class Base64UrlHelper
{
    /// <summary>
    /// Encodes bytes as Base64URL (no padding).
    /// </summary>
    internal static string Encode(byte[] bytes)
    {
        if (bytes is null || bytes.Length == 0) return string.Empty;
        return Convert.ToBase64String(bytes)
            .Replace('+', '-')
            .Replace('/', '_')
            .TrimEnd('=');
    }

    /// <summary>
    /// Decodes a Base64URL string (with or without padding) to bytes. Throws
    /// <see cref="FormatException"/> on malformed input.
    /// </summary>
    internal static byte[] Decode(string base64Url)
    {
        if (base64Url is null) throw new ArgumentNullException(nameof(base64Url));

        string standard = base64Url
            .Replace('-', '+')
            .Replace('_', '/');

        switch (standard.Length % 4)
        {
            case 2: standard += "=="; break;
            case 3: standard += "="; break;
            case 1: throw new FormatException("Invalid Base64URL length.");
        }

        return Convert.FromBase64String(standard);
    }
}
