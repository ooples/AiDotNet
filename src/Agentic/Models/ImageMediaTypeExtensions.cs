namespace AiDotNet.Agentic.Models;

/// <summary>
/// Conversions between <see cref="ImageMediaType"/> and its wire-format MIME string.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Providers expect image types written as <c>image/png</c>, <c>image/jpeg</c>,
/// etc. These helpers translate between the type-safe enum your code uses and that wire text.
/// </para>
/// </remarks>
public static class ImageMediaTypeExtensions
{
    /// <summary>
    /// Converts an <see cref="ImageMediaType"/> to its MIME string (e.g. <c>image/png</c>).
    /// </summary>
    /// <param name="mediaType">The media type.</param>
    /// <returns>The MIME type string.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when the value is not a defined member.</exception>
    public static string ToMimeType(this ImageMediaType mediaType) => mediaType switch
    {
        ImageMediaType.Png => "image/png",
        ImageMediaType.Jpeg => "image/jpeg",
        ImageMediaType.Gif => "image/gif",
        ImageMediaType.Webp => "image/webp",
        _ => throw new ArgumentOutOfRangeException(nameof(mediaType), mediaType, "Unknown image media type.")
    };

    /// <summary>
    /// Parses a MIME string into an <see cref="ImageMediaType"/>.
    /// </summary>
    /// <param name="mimeType">The MIME string (case-insensitive), e.g. <c>image/jpeg</c>. <c>image/jpg</c> is also accepted.</param>
    /// <param name="mediaType">The parsed media type when recognized.</param>
    /// <returns><c>true</c> when the MIME string maps to a known media type; otherwise <c>false</c>.</returns>
    public static bool TryParseMimeType(string? mimeType, out ImageMediaType mediaType)
    {
        switch (mimeType?.Trim().ToLowerInvariant())
        {
            case "image/png":
                mediaType = ImageMediaType.Png;
                return true;
            case "image/jpeg":
            case "image/jpg":
                mediaType = ImageMediaType.Jpeg;
                return true;
            case "image/gif":
                mediaType = ImageMediaType.Gif;
                return true;
            case "image/webp":
                mediaType = ImageMediaType.Webp;
                return true;
            default:
                // Return the enum default on failure rather than a specific type, so a false return is not
                // mistaken for a parsed PNG.
                mediaType = default;
                return false;
        }
    }
}
