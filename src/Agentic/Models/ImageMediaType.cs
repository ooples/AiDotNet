namespace AiDotNet.Agentic.Models;

/// <summary>
/// The image formats accepted by multimodal chat models.
/// </summary>
/// <remarks>
/// <para>
/// Vision-capable providers accept a small, fixed set of image formats. Modeling them as an enum (rather
/// than a raw MIME string like <c>"image/png"</c>) prevents typos such as <c>"image/pngg"</c> from
/// compiling and failing at request time, and gives callers autocomplete for the valid choices. The
/// wire-format MIME string is derived from the enum via <see cref="ImageMediaTypeExtensions.ToMimeType"/>.
/// </para>
/// <para><b>For Beginners:</b> Instead of passing a free-text string for the image type (easy to misspell),
/// you pick from a known list: PNG, JPEG, GIF, or WebP. The library turns your choice into the exact
/// text the provider expects.
/// </para>
/// </remarks>
public enum ImageMediaType
{
    /// <summary>PNG image (<c>image/png</c>).</summary>
    Png,

    /// <summary>JPEG image (<c>image/jpeg</c>).</summary>
    Jpeg,

    /// <summary>GIF image (<c>image/gif</c>).</summary>
    Gif,

    /// <summary>WebP image (<c>image/webp</c>).</summary>
    Webp
}
