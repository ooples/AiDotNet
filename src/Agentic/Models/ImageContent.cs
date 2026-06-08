namespace AiDotNet.Agentic.Models;

/// <summary>
/// An image content part within a <see cref="ChatMessage"/>, supplied either as raw bytes or as a URI.
/// </summary>
/// <remarks>
/// <para>
/// Multimodal models accept images alongside text. An image can be embedded directly as bytes
/// (which providers typically base64-encode on the wire) or referenced by a URL the provider fetches.
/// The format is a type-safe <see cref="ImageMediaType"/> rather than a raw MIME string. Use the
/// <see cref="FromBytes"/> and <see cref="FromUri"/> factories to construct instances.
/// </para>
/// <para><b>For Beginners:</b> If you want the model to "look at" a picture, wrap it in one of these.
/// You either hand over the actual image bytes (e.g., a PNG you loaded from disk) via
/// <see cref="FromBytes"/>, or point at a web address via <see cref="FromUri"/>. The image type is
/// chosen from the <see cref="ImageMediaType"/> enum, so you can't misspell it.
/// </para>
/// </remarks>
public sealed class ImageContent : AiContent
{
    private ImageContent(byte[]? data, string? uri, ImageMediaType? mediaType)
    {
        Data = data;
        Uri = uri;
        MediaType = mediaType;
    }

    /// <summary>
    /// Gets the raw image bytes, or <c>null</c> when the image is referenced by <see cref="Uri"/>.
    /// </summary>
    public byte[]? Data { get; }

    /// <summary>
    /// Gets the image URI, or <c>null</c> when the image is supplied inline via <see cref="Data"/>.
    /// </summary>
    public string? Uri { get; }

    /// <summary>
    /// Gets the image format, or <c>null</c> when referenced by URI with an unspecified format.
    /// </summary>
    public ImageMediaType? MediaType { get; }

    /// <summary>
    /// Gets a value indicating whether the image is supplied inline as bytes (rather than by URI).
    /// </summary>
    public bool HasData => Data is not null;

    /// <summary>
    /// Creates an image content part from raw bytes.
    /// </summary>
    /// <param name="data">The image bytes.</param>
    /// <param name="mediaType">The image format.</param>
    /// <returns>A new <see cref="ImageContent"/> wrapping the supplied bytes.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="data"/> is <c>null</c>.</exception>
    public static ImageContent FromBytes(byte[] data, ImageMediaType mediaType)
    {
        Guard.NotNull(data);
        return new ImageContent(data, uri: null, mediaType);
    }

    /// <summary>
    /// Creates an image content part that references an image by URI.
    /// </summary>
    /// <param name="uri">The image URI the provider will fetch.</param>
    /// <param name="mediaType">The image format, when known; <c>null</c> lets the provider infer it.</param>
    /// <returns>A new <see cref="ImageContent"/> referencing the supplied URI.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="uri"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">Thrown when <paramref name="uri"/> is empty/whitespace.</exception>
    public static ImageContent FromUri(string uri, ImageMediaType? mediaType = null)
    {
        Guard.NotNullOrWhiteSpace(uri);
        return new ImageContent(data: null, uri, mediaType);
    }
}
