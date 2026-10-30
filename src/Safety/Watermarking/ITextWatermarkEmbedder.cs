namespace AiDotNet.Safety.Watermarking;

/// <summary>
/// A text watermarker that embeds its keyed watermark into finished text.
/// </summary>
/// <remarks>
/// <para>
/// Post-hoc text watermarks rewrite text at sites that admit a meaning-preserving alternative (a synonym, or a
/// contraction) and choose the alternative from the key. They need no access to the language model, so they also
/// mark text from a model you do not run yourself, but they carry fewer bits per word than a generation-time
/// watermark and need longer text to reach the same confidence.
/// </para>
/// <para>
/// <b>For Beginners:</b> this kind of watermarker takes text that is already written and swaps a few words for
/// equivalent ones in a pattern only your key can recognise.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public interface ITextWatermarkEmbedder<T> : ITextWatermarker<T>, IKeyedWatermarker
{
    /// <summary>
    /// Returns <paramref name="text"/> with this watermarker's keyed watermark embedded.
    /// </summary>
    /// <param name="text">The text to watermark.</param>
    /// <returns>The watermarked text. Text without substitution sites is returned unchanged.</returns>
    string EmbedWatermark(string text);
}
