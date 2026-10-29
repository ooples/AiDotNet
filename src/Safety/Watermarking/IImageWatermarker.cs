using AiDotNet.Interfaces;
using AiDotNet.Models;

namespace AiDotNet.Safety.Watermarking;

/// <summary>
/// Interface for keyed image watermarkers that embed an imperceptible watermark in an image and detect it later.
/// </summary>
/// <remarks>
/// <para>
/// A watermarker derives a pseudo-random pattern from its secret key, adds it to a transform of the image
/// (for example the DCT or the Fourier spectrum) at an amplitude too small to see, and detects it by correlating the
/// same transform against the same pattern. Only a watermarker constructed with the same key detects the mark.
/// </para>
/// <para>
/// <b>For Beginners:</b> an image watermarker adds an invisible signature to images it produces, so the same
/// watermarker can later tell whether an image carries that signature, even after mild noise or re-quantization.
/// </para>
/// <para>
/// <b>References:</b>
/// - Cox et al., Secure Spread Spectrum Watermarking for Multimedia (IEEE TIP, 1997)
/// - Wen et al., Tree-Ring Watermarks (2023, arxiv:2305.20030)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public interface IImageWatermarker<T> : IImageSafetyModule<T>
{
    /// <summary>
    /// Returns a copy of <paramref name="image"/> carrying this watermarker's keyed watermark.
    /// </summary>
    /// <param name="image">The image tensor, shaped <c>[H, W]</c>, <c>[C, H, W]</c> or <c>[1, C, H, W]</c>.</param>
    /// <returns>The watermarked image, with the same shape. Values in [0, 1] or [0, 255] stay in that range.</returns>
    Tensor<T> EmbedWatermark(Tensor<T> image);

    /// <summary>
    /// Detects this watermarker's keyed watermark in the given image (0.0 = no evidence, 1.0 = certain).
    /// </summary>
    /// <param name="image">The image tensor to check.</param>
    /// <returns>A watermark detection confidence score.</returns>
    double DetectWatermark(Tensor<T> image);
}