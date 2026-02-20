using AiDotNet.Interfaces;
using AiDotNet.Models;

namespace AiDotNet.Safety.Watermarking;

/// <summary>
/// Interface for image watermarking modules that embed and detect watermarks in images.
/// </summary>
/// <remarks>
/// <para>
/// Image watermarkers embed imperceptible watermarks in images using frequency domain
/// (DCT/DWT), neural encoder-decoder, or spatial domain techniques. The watermark
/// survives common transformations like compression, resizing, and cropping.
/// </para>
/// <para>
/// <b>For Beginners:</b> An image watermarker adds an invisible signature to images.
/// Even if someone screenshots, crops, or compresses the image, the watermark can
/// still be detected to prove the image was AI-generated.
/// </para>
/// <para>
/// <b>References:</b>
/// - SynthID-Image: Internet-scale image watermarking (Google DeepMind, 2025, arxiv:2510.09263)
/// - Watermarking survey: unified framework (2025, arxiv:2504.03765)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public interface IImageWatermarker<T> : IImageSafetyModule<T>
{
    /// <summary>
    /// Detects the watermark confidence score in the given image (0.0 = no watermark, 1.0 = certain).
    /// </summary>
    /// <param name="image">The image tensor to check.</param>
    /// <returns>A watermark detection confidence score.</returns>
    double DetectWatermark(Tensor<T> image);
}
