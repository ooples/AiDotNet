namespace AiDotNet.ComputerVision.Segmentation.Salient;

/// <summary>
/// Separates the foreground object of an image from its background, as image-to-3D pipelines need before they fit
/// an object (DreamGaussian's <c>process.py</c> runs <c>rembg</c> for this).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <b>For Beginners:</b> give it a photo and it answers, for every pixel, how sure it is that the pixel belongs to the
/// main object (1) rather than the background (0).
/// </remarks>
public interface IForegroundSegmenter<T>
{
    /// <summary>The foreground alpha of an image.</summary>
    /// <param name="image">An RGB image [1, 3, H, W] with values in [0, 1].</param>
    /// <returns>The alpha matte [H, W] in [0, 1].</returns>
    Tensor<T> Segment(Tensor<T> image);
}
