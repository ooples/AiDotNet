namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for semantic segmentation models that assign a class label to every pixel.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Semantic segmentation classifies every pixel in an image into a predefined category without
/// distinguishing between individual object instances. For example, all cars are labeled "car"
/// regardless of how many there are.
/// </para>
/// <para>
/// <b>For Beginners:</b> Semantic segmentation answers "what is this pixel?" for every pixel.
///
/// Example output for a street scene:
/// - All road pixels labeled "road"
/// - All sky pixels labeled "sky"
/// - All car pixels labeled "car" (but car #1 and car #2 are not distinguished)
///
/// Models implementing this interface:
/// - SegFormer (lightweight transformer, NeurIPS 2021)
/// - SegNeXt (efficient CNN+attention, NeurIPS 2022)
/// - InternImage (large-scale CNN, CVPR 2023)
/// - ViT-Adapter, ViT-CoMer (transformer adapters)
/// - DiffCut, DiffSeg (diffusion-based, zero-shot)
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("SemanticSegmentation")]
public interface ISemanticSegmentation<T> : ISegmentationModel<T>
{
    /// <summary>
    /// Gets the per-pixel class map from the most recent segmentation.
    /// </summary>
    /// <returns>Integer class label tensor [H, W] where each value is a class index.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This gives you a simplified version of the segmentation output
    /// where each pixel is a single number representing its class (e.g., 0 = background,
    /// 1 = road, 2 = car). This is the argmax of the logits from <see cref="ISegmentationModel{T}.Segment"/>.
    /// </para>
    /// </remarks>
    Tensor<T> GetClassMap(Tensor<T> image);

    /// <summary>
    /// Gets the class-wise confidence scores for the segmentation.
    /// </summary>
    /// <param name="image">Input image tensor [C, H, W] or [B, C, H, W].</param>
    /// <returns>Probability map tensor [numClasses, H, W] or [B, numClasses, H, W] with values in [0, 1].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Instead of just the predicted class, this gives you a confidence
    /// score (0-100%) for every class at every pixel. Useful for understanding model uncertainty.
    /// </para>
    /// </remarks>
    Tensor<T> GetProbabilityMap(Tensor<T> image);
}
