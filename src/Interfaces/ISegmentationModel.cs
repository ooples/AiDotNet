namespace AiDotNet.Interfaces;

/// <summary>
/// Base interface for all image segmentation models that classify pixels into categories.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Image segmentation assigns a label to every pixel in an image, enabling fine-grained
/// scene understanding. This is the foundation interface that all segmentation model types
/// (semantic, instance, panoptic, medical, etc.) extend.
/// </para>
/// <para>
/// <b>For Beginners:</b> Segmentation is like coloring a picture where each color represents
/// a different object or category. Unlike object detection which draws boxes around things,
/// segmentation gives you the exact pixel-level outline.
///
/// Types of segmentation:
/// - Semantic: "Which pixels are road? Which are sky?" (classes, no instances)
/// - Instance: "Where is car #1? Car #2?" (individual objects)
/// - Panoptic: Both semantic + instance together
/// - Interactive: You point/click and the model segments what you indicated
///
/// Common use cases:
/// - Autonomous driving (road, lane, obstacle detection)
/// - Medical imaging (organ and tumor boundaries)
/// - Photo editing (background removal, object selection)
/// - Agriculture (crop vs. weed detection from drones)
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("SegmentationModel")]
public interface ISegmentationModel<T> : IFullModel<T, Tensor<T>, Tensor<T>>
{
    /// <summary>
    /// Gets the number of segmentation classes this model predicts.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is how many different "things" the model can identify.
    /// For example, a Cityscapes model has 19 classes (road, sidewalk, building, car, person, etc.),
    /// while ADE20K has 150 classes.
    /// </para>
    /// </remarks>
    int NumClasses { get; }

    /// <summary>
    /// Gets the expected input image height in pixels.
    /// </summary>
    int InputHeight { get; }

    /// <summary>
    /// Gets the expected input image width in pixels.
    /// </summary>
    int InputWidth { get; }

    /// <summary>
    /// Gets whether this model is running in ONNX inference mode.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When true, the model uses pre-trained ONNX weights for fast inference.
    /// When false, the model runs in native mode and supports both training and inference.
    /// </para>
    /// </remarks>
    bool IsOnnxMode { get; }

    /// <summary>
    /// Segments an image, producing a per-pixel class prediction map.
    /// </summary>
    /// <param name="image">Input image tensor [C, H, W] or [B, C, H, W].</param>
    /// <returns>Segmentation logits tensor [numClasses, H, W] or [B, numClasses, H, W].
    /// Take argmax along the class dimension to get the predicted class per pixel.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Pass in your image and get back a map where each pixel has
    /// a score for every possible class. The class with the highest score at each pixel
    /// is the model's prediction.
    /// </para>
    /// </remarks>
    Tensor<T> Segment(Tensor<T> image);
}
