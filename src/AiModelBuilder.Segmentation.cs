using AiDotNet.ComputerVision.Segmentation.Common;
using AiDotNet.ComputerVision.Segmentation.InstanceSegmentation;
using AiDotNet.Interfaces;
using AiDotNet.Tensors;

namespace AiDotNet;

/// <summary>
/// Segmentation-specific fields and configuration for AiModelBuilder.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input type.</typeparam>
/// <typeparam name="TOutput">The output type.</typeparam>
public partial class AiModelBuilder<T, TInput, TOutput>
{
    private SegmentationVisualizationConfig? _segmentationVisualizationConfig;

    /// <summary>
    /// Configures visualization settings for segmentation overlays.
    /// </summary>
    /// <param name="config">Visualization configuration. If null, uses defaults (alpha=0.5, contours on).</param>
    /// <returns>The builder for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> After segmenting an image, you often want to see the results overlaid
    /// on the original image with colored regions. This configures how that overlay looks — colors,
    /// transparency, labels, and contours.</para>
    /// </remarks>
    public AiModelBuilder<T, TInput, TOutput> ConfigureSegmentationVisualization(
        SegmentationVisualizationConfig? config = null)
    {
        _segmentationVisualizationConfig = config ?? new SegmentationVisualizationConfig();
        return this;
    }

    /// <summary>
    /// Gets the configured segmentation visualization settings.
    /// </summary>
    internal SegmentationVisualizationConfig? SegmentationVisualization => _segmentationVisualizationConfig;
}

/// <summary>
/// Extension methods for image segmentation operations through the AiModelBuilder facade.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> These methods let you perform segmentation on images after configuring
/// a model with <see cref="AiModelBuilder{T, TInput, TOutput}.ConfigureModel"/>. Each method
/// detects what kind of segmentation model you configured and calls the appropriate interface.
///
/// Usage pattern:
/// <code>
/// var builder = new AiModelBuilder&lt;float, Tensor&lt;float&gt;, Tensor&lt;float&gt;&gt;()
///     .ConfigureModel(new SegFormer&lt;float&gt;(architecture, "segformer_b2.onnx"));
///
/// // Semantic segmentation
/// var classMap = builder.GetSemanticClassMap(imageTensor);
/// var probMap = builder.GetSemanticProbabilities(imageTensor);
///
/// // Or with a promptable model like SAM:
/// builder.ConfigureModel(new SAM2&lt;float&gt;(architecture, "sam2_large.onnx"));
/// builder.SetSegmentationImage(imageTensor);
/// var result = builder.SegmentFromPoints(points, labels);
/// </code>
/// </para>
/// </remarks>
public static class SegmentationBuilderExtensions
{
    #region Semantic Segmentation

    /// <summary>
    /// Produces raw per-pixel class logits from a semantic segmentation model.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="builder">The builder with a configured segmentation model.</param>
    /// <param name="image">Input image tensor [C, H, W] or [B, C, H, W].</param>
    /// <returns>Logits tensor [numClasses, H, W] or [B, numClasses, H, W].</returns>
    public static Tensor<T> SegmentSemantic<T>(
        this AiModelBuilder<T, Tensor<T>, Tensor<T>> builder,
        Tensor<T> image)
    {
        var model = GetSegmentationModel(builder);
        return model.Segment(image);
    }

    /// <summary>
    /// Gets the per-pixel class map (argmax of logits) from a semantic segmentation model.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="builder">The builder with a configured semantic segmentation model.</param>
    /// <param name="image">Input image tensor [C, H, W] or [B, C, H, W].</param>
    /// <returns>Class label map [H, W] or [B, H, W] where each pixel is a class index.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the simplest way to segment an image. Each pixel gets
    /// assigned to the most likely class. For example, with a Cityscapes model, you'd get
    /// pixels labeled as road (0), sidewalk (1), building (2), etc.
    /// </para>
    /// </remarks>
    public static Tensor<T> GetSemanticClassMap<T>(
        this AiModelBuilder<T, Tensor<T>, Tensor<T>> builder,
        Tensor<T> image)
    {
        var model = GetSemanticModel(builder);
        return model.GetClassMap(image);
    }

    /// <summary>
    /// Gets per-pixel class probability maps from a semantic segmentation model.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="builder">The builder with a configured semantic segmentation model.</param>
    /// <param name="image">Input image tensor [C, H, W] or [B, C, H, W].</param>
    /// <returns>Probability map [numClasses, H, W] or [B, numClasses, H, W] with values in [0, 1].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Instead of just the predicted class, this gives confidence scores
    /// for every class at every pixel. Useful for understanding model uncertainty — if the model
    /// gives 90% to "road" and 10% to "sidewalk", it's fairly confident. If it gives 40% to
    /// "car" and 35% to "truck", it's uncertain.
    /// </para>
    /// </remarks>
    public static Tensor<T> GetSemanticProbabilities<T>(
        this AiModelBuilder<T, Tensor<T>, Tensor<T>> builder,
        Tensor<T> image)
    {
        var model = GetSemanticModel(builder);
        return model.GetProbabilityMap(image);
    }

    #endregion

    #region Instance Segmentation

    /// <summary>
    /// Detects and segments individual object instances in an image.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="builder">The builder with a configured instance segmentation model.</param>
    /// <param name="image">Input image tensor [C, H, W] or [B, C, H, W].</param>
    /// <returns>Instance segmentation result with per-instance masks, boxes, classes, and scores.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Finds each individual object (car #1, car #2, person #1) and gives
    /// you its exact pixel outline. Each detection includes a bounding box, mask, class label,
    /// and confidence score.
    /// </para>
    /// </remarks>
    public static InstanceSegmentationResult<T> SegmentInstances<T>(
        this AiModelBuilder<T, Tensor<T>, Tensor<T>> builder,
        Tensor<T> image)
    {
        var model = GetInstanceModel(builder);
        return model.DetectInstances(image);
    }

    #endregion

    #region Panoptic Segmentation

    /// <summary>
    /// Performs panoptic segmentation (unified semantic + instance) on an image.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="builder">The builder with a configured panoptic segmentation model.</param>
    /// <param name="image">Input image tensor [C, H, W] or [B, C, H, W].</param>
    /// <returns>Panoptic result with per-pixel class labels, instance IDs, and segment metadata.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Panoptic segmentation gives you the most complete understanding of a
    /// scene. Every pixel gets a class label, and countable objects (cars, people) also get unique
    /// instance IDs. Amorphous regions (sky, road) just get class labels without instance IDs.
    /// </para>
    /// </remarks>
    public static PanopticSegmentationResult<T> SegmentPanoptic<T>(
        this AiModelBuilder<T, Tensor<T>, Tensor<T>> builder,
        Tensor<T> image)
    {
        var model = GetPanopticModel(builder);
        return model.SegmentPanoptic(image);
    }

    #endregion

    #region Promptable Segmentation

    /// <summary>
    /// Encodes an image for subsequent prompt-based segmentation.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="builder">The builder with a configured promptable segmentation model.</param>
    /// <param name="image">Input image tensor [C, H, W].</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Call this once per image. The model encodes the image internally.
    /// Then you can call SegmentFromPoints, SegmentFromBox, etc. multiple times without
    /// re-encoding. This makes interactive segmentation very fast.
    /// </para>
    /// </remarks>
    public static void SetSegmentationImage<T>(
        this AiModelBuilder<T, Tensor<T>, Tensor<T>> builder,
        Tensor<T> image)
    {
        var model = GetPromptableModel(builder);
        model.SetImage(image);
    }

    /// <summary>
    /// Segments the region indicated by point prompts (click to segment).
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="builder">The builder with a configured promptable segmentation model.</param>
    /// <param name="points">Point coordinates [N, 2] as (x, y) pairs.</param>
    /// <param name="labels">Point labels [N] where 1 = foreground, 0 = background.</param>
    /// <returns>Mask proposals with confidence scores and stability scores.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Click on the object you want (label=1) and optionally click on
    /// background areas to exclude (label=0). The model returns mask proposals ranked by
    /// confidence.
    /// </para>
    /// </remarks>
    public static PromptedSegmentationResult<T> SegmentFromPoints<T>(
        this AiModelBuilder<T, Tensor<T>, Tensor<T>> builder,
        Tensor<T> points,
        Tensor<T> labels)
    {
        var model = GetPromptableModel(builder);
        return model.SegmentFromPoints(points, labels);
    }

    /// <summary>
    /// Segments the object inside a bounding box.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="builder">The builder with a configured promptable segmentation model.</param>
    /// <param name="box">Bounding box [4] as (x1, y1, x2, y2).</param>
    /// <returns>Mask proposals with confidence scores.</returns>
    public static PromptedSegmentationResult<T> SegmentFromBox<T>(
        this AiModelBuilder<T, Tensor<T>, Tensor<T>> builder,
        Tensor<T> box)
    {
        var model = GetPromptableModel(builder);
        return model.SegmentFromBox(box);
    }

    /// <summary>
    /// Refines a rough mask into a precise segmentation.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="builder">The builder with a configured promptable segmentation model.</param>
    /// <param name="mask">Rough mask [H, W] where positive values indicate region of interest.</param>
    /// <returns>Refined mask proposals with confidence scores.</returns>
    public static PromptedSegmentationResult<T> SegmentFromMask<T>(
        this AiModelBuilder<T, Tensor<T>, Tensor<T>> builder,
        Tensor<T> mask)
    {
        var model = GetPromptableModel(builder);
        return model.SegmentFromMask(mask);
    }

    /// <summary>
    /// Automatically segments everything in the image without prompts.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="builder">The builder with a configured promptable segmentation model.</param>
    /// <returns>All detected segments in the image.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The model automatically finds and segments every distinct object
    /// or region. No clicking or drawing required — just pass an image and get all segments.
    /// </para>
    /// </remarks>
    public static List<PromptedSegmentationResult<T>> SegmentEverything<T>(
        this AiModelBuilder<T, Tensor<T>, Tensor<T>> builder)
    {
        var model = GetPromptableModel(builder);
        return model.SegmentEverything();
    }

    #endregion

    #region Video Segmentation

    /// <summary>
    /// Initializes video object tracking with first-frame masks.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="builder">The builder with a configured video segmentation model.</param>
    /// <param name="firstFrame">First video frame [C, H, W].</param>
    /// <param name="masks">Object masks on the first frame [numObjects, H, W].</param>
    /// <param name="objectIds">Optional unique IDs for each tracked object.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Tell the model which objects to track by showing the first frame
    /// with masks drawn on each object. Then call PropagateSegmentation for each subsequent frame.
    /// </para>
    /// </remarks>
    public static void InitializeVideoTracking<T>(
        this AiModelBuilder<T, Tensor<T>, Tensor<T>> builder,
        Tensor<T> firstFrame,
        Tensor<T> masks,
        int[]? objectIds = null)
    {
        var model = GetVideoModel(builder);
        model.InitializeTracking(firstFrame, masks, objectIds);
    }

    /// <summary>
    /// Propagates tracked object masks to the next video frame.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="builder">The builder with a configured video segmentation model.</param>
    /// <param name="frame">Next video frame [C, H, W].</param>
    /// <returns>Tracked masks, object IDs, and confidence for this frame.</returns>
    public static VideoSegmentationResult<T> PropagateSegmentation<T>(
        this AiModelBuilder<T, Tensor<T>, Tensor<T>> builder,
        Tensor<T> frame)
    {
        var model = GetVideoModel(builder);
        return model.PropagateToFrame(frame);
    }

    /// <summary>
    /// Corrects a tracked object's mask at the current frame.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="builder">The builder with a configured video segmentation model.</param>
    /// <param name="objectId">ID of the object to correct.</param>
    /// <param name="correctionMask">Corrected mask [H, W].</param>
    public static void CorrectSegmentationMask<T>(
        this AiModelBuilder<T, Tensor<T>, Tensor<T>> builder,
        int objectId,
        Tensor<T> correctionMask)
    {
        var model = GetVideoModel(builder);
        model.AddCorrection(objectId, correctionMask);
    }

    /// <summary>
    /// Resets the video tracking state and memory.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="builder">The builder with a configured video segmentation model.</param>
    public static void ResetSegmentationTracking<T>(
        this AiModelBuilder<T, Tensor<T>, Tensor<T>> builder)
    {
        var model = GetVideoModel(builder);
        model.ResetTracking();
    }

    #endregion

    #region Medical Segmentation

    /// <summary>
    /// Segments a 2D medical image slice (CT, MRI, X-ray, etc.).
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="builder">The builder with a configured medical segmentation model.</param>
    /// <param name="slice">2D medical image [C, H, W].</param>
    /// <returns>Medical segmentation result with organ/structure labels and metadata.</returns>
    public static MedicalSegmentationResult<T> SegmentMedicalSlice<T>(
        this AiModelBuilder<T, Tensor<T>, Tensor<T>> builder,
        Tensor<T> slice)
    {
        var model = GetMedicalModel(builder);
        return model.SegmentSlice(slice);
    }

    /// <summary>
    /// Segments a 3D medical volume (full CT or MRI scan).
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="builder">The builder with a configured medical segmentation model.</param>
    /// <param name="volume">3D volume [C, D, H, W] where D is depth (number of slices).</param>
    /// <returns>Volumetric segmentation result with per-voxel labels.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> CT and MRI scans are 3D volumes (stacks of 2D slices). This method
    /// processes the entire volume at once, using 3D context for better accuracy than processing
    /// slice-by-slice.
    /// </para>
    /// </remarks>
    public static MedicalSegmentationResult<T> SegmentMedicalVolume<T>(
        this AiModelBuilder<T, Tensor<T>, Tensor<T>> builder,
        Tensor<T> volume)
    {
        var model = GetMedicalModel(builder);
        return model.SegmentVolume(volume);
    }

    /// <summary>
    /// Segments using few-shot examples (for models like UniverSeg, MedSAM).
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="builder">The builder with a configured few-shot medical segmentation model.</param>
    /// <param name="queryImage">Image to segment [C, H, W].</param>
    /// <param name="supportImages">Example images [N, C, H, W].</param>
    /// <param name="supportMasks">Example masks [N, 1, H, W].</param>
    /// <returns>Segmentation result based on the provided examples.</returns>
    public static MedicalSegmentationResult<T> SegmentMedicalFewShot<T>(
        this AiModelBuilder<T, Tensor<T>, Tensor<T>> builder,
        Tensor<T> queryImage,
        Tensor<T> supportImages,
        Tensor<T> supportMasks)
    {
        var model = GetMedicalModel(builder);
        return model.SegmentFewShot(queryImage, supportImages, supportMasks);
    }

    #endregion

    #region Open-Vocabulary Segmentation

    /// <summary>
    /// Segments objects described by text class names (open-vocabulary).
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="builder">The builder with a configured open-vocabulary segmentation model.</param>
    /// <param name="image">Input image tensor [C, H, W].</param>
    /// <param name="classNames">Text descriptions of classes to segment.</param>
    /// <returns>Per-class masks and confidence scores.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Describe what you want to find in plain text. For example:
    /// <code>
    /// var result = builder.SegmentWithTextClasses(image, ["car", "bicycle", "person on skateboard"]);
    /// </code>
    /// The model returns masks showing where each described thing is in the image.
    /// </para>
    /// </remarks>
    public static OpenVocabSegmentationResult<T> SegmentWithTextClasses<T>(
        this AiModelBuilder<T, Tensor<T>, Tensor<T>> builder,
        Tensor<T> image,
        IReadOnlyList<string> classNames)
    {
        var model = GetOpenVocabModel(builder);
        return model.SegmentWithText(image, classNames);
    }

    /// <summary>
    /// Segments objects matching a single text prompt (grounded segmentation).
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="builder">The builder with a configured open-vocabulary segmentation model.</param>
    /// <param name="image">Input image tensor [C, H, W].</param>
    /// <param name="prompt">Natural language description (e.g., "the largest dog in the image").</param>
    /// <returns>Segmentation result for the described object(s).</returns>
    public static OpenVocabSegmentationResult<T> SegmentWithTextPrompt<T>(
        this AiModelBuilder<T, Tensor<T>, Tensor<T>> builder,
        Tensor<T> image,
        string prompt)
    {
        var model = GetOpenVocabModel(builder);
        return model.SegmentWithPrompt(image, prompt);
    }

    #endregion

    #region Referring Segmentation

    /// <summary>
    /// Segments objects described by a natural language expression with reasoning.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="builder">The builder with a configured referring segmentation model.</param>
    /// <param name="image">Input image tensor [C, H, W].</param>
    /// <param name="expression">Referring expression (e.g., "the person standing behind the counter").</param>
    /// <returns>Mask(s), text response, and confidence for the referred object(s).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Referring segmentation understands complex descriptions that
    /// require reasoning about spatial relationships, attributes, and context. The model
    /// returns both a segmentation mask and a text explanation of what it found.
    /// </para>
    /// </remarks>
    public static ReferringSegmentationResult<T> SegmentFromExpression<T>(
        this AiModelBuilder<T, Tensor<T>, Tensor<T>> builder,
        Tensor<T> image,
        string expression)
    {
        var model = GetReferringModel(builder);
        return model.SegmentFromExpression(image, expression);
    }

    /// <summary>
    /// Segments objects from a multi-turn conversational context.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="builder">The builder with a configured referring segmentation model.</param>
    /// <param name="image">Input image tensor [C, H, W].</param>
    /// <param name="conversationHistory">Previous conversation turns as (role, message) pairs.</param>
    /// <param name="currentQuery">The current user query.</param>
    /// <returns>Segmentation result with mask(s) and conversational response.</returns>
    public static ReferringSegmentationResult<T> SegmentFromConversation<T>(
        this AiModelBuilder<T, Tensor<T>, Tensor<T>> builder,
        Tensor<T> image,
        IReadOnlyList<(string Role, string Message)> conversationHistory,
        string currentQuery)
    {
        var model = GetReferringModel(builder);
        return model.SegmentFromConversation(image, conversationHistory, currentQuery);
    }

    /// <summary>
    /// Segments and tracks objects in a video from a natural language description.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="builder">The builder with a configured referring segmentation model.</param>
    /// <param name="frames">Video frames [numFrames, C, H, W].</param>
    /// <param name="expression">Description of what to segment and track.</param>
    /// <returns>Per-frame segmentation results with tracking information.</returns>
    public static List<ReferringSegmentationResult<T>> SegmentVideoFromExpression<T>(
        this AiModelBuilder<T, Tensor<T>, Tensor<T>> builder,
        Tensor<T> frames,
        string expression)
    {
        var model = GetReferringModel(builder);
        return model.SegmentVideoFromExpression(frames, expression);
    }

    #endregion

    #region Model Queries

    /// <summary>
    /// Gets the number of segmentation classes the configured model predicts.
    /// </summary>
    public static int GetSegmentationClassCount<T>(
        this AiModelBuilder<T, Tensor<T>, Tensor<T>> builder)
    {
        var model = GetSegmentationModel(builder);
        return model.NumClasses;
    }

    /// <summary>
    /// Gets the expected input dimensions of the configured segmentation model.
    /// </summary>
    /// <returns>Tuple of (height, width).</returns>
    public static (int Height, int Width) GetSegmentationInputSize<T>(
        this AiModelBuilder<T, Tensor<T>, Tensor<T>> builder)
    {
        var model = GetSegmentationModel(builder);
        return (model.InputHeight, model.InputWidth);
    }

    /// <summary>
    /// Gets whether the configured segmentation model is running in ONNX mode.
    /// </summary>
    public static bool IsSegmentationOnnxMode<T>(
        this AiModelBuilder<T, Tensor<T>, Tensor<T>> builder)
    {
        var model = GetSegmentationModel(builder);
        return model.IsOnnxMode;
    }

    #endregion

    #region Private Helpers

    private static ISegmentationModel<T> GetSegmentationModel<T>(
        AiModelBuilder<T, Tensor<T>, Tensor<T>> builder)
    {
        if (builder.ConfiguredModel is ISegmentationModel<T> model)
        {
            return model;
        }

        if (builder.ConfiguredModel is null)
        {
            throw new InvalidOperationException(
                "No model configured. Use ConfigureModel() with a segmentation model before calling segmentation methods.");
        }

        throw new InvalidOperationException(
            $"The configured model ({builder.ConfiguredModel.GetType().Name}) does not support segmentation. " +
            $"Use ConfigureModel() with a model that implements ISegmentationModel<{typeof(T).Name}>.");
    }

    private static ISemanticSegmentation<T> GetSemanticModel<T>(
        AiModelBuilder<T, Tensor<T>, Tensor<T>> builder)
    {
        if (builder.ConfiguredModel is ISemanticSegmentation<T> model)
        {
            return model;
        }

        if (builder.ConfiguredModel is null)
        {
            throw new InvalidOperationException(
                "No model configured. Use ConfigureModel() with a semantic segmentation model " +
                "(e.g., SegFormer, SegNeXt, InternImage) before calling semantic segmentation methods.");
        }

        throw new InvalidOperationException(
            $"The configured model ({builder.ConfiguredModel.GetType().Name}) does not support semantic segmentation. " +
            $"Use ConfigureModel() with a model that implements ISemanticSegmentation<{typeof(T).Name}>.");
    }

    private static IInstanceSegmentation<T> GetInstanceModel<T>(
        AiModelBuilder<T, Tensor<T>, Tensor<T>> builder)
    {
        if (builder.ConfiguredModel is IInstanceSegmentation<T> model)
        {
            return model;
        }

        if (builder.ConfiguredModel is null)
        {
            throw new InvalidOperationException(
                "No model configured. Use ConfigureModel() with an instance segmentation model " +
                "(e.g., YOLO11-Seg, Mask2Former, MaskDINO) before calling instance segmentation methods.");
        }

        throw new InvalidOperationException(
            $"The configured model ({builder.ConfiguredModel.GetType().Name}) does not support instance segmentation. " +
            $"Use ConfigureModel() with a model that implements IInstanceSegmentation<{typeof(T).Name}>.");
    }

    private static IPanopticSegmentation<T> GetPanopticModel<T>(
        AiModelBuilder<T, Tensor<T>, Tensor<T>> builder)
    {
        if (builder.ConfiguredModel is IPanopticSegmentation<T> model)
        {
            return model;
        }

        if (builder.ConfiguredModel is null)
        {
            throw new InvalidOperationException(
                "No model configured. Use ConfigureModel() with a panoptic segmentation model " +
                "(e.g., Mask2Former, kMaX-DeepLab, OneFormer) before calling panoptic segmentation methods.");
        }

        throw new InvalidOperationException(
            $"The configured model ({builder.ConfiguredModel.GetType().Name}) does not support panoptic segmentation. " +
            $"Use ConfigureModel() with a model that implements IPanopticSegmentation<{typeof(T).Name}>.");
    }

    private static IPromptableSegmentation<T> GetPromptableModel<T>(
        AiModelBuilder<T, Tensor<T>, Tensor<T>> builder)
    {
        if (builder.ConfiguredModel is IPromptableSegmentation<T> model)
        {
            return model;
        }

        if (builder.ConfiguredModel is null)
        {
            throw new InvalidOperationException(
                "No model configured. Use ConfigureModel() with a promptable segmentation model " +
                "(e.g., SAM 2, SAM-HQ, SegGPT, SEEM) before calling promptable segmentation methods.");
        }

        throw new InvalidOperationException(
            $"The configured model ({builder.ConfiguredModel.GetType().Name}) does not support promptable segmentation. " +
            $"Use ConfigureModel() with a model that implements IPromptableSegmentation<{typeof(T).Name}>.");
    }

    private static IVideoSegmentation<T> GetVideoModel<T>(
        AiModelBuilder<T, Tensor<T>, Tensor<T>> builder)
    {
        if (builder.ConfiguredModel is IVideoSegmentation<T> model)
        {
            return model;
        }

        if (builder.ConfiguredModel is null)
        {
            throw new InvalidOperationException(
                "No model configured. Use ConfigureModel() with a video segmentation model " +
                "(e.g., SAM 2, Cutie, XMem, DEVA) before calling video segmentation methods.");
        }

        throw new InvalidOperationException(
            $"The configured model ({builder.ConfiguredModel.GetType().Name}) does not support video segmentation. " +
            $"Use ConfigureModel() with a model that implements IVideoSegmentation<{typeof(T).Name}>.");
    }

    private static IMedicalSegmentation<T> GetMedicalModel<T>(
        AiModelBuilder<T, Tensor<T>, Tensor<T>> builder)
    {
        if (builder.ConfiguredModel is IMedicalSegmentation<T> model)
        {
            return model;
        }

        if (builder.ConfiguredModel is null)
        {
            throw new InvalidOperationException(
                "No model configured. Use ConfigureModel() with a medical segmentation model " +
                "(e.g., nnU-Net, TransUNet, Swin-UNETR, MedSAM) before calling medical segmentation methods.");
        }

        throw new InvalidOperationException(
            $"The configured model ({builder.ConfiguredModel.GetType().Name}) does not support medical segmentation. " +
            $"Use ConfigureModel() with a model that implements IMedicalSegmentation<{typeof(T).Name}>.");
    }

    private static IOpenVocabSegmentation<T> GetOpenVocabModel<T>(
        AiModelBuilder<T, Tensor<T>, Tensor<T>> builder)
    {
        if (builder.ConfiguredModel is IOpenVocabSegmentation<T> model)
        {
            return model;
        }

        if (builder.ConfiguredModel is null)
        {
            throw new InvalidOperationException(
                "No model configured. Use ConfigureModel() with an open-vocabulary segmentation model " +
                "(e.g., SAN, CAT-Seg, Grounded SAM 2) before calling open-vocabulary segmentation methods.");
        }

        throw new InvalidOperationException(
            $"The configured model ({builder.ConfiguredModel.GetType().Name}) does not support open-vocabulary segmentation. " +
            $"Use ConfigureModel() with a model that implements IOpenVocabSegmentation<{typeof(T).Name}>.");
    }

    private static IReferringSegmentation<T> GetReferringModel<T>(
        AiModelBuilder<T, Tensor<T>, Tensor<T>> builder)
    {
        if (builder.ConfiguredModel is IReferringSegmentation<T> model)
        {
            return model;
        }

        if (builder.ConfiguredModel is null)
        {
            throw new InvalidOperationException(
                "No model configured. Use ConfigureModel() with a referring segmentation model " +
                "(e.g., LISA, VideoLISA, GLaMM, OMG-LLaVA) before calling referring segmentation methods.");
        }

        throw new InvalidOperationException(
            $"The configured model ({builder.ConfiguredModel.GetType().Name}) does not support referring segmentation. " +
            $"Use ConfigureModel() with a model that implements IReferringSegmentation<{typeof(T).Name}>.");
    }

    #endregion
}
