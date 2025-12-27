using AiDotNet.Augmentation.Image;
using AiDotNet.ComputerVision.Detection.ObjectDetection;
using AiDotNet.ComputerVision.OCR;
using AiDotNet.ComputerVision.OCR.EndToEnd;
using AiDotNet.ComputerVision.Segmentation.InstanceSegmentation;
using AiDotNet.ComputerVision.Tracking;
using AiDotNet.ComputerVision.Visualization;
using AiDotNet.Tensors;

namespace AiDotNet;

/// <summary>
/// Computer vision extensions for PredictionModelBuilder.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input type.</typeparam>
/// <typeparam name="TOutput">The output type.</typeparam>
public partial class PredictionModelBuilder<T, TInput, TOutput>
{
    private ObjectDetectorBase<T>? _objectDetector;
    private InstanceSegmenterBase<T>? _instanceSegmenter;
    private SceneTextReader<T>? _sceneTextReader;
    private ObjectTrackerBase<T>? _objectTracker;
    private DetectionVisualizer<T>? _detectionVisualizer;
    private MaskVisualizer<T>? _maskVisualizer;
    private OCRVisualizer<T>? _ocrVisualizer;

    /// <summary>
    /// Configures an object detector for the builder.
    /// </summary>
    /// <param name="detector">The object detector to use.</param>
    /// <returns>The builder for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this to add object detection capabilities
    /// to your model. The detector will identify and locate objects in images.</para>
    /// </remarks>
    /// <example>
    /// <code>
    /// var builder = new PredictionModelBuilder&lt;float, Tensor&lt;float&gt;, DetectionResult&lt;float&gt;&gt;()
    ///     .ConfigureObjectDetector(new YOLOv8&lt;float&gt;(options))
    ///     .Build();
    /// </code>
    /// </example>
    public PredictionModelBuilder<T, TInput, TOutput> ConfigureObjectDetector(ObjectDetectorBase<T> detector)
    {
        _objectDetector = detector;
        return this;
    }

    /// <summary>
    /// Configures an instance segmenter for the builder.
    /// </summary>
    /// <param name="segmenter">The instance segmenter to use.</param>
    /// <returns>The builder for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Instance segmentation provides pixel-level masks
    /// for each detected object, not just bounding boxes.</para>
    /// </remarks>
    public PredictionModelBuilder<T, TInput, TOutput> ConfigureInstanceSegmenter(InstanceSegmenterBase<T> segmenter)
    {
        _instanceSegmenter = segmenter;
        return this;
    }

    /// <summary>
    /// Configures a scene text reader for OCR.
    /// </summary>
    /// <param name="textReader">The scene text reader to use.</param>
    /// <returns>The builder for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> OCR (Optical Character Recognition) extracts
    /// text from images, useful for reading documents or scene text.</para>
    /// </remarks>
    public PredictionModelBuilder<T, TInput, TOutput> ConfigureSceneTextReader(SceneTextReader<T> textReader)
    {
        _sceneTextReader = textReader;
        return this;
    }

    /// <summary>
    /// Configures an object tracker for video tracking.
    /// </summary>
    /// <param name="tracker">The object tracker to use.</param>
    /// <returns>The builder for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Object tracking maintains identity of objects
    /// across video frames, assigning consistent IDs.</para>
    /// </remarks>
    public PredictionModelBuilder<T, TInput, TOutput> ConfigureObjectTracker(ObjectTrackerBase<T> tracker)
    {
        _objectTracker = tracker;
        return this;
    }

    /// <summary>
    /// Configures visualization options for detection results.
    /// </summary>
    /// <param name="options">Visualization options.</param>
    /// <returns>The builder for method chaining.</returns>
    public PredictionModelBuilder<T, TInput, TOutput> ConfigureVisualization(VisualizationOptions? options = null)
    {
        _detectionVisualizer = new DetectionVisualizer<T>(options);
        _maskVisualizer = new MaskVisualizer<T>(options);
        _ocrVisualizer = new OCRVisualizer<T>(options);
        return this;
    }

    /// <summary>
    /// Gets the configured object detector.
    /// </summary>
    public ObjectDetectorBase<T>? ObjectDetector => _objectDetector;

    /// <summary>
    /// Gets the configured instance segmenter.
    /// </summary>
    public InstanceSegmenterBase<T>? InstanceSegmenter => _instanceSegmenter;

    /// <summary>
    /// Gets the configured scene text reader.
    /// </summary>
    public SceneTextReader<T>? SceneTextReader => _sceneTextReader;

    /// <summary>
    /// Gets the configured object tracker.
    /// </summary>
    public ObjectTrackerBase<T>? ObjectTracker => _objectTracker;

    /// <summary>
    /// Gets the detection visualizer.
    /// </summary>
    public DetectionVisualizer<T>? DetectionVisualizer => _detectionVisualizer;

    /// <summary>
    /// Gets the mask visualizer.
    /// </summary>
    public MaskVisualizer<T>? MaskVisualizer => _maskVisualizer;

    /// <summary>
    /// Gets the OCR visualizer.
    /// </summary>
    public OCRVisualizer<T>? OCRVisualizer => _ocrVisualizer;
}

/// <summary>
/// Extension methods for computer vision operations.
/// </summary>
public static class ComputerVisionBuilderExtensions
{
    /// <summary>
    /// Performs object detection on an image.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="builder">The configured builder.</param>
    /// <param name="image">Input image tensor [batch, channels, height, width].</param>
    /// <returns>Detection result with bounding boxes and labels.</returns>
    public static DetectionResult<T> DetectObjects<T>(
        this PredictionModelBuilder<T, Tensor<T>, DetectionResult<T>> builder,
        Tensor<T> image)
    {
        if (builder.ObjectDetector == null)
        {
            throw new InvalidOperationException("No object detector configured. Use ConfigureObjectDetector first.");
        }

        return builder.ObjectDetector.Detect(image);
    }

    /// <summary>
    /// Performs instance segmentation on an image.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="builder">The configured builder.</param>
    /// <param name="image">Input image tensor.</param>
    /// <returns>Instance segmentation result with masks.</returns>
    public static InstanceSegmentationResult<T> SegmentInstances<T>(
        this PredictionModelBuilder<T, Tensor<T>, InstanceSegmentationResult<T>> builder,
        Tensor<T> image)
    {
        if (builder.InstanceSegmenter == null)
        {
            throw new InvalidOperationException("No instance segmenter configured. Use ConfigureInstanceSegmenter first.");
        }

        return builder.InstanceSegmenter.Segment(image);
    }

    /// <summary>
    /// Performs OCR on an image.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="builder">The configured builder.</param>
    /// <param name="image">Input image tensor.</param>
    /// <returns>OCR result with recognized text.</returns>
    public static OCRResult<T> RecognizeText<T>(
        this PredictionModelBuilder<T, Tensor<T>, OCRResult<T>> builder,
        Tensor<T> image)
    {
        if (builder.SceneTextReader == null)
        {
            throw new InvalidOperationException("No scene text reader configured. Use ConfigureSceneTextReader first.");
        }

        return builder.SceneTextReader.ReadText(image);
    }

    /// <summary>
    /// Updates object tracking with new detections.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="builder">The configured builder.</param>
    /// <param name="detections">Detections from the current frame.</param>
    /// <returns>Tracking result with updated tracks.</returns>
    public static TrackingResult<T> UpdateTracks<T>(
        this PredictionModelBuilder<T, Tensor<T>, TrackingResult<T>> builder,
        List<Detection<T>> detections)
    {
        if (builder.ObjectTracker == null)
        {
            throw new InvalidOperationException("No object tracker configured. Use ConfigureObjectTracker first.");
        }

        return builder.ObjectTracker.Update(detections);
    }

    /// <summary>
    /// Visualizes detection results on an image.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="builder">The configured builder.</param>
    /// <param name="image">Input image.</param>
    /// <param name="result">Detection result to visualize.</param>
    /// <param name="classNames">Optional class name mapping.</param>
    /// <returns>Image with visualizations drawn.</returns>
    public static Tensor<T> VisualizeDetections<T>(
        this PredictionModelBuilder<T, Tensor<T>, DetectionResult<T>> builder,
        Tensor<T> image,
        DetectionResult<T> result,
        string[]? classNames = null)
    {
        var visualizer = builder.DetectionVisualizer ?? new DetectionVisualizer<T>();
        return visualizer.Visualize(image, result, classNames);
    }

    /// <summary>
    /// Visualizes instance segmentation results on an image.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="builder">The configured builder.</param>
    /// <param name="image">Input image.</param>
    /// <param name="result">Segmentation result to visualize.</param>
    /// <param name="classNames">Optional class name mapping.</param>
    /// <returns>Image with visualizations drawn.</returns>
    public static Tensor<T> VisualizeSegmentation<T>(
        this PredictionModelBuilder<T, Tensor<T>, InstanceSegmentationResult<T>> builder,
        Tensor<T> image,
        InstanceSegmentationResult<T> result,
        string[]? classNames = null)
    {
        var visualizer = builder.MaskVisualizer ?? new MaskVisualizer<T>();
        return visualizer.Visualize(image, result, classNames);
    }

    /// <summary>
    /// Visualizes OCR results on an image.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="builder">The configured builder.</param>
    /// <param name="image">Input image.</param>
    /// <param name="result">OCR result to visualize.</param>
    /// <returns>Image with visualizations drawn.</returns>
    public static Tensor<T> VisualizeOCR<T>(
        this PredictionModelBuilder<T, Tensor<T>, OCRResult<T>> builder,
        Tensor<T> image,
        OCRResult<T> result)
    {
        var visualizer = builder.OCRVisualizer ?? new OCRVisualizer<T>();
        return visualizer.Visualize(image, result);
    }

    /// <summary>
    /// Performs detection and tracking in one step.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="builder">The configured builder.</param>
    /// <param name="image">Input image tensor.</param>
    /// <returns>Tracking result with tracked objects.</returns>
    public static TrackingResult<T> DetectAndTrack<T>(
        this PredictionModelBuilder<T, Tensor<T>, TrackingResult<T>> builder,
        Tensor<T> image)
    {
        if (builder.ObjectDetector == null)
        {
            throw new InvalidOperationException("No object detector configured. Use ConfigureObjectDetector first.");
        }

        if (builder.ObjectTracker == null)
        {
            throw new InvalidOperationException("No object tracker configured. Use ConfigureObjectTracker first.");
        }

        var detections = builder.ObjectDetector.Detect(image);
        return builder.ObjectTracker.Update(detections.Detections);
    }
}

/// <summary>
/// Factory methods for creating computer vision pipelines.
/// </summary>
public static class ComputerVisionPipelineFactory
{
    /// <summary>
    /// Creates a detection pipeline with YOLOv8.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="options">Detection options.</param>
    /// <returns>Configured builder for object detection.</returns>
    public static PredictionModelBuilder<T, Tensor<T>, DetectionResult<T>> CreateYOLOv8Pipeline<T>(
        ObjectDetectionOptions<T>? options = null)
    {
        var detectionOptions = options ?? new ObjectDetectionOptions<T>();
        var detector = new AiDotNet.ComputerVision.Detection.ObjectDetection.YOLO.YOLOv8<T>(detectionOptions);

        return new PredictionModelBuilder<T, Tensor<T>, DetectionResult<T>>()
            .ConfigureObjectDetector(detector)
            .ConfigureVisualization();
    }

    /// <summary>
    /// Creates an instance segmentation pipeline with Mask R-CNN.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="options">Segmentation options.</param>
    /// <returns>Configured builder for instance segmentation.</returns>
    public static PredictionModelBuilder<T, Tensor<T>, InstanceSegmentationResult<T>> CreateMaskRCNNPipeline<T>(
        InstanceSegmentationOptions<T>? options = null)
    {
        var segmentationOptions = options ?? new InstanceSegmentationOptions<T>();
        var segmenter = new MaskRCNN<T>(segmentationOptions);

        return new PredictionModelBuilder<T, Tensor<T>, InstanceSegmentationResult<T>>()
            .ConfigureInstanceSegmenter(segmenter)
            .ConfigureVisualization();
    }

    /// <summary>
    /// Creates an OCR pipeline for scene text.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="options">OCR options.</param>
    /// <returns>Configured builder for OCR.</returns>
    public static PredictionModelBuilder<T, Tensor<T>, OCRResult<T>> CreateSceneTextPipeline<T>(
        OCROptions<T>? options = null)
    {
        var ocrOptions = options ?? new OCROptions<T>();
        ocrOptions.Mode = OCRMode.SceneText;
        var reader = new AiDotNet.ComputerVision.OCR.EndToEnd.SceneTextReader<T>(ocrOptions);

        return new PredictionModelBuilder<T, Tensor<T>, OCRResult<T>>()
            .ConfigureSceneTextReader(reader)
            .ConfigureVisualization();
    }

    /// <summary>
    /// Creates a detection and tracking pipeline.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="detectionOptions">Detection options.</param>
    /// <param name="trackingOptions">Tracking options.</param>
    /// <returns>Configured builder for detection and tracking.</returns>
    public static PredictionModelBuilder<T, Tensor<T>, TrackingResult<T>> CreateDetectionTrackingPipeline<T>(
        ObjectDetectionOptions<T>? detectionOptions = null,
        TrackingOptions<T>? trackingOptions = null)
    {
        var detOptions = detectionOptions ?? new ObjectDetectionOptions<T>();
        var trkOptions = trackingOptions ?? new TrackingOptions<T>();

        var detector = new AiDotNet.ComputerVision.Detection.ObjectDetection.YOLO.YOLOv8<T>(detOptions);
        var tracker = new ByteTrack<T>(trkOptions);

        return new PredictionModelBuilder<T, Tensor<T>, TrackingResult<T>>()
            .ConfigureObjectDetector(detector)
            .ConfigureObjectTracker(tracker)
            .ConfigureVisualization();
    }
}
