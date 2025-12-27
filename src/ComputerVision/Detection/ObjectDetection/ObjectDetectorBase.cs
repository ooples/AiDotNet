using AiDotNet.ComputerVision.Detection.Backbones;
using AiDotNet.ComputerVision.Detection.Necks;
using AiDotNet.ComputerVision.Detection.PostProcessing;
using AiDotNet.ComputerVision.Weights;
using AiDotNet.Models.Options;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.ComputerVision.Detection.ObjectDetection;

/// <summary>
/// Base class for all object detection models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> An object detector takes an image and finds all objects in it,
/// returning their locations (bounding boxes), types (class labels), and confidence scores.
/// This base class provides the common structure and methods that all detection models share.</para>
///
/// <para>A typical detector has three parts:
/// - Backbone: Extracts features from the image
/// - Neck: Combines features at multiple scales
/// - Head: Produces final predictions (boxes, classes, scores)
/// </para>
/// </remarks>
public abstract class ObjectDetectorBase<T>
{
    /// <summary>
    /// Numeric operations for type T.
    /// </summary>
    protected readonly INumericOperations<T> NumOps;

    /// <summary>
    /// Configuration options for this detector.
    /// </summary>
    protected readonly ObjectDetectionOptions<T> Options;

    /// <summary>
    /// The backbone network for feature extraction.
    /// </summary>
    protected BackboneBase<T>? Backbone { get; set; }

    /// <summary>
    /// The neck module for feature fusion.
    /// </summary>
    protected NeckBase<T>? Neck { get; set; }

    /// <summary>
    /// NMS algorithm for removing duplicate detections.
    /// </summary>
    protected readonly NMS<T> Nms;

    /// <summary>
    /// Whether the model is in training mode.
    /// </summary>
    protected bool IsTrainingMode;

    /// <summary>
    /// Weight downloader for fetching pre-trained weights.
    /// </summary>
    protected readonly WeightDownloader WeightDownloader;

    /// <summary>
    /// Class names for detection labels.
    /// </summary>
    public string[] ClassNames { get; protected set; }

    /// <summary>
    /// Name of this detector architecture.
    /// </summary>
    public abstract string Name { get; }

    /// <summary>
    /// Creates a new object detector with the specified options.
    /// </summary>
    /// <param name="options">Configuration options for the detector.</param>
    protected ObjectDetectorBase(ObjectDetectionOptions<T> options)
    {
        NumOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        Options = options;
        Nms = new NMS<T>();
        WeightDownloader = new WeightDownloader();
        IsTrainingMode = false;

        // Initialize class names (default to COCO classes)
        ClassNames = options.ClassNames ?? GetCocoClassNames();
    }

    /// <summary>
    /// Detects objects in an image.
    /// </summary>
    /// <param name="image">Input image tensor with shape [batch, channels, height, width].</param>
    /// <returns>Detection results for each image in the batch.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the main method you call to detect objects.
    /// Pass in an image (as a tensor) and get back a list of detected objects with
    /// their bounding boxes, class labels, and confidence scores.</para>
    /// </remarks>
    public virtual DetectionResult<T> Detect(Tensor<T> image)
    {
        return Detect(image, Options.ConfidenceThreshold, Options.NmsThreshold);
    }

    /// <summary>
    /// Detects objects in an image with custom thresholds.
    /// </summary>
    /// <param name="image">Input image tensor.</param>
    /// <param name="confidenceThreshold">Minimum confidence to keep a detection.</param>
    /// <param name="nmsThreshold">IoU threshold for NMS.</param>
    /// <returns>Detection results.</returns>
    public abstract DetectionResult<T> Detect(
        Tensor<T> image,
        double confidenceThreshold,
        double nmsThreshold);

    /// <summary>
    /// Detects objects in a batch of images.
    /// </summary>
    /// <param name="images">Batch of images with shape [batch, channels, height, width].</param>
    /// <returns>Detection results for each image.</returns>
    public virtual BatchDetectionResult<T> DetectBatch(Tensor<T> images)
    {
        return DetectBatch(images, Options.ConfidenceThreshold, Options.NmsThreshold);
    }

    /// <summary>
    /// Detects objects in a batch of images with custom thresholds.
    /// </summary>
    /// <param name="images">Batch of images.</param>
    /// <param name="confidenceThreshold">Minimum confidence.</param>
    /// <param name="nmsThreshold">NMS threshold.</param>
    /// <returns>Batch detection results.</returns>
    public virtual BatchDetectionResult<T> DetectBatch(
        Tensor<T> images,
        double confidenceThreshold,
        double nmsThreshold)
    {
        var startTime = DateTime.UtcNow;
        var results = new List<DetectionResult<T>>();

        int batchSize = images.Shape[0];
        for (int i = 0; i < batchSize; i++)
        {
            // Extract single image from batch
            var singleImage = ExtractBatchItem(images, i);
            var result = Detect(singleImage, confidenceThreshold, nmsThreshold);
            results.Add(result);
        }

        return new BatchDetectionResult<T>
        {
            Results = results,
            TotalInferenceTime = DateTime.UtcNow - startTime
        };
    }

    /// <summary>
    /// Performs forward pass through the network.
    /// </summary>
    /// <param name="input">Input image tensor.</param>
    /// <returns>Raw network outputs before post-processing.</returns>
    protected abstract List<Tensor<T>> Forward(Tensor<T> input);

    /// <summary>
    /// Post-processes raw network outputs into detections.
    /// </summary>
    /// <param name="outputs">Raw network outputs.</param>
    /// <param name="imageWidth">Original image width.</param>
    /// <param name="imageHeight">Original image height.</param>
    /// <param name="confidenceThreshold">Minimum confidence threshold.</param>
    /// <param name="nmsThreshold">NMS IoU threshold.</param>
    /// <returns>List of detections after NMS.</returns>
    protected abstract List<Detection<T>> PostProcess(
        List<Tensor<T>> outputs,
        int imageWidth,
        int imageHeight,
        double confidenceThreshold,
        double nmsThreshold);

    /// <summary>
    /// Sets the model to training or inference mode.
    /// </summary>
    /// <param name="training">True for training mode, false for inference.</param>
    public virtual void SetTrainingMode(bool training)
    {
        IsTrainingMode = training;
        Backbone?.SetTrainingMode(training);
        Neck?.SetTrainingMode(training);
    }

    /// <summary>
    /// Loads pre-trained weights from a file or URL.
    /// </summary>
    /// <param name="pathOrUrl">Local file path or URL to weights.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    public abstract Task LoadWeightsAsync(
        string pathOrUrl,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Loads default pre-trained weights for this architecture and size.
    /// </summary>
    /// <param name="cancellationToken">Cancellation token.</param>
    public virtual async Task LoadPretrainedWeightsAsync(CancellationToken cancellationToken = default)
    {
        // Get the appropriate weights URL from the registry
        string? url = Options.WeightsUrl;
        if (string.IsNullOrEmpty(url))
        {
            var modelKey = PretrainedRegistry.GetDetectionModelKey(Options.Architecture, Options.Size);
            url = PretrainedRegistry.GetUrl(modelKey);
        }

        if (url is null || url.Length == 0)
        {
            throw new InvalidOperationException(
                $"No pre-trained weights URL found for {Options.Architecture} {Options.Size}. " +
                "Please provide a WeightsUrl in the options.");
        }

        var fileName = $"{Options.Architecture}_{Options.Size}.weights";
        var localPath = await WeightDownloader.DownloadIfNeededAsync(url, fileName, cancellationToken: cancellationToken);
        await LoadWeightsAsync(localPath, cancellationToken);
    }

    /// <summary>
    /// Saves model weights to a file.
    /// </summary>
    /// <param name="path">File path to save weights.</param>
    public abstract void SaveWeights(string path);

    /// <summary>
    /// Gets the total number of parameters in the model.
    /// </summary>
    /// <returns>Number of trainable parameters.</returns>
    public virtual long GetParameterCount()
    {
        long count = 0;
        if (Backbone is not null) count += Backbone.GetParameterCount();
        if (Neck is not null) count += Neck.GetParameterCount();
        count += GetHeadParameterCount();
        return count;
    }

    /// <summary>
    /// Gets the number of parameters in the detection head.
    /// </summary>
    /// <returns>Number of parameters.</returns>
    protected abstract long GetHeadParameterCount();

    /// <summary>
    /// Preprocesses an image for input to the network.
    /// </summary>
    /// <param name="image">Raw image tensor.</param>
    /// <returns>Preprocessed tensor ready for the network.</returns>
    protected virtual Tensor<T> Preprocess(Tensor<T> image)
    {
        // Default preprocessing: resize to input size and normalize
        int targetHeight = Options.InputSize[0];
        int targetWidth = Options.InputSize[1];

        // Resize if needed
        var resized = ResizeImage(image, targetHeight, targetWidth);

        // Normalize to [0, 1] range (assuming input is [0, 255])
        var normalized = Normalize(resized);

        return normalized;
    }

    /// <summary>
    /// Resizes an image tensor to the specified dimensions.
    /// </summary>
    /// <param name="image">Input image.</param>
    /// <param name="targetHeight">Target height.</param>
    /// <param name="targetWidth">Target width.</param>
    /// <returns>Resized image.</returns>
    protected virtual Tensor<T> ResizeImage(Tensor<T> image, int targetHeight, int targetWidth)
    {
        int batch = image.Shape[0];
        int channels = image.Shape[1];
        int srcHeight = image.Shape[2];
        int srcWidth = image.Shape[3];

        if (srcHeight == targetHeight && srcWidth == targetWidth)
        {
            return image;
        }

        var resized = new Tensor<T>(new[] { batch, channels, targetHeight, targetWidth });

        double scaleY = (double)srcHeight / targetHeight;
        double scaleX = (double)srcWidth / targetWidth;

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int h = 0; h < targetHeight; h++)
                {
                    for (int w = 0; w < targetWidth; w++)
                    {
                        // Bilinear interpolation
                        double srcY = h * scaleY;
                        double srcX = w * scaleX;

                        int y0 = (int)Math.Floor(srcY);
                        int x0 = (int)Math.Floor(srcX);
                        int y1 = Math.Min(y0 + 1, srcHeight - 1);
                        int x1 = Math.Min(x0 + 1, srcWidth - 1);

                        double dy = srcY - y0;
                        double dx = srcX - x0;

                        double v00 = NumOps.ToDouble(image[b, c, y0, x0]);
                        double v01 = NumOps.ToDouble(image[b, c, y0, x1]);
                        double v10 = NumOps.ToDouble(image[b, c, y1, x0]);
                        double v11 = NumOps.ToDouble(image[b, c, y1, x1]);

                        double value = v00 * (1 - dx) * (1 - dy)
                                     + v01 * dx * (1 - dy)
                                     + v10 * (1 - dx) * dy
                                     + v11 * dx * dy;

                        resized[b, c, h, w] = NumOps.FromDouble(value);
                    }
                }
            }
        }

        return resized;
    }

    /// <summary>
    /// Normalizes image values to [0, 1] range.
    /// </summary>
    /// <param name="image">Input image with values [0, 255].</param>
    /// <returns>Normalized image.</returns>
    protected virtual Tensor<T> Normalize(Tensor<T> image)
    {
        var normalized = new Tensor<T>(image.Shape);
        T scale = NumOps.FromDouble(1.0 / 255.0);

        for (int i = 0; i < image.Length; i++)
        {
            normalized[i] = NumOps.Multiply(image[i], scale);
        }

        return normalized;
    }

    /// <summary>
    /// Extracts a single image from a batch.
    /// </summary>
    /// <param name="batch">Batch of images.</param>
    /// <param name="index">Index of the image to extract.</param>
    /// <returns>Single image tensor.</returns>
    protected Tensor<T> ExtractBatchItem(Tensor<T> batch, int index)
    {
        int channels = batch.Shape[1];
        int height = batch.Shape[2];
        int width = batch.Shape[3];

        var single = new Tensor<T>(new[] { 1, channels, height, width });

        for (int c = 0; c < channels; c++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    single[0, c, h, w] = batch[index, c, h, w];
                }
            }
        }

        return single;
    }

    /// <summary>
    /// Gets the default COCO class names.
    /// </summary>
    /// <returns>Array of 80 COCO class names.</returns>
    protected static string[] GetCocoClassNames()
    {
        return new[]
        {
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
            "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
            "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
            "toothbrush"
        };
    }
}
