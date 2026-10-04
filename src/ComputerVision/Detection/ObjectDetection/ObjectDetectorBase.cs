using AiDotNet.ComputerVision.Detection.Necks;
using AiDotNet.Interfaces;
using AiDotNet.ComputerVision.Detection.PostProcessing;
using AiDotNet.ComputerVision.Weights;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
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
[AiDotNet.Configuration.YamlConfigurable("ObjectDetector")]
public abstract partial class ObjectDetectorBase<T> : VisionTaskModelBase<T>
{
    // Engine and NumOps inherited from ModelBase

    /// <summary>
    /// Configuration options for this detector.
    /// </summary>
    protected readonly ObjectDetectionOptions<T> Options;

    /// <summary>
    /// The backbone network for feature extraction. Typed against the
    /// <see cref="IDetectionBackbone{T}"/> contract (NeuralNetworkBase + multi-scale
    /// feature provider) instead of any concrete base class so any backbone that
    /// satisfies the contract — ResNet, CSPDarknet, EfficientNet, SwinTransformer,
    /// or a future custom implementation — can plug in.
    /// </summary>
    protected IDetectionBackbone<T>? Backbone { get; set; }

    /// <summary>
    /// The neck module for feature fusion. Optional: a detector such as DETR feeds the
    /// backbone straight into its transformer and has no neck at all.
    /// </summary>
    [AiDotNet.Attributes.TrainableParameter(Optional = true)]
    protected NeckBase<T>? Neck { get; set; }

    /// <summary>
    /// Gets the backbone network, throwing if not initialized.
    /// </summary>
    /// <exception cref="InvalidOperationException">Thrown when backbone has not been initialized.</exception>
    // An accessor over the Backbone property, not separate storage. Without the alias the generator
    // registered BOTH, so these weights were counted twice in the flat parameter vector, and for a
    // detector without a neck (DETR) reading parameters threw from the accessor's null check.
    [AiDotNet.Attributes.ParameterAlias(nameof(Backbone))]
    protected IDetectionBackbone<T> EnsureBackbone =>
        Backbone ?? throw new InvalidOperationException(
            $"{GetType().Name}: Backbone not initialized. Ensure the model is properly constructed.");

    /// <summary>
    /// Gets the neck module, throwing if not initialized.
    /// </summary>
    /// <exception cref="InvalidOperationException">Thrown when neck has not been initialized.</exception>
    // An accessor over the Neck property, not separate storage. Without the alias the generator
    // registered BOTH, so these weights were counted twice in the flat parameter vector, and for a
    // detector without a neck (DETR) reading parameters threw from the accessor's null check.
    [AiDotNet.Attributes.ParameterAlias(nameof(Neck))]
    protected NeckBase<T> EnsureNeck =>
        Neck ?? throw new InvalidOperationException(
            $"{GetType().Name}: Neck not initialized. Ensure the model is properly constructed.");

    /// <summary>
    /// NMS algorithm for removing duplicate detections.
    /// </summary>
    protected readonly NMS<T> Nms;

    /// <summary>
    /// Weight downloader for fetching pre-trained weights.
    /// </summary>
    protected readonly WeightDownloader WeightDownloader;

    /// <summary>
    /// Class names for detection labels.
    /// </summary>
    public string[] ClassNames { get; protected set; }

    /// <summary>
    /// Gets the number of object classes this detector was configured for.
    /// </summary>
    /// <remarks>
    /// Every <see cref="Detection{T}.ClassId"/> the detector emits indexes into a label set of
    /// this size, so callers need it to interpret the output.
    /// </remarks>
    public int NumClasses => Options.NumClasses;

    /// <summary>
    /// Gets the maximum number of detections kept for a single image after non-maximum suppression.
    /// </summary>
    public int MaxDetections => Options.MaxDetections;

    /// <summary>
    /// Gets the default minimum confidence a detection needs to be reported.
    /// </summary>
    public double ConfidenceThreshold => Options.ConfidenceThreshold;

    /// <summary>
    /// Gets the default IoU threshold used by non-maximum suppression.
    /// </summary>
    public double NmsThreshold => Options.NmsThreshold;

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
        Options = options;
        // Arm the per-layer initialization seed scope before the derived constructor builds any layer
        // (the same root-model contract DiffusionModelBase follows). Every layer, the BackboneLayerShims
        // adapters' inner layers, and the necks and query tables that draw from the scope then take a
        // deterministic seed, so two models built from equal options start from equal weights (#2201).
        // A null seed leaves the scope unarmed and initialization stays unseeded.
        AiDotNet.NeuralNetworks.Layers.LayerInitializationSeedScope.ResetForModelConstruction(options.RandomSeed);
        AiDotNet.NeuralNetworks.Layers.LayerInitializationSeedScope.OfferSeedToNestedBackbone();
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
        int imageHeight = images.Shape[2];
        int imageWidth = images.Shape[3];

        // Preprocess exactly as Detect does, then one forward pass for the whole batch. This used to
        // feed the RAW images straight to Forward, so a batch and a single Detect of the same image
        // ran the network on different inputs and disagreed; PostProcess then also mapped
        // coordinates from a frame the network never saw.
        var batchOutputs = Forward(Preprocess(images));

        // Post-process outputs for each image in the batch
        for (int i = 0; i < batchSize; i++)
        {
            // Extract outputs for this batch item
            var itemOutputs = ExtractBatchOutputs(batchOutputs, i);

            // Post-process to get detections for this image
            var detections = PostProcess(itemOutputs, imageWidth, imageHeight, confidenceThreshold, nmsThreshold);

            results.Add(new DetectionResult<T>
            {
                Detections = detections,
                InferenceTime = TimeSpan.Zero, // Individual times not tracked in batch mode
                ImageWidth = imageWidth,
                ImageHeight = imageHeight
            });
        }

        return new BatchDetectionResult<T>
        {
            Results = results,
            TotalInferenceTime = DateTime.UtcNow - startTime
        };
    }

    /// <summary>
    /// Extracts the outputs for a single batch item from batch outputs.
    /// </summary>
    protected virtual List<Tensor<T>> ExtractBatchOutputs(List<Tensor<T>> batchOutputs, int batchIndex)
        => DetectionOutputBatching<T>.SliceItem(batchOutputs, batchIndex);

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
    public override void SetTrainingMode(bool training)
    {
        base.SetTrainingMode(training);
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
        var prepared = PreprocessCore(image);
        NoteResolvedInput(prepared);
        return prepared;
    }

    private Tensor<T> PreprocessCore(Tensor<T> image)
    {
        // Default preprocessing: resize to input size and normalize
        var (targetHeight, targetWidth) = GetValidatedInputSize();

        // Resize if needed
        var resized = ResizeImage(image, targetHeight, targetWidth);

        // Normalize to [0, 1] range (assuming input is [0, 255])
        var normalized = Normalize(resized);

        return normalized;
    }

    private (int Height, int Width) GetValidatedInputSize()
    {
        // InputSize is publicly mutable, so validate at each consuming boundary rather than
        // only at construction. Return the dimensions, not the caller-owned array.
        var inputSize = Options.InputSize;
        if (inputSize is null || inputSize.Length != 2)
        {
            throw new ArgumentException(
                "InputSize must contain exactly two positive dimensions [height, width].",
                nameof(Options.InputSize));
        }

        int height = inputSize[0];
        int width = inputSize[1];
        if (height <= 0 || width <= 0)
        {
            throw new ArgumentException(
                "InputSize must contain exactly two positive dimensions [height, width].",
                nameof(Options.InputSize));
        }

        return (height, width);
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
        var normalized = new Tensor<T>(image._shape);
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

        // Use contiguous memory copy for better performance
        int pixelsPerImage = channels * height * width;
        int sourceOffset = index * pixelsPerImage;
        for (int i = 0; i < pixelsPerImage; i++)
        {
            single[i] = batch[sourceOffset + i];
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

    #region ModelBase Overrides

    /// <summary>
    /// Predicts by running the forward pass and returning raw network outputs concatenated.
    /// </summary>
    /// <remarks>
    /// Each output is flattened per image to <c>[batch, -1]</c> and the results are concatenated, so
    /// the prediction carries every head: all YOLO pyramid levels, both DETR's class logits and its
    /// boxes. It used to return <c>outputs[0]</c> alone despite this summary, so a detector trained
    /// against <see cref="Predict"/> never trained any head but the first. A single-output model is
    /// unchanged.
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        NoteResolvedInput(input);
        return CvTensorOps<T>.ConcatenateOutputs(Forward(input));
    }

    #endregion

    /// <summary>Trains structured heads with typed targets through the shared single-update path.</summary>
    /// <remarks>The derived model validates its task targets before calling this method.</remarks>
    protected void TrainWithTargets<TTarget>(Tensor<T> input, TTarget targets,
        Func<List<Tensor<T>>, TTarget, Tensor<T>> loss) where TTarget : class
        => TrainWithTargets(input, targets, Forward, loss);

    /// <summary>Trains heads produced by a training-specific forward, such as auxiliary heads inference drops.</summary>
    /// <remarks>The derived model validates its task targets before calling this method.</remarks>
    protected void TrainWithTargets<TTarget>(Tensor<T> input, TTarget targets,
        Func<Tensor<T>, List<Tensor<T>>> forward, Func<List<Tensor<T>>, TTarget, Tensor<T>> loss) where TTarget : class
        => TrainWithTargets<List<Tensor<T>>, TTarget>(input, targets, forward, loss);

    /// <inheritdoc />
    protected override int[] DeferredParameterProbeShape
    {
        get
        {
            var (height, width) = GetValidatedInputSize();
            return new[] { 1, InputChannels, height, width };
        }
    }

    /// <summary>
    /// Scale factors from the network-input frame (the <see cref="ObjectDetectionOptions{T}.InputSize"/>
    /// that <see cref="Preprocess"/> resizes to) to the source image's frame.
    /// </summary>
    /// <remarks>
    /// Boxes decode in network-input coordinates. Clipping them to the source image's size without
    /// this mapping produced inverted boxes (x1 beyond x2) whenever the source image was smaller than
    /// the input size, and silently dropped the boxes a two-stage detector's degenerate-box check
    /// then rejected.
    /// </remarks>
    protected (double ScaleX, double ScaleY) InputToImageScale(int imageWidth, int imageHeight)
        => (imageWidth / (double)Options.InputSize[1], imageHeight / (double)Options.InputSize[0]);

    /// <summary>
    /// Gets the IoU threshold non-maximum suppression actually applies for a requested threshold.
    /// </summary>
    /// <param name="requested">The threshold passed to <c>Detect</c>.</param>
    /// <returns>The requested threshold, unless the model deliberately suppresses less aggressively.</returns>
    /// <remarks>
    /// Set-prediction detectors (DETR, RT-DETR) are trained so that each object gets one query, and
    /// apply NMS only as a safety net at a high threshold rather than at the caller's value. That
    /// used to happen silently inside their post-processing; it is now declared here so callers can
    /// see it.
    /// </remarks>
    public virtual double EffectiveNmsThreshold(double requested) => requested;

}
