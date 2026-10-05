using AiDotNet.Augmentation.Image;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Tensors;

namespace AiDotNet.ComputerVision.Detection.TextDetection;

/// <summary>
/// Result of text detection containing detected text regions.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class TextDetectionResult<T>
{
    /// <summary>
    /// List of detected text regions.
    /// </summary>
    public List<TextRegion<T>> TextRegions { get; set; } = new();

    /// <summary>
    /// Time taken for inference.
    /// </summary>
    public TimeSpan InferenceTime { get; set; }

    /// <summary>
    /// Width of the input image.
    /// </summary>
    public int ImageWidth { get; set; }

    /// <summary>
    /// Height of the input image.
    /// </summary>
    public int ImageHeight { get; set; }
}

/// <summary>
/// Represents a detected text region in an image.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class TextRegion<T>
{
    private static readonly INumericOperations<T> NumOps =
        Tensors.Helpers.MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Bounding box around the text region.
    /// </summary>
    public BoundingBox<T> Box { get; set; }

    /// <summary>
    /// Polygon points defining the exact boundary (for rotated or curved text).
    /// </summary>
    public List<(T X, T Y)> Polygon { get; set; } = new();

    /// <summary>
    /// Confidence score of the detection.
    /// </summary>
    public T Confidence { get; set; }

    /// <summary>
    /// Rotation angle of the text region in degrees (if applicable).
    /// </summary>
    public double RotationAngle { get; set; }

    /// <summary>
    /// Whether this region is likely a word vs a text line.
    /// </summary>
    public TextRegionType RegionType { get; set; } = TextRegionType.Word;

    /// <summary>
    /// Creates a new text region.
    /// </summary>
    public TextRegion(BoundingBox<T> box, T confidence)
    {
        Box = box;
        Confidence = confidence;
    }

    /// <summary>
    /// Creates a text region from polygon points.
    /// </summary>
    public static TextRegion<T> FromPolygon(List<(T X, T Y)> polygon, T confidence)
    {
        if (polygon.Count < 4)
            throw new ArgumentException("Polygon must have at least 4 points", nameof(polygon));

        // Compute bounding box from polygon
        double minX = double.MaxValue, minY = double.MaxValue;
        double maxX = double.MinValue, maxY = double.MinValue;

        foreach (var (x, y) in polygon)
        {
            double px = NumOps.ToDouble(x);
            double py = NumOps.ToDouble(y);
            minX = Math.Min(minX, px);
            minY = Math.Min(minY, py);
            maxX = Math.Max(maxX, px);
            maxY = Math.Max(maxY, py);
        }

        var box = new BoundingBox<T>(
            NumOps.FromDouble(minX),
            NumOps.FromDouble(minY),
            NumOps.FromDouble(maxX),
            NumOps.FromDouble(maxY),
            BoundingBoxFormat.XYXY);

        return new TextRegion<T>(box, confidence)
        {
            Polygon = polygon
        };
    }
}

/// <summary>
/// Type of text region.
/// </summary>
public enum TextRegionType
{
    /// <summary>Character-level detection.</summary>
    Character,
    /// <summary>Word-level detection.</summary>
    Word,
    /// <summary>Line-level detection.</summary>
    Line,
    /// <summary>Paragraph-level detection.</summary>
    Paragraph
}

/// <summary>
/// Options for text detection models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class TextDetectionOptions<T>
{
    private static readonly INumericOperations<T> NumOps =
        Tensors.Helpers.MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Text detection architecture to use.
    /// </summary>
    public TextDetectionArchitecture Architecture { get; set; } = TextDetectionArchitecture.DBNet;

    /// <summary>
    /// Backbone network type.
    /// </summary>
    public BackboneType Backbone { get; set; } = BackboneType.ResNet50;

    /// <summary>
    /// Model size variant.
    /// </summary>
    public ModelSize Size { get; set; } = ModelSize.Medium;

    /// <summary>
    /// Input image size (height, width).
    /// </summary>
    public int[] InputSize { get; set; } = new[] { 640, 640 };

    /// <summary>
    /// Minimum confidence threshold for text detection.
    /// </summary>
    public T ConfidenceThreshold { get; set; } = NumOps.FromDouble(0.5);

    /// <summary>
    /// Threshold for text/background binarization.
    /// </summary>
    public T BinaryThreshold { get; set; } = NumOps.FromDouble(0.3);

    /// <summary>
    /// Polygon simplification threshold.
    /// </summary>
    public double PolygonSimplificationEpsilon { get; set; } = 0.002;

    /// <summary>
    /// Maximum number of detections.
    /// </summary>
    public int MaxDetections { get; set; } = 1000;

    /// <summary>
    /// Whether to detect at multiple scales.
    /// </summary>
    public bool UseMultiScale { get; set; } = false;

    /// <summary>
    /// Whether to use pretrained weights.
    /// </summary>
    public bool UsePretrained { get; set; } = true;

    /// <summary>
    /// URL for pretrained weights.
    /// </summary>
    public string? WeightsUrl { get; set; }

    /// <summary>
    /// Seed for weight initialization, so two models built from equal options start from equal weights.
    /// Default: 42, matching <see cref="AiDotNet.Models.Options.ObjectDetectionOptions{T}.RandomSeed"/>.
    /// Null leaves initialization unseeded.
    /// </summary>
    public int? RandomSeed { get; set; } = 42;
}

/// <summary>
/// Text detection architecture types.
/// </summary>
public enum TextDetectionArchitecture
{
    /// <summary>CRAFT: Character Region Awareness for Text detection.</summary>
    CRAFT,
    /// <summary>EAST: Efficient and Accurate Scene Text detector.</summary>
    EAST,
    /// <summary>DBNet: Differentiable Binarization Network.</summary>
    DBNet
}

/// <summary>
/// Base class for text detection models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract partial class TextDetectorBase<T> : VisionTaskModelBase<T>, AiDotNet.Interfaces.ITextDetectionTrainingModel<T>
{
    // NumOps inherited from ModelBase
    protected readonly TextDetectionOptions<T> Options;
    protected IDetectionBackbone<T>? Backbone;

    /// <summary>
    /// Gets the backbone network, throwing if not initialized.
    /// </summary>
    /// <exception cref="InvalidOperationException">Thrown when backbone has not been initialized.</exception>
    // An accessor over the Backbone field, not separate storage. Without the alias the generator
    // registered BOTH, so these weights were counted twice in the flat parameter vector, and for a
    // detector without a neck (DETR) reading parameters threw from the accessor's null check.
    [AiDotNet.Attributes.ParameterAlias(nameof(Backbone))]
    protected IDetectionBackbone<T> EnsureBackbone =>
        Backbone ?? throw new InvalidOperationException(
            $"{GetType().Name}: Backbone not initialized. Ensure the model is properly constructed.");

    /// <summary>
    /// Name of this text detector.
    /// </summary>
    public abstract string Name { get; }

    /// <summary>
    /// Gets the maximum number of text regions kept for a single image.
    /// </summary>
    public int MaxDetections => Options.MaxDetections;

    /// <summary>
    /// Gets the default minimum confidence a text region needs to be reported.
    /// </summary>
    public double ConfidenceThreshold => NumOps.ToDouble(Options.ConfidenceThreshold);

    /// <summary>Creates a new text detector.</summary>
    protected TextDetectorBase(TextDetectionOptions<T> options)
    {
        Options = options;
        // Arm the per-layer initialization seed scope before the derived constructor builds any layer
        // (the same root-model contract DiffusionModelBase follows). Every layer, the BackboneLayerShims
        // adapters' inner layers, and the necks and query tables that draw from the scope then take a
        // deterministic seed, so two models built from equal options start from equal weights (#2201).
        // A null seed leaves the scope unarmed and initialization stays unseeded.
        AiDotNet.NeuralNetworks.Layers.LayerInitializationSeedScope.ResetForModelConstruction(options.RandomSeed);
        AiDotNet.NeuralNetworks.Layers.LayerInitializationSeedScope.OfferSeedToNestedBackbone();
    }

    /// <summary>
    /// Detects text regions in one image.
    /// </summary>
    /// <param name="image">Input image tensor [1, channels, height, width].</param>
    /// <returns>Text detection result.</returns>
    public virtual TextDetectionResult<T> Detect(Tensor<T> image)
        => Detect(image, NumOps.ToDouble(Options.ConfidenceThreshold));

    /// <summary>
    /// Detects text regions in one image with a custom confidence threshold.
    /// </summary>
    /// <exception cref="ArgumentException">The tensor holds more than one image; use <see cref="DetectBatch(Tensor{T}, double)"/>.</exception>
    public virtual TextDetectionResult<T> Detect(Tensor<T> image, double confidenceThreshold)
    {
        if (image is null) throw new ArgumentNullException(nameof(image));
        // A single result cannot describe several images. Every detector used to decode batch position 0
        // of a batched forward and return it as the whole answer, silently dropping the rest.
        if (image.Rank == 4 && image.Shape[0] != 1)
            throw new ArgumentException(
                $"Detect takes one image but got a batch of {image.Shape[0]}; use DetectBatch for several images.",
                nameof(image));

        return DetectBatch(image, confidenceThreshold)[0];
    }

    /// <summary>
    /// Detects text regions in every image of a batch, with one forward pass.
    /// </summary>
    /// <param name="images">Input image tensor [batch, channels, height, width].</param>
    /// <returns>One result per image, in batch order.</returns>
    public virtual IReadOnlyList<TextDetectionResult<T>> DetectBatch(Tensor<T> images)
        => DetectBatch(images, NumOps.ToDouble(Options.ConfidenceThreshold));

    /// <summary>
    /// Detects text regions in every image of a batch with a custom confidence threshold.
    /// </summary>
    public virtual IReadOnlyList<TextDetectionResult<T>> DetectBatch(Tensor<T> images, double confidenceThreshold)
    {
        if (images is null) throw new ArgumentNullException(nameof(images));
        if (images.Rank != 4)
            throw new ArgumentException($"Expected [batch, channels, height, width], got rank {images.Rank}.", nameof(images));

        var startTime = DateTime.UtcNow;
        int batchSize = images.Shape[0];
        int originalHeight = images.Shape[2];
        int originalWidth = images.Shape[3];

        var batchOutputs = Forward(Preprocess(images));
        var results = new List<TextDetectionResult<T>>(batchSize);
        for (int i = 0; i < batchSize; i++)
        {
            var itemOutputs = DetectionOutputBatching<T>.SliceItem(batchOutputs, i);
            results.Add(new TextDetectionResult<T>
            {
                TextRegions = PostProcess(itemOutputs, originalWidth, originalHeight, confidenceThreshold),
                ImageWidth = originalWidth,
                ImageHeight = originalHeight
            });
        }

        // One forward serves the whole batch, so each image reports the batch's time.
        var elapsed = DateTime.UtcNow - startTime;
        foreach (var result in results) result.InferenceTime = elapsed;
        return results;
    }

    /// <summary>
    /// Preprocesses the input image.
    /// </summary>
    protected virtual Tensor<T> Preprocess(Tensor<T> image)
    {
        var prepared = PreprocessCore(image);
        NoteResolvedInput(prepared);
        return prepared;
    }

    private Tensor<T> PreprocessCore(Tensor<T> image)
    {
        var (height, width) = GetValidatedInputSize();
        // Keep the original asymmetric pixel mapping, but execute through the selected engine so
        // prediction/training share inference's pixel domain without forcing GPU data onto the CPU
        // or cutting the gradient path to an upstream image-producing model.
        var resized = CvTensorOps<T>.ResizeBilinearAsymmetric(
            image, height, width);
        return Engine.TensorMultiplyScalar(resized, NumOps.FromDouble(1.0 / 255.0));
    }

    private (int Height, int Width) GetValidatedInputSize()
    {
        // InputSize is publicly mutable: validate at each consuming boundary and return the
        // validated values rather than reading the caller-owned array again after validation.
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
    /// Forward pass through the network.
    /// </summary>
    protected abstract List<Tensor<T>> Forward(Tensor<T> input);

    /// <summary>
    /// Post-processes network outputs to get text regions.
    /// </summary>
    protected abstract List<TextRegion<T>> PostProcess(
        List<Tensor<T>> outputs,
        int imageWidth,
        int imageHeight,
        double confidenceThreshold);

    /// <summary>
    /// Gets the total parameter count of the model.
    /// </summary>
    public virtual long GetParameterCount()
    {
        long count = Backbone?.GetParameterCount() ?? 0;
        count += GetHeadParameterCount();
        return count;
    }

    /// <summary>
    /// Gets the parameter count of the detection head.
    /// </summary>
    protected abstract long GetHeadParameterCount();

    /// <summary>
    /// Loads pretrained weights.
    /// </summary>
    public abstract Task LoadWeightsAsync(string pathOrUrl, CancellationToken cancellationToken = default);

    /// <summary>
    /// Saves model weights.
    /// </summary>
    public abstract void SaveWeights(string path);

    /// <summary>
    /// Converts polygon points to a minimum area rotated rectangle.
    /// </summary>
    protected (double cx, double cy, double width, double height, double angle)
        FitMinAreaRect(List<(double X, double Y)> points)
    {
        if (points.Count < 4)
            return (0, 0, 0, 0, 0);

        // Simple axis-aligned bounding box (could be extended to rotated rect)
        double minX = points.Min(p => p.X);
        double maxX = points.Max(p => p.X);
        double minY = points.Min(p => p.Y);
        double maxY = points.Max(p => p.Y);

        double cx = (minX + maxX) / 2;
        double cy = (minY + maxY) / 2;
        double width = maxX - minX;
        double height = maxY - minY;

        return (cx, cy, width, height, 0);
    }

    /// <summary>
    /// Simplifies a polygon using Douglas-Peucker algorithm.
    /// </summary>
    protected List<(double X, double Y)> SimplifyPolygon(
        List<(double X, double Y)> points,
        double epsilon)
    {
        if (points.Count <= 4)
            return points;

        // Find point with max distance from line between first and last
        double maxDist = 0;
        int maxIdx = 0;

        var start = points[0];
        var end = points[^1];

        for (int i = 1; i < points.Count - 1; i++)
        {
            double dist = PerpendicularDistance(points[i], start, end);
            if (dist > maxDist)
            {
                maxDist = dist;
                maxIdx = i;
            }
        }

        if (maxDist > epsilon)
        {
            var left = SimplifyPolygon(points.Take(maxIdx + 1).ToList(), epsilon);
            var right = SimplifyPolygon(points.Skip(maxIdx).ToList(), epsilon);

            return left.Take(left.Count - 1).Concat(right).ToList();
        }

        return new List<(double X, double Y)> { start, end };
    }

    private double PerpendicularDistance(
        (double X, double Y) point,
        (double X, double Y) lineStart,
        (double X, double Y) lineEnd)
    {
        double dx = lineEnd.X - lineStart.X;
        double dy = lineEnd.Y - lineStart.Y;
        double length = Math.Sqrt(dx * dx + dy * dy);

        if (length < 1e-10)
            return Math.Sqrt(
                (point.X - lineStart.X) * (point.X - lineStart.X) +
                (point.Y - lineStart.Y) * (point.Y - lineStart.Y));

        return Math.Abs(
            dy * point.X - dx * point.Y + lineEnd.X * lineStart.Y - lineEnd.Y * lineStart.X
        ) / length;
    }

    #region ModelBase Overrides

    /// <summary>
    /// Predicts by running the forward pass and returning the primary output map.
    /// </summary>
    /// <remarks>
    /// This used to return <c>Preprocess(input)</c> -- the resized, normalised INPUT IMAGE -- so
    /// the model reported its own input back as a prediction. Nothing downstream could tell,
    /// because the returned tensor has a plausible shape. It now runs the network, matching
    /// <c>ObjectDetectorBase.Predict</c>: every output map is flattened per image and concatenated,
    /// so a model with a probability map and a threshold map (DBNet) or a score and a geometry map
    /// (EAST) exposes both, and training against the prediction reaches both heads.
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return CvTensorOps<T>.ConcatenateOutputs(Forward(Preprocess(input)));
    }

    #endregion

    /// <summary>
    /// One training step on page images and their text polygons. The model's paper target assignment and
    /// loss come from <see cref="TextDetectionLoss"/>.
    /// </summary>
    /// <param name="images">The page images, <c>[batch, channels, height, width]</c> or one <c>[channels, height, width]</c>.</param>
    /// <param name="targets">The text polygons of each image, in that image's pixel coordinates.</param>
    public void TrainTextDetections(Tensor<T> images, TextDetectionTrainingBatch targets)
    {
        if (images is null) throw new ArgumentNullException(nameof(images));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (images.Rank is not (3 or 4))
            throw new ArgumentException($"Images must be [C, H, W] or [B, C, H, W]; got rank {images.Rank}.", nameof(images));
        int batch = images.Rank == 4 ? images.Shape[0] : 1;
        if (targets.ImageCount != batch)
            throw new ArgumentException($"{targets.ImageCount} target lists for a batch of {batch} images.", nameof(targets));
        int height = images.Shape[images.Rank - 2], width = images.Shape[images.Rank - 1];
        TrainWithTargets<List<Tensor<T>>, TextDetectionTrainingBatch>(images, targets,
            input => Forward(Preprocess(input)),
            (outputs, batchTargets) => TextDetectionLoss(outputs, batchTargets, width, height));
    }

    /// <summary>
    /// The model's paper loss over its training heads <paramref name="outputs"/>. Targets are built from the
    /// polygons, which are given in <paramref name="imageWidth"/> x <paramref name="imageHeight"/> source pixels.
    /// The result must be a scalar on the gradient tape.
    /// </summary>
    protected abstract Tensor<T> TextDetectionLoss(List<Tensor<T>> outputs, TextDetectionTrainingBatch targets,
        int imageWidth, int imageHeight);

    /// <summary>A polygon in source pixels, mapped onto a <paramref name="mapWidth"/> x <paramref name="mapHeight"/> output map.</summary>
    /// <remarks>Every image is resized to the network input as a whole, so each output map spans the whole source image.</remarks>
    protected static (double X, double Y)[] ToMap(TextPolygonTarget target, int imageWidth, int imageHeight, int mapWidth, int mapHeight)
        => TextTargetGeometry.Scale(target.Points, (double)mapWidth / imageWidth, (double)mapHeight / imageHeight);

    /// <summary>Per-image target maps <c>[h, w]</c> stacked into a constant <c>[batch, channels, h, w]</c> tensor.</summary>
    protected Tensor<T> MapTensor(IReadOnlyList<double[][,]> perImageChannels)
    {
        int batch = perImageChannels.Count, channels = perImageChannels[0].Length;
        int height = perImageChannels[0][0].GetLength(0), width = perImageChannels[0][0].GetLength(1);
        var tensor = new Tensor<T>(new[] { batch, channels, height, width });
        for (int b = 0; b < batch; b++)
            for (int ch = 0; ch < channels; ch++)
                for (int y = 0; y < height; y++)
                    for (int x = 0; x < width; x++)
                        tensor[b, ch, y, x] = NumOps.FromDouble(perImageChannels[b][ch][y, x]);
        return tensor;
    }

    /// <summary>Elementwise binary cross-entropy of probabilities <paramref name="p"/> against <paramref name="y"/>, clamped away from 0 and 1.</summary>
    protected Tensor<T> BinaryCrossEntropyMap(Tensor<T> p, Tensor<T> y)
    {
        var clamped = Engine.TensorClamp(p, NumOps.FromDouble(1e-6), NumOps.FromDouble(1 - 1e-6));
        var ones = Engine.TensorAddScalar(Engine.TensorMultiplyScalar(y, NumOps.Zero), NumOps.One);
        var positive = Engine.TensorMultiply(y, Engine.TensorLog(clamped));
        var negative = Engine.TensorMultiply(Engine.TensorSubtract(ones, y), Engine.TensorLog(Engine.TensorSubtract(ones, clamped)));
        return Engine.TensorMultiplyScalar(Engine.TensorAdd(positive, negative), NumOps.FromDouble(-1.0));
    }

    /// <summary>
    /// <c>sum(values * weights) / sum(weights)</c>. The weights are constants, which can be a mask or a
    /// per-pixel weighting. Zero when the weights sum to zero.
    /// </summary>
    protected Tensor<T> WeightedMean(Tensor<T> values, Tensor<T> weights)
    {
        double total = 0;
        for (int i = 0; i < weights.Length; i++) total += NumOps.ToDouble(weights[i]);
        var sum = Engine.ReduceSum(Engine.TensorMultiply(values, weights), null);
        return Engine.TensorMultiplyScalar(sum, NumOps.FromDouble(total > 0 ? 1.0 / total : 0.0));
    }

    /// <summary>Dice loss <c>1 - 2 sum(p y m) / (sum(p m) + sum(y m) + eps)</c> over the masked pixels.</summary>
    protected Tensor<T> DiceLoss(Tensor<T> p, Tensor<T> y, Tensor<T> mask)
    {
        var intersection = Engine.ReduceSum(Engine.TensorMultiply(Engine.TensorMultiply(p, y), mask), null);
        var union = Engine.TensorAddScalar(Engine.TensorAdd(
            Engine.ReduceSum(Engine.TensorMultiply(p, mask), null),
            Engine.ReduceSum(Engine.TensorMultiply(y, mask), null)), NumOps.FromDouble(1e-6));
        return Engine.TensorSubtract(
            Engine.TensorAddScalar(Engine.TensorMultiplyScalar(intersection, NumOps.Zero), NumOps.One),
            Engine.TensorMultiplyScalar(Engine.TensorDivide(intersection, union), NumOps.FromDouble(2.0)));
    }
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
    /// Sets the model to training or inference mode.
    /// </summary>
    /// <param name="training">True for training mode, false for inference.</param>
    /// <remarks>
    /// Batch normalisation depends on it: batch statistics (and running-statistic updates) while
    /// training, running statistics at inference. The text detectors had no such switch, so their
    /// backbone's batch-norm layers never left inference mode, even inside <see cref="Train"/> -
    /// unlike the object detectors, which have always switched theirs. Override to forward the mode to
    /// head modules that depend on it, calling the base.
    /// </remarks>
    public override void SetTrainingMode(bool training)
    {
        base.SetTrainingMode(training);
        Backbone?.SetTrainingMode(training);
    }
}
