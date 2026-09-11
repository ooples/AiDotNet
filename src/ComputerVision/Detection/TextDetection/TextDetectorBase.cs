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
public abstract partial class TextDetectorBase<T> : ModelBase<T, Tensor<T>, Tensor<T>>
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
    }

    /// <summary>
    /// Detects text regions in an image.
    /// </summary>
    /// <param name="image">Input image tensor [batch, channels, height, width].</param>
    /// <returns>Text detection result.</returns>
    public abstract TextDetectionResult<T> Detect(Tensor<T> image);

    /// <summary>
    /// Detects text regions with custom threshold.
    /// </summary>
    public abstract TextDetectionResult<T> Detect(Tensor<T> image, double confidenceThreshold);

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
        // Keep the original asymmetric pixel mapping, but execute through the selected engine so
        // prediction/training share inference's pixel domain without forcing GPU data onto the CPU
        // or cutting the gradient path to an upstream image-producing model.
        var resized = CvTensorOps<T>.ResizeBilinearAsymmetric(
            image, Options.InputSize[0], Options.InputSize[1]);
        return Engine.TensorMultiplyScalar(resized, NumOps.FromDouble(1.0 / 255.0));
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

    /// <inheritdoc />
    /// <summary>
    /// Gets the step size used by <see cref="Train"/>.
    /// </summary>
    /// <remarks>
    /// Detection losses are large early in training, so this is deliberately conservative.
    /// Override it to match a paper recipe.
    /// </remarks>
    protected virtual double TrainingLearningRate => 0.001;

    /// <summary>
    /// Runs one training step against the model's public prediction.
    /// </summary>
    /// <param name="input">The training image.</param>
    /// <param name="expectedOutput">The desired output, shaped like <see cref="Predict"/>.</param>
    /// <remarks>
    /// Previously an empty method, so text detectors ignored training entirely. See
    /// <c>ObjectDetectorBase.Train</c> for the mechanism and its limits.
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (input is null)
        {
            throw new ArgumentNullException(nameof(input));
        }

        if (expectedOutput is null)
        {
            throw new ArgumentNullException(nameof(expectedOutput));
        }

        bool wasTraining = IsTrainingMode;
        SetTrainingMode(true);
        try
        {
            RecordTrainingLoss(TensorModelTrainer<T>.Step(
                this, input, expectedOutput, NumOps.FromDouble(TrainingLearningRate), Predict));
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    /// <inheritdoc />
    public override ILossFunction<T> DefaultLossFunction => new MeanSquaredErrorLoss<T>();

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> WithParameters(Vector<T> parameters)
    {
        var copy = DeepCopy();
        ((IParameterizable<T, Tensor<T>, Tensor<T>>)copy).SetParameters(parameters);
        return copy;
    }

    /// <inheritdoc />
    // See the note on ObjectDetectorBase: MemberwiseClone gave a shallow copy that shared
    // weights with the original. ModelBase's rebuild-and-reload DeepCopy is correct here.

    #endregion

    /// <summary>
    /// The shape of the first input this model's forward pass ran on. Its lazily-shaped layers sized
    /// their weights from it, so replaying it on a rebuilt copy reproduces the same parameter
    /// topology. Scratch: never persisted, and rebuilt copies record their own.
    /// </summary>
    [AiDotNet.Attributes.Scratch]
    private int[]? _resolvedInputShape;

    /// <summary>Records the input shape on the first forward pass.</summary>
    private void NoteResolvedInput(Tensor<T> input)
    {
        if (_resolvedInputShape is not null || input is null)
        {
            return;
        }

        var shape = new int[input.Shape.Length];
        for (int i = 0; i < shape.Length; i++)
        {
            shape[i] = input.Shape[i];
        }

        _resolvedInputShape = shape;
    }

    /// <inheritdoc />
    /// <remarks>
    /// Runs the copy once on a zero input of the shape this model has already processed, so its
    /// lazily-shaped layers (the convolutions behind the Conv2D adapter, the backbone's lazy layers)
    /// size their weights exactly as this model's did before its state is loaded into them.
    /// </remarks>
    protected override void PrepareCopyForStateRestore(ModelBase<T, Tensor<T>, Tensor<T>> copy)
    {
        if (_resolvedInputShape is not null && copy is TextDetectorBase<T> rebuilt)
        {
            var shape = (int[])_resolvedInputShape.Clone();
            shape[0] = 1;
            rebuilt.Predict(new Tensor<T>(shape));
        }
    }

    /// <summary>
    /// Gets the number of channels in the images this model reads.
    /// </summary>
    /// <remarks>RGB unless a model overrides it; every backbone here is built for three channels.</remarks>
    protected virtual int InputChannels => 3;

    /// <summary>
    /// Gives a model that has never run a concrete parameter topology, so its state can be captured.
    /// </summary>
    /// <remarks>
    /// Several layers size their weights on their first forward pass. Until then the model reports
    /// its parameters as shape-deferred, which is correct for a parameter query but made
    /// <see cref="Serialize"/> - and therefore <c>Clone</c> - throw on a freshly constructed model.
    /// Running the network once on a zero image of the configured input size resolves exactly the
    /// shapes the first real image would, because every image is resized to that size first.
    /// </remarks>
    private void ResolveDeferredParameters()
    {
        if (_resolvedInputShape is not null)
        {
            return;
        }

        Predict(new Tensor<T>(new[] { 1, InputChannels, Options.InputSize[0], Options.InputSize[1] }));
    }

    /// <inheritdoc />
    /// <remarks>Resolves shape-deferred layers first; see <see cref="ResolveDeferredParameters"/>.</remarks>
    public override byte[] Serialize()
    {
        ResolveDeferredParameters();
        return base.Serialize();
    }

    /// <summary>
    /// The loss of the most recent <see cref="Train"/> call, measured before its update.
    /// </summary>
    [AiDotNet.Attributes.Scratch]
    private T _lastTrainingLoss = MathHelper.GetNumericOperations<T>().Zero;

    /// <summary>
    /// Gets the loss of the most recent <see cref="Train"/> call, measured on that call's input before
    /// its update (zero before the first call).
    /// </summary>
    /// <returns>The training objective's value: mean squared error, or the model's own loss where it
    /// has one.</returns>
    /// <remarks>Same contract as <c>INeuralNetwork&lt;T&gt;.GetLastLoss</c>.</remarks>
    public T GetLastLoss() => _lastTrainingLoss;

    /// <summary>Records the loss a training step reported.</summary>
    /// <param name="loss">The step's loss.</param>
    protected void RecordTrainingLoss(T loss) => _lastTrainingLoss = loss;

    /// <summary>
    /// Whether the model is in training mode.
    /// </summary>
    protected bool IsTrainingMode;

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
    public virtual void SetTrainingMode(bool training)
    {
        IsTrainingMode = training;
        Backbone?.SetTrainingMode(training);
    }
}
