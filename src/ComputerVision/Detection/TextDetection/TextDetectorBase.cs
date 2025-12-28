using AiDotNet.Augmentation.Image;
using AiDotNet.ComputerVision.Detection.Backbones;
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
public abstract class TextDetectorBase<T>
{
    protected readonly INumericOperations<T> NumOps;
    protected readonly TextDetectionOptions<T> Options;
    protected BackboneBase<T>? Backbone;

    /// <summary>
    /// Name of this text detector.
    /// </summary>
    public abstract string Name { get; }

    /// <summary>
    /// Creates a new text detector.
    /// </summary>
    protected TextDetectorBase(TextDetectionOptions<T> options)
    {
        NumOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
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
        // Standard preprocessing: resize to input size, normalize
        int targetH = Options.InputSize[0];
        int targetW = Options.InputSize[1];

        int batch = image.Shape[0];
        int channels = image.Shape[1];
        int height = image.Shape[2];
        int width = image.Shape[3];

        // Create resized output
        var output = new Tensor<T>(new[] { batch, channels, targetH, targetW });

        // Bilinear interpolation resize
        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int h = 0; h < targetH; h++)
                {
                    for (int w = 0; w < targetW; w++)
                    {
                        double srcY = (double)h / targetH * height;
                        double srcX = (double)w / targetW * width;

                        int y0 = (int)Math.Floor(srcY);
                        int x0 = (int)Math.Floor(srcX);
                        int y1 = Math.Min(y0 + 1, height - 1);
                        int x1 = Math.Min(x0 + 1, width - 1);

                        double wy1 = srcY - y0;
                        double wy0 = 1.0 - wy1;
                        double wx1 = srcX - x0;
                        double wx0 = 1.0 - wx1;

                        double v00 = NumOps.ToDouble(image[b, c, y0, x0]);
                        double v01 = NumOps.ToDouble(image[b, c, y0, x1]);
                        double v10 = NumOps.ToDouble(image[b, c, y1, x0]);
                        double v11 = NumOps.ToDouble(image[b, c, y1, x1]);

                        double val = wy0 * (wx0 * v00 + wx1 * v01) + wy1 * (wx0 * v10 + wx1 * v11);

                        // Normalize to [0, 1]
                        val /= 255.0;

                        output[b, c, h, w] = NumOps.FromDouble(val);
                    }
                }
            }
        }

        return output;
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
}
