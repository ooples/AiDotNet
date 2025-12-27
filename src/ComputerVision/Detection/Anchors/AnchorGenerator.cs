using AiDotNet.Augmentation.Image;
using AiDotNet.Tensors;

namespace AiDotNet.ComputerVision.Detection.Anchors;

/// <summary>
/// Generates anchor boxes for object detection models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Anchor boxes (also called prior boxes) are pre-defined boxes
/// of various sizes and aspect ratios placed at each location in the feature map.
/// The detector predicts how to adjust these anchors to fit actual objects.
/// Using anchors helps the model handle objects of different sizes and shapes.</para>
///
/// <para>For example, a feature map of 80x80 with 3 anchors per location generates
/// 80 x 80 x 3 = 19,200 anchor boxes, each representing a potential object location.</para>
/// </remarks>
public class AnchorGenerator<T>
{
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Base sizes for anchors at each feature level.
    /// </summary>
    public double[] BaseSizes { get; }

    /// <summary>
    /// Aspect ratios for anchors (height/width).
    /// </summary>
    public double[] AspectRatios { get; }

    /// <summary>
    /// Scales to apply to base sizes.
    /// </summary>
    public double[] Scales { get; }

    /// <summary>
    /// Strides (downsampling factors) at each feature level.
    /// </summary>
    public int[] Strides { get; }

    /// <summary>
    /// Creates a new anchor generator with default YOLO-style anchors.
    /// </summary>
    public AnchorGenerator()
        : this(
            baseSizes: new double[] { 32, 64, 128 },
            aspectRatios: new double[] { 0.5, 1.0, 2.0 },
            scales: new double[] { 1.0, 1.26, 1.59 },
            strides: new int[] { 8, 16, 32 })
    {
    }

    /// <summary>
    /// Creates a new anchor generator with custom settings.
    /// </summary>
    /// <param name="baseSizes">Base anchor sizes at each feature level.</param>
    /// <param name="aspectRatios">Anchor aspect ratios (height/width).</param>
    /// <param name="scales">Scales to apply to base sizes.</param>
    /// <param name="strides">Feature map strides.</param>
    public AnchorGenerator(
        double[] baseSizes,
        double[] aspectRatios,
        double[] scales,
        int[] strides)
    {
        _numOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        BaseSizes = baseSizes;
        AspectRatios = aspectRatios;
        Scales = scales;
        Strides = strides;
    }

    /// <summary>
    /// Generates anchors for a single feature map level.
    /// </summary>
    /// <param name="featureHeight">Height of the feature map.</param>
    /// <param name="featureWidth">Width of the feature map.</param>
    /// <param name="stride">Stride (downsampling factor) of this level.</param>
    /// <param name="baseSize">Base anchor size for this level.</param>
    /// <returns>List of anchor boxes centered at each feature map location.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> For each location (x, y) in the feature map, this method
    /// creates multiple anchor boxes of different sizes and aspect ratios. The anchor centers
    /// are converted to image coordinates using the stride.</para>
    /// </remarks>
    public List<BoundingBox<T>> GenerateAnchorsForLevel(
        int featureHeight,
        int featureWidth,
        int stride,
        double baseSize)
    {
        var anchors = new List<BoundingBox<T>>();

        // Generate base anchors (at origin) for all aspect ratios and scales
        var baseAnchors = GenerateBaseAnchors(baseSize);

        // Place anchors at each feature map location
        for (int y = 0; y < featureHeight; y++)
        {
            for (int x = 0; x < featureWidth; x++)
            {
                // Center of this feature map cell in image coordinates
                double centerX = (x + 0.5) * stride;
                double centerY = (y + 0.5) * stride;

                // Shift base anchors to this location
                foreach (var (width, height) in baseAnchors)
                {
                    double x1 = centerX - width / 2;
                    double y1 = centerY - height / 2;
                    double x2 = centerX + width / 2;
                    double y2 = centerY + height / 2;

                    anchors.Add(new BoundingBox<T>(
                        _numOps.FromDouble(x1),
                        _numOps.FromDouble(y1),
                        _numOps.FromDouble(x2),
                        _numOps.FromDouble(y2),
                        BoundingBoxFormat.XYXY));
                }
            }
        }

        return anchors;
    }

    /// <summary>
    /// Generates anchors for all feature levels.
    /// </summary>
    /// <param name="featureSizes">List of (height, width) for each feature level.</param>
    /// <returns>List of anchors for each level.</returns>
    public List<List<BoundingBox<T>>> GenerateAnchors(List<(int Height, int Width)> featureSizes)
    {
        var allAnchors = new List<List<BoundingBox<T>>>();

        for (int level = 0; level < featureSizes.Count; level++)
        {
            var (height, width) = featureSizes[level];
            int stride = level < Strides.Length ? Strides[level] : Strides[^1];
            double baseSize = level < BaseSizes.Length ? BaseSizes[level] : BaseSizes[^1];

            var levelAnchors = GenerateAnchorsForLevel(height, width, stride, baseSize);
            allAnchors.Add(levelAnchors);
        }

        return allAnchors;
    }

    /// <summary>
    /// Generates anchors for an image of specified size.
    /// </summary>
    /// <param name="imageHeight">Image height in pixels.</param>
    /// <param name="imageWidth">Image width in pixels.</param>
    /// <returns>Flattened list of all anchors across all levels.</returns>
    public List<BoundingBox<T>> GenerateAnchorsForImage(int imageHeight, int imageWidth)
    {
        var allAnchors = new List<BoundingBox<T>>();

        for (int level = 0; level < Strides.Length; level++)
        {
            int stride = Strides[level];
            int featureHeight = (imageHeight + stride - 1) / stride;
            int featureWidth = (imageWidth + stride - 1) / stride;
            double baseSize = level < BaseSizes.Length ? BaseSizes[level] : BaseSizes[^1];

            var levelAnchors = GenerateAnchorsForLevel(featureHeight, featureWidth, stride, baseSize);
            allAnchors.AddRange(levelAnchors);
        }

        return allAnchors;
    }

    /// <summary>
    /// Generates base anchors (centered at origin) for all aspect ratios and scales.
    /// </summary>
    /// <param name="baseSize">Base size for anchors.</param>
    /// <returns>List of (width, height) tuples for base anchors.</returns>
    private List<(double Width, double Height)> GenerateBaseAnchors(double baseSize)
    {
        var baseAnchors = new List<(double Width, double Height)>();

        foreach (double scale in Scales)
        {
            double scaledSize = baseSize * scale;

            foreach (double aspectRatio in AspectRatios)
            {
                // aspectRatio = height / width
                // area = width * height = scaledSize^2
                // width = sqrt(area / aspectRatio)
                // height = aspectRatio * width
                double width = scaledSize / Math.Sqrt(aspectRatio);
                double height = scaledSize * Math.Sqrt(aspectRatio);

                baseAnchors.Add((width, height));
            }
        }

        return baseAnchors;
    }

    /// <summary>
    /// Gets the number of anchors per feature map location.
    /// </summary>
    public int NumAnchorsPerLocation => AspectRatios.Length * Scales.Length;

    /// <summary>
    /// Gets the total number of anchors for an image.
    /// </summary>
    /// <param name="imageHeight">Image height.</param>
    /// <param name="imageWidth">Image width.</param>
    /// <returns>Total anchor count.</returns>
    public int GetTotalAnchorCount(int imageHeight, int imageWidth)
    {
        int total = 0;
        for (int level = 0; level < Strides.Length; level++)
        {
            int stride = Strides[level];
            int featureHeight = (imageHeight + stride - 1) / stride;
            int featureWidth = (imageWidth + stride - 1) / stride;
            total += featureHeight * featureWidth * NumAnchorsPerLocation;
        }
        return total;
    }

    /// <summary>
    /// Creates an anchor generator with YOLO-style anchors.
    /// </summary>
    /// <param name="anchors">Array of (width, height) anchor pairs for each level.</param>
    /// <param name="strides">Feature map strides.</param>
    /// <returns>Configured anchor generator.</returns>
    public static AnchorGenerator<T> CreateYOLOAnchors(
        double[,] anchors,
        int[] strides)
    {
        // For YOLO, we use fixed anchor sizes directly instead of base sizes with scales
        // This is a simplified version that works with the standard approach
        int numLevels = strides.Length;
        var baseSizes = new double[numLevels];

        // Extract base sizes from anchor definitions
        for (int i = 0; i < numLevels && i < anchors.GetLength(0) / 3; i++)
        {
            // Take average of anchors at each level as base size
            double avgWidth = 0, avgHeight = 0;
            for (int j = 0; j < 3; j++)
            {
                int idx = i * 3 + j;
                if (idx < anchors.GetLength(0))
                {
                    avgWidth += anchors[idx, 0];
                    avgHeight += anchors[idx, 1];
                }
            }
            baseSizes[i] = Math.Sqrt(avgWidth * avgHeight / 9);
        }

        return new AnchorGenerator<T>(
            baseSizes,
            aspectRatios: new double[] { 0.5, 1.0, 2.0 },
            scales: new double[] { 1.0 },
            strides: strides);
    }

    /// <summary>
    /// Creates an anchor generator with Faster R-CNN style anchors.
    /// </summary>
    /// <returns>Anchor generator for Faster R-CNN.</returns>
    public static AnchorGenerator<T> CreateFasterRCNNAnchors()
    {
        return new AnchorGenerator<T>(
            baseSizes: new double[] { 32, 64, 128, 256, 512 },
            aspectRatios: new double[] { 0.5, 1.0, 2.0 },
            scales: new double[] { 1.0 },
            strides: new int[] { 4, 8, 16, 32, 64 });
    }

    /// <summary>
    /// Creates an anchor generator with RetinaNet style anchors.
    /// </summary>
    /// <returns>Anchor generator for RetinaNet.</returns>
    public static AnchorGenerator<T> CreateRetinaNetAnchors()
    {
        return new AnchorGenerator<T>(
            baseSizes: new double[] { 32, 64, 128, 256, 512 },
            aspectRatios: new double[] { 0.5, 1.0, 2.0 },
            scales: new double[] { 1.0, 1.26, 1.59 },  // 2^0, 2^(1/3), 2^(2/3)
            strides: new int[] { 8, 16, 32, 64, 128 });
    }
}

/// <summary>
/// Pre-defined anchor configurations for common models.
/// </summary>
public static class AnchorPresets
{
    /// <summary>
    /// YOLOv8 default anchors per level (not used in anchor-free YOLOv8, but available for compatibility).
    /// </summary>
    public static readonly double[,] YOLOv8Anchors = new double[,]
    {
        // Level 0 (stride 8) - small objects
        { 10, 13 }, { 16, 30 }, { 33, 23 },
        // Level 1 (stride 16) - medium objects
        { 30, 61 }, { 62, 45 }, { 59, 119 },
        // Level 2 (stride 32) - large objects
        { 116, 90 }, { 156, 198 }, { 373, 326 }
    };

    /// <summary>
    /// YOLOv5 default anchors.
    /// </summary>
    public static readonly double[,] YOLOv5Anchors = new double[,]
    {
        { 10, 13 }, { 16, 30 }, { 33, 23 },
        { 30, 61 }, { 62, 45 }, { 59, 119 },
        { 116, 90 }, { 156, 198 }, { 373, 326 }
    };

    /// <summary>
    /// Standard strides for YOLO models.
    /// </summary>
    public static readonly int[] YOLOStrides = new[] { 8, 16, 32 };

    /// <summary>
    /// Standard strides for 5-level FPN (Faster R-CNN, RetinaNet).
    /// </summary>
    public static readonly int[] FPNStrides = new[] { 4, 8, 16, 32, 64 };
}
