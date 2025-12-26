using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Specifies the type of segmentation mask.
/// </summary>
public enum MaskType
{
    /// <summary>
    /// Binary mask where each pixel is 0 or 1.
    /// </summary>
    Binary,

    /// <summary>
    /// Semantic segmentation where each pixel has a class label.
    /// </summary>
    Semantic,

    /// <summary>
    /// Instance segmentation where each object instance has a unique ID.
    /// </summary>
    Instance,

    /// <summary>
    /// Panoptic segmentation combining semantic and instance.
    /// </summary>
    Panoptic
}

/// <summary>
/// Specifies the encoding format for segmentation masks.
/// </summary>
public enum MaskEncoding
{
    /// <summary>
    /// Raw 2D array of pixel values.
    /// </summary>
    Dense,

    /// <summary>
    /// Run-length encoding (COCO-style).
    /// </summary>
    RLE,

    /// <summary>
    /// Polygon vertices.
    /// </summary>
    Polygon
}

/// <summary>
/// Represents a segmentation mask for pixel-level annotations.
/// </summary>
/// <remarks>
/// <para>
/// Segmentation masks provide pixel-level object delineation, used for:
/// - Semantic segmentation (what class is each pixel)
/// - Instance segmentation (which object instance is each pixel)
/// - Panoptic segmentation (both semantic and instance)
/// </para>
/// <para><b>For Beginners:</b> While a bounding box draws a rectangle around an object,
/// a segmentation mask precisely outlines the object's shape. When you rotate or
/// flip an image, the mask must be transformed identically.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for mask values.</typeparam>
public class SegmentationMask<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets or sets the mask data as a 2D array [height, width].
    /// </summary>
    /// <remarks>
    /// For binary masks: 0 = background, 1 = foreground.
    /// For semantic masks: each value is a class index.
    /// For instance masks: each value is an instance ID.
    /// </remarks>
    public T[,]? MaskData { get; set; }

    /// <summary>
    /// Gets or sets the RLE-encoded mask (if using RLE encoding).
    /// </summary>
    public int[]? RLECounts { get; set; }

    /// <summary>
    /// Gets or sets the polygon vertices (if using polygon encoding).
    /// </summary>
    /// <remarks>
    /// List of polygons, each polygon is a list of (x, y) coordinates.
    /// </remarks>
    public IList<IList<(double x, double y)>>? Polygons { get; set; }

    /// <summary>
    /// Gets or sets the mask type.
    /// </summary>
    public MaskType Type { get; set; } = MaskType.Binary;

    /// <summary>
    /// Gets or sets the current encoding format.
    /// </summary>
    public MaskEncoding Encoding { get; set; } = MaskEncoding.Dense;

    /// <summary>
    /// Gets or sets the mask width.
    /// </summary>
    public int Width { get; set; }

    /// <summary>
    /// Gets or sets the mask height.
    /// </summary>
    public int Height { get; set; }

    /// <summary>
    /// Gets or sets the class index (for semantic/instance masks).
    /// </summary>
    public int ClassIndex { get; set; }

    /// <summary>
    /// Gets or sets the class name.
    /// </summary>
    public string? ClassName { get; set; }

    /// <summary>
    /// Gets or sets the instance ID (for instance/panoptic masks).
    /// </summary>
    public int InstanceId { get; set; }

    /// <summary>
    /// Gets or sets the confidence score (if from a segmenter).
    /// </summary>
    public T? Confidence { get; set; }

    /// <summary>
    /// Gets or sets additional metadata.
    /// </summary>
    public IDictionary<string, object>? Metadata { get; set; }

    /// <summary>
    /// Creates an empty segmentation mask.
    /// </summary>
    public SegmentationMask()
    {
    }

    /// <summary>
    /// Creates a segmentation mask from dense data.
    /// </summary>
    /// <param name="maskData">The 2D mask array [height, width].</param>
    /// <param name="type">The mask type.</param>
    /// <param name="classIndex">The class index.</param>
    public SegmentationMask(T[,] maskData, MaskType type = MaskType.Binary, int classIndex = 0)
    {
        MaskData = maskData;
        Height = maskData.GetLength(0);
        Width = maskData.GetLength(1);
        Type = type;
        Encoding = MaskEncoding.Dense;
        ClassIndex = classIndex;
    }

    /// <summary>
    /// Creates a segmentation mask from RLE encoding.
    /// </summary>
    /// <param name="rleCounts">The RLE counts.</param>
    /// <param name="width">The mask width.</param>
    /// <param name="height">The mask height.</param>
    /// <param name="type">The mask type.</param>
    public SegmentationMask(int[] rleCounts, int width, int height, MaskType type = MaskType.Binary)
    {
        RLECounts = rleCounts;
        Width = width;
        Height = height;
        Type = type;
        Encoding = MaskEncoding.RLE;
    }

    /// <summary>
    /// Creates a segmentation mask from polygon vertices.
    /// </summary>
    /// <param name="polygons">The polygon vertices.</param>
    /// <param name="width">The image width.</param>
    /// <param name="height">The image height.</param>
    /// <param name="type">The mask type.</param>
    public SegmentationMask(IList<IList<(double x, double y)>> polygons, int width, int height, MaskType type = MaskType.Binary)
    {
        Polygons = polygons;
        Width = width;
        Height = height;
        Type = type;
        Encoding = MaskEncoding.Polygon;
    }

    /// <summary>
    /// Creates a deep copy of this mask.
    /// </summary>
    /// <returns>A new mask with copied data.</returns>
    public SegmentationMask<T> Clone()
    {
        var clone = new SegmentationMask<T>
        {
            Type = Type,
            Encoding = Encoding,
            Width = Width,
            Height = Height,
            ClassIndex = ClassIndex,
            ClassName = ClassName,
            InstanceId = InstanceId,
            Confidence = Confidence,
            Metadata = Metadata is not null ? new Dictionary<string, object>(Metadata) : null
        };

        if (MaskData is not null)
        {
            clone.MaskData = new T[Height, Width];
            Array.Copy(MaskData, clone.MaskData, MaskData.Length);
        }

        if (RLECounts is not null)
        {
            clone.RLECounts = new int[RLECounts.Length];
            Array.Copy(RLECounts, clone.RLECounts, RLECounts.Length);
        }

        if (Polygons is not null)
        {
            clone.Polygons = new List<IList<(double x, double y)>>();
            foreach (var polygon in Polygons)
            {
                clone.Polygons.Add(new List<(double x, double y)>(polygon));
            }
        }

        return clone;
    }

    /// <summary>
    /// Converts this mask to dense format.
    /// </summary>
    /// <returns>The dense mask data.</returns>
    public T[,] ToDense()
    {
        if (MaskData is not null)
        {
            return MaskData;
        }

        if (Encoding == MaskEncoding.RLE && RLECounts is not null)
        {
            return DecodeRLE();
        }

        if (Encoding == MaskEncoding.Polygon && Polygons is not null)
        {
            return RasterizePolygons();
        }

        throw new InvalidOperationException("No mask data available to convert.");
    }

    /// <summary>
    /// Converts this mask to RLE format.
    /// </summary>
    /// <returns>The RLE counts.</returns>
    public int[] ToRLE()
    {
        if (RLECounts is not null)
        {
            return RLECounts;
        }

        var dense = ToDense();
        return EncodeRLE(dense);
    }

    /// <summary>
    /// Calculates the area (number of foreground pixels).
    /// </summary>
    /// <returns>The mask area in pixels.</returns>
    public int Area()
    {
        if (Encoding == MaskEncoding.RLE && RLECounts is not null)
        {
            // In RLE, odd indices are foreground runs
            int area = 0;
            for (int i = 1; i < RLECounts.Length; i += 2)
            {
                area += RLECounts[i];
            }
            return area;
        }

        var dense = ToDense();
        int count = 0;
        for (int y = 0; y < Height; y++)
        {
            for (int x = 0; x < Width; x++)
            {
                if (NumOps.ToDouble(dense[y, x]) > 0)
                {
                    count++;
                }
            }
        }
        return count;
    }

    /// <summary>
    /// Calculates the bounding box of the mask.
    /// </summary>
    /// <returns>The bounding box (xMin, yMin, xMax, yMax).</returns>
    public (int xMin, int yMin, int xMax, int yMax) GetBoundingBox()
    {
        var dense = ToDense();
        int xMin = Width, yMin = Height, xMax = -1, yMax = -1;

        for (int y = 0; y < Height; y++)
        {
            for (int x = 0; x < Width; x++)
            {
                if (NumOps.ToDouble(dense[y, x]) > 0)
                {
                    xMin = Math.Min(xMin, x);
                    yMin = Math.Min(yMin, y);
                    xMax = Math.Max(xMax, x);
                    yMax = Math.Max(yMax, y);
                }
            }
        }

        if (xMax < 0)
        {
            return (0, 0, 0, 0); // Empty mask
        }

        return (xMin, yMin, xMax + 1, yMax + 1);
    }

    /// <summary>
    /// Calculates the IoU (Intersection over Union) with another mask.
    /// </summary>
    /// <param name="other">The other mask.</param>
    /// <returns>The IoU value between 0 and 1.</returns>
    public double IoU(SegmentationMask<T> other)
    {
        var mask1 = ToDense();
        var mask2 = other.ToDense();

        if (Width != other.Width || Height != other.Height)
        {
            throw new ArgumentException("Masks must have the same dimensions.");
        }

        int intersection = 0;
        int union = 0;

        for (int y = 0; y < Height; y++)
        {
            for (int x = 0; x < Width; x++)
            {
                bool m1 = NumOps.ToDouble(mask1[y, x]) > 0;
                bool m2 = NumOps.ToDouble(mask2[y, x]) > 0;

                if (m1 && m2) intersection++;
                if (m1 || m2) union++;
            }
        }

        return union > 0 ? (double)intersection / union : 0;
    }

    /// <summary>
    /// Calculates the Dice coefficient with another mask.
    /// </summary>
    /// <param name="other">The other mask.</param>
    /// <returns>The Dice coefficient between 0 and 1.</returns>
    public double Dice(SegmentationMask<T> other)
    {
        var mask1 = ToDense();
        var mask2 = other.ToDense();

        if (Width != other.Width || Height != other.Height)
        {
            throw new ArgumentException("Masks must have the same dimensions.");
        }

        int intersection = 0;
        int sum = 0;

        for (int y = 0; y < Height; y++)
        {
            for (int x = 0; x < Width; x++)
            {
                bool m1 = NumOps.ToDouble(mask1[y, x]) > 0;
                bool m2 = NumOps.ToDouble(mask2[y, x]) > 0;

                if (m1 && m2) intersection++;
                if (m1) sum++;
                if (m2) sum++;
            }
        }

        return sum > 0 ? 2.0 * intersection / sum : 0;
    }

    /// <summary>
    /// Decodes RLE to dense format.
    /// </summary>
    private T[,] DecodeRLE()
    {
        if (RLECounts is null)
        {
            throw new InvalidOperationException("RLE counts are not set.");
        }

        var result = new T[Height, Width];
        int position = 0;
        bool isForeground = false;

        foreach (int count in RLECounts)
        {
            for (int i = 0; i < count && position < Width * Height; i++)
            {
                int y = position / Width;
                int x = position % Width;

                if (isForeground)
                {
                    result[y, x] = NumOps.FromDouble(1);
                }

                position++;
            }
            isForeground = !isForeground;
        }

        return result;
    }

    /// <summary>
    /// Encodes dense mask to RLE format.
    /// </summary>
    private int[] EncodeRLE(T[,] dense)
    {
        var counts = new List<int>();
        bool currentValue = false;
        int currentCount = 0;

        for (int y = 0; y < Height; y++)
        {
            for (int x = 0; x < Width; x++)
            {
                bool pixelValue = NumOps.ToDouble(dense[y, x]) > 0;

                if (pixelValue == currentValue)
                {
                    currentCount++;
                }
                else
                {
                    counts.Add(currentCount);
                    currentValue = pixelValue;
                    currentCount = 1;
                }
            }
        }

        counts.Add(currentCount);
        return counts.ToArray();
    }

    /// <summary>
    /// Rasterizes polygons to dense format.
    /// </summary>
    private T[,] RasterizePolygons()
    {
        if (Polygons is null || Polygons.Count == 0)
        {
            return new T[Height, Width];
        }

        var result = new T[Height, Width];

        foreach (var polygon in Polygons.Where(p => p.Count >= 3))
        {
            // Use scanline algorithm to fill polygon
            FillPolygon(result, polygon);
        }

        return result;
    }

    /// <summary>
    /// Fills a polygon using scanline algorithm.
    /// </summary>
    private void FillPolygon(T[,] result, IList<(double x, double y)> polygon)
    {
        // Find y bounds
        double minY = double.MaxValue, maxY = double.MinValue;
        foreach (var (x, y) in polygon)
        {
            minY = Math.Min(minY, y);
            maxY = Math.Max(maxY, y);
        }

        int yStart = Math.Max(0, (int)Math.Floor(minY));
        int yEnd = Math.Min(Height - 1, (int)Math.Ceiling(maxY));

        // Scanline for each y
        for (int y = yStart; y <= yEnd; y++)
        {
            var intersections = new List<double>();

            // Find intersections with polygon edges
            for (int i = 0; i < polygon.Count; i++)
            {
                var p1 = polygon[i];
                var p2 = polygon[(i + 1) % polygon.Count];

                if ((p1.y <= y && p2.y > y) || (p2.y <= y && p1.y > y))
                {
                    double xIntersect = p1.x + (y - p1.y) / (p2.y - p1.y) * (p2.x - p1.x);
                    intersections.Add(xIntersect);
                }
            }

            intersections.Sort();

            // Fill between pairs of intersections
            for (int i = 0; i + 1 < intersections.Count; i += 2)
            {
                int xStart = Math.Max(0, (int)Math.Ceiling(intersections[i]));
                int xEnd = Math.Min(Width - 1, (int)Math.Floor(intersections[i + 1]));

                for (int x = xStart; x <= xEnd; x++)
                {
                    result[y, x] = NumOps.FromDouble(1);
                }
            }
        }
    }
}
