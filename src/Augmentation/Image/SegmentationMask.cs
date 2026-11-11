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

        var engine = AiDotNetEngine.Current;
        return (int)NumOps.ToDouble(engine.TensorSum(Foreground(ToDense())));
    }

    /// <summary>
    /// Calculates the bounding box of the mask.
    /// </summary>
    /// <returns>The bounding box (xMin, yMin, xMax, yMax).</returns>
    public (int xMin, int yMin, int xMax, int yMax) GetBoundingBox()
    {
        var dense = ToDense();
        var engine = AiDotNetEngine.Current;
        var foreground = Foreground(dense);
        if (!NumOps.GreaterThan(engine.TensorMaxValue(foreground), NumOps.Zero))
        {
            return (0, 0, 0, 0); // Empty mask
        }

        // First/last row and column that contain a foreground pixel
        var rows = engine.ReduceMax(foreground, [1], false);
        var cols = engine.ReduceMax(foreground, [0], false);
        int yMin = engine.TensorArgMax(rows, 0).GetFlat(0);
        int xMin = engine.TensorArgMax(cols, 0).GetFlat(0);
        int yMax = rows.Length - 1 - engine.TensorArgMax(engine.TensorFlip(rows, [0]), 0).GetFlat(0);
        int xMax = cols.Length - 1 - engine.TensorArgMax(engine.TensorFlip(cols, [0]), 0).GetFlat(0);

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

        var engine = AiDotNetEngine.Current;
        var m1 = Foreground(mask1);
        var m2 = Foreground(mask2);
        int intersection = (int)NumOps.ToDouble(engine.TensorSum(engine.TensorMultiply(m1, m2)));
        int union = (int)NumOps.ToDouble(engine.TensorSum(engine.TensorMax(m1, m2)));

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

        var engine = AiDotNetEngine.Current;
        var m1 = Foreground(mask1);
        var m2 = Foreground(mask2);
        int intersection = (int)NumOps.ToDouble(engine.TensorSum(engine.TensorMultiply(m1, m2)));
        int sum = (int)NumOps.ToDouble(engine.TensorSum(m1)) + (int)NumOps.ToDouble(engine.TensorSum(m2));

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

        // Runs alternate background/foreground starting with background; pixel p lies in the first
        // run whose cumulative end exceeds p, and is foreground when that run's index is odd
        var engine = AiDotNetEngine.Current;
        int total = Width * Height;
        if (total == 0 || RLECounts.Length == 0)
        {
            return new T[Height, Width];
        }

        var ends = engine.TensorCumSum(new Tensor<int>((int[])RLECounts.Clone(), [RLECounts.Length]), 0);
        var positions = engine.TensorArange<int>(0, total, 1);
        var run = engine.TensorSearchSorted(ends, positions, true);
        var odd = engine.TensorSubtract(run, engine.TensorMultiplyScalar(engine.TensorDivideScalar(run, 2), 2));
        var foreground = engine.TensorMultiply(odd, engine.TensorLessThan(run, RLECounts.Length));
        return ToArray2D(foreground.Cast<T>(), Height, Width);
    }

    /// <summary>
    /// Encodes dense mask to RLE format.
    /// </summary>
    private int[] EncodeRLE(T[,] dense)
    {
        // Runs of equal foreground/background values in row-major order; RLE starts with a
        // background run, so a leading foreground run gets a zero-length background run first
        var engine = AiDotNetEngine.Current;
        var foreground = Foreground(dense);
        if (foreground.Length == 0)
        {
            return [0];
        }

        var flat = engine.Reshape(foreground, [foreground.Length]);
        var (values, _, runCounts) = engine.TensorUniqueConsecutiveWithInfo(flat, false, true);
        int[] counts = runCounts is null ? [foreground.Length] : runCounts.ToArray();
        bool startsWithForeground = NumOps.GreaterThan(values.GetFlat(0), NumOps.Zero);
        return startsWithForeground ? [0, .. counts] : counts;
    }

    /// <summary>
    /// Copies a dense mask into a tensor (row-major) of the same shape.
    /// </summary>
    private static Tensor<T> ToTensor(T[,] dense)
    {
        return new Tensor<T>(dense.Cast<T>().ToArray(), [dense.GetLength(0), dense.GetLength(1)]);
    }

    /// <summary>
    /// 1 where the dense mask is greater than zero, 0 elsewhere.
    /// </summary>
    private static Tensor<T> Foreground(T[,] dense)
    {
        return AiDotNetEngine.Current.TensorGreaterThan(ToTensor(dense), NumOps.Zero);
    }

    /// <summary>
    /// Copies a row-major tensor into a new [rows, columns] array in one block copy (an element copy
    /// for non-primitive element types, which block copies do not support).
    /// </summary>
    private static T[,] ToArray2D(Tensor<T> values, int rows, int columns)
    {
        var flat = values.Contiguous().ToArray();
        var result = new T[rows, columns];
        if (typeof(T).IsPrimitive)
        {
            Buffer.BlockCopy(flat, 0, result, 0, Buffer.ByteLength(flat));
            return result;
        }

        for (int i = 0; i < flat.Length; i++)
        {
            result[i / columns, i % columns] = flat[i];
        }

        return result;
    }

    /// <summary>
    /// Rasterizes polygons to dense format.
    /// </summary>
    private T[,] RasterizePolygons()
    {
        if (Polygons is null || Polygons.Count == 0 || Width == 0 || Height == 0)
        {
            return new T[Height, Width];
        }

        var engine = AiDotNetEngine.Current;
        var filled = new Tensor<int>([Height, Width]);
        foreach (var polygon in Polygons.Where(p => p.Count >= 3))
        {
            filled = engine.TensorMax(filled, FillPolygon(polygon));
        }

        return ToArray2D(filled.Cast<T>(), Height, Width);
    }

    /// <summary>
    /// Scanline fill of one polygon as a [height, width] 0/1 tensor. Row y crosses edge (p1, p2) when
    /// y lies in [min(y1, y2), max(y1, y2)) at x = x1 + (y - y1) / (y2 - y1) * (x2 - x1); pixel x is
    /// filled when it lies between a pair of sorted crossings (inclusive), i.e. when the number of
    /// crossings below x is odd or a crossing equals x.
    /// </summary>
    private Tensor<int> FillPolygon(IList<(double x, double y)> polygon)
    {
        var engine = AiDotNetEngine.Current;
        int edges = polygon.Count;
        var vertexX = polygon.Select(v => v.x).ToArray();
        var vertexY = polygon.Select(v => v.y).ToArray();
        var x1 = new Tensor<T>(Array.ConvertAll(vertexX, NumOps.FromDouble), [1, edges]);
        var y1 = new Tensor<T>(Array.ConvertAll(vertexY, NumOps.FromDouble), [1, edges]);
        var x2 = engine.TensorRoll(x1, [-1], [1]);
        var y2 = engine.TensorRoll(y1, [-1], [1]);

        // Crossing of every row with every edge: [height, edges]
        int[] rowEdge = [Height, edges];
        var y = engine.TensorBroadcastTo(engine.Reshape(engine.TensorArange<T>(0, Height, 1), [Height, 1]), rowEdge);
        var y1b = engine.TensorBroadcastTo(y1, rowEdge);
        var y2b = engine.TensorBroadcastTo(y2, rowEdge);
        var upward = engine.TensorMultiply(
            engine.ScalarMinusTensor(NumOps.One, engine.TensorGreaterThan(y1b, y)), engine.TensorGreaterThan(y2b, y));
        var downward = engine.TensorMultiply(
            engine.ScalarMinusTensor(NumOps.One, engine.TensorGreaterThan(y2b, y)), engine.TensorGreaterThan(y1b, y));
        var crosses = engine.TensorAdd(upward, downward);
        var rise = engine.TensorSubtract(y2, y1);
        var safeRise = engine.TensorWhere(engine.TensorEquals(rise, NumOps.Zero), ImageTensor<T>.Constant(rise, 1.0), rise);
        var slope = engine.TensorDivide(engine.TensorSubtract(x2, x1), safeRise);
        var crossingX = engine.TensorAdd(
            engine.TensorBroadcastTo(x1, rowEdge),
            engine.TensorMultiply(engine.TensorSubtract(y, y1b), engine.TensorBroadcastTo(slope, rowEdge)));

        // Crossings strictly below and at-or-below every pixel column: [height, width]
        int[] grid = [Height, Width, edges];
        var column = engine.TensorBroadcastTo(engine.Reshape(engine.TensorArange<T>(0, Width, 1), [1, Width, 1]), grid);
        var crossing = engine.TensorBroadcastTo(engine.Reshape(crossingX, [Height, 1, edges]), grid);
        var counted = engine.TensorBroadcastTo(engine.Reshape(crosses, [Height, 1, edges]), grid);
        var below = engine.ReduceSum(engine.TensorMultiply(counted, engine.TensorLessThan(crossing, column)), [2], false).Cast<int>();
        var atOrBelow = engine.ReduceSum(
            engine.TensorMultiply(counted, engine.ScalarMinusTensor(NumOps.One, engine.TensorGreaterThan(crossing, column))), [2], false)
            .Cast<int>();

        var oddBelow = engine.TensorSubtract(below, engine.TensorMultiplyScalar(engine.TensorDivideScalar(below, 2), 2));
        var onCrossing = engine.TensorGreaterThan(atOrBelow, below);
        return engine.TensorMax(oddBelow, onCrossing);
    }
}
