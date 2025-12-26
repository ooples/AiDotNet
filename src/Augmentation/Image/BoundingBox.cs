namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Specifies the format of bounding box coordinates.
/// </summary>
public enum BoundingBoxFormat
{
    /// <summary>
    /// [x_min, y_min, x_max, y_max] - Absolute pixel coordinates.
    /// </summary>
    XYXY,

    /// <summary>
    /// [x_min, y_min, width, height] - Top-left corner with dimensions.
    /// </summary>
    XYWH,

    /// <summary>
    /// [x_center, y_center, width, height] - Center point with dimensions.
    /// </summary>
    CXCYWH,

    /// <summary>
    /// [x_center, y_center, width, height] normalized to [0, 1] - YOLO format.
    /// </summary>
    YOLO,

    /// <summary>
    /// [x_min, y_min, width, height] - COCO format (same as XYWH).
    /// </summary>
    COCO,

    /// <summary>
    /// [x_min, y_min, x_max, y_max] - Pascal VOC format (same as XYXY).
    /// </summary>
    PascalVOC
}

/// <summary>
/// Represents a bounding box annotation for object detection.
/// </summary>
/// <remarks>
/// <para>
/// Bounding boxes are rectangular regions that localize objects in images.
/// This class supports multiple coordinate formats used by different frameworks
/// and can convert between them.
/// </para>
/// <para><b>For Beginners:</b> A bounding box is simply a rectangle around an object
/// in an image, defined by its coordinates. Different systems use different ways
/// to specify these coordinates (corners vs center, pixels vs percentages).
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for coordinates.</typeparam>
public class BoundingBox<T>
{
    /// <summary>
    /// Gets or sets the X coordinate of the first point.
    /// </summary>
    /// <remarks>
    /// Interpretation depends on format: x_min (XYXY, XYWH), x_center (CXCYWH, YOLO).
    /// </remarks>
    public T X1 { get; set; }

    /// <summary>
    /// Gets or sets the Y coordinate of the first point.
    /// </summary>
    public T Y1 { get; set; }

    /// <summary>
    /// Gets or sets the X coordinate of the second point or width.
    /// </summary>
    public T X2 { get; set; }

    /// <summary>
    /// Gets or sets the Y coordinate of the second point or height.
    /// </summary>
    public T Y2 { get; set; }

    /// <summary>
    /// Gets or sets the class label index.
    /// </summary>
    public int ClassIndex { get; set; }

    /// <summary>
    /// Gets or sets the class label name.
    /// </summary>
    public string? ClassName { get; set; }

    /// <summary>
    /// Gets or sets the confidence score (if from a detector).
    /// </summary>
    public T? Confidence { get; set; }

    /// <summary>
    /// Gets or sets the current coordinate format.
    /// </summary>
    public BoundingBoxFormat Format { get; set; }

    /// <summary>
    /// Gets or sets the image width (needed for normalized formats).
    /// </summary>
    public int ImageWidth { get; set; }

    /// <summary>
    /// Gets or sets the image height (needed for normalized formats).
    /// </summary>
    public int ImageHeight { get; set; }

    /// <summary>
    /// Gets or sets additional metadata.
    /// </summary>
    public IDictionary<string, object>? Metadata { get; set; }

    /// <summary>
    /// Creates an empty bounding box.
    /// </summary>
    public BoundingBox()
    {
        X1 = default!;
        Y1 = default!;
        X2 = default!;
        Y2 = default!;
        Format = BoundingBoxFormat.XYXY;
    }

    /// <summary>
    /// Creates a bounding box with the specified coordinates.
    /// </summary>
    /// <param name="x1">The first X coordinate.</param>
    /// <param name="y1">The first Y coordinate.</param>
    /// <param name="x2">The second X coordinate or width.</param>
    /// <param name="y2">The second Y coordinate or height.</param>
    /// <param name="format">The coordinate format.</param>
    /// <param name="classIndex">The class label index.</param>
    public BoundingBox(T x1, T y1, T x2, T y2, BoundingBoxFormat format = BoundingBoxFormat.XYXY, int classIndex = 0)
    {
        X1 = x1;
        Y1 = y1;
        X2 = x2;
        Y2 = y2;
        Format = format;
        ClassIndex = classIndex;
    }

    /// <summary>
    /// Creates a deep copy of this bounding box.
    /// </summary>
    /// <returns>A new bounding box with the same values.</returns>
    public BoundingBox<T> Clone()
    {
        return new BoundingBox<T>
        {
            X1 = X1,
            Y1 = Y1,
            X2 = X2,
            Y2 = Y2,
            ClassIndex = ClassIndex,
            ClassName = ClassName,
            Confidence = Confidence,
            Format = Format,
            ImageWidth = ImageWidth,
            ImageHeight = ImageHeight,
            Metadata = Metadata is not null ? new Dictionary<string, object>(Metadata) : null
        };
    }

    /// <summary>
    /// Converts this bounding box to XYXY format.
    /// </summary>
    /// <returns>Coordinates as [x_min, y_min, x_max, y_max].</returns>
    public (double xMin, double yMin, double xMax, double yMax) ToXYXY()
    {
        double x1 = Convert.ToDouble(X1);
        double y1 = Convert.ToDouble(Y1);
        double x2 = Convert.ToDouble(X2);
        double y2 = Convert.ToDouble(Y2);

        return Format switch
        {
            BoundingBoxFormat.XYXY or BoundingBoxFormat.PascalVOC => (x1, y1, x2, y2),
            BoundingBoxFormat.XYWH or BoundingBoxFormat.COCO => (x1, y1, x1 + x2, y1 + y2),
            BoundingBoxFormat.CXCYWH => (x1 - x2 / 2, y1 - y2 / 2, x1 + x2 / 2, y1 + y2 / 2),
            BoundingBoxFormat.YOLO => ImageWidth <= 0 || ImageHeight <= 0
                ? throw new InvalidOperationException("ImageWidth and ImageHeight must be set for YOLO format conversion.")
                : (
                (x1 - x2 / 2) * ImageWidth,
                (y1 - y2 / 2) * ImageHeight,
                (x1 + x2 / 2) * ImageWidth,
                (y1 + y2 / 2) * ImageHeight),
            _ => throw new InvalidOperationException($"Unknown format: {Format}")
        };
    }

    /// <summary>
    /// Converts this bounding box to XYWH format.
    /// </summary>
    /// <returns>Coordinates as [x_min, y_min, width, height].</returns>
    public (double x, double y, double width, double height) ToXYWH()
    {
        var (xMin, yMin, xMax, yMax) = ToXYXY();
        return (xMin, yMin, xMax - xMin, yMax - yMin);
    }

    /// <summary>
    /// Converts this bounding box to CXCYWH format.
    /// </summary>
    /// <returns>Coordinates as [x_center, y_center, width, height].</returns>
    public (double cx, double cy, double width, double height) ToCXCYWH()
    {
        var (xMin, yMin, xMax, yMax) = ToXYXY();
        double width = xMax - xMin;
        double height = yMax - yMin;
        return (xMin + width / 2, yMin + height / 2, width, height);
    }

    /// <summary>
    /// Converts this bounding box to YOLO format (normalized).
    /// </summary>
    /// <returns>Coordinates as [x_center, y_center, width, height] normalized to [0, 1].</returns>
    public (double cx, double cy, double width, double height) ToYOLO()
    {
        if (ImageWidth <= 0 || ImageHeight <= 0)
        {
            throw new InvalidOperationException("ImageWidth and ImageHeight must be set for YOLO format conversion.");
        }

        var (cx, cy, w, h) = ToCXCYWH();
        return (cx / ImageWidth, cy / ImageHeight, w / ImageWidth, h / ImageHeight);
    }

    /// <summary>
    /// Calculates the area of this bounding box.
    /// </summary>
    /// <returns>The area in square pixels.</returns>
    public double Area()
    {
        var (xMin, yMin, xMax, yMax) = ToXYXY();
        double width = Math.Max(0, xMax - xMin);
        double height = Math.Max(0, yMax - yMin);
        return width * height;
    }

    /// <summary>
    /// Calculates the IoU (Intersection over Union) with another bounding box.
    /// </summary>
    /// <param name="other">The other bounding box.</param>
    /// <returns>The IoU value between 0 and 1.</returns>
    public double IoU(BoundingBox<T> other)
    {
        var (xMin1, yMin1, xMax1, yMax1) = ToXYXY();
        var (xMin2, yMin2, xMax2, yMax2) = other.ToXYXY();

        double interXMin = Math.Max(xMin1, xMin2);
        double interYMin = Math.Max(yMin1, yMin2);
        double interXMax = Math.Min(xMax1, xMax2);
        double interYMax = Math.Min(yMax1, yMax2);

        double interWidth = Math.Max(0, interXMax - interXMin);
        double interHeight = Math.Max(0, interYMax - interYMin);
        double intersection = interWidth * interHeight;

        double area1 = Area();
        double area2 = other.Area();
        double union = area1 + area2 - intersection;

        return union > 0 ? intersection / union : 0;
    }

    /// <summary>
    /// Clips this bounding box to image boundaries.
    /// </summary>
    /// <param name="width">The image width.</param>
    /// <param name="height">The image height.</param>
    public void Clip(int width, int height)
    {
        var (xMin, yMin, xMax, yMax) = ToXYXY();

        xMin = Math.Max(0, Math.Min(width, xMin));
        yMin = Math.Max(0, Math.Min(height, yMin));
        xMax = Math.Max(0, Math.Min(width, xMax));
        yMax = Math.Max(0, Math.Min(height, yMax));

        // Update coordinates based on format
        X1 = (T)Convert.ChangeType(xMin, typeof(T));
        Y1 = (T)Convert.ChangeType(yMin, typeof(T));
        X2 = (T)Convert.ChangeType(xMax, typeof(T));
        Y2 = (T)Convert.ChangeType(yMax, typeof(T));
        Format = BoundingBoxFormat.XYXY;
        ImageWidth = width;
        ImageHeight = height;
    }

    /// <summary>
    /// Checks if this bounding box is valid (has positive area).
    /// </summary>
    /// <returns>True if the box has positive width and height.</returns>
    public bool IsValid()
    {
        var (xMin, yMin, xMax, yMax) = ToXYXY();
        return xMax > xMin && yMax > yMin;
    }
}
