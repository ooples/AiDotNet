namespace AiDotNet.ComputerVision.Detection;

/// <summary>A foreground label and a normalized center-format box used to train a detector.</summary>
/// <typeparam name="T">The detector's numeric type.</typeparam>
/// <remarks>
/// Coordinates are center-x, center-y, width and height, each relative to its own image dimension.
/// Width and height must be positive. This type does not clip boxes or infer a coordinate format.
/// </remarks>
public sealed class DetectionTrainingTarget<T>
{
    /// <summary>Creates an immutable target with normalized center-format coordinates.</summary>
    public DetectionTrainingTarget(int classId, T centerX, T centerY, T width, T height)
    {
        if (classId < 0)
            throw new ArgumentOutOfRangeException(nameof(classId), "A target must have a nonnegative foreground class.");
        ValidateCoordinate(centerX, nameof(centerX), positive: false);
        ValidateCoordinate(centerY, nameof(centerY), positive: false);
        ValidateCoordinate(width, nameof(width), positive: true);
        ValidateCoordinate(height, nameof(height), positive: true);
        ClassId = classId;
        CenterX = centerX;
        CenterY = centerY;
        Width = width;
        Height = height;
    }

    /// <summary>Gets the zero-based foreground class; the no-object class is never a target.</summary>
    public int ClassId { get; }
    /// <summary>Gets the horizontal center divided by image width.</summary>
    public T CenterX { get; }
    /// <summary>Gets the vertical center divided by image height.</summary>
    public T CenterY { get; }
    /// <summary>Gets box width divided by image width.</summary>
    public T Width { get; }
    /// <summary>Gets box height divided by image height.</summary>
    public T Height { get; }

    /// <summary>Converts a pixel-space top-left xywh box without assuming a square image.</summary>
    public static DetectionTrainingTarget<T> FromPixelXywh(
        int classId, T x, T y, T width, T height, int imageWidth, int imageHeight)
    {
        if (imageWidth <= 0) throw new ArgumentOutOfRangeException(nameof(imageWidth));
        if (imageHeight <= 0) throw new ArgumentOutOfRangeException(nameof(imageHeight));
        var ops = MathHelper.GetNumericOperations<T>();
        return FromNormalizedXywh(classId,
            ops.Divide(x, ops.FromDouble(imageWidth)),
            ops.Divide(y, ops.FromDouble(imageHeight)),
            ops.Divide(width, ops.FromDouble(imageWidth)),
            ops.Divide(height, ops.FromDouble(imageHeight)));
    }

    internal static DetectionTrainingTarget<T> FromNormalizedXywh(int classId, T x, T y, T width, T height)
    {
        ValidateCoordinate(x, nameof(x), positive: false);
        ValidateCoordinate(y, nameof(y), positive: false);
        var ops = MathHelper.GetNumericOperations<T>();
        T half = ops.FromDouble(0.5);
        return new DetectionTrainingTarget<T>(classId,
            ops.Add(x, ops.Multiply(width, half)), ops.Add(y, ops.Multiply(height, half)), width, height);
    }

    private static void ValidateCoordinate(T value, string parameterName, bool positive)
    {
        double coordinate = MathHelper.GetNumericOperations<T>().ToDouble(value);
        if (double.IsNaN(coordinate) || double.IsInfinity(coordinate))
            throw new ArgumentOutOfRangeException(parameterName, "Box coordinates must be finite.");
        if (coordinate < 0 || coordinate > 1 || (positive && coordinate == 0))
            throw new ArgumentOutOfRangeException(parameterName, "Center coordinates must be in [0, 1] and extents in (0, 1].");
    }
}
