
namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Rotates an image by a random angle within a specified range.
/// </summary>
/// <remarks>
/// <para>
/// Rotation randomly rotates the image around its center point by an angle sampled
/// from the specified range. This simulates viewing objects from slightly different
/// angles, which helps the model become robust to orientation variations.
/// </para>
/// <para><b>For Beginners:</b> Imagine tilting your camera slightly when taking a photo.
/// The same object photographed at a slight angle is still the same object. This augmentation
/// teaches your model to recognize objects even when they're not perfectly aligned.
/// </para>
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>Object classification where objects might appear at different angles</item>
/// <item>Document analysis (scanned documents may be slightly tilted)</item>
/// <item>Medical imaging where acquisition angle varies</item>
/// <item>Satellite imagery where orientation is arbitrary</item>
/// </list>
/// </para>
/// <para><b>When NOT to use:</b>
/// <list type="bullet">
/// <item>Facial recognition (faces should be upright)</item>
/// <item>Handwriting recognition (letters need consistent orientation)</item>
/// <item>Tasks where specific orientation is part of the classification</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class Rotation<T> : SpatialAugmentationBase<T, ImageTensor<T>>
{
    /// <summary>
    /// Gets the minimum rotation angle in degrees (can be negative for counter-clockwise).
    /// </summary>
    public double MinAngle { get; }

    /// <summary>
    /// Gets the maximum rotation angle in degrees.
    /// </summary>
    public double MaxAngle { get; }

    /// <summary>
    /// Gets the border fill mode when pixels fall outside the original image bounds.
    /// </summary>
    public BorderMode BorderMode { get; }

    /// <summary>
    /// Gets the constant value used when BorderMode is Constant.
    /// </summary>
    public T BorderValue { get; }

    /// <summary>
    /// Gets the interpolation mode for pixel sampling.
    /// </summary>
    public InterpolationMode Interpolation { get; }

    /// <summary>
    /// Creates a new rotation augmentation.
    /// </summary>
    /// <param name="minAngle">
    /// The minimum rotation angle in degrees. Use negative values for counter-clockwise rotation.
    /// Industry standard default is -15 degrees.
    /// </param>
    /// <param name="maxAngle">
    /// The maximum rotation angle in degrees.
    /// Industry standard default is 15 degrees.
    /// </param>
    /// <param name="probability">
    /// The probability of applying this augmentation (0.0 to 1.0).
    /// Industry standard default is 0.5.
    /// </param>
    /// <param name="borderMode">
    /// How to fill pixels that fall outside the original image bounds.
    /// Industry standard default is Reflect (mirror pixels at the edge).
    /// </param>
    /// <param name="borderValue">
    /// The constant value to use when borderMode is Constant.
    /// </param>
    /// <param name="interpolation">
    /// The interpolation mode for sampling pixels.
    /// Industry standard default is Bilinear for smooth results.
    /// </param>
    /// <remarks>
    /// <para><b>For Beginners:</b> The angle range controls how much rotation is applied.
    /// A range of -15 to +15 degrees creates subtle rotations that look natural.
    /// Larger ranges (like -45 to +45) create more dramatic rotations.
    /// </para>
    /// </remarks>
    public Rotation(
        double minAngle = -15.0,
        double maxAngle = 15.0,
        double probability = 0.5,
        BorderMode borderMode = BorderMode.Reflect,
        T? borderValue = default,
        InterpolationMode interpolation = InterpolationMode.Bilinear)
        : base(probability)
    {
        if (minAngle > maxAngle)
        {
            throw new ArgumentException("minAngle must be less than or equal to maxAngle", nameof(minAngle));
        }

        MinAngle = minAngle;
        MaxAngle = maxAngle;
        BorderMode = borderMode;
        BorderValue = borderValue ?? default!;
        Interpolation = interpolation;
    }

    /// <summary>
    /// Applies the rotation transformation and returns transform parameters.
    /// </summary>
    protected override (ImageTensor<T> data, IDictionary<string, object> parameters) ApplyWithTransformParams(
        ImageTensor<T> data,
        AugmentationContext<T> context)
    {
        // Sample a random angle
        double angleDegrees = context.GetRandomDouble(MinAngle, MaxAngle);
        double angleRadians = angleDegrees * Math.PI / 180.0;

        var result = data.Clone();
        int height = data.Height;
        int width = data.Width;
        int channels = data.Channels;

        // Center of rotation
        double centerX = width / 2.0;
        double centerY = height / 2.0;

        // Precompute rotation matrix components
        double cos = Math.Cos(angleRadians);
        double sin = Math.Sin(angleRadians);

        // For each pixel in the output, find the corresponding input pixel
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                // Translate to center, rotate (inverse), translate back
                double dx = x - centerX;
                double dy = y - centerY;

                // Inverse rotation to find source pixel
                double srcX = cos * dx + sin * dy + centerX;
                double srcY = -sin * dx + cos * dy + centerY;

                // Sample the source pixel
                for (int c = 0; c < channels; c++)
                {
                    T value = SamplePixel(data, srcX, srcY, c);
                    result.SetPixel(y, x, c, value);
                }
            }
        }

        var parameters = new Dictionary<string, object>
        {
            ["angle_degrees"] = angleDegrees,
            ["angle_radians"] = angleRadians,
            ["center_x"] = centerX,
            ["center_y"] = centerY,
            ["image_width"] = width,
            ["image_height"] = height
        };

        return (result, parameters);
    }

    /// <summary>
    /// Samples a pixel value at non-integer coordinates using the configured interpolation.
    /// </summary>
    private T SamplePixel(ImageTensor<T> image, double x, double y, int channel)
    {
        int width = image.Width;
        int height = image.Height;

        // Handle out-of-bounds coordinates based on border mode
        if (x < 0 || x >= width || y < 0 || y >= height)
        {
            switch (BorderMode)
            {
                case BorderMode.Constant:
                    return BorderValue;

                case BorderMode.Reflect:
                    x = ReflectCoordinate(x, width);
                    y = ReflectCoordinate(y, height);
                    break;

                case BorderMode.Wrap:
                    x = WrapCoordinate(x, width);
                    y = WrapCoordinate(y, height);
                    break;

                case BorderMode.Clamp:
                default:
                    x = Math.Max(0, Math.Min(width - 1, x));
                    y = Math.Max(0, Math.Min(height - 1, y));
                    break;
            }
        }

        // Apply interpolation
        return Interpolation switch
        {
            InterpolationMode.Nearest => NearestNeighbor(image, x, y, channel),
            InterpolationMode.Bilinear => BilinearInterpolate(image, x, y, channel),
            _ => BilinearInterpolate(image, x, y, channel)
        };
    }

    private static double ReflectCoordinate(double coord, int size)
    {
        if (size <= 1)
        {
            return 0;
        }

        // Use modular arithmetic for deterministic reflection
        // Period is 2 * (size - 1) for reflection pattern
        // Use long arithmetic to prevent overflow for large sizes
        long period = 2L * (size - 1);

        // Handle negative coordinates
        coord = Math.Abs(coord);

        // Reduce to single period
        double reduced = coord % period;

        // If in second half of period, reflect back
        if (reduced >= size)
        {
            reduced = period - reduced;
        }

        return Math.Max(0, Math.Min(size - 1, reduced));
    }

    private static double WrapCoordinate(double coord, int size)
    {
        coord = coord % size;
        if (coord < 0) coord += size;
        return coord;
    }

    private static T NearestNeighbor(ImageTensor<T> image, double x, double y, int channel)
    {
        int ix = (int)Math.Round(x);
        int iy = (int)Math.Round(y);
        ix = Math.Max(0, Math.Min(image.Width - 1, ix));
        iy = Math.Max(0, Math.Min(image.Height - 1, iy));
        return image.GetPixel(iy, ix, channel);
    }

    private static T BilinearInterpolate(ImageTensor<T> image, double x, double y, int channel)
    {
        int x0 = (int)Math.Floor(x);
        int x1 = x0 + 1;
        int y0 = (int)Math.Floor(y);
        int y1 = y0 + 1;

        // Clamp to valid range
        x0 = Math.Max(0, Math.Min(image.Width - 1, x0));
        x1 = Math.Max(0, Math.Min(image.Width - 1, x1));
        y0 = Math.Max(0, Math.Min(image.Height - 1, y0));
        y1 = Math.Max(0, Math.Min(image.Height - 1, y1));

        double xFrac = x - Math.Floor(x);
        double yFrac = y - Math.Floor(y);

        // Get the four surrounding pixels
        double v00 = NumOps.ToDouble(image.GetPixel(y0, x0, channel));
        double v01 = NumOps.ToDouble(image.GetPixel(y0, x1, channel));
        double v10 = NumOps.ToDouble(image.GetPixel(y1, x0, channel));
        double v11 = NumOps.ToDouble(image.GetPixel(y1, x1, channel));

        // Bilinear interpolation
        double v0 = v00 * (1 - xFrac) + v01 * xFrac;
        double v1 = v10 * (1 - xFrac) + v11 * xFrac;
        double result = v0 * (1 - yFrac) + v1 * yFrac;

        return NumOps.FromDouble(result);
    }

    /// <summary>
    /// Transforms a bounding box after rotation.
    /// </summary>
    protected override BoundingBox<T> TransformBoundingBox(
        BoundingBox<T> box,
        IDictionary<string, object> transformParams,
        AugmentationContext<T> context)
    {
        double angleRadians = (double)transformParams["angle_radians"];
        double centerX = (double)transformParams["center_x"];
        double centerY = (double)transformParams["center_y"];

        // Get box coordinates in XYWH format
        var (x, y, w, h) = box.ToXYWH();

        // Get the four corners of the box
        double[][] corners = [
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ];

        // Rotate all corners
        double cos = Math.Cos(angleRadians);
        double sin = Math.Sin(angleRadians);

        double minX = double.MaxValue, minY = double.MaxValue;
        double maxX = double.MinValue, maxY = double.MinValue;

        foreach (var corner in corners)
        {
            double dx = corner[0] - centerX;
            double dy = corner[1] - centerY;
            double newX = cos * dx - sin * dy + centerX;
            double newY = sin * dx + cos * dy + centerY;

            minX = Math.Min(minX, newX);
            minY = Math.Min(minY, newY);
            maxX = Math.Max(maxX, newX);
            maxY = Math.Max(maxY, newY);
        }

        var result = box.Clone();
        result.X1 = NumOps.FromDouble(minX);
        result.Y1 = NumOps.FromDouble(minY);
        result.X2 = NumOps.FromDouble(maxX);
        result.Y2 = NumOps.FromDouble(maxY);
        result.Format = BoundingBoxFormat.XYXY;
        return result;
    }

    /// <summary>
    /// Transforms a keypoint after rotation.
    /// </summary>
    protected override Keypoint<T> TransformKeypoint(
        Keypoint<T> keypoint,
        IDictionary<string, object> transformParams,
        AugmentationContext<T> context)
    {
        double angleRadians = (double)transformParams["angle_radians"];
        double centerX = (double)transformParams["center_x"];
        double centerY = (double)transformParams["center_y"];

        double x = NumOps.ToDouble(keypoint.X);
        double y = NumOps.ToDouble(keypoint.Y);

        // Rotate around center
        double dx = x - centerX;
        double dy = y - centerY;
        double cos = Math.Cos(angleRadians);
        double sin = Math.Sin(angleRadians);

        double newX = cos * dx - sin * dy + centerX;
        double newY = sin * dx + cos * dy + centerY;

        var result = keypoint.Clone();
        result.X = NumOps.FromDouble(newX);
        result.Y = NumOps.FromDouble(newY);
        return result;
    }

    /// <summary>
    /// Transforms a segmentation mask after rotation.
    /// </summary>
    /// <remarks>
    /// Uses the same rotation center as the image transformation, scaled to mask dimensions
    /// if they differ from image dimensions, to ensure proper alignment.
    /// </remarks>
    protected override SegmentationMask<T> TransformMask(
        SegmentationMask<T> mask,
        IDictionary<string, object> transformParams,
        AugmentationContext<T> context)
    {
        double angleRadians = (double)transformParams["angle_radians"];
        int imageWidth = (int)transformParams["image_width"];
        int imageHeight = (int)transformParams["image_height"];
        double imageCenterX = (double)transformParams["center_x"];
        double imageCenterY = (double)transformParams["center_y"];

        var result = mask.Clone();
        int maskHeight = mask.Height;
        int maskWidth = mask.Width;

        // Scale the center to mask dimensions if they differ from image dimensions
        // This ensures the rotation pivot point is consistent between image and mask
        double centerX = imageCenterX * maskWidth / imageWidth;
        double centerY = imageCenterY * maskHeight / imageHeight;

        double cos = Math.Cos(angleRadians);
        double sin = Math.Sin(angleRadians);

        // Get the dense mask data
        var data = mask.ToDense();
        var rotated = new T[maskHeight, maskWidth];

        // Rotate the mask using nearest neighbor (to preserve class labels)
        for (int y = 0; y < maskHeight; y++)
        {
            for (int x = 0; x < maskWidth; x++)
            {
                double dx = x - centerX;
                double dy = y - centerY;

                // Inverse rotation to find source pixel
                double srcX = cos * dx + sin * dy + centerX;
                double srcY = -sin * dx + cos * dy + centerY;

                int srcXi = (int)Math.Round(srcX);
                int srcYi = (int)Math.Round(srcY);

                if (srcXi >= 0 && srcXi < maskWidth && srcYi >= 0 && srcYi < maskHeight)
                {
                    rotated[y, x] = data[srcYi, srcXi];
                }
                // else: default value (0) for out of bounds
            }
        }

        result.MaskData = rotated;
        result.Encoding = MaskEncoding.Dense;
        return result;
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var parameters = base.GetParameters();
        parameters["min_angle"] = MinAngle;
        parameters["max_angle"] = MaxAngle;
        parameters["border_mode"] = BorderMode.ToString();
        parameters["interpolation"] = Interpolation.ToString();
        return parameters;
    }
}

/// <summary>
/// Specifies how to handle pixels that fall outside the image bounds during transformation.
/// </summary>
public enum BorderMode
{
    /// <summary>
    /// Fill with a constant value.
    /// </summary>
    Constant,

    /// <summary>
    /// Reflect pixels at the edge (mirror).
    /// </summary>
    Reflect,

    /// <summary>
    /// Wrap around to the opposite edge.
    /// </summary>
    Wrap,

    /// <summary>
    /// Clamp to the nearest edge pixel.
    /// </summary>
    Clamp
}
