
namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Applies random affine transformations (rotation, scale, shear, translation) to an image.
/// </summary>
/// <remarks>
/// <para>
/// Affine transformation is a general geometric transformation that preserves lines and
/// parallelism. It combines rotation, scaling, shearing, and translation in a single operation,
/// providing a powerful way to augment images with realistic geometric variations.
/// </para>
/// <para><b>For Beginners:</b> Think of this as combining multiple geometric operations into one.
/// The image can be rotated, stretched, tilted (sheared), and moved around - all at once.
/// This creates more diverse training examples than applying each transformation separately.
/// </para>
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>General image classification with geometric invariance</item>
/// <item>Object detection where objects may be viewed from different angles</item>
/// <item>When you need combined geometric variations efficiently</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class Affine<T> : SpatialAugmentationBase<T, ImageTensor<T>>
{
    /// <summary>
    /// Gets the rotation angle range in degrees.
    /// </summary>
    public (double Min, double Max) RotationRange { get; }

    /// <summary>
    /// Gets the scale factor range.
    /// </summary>
    public (double Min, double Max) ScaleRange { get; }

    /// <summary>
    /// Gets the shear angle range in degrees.
    /// </summary>
    public (double Min, double Max) ShearRange { get; }

    /// <summary>
    /// Gets the translation range as a fraction of image dimensions.
    /// </summary>
    public (double X, double Y) TranslationRange { get; }

    /// <summary>
    /// Gets the interpolation mode for pixel sampling.
    /// </summary>
    public InterpolationMode Interpolation { get; }

    /// <summary>
    /// Gets the border mode when pixels fall outside the original image bounds.
    /// </summary>
    public BorderMode BorderMode { get; }

    /// <summary>
    /// Gets the constant value used when BorderMode is Constant.
    /// </summary>
    public T BorderValue { get; }

    /// <summary>
    /// Creates a new affine transformation augmentation.
    /// </summary>
    /// <param name="rotationRange">
    /// The range of rotation angles in degrees (min, max).
    /// Industry standard default is (-15, 15).
    /// </param>
    /// <param name="scaleRange">
    /// The range of scale factors (min, max).
    /// Industry standard default is (0.9, 1.1).
    /// </param>
    /// <param name="shearRange">
    /// The range of shear angles in degrees (min, max).
    /// Industry standard default is (-10, 10).
    /// </param>
    /// <param name="translationRange">
    /// The range of translation as fraction of image dimensions (x, y).
    /// Industry standard default is (0.1, 0.1) meaning up to 10% in each direction.
    /// </param>
    /// <param name="probability">
    /// The probability of applying this augmentation (0.0 to 1.0).
    /// Industry standard default is 0.5.
    /// </param>
    /// <param name="interpolation">
    /// The interpolation mode for sampling pixels.
    /// Industry standard default is Bilinear.
    /// </param>
    /// <param name="borderMode">
    /// How to fill pixels that fall outside the original image bounds.
    /// Industry standard default is Reflect.
    /// </param>
    /// <param name="borderValue">
    /// The constant value to use when borderMode is Constant.
    /// </param>
    public Affine(
        (double Min, double Max)? rotationRange = null,
        (double Min, double Max)? scaleRange = null,
        (double Min, double Max)? shearRange = null,
        (double X, double Y)? translationRange = null,
        double probability = 0.5,
        InterpolationMode interpolation = InterpolationMode.Bilinear,
        BorderMode borderMode = BorderMode.Reflect,
        T? borderValue = default)
        : base(probability)
    {
        RotationRange = rotationRange ?? (-15.0, 15.0);
        ScaleRange = scaleRange ?? (0.9, 1.1);
        ShearRange = shearRange ?? (-10.0, 10.0);
        TranslationRange = translationRange ?? (0.1, 0.1);
        Interpolation = interpolation;
        BorderMode = borderMode;
        BorderValue = borderValue ?? default!;

        ValidateRanges();
    }

    private void ValidateRanges()
    {
        if (RotationRange.Min > RotationRange.Max)
            throw new ArgumentException("Rotation min must be <= max");
        if (ScaleRange.Min <= 0 || ScaleRange.Max <= 0)
            throw new ArgumentException("Scale values must be positive");
        if (ScaleRange.Min > ScaleRange.Max)
            throw new ArgumentException("Scale min must be <= max");
        if (ShearRange.Min > ShearRange.Max)
            throw new ArgumentException("Shear min must be <= max");
        if (TranslationRange.X < 0 || TranslationRange.Y < 0)
            throw new ArgumentException("Translation ranges must be non-negative");
    }

    /// <summary>
    /// Applies the affine transformation and returns transform parameters.
    /// </summary>
    protected override (ImageTensor<T> data, IDictionary<string, object> parameters) ApplyWithTransformParams(
        ImageTensor<T> data,
        AugmentationContext<T> context)
    {
        int height = data.Height;
        int width = data.Width;
        int channels = data.Channels;

        // Sample random parameters
        double angle = context.GetRandomDouble(RotationRange.Min, RotationRange.Max);
        double scale = context.GetRandomDouble(ScaleRange.Min, ScaleRange.Max);
        double shear = context.GetRandomDouble(ShearRange.Min, ShearRange.Max);
        double translateX = context.GetRandomDouble(-TranslationRange.X, TranslationRange.X) * width;
        double translateY = context.GetRandomDouble(-TranslationRange.Y, TranslationRange.Y) * height;

        // Convert to radians
        double angleRad = angle * Math.PI / 180.0;
        double shearRad = shear * Math.PI / 180.0;

        // Center of the image
        double centerX = width / 2.0;
        double centerY = height / 2.0;

        // Build affine transformation matrix components
        double cos = Math.Cos(angleRad);
        double sin = Math.Sin(angleRad);
        double shearTan = Math.Tan(shearRad);

        // Combined transformation matrix: T * R * Sh * S (applied right to left)
        // Rotation matrix
        double r00 = cos;
        double r01 = -sin;
        double r10 = sin;
        double r11 = cos;

        // Shear matrix (horizontal shear)
        double sh00 = 1;
        double sh01 = shearTan;
        double sh10 = 0;
        double sh11 = 1;

        // Combined: R * Sh * Scale
        double m00 = (r00 * sh00 + r01 * sh10) * scale;
        double m01 = (r00 * sh01 + r01 * sh11) * scale;
        double m10 = (r10 * sh00 + r11 * sh10) * scale;
        double m11 = (r10 * sh01 + r11 * sh11) * scale;

        // Inverse matrix for sampling
        double det = m00 * m11 - m01 * m10;
        if (Math.Abs(det) < 1e-10)
        {
            // Degenerate case, return original
            return (data.Clone(), new Dictionary<string, object>());
        }

        double invM00 = m11 / det;
        double invM01 = -m01 / det;
        double invM10 = -m10 / det;
        double invM11 = m00 / det;

        var result = data.Clone();

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                // Transform to center coordinates, apply inverse, transform back
                double dx = x - centerX - translateX;
                double dy = y - centerY - translateY;

                double srcX = invM00 * dx + invM01 * dy + centerX;
                double srcY = invM10 * dx + invM11 * dy + centerY;

                for (int c = 0; c < channels; c++)
                {
                    T value = SamplePixel(data, srcX, srcY, c);
                    result.SetPixel(y, x, c, value);
                }
            }
        }

        var parameters = new Dictionary<string, object>
        {
            ["angle_degrees"] = angle,
            ["scale_factor"] = scale,
            ["shear_degrees"] = shear,
            ["translate_x"] = translateX,
            ["translate_y"] = translateY,
            ["center_x"] = centerX,
            ["center_y"] = centerY,
            ["m00"] = m00,
            ["m01"] = m01,
            ["m10"] = m10,
            ["m11"] = m11,
            ["image_width"] = width,
            ["image_height"] = height
        };

        return (result, parameters);
    }

    private T SamplePixel(ImageTensor<T> image, double x, double y, int channel)
    {
        int width = image.Width;
        int height = image.Height;

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

        x0 = Math.Max(0, Math.Min(image.Width - 1, x0));
        x1 = Math.Max(0, Math.Min(image.Width - 1, x1));
        y0 = Math.Max(0, Math.Min(image.Height - 1, y0));
        y1 = Math.Max(0, Math.Min(image.Height - 1, y1));

        double xFrac = x - Math.Floor(x);
        double yFrac = y - Math.Floor(y);

        double v00 = NumOps.ToDouble(image.GetPixel(y0, x0, channel));
        double v01 = NumOps.ToDouble(image.GetPixel(y0, x1, channel));
        double v10 = NumOps.ToDouble(image.GetPixel(y1, x0, channel));
        double v11 = NumOps.ToDouble(image.GetPixel(y1, x1, channel));

        double v0 = v00 * (1 - xFrac) + v01 * xFrac;
        double v1 = v10 * (1 - xFrac) + v11 * xFrac;
        double result = v0 * (1 - yFrac) + v1 * yFrac;

        return NumOps.FromDouble(result);
    }

    /// <summary>
    /// Transforms a bounding box after affine transformation.
    /// </summary>
    protected override BoundingBox<T> TransformBoundingBox(
        BoundingBox<T> box,
        IDictionary<string, object> transformParams,
        AugmentationContext<T> context)
    {
        double m00 = (double)transformParams["m00"];
        double m01 = (double)transformParams["m01"];
        double m10 = (double)transformParams["m10"];
        double m11 = (double)transformParams["m11"];
        double translateX = (double)transformParams["translate_x"];
        double translateY = (double)transformParams["translate_y"];
        double centerX = (double)transformParams["center_x"];
        double centerY = (double)transformParams["center_y"];

        var (x, y, w, h) = box.ToXYWH();

        // Get four corners
        double[][] corners =
        [
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ];

        double minX = double.MaxValue, minY = double.MaxValue;
        double maxX = double.MinValue, maxY = double.MinValue;

        foreach (var corner in corners)
        {
            double dx = corner[0] - centerX;
            double dy = corner[1] - centerY;
            double newX = m00 * dx + m01 * dy + centerX + translateX;
            double newY = m10 * dx + m11 * dy + centerY + translateY;

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
    /// Transforms a keypoint after affine transformation.
    /// </summary>
    protected override Keypoint<T> TransformKeypoint(
        Keypoint<T> keypoint,
        IDictionary<string, object> transformParams,
        AugmentationContext<T> context)
    {
        double m00 = (double)transformParams["m00"];
        double m01 = (double)transformParams["m01"];
        double m10 = (double)transformParams["m10"];
        double m11 = (double)transformParams["m11"];
        double translateX = (double)transformParams["translate_x"];
        double translateY = (double)transformParams["translate_y"];
        double centerX = (double)transformParams["center_x"];
        double centerY = (double)transformParams["center_y"];

        double x = NumOps.ToDouble(keypoint.X);
        double y = NumOps.ToDouble(keypoint.Y);

        double dx = x - centerX;
        double dy = y - centerY;
        double newX = m00 * dx + m01 * dy + centerX + translateX;
        double newY = m10 * dx + m11 * dy + centerY + translateY;

        var result = keypoint.Clone();
        result.X = NumOps.FromDouble(newX);
        result.Y = NumOps.FromDouble(newY);
        return result;
    }

    /// <summary>
    /// Transforms a segmentation mask after affine transformation.
    /// </summary>
    protected override SegmentationMask<T> TransformMask(
        SegmentationMask<T> mask,
        IDictionary<string, object> transformParams,
        AugmentationContext<T> context)
    {
        double m00 = (double)transformParams["m00"];
        double m01 = (double)transformParams["m01"];
        double m10 = (double)transformParams["m10"];
        double m11 = (double)transformParams["m11"];
        double translateX = (double)transformParams["translate_x"];
        double translateY = (double)transformParams["translate_y"];
        double centerX = (double)transformParams["center_x"];
        double centerY = (double)transformParams["center_y"];

        // Compute inverse for sampling
        double det = m00 * m11 - m01 * m10;
        double invM00 = m11 / det;
        double invM01 = -m01 / det;
        double invM10 = -m10 / det;
        double invM11 = m00 / det;

        var result = mask.Clone();
        int height = mask.Height;
        int width = mask.Width;

        var data = mask.ToDense();
        var transformed = new T[height, width];

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                double dx = x - centerX - translateX;
                double dy = y - centerY - translateY;

                double srcX = invM00 * dx + invM01 * dy + centerX;
                double srcY = invM10 * dx + invM11 * dy + centerY;

                int srcXi = (int)Math.Round(srcX);
                int srcYi = (int)Math.Round(srcY);

                if (srcXi >= 0 && srcXi < width && srcYi >= 0 && srcYi < height)
                {
                    transformed[y, x] = data[srcYi, srcXi];
                }
            }
        }

        result.MaskData = transformed;
        result.Encoding = MaskEncoding.Dense;
        return result;
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var parameters = base.GetParameters();
        parameters["rotation_range"] = $"({RotationRange.Min}, {RotationRange.Max})";
        parameters["scale_range"] = $"({ScaleRange.Min}, {ScaleRange.Max})";
        parameters["shear_range"] = $"({ShearRange.Min}, {ShearRange.Max})";
        parameters["translation_range"] = $"({TranslationRange.X}, {TranslationRange.Y})";
        parameters["interpolation"] = Interpolation.ToString();
        parameters["border_mode"] = BorderMode.ToString();
        return parameters;
    }
}
