using AiDotNet.Augmentation.Base;
using AiDotNet.Augmentation.Data;

namespace AiDotNet.Augmentation.Image.Geometric;

/// <summary>
/// Scales an image by a random factor within a specified range.
/// </summary>
/// <remarks>
/// <para>
/// Scale randomly resizes the image by a factor sampled from the specified range.
/// This simulates viewing objects from different distances, helping the model
/// become robust to scale variations.
/// </para>
/// <para><b>For Beginners:</b> Think of this like zooming in or out on your camera.
/// The same object photographed from closer or farther appears at different sizes.
/// This augmentation teaches your model to recognize objects regardless of their size.
/// </para>
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>Object detection where objects appear at various sizes</item>
/// <item>Image classification with variable-sized subjects</item>
/// <item>When training data lacks size diversity</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class Scale<T> : SpatialAugmentationBase<T, ImageTensor<T>>
{
    /// <summary>
    /// Gets the minimum scale factor.
    /// </summary>
    public double MinScale { get; }

    /// <summary>
    /// Gets the maximum scale factor.
    /// </summary>
    public double MaxScale { get; }

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
    /// Creates a new scale augmentation.
    /// </summary>
    /// <param name="minScale">
    /// The minimum scale factor. Values less than 1.0 zoom out, greater than 1.0 zoom in.
    /// Industry standard default is 0.8 (80% of original size).
    /// </param>
    /// <param name="maxScale">
    /// The maximum scale factor.
    /// Industry standard default is 1.2 (120% of original size).
    /// </param>
    /// <param name="probability">
    /// The probability of applying this augmentation (0.0 to 1.0).
    /// Industry standard default is 0.5.
    /// </param>
    /// <param name="interpolation">
    /// The interpolation mode for sampling pixels.
    /// Industry standard default is Bilinear for smooth results.
    /// </param>
    /// <param name="borderMode">
    /// How to fill pixels that fall outside the original image bounds.
    /// Industry standard default is Reflect.
    /// </param>
    /// <param name="borderValue">
    /// The constant value to use when borderMode is Constant.
    /// </param>
    public Scale(
        double minScale = 0.8,
        double maxScale = 1.2,
        double probability = 0.5,
        InterpolationMode interpolation = InterpolationMode.Bilinear,
        BorderMode borderMode = BorderMode.Reflect,
        T? borderValue = default)
        : base(probability)
    {
        if (minScale <= 0)
            throw new ArgumentOutOfRangeException(nameof(minScale), "Scale must be positive");
        if (maxScale <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxScale), "Scale must be positive");
        if (minScale > maxScale)
            throw new ArgumentException("minScale must be less than or equal to maxScale", nameof(minScale));

        MinScale = minScale;
        MaxScale = maxScale;
        Interpolation = interpolation;
        BorderMode = borderMode;
        BorderValue = borderValue ?? default!;
    }

    /// <summary>
    /// Applies the scale transformation and returns transform parameters.
    /// </summary>
    protected override (ImageTensor<T> data, IDictionary<string, object> parameters) ApplyWithTransformParams(
        ImageTensor<T> data,
        AugmentationContext<T> context)
    {
        // Sample a random scale factor
        double scaleFactor = context.GetRandomDouble(MinScale, MaxScale);

        var result = data.Clone();
        int height = data.Height;
        int width = data.Width;
        int channels = data.Channels;

        // Center of the image
        double centerX = width / 2.0;
        double centerY = height / 2.0;

        // For each pixel in the output, find the corresponding input pixel
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                // Inverse scaling to find source pixel
                double srcX = (x - centerX) / scaleFactor + centerX;
                double srcY = (y - centerY) / scaleFactor + centerY;

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
            ["scale_factor"] = scaleFactor,
            ["center_x"] = centerX,
            ["center_y"] = centerY,
            ["image_width"] = width,
            ["image_height"] = height
        };

        return (result, parameters);
    }

    /// <summary>
    /// Samples a pixel value at non-integer coordinates.
    /// </summary>
    private T SamplePixel(ImageTensor<T> image, double x, double y, int channel)
    {
        int width = image.Width;
        int height = image.Height;

        // Handle out-of-bounds
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
        while (coord < 0 || coord >= size)
        {
            if (coord < 0)
                coord = -coord;
            if (coord >= size)
                coord = 2 * (size - 1) - coord;
        }
        return Math.Max(0, Math.Min(size - 1, coord));
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

        double v00 = Convert.ToDouble(image.GetPixel(y0, x0, channel));
        double v01 = Convert.ToDouble(image.GetPixel(y0, x1, channel));
        double v10 = Convert.ToDouble(image.GetPixel(y1, x0, channel));
        double v11 = Convert.ToDouble(image.GetPixel(y1, x1, channel));

        double v0 = v00 * (1 - xFrac) + v01 * xFrac;
        double v1 = v10 * (1 - xFrac) + v11 * xFrac;
        double result = v0 * (1 - yFrac) + v1 * yFrac;

        return (T)Convert.ChangeType(result, typeof(T));
    }

    /// <summary>
    /// Transforms a bounding box after scaling.
    /// </summary>
    protected override BoundingBox<T> TransformBoundingBox(
        BoundingBox<T> box,
        IDictionary<string, object> transformParams,
        AugmentationContext<T> context)
    {
        double scaleFactor = (double)transformParams["scale_factor"];
        double centerX = (double)transformParams["center_x"];
        double centerY = (double)transformParams["center_y"];

        var (x, y, w, h) = box.ToXYWH();

        // Scale coordinates around center
        double newX = (x - centerX) * scaleFactor + centerX;
        double newY = (y - centerY) * scaleFactor + centerY;
        double newW = w * scaleFactor;
        double newH = h * scaleFactor;

        var result = box.Clone();
        result.X1 = (T)Convert.ChangeType(newX, typeof(T));
        result.Y1 = (T)Convert.ChangeType(newY, typeof(T));
        result.X2 = (T)Convert.ChangeType(newX + newW, typeof(T));
        result.Y2 = (T)Convert.ChangeType(newY + newH, typeof(T));
        result.Format = BoundingBoxFormat.XYXY;
        return result;
    }

    /// <summary>
    /// Transforms a keypoint after scaling.
    /// </summary>
    protected override Keypoint<T> TransformKeypoint(
        Keypoint<T> keypoint,
        IDictionary<string, object> transformParams,
        AugmentationContext<T> context)
    {
        double scaleFactor = (double)transformParams["scale_factor"];
        double centerX = (double)transformParams["center_x"];
        double centerY = (double)transformParams["center_y"];

        double x = Convert.ToDouble(keypoint.X);
        double y = Convert.ToDouble(keypoint.Y);

        double newX = (x - centerX) * scaleFactor + centerX;
        double newY = (y - centerY) * scaleFactor + centerY;

        var result = keypoint.Clone();
        result.X = (T)Convert.ChangeType(newX, typeof(T));
        result.Y = (T)Convert.ChangeType(newY, typeof(T));
        return result;
    }

    /// <summary>
    /// Transforms a segmentation mask after scaling.
    /// </summary>
    protected override SegmentationMask<T> TransformMask(
        SegmentationMask<T> mask,
        IDictionary<string, object> transformParams,
        AugmentationContext<T> context)
    {
        double scaleFactor = (double)transformParams["scale_factor"];

        var result = mask.Clone();
        int height = mask.Height;
        int width = mask.Width;

        double centerX = width / 2.0;
        double centerY = height / 2.0;

        var data = mask.ToDense();
        var scaled = new T[height, width];

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                double srcX = (x - centerX) / scaleFactor + centerX;
                double srcY = (y - centerY) / scaleFactor + centerY;

                int srcXi = (int)Math.Round(srcX);
                int srcYi = (int)Math.Round(srcY);

                if (srcXi >= 0 && srcXi < width && srcYi >= 0 && srcYi < height)
                {
                    scaled[y, x] = data[srcYi, srcXi];
                }
            }
        }

        result.MaskData = scaled;
        result.Encoding = MaskEncoding.Dense;
        return result;
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var parameters = base.GetParameters();
        parameters["min_scale"] = MinScale;
        parameters["max_scale"] = MaxScale;
        parameters["interpolation"] = Interpolation.ToString();
        parameters["border_mode"] = BorderMode.ToString();
        return parameters;
    }
}
