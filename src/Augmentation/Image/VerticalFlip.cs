
namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Flips an image vertically (top-bottom mirror).
/// </summary>
/// <remarks>
/// <para>
/// Vertical flipping mirrors the image along its horizontal center axis, swapping the top
/// and bottom portions. This is less commonly used than horizontal flipping because
/// most real-world objects have a consistent "up" orientation (gravity matters!).
/// </para>
/// <para><b>For Beginners:</b> Think of this like flipping a photo upside down. The top of
/// the image becomes the bottom, and vice versa. Use this carefully because many objects
/// look unnatural when flipped vertically (imagine an upside-down car or person).
/// </para>
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>Satellite or aerial imagery where "up" is arbitrary</item>
/// <item>Microscopy images with no inherent orientation</item>
/// <item>Abstract pattern recognition</item>
/// <item>Medical imaging where orientation varies by acquisition</item>
/// </list>
/// </para>
/// <para><b>When NOT to use:</b>
/// <list type="bullet">
/// <item>Natural photography with gravity-oriented subjects</item>
/// <item>Facial recognition or human pose estimation</item>
/// <item>Vehicle recognition (cars don't drive upside down)</item>
/// <item>Any task where vertical orientation is meaningful</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class VerticalFlip<T> : SpatialAugmentationBase<T, ImageTensor<T>>
{
    /// <summary>
    /// Creates a new vertical flip augmentation.
    /// </summary>
    /// <param name="probability">
    /// The probability of applying this augmentation (0.0 to 1.0).
    /// Industry standard default is 0.5 (50% chance of flipping each image).
    /// </param>
    /// <remarks>
    /// <para><b>For Beginners:</b> A probability of 0.5 means each image has a 50-50 chance
    /// of being flipped. Consider using a lower probability (like 0.25) if vertical flips
    /// might create unrealistic-looking images for your use case.
    /// </para>
    /// </remarks>
    public VerticalFlip(double probability = 0.5) : base(probability)
    {
    }

    /// <summary>
    /// Applies the vertical flip transformation and returns transform parameters.
    /// </summary>
    protected override (ImageTensor<T> data, IDictionary<string, object> parameters) ApplyWithTransformParams(
        ImageTensor<T> data,
        AugmentationContext<T> context)
    {
        var result = data.Clone();
        int height = result.Height;
        int width = result.Width;
        int channels = result.Channels;

        // Flip vertically by swapping pixels top-to-bottom
        for (int y = 0; y < height / 2; y++)
        {
            int mirrorY = height - 1 - y;
            for (int x = 0; x < width; x++)
            {
                for (int c = 0; c < channels; c++)
                {
                    // Swap pixel at (y, x) with pixel at (mirrorY, x)
                    T temp = result.GetPixel(y, x, c);
                    result.SetPixel(y, x, c, result.GetPixel(mirrorY, x, c));
                    result.SetPixel(mirrorY, x, c, temp);
                }
            }
        }

        var parameters = new Dictionary<string, object>
        {
            ["flip_type"] = "vertical",
            ["image_width"] = width,
            ["image_height"] = height
        };

        return (result, parameters);
    }

    /// <summary>
    /// Transforms a bounding box after vertical flip.
    /// </summary>
    protected override BoundingBox<T> TransformBoundingBox(
        BoundingBox<T> box,
        IDictionary<string, object> transformParams,
        AugmentationContext<T> context)
    {
        int imageHeight = (int)transformParams["image_height"];

        // Get box coordinates in XYWH format
        var (x, y, width, height) = box.ToXYWH();

        // For vertical flip: new_y = image_height - old_y - box_height
        double newY = imageHeight - y - height;

        var result = box.Clone();
        result.X1 = (T)Convert.ChangeType(x, typeof(T));
        result.Y1 = (T)Convert.ChangeType(newY, typeof(T));
        result.X2 = (T)Convert.ChangeType(x + width, typeof(T));
        result.Y2 = (T)Convert.ChangeType(newY + height, typeof(T));
        result.Format = BoundingBoxFormat.XYXY;
        return result;
    }

    /// <summary>
    /// Transforms a keypoint after vertical flip.
    /// </summary>
    protected override Keypoint<T> TransformKeypoint(
        Keypoint<T> keypoint,
        IDictionary<string, object> transformParams,
        AugmentationContext<T> context)
    {
        int imageHeight = (int)transformParams["image_height"];

        // For vertical flip: new_y = image_height - 1 - old_y (0-indexed)
        double y = Convert.ToDouble(keypoint.Y);
        double newY = imageHeight - 1 - y;

        var result = keypoint.Clone();
        result.Y = (T)Convert.ChangeType(newY, typeof(T));
        return result;
    }

    /// <summary>
    /// Transforms a segmentation mask after vertical flip.
    /// </summary>
    protected override SegmentationMask<T> TransformMask(
        SegmentationMask<T> mask,
        IDictionary<string, object> transformParams,
        AugmentationContext<T> context)
    {
        var result = mask.Clone();
        int width = result.Width;
        int height = result.Height;

        // Get the dense mask data and flip it vertically
        var data = result.ToDense();
        var flipped = new T[height, width];

        for (int y = 0; y < height; y++)
        {
            int mirrorY = height - 1 - y;
            for (int x = 0; x < width; x++)
            {
                flipped[y, x] = data[mirrorY, x];
            }
        }

        result.MaskData = flipped;
        result.Encoding = MaskEncoding.Dense;
        return result;
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var parameters = base.GetParameters();
        parameters["flip_type"] = "vertical";
        return parameters;
    }
}
