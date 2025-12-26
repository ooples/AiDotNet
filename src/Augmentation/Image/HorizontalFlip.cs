
namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Flips an image horizontally (left-right mirror).
/// </summary>
/// <remarks>
/// <para>
/// Horizontal flipping mirrors the image along its vertical center axis, swapping the left
/// and right sides. This is one of the most commonly used augmentations because many objects
/// look the same when flipped horizontally (e.g., cars, animals, faces in frontal view).
/// </para>
/// <para><b>For Beginners:</b> Think of this like looking in a mirror. The left side of the
/// image becomes the right side, and vice versa. This is useful when training image classifiers
/// because a cat is still a cat whether it's facing left or right.
/// </para>
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>Image classification where horizontal orientation doesn't matter</item>
/// <item>Object detection (boxes will be flipped automatically)</item>
/// <item>Pose estimation (keypoints will be swapped correctly)</item>
/// </list>
/// </para>
/// <para><b>When NOT to use:</b>
/// <list type="bullet">
/// <item>Text recognition (text would become unreadable when mirrored)</item>
/// <item>Reading direction matters (left-to-right vs right-to-left)</item>
/// <item>Asymmetric objects where orientation is meaningful (e.g., traffic signs with arrows)</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class HorizontalFlip<T> : SpatialAugmentationBase<T, ImageTensor<T>>
{
    /// <summary>
    /// Creates a new horizontal flip augmentation.
    /// </summary>
    /// <param name="probability">
    /// The probability of applying this augmentation (0.0 to 1.0).
    /// Industry standard default is 0.5 (50% chance of flipping each image).
    /// </param>
    /// <remarks>
    /// <para><b>For Beginners:</b> A probability of 0.5 means each image has a 50-50 chance
    /// of being flipped. This creates good variety in your training data.
    /// </para>
    /// </remarks>
    public HorizontalFlip(double probability = 0.5) : base(probability)
    {
    }

    /// <summary>
    /// Applies the horizontal flip transformation and returns transform parameters.
    /// </summary>
    protected override (ImageTensor<T> data, IDictionary<string, object> parameters) ApplyWithTransformParams(
        ImageTensor<T> data,
        AugmentationContext<T> context)
    {
        var result = data.Clone();
        int height = result.Height;
        int width = result.Width;
        int channels = result.Channels;

        // Flip horizontally by swapping pixels left-to-right
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width / 2; x++)
            {
                int mirrorX = width - 1 - x;
                for (int c = 0; c < channels; c++)
                {
                    // Swap pixel at (y, x) with pixel at (y, mirrorX)
                    T temp = result.GetPixel(y, x, c);
                    result.SetPixel(y, x, c, result.GetPixel(y, mirrorX, c));
                    result.SetPixel(y, mirrorX, c, temp);
                }
            }
        }

        var parameters = new Dictionary<string, object>
        {
            ["flip_type"] = "horizontal",
            ["image_width"] = width,
            ["image_height"] = height
        };

        return (result, parameters);
    }

    /// <summary>
    /// Transforms a bounding box after horizontal flip.
    /// </summary>
    protected override BoundingBox<T> TransformBoundingBox(
        BoundingBox<T> box,
        IDictionary<string, object> transformParams,
        AugmentationContext<T> context)
    {
        int imageWidth = (int)transformParams["image_width"];

        // Get box coordinates in XYWH format
        var (x, y, width, height) = box.ToXYWH();

        // For horizontal flip: new_x = image_width - old_x - box_width
        double newX = imageWidth - x - width;

        var result = box.Clone();
        result.X1 = NumOps.FromDouble(newX);
        result.Y1 = NumOps.FromDouble(y);
        result.X2 = NumOps.FromDouble(newX + width);
        result.Y2 = NumOps.FromDouble(y + height);
        result.Format = BoundingBoxFormat.XYXY;
        return result;
    }

    /// <summary>
    /// Transforms a keypoint after horizontal flip.
    /// </summary>
    /// <remarks>
    /// Handles both normalized coordinates (0.0-1.0) and absolute pixel coordinates.
    /// </remarks>
    protected override Keypoint<T> TransformKeypoint(
        Keypoint<T> keypoint,
        IDictionary<string, object> transformParams,
        AugmentationContext<T> context)
    {
        double x = NumOps.ToDouble(keypoint.X);
        double newX;

        if (keypoint.IsNormalized)
        {
            // For normalized coordinates: new_x = 1.0 - old_x
            newX = 1.0 - x;
        }
        else
        {
            // For pixel coordinates: new_x = image_width - 1 - old_x (0-indexed)
            int imageWidth = (int)transformParams["image_width"];
            newX = imageWidth - 1 - x;
        }

        var result = keypoint.Clone();
        result.X = NumOps.FromDouble(newX);
        return result;
    }

    /// <summary>
    /// Transforms a segmentation mask after horizontal flip.
    /// </summary>
    protected override SegmentationMask<T> TransformMask(
        SegmentationMask<T> mask,
        IDictionary<string, object> transformParams,
        AugmentationContext<T> context)
    {
        var result = mask.Clone();
        int width = result.Width;
        int height = result.Height;

        // Get the dense mask data and flip it horizontally
        var data = result.ToDense();
        var flipped = new T[height, width];

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int mirrorX = width - 1 - x;
                flipped[y, x] = data[y, mirrorX];
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
        parameters["flip_type"] = "horizontal";
        return parameters;
    }
}
