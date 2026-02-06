namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Crops the center region of an image to a specified size.
/// </summary>
/// <remarks>
/// <para>
/// Center cropping extracts a fixed-size region from the center of the image. This is commonly
/// used during evaluation/inference to ensure consistent framing, and as part of the standard
/// ImageNet preprocessing pipeline (resize to 256, then center crop to 224).
/// </para>
/// <para><b>For Beginners:</b> Think of this like taking a photo and cutting out the middle
/// rectangle. The center of the image usually contains the most important content, so this
/// is a reliable way to get a fixed-size input for your model during testing.</para>
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>Evaluation/inference preprocessing (standard for ImageNet models)</item>
/// <item>When you need deterministic cropping (no randomness)</item>
/// <item>As part of a resize-then-crop pipeline</item>
/// </list>
/// </para>
/// <para><b>When NOT to use:</b>
/// <list type="bullet">
/// <item>Training (use RandomCrop instead for data diversity)</item>
/// <item>When important content is near the edges</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class CenterCrop<T> : SpatialImageAugmenterBase<T>
{
    /// <summary>
    /// Gets the output height after cropping.
    /// </summary>
    public int CropHeight { get; }

    /// <summary>
    /// Gets the output width after cropping.
    /// </summary>
    public int CropWidth { get; }

    /// <summary>
    /// Creates a new center crop augmentation.
    /// </summary>
    /// <param name="cropHeight">The output height. Must be positive.</param>
    /// <param name="cropWidth">The output width. Must be positive.</param>
    /// <param name="probability">
    /// The probability of applying this augmentation (0.0 to 1.0).
    /// Default is 1.0 (always apply) since center crop is typically deterministic.
    /// </param>
    public CenterCrop(int cropHeight, int cropWidth, double probability = 1.0)
        : base(probability)
    {
        if (cropHeight <= 0)
            throw new ArgumentOutOfRangeException(nameof(cropHeight), "Crop height must be positive.");
        if (cropWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(cropWidth), "Crop width must be positive.");

        CropHeight = cropHeight;
        CropWidth = cropWidth;
    }

    /// <summary>
    /// Creates a new center crop augmentation with square output.
    /// </summary>
    /// <param name="size">The output size (both height and width). Must be positive.</param>
    /// <param name="probability">The probability of applying this augmentation.</param>
    public CenterCrop(int size, double probability = 1.0)
        : this(size, size, probability)
    {
    }

    /// <summary>
    /// Applies the center crop and returns transformation parameters.
    /// </summary>
    protected override (ImageTensor<T> data, IDictionary<string, object> parameters) ApplyWithTransformParams(
        ImageTensor<T> data, AugmentationContext<T> context)
    {
        int height = data.Height;
        int width = data.Width;

        int cropW = Math.Min(CropWidth, width);
        int cropH = Math.Min(CropHeight, height);
        int cropX = (width - cropW) / 2;
        int cropY = (height - cropH) / 2;

        var result = new ImageTensor<T>(CropHeight, CropWidth, data.Channels, data.ChannelOrder, data.ColorSpace)
        {
            IsNormalized = data.IsNormalized,
            NormalizationMean = data.NormalizationMean,
            NormalizationStd = data.NormalizationStd,
            OriginalRange = data.OriginalRange
        };

        // Copy the center region
        for (int y = 0; y < cropH; y++)
        {
            for (int x = 0; x < cropW; x++)
            {
                // Offset into the result if crop is larger than source (zero-padded)
                int destY = (CropHeight - cropH) / 2 + y;
                int destX = (CropWidth - cropW) / 2 + x;

                for (int c = 0; c < data.Channels; c++)
                {
                    result.SetPixel(destY, destX, c, data.GetPixel(cropY + y, cropX + x, c));
                }
            }
        }

        var parameters = new Dictionary<string, object>
        {
            ["crop_x"] = cropX,
            ["crop_y"] = cropY,
            ["crop_width"] = cropW,
            ["crop_height"] = cropH,
            ["output_width"] = CropWidth,
            ["output_height"] = CropHeight,
            ["original_width"] = width,
            ["original_height"] = height
        };

        return (result, parameters);
    }

    /// <summary>
    /// Transforms a bounding box after center crop.
    /// </summary>
    protected override BoundingBox<T> TransformBoundingBox(
        BoundingBox<T> box,
        IDictionary<string, object> transformParams,
        AugmentationContext<T> context)
    {
        int cropX = (int)transformParams["crop_x"];
        int cropY = (int)transformParams["crop_y"];

        var (x, y, w, h) = box.ToXYWH();

        double newX1 = x - cropX;
        double newY1 = y - cropY;
        double newX2 = x + w - cropX;
        double newY2 = y + h - cropY;

        newX1 = Math.Max(0, Math.Min(CropWidth, newX1));
        newY1 = Math.Max(0, Math.Min(CropHeight, newY1));
        newX2 = Math.Max(0, Math.Min(CropWidth, newX2));
        newY2 = Math.Max(0, Math.Min(CropHeight, newY2));

        var result = box.Clone();
        result.X1 = NumOps.FromDouble(newX1);
        result.Y1 = NumOps.FromDouble(newY1);
        result.X2 = NumOps.FromDouble(newX2);
        result.Y2 = NumOps.FromDouble(newY2);
        result.Format = BoundingBoxFormat.XYXY;
        return result;
    }

    /// <summary>
    /// Transforms a keypoint after center crop.
    /// </summary>
    protected override Keypoint<T> TransformKeypoint(
        Keypoint<T> keypoint,
        IDictionary<string, object> transformParams,
        AugmentationContext<T> context)
    {
        int cropX = (int)transformParams["crop_x"];
        int cropY = (int)transformParams["crop_y"];

        double x = NumOps.ToDouble(keypoint.X);
        double y = NumOps.ToDouble(keypoint.Y);

        double newX = x - cropX;
        double newY = y - cropY;

        var result = keypoint.Clone();
        result.X = NumOps.FromDouble(newX);
        result.Y = NumOps.FromDouble(newY);

        if (newX < 0 || newX >= CropWidth || newY < 0 || newY >= CropHeight)
        {
            result.Visibility = 0;
        }

        return result;
    }

    /// <summary>
    /// Transforms a segmentation mask after center crop.
    /// </summary>
    protected override SegmentationMask<T> TransformMask(
        SegmentationMask<T> mask,
        IDictionary<string, object> transformParams,
        AugmentationContext<T> context)
    {
        int cropX = (int)transformParams["crop_x"];
        int cropY = (int)transformParams["crop_y"];
        int cropW = (int)transformParams["crop_width"];
        int cropH = (int)transformParams["crop_height"];

        var dense = mask.ToDense();
        var cropped = new T[CropHeight, CropWidth];

        for (int y = 0; y < cropH; y++)
        {
            for (int x = 0; x < cropW; x++)
            {
                int srcX = cropX + x;
                int srcY = cropY + y;
                if (srcX >= 0 && srcX < mask.Width && srcY >= 0 && srcY < mask.Height)
                {
                    int destY = (CropHeight - cropH) / 2 + y;
                    int destX = (CropWidth - cropW) / 2 + x;
                    cropped[destY, destX] = dense[srcY, srcX];
                }
            }
        }

        return new SegmentationMask<T>(cropped, mask.Type, mask.ClassIndex)
        {
            ClassName = mask.ClassName,
            InstanceId = mask.InstanceId
        };
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var parameters = base.GetParameters();
        parameters["crop_height"] = CropHeight;
        parameters["crop_width"] = CropWidth;
        return parameters;
    }
}
