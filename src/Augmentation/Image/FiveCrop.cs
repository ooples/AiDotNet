namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Extracts five crops from an image: four corners and center.
/// </summary>
/// <remarks>
/// <para>
/// FiveCrop produces five fixed crops of the specified size from one image: top-left,
/// top-right, bottom-left, bottom-right, and center. This is commonly used at test time
/// to improve accuracy by averaging predictions over multiple views of the same image.
/// </para>
/// <para><b>For Beginners:</b> Instead of just looking at the center of an image, this
/// creates 5 different views by cropping from each corner and the center. During testing,
/// you can run your model on all 5 crops and average the predictions for better accuracy.
/// </para>
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>Test-time augmentation (TTA) for improved classification accuracy</item>
/// <item>When you want deterministic multi-crop evaluation</item>
/// </list>
/// </para>
/// <para><b>When NOT to use:</b>
/// <list type="bullet">
/// <item>Training (use RandomCrop instead)</item>
/// <item>When inference speed is critical (5x the computation)</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class FiveCrop<T> : ImageAugmenterBase<T>
{
    /// <summary>
    /// Gets the height of each crop.
    /// </summary>
    public int CropHeight { get; }

    /// <summary>
    /// Gets the width of each crop.
    /// </summary>
    public int CropWidth { get; }

    /// <summary>
    /// Creates a new FiveCrop augmentation.
    /// </summary>
    /// <param name="cropHeight">The height of each crop. Must be positive.</param>
    /// <param name="cropWidth">The width of each crop. Must be positive.</param>
    /// <param name="probability">The probability of applying this augmentation. Default is 1.0.</param>
    public FiveCrop(int cropHeight, int cropWidth, double probability = 1.0)
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
    /// Creates a new FiveCrop augmentation with square crops.
    /// </summary>
    /// <param name="size">The crop size (both height and width). Must be positive.</param>
    /// <param name="probability">The probability of applying this augmentation.</param>
    public FiveCrop(int size, double probability = 1.0)
        : this(size, size, probability)
    {
    }

    /// <summary>
    /// Extracts the five crops from the image.
    /// </summary>
    /// <param name="data">The input image.</param>
    /// <returns>A list of five cropped images: top-left, top-right, bottom-left, bottom-right, center.</returns>
    public List<ImageTensor<T>> GetCrops(ImageTensor<T> data)
    {
        int height = data.Height;
        int width = data.Width;

        if (CropHeight > height || CropWidth > width)
        {
            throw new ArgumentException(
                $"Crop size ({CropHeight}x{CropWidth}) exceeds image size ({height}x{width}).");
        }

        var crops = new List<ImageTensor<T>>(5);

        // Top-left
        crops.Add(ExtractCrop(data, 0, 0));
        // Top-right
        crops.Add(ExtractCrop(data, 0, width - CropWidth));
        // Bottom-left
        crops.Add(ExtractCrop(data, height - CropHeight, 0));
        // Bottom-right
        crops.Add(ExtractCrop(data, height - CropHeight, width - CropWidth));
        // Center
        crops.Add(ExtractCrop(data, (height - CropHeight) / 2, (width - CropWidth) / 2));

        return crops;
    }

    /// <summary>
    /// Applies the augmentation. Returns the center crop by default.
    /// Use <see cref="GetCrops"/> to get all five crops.
    /// </summary>
    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        // When used in a pipeline, return just the center crop
        int height = data.Height;
        int width = data.Width;

        if (CropHeight > height || CropWidth > width)
        {
            throw new ArgumentException(
                $"Crop size ({CropHeight}x{CropWidth}) exceeds image size ({height}x{width}).");
        }

        return ExtractCrop(data, (height - CropHeight) / 2, (width - CropWidth) / 2);
    }

    private ImageTensor<T> ExtractCrop(ImageTensor<T> source, int startY, int startX)
    {
        var crop = new ImageTensor<T>(CropHeight, CropWidth, source.Channels, source.ChannelOrder, source.ColorSpace)
        {
            IsNormalized = source.IsNormalized,
            NormalizationMean = source.NormalizationMean,
            NormalizationStd = source.NormalizationStd,
            OriginalRange = source.OriginalRange
        };

        for (int y = 0; y < CropHeight; y++)
        {
            for (int x = 0; x < CropWidth; x++)
            {
                for (int c = 0; c < source.Channels; c++)
                {
                    crop.SetPixel(y, x, c, source.GetPixel(startY + y, startX + x, c));
                }
            }
        }

        return crop;
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var parameters = base.GetParameters();
        parameters["crop_height"] = CropHeight;
        parameters["crop_width"] = CropWidth;
        parameters["num_crops"] = 5;
        return parameters;
    }
}
