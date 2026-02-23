namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Extracts ten crops from an image: five crops plus their horizontal flips.
/// </summary>
/// <remarks>
/// <para>
/// TenCrop extends <see cref="FiveCrop{T}"/> by also flipping each of the five crops
/// horizontally, yielding 10 total views. This provides more diverse test-time augmentation
/// for improved classification accuracy.
/// </para>
/// <para><b>For Beginners:</b> This creates 10 different views of your image: 5 crops
/// (4 corners + center) and their mirror images. By averaging your model's predictions
/// over all 10 views, you can get more reliable results at test time.</para>
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>Test-time augmentation when accuracy is paramount</item>
/// <item>Competition settings where every fraction of a percent matters</item>
/// </list>
/// </para>
/// <para><b>When NOT to use:</b>
/// <list type="bullet">
/// <item>Training (use RandomCrop + RandomHorizontalFlip instead)</item>
/// <item>Real-time applications (10x computation overhead)</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class TenCrop<T> : ImageAugmenterBase<T>
{
    private readonly FiveCrop<T> _fiveCrop;

    /// <summary>
    /// Gets the height of each crop.
    /// </summary>
    public int CropHeight { get; }

    /// <summary>
    /// Gets the width of each crop.
    /// </summary>
    public int CropWidth { get; }

    /// <summary>
    /// Gets whether to use vertical flips instead of horizontal flips.
    /// </summary>
    public bool UseVerticalFlip { get; }

    /// <summary>
    /// Creates a new TenCrop augmentation.
    /// </summary>
    /// <param name="cropHeight">The height of each crop. Must be positive.</param>
    /// <param name="cropWidth">The width of each crop. Must be positive.</param>
    /// <param name="useVerticalFlip">If true, use vertical flips instead of horizontal. Default is false.</param>
    /// <param name="probability">The probability of applying this augmentation. Default is 1.0.</param>
    public TenCrop(int cropHeight, int cropWidth, bool useVerticalFlip = false, double probability = 1.0)
        : base(probability)
    {
        if (cropHeight <= 0)
            throw new ArgumentOutOfRangeException(nameof(cropHeight), "Crop height must be positive.");
        if (cropWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(cropWidth), "Crop width must be positive.");

        CropHeight = cropHeight;
        CropWidth = cropWidth;
        UseVerticalFlip = useVerticalFlip;
        _fiveCrop = new FiveCrop<T>(cropHeight, cropWidth);
    }

    /// <summary>
    /// Creates a new TenCrop augmentation with square crops.
    /// </summary>
    /// <param name="size">The crop size. Must be positive.</param>
    /// <param name="useVerticalFlip">If true, use vertical flips instead of horizontal.</param>
    /// <param name="probability">The probability of applying this augmentation.</param>
    public TenCrop(int size, bool useVerticalFlip = false, double probability = 1.0)
        : this(size, size, useVerticalFlip, probability)
    {
    }

    /// <summary>
    /// Extracts the ten crops from the image.
    /// </summary>
    /// <param name="data">The input image.</param>
    /// <returns>A list of ten cropped images: 5 original crops followed by 5 flipped crops.</returns>
    public List<ImageTensor<T>> GetCrops(ImageTensor<T> data)
    {
        var fiveCrops = _fiveCrop.GetCrops(data);
        var allCrops = new List<ImageTensor<T>>(10);

        allCrops.AddRange(fiveCrops);

        foreach (var crop in fiveCrops)
        {
            allCrops.Add(FlipCrop(crop));
        }

        return allCrops;
    }

    /// <summary>
    /// Applies the augmentation. Returns the center crop by default.
    /// Use <see cref="GetCrops"/> to get all ten crops.
    /// </summary>
    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        int height = data.Height;
        int width = data.Width;

        if (CropHeight > height || CropWidth > width)
        {
            throw new ArgumentException(
                $"Crop size ({CropHeight}x{CropWidth}) exceeds image size ({height}x{width}).");
        }

        // Return center crop when used in pipeline
        int startY = (height - CropHeight) / 2;
        int startX = (width - CropWidth) / 2;

        var result = new ImageTensor<T>(CropHeight, CropWidth, data.Channels, data.ChannelOrder, data.ColorSpace)
        {
            IsNormalized = data.IsNormalized,
            NormalizationMean = data.NormalizationMean,
            NormalizationStd = data.NormalizationStd,
            OriginalRange = data.OriginalRange
        };

        for (int y = 0; y < CropHeight; y++)
        {
            for (int x = 0; x < CropWidth; x++)
            {
                for (int c = 0; c < data.Channels; c++)
                {
                    result.SetPixel(y, x, c, data.GetPixel(startY + y, startX + x, c));
                }
            }
        }

        return result;
    }

    private ImageTensor<T> FlipCrop(ImageTensor<T> crop)
    {
        var flipped = new ImageTensor<T>(crop.Height, crop.Width, crop.Channels, crop.ChannelOrder, crop.ColorSpace)
        {
            IsNormalized = crop.IsNormalized,
            NormalizationMean = crop.NormalizationMean,
            NormalizationStd = crop.NormalizationStd,
            OriginalRange = crop.OriginalRange
        };

        for (int y = 0; y < crop.Height; y++)
        {
            for (int x = 0; x < crop.Width; x++)
            {
                for (int c = 0; c < crop.Channels; c++)
                {
                    if (UseVerticalFlip)
                    {
                        flipped.SetPixel(crop.Height - 1 - y, x, c, crop.GetPixel(y, x, c));
                    }
                    else
                    {
                        flipped.SetPixel(y, crop.Width - 1 - x, c, crop.GetPixel(y, x, c));
                    }
                }
            }
        }

        return flipped;
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var parameters = base.GetParameters();
        parameters["crop_height"] = CropHeight;
        parameters["crop_width"] = CropWidth;
        parameters["use_vertical_flip"] = UseVerticalFlip;
        parameters["num_crops"] = 10;
        return parameters;
    }
}
