namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Crops or pads an image to reach the target size, centering the content.
/// </summary>
/// <remarks>
/// <para>
/// CenterCropOrPad handles both cases: if the image is larger than the target, it center crops;
/// if smaller, it center pads with the specified fill value. This ensures a fixed output size
/// regardless of input dimensions.
/// </para>
/// <para><b>For Beginners:</b> This is a "smart resize" that doesn't stretch your image.
/// If the image is too big, it cuts away the edges. If too small, it adds borders. Either
/// way, the original content stays centered in the output.</para>
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>When input images vary in size and you need fixed dimensions</item>
/// <item>When you want to avoid any scaling/interpolation artifacts</item>
/// <item>Evaluation/inference preprocessing</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class CenterCropOrPad<T> : ImageAugmenterBase<T>
{
    /// <summary>
    /// Gets the target height.
    /// </summary>
    public int TargetHeight { get; }

    /// <summary>
    /// Gets the target width.
    /// </summary>
    public int TargetWidth { get; }

    /// <summary>
    /// Gets the fill value for padding (when image is smaller than target).
    /// </summary>
    public double FillValue { get; }

    /// <summary>
    /// Creates a new CenterCropOrPad augmentation.
    /// </summary>
    /// <param name="targetHeight">Target output height. Must be positive.</param>
    /// <param name="targetWidth">Target output width. Must be positive.</param>
    /// <param name="fillValue">Fill value for padding. Default is 0.</param>
    /// <param name="probability">Probability of applying. Default is 1.0.</param>
    public CenterCropOrPad(int targetHeight, int targetWidth, double fillValue = 0, double probability = 1.0)
        : base(probability)
    {
        if (targetHeight <= 0) throw new ArgumentOutOfRangeException(nameof(targetHeight));
        if (targetWidth <= 0) throw new ArgumentOutOfRangeException(nameof(targetWidth));

        TargetHeight = targetHeight;
        TargetWidth = targetWidth;
        FillValue = fillValue;
    }

    /// <summary>
    /// Creates a square CenterCropOrPad.
    /// </summary>
    public CenterCropOrPad(int targetSize, double fillValue = 0, double probability = 1.0)
        : this(targetSize, targetSize, fillValue, probability)
    {
    }

    /// <summary>
    /// Applies center crop or pad to reach target size.
    /// </summary>
    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        if (data.Height == TargetHeight && data.Width == TargetWidth)
            return data.Clone();

        var result = new ImageTensor<T>(TargetHeight, TargetWidth, data.Channels, data.ChannelOrder, data.ColorSpace)
        {
            IsNormalized = data.IsNormalized,
            NormalizationMean = data.NormalizationMean,
            NormalizationStd = data.NormalizationStd,
            OriginalRange = data.OriginalRange
        };

        // Fill with pad value
        T fill = NumOps.FromDouble(FillValue);
        for (int y = 0; y < TargetHeight; y++)
            for (int x = 0; x < TargetWidth; x++)
                for (int c = 0; c < data.Channels; c++)
                    result.SetPixel(y, x, c, fill);

        // Calculate source and destination offsets
        int srcOffY = data.Height > TargetHeight ? (data.Height - TargetHeight) / 2 : 0;
        int srcOffX = data.Width > TargetWidth ? (data.Width - TargetWidth) / 2 : 0;
        int dstOffY = TargetHeight > data.Height ? (TargetHeight - data.Height) / 2 : 0;
        int dstOffX = TargetWidth > data.Width ? (TargetWidth - data.Width) / 2 : 0;

        int copyH = Math.Min(data.Height, TargetHeight);
        int copyW = Math.Min(data.Width, TargetWidth);

        for (int y = 0; y < copyH; y++)
        {
            for (int x = 0; x < copyW; x++)
            {
                for (int c = 0; c < data.Channels; c++)
                {
                    result.SetPixel(dstOffY + y, dstOffX + x, c,
                        data.GetPixel(srcOffY + y, srcOffX + x, c));
                }
            }
        }

        return result;
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var parameters = base.GetParameters();
        parameters["target_height"] = TargetHeight;
        parameters["target_width"] = TargetWidth;
        parameters["fill_value"] = FillValue;
        return parameters;
    }
}
