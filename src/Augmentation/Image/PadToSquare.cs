namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Pads an image to make it square while preserving the original content centered.
/// </summary>
/// <remarks>
/// <para>
/// PadToSquare adds padding to the shorter dimension of an image so that height equals width.
/// The original image is centered within the square, with padding distributed evenly on both sides.
/// This is useful when models require square inputs but you want to preserve aspect ratio.
/// </para>
/// <para><b>For Beginners:</b> If your image is 200x300 pixels (tall rectangle), this will add
/// 50 pixels of padding to the left and right, making it 300x300 (square). The original image
/// stays in the center.</para>
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>When your model requires square input but images have varying aspect ratios</item>
/// <item>When you want to avoid distortion from non-uniform resizing</item>
/// <item>Object detection where distortion could change bounding box proportions</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class PadToSquare<T> : ImageAugmenterBase<T>
{
    /// <summary>
    /// Gets the padding mode.
    /// </summary>
    public PaddingMode Mode { get; }

    /// <summary>
    /// Gets the constant fill value (used when Mode is Constant).
    /// </summary>
    public double FillValue { get; }

    /// <summary>
    /// Creates a new PadToSquare augmentation.
    /// </summary>
    /// <param name="mode">The padding mode. Default is Constant.</param>
    /// <param name="fillValue">The fill value for Constant mode. Default is 0 (black).</param>
    /// <param name="probability">The probability of applying this augmentation.</param>
    public PadToSquare(PaddingMode mode = PaddingMode.Constant, double fillValue = 0,
        double probability = 1.0)
        : base(probability)
    {
        Mode = mode;
        FillValue = fillValue;
    }

    /// <summary>
    /// Pads the image to a square.
    /// </summary>
    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        int height = data.Height;
        int width = data.Width;

        if (height == width)
            return data.Clone();

        int padTop = 0, padBottom = 0, padLeft = 0, padRight = 0;

        if (height < width)
        {
            int totalPad = width - height;
            padTop = totalPad / 2;
            padBottom = totalPad - padTop;
        }
        else
        {
            int totalPad = height - width;
            padLeft = totalPad / 2;
            padRight = totalPad - padLeft;
        }

        var pad = new Pad<T>(padLeft, padRight, padTop, padBottom, Mode, FillValue);
        return pad.Apply(data, context);
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var parameters = base.GetParameters();
        parameters["mode"] = Mode.ToString();
        parameters["fill_value"] = FillValue;
        return parameters;
    }
}
