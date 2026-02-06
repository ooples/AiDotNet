namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Resizes an image to fit within target dimensions while preserving the aspect ratio,
/// then pads to reach the exact target size.
/// </summary>
/// <remarks>
/// <para>
/// This combines resizing and padding in one step. First, the image is scaled so that it
/// fits within the target dimensions without exceeding them. Then, padding is added to
/// reach the exact target size. This ensures no distortion while producing a fixed-size output.
/// </para>
/// <para><b>For Beginners:</b> If you have a wide 400x200 image and want 300x300 output,
/// this first shrinks it to 300x150 (keeping proportions), then adds 75 pixels of padding
/// top and bottom to reach 300x300. The image looks correct, just with borders.</para>
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>When model requires fixed input size but aspect ratio matters</item>
/// <item>Object detection where distorted proportions would hurt accuracy</item>
/// <item>YOLO-style preprocessing (letterbox resizing)</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class ResizeWithAspectRatio<T> : ImageAugmenterBase<T>
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
    /// Gets the interpolation mode used for resizing.
    /// </summary>
    public InterpolationMode Interpolation { get; }

    /// <summary>
    /// Gets the padding mode for the fill areas.
    /// </summary>
    public PaddingMode PadMode { get; }

    /// <summary>
    /// Gets the fill value for constant padding.
    /// </summary>
    public double FillValue { get; }

    /// <summary>
    /// Creates a new ResizeWithAspectRatio augmentation.
    /// </summary>
    /// <param name="targetHeight">The target output height. Must be positive.</param>
    /// <param name="targetWidth">The target output width. Must be positive.</param>
    /// <param name="interpolation">The interpolation mode for resizing. Default is Bilinear.</param>
    /// <param name="padMode">The padding mode for fill areas. Default is Constant.</param>
    /// <param name="fillValue">The fill value for constant padding. Default is 0 (black).</param>
    /// <param name="probability">The probability of applying this augmentation.</param>
    public ResizeWithAspectRatio(
        int targetHeight,
        int targetWidth,
        InterpolationMode interpolation = InterpolationMode.Bilinear,
        PaddingMode padMode = PaddingMode.Constant,
        double fillValue = 0,
        double probability = 1.0) : base(probability)
    {
        if (targetHeight <= 0)
            throw new ArgumentOutOfRangeException(nameof(targetHeight), "Target height must be positive.");
        if (targetWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(targetWidth), "Target width must be positive.");

        TargetHeight = targetHeight;
        TargetWidth = targetWidth;
        Interpolation = interpolation;
        PadMode = padMode;
        FillValue = fillValue;
    }

    /// <summary>
    /// Creates a square ResizeWithAspectRatio augmentation.
    /// </summary>
    /// <param name="targetSize">The target size for both dimensions.</param>
    /// <param name="interpolation">The interpolation mode.</param>
    /// <param name="padMode">The padding mode.</param>
    /// <param name="fillValue">The fill value for constant padding.</param>
    /// <param name="probability">The probability of applying this augmentation.</param>
    public ResizeWithAspectRatio(
        int targetSize,
        InterpolationMode interpolation = InterpolationMode.Bilinear,
        PaddingMode padMode = PaddingMode.Constant,
        double fillValue = 0,
        double probability = 1.0)
        : this(targetSize, targetSize, interpolation, padMode, fillValue, probability)
    {
    }

    /// <summary>
    /// Resizes the image preserving aspect ratio, then pads to target size.
    /// </summary>
    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        int srcH = data.Height;
        int srcW = data.Width;

        // Calculate scale to fit within target while preserving aspect ratio
        double scale = Math.Min((double)TargetHeight / srcH, (double)TargetWidth / srcW);
        int resizedH = (int)Math.Round(srcH * scale);
        int resizedW = (int)Math.Round(srcW * scale);

        // Clamp to target dimensions
        resizedH = Math.Min(resizedH, TargetHeight);
        resizedW = Math.Min(resizedW, TargetWidth);

        // Step 1: Resize
        var resize = new Resize<T>(resizedH, resizedW, Interpolation);
        var resizeContext = new AugmentationContext<T>(isTraining: false);
        var resized = resize.Apply(data, resizeContext);

        // Step 2: Pad to target size if needed
        if (resizedH == TargetHeight && resizedW == TargetWidth)
            return resized;

        int padTop = (TargetHeight - resizedH) / 2;
        int padBottom = TargetHeight - resizedH - padTop;
        int padLeft = (TargetWidth - resizedW) / 2;
        int padRight = TargetWidth - resizedW - padLeft;

        var pad = new Pad<T>(padLeft, padRight, padTop, padBottom, PadMode, FillValue);
        var padContext = new AugmentationContext<T>(isTraining: false);
        return pad.Apply(resized, padContext);
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var parameters = base.GetParameters();
        parameters["target_height"] = TargetHeight;
        parameters["target_width"] = TargetWidth;
        parameters["interpolation"] = Interpolation.ToString();
        parameters["pad_mode"] = PadMode.ToString();
        parameters["fill_value"] = FillValue;
        return parameters;
    }
}
