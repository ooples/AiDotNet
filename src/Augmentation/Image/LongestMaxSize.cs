namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Resizes the image so that the longest edge equals the specified max size, preserving aspect ratio.
/// </summary>
/// <remarks>
/// <para>
/// LongestMaxSize scales the image so the longer dimension matches the target size,
/// with the shorter dimension scaled proportionally. This ensures the image fits within
/// a max_size x max_size box without exceeding it.
/// </para>
/// <para><b>For Beginners:</b> If you have a 400x200 image and set max_size to 300,
/// the long side (400) gets scaled to 300, and the short side scales proportionally
/// to 150, giving you a 300x150 image.</para>
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>Preprocessing for object detection (DETR, Faster R-CNN)</item>
/// <item>When you want to limit maximum dimension without distortion</item>
/// <item>Batch processing with variable-size images</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class LongestMaxSize<T> : ImageAugmenterBase<T>
{
    /// <summary>
    /// Gets the maximum size for the longest edge.
    /// </summary>
    public int MaxSize { get; }

    /// <summary>
    /// Gets the interpolation mode.
    /// </summary>
    public InterpolationMode Interpolation { get; }

    /// <summary>
    /// Creates a new LongestMaxSize augmentation.
    /// </summary>
    /// <param name="maxSize">Maximum size for the longest edge. Must be positive.</param>
    /// <param name="interpolation">Interpolation mode. Default is Bilinear.</param>
    /// <param name="probability">Probability of applying. Default is 1.0.</param>
    public LongestMaxSize(int maxSize, InterpolationMode interpolation = InterpolationMode.Bilinear,
        double probability = 1.0) : base(probability)
    {
        if (maxSize <= 0) throw new ArgumentOutOfRangeException(nameof(maxSize));
        MaxSize = maxSize;
        Interpolation = interpolation;
    }

    /// <summary>
    /// Resizes so the longest edge equals MaxSize.
    /// </summary>
    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        int longest = Math.Max(data.Height, data.Width);
        if (longest == MaxSize)
            return data.Clone();

        double scale = (double)MaxSize / longest;
        int newH = Math.Max(1, (int)Math.Round(data.Height * scale));
        int newW = Math.Max(1, (int)Math.Round(data.Width * scale));

        var resize = new Resize<T>(newH, newW, Interpolation);
        return resize.Apply(data, context);
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var parameters = base.GetParameters();
        parameters["max_size"] = MaxSize;
        parameters["interpolation"] = Interpolation.ToString();
        return parameters;
    }
}
