namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Resizes the image so that the shortest edge equals the specified max size, preserving aspect ratio.
/// </summary>
/// <remarks>
/// <para>
/// SmallestMaxSize scales the image so the shorter dimension matches the target size,
/// with the longer dimension scaled proportionally. This is the standard preprocessing
/// for ImageNet evaluation: resize shortest side to 256, then center crop to 224.
/// </para>
/// <para><b>For Beginners:</b> If you have a 400x200 image and set max_size to 256,
/// the short side (200) gets scaled to 256, and the long side scales proportionally
/// to 512, giving you a 512x256 image. This is typically followed by a center crop.</para>
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>ImageNet evaluation preprocessing (resize 256 → center crop 224)</item>
/// <item>When you want all images to have at least a minimum dimension</item>
/// <item>Before cropping operations to ensure sufficient image size</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class SmallestMaxSize<T> : ImageAugmenterBase<T>
{
    /// <summary>
    /// Gets the target size for the shortest edge.
    /// </summary>
    public int MaxSize { get; }

    /// <summary>
    /// Gets the interpolation mode.
    /// </summary>
    public InterpolationMode Interpolation { get; }

    /// <summary>
    /// Creates a new SmallestMaxSize augmentation.
    /// </summary>
    /// <param name="maxSize">Target size for the shortest edge. Must be positive.</param>
    /// <param name="interpolation">Interpolation mode. Default is Bilinear.</param>
    /// <param name="probability">Probability of applying. Default is 1.0.</param>
    public SmallestMaxSize(int maxSize, InterpolationMode interpolation = InterpolationMode.Bilinear,
        double probability = 1.0) : base(probability)
    {
        if (maxSize <= 0) throw new ArgumentOutOfRangeException(nameof(maxSize));
        MaxSize = maxSize;
        Interpolation = interpolation;
    }

    /// <summary>
    /// Resizes so the shortest edge equals MaxSize.
    /// </summary>
    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        int shortest = Math.Min(data.Height, data.Width);
        if (shortest == MaxSize)
            return data.Clone();

        double scale = (double)MaxSize / shortest;
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
