using AiDotNet.Augmentation.Image;

namespace AiDotNet.Augmentation.Video;

/// <summary>
/// Applies color jitter (brightness, contrast, saturation) to video frames.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Color jitter randomly adjusts brightness, contrast,
/// and saturation of video frames. This helps models become robust to different
/// lighting conditions and camera settings.</para>
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>Videos captured with different cameras or lighting</item>
/// <item>Outdoor videos with varying light conditions</item>
/// <item>Reducing overfitting to specific color characteristics</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class VideoColorJitter<T> : SpatialVideoAugmenterBase<T>
{
    /// <summary>
    /// Gets the maximum brightness adjustment.
    /// </summary>
    /// <remarks>
    /// <para>Default: 0.2 (±20%)</para>
    /// </remarks>
    public double BrightnessRange { get; }

    /// <summary>
    /// Gets the maximum contrast adjustment.
    /// </summary>
    /// <remarks>
    /// <para>Default: 0.2 (±20%)</para>
    /// </remarks>
    public double ContrastRange { get; }

    /// <summary>
    /// Gets the maximum saturation adjustment.
    /// </summary>
    /// <remarks>
    /// <para>Default: 0.2 (±20%)</para>
    /// </remarks>
    public double SaturationRange { get; }

    /// <summary>
    /// Creates a new video color jitter augmentation.
    /// </summary>
    /// <param name="brightnessRange">Brightness adjustment range (default: 0.2).</param>
    /// <param name="contrastRange">Contrast adjustment range (default: 0.2).</param>
    /// <param name="saturationRange">Saturation adjustment range (default: 0.2).</param>
    /// <param name="probability">Probability of applying this augmentation (default: 0.5).</param>
    /// <param name="frameRate">Frame rate of the video (default: 30).</param>
    public VideoColorJitter(
        double brightnessRange = 0.2,
        double contrastRange = 0.2,
        double saturationRange = 0.2,
        double probability = 0.5,
        double frameRate = 30.0) : base(probability, frameRate)
    {
        if (brightnessRange < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(brightnessRange),
                "Brightness range must be non-negative.");
        }

        if (contrastRange < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(contrastRange),
                "Contrast range must be non-negative.");
        }

        if (saturationRange < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(saturationRange),
                "Saturation range must be non-negative.");
        }

        BrightnessRange = brightnessRange;
        ContrastRange = contrastRange;
        SaturationRange = saturationRange;
    }

    /// <inheritdoc />
    protected override ImageTensor<T>[] ApplyAugmentation(ImageTensor<T>[] data, AugmentationContext<T> context)
    {
        if (GetFrameCount(data) == 0) return data;

        ValidateFrameDimensions(data);

        // Sample adjustment parameters (consistent across all frames if ConsistentAcrossFrames is true)
        double brightnessAdjust = context.GetRandomDouble(-BrightnessRange, BrightnessRange);
        double contrastAdjust = context.GetRandomDouble(1 - ContrastRange, 1 + ContrastRange);
        double saturationAdjust = context.GetRandomDouble(1 - SaturationRange, 1 + SaturationRange);

        var result = new ImageTensor<T>[data.Length];

        for (int f = 0; f < data.Length; f++)
        {
            result[f] = ApplyColorJitterToFrame(
                data[f],
                brightnessAdjust,
                contrastAdjust,
                saturationAdjust);
        }

        return result;
    }

    private ImageTensor<T> ApplyColorJitterToFrame(
        ImageTensor<T> frame,
        double brightness,
        double contrast,
        double saturation)
    {
        int height = frame.Height;
        int width = frame.Width;
        int channels = frame.Channels;

        var result = new ImageTensor<T>(height, width, channels, frame.ChannelOrder, frame.ColorSpace);

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                if (channels >= 3)
                {
                    // RGB processing
                    double r = NumOps.ToDouble(frame.GetPixel(y, x, 0));
                    double g = NumOps.ToDouble(frame.GetPixel(y, x, 1));
                    double b = NumOps.ToDouble(frame.GetPixel(y, x, 2));

                    // Apply brightness (additive)
                    r += brightness;
                    g += brightness;
                    b += brightness;

                    // Apply contrast (multiplicative around 0.5)
                    r = (r - 0.5) * contrast + 0.5;
                    g = (g - 0.5) * contrast + 0.5;
                    b = (b - 0.5) * contrast + 0.5;

                    // Apply saturation
                    double gray = 0.299 * r + 0.587 * g + 0.114 * b;
                    r = gray + (r - gray) * saturation;
                    g = gray + (g - gray) * saturation;
                    b = gray + (b - gray) * saturation;

                    // Clamp to [0, 1]
                    result.SetPixel(y, x, 0, NumOps.FromDouble(Math.Max(0, Math.Min(1, r))));
                    result.SetPixel(y, x, 1, NumOps.FromDouble(Math.Max(0, Math.Min(1, g))));
                    result.SetPixel(y, x, 2, NumOps.FromDouble(Math.Max(0, Math.Min(1, b))));

                    // Copy alpha channel if present
                    if (channels == 4)
                    {
                        result.SetPixel(y, x, 3, frame.GetPixel(y, x, 3));
                    }
                }
                else
                {
                    // Grayscale - apply brightness and contrast only
                    double val = NumOps.ToDouble(frame.GetPixel(y, x, 0));
                    val += brightness;
                    val = (val - 0.5) * contrast + 0.5;
                    result.SetPixel(y, x, 0, NumOps.FromDouble(Math.Max(0, Math.Min(1, val))));
                }
            }
        }

        return result;
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var parameters = base.GetParameters();
        parameters["brightnessRange"] = BrightnessRange;
        parameters["contrastRange"] = ContrastRange;
        parameters["saturationRange"] = SaturationRange;
        return parameters;
    }
}
