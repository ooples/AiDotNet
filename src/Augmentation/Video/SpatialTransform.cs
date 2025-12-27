using AiDotNet.Augmentation.Image;

namespace AiDotNet.Augmentation.Video;

/// <summary>
/// Applies spatial transformations (flips, rotations) consistently to all frames.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Spatial transform applies image augmentations like
/// horizontal flip or rotation to all frames in the video consistently.
/// This ensures the transformation is coherent across the entire video.</para>
/// <para><b>Available transforms:</b>
/// <list type="bullet">
/// <item>Horizontal flip (mirror left-right)</item>
/// <item>Vertical flip (mirror top-bottom)</item>
/// <item>90° rotation (clockwise)</item>
/// <item>180° rotation</item>
/// <item>270° rotation (counter-clockwise)</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class SpatialTransform<T> : SpatialVideoAugmenterBase<T>
{
    /// <summary>
    /// Gets or sets whether to enable horizontal flip.
    /// </summary>
    public bool EnableHorizontalFlip { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to enable vertical flip.
    /// </summary>
    public bool EnableVerticalFlip { get; set; } = false;

    /// <summary>
    /// Gets or sets whether to enable 90° rotations.
    /// </summary>
    public bool EnableRotation90 { get; set; } = false;

    /// <summary>
    /// Gets the probability of horizontal flip (when enabled).
    /// </summary>
    public double HorizontalFlipProbability { get; }

    /// <summary>
    /// Gets the probability of vertical flip (when enabled).
    /// </summary>
    public double VerticalFlipProbability { get; }

    /// <summary>
    /// Creates a new spatial transform augmentation.
    /// </summary>
    /// <param name="horizontalFlipProbability">Probability of horizontal flip (default: 0.5).</param>
    /// <param name="verticalFlipProbability">Probability of vertical flip (default: 0.5).</param>
    /// <param name="probability">Probability of applying any transform (default: 0.5).</param>
    /// <param name="frameRate">Frame rate of the video (default: 30).</param>
    public SpatialTransform(
        double horizontalFlipProbability = 0.5,
        double verticalFlipProbability = 0.5,
        double probability = 0.5,
        double frameRate = 30.0) : base(probability, frameRate)
    {
        HorizontalFlipProbability = horizontalFlipProbability;
        VerticalFlipProbability = verticalFlipProbability;
    }

    /// <inheritdoc />
    protected override ImageTensor<T>[] ApplyAugmentation(ImageTensor<T>[] data, AugmentationContext<T> context)
    {
        if (GetFrameCount(data) == 0) return data;

        ValidateFrameDimensions(data);

        // Determine transforms to apply (consistent across all frames if ConsistentAcrossFrames is true)
        bool doHorizontalFlip = EnableHorizontalFlip && context.Random.NextDouble() < HorizontalFlipProbability;
        bool doVerticalFlip = EnableVerticalFlip && context.Random.NextDouble() < VerticalFlipProbability;
        int rotation = 0;
        if (EnableRotation90)
        {
            rotation = context.Random.Next(4) * 90; // 0, 90, 180, or 270
        }

        if (!doHorizontalFlip && !doVerticalFlip && rotation == 0)
        {
            return data; // No transform needed
        }

        var result = new ImageTensor<T>[data.Length];

        for (int f = 0; f < data.Length; f++)
        {
            var frame = data[f];
            result[f] = ApplyTransformToFrame(frame, doHorizontalFlip, doVerticalFlip, rotation);
        }

        return result;
    }

    private ImageTensor<T> ApplyTransformToFrame(
        ImageTensor<T> frame,
        bool horizontalFlip,
        bool verticalFlip,
        int rotation)
    {
        int height = frame.Height;
        int width = frame.Width;
        int channels = frame.Channels;

        // Determine output dimensions (swap for 90/270 rotation)
        int outHeight = (rotation == 90 || rotation == 270) ? width : height;
        int outWidth = (rotation == 90 || rotation == 270) ? height : width;

        var result = new ImageTensor<T>(outHeight, outWidth, channels, frame.ChannelOrder, frame.ColorSpace);

        for (int y = 0; y < outHeight; y++)
        {
            for (int x = 0; x < outWidth; x++)
            {
                // Map output coordinates to input coordinates
                int srcY = y;
                int srcX = x;

                // Apply rotation (reverse mapping)
                switch (rotation)
                {
                    case 90:
                        srcY = width - 1 - x;
                        srcX = y;
                        break;
                    case 180:
                        srcY = height - 1 - y;
                        srcX = width - 1 - x;
                        break;
                    case 270:
                        srcY = x;
                        srcX = height - 1 - y;
                        break;
                }

                // Apply flips (on the rotated coordinates)
                if (horizontalFlip)
                {
                    srcX = (rotation == 90 || rotation == 270 ? height : width) - 1 - srcX;
                }
                if (verticalFlip)
                {
                    srcY = (rotation == 90 || rotation == 270 ? width : height) - 1 - srcY;
                }

                // Clamp to valid range
                srcY = Math.Max(0, Math.Min(srcY, height - 1));
                srcX = Math.Max(0, Math.Min(srcX, width - 1));

                // Copy all channels
                for (int c = 0; c < channels; c++)
                {
                    result.SetPixel(y, x, c, frame.GetPixel(srcY, srcX, c));
                }
            }
        }

        return result;
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var parameters = base.GetParameters();
        parameters["enableHorizontalFlip"] = EnableHorizontalFlip;
        parameters["enableVerticalFlip"] = EnableVerticalFlip;
        parameters["enableRotation90"] = EnableRotation90;
        parameters["horizontalFlipProbability"] = HorizontalFlipProbability;
        parameters["verticalFlipProbability"] = VerticalFlipProbability;
        return parameters;
    }
}
