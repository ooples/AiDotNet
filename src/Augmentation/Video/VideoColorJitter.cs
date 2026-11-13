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
    /// <para>Default: 0.2, a factor drawn from [0.8, 1.2] (torchvision ColorJitter).</para>
    /// </remarks>
    public double BrightnessRange { get; }

    /// <summary>
    /// Gets the maximum contrast adjustment.
    /// </summary>
    /// <remarks>
    /// <para>Default: 0.2, a factor drawn from [0.8, 1.2] (torchvision ColorJitter).</para>
    /// </remarks>
    public double ContrastRange { get; }

    /// <summary>
    /// Gets the maximum saturation adjustment.
    /// </summary>
    /// <remarks>
    /// <para>Default: 0.2, a factor drawn from [0.8, 1.2] (torchvision ColorJitter).</para>
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
    /// <remarks>
    /// Follows torchvision's <c>ColorJitter</c>: brightness, contrast and saturation factors are drawn from
    /// [max(0, 1 - range), 1 + range] once per clip (so every frame gets the same jitter), the three
    /// adjustments run in a random order, and the image is clipped to its value range after each one.
    /// </remarks>
    protected override ImageTensor<T>[] ApplyAugmentation(ImageTensor<T>[] data, AugmentationContext<T> context)
    {
        if (GetFrameCount(data) == 0) return data;

        ValidateFrameDimensions(data);

        // Sample adjustment parameters (consistent across all frames)
        double brightnessFactor = context.GetRandomDouble(Math.Max(0.0, 1 - BrightnessRange), 1 + BrightnessRange);
        double contrastFactor = context.GetRandomDouble(Math.Max(0.0, 1 - ContrastRange), 1 + ContrastRange);
        double saturationFactor = context.GetRandomDouble(Math.Max(0.0, 1 - SaturationRange), 1 + SaturationRange);

        // torchvision applies the adjustments in a random order (fn_idx = randperm).
        var order = new[] { ColorAdjustment.Brightness, ColorAdjustment.Contrast, ColorAdjustment.Saturation };
        for (int i = order.Length - 1; i > 0; i--)
        {
            int j = context.Random.Next(i + 1);
            (order[i], order[j]) = (order[j], order[i]);
        }

        var result = new ImageTensor<T>[data.Length];
        for (int f = 0; f < data.Length; f++)
        {
            result[f] = ApplyColorJitterToFrame(data[f], order, brightnessFactor, contrastFactor, saturationFactor);
        }

        return result;
    }

    private enum ColorAdjustment
    {
        Brightness,
        Contrast,
        Saturation
    }

    private ImageTensor<T> ApplyColorJitterToFrame(
        ImageTensor<T> frame,
        ColorAdjustment[] order,
        double brightness,
        double contrast,
        double saturation)
    {
        var engine = AiDotNetEngine.Current;
        int height = frame.Height;
        int width = frame.Width;
        int channels = frame.Channels;
        T maxVal = NumOps.FromDouble(frame.IsNormalized ? 1.0 : 255.0);

        // Work in CHW; colour channels are the first three (or the single gray channel), the rest
        // (alpha) pass through unchanged.
        var chw = frame.ToChannelOrder(ChannelOrder.CHW).Data;
        int colorChannels = channels >= 3 ? 3 : 1;
        var color = engine.TensorSlice(chw, new[] { 0, 0, 0 }, new[] { colorChannels, height, width });
        T brightnessFactor = NumOps.FromDouble(brightness);
        T contrastFactor = NumOps.FromDouble(contrast);
        T saturationFactor = NumOps.FromDouble(saturation);

        foreach (var adjustment in order)
        {
            color = adjustment switch
            {
                ColorAdjustment.Brightness => Blend(color, brightnessFactor, null, maxVal),
                ColorAdjustment.Contrast => Blend(
                    color,
                    contrastFactor,
                    engine.TensorAddScalar(new Tensor<T>(new[] { colorChannels, height, width }), engine.TensorMean(Grayscale(color))),
                    maxVal),
                ColorAdjustment.Saturation => colorChannels == 3
                    ? Blend(
                        color,
                        saturationFactor,
                        engine.TensorBroadcastTo(Grayscale(color), new[] { 3, height, width }),
                        maxVal)
                    : color,
                _ => color
            };
        }

        var adjusted = colorChannels < channels
            ? engine.TensorConcatenate(
                new[] { color, engine.TensorSlice(chw, new[] { colorChannels, 0, 0 }, new[] { channels - colorChannels, height, width }) },
                0)
            : color;

        var resultChw = new ImageTensor<T>(adjusted, ChannelOrder.CHW, frame.ColorSpace)
        {
            IsNormalized = frame.IsNormalized,
            NormalizationMean = frame.NormalizationMean,
            NormalizationStd = frame.NormalizationStd,
            OriginalRange = frame.OriginalRange,
            Metadata = frame.Metadata
        };
        var result = resultChw.ToChannelOrder(frame.ChannelOrder);
        result.IsNormalized = frame.IsNormalized;
        return result;
    }

    /// <summary>
    /// clip(factor * image + (1 - factor) * other, 0, max) with other = 0 when null (torchvision _blend).
    /// </summary>
    private static Tensor<T> Blend(Tensor<T> image, T factor, Tensor<T>? other, T maxVal)
    {
        var engine = AiDotNetEngine.Current;
        var blended = engine.TensorMultiplyScalar(image, factor);
        if (other is not null)
        {
            blended = engine.TensorAdd(blended, engine.TensorMultiplyScalar(other, NumOps.Subtract(NumOps.One, factor)));
        }

        return engine.TensorClamp(blended, NumOps.Zero, maxVal);
    }

    /// <summary>
    /// ITU-R 601 luma 0.2989 R + 0.587 G + 0.114 B as [1, H, W] (torchvision rgb_to_grayscale); a single
    /// channel is its own grayscale.
    /// </summary>
    private static Tensor<T> Grayscale(Tensor<T> color)
    {
        var engine = AiDotNetEngine.Current;
        int channels = color.Shape[0];
        if (channels == 1)
        {
            return color;
        }

        int height = color.Shape[1];
        int width = color.Shape[2];
        Tensor<T> Channel(int c) => engine.TensorSlice(color, new[] { c, 0, 0 }, new[] { 1, height, width });
        return engine.TensorAdd(
            engine.TensorAdd(
                engine.TensorMultiplyScalar(Channel(0), NumOps.FromDouble(0.2989)),
                engine.TensorMultiplyScalar(Channel(1), NumOps.FromDouble(0.587))),
            engine.TensorMultiplyScalar(Channel(2), NumOps.FromDouble(0.114)));
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
