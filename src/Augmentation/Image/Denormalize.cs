namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Reverses normalization of an image tensor, restoring original pixel value ranges.
/// </summary>
/// <remarks>
/// <para>
/// Denormalization reverses the normalization operation: <c>output = input * std + mean</c>.
/// This is the inverse of <see cref="Normalize{T}"/> and restores pixel values to their
/// original range for visualization or saving.
/// </para>
/// <para><b>For Beginners:</b> After your model processes an image, the pixel values are
/// in a normalized range (roughly -2 to +2). To display or save the image, you need to
/// convert back to normal pixel values (0-255 or 0-1). This operation does that.</para>
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>Visualizing model inputs or intermediate activations</item>
/// <item>Saving processed images back to disk</item>
/// <item>Converting model output back to displayable format</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class Denormalize<T> : ImageAugmenterBase<T>
{
    /// <summary>
    /// Gets the per-channel mean values used in the original normalization.
    /// </summary>
    public double[] Mean { get; }

    /// <summary>
    /// Gets the per-channel standard deviation values used in the original normalization.
    /// </summary>
    public double[] Std { get; }

    /// <summary>
    /// Gets whether to clamp output values to [0, 1].
    /// </summary>
    public bool ClampOutput { get; }

    /// <summary>
    /// Creates a new denormalization augmentation.
    /// </summary>
    /// <param name="mean">Per-channel mean values from the original normalization.</param>
    /// <param name="std">Per-channel standard deviation values from the original normalization.</param>
    /// <param name="clampOutput">Whether to clamp output to [0, 1]. Default is true.</param>
    /// <param name="probability">The probability of applying this augmentation. Default is 1.0.</param>
    public Denormalize(double[] mean, double[] std, bool clampOutput = true, double probability = 1.0)
        : base(probability)
    {
        if (mean == null || mean.Length == 0)
            throw new ArgumentException("Mean array must not be null or empty.", nameof(mean));
        if (std == null || std.Length == 0)
            throw new ArgumentException("Std array must not be null or empty.", nameof(std));
        if (mean.Length != std.Length)
            throw new ArgumentException("Mean and std arrays must have the same length.");

        Mean = (double[])mean.Clone();
        Std = (double[])std.Clone();
        ClampOutput = clampOutput;
    }

    /// <summary>
    /// Creates a denormalization for ImageNet statistics.
    /// </summary>
    /// <param name="clampOutput">Whether to clamp output to [0, 1].</param>
    /// <param name="probability">The probability of applying this augmentation.</param>
    public static Denormalize<T> ImageNet(bool clampOutput = true, double probability = 1.0) =>
        new([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], clampOutput, probability);

    /// <summary>
    /// Creates a denormalization for CLIP statistics.
    /// </summary>
    /// <param name="clampOutput">Whether to clamp output to [0, 1].</param>
    /// <param name="probability">The probability of applying this augmentation.</param>
    public static Denormalize<T> Clip(bool clampOutput = true, double probability = 1.0) =>
        new([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711], clampOutput, probability);

    /// <summary>
    /// Applies denormalization: output = input * std + mean.
    /// </summary>
    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();
        int height = data.Height;
        int width = data.Width;
        int channels = data.Channels;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                for (int c = 0; c < channels; c++)
                {
                    double value = NumOps.ToDouble(data.GetPixel(y, x, c));
                    int idx = c % Mean.Length;
                    double denormalized = value * Std[idx] + Mean[idx];

                    if (ClampOutput)
                    {
                        denormalized = Math.Max(0.0, Math.Min(1.0, denormalized));
                    }

                    result.SetPixel(y, x, c, NumOps.FromDouble(denormalized));
                }
            }
        }

        result.IsNormalized = false;
        result.NormalizationMean = null;
        result.NormalizationStd = null;

        return result;
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var parameters = base.GetParameters();
        parameters["mean"] = Mean;
        parameters["std"] = Std;
        parameters["clamp_output"] = ClampOutput;
        return parameters;
    }
}
