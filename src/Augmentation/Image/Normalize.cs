namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Normalizes an image tensor with per-channel mean and standard deviation.
/// </summary>
/// <remarks>
/// <para>
/// Normalization transforms pixel values using: <c>output = (input - mean) / std</c> for each
/// channel. This centers the data around zero and scales it to unit variance, which helps
/// neural networks train faster and converge more reliably.
/// </para>
/// <para><b>For Beginners:</b> Neural networks work better when input values are small numbers
/// centered around zero. Raw images have pixel values from 0 to 255 (or 0 to 1). Normalization
/// adjusts these values so they have a mean of 0 and standard deviation of 1, which makes
/// training more stable.</para>
/// <para><b>Common normalization values:</b>
/// <list type="bullet">
/// <item><b>ImageNet</b>: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]</item>
/// <item><b>CLIP</b>: mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]</item>
/// <item><b>Simple</b>: mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5] (maps [0,1] to [-1,1])</item>
/// </list>
/// </para>
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>As the last step in preprocessing before model input</item>
/// <item>When using pretrained models (must match training normalization)</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class Normalize<T> : ImageAugmenterBase<T>
{
    /// <summary>
    /// Gets the per-channel mean values.
    /// </summary>
    public double[] Mean { get; }

    /// <summary>
    /// Gets the per-channel standard deviation values.
    /// </summary>
    public double[] Std { get; }

    /// <summary>
    /// Creates a new normalization augmentation with per-channel mean and std.
    /// </summary>
    /// <param name="mean">Per-channel mean values. Must have one value per channel.</param>
    /// <param name="std">Per-channel standard deviation values. Must have one value per channel. All values must be non-zero.</param>
    /// <param name="probability">The probability of applying this augmentation. Default is 1.0.</param>
    public Normalize(double[] mean, double[] std, double probability = 1.0)
        : base(probability)
    {
        if (mean == null || mean.Length == 0)
            throw new ArgumentException("Mean array must not be null or empty.", nameof(mean));
        if (std == null || std.Length == 0)
            throw new ArgumentException("Std array must not be null or empty.", nameof(std));
        if (mean.Length != std.Length)
            throw new ArgumentException("Mean and std arrays must have the same length.");

        for (int i = 0; i < std.Length; i++)
        {
            if (Math.Abs(std[i]) < 1e-10)
                throw new ArgumentException($"Std value at index {i} must be non-zero.", nameof(std));
        }

        Mean = (double[])mean.Clone();
        Std = (double[])std.Clone();
    }

    /// <summary>
    /// Creates a normalization with ImageNet statistics.
    /// </summary>
    /// <param name="probability">The probability of applying this augmentation.</param>
    /// <returns>A Normalize instance with ImageNet mean and std.</returns>
    public static Normalize<T> ImageNet(double probability = 1.0) =>
        new([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], probability);

    /// <summary>
    /// Creates a normalization with CLIP statistics.
    /// </summary>
    /// <param name="probability">The probability of applying this augmentation.</param>
    /// <returns>A Normalize instance with CLIP mean and std.</returns>
    public static Normalize<T> Clip(double probability = 1.0) =>
        new([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711], probability);

    /// <summary>
    /// Creates a normalization that maps [0,1] to [-1,1].
    /// </summary>
    /// <param name="channels">The number of channels. Default is 3.</param>
    /// <param name="probability">The probability of applying this augmentation.</param>
    /// <returns>A Normalize instance that maps to [-1,1] range.</returns>
    public static Normalize<T> NegativeOneToOne(int channels = 3, double probability = 1.0)
    {
        var mean = new double[channels];
        var std = new double[channels];
        for (int i = 0; i < channels; i++)
        {
            mean[i] = 0.5;
            std[i] = 0.5;
        }
        return new Normalize<T>(mean, std, probability);
    }

    /// <summary>
    /// Applies normalization: output = (input - mean) / std.
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
                    int meanIdx = c % Mean.Length;
                    double normalized = (value - Mean[meanIdx]) / Std[meanIdx];
                    result.SetPixel(y, x, c, NumOps.FromDouble(normalized));
                }
            }
        }

        // Track normalization state
        result.IsNormalized = true;
        result.NormalizationMean = Array.ConvertAll(Mean, m => NumOps.FromDouble(m));
        result.NormalizationStd = Array.ConvertAll(Std, s => NumOps.FromDouble(s));

        return result;
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var parameters = base.GetParameters();
        parameters["mean"] = Mean;
        parameters["std"] = Std;
        return parameters;
    }
}
