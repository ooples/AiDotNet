using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.Video;

/// <summary>
/// Base class for video denoising models that remove noise from video sequences.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Video denoising removes noise and grain from video while preserving detail and temporal
/// consistency. This base class provides:
///
/// - Noise level (sigma) handling for both blind and non-blind denoising
/// - Noise estimation utilities
/// - Temporal buffer management for multi-frame denoising
///
/// Derived classes implement specific architectures like FastDVDNet, BSVD, FloRNN, etc.
/// </para>
/// <para>
/// <b>For Beginners:</b> Video denoising cleans up grainy or noisy video footage. This is
/// common in low-light video, security camera footage, or video shot at high ISO settings.
/// The model learns to distinguish real detail from random noise, removing the noise while
/// keeping important details sharp. Using multiple frames helps because noise is random
/// (different in each frame) while real content is consistent across frames.
/// </para>
/// </remarks>
public abstract class VideoDenoisingBase<T> : VideoNeuralNetworkBase<T>
{
    /// <summary>
    /// Gets or sets the noise sigma level for non-blind denoising.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For non-blind denoising, the noise level must be specified.
    /// For blind denoising, this is estimated automatically.
    /// Common values: 5-50 for uint8 images (0-255 range).
    /// </para>
    /// </remarks>
    public double NoiseSigma { get; protected set; } = 25.0;

    /// <summary>
    /// Gets whether this model performs blind denoising (estimates noise level automatically).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Blind denoising models don't require knowing the noise level in advance.
    /// They estimate the noise characteristics from the input itself.
    /// </para>
    /// </remarks>
    public bool IsBlindDenoising { get; protected set; }

    /// <summary>
    /// Gets the temporal radius (number of frames before and after used for context).
    /// </summary>
    /// <remarks>
    /// <para>
    /// A temporal radius of 2 means the model uses 5 frames total: 2 before, current, 2 after.
    /// Larger radius provides more temporal information but increases memory and latency.
    /// </para>
    /// </remarks>
    public int TemporalRadius { get; protected set; } = 2;

    /// <summary>
    /// Initializes a new instance of the VideoDenoisingBase class.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="lossFunction">The loss function to use. If null, MSE loss is used.</param>
    /// <param name="maxGradNorm">Maximum gradient norm for gradient clipping.</param>
    protected VideoDenoisingBase(
        NeuralNetworkArchitecture<T> architecture,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 1.0)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), maxGradNorm)
    {
    }

    /// <summary>
    /// Denoises a sequence of video frames.
    /// </summary>
    /// <param name="noisyFrames">Noisy input frames [numFrames, channels, height, width].</param>
    /// <returns>Denoised frames [numFrames, channels, height, width].</returns>
    public abstract Tensor<T> Denoise(Tensor<T> noisyFrames);

    /// <summary>
    /// Estimates the noise level (sigma) from noisy input frames.
    /// </summary>
    /// <param name="noisyFrames">Noisy input frames [numFrames, channels, height, width].</param>
    /// <returns>Estimated noise sigma value.</returns>
    /// <remarks>
    /// <para>
    /// Uses the Median Absolute Deviation (MAD) estimator on high-frequency components.
    /// This provides a robust estimate of Gaussian noise standard deviation.
    /// </para>
    /// </remarks>
    public virtual double EstimateNoiseLevel(Tensor<T> noisyFrames)
    {
        if (noisyFrames is null) throw new ArgumentNullException(nameof(noisyFrames));
        if (noisyFrames.Rank < 4)
            throw new ArgumentException("Input must have 4 dimensions [numFrames, channels, height, width].", nameof(noisyFrames));

        // Simple MAD-based noise estimation on first frame
        int channels = noisyFrames.Shape[1];
        int height = noisyFrames.Shape[2];
        int width = noisyFrames.Shape[3];

        var diffs = new List<double>();

        // Compute high-frequency (Laplacian) response
        for (int c = 0; c < channels; c++)
        {
            for (int h = 1; h < height - 1; h++)
            {
                for (int w = 1; w < width - 1; w++)
                {
                    int idx = c * height * width + h * width + w;
                    int idxUp = c * height * width + (h - 1) * width + w;
                    int idxDown = c * height * width + (h + 1) * width + w;
                    int idxLeft = c * height * width + h * width + (w - 1);
                    int idxRight = c * height * width + h * width + (w + 1);

                    double center = NumOps.ToDouble(noisyFrames.Data.Span[idx]);
                    double laplacian = 4 * center
                        - NumOps.ToDouble(noisyFrames.Data.Span[idxUp])
                        - NumOps.ToDouble(noisyFrames.Data.Span[idxDown])
                        - NumOps.ToDouble(noisyFrames.Data.Span[idxLeft])
                        - NumOps.ToDouble(noisyFrames.Data.Span[idxRight]);

                    diffs.Add(Math.Abs(laplacian));
                }
            }
        }

        diffs.Sort();
        double median = diffs.Count > 0 ? diffs[diffs.Count / 2] : 0;

        // MAD estimator: sigma = median / 0.6745
        return median / 0.6745;
    }

    /// <inheritdoc />
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return Denoise(input);
    }
}
