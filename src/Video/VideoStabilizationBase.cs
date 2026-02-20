using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.Video;

/// <summary>
/// Base class for video stabilization models that remove camera shake from video sequences.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Video stabilization compensates for unwanted camera motion to produce smooth footage.
/// This base class provides:
///
/// - Camera trajectory estimation and smoothing
/// - Crop ratio management (stabilization typically requires cropping)
/// - Homography/affine transform estimation utilities
/// - Full-frame inpainting support for crop-free stabilization
///
/// Derived classes implement specific architectures like DIFRINT, DUT, FuSta, etc.
/// </para>
/// <para>
/// <b>For Beginners:</b> Video stabilization removes camera shake from handheld video footage.
/// It analyzes how the camera moved between frames and compensates by shifting/warping each
/// frame to create a smooth viewing experience. Some methods crop the edges (like smartphone
/// stabilization), while advanced neural methods can fill in the missing edges.
/// </para>
/// </remarks>
public abstract class VideoStabilizationBase<T> : VideoNeuralNetworkBase<T>
{
    /// <summary>
    /// Gets the crop ratio used for stabilization (fraction of frame that may be cropped).
    /// </summary>
    /// <remarks>
    /// <para>
    /// A value of 0.1 means up to 10% of the frame edges may be lost to stabilization.
    /// Full-frame methods (crop ratio = 0) inpaint the missing edges instead.
    /// </para>
    /// </remarks>
    public double CropRatio { get; protected set; } = 0.1;

    /// <summary>
    /// Gets whether this model supports full-frame stabilization (no cropping).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Full-frame methods use neural inpainting to fill edges that would otherwise be cropped,
    /// preserving the original field of view.
    /// </para>
    /// </remarks>
    public bool SupportsFullFrame { get; protected set; }

    /// <summary>
    /// Gets the smoothing window size for trajectory smoothing.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Larger window sizes produce smoother camera motion but may lose intentional camera movements.
    /// Common values: 15-60 frames.
    /// </para>
    /// </remarks>
    public int SmoothingWindowSize { get; protected set; } = 30;

    /// <summary>
    /// Initializes a new instance of the VideoStabilizationBase class.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="lossFunction">The loss function to use. If null, MSE loss is used.</param>
    /// <param name="maxGradNorm">Maximum gradient norm for gradient clipping.</param>
    protected VideoStabilizationBase(
        NeuralNetworkArchitecture<T> architecture,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 1.0)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), maxGradNorm)
    {
    }

    /// <summary>
    /// Stabilizes a sequence of video frames.
    /// </summary>
    /// <param name="unstableFrames">Unstable input frames [numFrames, channels, height, width].</param>
    /// <returns>Stabilized frames [numFrames, channels, height, width].</returns>
    public abstract Tensor<T> Stabilize(Tensor<T> unstableFrames);

    /// <summary>
    /// Estimates the camera trajectory (per-frame transforms) from the input sequence.
    /// </summary>
    /// <param name="frames">Input frames [numFrames, channels, height, width].</param>
    /// <returns>List of 3x3 homography matrices (as flat tensors) representing per-frame camera transforms.</returns>
    public virtual List<Tensor<T>> EstimateTrajectory(Tensor<T> frames)
    {
        int numFrames = frames.Shape[0];
        var trajectory = new List<Tensor<T>>();

        // Default: identity transforms
        for (int i = 0; i < numFrames; i++)
        {
            var identity = new Tensor<T>([3, 3]);
            identity.Data.Span[0] = NumOps.FromDouble(1.0); // [0,0]
            identity.Data.Span[4] = NumOps.FromDouble(1.0); // [1,1]
            identity.Data.Span[8] = NumOps.FromDouble(1.0); // [2,2]
            trajectory.Add(identity);
        }

        return trajectory;
    }

    /// <summary>
    /// Smooths a camera trajectory using a moving average filter.
    /// </summary>
    /// <param name="trajectory">Raw trajectory (list of 3x3 transforms).</param>
    /// <returns>Smoothed trajectory.</returns>
    protected List<Tensor<T>> SmoothTrajectory(List<Tensor<T>> trajectory)
    {
        int numFrames = trajectory.Count;
        var smoothed = new List<Tensor<T>>();

        int halfWindow = SmoothingWindowSize / 2;

        for (int i = 0; i < numFrames; i++)
        {
            var avgTransform = new Tensor<T>([3, 3]);
            int count = 0;

            for (int j = Math.Max(0, i - halfWindow); j <= Math.Min(numFrames - 1, i + halfWindow); j++)
            {
                for (int k = 0; k < 9; k++)
                {
                    avgTransform.Data.Span[k] = NumOps.Add(
                        avgTransform.Data.Span[k],
                        trajectory[j].Data.Span[k]);
                }
                count++;
            }

            for (int k = 0; k < 9; k++)
            {
                avgTransform.Data.Span[k] = NumOps.Multiply(
                    avgTransform.Data.Span[k],
                    NumOps.FromDouble(1.0 / count));
            }

            smoothed.Add(avgTransform);
        }

        return smoothed;
    }

    /// <inheritdoc />
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return Stabilize(input);
    }
}
