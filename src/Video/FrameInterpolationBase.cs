using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.Video;

/// <summary>
/// Base class for frame interpolation models that generate intermediate frames between existing frames.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Frame interpolation increases video frame rate by generating new frames between existing ones.
/// This base class provides:
///
/// - Arbitrary timestep interpolation (not just midpoint)
/// - Multi-frame interpolation (generate multiple intermediate frames)
/// - Flow-based and kernel-based interpolation utilities
/// - Temporal consistency support
///
/// Derived classes implement specific architectures like RIFE, AMT, EMA-VFI, VFIMamba, etc.
/// </para>
/// <para>
/// <b>For Beginners:</b> Frame interpolation makes choppy video smoother by adding new frames
/// in between existing ones. For example, converting 30fps video to 60fps or even 120fps.
/// The model "imagines" what the scene looks like at the intermediate time points.
/// </para>
/// </remarks>
public abstract class FrameInterpolationBase<T> : VideoNeuralNetworkBase<T>
{
    /// <summary>
    /// Gets the temporal scale factor (e.g., 2 for doubling frame rate).
    /// </summary>
    /// <remarks>
    /// <para>
    /// A value of 2 means one intermediate frame is generated between each pair (30fps to 60fps).
    /// A value of 4 means three intermediate frames are generated (30fps to 120fps).
    /// </para>
    /// </remarks>
    public int TemporalScaleFactor { get; protected set; } = 2;

    /// <summary>
    /// Gets whether this model supports arbitrary timestep interpolation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When true, the model can interpolate at any time t in (0, 1) between two frames.
    /// When false, the model only supports fixed timesteps (e.g., midpoint t=0.5).
    /// </para>
    /// </remarks>
    public bool SupportsArbitraryTimestep { get; protected set; }

    /// <summary>
    /// Initializes a new instance of the FrameInterpolationBase class.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="lossFunction">The loss function to use. If null, MSE loss is used.</param>
    /// <param name="maxGradNorm">Maximum gradient norm for gradient clipping.</param>
    protected FrameInterpolationBase(
        NeuralNetworkArchitecture<T> architecture,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 1.0)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), maxGradNorm)
    {
    }

    /// <summary>
    /// Interpolates a single intermediate frame between two input frames at the given timestep.
    /// </summary>
    /// <param name="frame0">First frame [channels, height, width].</param>
    /// <param name="frame1">Second frame [channels, height, width].</param>
    /// <param name="t">Timestep in (0, 1). t=0.5 is the midpoint.</param>
    /// <returns>Interpolated frame [channels, height, width].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Given two consecutive video frames, this generates a new frame
    /// that shows what the scene looks like at time t between them.
    /// t=0.5 gives you the exact midpoint. t=0.25 gives a frame closer to frame0.
    /// </para>
    /// </remarks>
    public abstract Tensor<T> Interpolate(Tensor<T> frame0, Tensor<T> frame1, double t = 0.5);

    /// <summary>
    /// Interpolates multiple intermediate frames between two input frames.
    /// </summary>
    /// <param name="frame0">First frame [channels, height, width].</param>
    /// <param name="frame1">Second frame [channels, height, width].</param>
    /// <param name="numIntermediate">Number of intermediate frames to generate.</param>
    /// <returns>Tensor containing all intermediate frames [numIntermediate, channels, height, width].</returns>
    public virtual Tensor<T> InterpolateMulti(Tensor<T> frame0, Tensor<T> frame1, int numIntermediate)
    {
        if (numIntermediate < 1)
            throw new ArgumentOutOfRangeException(nameof(numIntermediate), "Must generate at least 1 intermediate frame.");

        int channels = frame0.Shape[0];
        int height = frame0.Shape[1];
        int width = frame0.Shape[2];

        var result = new Tensor<T>([numIntermediate, channels, height, width]);

        for (int i = 0; i < numIntermediate; i++)
        {
            double t = (i + 1.0) / (numIntermediate + 1.0);
            var interpolated = Interpolate(frame0, frame1, t);
            StoreFrame(result, interpolated, i);
        }

        return result;
    }

    /// <summary>
    /// Interpolates all frames in a video sequence to increase frame rate.
    /// </summary>
    /// <param name="frames">Input frames [numFrames, channels, height, width].</param>
    /// <returns>Interpolated sequence with (numFrames - 1) * scaleFactor + 1 frames.</returns>
    public virtual Tensor<T> InterpolateSequence(Tensor<T> frames)
    {
        int numFrames = frames.Shape[0];
        int channels = frames.Shape[1];
        int height = frames.Shape[2];
        int width = frames.Shape[3];

        int numIntermediate = TemporalScaleFactor - 1;
        int outputFrames = (numFrames - 1) * TemporalScaleFactor + 1;

        var result = new Tensor<T>([outputFrames, channels, height, width]);

        for (int i = 0; i < numFrames; i++)
        {
            var frame = ExtractFrame(frames, i);
            StoreFrame(result, frame, i * TemporalScaleFactor);

            if (i < numFrames - 1)
            {
                var nextFrame = ExtractFrame(frames, i + 1);
                for (int j = 1; j <= numIntermediate; j++)
                {
                    double t = (double)j / TemporalScaleFactor;
                    var interpolated = Interpolate(frame, nextFrame, t);
                    StoreFrame(result, interpolated, i * TemporalScaleFactor + j);
                }
            }
        }

        return result;
    }

    /// <inheritdoc />
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return InterpolateSequence(input);
    }
}
