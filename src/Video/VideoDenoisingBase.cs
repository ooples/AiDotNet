using System.Collections.Generic;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;

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
[TensorLayout(TensorAxis.Frames, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Frames, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Output)]
public abstract partial class VideoDenoisingBase<T> : VideoNeuralNetworkBase<T>, IShapeContract
{
    /// <summary>
    /// Video denoisers accept public frame tensors on the unsigned 8-bit pixel scale.
    /// The native model path normalizes these values to [0, 1] before forwarding.
    /// </summary>
    public override LayerInputDomain GetInputDomain(int[]? inputShape) => VideoPixelInputDomain.Value;

    /// <summary>
    /// Denoised public outputs use the same unsigned 8-bit pixel scale as their inputs.
    /// </summary>
    public override LayerInputDomain GetOutputDomain(int[]? outputShape) => VideoPixelInputDomain.Value;

    /// <inheritdoc />
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
        => inputRank == 4
            ?
            [
                new OutputAxisContract(TensorAxis.Frames, AxisRelation.Same(TensorAxis.Frames)),
                new OutputAxisContract(TensorAxis.Channels, AxisRelation.Same(TensorAxis.Channels)),
                new OutputAxisContract(TensorAxis.Height, AxisRelation.Same(TensorAxis.Height)),
                new OutputAxisContract(TensorAxis.Width, AxisRelation.Same(TensorAxis.Width)),
            ]
            : null;

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
    /// Runs the native denoising stack on a public raw-frame tensor while keeping the
    /// tape objective in the normalized model domain.
    /// </summary>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
    {
        var preprocessed = PreprocessFrames(input);
        return ForwardPreprocessedForTraining(preprocessed);
    }

    /// <summary>
    /// Runs the registered layer graph for an input that has already passed through
    /// <see cref="VideoNeuralNetworkBase{T}.PreprocessFrames"/>.
    /// </summary>
    /// <remarks>
    /// Architecture-specific denoisers use this hook when they add a residual readout
    /// or restore spatial resolution. Calling <c>base.ForwardForTraining</c> from such
    /// an override would preprocess the tensor a second time.
    /// </remarks>
    protected Tensor<T> ForwardPreprocessedForTraining(Tensor<T> preprocessedInput)
        => base.ForwardForTraining(preprocessedInput);

    /// <inheritdoc />
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // A model of this family that declares its own objective or learning rule takes its step through it.
        if (TryTrainDeclaredObjectives(input, expectedOutput)) return;
        if (IsOnnxMode)
            throw new NotSupportedException("Training is not supported in ONNX mode.");

        SetTrainingMode(true);
        try
        {
            // ForwardForTraining performs the model's input preprocessing. Supervision
            // must be placed in that same normalized domain, while callers continue to
            // provide and receive raw [0,255] frames through Train/Predict.
            var normalizedExpected = PreprocessTarget(expectedOutput);
            TrainWithTape(input, normalizedExpected);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    /// <summary>
    /// Maps public supervision into the native model's output domain. Most denoisers
    /// expose raw pixels and therefore normalize the target; models whose published
    /// API is already normalized can override this independently of input reshaping.
    /// </summary>
    protected virtual Tensor<T> PreprocessTarget(Tensor<T> expectedOutput)
        => NormalizeFrames(expectedOutput);

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

        var frame = Engine.Reshape(
            Engine.TensorNarrow(Engine.Reshape(noisyFrames, [noisyFrames.Length]), 0, 0, channels * height * width),
            [channels, height, width]);
        return LaplacianNoiseSigma(frame);
    }

    /// <summary>
    /// MAD noise estimate from the 4-neighbour Laplacian response over the interior pixels of
    /// <c>[planes, height, width]</c> data.
    /// </summary>
    /// <remarks>
    /// The Laplacian <c>4x - up - down - left - right</c> of i.i.d. Gaussian noise has standard deviation
    /// <c>sqrt(4² + 4)·sigma = sqrt(20)·sigma</c>, and the median of <c>|N(0, s²)|</c> is <c>0.6745·s</c>, so
    /// <c>sigma = median(|response|) / (0.6745·sqrt(20))</c>. The median is element <c>n/2</c> of the sorted
    /// responses (the upper median).
    /// </remarks>
    private protected double LaplacianNoiseSigma(Tensor<T> planes)
    {
        int height = planes.Shape[1];
        int width = planes.Shape[2];
        if (height < 3 || width < 3)
        {
            return 0;
        }

        Tensor<T> Shifted(int dh, int dw) =>
            Engine.TensorNarrow(Engine.TensorNarrow(planes, 1, 1 + dh, height - 2), 2, 1 + dw, width - 2);

        var neighbours = Engine.TensorAdd(
            Engine.TensorAdd(Shifted(-1, 0), Shifted(1, 0)),
            Engine.TensorAdd(Shifted(0, -1), Shifted(0, 1)));
        var laplacian = Engine.TensorSubtract(
            Engine.TensorMultiplyScalar(Shifted(0, 0), NumOps.FromDouble(4.0)), neighbours);
        var magnitude = Engine.TensorAbs(laplacian);

        var (median, _) = Engine.TensorKthvalue(magnitude, magnitude.Length / 2 + 1);
        return NumOps.ToDouble(median) / (0.6745 * Math.Sqrt(20.0));
    }

    /// <inheritdoc />
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        return Denoise(input);
    }
}
