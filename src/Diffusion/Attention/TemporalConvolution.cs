using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Diffusion.Attention;

/// <summary>
/// 1D temporal convolution layer for video diffusion models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet">
/// <item>Paper: "Make-A-Video: Text-to-Video Generation without Text-Video Data" (Singer et al., 2022)</item>
/// <item>Paper: "Video Diffusion Models" (Ho et al., 2022)</item>
/// </list></para>
/// <para><b>For Beginners:</b> Temporal Convolution applies 1D convolution along the time dimension, treating each spatial position independently. This pseudo-3D approach is much faster than full 3D attention while still modeling temporal relationships.</para>
/// <para>
/// Temporal convolution applies 1D convolution across the time dimension for each spatial position.
/// This provides local temporal modeling (mixing information from adjacent frames) as a complement
/// to temporal attention (which provides global temporal modeling). Temporal convolutions are:
/// - More efficient than attention for short-range temporal dependencies
/// - Often used alongside temporal attention in video UNets
/// - Optionally causal (only looking at past frames) for streaming generation
/// </para>
/// </remarks>
// Shape-preserving at rank 3 [Batch, Time, Features]; only that rank was probed, so only it is declared.
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class TemporalConvolution<T> : LayerBase<T>, IShapeContract
{
    private readonly int _channels;
    private readonly int _kernelSize;
    private readonly int _numFrames;
    private readonly bool _causal;
    private readonly DenseLayer<T> _conv;
    private readonly LayerNormalizationLayer<T> _norm;
    private Tensor<T>? _lastInput;

    private Tensor<T> AddTensors(Tensor<T> a, Tensor<T> b)
    {
        return AiDotNetEngine.Current.TensorAdd(a, b);
    }

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets the number of channels.
    /// </summary>
    public int Channels => _channels;

    /// <summary>
    /// Gets the temporal kernel size.
    /// </summary>
    public int KernelSize => _kernelSize;

    /// <summary>
    /// Gets whether causal convolution is used.
    /// </summary>
    public bool IsCausal => _causal;

    /// <summary>
    /// Initializes a new temporal convolution layer.
    /// </summary>
    /// <param name="channels">Number of input/output channels.</param>
    /// <param name="kernelSize">Temporal convolution kernel size.</param>
    /// <param name="numFrames">Number of video frames.</param>
    /// <param name="causal">Whether to use causal convolution (only past frames).</param>
    public TemporalConvolution(
        int channels,
        int kernelSize = 3,
        int numFrames = 16,
        bool causal = false)
        : base(
            new[] { 1, numFrames, channels },
            new[] { 1, numFrames, channels })
    {
        if (channels <= 0)
            throw new ArgumentOutOfRangeException(nameof(channels), "Channels must be positive.");
        if (kernelSize <= 0 || kernelSize % 2 == 0)
            throw new ArgumentOutOfRangeException(nameof(kernelSize), "Kernel size must be a positive odd number.");
        if (numFrames <= 0)
            throw new ArgumentOutOfRangeException(nameof(numFrames), "Number of frames must be positive.");

        _channels = channels;
        _kernelSize = kernelSize;
        _numFrames = numFrames;
        _causal = causal;

        // Approximates temporal 1D convolution via dense projection across channels per frame.
        // TODO: Replace with depthwise 1D convolution that uses kernelSize along the time axis
        // and applies causal masking when _causal is true (zero-pad left, no right context).
        // The current dense layer captures per-frame channel mixing but does not model
        // cross-frame temporal dependencies. This serves as a placeholder for ONNX inference.
        _conv = new DenseLayer<T>(channels, (IActivationFunction<T>)new GELUActivation<T>());

        _norm = new LayerNormalizationLayer<T>();
    }

    /// <summary>
    /// Applies temporal convolution across frames.
    /// </summary>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        // #1668: skip the backward-activation cache in inference (denoise-loop arena safety).
        _lastInput = ShouldCacheForBackward ? input : null;

        var normed = _norm.Forward(input);
        var convOut = _conv.Forward(normed);
        return AddTensors(input, convOut);
    }

    /// <inheritdoc />
    public override void UpdateParameters(T learningRate)
    {
        _conv.UpdateParameters(learningRate);
        _norm.UpdateParameters(learningRate);
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        _lastInput = null;
        _conv.ResetState();
        _norm.ResetState();
    }


}
