using AiDotNet.ActivationFunctions;
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
public class TemporalConvolution<T> : LayerBase<T>
{
    private readonly int _channels;
    private readonly int _kernelSize;
    private readonly int _numFrames;
    private readonly bool _causal;
    private readonly DenseLayer<T> _conv;
    private readonly LayerNormalizationLayer<T> _norm;
    private static new readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private Tensor<T>? _lastInput;

    private static Tensor<T> AddTensors(Tensor<T> a, Tensor<T> b)
    {
        return a.Transform((v, idx) => NumOps.Add(v, b.Data.Span[idx]));
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
        _conv = new DenseLayer<T>(channels, channels, (IActivationFunction<T>)new GELUActivation<T>());

        _norm = new LayerNormalizationLayer<T>(channels);
    }

    /// <summary>
    /// Applies temporal convolution across frames.
    /// </summary>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;

        var normed = _norm.Forward(input);
        var convOut = _conv.Forward(normed);
        return AddTensors(input, convOut);
    }

    /// <inheritdoc />
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        var convGrad = _conv.Backward(outputGradient);
        var normGrad = _norm.Backward(convGrad);
        return AddTensors(outputGradient, normGrad);
    }

    /// <inheritdoc />
    public override void UpdateParameters(T learningRate)
    {
        _conv.UpdateParameters(learningRate);
        _norm.UpdateParameters(learningRate);
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var convParams = _conv.GetParameters();
        var normParams = _norm.GetParameters();
        var combined = new Vector<T>(convParams.Length + normParams.Length);
        for (int i = 0; i < convParams.Length; i++)
            combined[i] = convParams[i];
        for (int i = 0; i < normParams.Length; i++)
            combined[convParams.Length + i] = normParams[i];
        return combined;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        var convParams = _conv.GetParameters();
        int convCount = convParams.Length;
        int normCount = parameters.Length - convCount;

        var newConvParams = new Vector<T>(convCount);
        var newNormParams = new Vector<T>(normCount);
        for (int i = 0; i < convCount; i++)
            newConvParams[i] = parameters[i];
        for (int i = 0; i < normCount; i++)
            newNormParams[i] = parameters[convCount + i];

        _conv.SetParameters(newConvParams);
        _norm.SetParameters(newNormParams);
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        _lastInput = null;
        _conv.ResetState();
        _norm.ResetState();
    }

    /// <inheritdoc />
    public override bool SupportsJitCompilation => false;

    /// <inheritdoc />
    public override Autodiff.ComputationNode<T> ExportComputationGraph(List<Autodiff.ComputationNode<T>> inputNodes)
    {
        throw new NotSupportedException("TemporalConvolution does not support JIT compilation.");
    }
}
