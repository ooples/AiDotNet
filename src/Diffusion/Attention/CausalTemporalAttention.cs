using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Attention;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Diffusion.Attention;

/// <summary>
/// Causal temporal attention for autoregressive video generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet">
/// <item>Paper: "CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer" (2024)</item>
/// <item>Paper: "Cosmos World Foundation Model" (NVIDIA, 2024)</item>
/// </list></para>
/// <para>
/// Causal temporal attention ensures that each frame can only attend to previous frames
/// and itself, not future frames. This is essential for:
/// - Streaming/real-time video generation
/// - Autoregressive frame-by-frame generation
/// - World models where future depends only on past
/// </para>
/// <para>
/// Architecture:
/// - Uses a causal mask to prevent attending to future frames
/// - Frame t can attend to frames 0, 1, ..., t but not t+1, t+2, ...
/// - Combined with spatial attention for full spatio-temporal modeling
/// </para>
/// </remarks>
public class CausalTemporalAttention<T> : LayerBase<T>
{
    private readonly int _channels;
    private readonly int _numHeads;
    private readonly int _numFrames;
    private readonly int _spatialSize;
    private readonly FlashAttentionLayer<T> _causalAttention;
    private Tensor<T>? _lastInput;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets the number of channels.
    /// </summary>
    public int Channels => _channels;

    /// <summary>
    /// Gets the number of frames.
    /// </summary>
    public int NumFrames => _numFrames;

    /// <summary>
    /// Whether causal masking is enabled.
    /// </summary>
    public bool IsCausal => true;

    /// <summary>
    /// Initializes a new causal temporal attention layer.
    /// </summary>
    /// <param name="channels">Number of feature channels.</param>
    /// <param name="numHeads">Number of attention heads.</param>
    /// <param name="numFrames">Maximum number of video frames.</param>
    /// <param name="spatialSize">Spatial size of feature maps.</param>
    public CausalTemporalAttention(
        int channels,
        int numHeads = 8,
        int numFrames = 16,
        int spatialSize = 64)
        : base(
            new[] { 1, numFrames, channels },
            new[] { 1, numFrames, channels })
    {
        if (channels <= 0)
            throw new ArgumentOutOfRangeException(nameof(channels), "Channels must be positive.");
        if (numHeads <= 0)
            throw new ArgumentOutOfRangeException(nameof(numHeads), "Number of heads must be positive.");
        if (channels % numHeads != 0)
            throw new ArgumentException($"Channels ({channels}) must be divisible by numHeads ({numHeads}).");

        _channels = channels;
        _numHeads = numHeads;
        _numFrames = numFrames;
        _spatialSize = spatialSize;

        var config = new FlashAttentionConfig
        {
            UseCausalMask = true,
            BlockSizeQ = 64,
            BlockSizeKV = 64,
            RecomputeInBackward = true,
            ReturnAttentionWeights = false,
            Precision = FlashAttentionPrecision.Float32
        };

        _causalAttention = new FlashAttentionLayer<T>(
            sequenceLength: numFrames,
            embeddingDimension: channels,
            headCount: numHeads,
            config: config);
    }

    /// <summary>
    /// Performs causal temporal attention where each frame attends only to past frames.
    /// </summary>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        return _causalAttention.Forward(input);
    }

    /// <inheritdoc />
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        return _causalAttention.Backward(outputGradient);
    }

    /// <inheritdoc />
    public override void UpdateParameters(T learningRate)
    {
        _causalAttention.UpdateParameters(learningRate);
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        return _causalAttention.GetParameters();
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        _causalAttention.SetParameters(parameters);
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        _lastInput = null;
        _causalAttention.ResetState();
    }

    /// <inheritdoc />
    public override bool SupportsJitCompilation => false;

    /// <inheritdoc />
    public override Autodiff.ComputationNode<T> ExportComputationGraph(List<Autodiff.ComputationNode<T>> inputNodes)
    {
        return _causalAttention.ExportComputationGraph(inputNodes);
    }
}
