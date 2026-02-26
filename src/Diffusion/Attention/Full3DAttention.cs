using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Attention;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Diffusion.Attention;

/// <summary>
/// Full 3D attention across all spatio-temporal positions simultaneously.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet">
/// <item>Paper: "Sora: Creating video from text" (OpenAI, 2024)</item>
/// <item>Paper: "MovieGen: A Cast of Media Foundation Models" (Meta, 2024)</item>
/// </list></para>
/// <para><b>For Beginners:</b> Full 3D Attention jointly attends to all spatial and temporal positions simultaneously. While computationally expensive, it captures the richest interactions between space and time for highest quality results.</para>
/// <para>
/// Full 3D attention computes attention across all tokens in the video volume simultaneously
/// (all frames, all spatial positions). While computationally expensive at O((T*H*W)^2),
/// it captures the richest spatio-temporal interactions and is used in the most capable
/// video generation models like Sora and MovieGen.
/// </para>
/// <para>
/// Architecture:
/// - Flattens all frames and spatial positions into a single sequence
/// - Applies multi-head self-attention across the entire video volume
/// - Uses Flash Attention for memory efficiency
/// - Uses Flash Attention implementation for O(N) memory
/// </para>
/// </remarks>
public class Full3DAttention<T> : LayerBase<T>
{
    private readonly int _channels;
    private readonly int _numHeads;
    private readonly int _numFrames;
    private readonly int _spatialSize;
    private readonly FlashAttentionLayer<T> _fullAttention;
    private Tensor<T>? _lastInput;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets the number of channels.
    /// </summary>
    public int Channels => _channels;

    /// <summary>
    /// Gets the total sequence length (frames * height * width).
    /// </summary>
    public int TotalSequenceLength => _numFrames * _spatialSize * _spatialSize;

    /// <summary>
    /// Initializes a new full 3D attention layer.
    /// </summary>
    /// <param name="channels">Number of feature channels.</param>
    /// <param name="numHeads">Number of attention heads.</param>
    /// <param name="numFrames">Number of video frames.</param>
    /// <param name="spatialSize">Spatial size of feature maps.</param>
    public Full3DAttention(
        int channels,
        int numHeads = 16,
        int numFrames = 16,
        int spatialSize = 32)
        : base(
            new[] { 1, numFrames * spatialSize * spatialSize, channels },
            new[] { 1, numFrames * spatialSize * spatialSize, channels })
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

        int totalSeqLen = numFrames * spatialSize * spatialSize;

        var config = new FlashAttentionConfig
        {
            UseCausalMask = false,
            BlockSizeQ = 128,
            BlockSizeKV = 128,
            RecomputeInBackward = true,
            ReturnAttentionWeights = false,
            Precision = FlashAttentionPrecision.Float32
        };

        _fullAttention = new FlashAttentionLayer<T>(
            sequenceLength: totalSeqLen,
            embeddingDimension: channels,
            headCount: numHeads,
            config: config);
    }

    /// <summary>
    /// Applies full 3D attention across all spatio-temporal positions.
    /// </summary>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        return _fullAttention.Forward(input);
    }

    /// <inheritdoc />
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        return _fullAttention.Backward(outputGradient);
    }

    /// <inheritdoc />
    public override void UpdateParameters(T learningRate)
    {
        _fullAttention.UpdateParameters(learningRate);
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        return _fullAttention.GetParameters();
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        _fullAttention.SetParameters(parameters);
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        _lastInput = null;
        _fullAttention.ResetState();
    }

    /// <inheritdoc />
    public override bool SupportsJitCompilation => false;

    /// <inheritdoc />
    public override Autodiff.ComputationNode<T> ExportComputationGraph(List<Autodiff.ComputationNode<T>> inputNodes)
    {
        return _fullAttention.ExportComputationGraph(inputNodes);
    }
}
