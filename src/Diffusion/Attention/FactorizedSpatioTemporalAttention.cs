using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Attention;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Diffusion.Attention;

/// <summary>
/// Factorized spatio-temporal attention that applies spatial and temporal attention separately.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet">
/// <item>Paper: "Scalable Diffusion Models with Transformers" (Peebles and Xie, 2023)</item>
/// <item>Paper: "Video Diffusion Models" (Ho et al., 2022)</item>
/// </list></para>
/// <para><b>For Beginners:</b> Factorized Spatio-Temporal Attention processes spatial (within-frame) and temporal (across-frame) relationships separately. This is much more efficient than joint attention while still capturing both spatial detail and temporal motion.</para>
/// <para>
/// Factorized spatio-temporal attention decomposes full 3D attention into separate spatial
/// and temporal components. This reduces computational complexity from O((T*H*W)^2) to
/// O(T*(H*W)^2 + H*W*T^2), making it feasible for high-resolution long videos.
/// </para>
/// <para>
/// Architecture:
/// - Spatial attention: self-attention within each frame (across H*W positions)
/// - Temporal attention: self-attention across frames (for each spatial position)
/// - LayerNorm + residual connections around each attention block
/// </para>
/// </remarks>
public class FactorizedSpatioTemporalAttention<T> : LayerBase<T>
{
    private readonly int _channels;
    private readonly int _numHeads;
    private readonly int _numFrames;
    private readonly int _spatialSize;
    private readonly DiffusionAttention<T> _spatialAttention;
    private readonly TemporalSelfAttention<T> _temporalAttention;
    private readonly LayerNormalizationLayer<T> _spatialNorm;
    private readonly LayerNormalizationLayer<T> _temporalNorm;
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
    /// Initializes a new factorized spatio-temporal attention layer.
    /// </summary>
    /// <param name="channels">Number of feature channels.</param>
    /// <param name="numHeads">Number of attention heads.</param>
    /// <param name="numFrames">Number of video frames.</param>
    /// <param name="spatialSize">Spatial size of feature maps.</param>
    public FactorizedSpatioTemporalAttention(
        int channels,
        int numHeads = 8,
        int numFrames = 16,
        int spatialSize = 64)
        : base(
            new[] { 1, numFrames * spatialSize * spatialSize, channels },
            new[] { 1, numFrames * spatialSize * spatialSize, channels })
    {
        _channels = channels;
        _numHeads = numHeads;
        _numFrames = numFrames;
        _spatialSize = spatialSize;

        _spatialAttention = new DiffusionAttention<T>(
            channels: channels,
            numHeads: numHeads,
            spatialSize: spatialSize);

        _temporalAttention = new TemporalSelfAttention<T>(
            channels: channels,
            numHeads: numHeads,
            numFrames: numFrames,
            spatialSize: spatialSize);

        _spatialNorm = new LayerNormalizationLayer<T>(channels);
        _temporalNorm = new LayerNormalizationLayer<T>(channels);
    }

    /// <summary>
    /// Applies spatial attention then temporal attention with residual connections.
    /// </summary>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;

        // Spatial attention with residual
        var spatialNormed = _spatialNorm.Forward(input);
        var spatialOut = _spatialAttention.Forward(spatialNormed);
        var afterSpatial = AddTensors(input, spatialOut);

        // Temporal attention with residual
        var temporalNormed = _temporalNorm.Forward(afterSpatial);
        var temporalOut = _temporalAttention.Forward(temporalNormed);
        var output = AddTensors(afterSpatial, temporalOut);

        return output;
    }

    /// <inheritdoc />
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // Backward through temporal attention
        var temporalGrad = _temporalAttention.Backward(outputGradient);
        var temporalNormGrad = _temporalNorm.Backward(temporalGrad);

        // Add residual gradient
        var afterTemporalGrad = AddTensors(outputGradient, temporalNormGrad);

        // Backward through spatial attention
        var spatialGrad = _spatialAttention.Backward(afterTemporalGrad);
        var spatialNormGrad = _spatialNorm.Backward(spatialGrad);

        return AddTensors(afterTemporalGrad, spatialNormGrad);
    }

    /// <inheritdoc />
    public override void UpdateParameters(T learningRate)
    {
        _spatialAttention.UpdateParameters(learningRate);
        _temporalAttention.UpdateParameters(learningRate);
        _spatialNorm.UpdateParameters(learningRate);
        _temporalNorm.UpdateParameters(learningRate);
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var spatialParams = _spatialAttention.GetParameters();
        var temporalParams = _temporalAttention.GetParameters();
        var spatialNormParams = _spatialNorm.GetParameters();
        var temporalNormParams = _temporalNorm.GetParameters();

        int total = spatialParams.Length + temporalParams.Length +
                    spatialNormParams.Length + temporalNormParams.Length;
        var combined = new Vector<T>(total);
        int offset = 0;
        CopyParams(spatialParams, combined, ref offset);
        CopyParams(temporalParams, combined, ref offset);
        CopyParams(spatialNormParams, combined, ref offset);
        CopyParams(temporalNormParams, combined, ref offset);
        return combined;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        int offset = 0;
        var spatialParams = ExtractParams(parameters, _spatialAttention.GetParameters().Length, ref offset);
        var temporalParams = ExtractParams(parameters, _temporalAttention.GetParameters().Length, ref offset);
        var spatialNormParams = ExtractParams(parameters, _spatialNorm.GetParameters().Length, ref offset);
        var temporalNormParams = ExtractParams(parameters, _temporalNorm.GetParameters().Length, ref offset);

        _spatialAttention.SetParameters(spatialParams);
        _temporalAttention.SetParameters(temporalParams);
        _spatialNorm.SetParameters(spatialNormParams);
        _temporalNorm.SetParameters(temporalNormParams);
    }

    private static void CopyParams(Vector<T> src, Vector<T> dst, ref int offset)
    {
        for (int i = 0; i < src.Length; i++)
            dst[offset + i] = src[i];
        offset += src.Length;
    }

    private static Vector<T> ExtractParams(Vector<T> src, int count, ref int offset)
    {
        var result = new Vector<T>(count);
        for (int i = 0; i < count; i++)
            result[i] = src[offset + i];
        offset += count;
        return result;
    }

    private static new readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private Tensor<T> AddTensors(Tensor<T> a, Tensor<T> b)
    {
        return Engine.TensorAdd<T>(a, b);
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        _lastInput = null;
        _spatialAttention.ResetState();
        _temporalAttention.ResetState();
        _spatialNorm.ResetState();
        _temporalNorm.ResetState();
    }

    /// <inheritdoc />
    public override bool SupportsJitCompilation => false;

    /// <inheritdoc />
    public override Autodiff.ComputationNode<T> ExportComputationGraph(List<Autodiff.ComputationNode<T>> inputNodes)
    {
        throw new NotSupportedException(
            "FactorizedSpatioTemporalAttention does not support JIT compilation.");
    }
}
