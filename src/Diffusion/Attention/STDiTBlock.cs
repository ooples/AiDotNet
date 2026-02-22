using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Attention;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Diffusion.Attention;

/// <summary>
/// Spatial-Temporal DiT (STDiT) block for efficient video generation transformers.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet">
/// <item>Paper: "Open-Sora: Democratizing Efficient Video Production for All" (2024)</item>
/// <item>Paper: "Latte: Latent Diffusion Transformer for Video Generation" (Ma et al., 2024)</item>
/// </list></para>
/// <para><b>For Beginners:</b> The STDiT (Spatial-Temporal DiT) Block alternates between spatial and temporal attention within a single transformer block. This efficient design is used in Open-Sora for balanced spatial and temporal processing.</para>
/// <para>
/// STDiT is the core building block for video DiT architectures. It combines spatial and temporal
/// attention with cross-attention conditioning in a single transformer block:
/// 1. Spatial self-attention (within each frame)
/// 2. Temporal self-attention (across frames per spatial position)
/// 3. Cross-attention to text conditioning
/// 4. Feed-forward network
/// Each sub-layer uses adaptive layer normalization (adaLN-Zero) for timestep conditioning.
/// </para>
/// </remarks>
public class STDiTBlock<T> : LayerBase<T>
{
    private readonly int _channels;
    private readonly int _numHeads;
    private readonly int _contextDim;
    private readonly int _numFrames;
    private readonly int _spatialSize;

    private readonly DiffusionAttention<T> _spatialAttention;
    private readonly TemporalSelfAttention<T> _temporalAttention;
    private readonly DiffusionCrossAttention<T> _crossAttention;
    private readonly DenseLayer<T> _ffnIn;
    private readonly DenseLayer<T> _ffnOut;
    private readonly LayerNormalizationLayer<T> _spatialNorm;
    private readonly LayerNormalizationLayer<T> _temporalNorm;
    private readonly LayerNormalizationLayer<T> _crossNorm;
    private readonly LayerNormalizationLayer<T> _ffnNorm;

    // Intentionally shadows base class NumOps with static field for use in static AddTensors helper
    private static new readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private Tensor<T>? _lastInput;
    private Tensor<T>? _afterSpatial;
    private Tensor<T>? _afterTemporal;
    private Tensor<T>? _afterCross;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    private static Tensor<T> AddTensors(Tensor<T> a, Tensor<T> b)
    {
        return a.Transform((v, idx) => NumOps.Add(v, b.Data.Span[idx]));
    }

    /// <summary>
    /// Gets the hidden dimension.
    /// </summary>
    public int Channels => _channels;

    /// <summary>
    /// Gets the context dimension for cross-attention.
    /// </summary>
    public int ContextDim => _contextDim;

    /// <summary>
    /// Initializes a new STDiT block.
    /// </summary>
    /// <param name="channels">Hidden dimension / number of channels.</param>
    /// <param name="numHeads">Number of attention heads.</param>
    /// <param name="contextDim">Dimension of text conditioning.</param>
    /// <param name="numFrames">Number of video frames.</param>
    /// <param name="spatialSize">Spatial resolution of latent feature maps.</param>
    /// <param name="ffnMultiplier">FFN hidden dimension multiplier.</param>
    public STDiTBlock(
        int channels,
        int numHeads = 16,
        int contextDim = 4096,
        int numFrames = 16,
        int spatialSize = 32,
        int ffnMultiplier = 4)
        : base(
            new[] { 1, numFrames * spatialSize * spatialSize, channels },
            new[] { 1, numFrames * spatialSize * spatialSize, channels })
    {
        _channels = channels;
        _numHeads = numHeads;
        _contextDim = contextDim;
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

        _crossAttention = new DiffusionCrossAttention<T>(
            queryDim: channels,
            contextDim: contextDim,
            numHeads: numHeads,
            spatialSize: spatialSize);

        int ffnHidden = channels * ffnMultiplier;
        _ffnIn = new DenseLayer<T>(channels, ffnHidden, (IActivationFunction<T>)new GELUActivation<T>());
        _ffnOut = new DenseLayer<T>(ffnHidden, channels, (IActivationFunction<T>)new IdentityActivation<T>());

        _spatialNorm = new LayerNormalizationLayer<T>(channels);
        _temporalNorm = new LayerNormalizationLayer<T>(channels);
        _crossNorm = new LayerNormalizationLayer<T>(channels);
        _ffnNorm = new LayerNormalizationLayer<T>(channels);
    }

    /// <summary>
    /// Applies the STDiT block: spatial attn -> temporal attn -> cross attn -> FFN.
    /// </summary>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;

        // 1. Spatial self-attention with residual
        _afterSpatial = AddTensors(input, _spatialAttention.Forward(_spatialNorm.Forward(input)));

        // 2. Temporal self-attention with residual
        _afterTemporal = AddTensors(_afterSpatial, _temporalAttention.Forward(_temporalNorm.Forward(_afterSpatial)));

        // 3. Cross-attention with residual
        _afterCross = AddTensors(_afterTemporal, _crossAttention.Forward(_crossNorm.Forward(_afterTemporal)));

        // 4. Feed-forward network with residual
        var ffnInput = _ffnNorm.Forward(_afterCross);
        var ffnHidden = _ffnIn.Forward(ffnInput);
        var ffnOutput = _ffnOut.Forward(ffnHidden);
        return AddTensors(_afterCross, ffnOutput);
    }

    /// <inheritdoc />
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // Backward through FFN residual: output = afterCross + ffn(norm(afterCross))
        // Gradient flows both through the FFN branch and the skip connection
        var ffnOutGrad = _ffnOut.Backward(outputGradient);
        var ffnInGrad = _ffnIn.Backward(ffnOutGrad);
        var ffnNormGrad = _ffnNorm.Backward(ffnInGrad);
        var afterCrossGrad = AddTensors(outputGradient, ffnNormGrad);

        // Backward through cross-attention residual: afterCross = afterTemporal + cross(norm(afterTemporal))
        var crossGrad = _crossAttention.Backward(afterCrossGrad);
        var crossNormGrad = _crossNorm.Backward(crossGrad);
        var afterTemporalGrad = AddTensors(afterCrossGrad, crossNormGrad);

        // Backward through temporal attention residual: afterTemporal = afterSpatial + temporal(norm(afterSpatial))
        var temporalGrad = _temporalAttention.Backward(afterTemporalGrad);
        var temporalNormGrad = _temporalNorm.Backward(temporalGrad);
        var afterSpatialGrad = AddTensors(afterTemporalGrad, temporalNormGrad);

        // Backward through spatial attention residual: afterSpatial = input + spatial(norm(input))
        var spatialGrad = _spatialAttention.Backward(afterSpatialGrad);
        var spatialNormGrad = _spatialNorm.Backward(spatialGrad);
        return AddTensors(afterSpatialGrad, spatialNormGrad);
    }

    /// <inheritdoc />
    public override void UpdateParameters(T learningRate)
    {
        _spatialAttention.UpdateParameters(learningRate);
        _temporalAttention.UpdateParameters(learningRate);
        _crossAttention.UpdateParameters(learningRate);
        _ffnIn.UpdateParameters(learningRate);
        _ffnOut.UpdateParameters(learningRate);
        _spatialNorm.UpdateParameters(learningRate);
        _temporalNorm.UpdateParameters(learningRate);
        _crossNorm.UpdateParameters(learningRate);
        _ffnNorm.UpdateParameters(learningRate);
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var parts = new LayerBase<T>[]
        {
            _spatialAttention, _temporalAttention, _crossAttention,
            _ffnIn, _ffnOut,
            _spatialNorm, _temporalNorm, _crossNorm, _ffnNorm
        };

        int total = 0;
        foreach (var p in parts) total += p.GetParameters().Length;

        var combined = new Vector<T>(total);
        int offset = 0;
        foreach (var p in parts)
        {
            var parms = p.GetParameters();
            for (int i = 0; i < parms.Length; i++)
                combined[offset + i] = parms[i];
            offset += parms.Length;
        }
        return combined;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        var parts = new LayerBase<T>[]
        {
            _spatialAttention, _temporalAttention, _crossAttention,
            _ffnIn, _ffnOut,
            _spatialNorm, _temporalNorm, _crossNorm, _ffnNorm
        };

        int offset = 0;
        foreach (var layer in parts)
        {
            int count = layer.GetParameters().Length;
            var sub = new Vector<T>(count);
            for (int i = 0; i < count; i++)
                sub[i] = parameters[offset + i];
            layer.SetParameters(sub);
            offset += count;
        }
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        _lastInput = null;
        _afterSpatial = null;
        _afterTemporal = null;
        _afterCross = null;
        _spatialAttention.ResetState();
        _temporalAttention.ResetState();
        _crossAttention.ResetState();
        _ffnIn.ResetState();
        _ffnOut.ResetState();
        _spatialNorm.ResetState();
        _temporalNorm.ResetState();
        _crossNorm.ResetState();
        _ffnNorm.ResetState();
    }

    /// <inheritdoc />
    public override bool SupportsJitCompilation => false;

    /// <inheritdoc />
    public override Autodiff.ComputationNode<T> ExportComputationGraph(List<Autodiff.ComputationNode<T>> inputNodes)
    {
        throw new NotSupportedException("STDiTBlock does not support JIT compilation.");
    }
}
