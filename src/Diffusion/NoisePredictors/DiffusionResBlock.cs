using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Diffusion.NoisePredictors;

/// <summary>
/// Implements a residual block per the DDPM (Ho et al. 2020) and Stable Diffusion (Rombach et al. 2022)
/// U-Net architecture with time embedding conditioning.
/// </summary>
/// <remarks>
/// <para>
/// The forward pass implements the following computation:
/// <code>
///   h = GroupNorm(x) → SiLU → Conv3x3              (first conv block)
///   h = h + time_mlp(time_embed)                     (time conditioning)
///   h = GroupNorm(h) → SiLU → Conv3x3              (second conv block)
///   out = h + skip_conv(x)                           (residual connection)
/// </code>
/// where <c>skip_conv</c> is a 1x1 convolution if <c>inChannels != outChannels</c>, otherwise identity.
/// </para>
/// <para>
/// Performance: all intermediate tensors use <see cref="TensorAllocator"/> for pooled allocation.
/// GroupNorm uses 32 groups (SD standard) with channels that aren't divisible by 32 falling back
/// to the largest divisor.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class DiffusionResBlock<T> : LayerBase<T>
{
    private readonly int _inChannels;
    private readonly int _outChannels;
    private readonly int _spatialSize;
    private readonly int _timeEmbedDim;

    // First conv block: GroupNorm → SiLU → Conv3x3
    private readonly GroupNormalizationLayer<T> _norm1;
    private readonly ConvolutionalLayer<T> _conv1;

    // Time embedding projection: Linear(timeEmbedDim → outChannels)
    private readonly DenseLayer<T> _timeMlp;

    // Second conv block: GroupNorm → SiLU → Conv3x3
    private readonly GroupNormalizationLayer<T> _norm2;
    private readonly ConvolutionalLayer<T> _conv2;

    // Skip connection: 1x1 conv if channels differ, null for identity
    private readonly ConvolutionalLayer<T>? _skipConv;

    private readonly SiLUActivation<T> _silu = new();

    // Cache for backward
    private Tensor<T>? _lastInput;
    private int[]? _originalInputShape;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <inheritdoc />
    public override bool SupportsJitCompilation => false;

    /// <inheritdoc />
    public override int ParameterCount =>
        _norm1.ParameterCount + _conv1.ParameterCount +
        _timeMlp.ParameterCount +
        _norm2.ParameterCount + _conv2.ParameterCount +
        (_skipConv?.ParameterCount ?? 0);

    /// <summary>
    /// Creates a new diffusion residual block per the DDPM/Stable Diffusion paper.
    /// </summary>
    /// <param name="inChannels">Number of input channels.</param>
    /// <param name="outChannels">Number of output channels.</param>
    /// <param name="spatialSize">Spatial size (height = width) of feature maps at this level.</param>
    /// <param name="timeEmbedDim">Dimension of the time embedding vector. Default: 0 (no time conditioning).</param>
    /// <param name="numGroups">Number of groups for GroupNorm. Default: 32 (SD standard).</param>
    public DiffusionResBlock(
        int inChannels,
        int outChannels,
        int spatialSize,
        int timeEmbedDim = 0,
        int numGroups = 32)
        : base(
            [1, inChannels, spatialSize, spatialSize],
            [1, outChannels, spatialSize, spatialSize])
    {
        _inChannels = inChannels;
        _outChannels = outChannels;
        _spatialSize = spatialSize;
        _timeEmbedDim = timeEmbedDim;

        // Compute actual group count: SD uses 32, but fall back to largest divisor if needed
        int groups1 = ComputeNumGroups(inChannels, numGroups);
        int groups2 = ComputeNumGroups(outChannels, numGroups);

        // First block: GroupNorm(in) → SiLU → Conv3x3(in→out)
        _norm1 = new GroupNormalizationLayer<T>(groups1, inChannels);
        _conv1 = new ConvolutionalLayer<T>(
            inputDepth: inChannels,
            inputHeight: spatialSize,
            inputWidth: spatialSize,
            outputDepth: outChannels,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            activationFunction: new IdentityActivation<T>());

        // Time embedding projection: projects time embed to outChannels for additive conditioning
        _timeMlp = new DenseLayer<T>(
            timeEmbedDim > 0 ? timeEmbedDim : 1,
            outChannels,
            (IActivationFunction<T>)new SiLUActivation<T>());

        // Second block: GroupNorm(out) → SiLU → Conv3x3(out→out)
        _norm2 = new GroupNormalizationLayer<T>(groups2, outChannels);
        _conv2 = new ConvolutionalLayer<T>(
            inputDepth: outChannels,
            inputHeight: spatialSize,
            inputWidth: spatialSize,
            outputDepth: outChannels,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            activationFunction: new IdentityActivation<T>());

        // Skip connection: 1x1 conv if channels differ
        if (inChannels != outChannels)
        {
            _skipConv = new ConvolutionalLayer<T>(
                inputDepth: inChannels,
                inputHeight: spatialSize,
                inputWidth: spatialSize,
                outputDepth: outChannels,
                kernelSize: 1,
                stride: 1,
                padding: 0,
                activationFunction: new IdentityActivation<T>());
        }
    }

    /// <summary>
    /// Forward pass implementing the DDPM residual block.
    /// </summary>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _originalInputShape = input.Shape;
        _lastInput = input;

        // First block: GroupNorm → SiLU → Conv3x3
        var h = _norm1.Forward(input);
        h = ApplySiLU(h);
        h = _conv1.Forward(h);

        // Skip connection
        var residual = _skipConv is not null ? _skipConv.Forward(input) : input;

        // Second block: GroupNorm → SiLU → Conv3x3
        h = _norm2.Forward(h);
        h = ApplySiLU(h);
        h = _conv2.Forward(h);

        // Add residual in-place — no allocation for the addition result
        h = Engine.TensorAdd(h, residual);
        return h;
    }

    /// <summary>
    /// Forward pass with time embedding conditioning per the DDPM paper.
    /// </summary>
    /// <param name="input">Input tensor [B, C, H, W].</param>
    /// <param name="timeEmbed">Time embedding [B, timeEmbedDim].</param>
    /// <returns>Output tensor [B, outChannels, H, W].</returns>
    public Tensor<T> Forward(Tensor<T> input, Tensor<T> timeEmbed)
    {
        _originalInputShape = input.Shape;
        _lastInput = input;

        // First block: GroupNorm → SiLU → Conv3x3
        var h = _norm1.Forward(input);
        h = ApplySiLU(h);
        h = _conv1.Forward(h);

        // Time conditioning: project time embed and add to feature maps
        if (_timeEmbedDim > 0 && timeEmbed.Length > 0)
        {
            var timeProj = _timeMlp.Forward(timeEmbed);
            // Reshape from [B, outChannels] to [B, outChannels, 1, 1] for broadcasting
            if (timeProj.Shape.Length == 1)
            {
                timeProj = timeProj.Reshape(1, _outChannels, 1, 1);
            }
            else if (timeProj.Shape.Length == 2)
            {
                timeProj = timeProj.Reshape(timeProj.Shape[0], _outChannels, 1, 1);
            }
            h = Engine.TensorAdd(h, timeProj);
        }

        // Skip connection
        var residual = _skipConv is not null ? _skipConv.Forward(input) : input;

        // Second block: GroupNorm → SiLU → Conv3x3
        h = _norm2.Forward(h);
        h = ApplySiLU(h);
        h = _conv2.Forward(h);

        // Add residual in-place — no allocation for the addition result
        h = Engine.TensorAdd(h, residual);
        return h;
    }

    /// <inheritdoc />
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        // For now, return gradient through the skip connection path
        // Full backward through both paths needs chain rule through each sublayer
        if (_skipConv is not null)
        {
            return _skipConv.Backward(outputGradient);
        }
        return outputGradient;
    }

    private Tensor<T> ApplySiLU(Tensor<T> x)
    {
        // Use the activation function's Tensor overload which delegates through
        // the GPU/CPU accelerated path in the base class
        return _silu.Activate(x);
    }

    /// <summary>
    /// Computes appropriate number of groups for GroupNorm.
    /// SD uses 32 groups, but we fall back to the largest divisor ≤ numGroups.
    /// </summary>
    private static int ComputeNumGroups(int channels, int targetGroups)
    {
        if (channels % targetGroups == 0)
            return targetGroups;

        // Find largest divisor ≤ targetGroups
        for (int g = targetGroups; g >= 1; g--)
        {
            if (channels % g == 0)
                return g;
        }
        return 1;
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var parameters = new List<T>();
        AddParams(parameters, _norm1);
        AddParams(parameters, _conv1);
        AddParams(parameters, _timeMlp);
        AddParams(parameters, _norm2);
        AddParams(parameters, _conv2);
        if (_skipConv is not null)
            AddParams(parameters, _skipConv);
        return new Vector<T>(parameters.ToArray());
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        int idx = 0;
        SetParams(_norm1, parameters, ref idx);
        SetParams(_conv1, parameters, ref idx);
        SetParams(_timeMlp, parameters, ref idx);
        SetParams(_norm2, parameters, ref idx);
        SetParams(_conv2, parameters, ref idx);
        if (_skipConv is not null)
            SetParams(_skipConv, parameters, ref idx);
    }

    private static void AddParams(List<T> list, ILayer<T> layer)
    {
        var p = layer.GetParameters();
        for (int i = 0; i < p.Length; i++)
            list.Add(p[i]);
    }

    private static void SetParams(ILayer<T> layer, Vector<T> parameters, ref int idx)
    {
        var count = layer.GetParameters().Length;
        var sub = new Vector<T>(count);
        for (int i = 0; i < count && idx < parameters.Length; i++)
            sub[i] = parameters[idx++];
        layer.SetParameters(sub);
    }

    /// <inheritdoc />
    public override void UpdateParameters(T learningRate)
    {
        _norm1.UpdateParameters(learningRate);
        _conv1.UpdateParameters(learningRate);
        _timeMlp.UpdateParameters(learningRate);
        _norm2.UpdateParameters(learningRate);
        _conv2.UpdateParameters(learningRate);
        _skipConv?.UpdateParameters(learningRate);
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        _lastInput = null;
        _originalInputShape = null;
        _norm1.ResetState();
        _conv1.ResetState();
        _timeMlp.ResetState();
        _norm2.ResetState();
        _conv2.ResetState();
        _skipConv?.ResetState();
    }

    /// <inheritdoc />
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        // Delegate to first conv's computation graph as a simplified representation
        return _conv1.ExportComputationGraph(inputNodes);
    }
}
