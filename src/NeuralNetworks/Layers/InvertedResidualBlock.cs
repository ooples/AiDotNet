using AiDotNet.ActivationFunctions;
using AiDotNet.Autodiff;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Implements an Inverted Residual Block (MBConv) used in MobileNetV2 and MobileNetV3.
/// </summary>
/// <remarks>
/// <para>
/// The Inverted Residual Block is the core building block for MobileNet architectures.
/// Unlike traditional residual blocks that narrow then widen, inverted residual blocks
/// expand then narrow, hence the name "inverted."
/// </para>
/// <para>
/// Architecture:
/// <code>
/// Input ────────────────────────────────────────────────────────────────┐
///   │                                                                   │
///   └─► [Expand 1x1 ─► BN ─► Act] ─► DW 3x3 ─► BN ─► Act ─► [SE] ─►    │
///                                                              │        │
///                                           Project 1x1 ─► BN ─┘─► (+) ─► Output
///                                                                    ↑
///                                                             (if skip connection)
/// </code>
/// </para>
/// <para>
/// <b>For Beginners:</b> The Inverted Residual Block is designed for efficient mobile inference.
///
/// Key innovations:
/// - Expansion: First EXPANDS the channels (opposite of traditional bottlenecks)
/// - Depthwise separable convolution: Filters each channel independently, then mixes
/// - Linear bottleneck: The final projection has NO activation (preserves information)
/// - Skip connection: Only when input and output dimensions match
///
/// This design reduces computational cost while maintaining model accuracy.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class InvertedResidualBlock<T> : LayerBase<T>, IChainableComputationGraph<T>
{
    private readonly ConvolutionalLayer<T>? _expandConv;
    private readonly BatchNormalizationLayer<T>? _expandBn;
    private readonly ConvolutionalLayer<T> _dwConv;
    private readonly BatchNormalizationLayer<T> _dwBn;
    private readonly SqueezeAndExcitationLayer<T>? _se;
    private readonly ConvolutionalLayer<T> _projectConv;
    private readonly BatchNormalizationLayer<T> _projectBn;

    private readonly IActivationFunction<T> _activation;
    private readonly bool _useResidual;
    private readonly bool _hasExpansion;
    private readonly bool _useSE;

    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastExpandOut;
    private Tensor<T>? _lastExpandBnOut;
    private Tensor<T>? _lastExpandActOut;
    private Tensor<T>? _lastDwOut;
    private Tensor<T>? _lastDwBnOut;
    private Tensor<T>? _lastDwActOut;
    private Tensor<T>? _lastSeOut;
    private Tensor<T>? _lastProjectOut;
    private Tensor<T>? _lastProjectBnOut;

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets the number of input channels.
    /// </summary>
    public int InChannels { get; }

    /// <summary>
    /// Gets the number of output channels.
    /// </summary>
    public int OutChannels { get; }

    /// <summary>
    /// Gets the expansion ratio.
    /// </summary>
    public int ExpansionRatio { get; }

    /// <summary>
    /// Gets the stride used in the depthwise convolution.
    /// </summary>
    public int Stride { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="InvertedResidualBlock{T}"/> class.
    /// </summary>
    /// <param name="inChannels">The number of input channels.</param>
    /// <param name="outChannels">The number of output channels.</param>
    /// <param name="inputHeight">The height of the input feature map.</param>
    /// <param name="inputWidth">The width of the input feature map.</param>
    /// <param name="expansionRatio">The expansion ratio for the hidden layer (default: 6).</param>
    /// <param name="stride">The stride for the depthwise convolution (default: 1).</param>
    /// <param name="useSE">Whether to use Squeeze-and-Excitation (for MobileNetV3).</param>
    /// <param name="seRatio">The reduction ratio for SE block (default: 4).</param>
    /// <param name="activation">The activation function to use. Default is ReLU6.</param>
    public InvertedResidualBlock(
        int inChannels,
        int outChannels,
        int inputHeight,
        int inputWidth,
        int expansionRatio = 6,
        int stride = 1,
        bool useSE = false,
        int seRatio = 4,
        IActivationFunction<T>? activationFunction = null)
        : base(
            inputShape: [inChannels, inputHeight, inputWidth],
            outputShape: [outChannels, (inputHeight + stride - 1) / stride, (inputWidth + stride - 1) / stride])
    {
        InChannels = inChannels;
        OutChannels = outChannels;
        ExpansionRatio = expansionRatio;
        Stride = stride;

        int hiddenDim = inChannels * expansionRatio;
        _activation = activationFunction ?? new ReLU6Activation<T>();
        _hasExpansion = expansionRatio != 1;
        _useSE = useSE;

        // Skip connection only when stride=1 and input/output channels match
        _useResidual = stride == 1 && inChannels == outChannels;

        int currentHeight = inputHeight;
        int currentWidth = inputWidth;

        // Expansion layer (1x1 conv) - only if expansion ratio > 1
        if (_hasExpansion)
        {
            _expandConv = new ConvolutionalLayer<T>(
                inputDepth: inChannels,
                outputDepth: hiddenDim,
                kernelSize: 1,
                inputHeight: currentHeight,
                inputWidth: currentWidth,
                stride: 1,
                padding: 0,
                activationFunction: new IdentityActivation<T>());

            _expandBn = new BatchNormalizationLayer<T>(hiddenDim);
        }

        // Calculate output dimensions after depthwise conv
        int dwOutputHeight = (currentHeight + 2 * 1 - 3) / stride + 1; // padding=1, kernel=3
        int dwOutputWidth = (currentWidth + 2 * 1 - 3) / stride + 1;

        // Depthwise separable convolution (3x3)
        // Note: We use a regular conv with groups=channels for depthwise
        // Here we use ConvolutionalLayer configured for depthwise operation
        int dwInputChannels = _hasExpansion ? hiddenDim : inChannels;
        _dwConv = new ConvolutionalLayer<T>(
            inputDepth: dwInputChannels,
            outputDepth: dwInputChannels, // Same as input for depthwise
            kernelSize: 3,
            inputHeight: currentHeight,
            inputWidth: currentWidth,
            stride: stride,
            padding: 1,
            activationFunction: new IdentityActivation<T>());

        _dwBn = new BatchNormalizationLayer<T>(dwInputChannels);

        // Squeeze-and-Excitation block (optional, for MobileNetV3)
        if (_useSE)
        {
            _se = new SqueezeAndExcitationLayer<T>(
                dwInputChannels,
                seRatio,
                firstActivation: (IActivationFunction<T>?)null,
                secondActivation: (IActivationFunction<T>?)null);
        }

        // Projection layer (1x1 conv) - LINEAR (no activation)
        _projectConv = new ConvolutionalLayer<T>(
            inputDepth: dwInputChannels,
            outputDepth: outChannels,
            kernelSize: 1,
            inputHeight: dwOutputHeight,
            inputWidth: dwOutputWidth,
            stride: 1,
            padding: 0,
            activationFunction: new IdentityActivation<T>());

        _projectBn = new BatchNormalizationLayer<T>(outChannels);
    }

    /// <summary>
    /// Performs the forward pass of the Inverted Residual Block.
    /// </summary>
    /// <param name="input">The input tensor [C, H, W] or [B, C, H, W].</param>
    /// <returns>The output tensor after the inverted residual computation.</returns>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        Tensor<T> x = input;

        // Expansion phase (if expansion > 1)
        if (_hasExpansion && _expandConv is not null && _expandBn is not null)
        {
            _lastExpandOut = _expandConv.Forward(x);
            _lastExpandBnOut = _expandBn.Forward(_lastExpandOut);
            _lastExpandActOut = ApplyBlockActivation(_lastExpandBnOut);
            x = _lastExpandActOut;
        }

        // Depthwise convolution phase
        _lastDwOut = _dwConv.Forward(x);
        _lastDwBnOut = _dwBn.Forward(_lastDwOut);
        _lastDwActOut = ApplyBlockActivation(_lastDwBnOut);
        x = _lastDwActOut;

        // Squeeze-and-Excitation phase (optional)
        // Note: SE layer expects NHWC format, but our tensors are in NCHW format
        if (_useSE && _se is not null)
        {
            // Transpose NCHW [B, C, H, W] -> NHWC [B, H, W, C]
            var seInput = TransposeNCHWToNHWC(x);
            var seOutput = _se.Forward(seInput);
            // Transpose NHWC [B, H, W, C] -> NCHW [B, C, H, W]
            _lastSeOut = TransposeNHWCToNCHW(seOutput);
            x = _lastSeOut;
        }

        // Projection phase (LINEAR - no activation)
        _lastProjectOut = _projectConv.Forward(x);
        _lastProjectBnOut = _projectBn.Forward(_lastProjectOut);

        // Residual connection (only if dimensions match)
        if (_useResidual)
        {
            return AddTensors(_lastProjectBnOut, input);
        }

        return _lastProjectBnOut;
    }

    /// <summary>
    /// Performs the backward pass of the Inverted Residual Block.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the output.</param>
    /// <returns>The gradient of the loss with respect to the input.</returns>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput is null || _lastDwOut is null || _lastDwBnOut is null ||
            _lastProjectOut is null || _lastProjectBnOut is null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass.");
        }

        Tensor<T> grad = outputGradient;
        Tensor<T>? gradToInput = null;

        // If we have a residual connection, gradient flows to both paths
        if (_useResidual)
        {
            gradToInput = grad;
        }

        // Backward through projection (no activation)
        var gradProjectBn = _projectBn.Backward(grad);
        var gradProjectConv = _projectConv.Backward(gradProjectBn);
        grad = gradProjectConv;

        // Backward through SE (if used)
        // Note: SE layer expects NHWC format, so transpose gradients
        if (_useSE && _se is not null && _lastDwActOut is not null)
        {
            // Transpose gradient from NCHW to NHWC for SE backward
            var gradNHWC = TransposeNCHWToNHWC(grad);
            var seGrad = _se.Backward(gradNHWC);
            // Transpose gradient back from NHWC to NCHW
            grad = TransposeNHWCToNCHW(seGrad);
        }

        // Backward through depthwise conv activation
        grad = ApplyBlockActivationDerivative(_lastDwBnOut, grad);

        // Backward through depthwise conv BN and conv
        var gradDwBn = _dwBn.Backward(grad);
        var gradDwConv = _dwConv.Backward(gradDwBn);
        grad = gradDwConv;

        // Backward through expansion (if used)
        if (_hasExpansion && _expandConv is not null && _expandBn is not null && _lastExpandBnOut is not null)
        {
            grad = ApplyBlockActivationDerivative(_lastExpandBnOut, grad);
            var gradExpandBn = _expandBn.Backward(grad);
            grad = _expandConv.Backward(gradExpandBn);
        }

        // Combine gradients if we have residual connection
        if (gradToInput is not null)
        {
            return AddTensors(grad, gradToInput);
        }

        return grad;
    }

    /// <summary>
    /// Updates the parameters of all sub-layers.
    /// </summary>
    /// <param name="learningRate">The learning rate for parameter updates.</param>
    public override void UpdateParameters(T learningRate)
    {
        _expandConv?.UpdateParameters(learningRate);
        _expandBn?.UpdateParameters(learningRate);
        _dwConv.UpdateParameters(learningRate);
        _dwBn.UpdateParameters(learningRate);
        _se?.UpdateParameters(learningRate);
        _projectConv.UpdateParameters(learningRate);
        _projectBn.UpdateParameters(learningRate);
    }

    /// <summary>
    /// Gets all trainable parameters from the block.
    /// </summary>
    /// <returns>A vector containing all trainable parameters.</returns>
    public override Vector<T> GetParameters()
    {
        var parameters = new List<T>();

        if (_expandConv is not null)
            parameters.AddRange(_expandConv.GetParameters().ToArray());
        if (_expandBn is not null)
            parameters.AddRange(_expandBn.GetParameters().ToArray());

        parameters.AddRange(_dwConv.GetParameters().ToArray());
        parameters.AddRange(_dwBn.GetParameters().ToArray());

        if (_se is not null)
            parameters.AddRange(_se.GetParameters().ToArray());

        parameters.AddRange(_projectConv.GetParameters().ToArray());
        parameters.AddRange(_projectBn.GetParameters().ToArray());

        return new Vector<T>(parameters.ToArray());
    }

    /// <summary>
    /// Sets all trainable parameters from the given parameter vector.
    /// </summary>
    /// <param name="parameters">The parameter vector containing all layer parameters.</param>
    public override void SetParameters(Vector<T> parameters)
    {
        int offset = 0;

        if (_expandConv is not null)
        {
            int count = _expandConv.GetParameters().Length;
            _expandConv.SetParameters(parameters.SubVector(offset, count));
            offset += count;
        }
        if (_expandBn is not null)
        {
            int count = _expandBn.GetParameters().Length;
            _expandBn.SetParameters(parameters.SubVector(offset, count));
            offset += count;
        }

        {
            int count = _dwConv.GetParameters().Length;
            _dwConv.SetParameters(parameters.SubVector(offset, count));
            offset += count;
        }
        {
            int count = _dwBn.GetParameters().Length;
            _dwBn.SetParameters(parameters.SubVector(offset, count));
            offset += count;
        }

        if (_se is not null)
        {
            int count = _se.GetParameters().Length;
            _se.SetParameters(parameters.SubVector(offset, count));
            offset += count;
        }

        {
            int count = _projectConv.GetParameters().Length;
            _projectConv.SetParameters(parameters.SubVector(offset, count));
            offset += count;
        }
        {
            int count = _projectBn.GetParameters().Length;
            _projectBn.SetParameters(parameters.SubVector(offset, count));
            // offset not incremented since this is the last parameter block
        }
    }

    /// <summary>
    /// Resets the internal state of the block.
    /// </summary>
    public override void ResetState()
    {
        _lastInput = null;
        _lastExpandOut = null;
        _lastExpandBnOut = null;
        _lastExpandActOut = null;
        _lastDwOut = null;
        _lastDwBnOut = null;
        _lastDwActOut = null;
        _lastSeOut = null;
        _lastProjectOut = null;
        _lastProjectBnOut = null;

        _expandConv?.ResetState();
        _expandBn?.ResetState();
        _dwConv.ResetState();
        _dwBn.ResetState();
        _se?.ResetState();
        _projectConv.ResetState();
        _projectBn.ResetState();
    }

    /// <summary>
    /// Gets whether this block supports JIT compilation.
    /// </summary>
    public override bool SupportsJitCompilation
    {
        get
        {
            // Check all required sub-layers support JIT
            if (_hasExpansion)
            {
                if (_expandConv is null || !_expandConv.SupportsJitCompilation ||
                    _expandBn is null || !_expandBn.SupportsJitCompilation)
                    return false;
            }

            if (!_dwConv.SupportsJitCompilation || !_dwBn.SupportsJitCompilation)
                return false;

            if (_useSE && _se is not null && !_se.SupportsJitCompilation)
                return false;

            if (!_projectConv.SupportsJitCompilation || !_projectBn.SupportsJitCompilation)
                return false;

            // Check activation supports JIT
            if (!_activation.SupportsJitCompilation)
                return false;

            return true;
        }
    }

    /// <summary>
    /// Exports the computation graph for JIT compilation.
    /// </summary>
    /// <param name="inputNodes">List to populate with input computation nodes.</param>
    /// <returns>The output computation node.</returns>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes is null)
        {
            throw new ArgumentNullException(nameof(inputNodes));
        }

        if (InputShape is null || InputShape.Length == 0)
        {
            throw new InvalidOperationException("Layer input shape not configured.");
        }

        // Create symbolic input node with batch dimension [B, C, H, W]
        var symbolicInput = new Tensor<T>(new int[] { 1 }.Concat(InputShape).ToArray());
        var inputNode = TensorOperations<T>.Variable(symbolicInput, "input");
        inputNodes.Add(inputNode);

        return BuildComputationGraph(inputNode, "");
    }

    /// <inheritdoc />
    public ComputationNode<T> BuildComputationGraph(ComputationNode<T> inputNode, string namePrefix)
    {
        var current = inputNode;

        // Expansion phase (if expansion > 1)
        if (_hasExpansion && _expandConv is not null && _expandBn is not null)
        {
            // Conv1x1 expansion
            var expandBiases = _expandConv.GetBiases();
            current = TensorOperations<T>.Conv2D(
                current,
                TensorOperations<T>.Constant(_expandConv.GetFilters(), $"{namePrefix}expand_kernel"),
                expandBiases is not null ? TensorOperations<T>.Constant(expandBiases, $"{namePrefix}expand_bias") : null,
                stride: new int[] { _expandConv.Stride, _expandConv.Stride },
                padding: new int[] { _expandConv.Padding, _expandConv.Padding });

            // BN
            current = TensorOperations<T>.BatchNorm(
                current,
                gamma: TensorOperations<T>.Constant(_expandBn.GetGamma(), $"{namePrefix}expand_bn_gamma"),
                beta: TensorOperations<T>.Constant(_expandBn.GetBeta(), $"{namePrefix}expand_bn_beta"),
                runningMean: _expandBn.GetRunningMean(),
                runningVar: _expandBn.GetRunningVariance(),
                training: false,
                epsilon: NumOps.ToDouble(_expandBn.GetEpsilon()));

            // Apply activation using proper JIT graph integration
            current = _activation.ApplyToGraph(current);
        }

        // Depthwise convolution phase
        {
            var dwBiases = _dwConv.GetBiases();
            current = TensorOperations<T>.Conv2D(
                current,
                TensorOperations<T>.Constant(_dwConv.GetFilters(), $"{namePrefix}dw_kernel"),
                dwBiases is not null ? TensorOperations<T>.Constant(dwBiases, $"{namePrefix}dw_bias") : null,
                stride: new int[] { _dwConv.Stride, _dwConv.Stride },
                padding: new int[] { _dwConv.Padding, _dwConv.Padding });

            // BN
            current = TensorOperations<T>.BatchNorm(
                current,
                gamma: TensorOperations<T>.Constant(_dwBn.GetGamma(), $"{namePrefix}dw_bn_gamma"),
                beta: TensorOperations<T>.Constant(_dwBn.GetBeta(), $"{namePrefix}dw_bn_beta"),
                runningMean: _dwBn.GetRunningMean(),
                runningVar: _dwBn.GetRunningVariance(),
                training: false,
                epsilon: NumOps.ToDouble(_dwBn.GetEpsilon()));

            // Apply activation using proper JIT graph integration
            current = _activation.ApplyToGraph(current);
        }

        // Squeeze-and-Excitation phase (if used)
        // Note: SE layer expects NHWC format, so we need to transpose
        if (_useSE && _se is not null)
        {
            // Transpose NCHW [B, C, H, W] -> NHWC [B, H, W, C]
            current = TensorOperations<T>.Permute(current, new int[] { 0, 2, 3, 1 });

            // Apply SE layer
            current = _se.BuildComputationGraph(current, $"{namePrefix}se_");

            // Transpose NHWC [B, H, W, C] -> NCHW [B, C, H, W]
            current = TensorOperations<T>.Permute(current, new int[] { 0, 3, 1, 2 });
        }

        // Projection phase (LINEAR - no activation)
        {
            var projectBiases = _projectConv.GetBiases();
            current = TensorOperations<T>.Conv2D(
                current,
                TensorOperations<T>.Constant(_projectConv.GetFilters(), $"{namePrefix}project_kernel"),
                projectBiases is not null ? TensorOperations<T>.Constant(projectBiases, $"{namePrefix}project_bias") : null,
                stride: new int[] { _projectConv.Stride, _projectConv.Stride },
                padding: new int[] { _projectConv.Padding, _projectConv.Padding });

            // BN only, no activation (linear bottleneck)
            current = TensorOperations<T>.BatchNorm(
                current,
                gamma: TensorOperations<T>.Constant(_projectBn.GetGamma(), $"{namePrefix}project_bn_gamma"),
                beta: TensorOperations<T>.Constant(_projectBn.GetBeta(), $"{namePrefix}project_bn_beta"),
                runningMean: _projectBn.GetRunningMean(),
                runningVar: _projectBn.GetRunningVariance(),
                training: false,
                epsilon: NumOps.ToDouble(_projectBn.GetEpsilon()));
        }

        // Residual connection (only if dimensions match)
        if (_useResidual)
        {
            current = TensorOperations<T>.Add(current, inputNode);
        }

        return current;
    }

    #region Helper Methods

    private Tensor<T> ApplyBlockActivation(Tensor<T> input)
    {
        return _activation.Activate(input);
    }

    private Tensor<T> ApplyBlockActivationDerivative(Tensor<T> input, Tensor<T> gradient)
    {
        var derivative = _activation.Derivative(input);
        return MultiplyTensors(gradient, derivative);
    }

    /// <summary>
    /// Multiplies two tensors element-wise using vectorized Engine operations.
    /// </summary>
    private Tensor<T> MultiplyTensors(Tensor<T> a, Tensor<T> b)
    {
        return Engine.TensorMultiply(a, b);
    }

    /// <summary>
    /// Adds two tensors element-wise using vectorized Engine operations.
    /// </summary>
    private Tensor<T> AddTensors(Tensor<T> a, Tensor<T> b)
    {
        return Engine.TensorAdd(a, b);
    }

    /// <summary>
    /// Transposes a tensor from NCHW to NHWC format.
    /// Supports both 3D [C, H, W] and 4D [B, C, H, W] inputs.
    /// Uses vectorized Engine.TensorPermute operation.
    /// </summary>
    private Tensor<T> TransposeNCHWToNHWC(Tensor<T> input)
    {
        int rank = input.Shape.Length;
        if (rank == 3)
        {
            // 3D input [C, H, W] -> add batch dimension -> [1, C, H, W]
            var input4D = input.Reshape([1, input.Shape[0], input.Shape[1], input.Shape[2]]);
            // NCHW [0, 1, 2, 3] -> NHWC [0, 2, 3, 1]
            var result4D = Engine.TensorPermute(input4D, [0, 2, 3, 1]);
            // Remove batch dimension -> [H, W, C]
            return result4D.Reshape([result4D.Shape[1], result4D.Shape[2], result4D.Shape[3]]);
        }
        else if (rank == 4)
        {
            // 4D input: NCHW [0, 1, 2, 3] -> NHWC [0, 2, 3, 1]
            return Engine.TensorPermute(input, [0, 2, 3, 1]);
        }
        else
        {
            // Higher rank: flatten leading dims, transpose, restore
            int flatBatch = 1;
            for (int d = 0; d < rank - 3; d++)
                flatBatch *= input.Shape[d];
            var input4D = input.Reshape([flatBatch, input.Shape[rank - 3], input.Shape[rank - 2], input.Shape[rank - 1]]);
            var result4D = Engine.TensorPermute(input4D, [0, 2, 3, 1]);
            var outputShape = new int[rank];
            for (int d = 0; d < rank - 3; d++)
                outputShape[d] = input.Shape[d];
            outputShape[rank - 3] = result4D.Shape[1];
            outputShape[rank - 2] = result4D.Shape[2];
            outputShape[rank - 1] = result4D.Shape[3];
            return result4D.Reshape(outputShape);
        }
    }

    /// <summary>
    /// Transposes a tensor from NHWC to NCHW format.
    /// Supports both 3D [H, W, C] and 4D [B, H, W, C] inputs.
    /// Uses vectorized Engine.TensorPermute operation.
    /// </summary>
    private Tensor<T> TransposeNHWCToNCHW(Tensor<T> input)
    {
        int rank = input.Shape.Length;
        if (rank == 3)
        {
            // 3D input [H, W, C] -> add batch dimension -> [1, H, W, C]
            var input4D = input.Reshape([1, input.Shape[0], input.Shape[1], input.Shape[2]]);
            // NHWC [0, 1, 2, 3] -> NCHW [0, 3, 1, 2]
            var result4D = Engine.TensorPermute(input4D, [0, 3, 1, 2]);
            // Remove batch dimension -> [C, H, W]
            return result4D.Reshape([result4D.Shape[1], result4D.Shape[2], result4D.Shape[3]]);
        }
        else if (rank == 4)
        {
            // 4D input: NHWC [0, 1, 2, 3] -> NCHW [0, 3, 1, 2]
            return Engine.TensorPermute(input, [0, 3, 1, 2]);
        }
        else
        {
            // Higher rank: flatten leading dims, transpose, restore
            int flatBatch = 1;
            for (int d = 0; d < rank - 3; d++)
                flatBatch *= input.Shape[d];
            var input4D = input.Reshape([flatBatch, input.Shape[rank - 3], input.Shape[rank - 2], input.Shape[rank - 1]]);
            var result4D = Engine.TensorPermute(input4D, [0, 3, 1, 2]);
            var outputShape = new int[rank];
            for (int d = 0; d < rank - 3; d++)
                outputShape[d] = input.Shape[d];
            outputShape[rank - 3] = result4D.Shape[1];
            outputShape[rank - 2] = result4D.Shape[2];
            outputShape[rank - 1] = result4D.Shape[3];
            return result4D.Reshape(outputShape);
        }
    }

    #endregion

    /// <summary>
    /// Returns layer-specific metadata for serialization purposes.
    /// </summary>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["InChannels"] = InChannels.ToString();
        metadata["OutChannels"] = OutChannels.ToString();
        metadata["ExpansionRatio"] = ExpansionRatio.ToString();
        metadata["Stride"] = Stride.ToString();
        metadata["UseSE"] = _useSE.ToString();
        return metadata;
    }
}
