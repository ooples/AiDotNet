using AiDotNet.ActivationFunctions;
using AiDotNet.Engines;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Implements the BottleneckBlock used in ResNet50, ResNet101, and ResNet152 architectures.
/// </summary>
/// <remarks>
/// <para>
/// The BottleneckBlock uses a 1x1-3x3-1x1 convolution pattern, where the 1x1 layers reduce and then
/// restore dimensions (with expansion), and the 3x3 layer is the bottleneck with smaller channels.
/// This design is more computationally efficient than stacking 3x3 convolutions for deep networks.
/// </para>
/// <para>
/// <b>Architecture:</b>
/// <code>
/// Input ─┬─ Conv1x1 ─ BN ─ ReLU ─ Conv3x3 ─ BN ─ ReLU ─ Conv1x1 ─ BN ─┬─ (+) ─ ReLU ─ Output
///        │                                                             │
///        └────────────────────── [Downsample?] ────────────────────────┘
/// </code>
/// The first 1x1 conv reduces channels, the 3x3 processes at reduced channels,
/// and the final 1x1 expands channels by a factor of 4.
/// </para>
/// <para>
/// <b>For Beginners:</b> The BottleneckBlock is like a compressed processing pipeline.
///
/// Think of it as:
/// 1. First 1x1 conv: "Compress" - reduce the number of channels (like compressing a file)
/// 2. 3x3 conv: "Process" - do the heavy computation on the compressed representation
/// 3. Second 1x1 conv: "Expand" - restore and expand the channels
///
/// This is more efficient because:
/// - The expensive 3x3 convolution works on fewer channels
/// - The overall result has high capacity (4x expansion)
/// - Much fewer parameters than three 3x3 convolutions
///
/// The expansion factor of 4 means if the base channels is 64, the output will have 256 channels.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class BottleneckBlock<T> : LayerBase<T>
{
    /// <summary>
    /// The expansion factor for BottleneckBlock. Output channels = base channels * 4.
    /// </summary>
    public const int Expansion = 4;

    private readonly ConvolutionalLayer<T> _conv1;  // 1x1 reduce
    private readonly BatchNormalizationLayer<T> _bn1;
    private readonly ConvolutionalLayer<T> _conv2;  // 3x3 bottleneck
    private readonly BatchNormalizationLayer<T> _bn2;
    private readonly ConvolutionalLayer<T> _conv3;  // 1x1 expand
    private readonly BatchNormalizationLayer<T> _bn3;
    private readonly ConvolutionalLayer<T>? _downsampleConv;
    private readonly BatchNormalizationLayer<T>? _downsampleBn;
    private readonly IActivationFunction<T> _relu;
    private readonly bool _hasDownsample;

    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastConv1Output;
    private Tensor<T>? _lastBn1Output;
    private Tensor<T>? _lastRelu1Output;
    private Tensor<T>? _lastConv2Output;
    private Tensor<T>? _lastBn2Output;
    private Tensor<T>? _lastRelu2Output;
    private Tensor<T>? _lastConv3Output;
    private Tensor<T>? _lastBn3Output;
    private Tensor<T>? _lastIdentity;
    private Tensor<T>? _lastPreActivation;

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new instance of the <see cref="BottleneckBlock{T}"/> class.
    /// </summary>
    /// <param name="inChannels">The number of input channels.</param>
    /// <param name="baseChannels">The base channel count (output will be baseChannels * 4).</param>
    /// <param name="stride">The stride for the 3x3 convolution (default: 1).</param>
    /// <param name="inputHeight">The input spatial height.</param>
    /// <param name="inputWidth">The input spatial width.</param>
    /// <param name="zeroInitResidual">If true, initialize the last BN to zero for better training stability.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The baseChannels parameter specifies the "bottleneck" width.
    /// The actual output channels will be baseChannels * 4 due to the expansion factor.
    /// For example, if baseChannels = 64, the output will have 256 channels.
    /// </para>
    /// </remarks>
    public BottleneckBlock(
        int inChannels,
        int baseChannels,
        int stride = 1,
        int inputHeight = 56,
        int inputWidth = 56,
        bool zeroInitResidual = true)
        : base(
            inputShape: [inChannels, inputHeight, inputWidth],
            outputShape: [baseChannels * Expansion, inputHeight / stride, inputWidth / stride])
    {
        int outChannels = baseChannels * Expansion;
        _relu = new ReLUActivation<T>();

        // First 1x1 conv: reduce channels to baseChannels
        _conv1 = new ConvolutionalLayer<T>(
            inputDepth: inChannels,
            inputHeight: inputHeight,
            inputWidth: inputWidth,
            outputDepth: baseChannels,
            kernelSize: 1,
            stride: 1,
            padding: 0,
            activation: new IdentityActivation<T>());

        _bn1 = new BatchNormalizationLayer<T>(baseChannels);

        // Second 3x3 conv: process at bottleneck width with stride
        _conv2 = new ConvolutionalLayer<T>(
            inputDepth: baseChannels,
            inputHeight: inputHeight,
            inputWidth: inputWidth,
            outputDepth: baseChannels,
            kernelSize: 3,
            stride: stride,
            padding: 1,
            activation: new IdentityActivation<T>());

        int outHeight = inputHeight / stride;
        int outWidth = inputWidth / stride;

        _bn2 = new BatchNormalizationLayer<T>(baseChannels);

        // Third 1x1 conv: expand channels to outChannels (baseChannels * 4)
        _conv3 = new ConvolutionalLayer<T>(
            inputDepth: baseChannels,
            inputHeight: outHeight,
            inputWidth: outWidth,
            outputDepth: outChannels,
            kernelSize: 1,
            stride: 1,
            padding: 0,
            activation: new IdentityActivation<T>());

        _bn3 = new BatchNormalizationLayer<T>(outChannels);

        // Zero-init residual: initialize last BN's gamma to zero so residual blocks
        // start as identity mappings, improving training stability
        if (zeroInitResidual)
        {
            _bn3.ZeroInitGamma();
        }

        // Downsample if dimensions change
        _hasDownsample = stride != 1 || inChannels != outChannels;
        if (_hasDownsample)
        {
            _downsampleConv = new ConvolutionalLayer<T>(
                inputDepth: inChannels,
                inputHeight: inputHeight,
                inputWidth: inputWidth,
                outputDepth: outChannels,
                kernelSize: 1,
                stride: stride,
                padding: 0,
                activation: new IdentityActivation<T>());

            _downsampleBn = new BatchNormalizationLayer<T>(outChannels);
        }
    }

    /// <summary>
    /// Performs the forward pass through the BottleneckBlock.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>The output tensor after the residual connection.</returns>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;

        // Main branch: conv1 -> bn1 -> relu -> conv2 -> bn2 -> relu -> conv3 -> bn3
        _lastConv1Output = _conv1.Forward(input);
        _lastBn1Output = _bn1.Forward(_lastConv1Output);
        _lastRelu1Output = ApplyReLU(_lastBn1Output);

        _lastConv2Output = _conv2.Forward(_lastRelu1Output);
        _lastBn2Output = _bn2.Forward(_lastConv2Output);
        _lastRelu2Output = ApplyReLU(_lastBn2Output);

        _lastConv3Output = _conv3.Forward(_lastRelu2Output);
        _lastBn3Output = _bn3.Forward(_lastConv3Output);

        // Identity/skip branch
        if (_hasDownsample && _downsampleConv is not null && _downsampleBn is not null)
        {
            var dsConvOut = _downsampleConv.Forward(input);
            _lastIdentity = _downsampleBn.Forward(dsConvOut);
        }
        else
        {
            _lastIdentity = input;
        }

        // Add residual connection
        _lastPreActivation = Engine.TensorAdd(_lastBn3Output, _lastIdentity);

        // Final ReLU
        return ApplyReLU(_lastPreActivation);
    }

    /// <summary>
    /// Performs the backward pass through the BottleneckBlock.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the output.</param>
    /// <returns>The gradient of the loss with respect to the input.</returns>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput is null || _lastPreActivation is null || _lastIdentity is null ||
            _lastBn3Output is null || _lastConv3Output is null || _lastRelu2Output is null ||
            _lastBn2Output is null || _lastConv2Output is null || _lastRelu1Output is null ||
            _lastBn1Output is null || _lastConv1Output is null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass.");
        }

        // Gradient through final ReLU
        var gradPreActivation = ApplyReLUDerivative(_lastPreActivation, outputGradient);

        // The gradient splits to both branches (residual add)
        var gradMain = gradPreActivation;
        var gradIdentity = gradPreActivation;

        // Backward through main branch: bn3 -> conv3 -> relu2 -> bn2 -> conv2 -> relu1 -> bn1 -> conv1
        var gradBn3 = _bn3.Backward(gradMain);
        var gradConv3 = _conv3.Backward(gradBn3);
        var gradRelu2 = ApplyReLUDerivative(_lastBn2Output, gradConv3);
        var gradBn2 = _bn2.Backward(gradRelu2);
        var gradConv2 = _conv2.Backward(gradBn2);
        var gradRelu1 = ApplyReLUDerivative(_lastBn1Output, gradConv2);
        var gradBn1 = _bn1.Backward(gradRelu1);
        var gradConv1 = _conv1.Backward(gradBn1);

        // Backward through identity branch
        Tensor<T> gradInput;
        if (_hasDownsample && _downsampleBn is not null && _downsampleConv is not null)
        {
            var gradDsBn = _downsampleBn.Backward(gradIdentity);
            var gradDsConv = _downsampleConv.Backward(gradDsBn);
            gradInput = Engine.TensorAdd(gradConv1, gradDsConv);
        }
        else
        {
            gradInput = Engine.TensorAdd(gradConv1, gradIdentity);
        }

        return gradInput;
    }

    /// <summary>
    /// Updates the parameters of all internal layers.
    /// </summary>
    /// <param name="learningRate">The learning rate.</param>
    public override void UpdateParameters(T learningRate)
    {
        _conv1.UpdateParameters(learningRate);
        _bn1.UpdateParameters(learningRate);
        _conv2.UpdateParameters(learningRate);
        _bn2.UpdateParameters(learningRate);
        _conv3.UpdateParameters(learningRate);
        _bn3.UpdateParameters(learningRate);
        _downsampleConv?.UpdateParameters(learningRate);
        _downsampleBn?.UpdateParameters(learningRate);
    }

    /// <summary>
    /// Gets all trainable parameters.
    /// </summary>
    /// <returns>A vector containing all parameters.</returns>
    public override Vector<T> GetParameters()
    {
        var allParams = new List<T>();
        allParams.AddRange(_conv1.GetParameters().ToArray());
        allParams.AddRange(_bn1.GetParameters().ToArray());
        allParams.AddRange(_conv2.GetParameters().ToArray());
        allParams.AddRange(_bn2.GetParameters().ToArray());
        allParams.AddRange(_conv3.GetParameters().ToArray());
        allParams.AddRange(_bn3.GetParameters().ToArray());
        if (_downsampleConv is not null && _downsampleBn is not null)
        {
            allParams.AddRange(_downsampleConv.GetParameters().ToArray());
            allParams.AddRange(_downsampleBn.GetParameters().ToArray());
        }
        return new Vector<T>([.. allParams]);
    }

    /// <summary>
    /// Resets the internal state of the block.
    /// </summary>
    public override void ResetState()
    {
        _lastInput = null;
        _lastConv1Output = null;
        _lastBn1Output = null;
        _lastRelu1Output = null;
        _lastConv2Output = null;
        _lastBn2Output = null;
        _lastRelu2Output = null;
        _lastConv3Output = null;
        _lastBn3Output = null;
        _lastIdentity = null;
        _lastPreActivation = null;

        _conv1.ResetState();
        _bn1.ResetState();
        _conv2.ResetState();
        _bn2.ResetState();
        _conv3.ResetState();
        _bn3.ResetState();
        _downsampleConv?.ResetState();
        _downsampleBn?.ResetState();
    }

    /// <summary>
    /// Gets whether this layer supports JIT compilation.
    /// </summary>
    /// <remarks>
    /// BottleneckBlock supports JIT compilation when all its sub-layers support JIT.
    /// This includes conv1, bn1, conv2, bn2, conv3, bn3, and optionally the downsample layers.
    /// </remarks>
    public override bool SupportsJitCompilation
    {
        get
        {
            // Check all required layers
            if (!_conv1.SupportsJitCompilation || !_bn1.SupportsJitCompilation ||
                !_conv2.SupportsJitCompilation || !_bn2.SupportsJitCompilation ||
                !_conv3.SupportsJitCompilation || !_bn3.SupportsJitCompilation)
            {
                return false;
            }

            // Check downsample layers if present
            if (_hasDownsample &&
                (_downsampleConv is null || !_downsampleConv.SupportsJitCompilation ||
                 _downsampleBn is null || !_downsampleBn.SupportsJitCompilation))
            {
                return false;
            }

            return true;
        }
    }

    /// <summary>
    /// Exports the computation graph for JIT compilation.
    /// </summary>
    /// <param name="inputNodes">List to populate with input computation nodes.</param>
    /// <returns>The output computation node representing the BottleneckBlock.</returns>
    /// <remarks>
    /// <para>
    /// This method builds a computation graph representing the BottleneckBlock:
    /// Input -> Conv1(1x1) -> BN1 -> ReLU -> Conv2(3x3) -> BN2 -> ReLU -> Conv3(1x1) -> BN3 -> (+Identity) -> ReLU -> Output
    /// </para>
    /// <para>
    /// For JIT compilation, we chain the sub-layer computation graphs together
    /// and add the residual connection using TensorOperations.Add.
    /// </para>
    /// </remarks>
    public override Autodiff.ComputationNode<T> ExportComputationGraph(List<Autodiff.ComputationNode<T>> inputNodes)
    {
        if (inputNodes is null)
        {
            throw new ArgumentNullException(nameof(inputNodes));
        }

        if (InputShape is null || InputShape.Length == 0)
        {
            throw new InvalidOperationException("Layer input shape not configured.");
        }

        // Create symbolic input node with batch dimension
        var symbolicInput = new Tensor<T>(new int[] { 1 }.Concat(InputShape).ToArray());
        var inputNode = Autodiff.TensorOperations<T>.Variable(symbolicInput, "input");
        inputNodes.Add(inputNode);

        // Main branch: conv1 (1x1) -> bn1 -> relu -> conv2 (3x3) -> bn2 -> relu -> conv3 (1x1) -> bn3

        // Conv1 (1x1 reduce)
        var conv1Biases = _conv1.GetBiases();
        var conv1Node = Autodiff.TensorOperations<T>.Conv2D(
            inputNode,
            Autodiff.TensorOperations<T>.Constant(_conv1.GetFilters(), "conv1_kernel"),
            conv1Biases is not null ? Autodiff.TensorOperations<T>.Constant(conv1Biases, "conv1_bias") : null,
            stride: new int[] { _conv1.Stride, _conv1.Stride },
            padding: new int[] { _conv1.Padding, _conv1.Padding });

        // BN1
        var bn1Node = Autodiff.TensorOperations<T>.BatchNorm(
            conv1Node,
            gamma: Autodiff.TensorOperations<T>.Constant(_bn1.GetGamma(), "bn1_gamma"),
            beta: Autodiff.TensorOperations<T>.Constant(_bn1.GetBeta(), "bn1_beta"),
            runningMean: _bn1.GetRunningMean(),
            runningVar: _bn1.GetRunningVariance(),
            training: false,
            epsilon: NumOps.ToDouble(_bn1.GetEpsilon()));

        // ReLU1
        var relu1Node = Autodiff.TensorOperations<T>.ReLU(bn1Node);

        // Conv2 (3x3 bottleneck)
        var conv2Biases = _conv2.GetBiases();
        var conv2Node = Autodiff.TensorOperations<T>.Conv2D(
            relu1Node,
            Autodiff.TensorOperations<T>.Constant(_conv2.GetFilters(), "conv2_kernel"),
            conv2Biases is not null ? Autodiff.TensorOperations<T>.Constant(conv2Biases, "conv2_bias") : null,
            stride: new int[] { _conv2.Stride, _conv2.Stride },
            padding: new int[] { _conv2.Padding, _conv2.Padding });

        // BN2
        var bn2Node = Autodiff.TensorOperations<T>.BatchNorm(
            conv2Node,
            gamma: Autodiff.TensorOperations<T>.Constant(_bn2.GetGamma(), "bn2_gamma"),
            beta: Autodiff.TensorOperations<T>.Constant(_bn2.GetBeta(), "bn2_beta"),
            runningMean: _bn2.GetRunningMean(),
            runningVar: _bn2.GetRunningVariance(),
            training: false,
            epsilon: NumOps.ToDouble(_bn2.GetEpsilon()));

        // ReLU2
        var relu2Node = Autodiff.TensorOperations<T>.ReLU(bn2Node);

        // Conv3 (1x1 expand)
        var conv3Biases = _conv3.GetBiases();
        var conv3Node = Autodiff.TensorOperations<T>.Conv2D(
            relu2Node,
            Autodiff.TensorOperations<T>.Constant(_conv3.GetFilters(), "conv3_kernel"),
            conv3Biases is not null ? Autodiff.TensorOperations<T>.Constant(conv3Biases, "conv3_bias") : null,
            stride: new int[] { _conv3.Stride, _conv3.Stride },
            padding: new int[] { _conv3.Padding, _conv3.Padding });

        // BN3
        var bn3Node = Autodiff.TensorOperations<T>.BatchNorm(
            conv3Node,
            gamma: Autodiff.TensorOperations<T>.Constant(_bn3.GetGamma(), "bn3_gamma"),
            beta: Autodiff.TensorOperations<T>.Constant(_bn3.GetBeta(), "bn3_beta"),
            runningMean: _bn3.GetRunningMean(),
            runningVar: _bn3.GetRunningVariance(),
            training: false,
            epsilon: NumOps.ToDouble(_bn3.GetEpsilon()));

        // Identity/skip branch
        Autodiff.ComputationNode<T> identityNode;
        if (_hasDownsample && _downsampleConv is not null && _downsampleBn is not null)
        {
            // Downsample: conv1x1 -> bn
            var dsConvBiases = _downsampleConv.GetBiases();
            var dsConvNode = Autodiff.TensorOperations<T>.Conv2D(
                inputNode,
                Autodiff.TensorOperations<T>.Constant(_downsampleConv.GetFilters(), "ds_conv_kernel"),
                dsConvBiases is not null ? Autodiff.TensorOperations<T>.Constant(dsConvBiases, "ds_conv_bias") : null,
                stride: new int[] { _downsampleConv.Stride, _downsampleConv.Stride },
                padding: new int[] { _downsampleConv.Padding, _downsampleConv.Padding });

            identityNode = Autodiff.TensorOperations<T>.BatchNorm(
                dsConvNode,
                gamma: Autodiff.TensorOperations<T>.Constant(_downsampleBn.GetGamma(), "ds_bn_gamma"),
                beta: Autodiff.TensorOperations<T>.Constant(_downsampleBn.GetBeta(), "ds_bn_beta"),
                runningMean: _downsampleBn.GetRunningMean(),
                runningVar: _downsampleBn.GetRunningVariance(),
                training: false,
                epsilon: NumOps.ToDouble(_downsampleBn.GetEpsilon()));
        }
        else
        {
            // Identity shortcut - just use input directly
            identityNode = inputNode;
        }

        // Residual connection: add main branch and identity
        var addNode = Autodiff.TensorOperations<T>.Add(bn3Node, identityNode);

        // Final ReLU
        var outputNode = Autodiff.TensorOperations<T>.ReLU(addNode);

        return outputNode;
    }

    private Tensor<T> ApplyReLU(Tensor<T> input)
    {
        return input.Transform((x, _) => _relu.Activate(x));
    }

    private Tensor<T> ApplyReLUDerivative(Tensor<T> preActivation, Tensor<T> gradient)
    {
        // ReLU derivative: 1 if x > 0, 0 otherwise
        var derivative = preActivation.Transform((x, _) => _relu.Derivative(x));
        return Engine.TensorMultiply(gradient, derivative);
    }
}
