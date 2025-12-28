using AiDotNet.ActivationFunctions;
using AiDotNet.Engines;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Implements the BasicBlock used in ResNet18 and ResNet34 architectures.
/// </summary>
/// <remarks>
/// <para>
/// The BasicBlock contains two 3x3 convolutional layers with batch normalization and ReLU activation.
/// A skip connection adds the input directly to the output, enabling gradient flow through very deep networks.
/// </para>
/// <para>
/// <b>Architecture:</b>
/// <code>
/// Input ─┬─ Conv3x3 ─ BN ─ ReLU ─ Conv3x3 ─ BN ─┬─ (+) ─ ReLU ─ Output
///        │                                       │
///        └───────────── [Downsample?] ───────────┘
/// </code>
/// </para>
/// <para>
/// <b>For Beginners:</b> The BasicBlock is like a "learning module" with a shortcut.
///
/// The key insight is:
/// - The two conv layers learn to predict what needs to be ADDED to the input (the "residual")
/// - The skip connection adds the original input back to this learned residual
/// - This makes it easier to train very deep networks because gradients can flow directly through the skip connection
///
/// When the input and output have different dimensions (due to stride or channel changes),
/// a downsample layer (1x1 conv + BN) is used to match the dimensions before adding.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class BasicBlock<T> : LayerBase<T>
{
    /// <summary>
    /// The expansion factor for BasicBlock. BasicBlock does not expand channels.
    /// </summary>
    public const int Expansion = 1;

    private readonly ConvolutionalLayer<T> _conv1;
    private readonly BatchNormalizationLayer<T> _bn1;
    private readonly ConvolutionalLayer<T> _conv2;
    private readonly BatchNormalizationLayer<T> _bn2;
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
    private Tensor<T>? _lastIdentity;
    private Tensor<T>? _lastPreActivation;

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new instance of the <see cref="BasicBlock{T}"/> class.
    /// </summary>
    /// <param name="inChannels">The number of input channels.</param>
    /// <param name="outChannels">The number of output channels.</param>
    /// <param name="stride">The stride for the first convolution (default: 1).</param>
    /// <param name="inputHeight">The input spatial height.</param>
    /// <param name="inputWidth">The input spatial width.</param>
    /// <param name="zeroInitResidual">If true, initialize the last BN to zero for better training stability.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When stride > 1, the block will downsample the spatial dimensions.
    /// When inChannels != outChannels, a projection shortcut is used to match dimensions.
    /// </para>
    /// </remarks>
    public BasicBlock(
        int inChannels,
        int outChannels,
        int stride = 1,
        int inputHeight = 56,
        int inputWidth = 56,
        bool zeroInitResidual = true)
        : base(
            inputShape: [inChannels, inputHeight, inputWidth],
            outputShape: [outChannels, inputHeight / stride, inputWidth / stride])
    {
        _relu = new ReLUActivation<T>();

        // First conv: 3x3, stride = stride
        _conv1 = new ConvolutionalLayer<T>(
            inputDepth: inChannels,
            inputHeight: inputHeight,
            inputWidth: inputWidth,
            outputDepth: outChannels,
            kernelSize: 3,
            stride: stride,
            padding: 1,
            activationFunction: new IdentityActivation<T>());

        int outHeight = inputHeight / stride;
        int outWidth = inputWidth / stride;

        _bn1 = new BatchNormalizationLayer<T>(outChannels);

        // Second conv: 3x3, stride = 1
        _conv2 = new ConvolutionalLayer<T>(
            inputDepth: outChannels,
            inputHeight: outHeight,
            inputWidth: outWidth,
            outputDepth: outChannels,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            activationFunction: new IdentityActivation<T>());

        _bn2 = new BatchNormalizationLayer<T>(outChannels);

        // Zero-init residual: initialize last BN's gamma to zero so residual blocks
        // start as identity mappings, improving training stability
        if (zeroInitResidual)
        {
            _bn2.ZeroInitGamma();
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
                activationFunction: new IdentityActivation<T>());

            _downsampleBn = new BatchNormalizationLayer<T>(outChannels);
        }
    }

    /// <summary>
    /// Performs the forward pass through the BasicBlock.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>The output tensor after the residual connection.</returns>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;

        // Main branch: conv1 -> bn1 -> relu -> conv2 -> bn2
        _lastConv1Output = _conv1.Forward(input);
        _lastBn1Output = _bn1.Forward(_lastConv1Output);
        _lastRelu1Output = ApplyReLU(_lastBn1Output);
        _lastConv2Output = _conv2.Forward(_lastRelu1Output);
        _lastBn2Output = _bn2.Forward(_lastConv2Output);

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
        _lastPreActivation = Engine.TensorAdd(_lastBn2Output, _lastIdentity);

        // Final ReLU
        return ApplyReLU(_lastPreActivation);
    }

    /// <summary>
    /// Performs the backward pass through the BasicBlock.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the output.</param>
    /// <returns>The gradient of the loss with respect to the input.</returns>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput is null || _lastPreActivation is null || _lastIdentity is null ||
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

        // Backward through main branch: bn2 -> conv2 -> relu1 -> bn1 -> conv1
        var gradBn2 = _bn2.Backward(gradMain);
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
        _lastIdentity = null;
        _lastPreActivation = null;

        _conv1.ResetState();
        _bn1.ResetState();
        _conv2.ResetState();
        _bn2.ResetState();
        _downsampleConv?.ResetState();
        _downsampleBn?.ResetState();
    }

    /// <summary>
    /// Gets whether this layer supports JIT compilation.
    /// </summary>
    /// <remarks>
    /// BasicBlock supports JIT compilation when all its sub-layers support JIT.
    /// This includes conv1, bn1, conv2, bn2, and optionally the downsample layers.
    /// </remarks>
    public override bool SupportsJitCompilation
    {
        get
        {
            // Check all required layers
            if (!_conv1.SupportsJitCompilation || !_bn1.SupportsJitCompilation ||
                !_conv2.SupportsJitCompilation || !_bn2.SupportsJitCompilation)
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
    /// <returns>The output computation node representing the BasicBlock.</returns>
    /// <remarks>
    /// <para>
    /// This method builds a computation graph representing the BasicBlock:
    /// Input -> Conv1 -> BN1 -> ReLU -> Conv2 -> BN2 -> (+Identity) -> ReLU -> Output
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

        // Main branch: conv1 -> bn1 -> relu -> conv2 -> bn2
        // Conv1: Build the convolution node using the layer's parameters
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

        // Conv2
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
        var addNode = Autodiff.TensorOperations<T>.Add(bn2Node, identityNode);

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
