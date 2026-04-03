using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Engines;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;

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
[LayerCategory(LayerCategory.Residual)]
[LayerCategory(LayerCategory.Convolution)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerTask(LayerTask.SpatialProcessing)]
[LayerProperty(IsTrainable = true, ChangesShape = true, ExpectedInputRank = 3, Cost = ComputeCost.High, TestInputShape = "1, 8, 8", TestConstructorArgs = "1, 1, 8, 8")]
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

    // GPU cached tensors for backward pass
    private Tensor<T>? _gpuBn1Out;
    private Tensor<T>? _gpuBn2Out;
    private Tensor<T>? _gpuPreActivation;

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    public override int ParameterCount =>
        _conv1.ParameterCount + _bn1.ParameterCount + _conv2.ParameterCount + _bn2.ParameterCount +
        (_downsampleConv?.ParameterCount ?? 0) + (_downsampleBn?.ParameterCount ?? 0);
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets a value indicating whether this layer has a GPU implementation.
    /// </summary>
    protected override bool SupportsGpuExecution => true;

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

        RegisterSubLayer(_conv1);
        RegisterSubLayer(_bn1);
        RegisterSubLayer(_conv2);
        RegisterSubLayer(_bn2);
        if (_downsampleConv is not null) RegisterSubLayer(_downsampleConv);
        if (_downsampleBn is not null) RegisterSubLayer(_downsampleBn);
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
    /// Performs the forward pass on GPU, keeping data GPU-resident.
    /// </summary>
    /// <param name="inputs">The input tensors (expects single input).</param>
    /// <returns>The output tensor on GPU.</returns>
    public override Tensor<T> ForwardGpu(params Tensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));

        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("ForwardGpu requires a DirectGpuTensorEngine.");

        var input = inputs[0];

        // Main branch: conv1 -> bn1 -> relu -> conv2 -> bn2
        var conv1Out = _conv1.ForwardGpu(input);
        var bn1Out = _bn1.ForwardGpu(conv1Out);

        // Cache bn1Out for backward pass (ReLU1 backward needs it)
        _gpuBn1Out = bn1Out;

        var relu1Out = gpuEngine.ReluGpu(bn1Out);
        var conv2Out = _conv2.ForwardGpu(relu1Out);
        var bn2Out = _bn2.ForwardGpu(conv2Out);

        // Cache bn2Out for backward pass (ReLU2/final backward needs it)
        _gpuBn2Out = bn2Out;

        // Identity/skip branch
        Tensor<T> identity;
        if (_hasDownsample && _downsampleConv is not null && _downsampleBn is not null)
        {
            var dsConvOut = _downsampleConv.ForwardGpu(input);
            identity = _downsampleBn.ForwardGpu(dsConvOut);
        }
        else
        {
            identity = input;
        }

        // Add residual connection
        var preActivation = gpuEngine.AddGpu(bn2Out, identity);

        // Cache preActivation for backward pass (final ReLU backward needs it)
        _gpuPreActivation = preActivation;

        // Final ReLU
        return gpuEngine.ReluGpu(preActivation);
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

    public override Vector<T> GetParameterGradients()
    {
        var grads = new List<T>();
        grads.AddRange(_conv1.GetParameterGradients().ToArray());
        grads.AddRange(_bn1.GetParameterGradients().ToArray());
        grads.AddRange(_conv2.GetParameterGradients().ToArray());
        grads.AddRange(_bn2.GetParameterGradients().ToArray());
        if (_downsampleConv is not null && _downsampleBn is not null)
        {
            grads.AddRange(_downsampleConv.GetParameterGradients().ToArray());
            grads.AddRange(_downsampleBn.GetParameterGradients().ToArray());
        }
        return new Vector<T>([.. grads]);
    }

    public override void ClearGradients()
    {
        base.ClearGradients();
        _conv1.ClearGradients(); _bn1.ClearGradients();
        _conv2.ClearGradients(); _bn2.ClearGradients();
        _downsampleConv?.ClearGradients(); _downsampleBn?.ClearGradients();
    }

    public override void SetParameters(Vector<T> parameters)
    {
        int idx = 0;
        void Set(ILayer<T> layer)
        {
            int count = layer.ParameterCount;
            layer.SetParameters(parameters.Slice(idx, count));
            idx += count;
        }
        Set(_conv1); Set(_bn1); Set(_conv2); Set(_bn2);
        if (_downsampleConv is not null && _downsampleBn is not null)
        {
            Set(_downsampleConv); Set(_downsampleBn);
        }
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
