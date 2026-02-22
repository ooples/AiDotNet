using AiDotNet.ActivationFunctions;
using AiDotNet.Engines;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Diffusion.VAE;

/// <summary>
/// Residual block for VAE encoder/decoder with GroupNorm and skip connections.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This implements a proper VAE residual block following the Stable Diffusion VAE architecture:
/// - GroupNorm -> SiLU -> Conv -> GroupNorm -> SiLU -> Conv
/// - Skip connection with optional 1x1 convolution when input/output channels differ
/// </para>
/// <para>
/// <b>For Beginners:</b> A residual block helps the network learn more effectively.
///
/// Think of it like taking notes during a lecture:
/// - The main path (two convolutions) learns new features
/// - The skip connection preserves the original information
/// - Adding them together means you learn the "difference" or "improvement"
///
/// The GroupNorm helps stabilize training by normalizing activations within groups
/// of channels, which works well even with small batch sizes commonly used in
/// image generation tasks.
///
/// Structure:
/// ```
///     input ─────────────────────────────────┐
///       │                                    │
///       ├─→ GroupNorm → SiLU → Conv3x3 ─→ h  │ (skip connection)
///       │                                    │
///       │        ↓                           │
///       │                                    │
///       │   GroupNorm → SiLU → Conv3x3 ─→ h  │
///       │                                    │
///       │        ↓                           ↓
///       │                                 [1x1 Conv if channels differ]
///       │        ↓                           ↓
///       └────────────────→ (+) ←─────────────┘
///                          │
///                       output
/// ```
/// </para>
/// </remarks>
public class VAEResBlock<T> : LayerBase<T>
{
    /// <summary>
    /// First GroupNorm layer.
    /// </summary>
    private readonly GroupNormalizationLayer<T> _norm1;

    /// <summary>
    /// Second GroupNorm layer.
    /// </summary>
    private readonly GroupNormalizationLayer<T> _norm2;

    /// <summary>
    /// First convolution layer.
    /// </summary>
    private readonly ConvolutionalLayer<T> _conv1;

    /// <summary>
    /// Second convolution layer.
    /// </summary>
    private readonly ConvolutionalLayer<T> _conv2;

    /// <summary>
    /// Optional 1x1 convolution for skip connection when channels differ.
    /// </summary>
    private readonly ConvolutionalLayer<T>? _skipConv;

    /// <summary>
    /// SiLU activation function.
    /// </summary>
    private readonly IActivationFunction<T> _silu;

    /// <summary>
    /// Number of input channels.
    /// </summary>
    private readonly int _inChannels;

    /// <summary>
    /// Number of output channels.
    /// </summary>
    private readonly int _outChannels;

    /// <summary>
    /// Number of groups for GroupNorm.
    /// </summary>
    private readonly int _numGroups;

    /// <summary>
    /// Cached input from forward pass for backward.
    /// </summary>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Cached intermediate values for backward pass.
    /// </summary>
    private Tensor<T>? _norm1Output;
    private Tensor<T>? _silu1Output;
    private Tensor<T>? _conv1Output;
    private Tensor<T>? _norm2Output;
    private Tensor<T>? _silu2Output;
    private Tensor<T>? _conv2Output;
    private Tensor<T>? _skipOutput;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets the number of input channels.
    /// </summary>
    public int InputChannels => _inChannels;

    /// <summary>
    /// Gets the number of output channels.
    /// </summary>
    public int OutputChannels => _outChannels;

    /// <summary>
    /// Gets the number of groups for GroupNorm.
    /// </summary>
    public int NumGroups => _numGroups;

    /// <summary>
    /// Initializes a new instance of the VAEResBlock class.
    /// </summary>
    /// <param name="inChannels">Number of input channels.</param>
    /// <param name="outChannels">Number of output channels.</param>
    /// <param name="numGroups">Number of groups for GroupNorm (default: 32).</param>
    /// <param name="spatialSize">Spatial dimensions (height/width) for conv layer setup.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Create a VAE residual block with the specified channel configuration.
    ///
    /// Typical configurations:
    /// - numGroups=32 for 256+ channels
    /// - numGroups=16 for 128 channels
    /// - numGroups=8 for 64 channels
    ///
    /// The numGroups should evenly divide the channel count for proper normalization.
    /// </para>
    /// </remarks>
    public VAEResBlock(int inChannels, int outChannels, int numGroups = 32, int spatialSize = 32)
        : base(CalculateInputShape(inChannels, spatialSize), CalculateOutputShape(outChannels, spatialSize))
    {
        if (inChannels <= 0)
            throw new ArgumentOutOfRangeException(nameof(inChannels), "Input channels must be positive.");
        if (outChannels <= 0)
            throw new ArgumentOutOfRangeException(nameof(outChannels), "Output channels must be positive.");
        if (numGroups <= 0)
            throw new ArgumentOutOfRangeException(nameof(numGroups), "Number of groups must be positive.");

        // Adjust numGroups if it doesn't divide evenly
        numGroups = Math.Min(numGroups, Math.Min(inChannels, outChannels));
        while (inChannels % numGroups != 0 || outChannels % numGroups != 0)
        {
            numGroups--;
            if (numGroups <= 0) numGroups = 1;
        }

        _inChannels = inChannels;
        _outChannels = outChannels;
        _numGroups = numGroups;
        _silu = new SiLUActivation<T>();

        // GroupNorm layers
        _norm1 = new GroupNormalizationLayer<T>(numGroups, inChannels);
        _norm2 = new GroupNormalizationLayer<T>(numGroups, outChannels);

        // Convolutional layers (3x3 with padding=1 preserves spatial dimensions)
        _conv1 = new ConvolutionalLayer<T>(
            inputDepth: inChannels,
            outputDepth: outChannels,
            kernelSize: 3,
            inputHeight: spatialSize,
            inputWidth: spatialSize,
            stride: 1,
            padding: 1,
            activationFunction: new IdentityActivation<T>());

        _conv2 = new ConvolutionalLayer<T>(
            inputDepth: outChannels,
            outputDepth: outChannels,
            kernelSize: 3,
            inputHeight: spatialSize,
            inputWidth: spatialSize,
            stride: 1,
            padding: 1,
            activationFunction: new IdentityActivation<T>());

        // Skip connection: 1x1 conv if channels differ
        if (inChannels != outChannels)
        {
            _skipConv = new ConvolutionalLayer<T>(
                inputDepth: inChannels,
                outputDepth: outChannels,
                kernelSize: 1,
                inputHeight: spatialSize,
                inputWidth: spatialSize,
                stride: 1,
                padding: 0,
                activationFunction: new IdentityActivation<T>());
        }
    }

    private static int[] CalculateInputShape(int channels, int spatialSize)
    {
        return new[] { channels, spatialSize, spatialSize };
    }

    private static int[] CalculateOutputShape(int channels, int spatialSize)
    {
        return new[] { channels, spatialSize, spatialSize };
    }

    /// <summary>
    /// Performs the forward pass through the residual block.
    /// </summary>
    /// <param name="input">Input tensor with shape [batch, channels, height, width].</param>
    /// <returns>Output tensor with shape [batch, outChannels, height, width].</returns>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;

        // Main path: GroupNorm -> SiLU -> Conv -> GroupNorm -> SiLU -> Conv
        _norm1Output = _norm1.Forward(input);
        _silu1Output = ApplySiLU(_norm1Output);
        _conv1Output = _conv1.Forward(_silu1Output);

        _norm2Output = _norm2.Forward(_conv1Output);
        _silu2Output = ApplySiLU(_norm2Output);
        _conv2Output = _conv2.Forward(_silu2Output);

        // Skip connection
        _skipOutput = _skipConv != null ? _skipConv.Forward(input) : input;

        // Add main path and skip connection
        return Engine.TensorAdd(_conv2Output, _skipOutput);
    }

    /// <summary>
    /// Applies SiLU activation to a tensor.
    /// </summary>
    private Tensor<T> ApplySiLU(Tensor<T> input)
    {
        var output = new Tensor<T>(input.Shape);
        var inputSpan = input.AsSpan();
        var outputSpan = output.AsWritableSpan();

        for (int i = 0; i < inputSpan.Length; i++)
        {
            outputSpan[i] = _silu.Activate(inputSpan[i]);
        }

        return output;
    }

    /// <summary>
    /// Computes the SiLU derivative for a tensor.
    /// </summary>
    private Tensor<T> ApplySiLUDerivative(Tensor<T> input, Tensor<T> gradient)
    {
        var output = new Tensor<T>(input.Shape);
        var inputSpan = input.AsSpan();
        var gradSpan = gradient.AsSpan();
        var outputSpan = output.AsWritableSpan();

        for (int i = 0; i < inputSpan.Length; i++)
        {
            outputSpan[i] = NumOps.Multiply(_silu.Derivative(inputSpan[i]), gradSpan[i]);
        }

        return output;
    }

    /// <summary>
    /// Performs the backward pass through the residual block.
    /// </summary>
    /// <param name="outputGradient">Gradient of loss with respect to output.</param>
    /// <returns>Gradient of loss with respect to input.</returns>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _norm1Output == null || _silu1Output == null ||
            _conv1Output == null || _norm2Output == null || _silu2Output == null ||
            _conv2Output == null || _skipOutput == null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass.");
        }

        // Gradient flows through both main path and skip connection
        var mainGradient = outputGradient;
        var skipGradient = outputGradient;

        // Backward through skip connection
        Tensor<T> skipInputGrad;
        if (_skipConv != null)
        {
            skipInputGrad = _skipConv.Backward(skipGradient);
        }
        else
        {
            skipInputGrad = skipGradient;
        }

        // Backward through main path: conv2 -> SiLU -> norm2 -> conv1 -> SiLU -> norm1
        var conv2Grad = _conv2.Backward(mainGradient);
        var silu2Grad = ApplySiLUDerivative(_norm2Output, conv2Grad);
        var norm2Grad = _norm2.Backward(silu2Grad);

        var conv1Grad = _conv1.Backward(norm2Grad);
        var silu1Grad = ApplySiLUDerivative(_norm1Output, conv1Grad);
        var norm1Grad = _norm1.Backward(silu1Grad);

        // Sum gradients from main path and skip connection
        return Engine.TensorAdd(norm1Grad, skipInputGrad);
    }

    /// <summary>
    /// Updates all learnable parameters using gradient descent.
    /// </summary>
    /// <param name="learningRate">The learning rate for the update.</param>
    public override void UpdateParameters(T learningRate)
    {
        _norm1.UpdateParameters(learningRate);
        _norm2.UpdateParameters(learningRate);
        _conv1.UpdateParameters(learningRate);
        _conv2.UpdateParameters(learningRate);
        _skipConv?.UpdateParameters(learningRate);
    }

    /// <summary>
    /// Gets all trainable parameters as a single vector.
    /// </summary>
    public override Vector<T> GetParameters()
    {
        var paramsList = new List<T>();

        AddParameters(paramsList, _norm1.GetParameters());
        AddParameters(paramsList, _conv1.GetParameters());
        AddParameters(paramsList, _norm2.GetParameters());
        AddParameters(paramsList, _conv2.GetParameters());

        if (_skipConv != null)
        {
            AddParameters(paramsList, _skipConv.GetParameters());
        }

        return new Vector<T>(paramsList.ToArray());
    }

    private static void AddParameters(List<T> list, Vector<T> parameters)
    {
        for (int i = 0; i < parameters.Length; i++)
        {
            list.Add(parameters[i]);
        }
    }

    /// <summary>
    /// Sets all trainable parameters from a single vector.
    /// </summary>
    public override void SetParameters(Vector<T> parameters)
    {
        int index = 0;

        SetLayerParams(_norm1, parameters, ref index);
        SetLayerParams(_conv1, parameters, ref index);
        SetLayerParams(_norm2, parameters, ref index);
        SetLayerParams(_conv2, parameters, ref index);

        if (_skipConv != null)
        {
            SetLayerParams(_skipConv, parameters, ref index);
        }
    }

    private static void SetLayerParams(ILayer<T> layer, Vector<T> parameters, ref int index)
    {
        var layerParams = layer.GetParameters();
        var newParams = new Vector<T>(layerParams.Length);

        for (int i = 0; i < layerParams.Length && index < parameters.Length; i++)
        {
            newParams[i] = parameters[index++];
        }

        layer.SetParameters(newParams);
    }

    /// <summary>
    /// Resets the internal state of the block.
    /// </summary>
    public override void ResetState()
    {
        _lastInput = null;
        _norm1Output = null;
        _silu1Output = null;
        _conv1Output = null;
        _norm2Output = null;
        _silu2Output = null;
        _conv2Output = null;
        _skipOutput = null;

        _norm1.ResetState();
        _norm2.ResetState();
        _conv1.ResetState();
        _conv2.ResetState();
        _skipConv?.ResetState();
    }

    /// <inheritdoc />
    public override bool SupportsJitCompilation => false;

    /// <inheritdoc />
    public override Autodiff.ComputationNode<T> ExportComputationGraph(List<Autodiff.ComputationNode<T>> inputNodes)
    {
        throw new NotSupportedException(
            "VAEResBlock JIT compilation is not yet implemented. " +
            "Use the layer in interpreted mode by setting SupportsJitCompilation = false.");
    }

    /// <summary>
    /// Saves the block's state to a binary writer.
    /// </summary>
    public override void Serialize(BinaryWriter writer)
    {
        base.Serialize(writer);
        writer.Write(_inChannels);
        writer.Write(_outChannels);
        writer.Write(_numGroups);

        _norm1.Serialize(writer);
        _conv1.Serialize(writer);
        _norm2.Serialize(writer);
        _conv2.Serialize(writer);

        writer.Write(_skipConv != null);
        _skipConv?.Serialize(writer);
    }

    /// <summary>
    /// Loads the block's state from a binary reader.
    /// </summary>
    public override void Deserialize(BinaryReader reader)
    {
        base.Deserialize(reader);
        var inChannels = reader.ReadInt32();
        var outChannels = reader.ReadInt32();
        var numGroups = reader.ReadInt32();

        if (inChannels != _inChannels || outChannels != _outChannels || numGroups != _numGroups)
        {
            throw new InvalidOperationException(
                $"Architecture mismatch: expected ({_inChannels}, {_outChannels}, {_numGroups}) " +
                $"but got ({inChannels}, {outChannels}, {numGroups}).");
        }

        _norm1.Deserialize(reader);
        _conv1.Deserialize(reader);
        _norm2.Deserialize(reader);
        _conv2.Deserialize(reader);

        var hasSkipConv = reader.ReadBoolean();
        if (hasSkipConv && _skipConv != null)
        {
            _skipConv.Deserialize(reader);
        }
    }
}
