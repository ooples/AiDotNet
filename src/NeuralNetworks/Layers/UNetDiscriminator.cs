using AiDotNet.ActivationFunctions;
using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// U-Net Discriminator as used in Real-ESRGAN for improved perceptual quality.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This implements the U-Net discriminator from the Real-ESRGAN paper (Wang et al., 2021).
/// Unlike traditional patch discriminators, U-Net discriminator provides pixel-level feedback
/// which helps the generator produce finer details.
/// </para>
/// <para>
/// The architecture has an encoder-decoder structure:
/// <code>
/// Input (3 channels, HR image)
///   ↓
/// Encoder (progressively downsample with skip connections)
///   ↓
/// Bottleneck
///   ↓
/// Decoder (progressively upsample, concat with skip connections)
///   ↓
/// Output (1 channel, per-pixel real/fake prediction)
/// </code>
/// </para>
/// <para>
/// <b>For Beginners:</b> The discriminator judges whether an image is real or fake.
///
/// Traditional discriminators output a single "real/fake" score for the whole image.
/// U-Net discriminator outputs a "real/fake" prediction for EVERY PIXEL, which:
/// - Provides more detailed feedback to the generator
/// - Helps produce sharper details and textures
/// - Enables better reconstruction of fine features
///
/// The U-Net architecture (encoder + decoder with skip connections) allows the
/// discriminator to consider both local details and global context.
/// </para>
/// <para>
/// <b>Reference:</b> Wang et al., "Real-ESRGAN: Training Real-World Blind Super-Resolution
/// with Pure Synthetic Data", ICCV 2021. https://arxiv.org/abs/2107.10833
/// </para>
/// </remarks>
public class UNetDiscriminator<T> : LayerBase<T>, IChainableComputationGraph<T>
{
    #region Fields

    /// <summary>
    /// Encoder blocks (downsampling path).
    /// </summary>
    private readonly UNetConvBlock<T>[] _encoderBlocks;

    /// <summary>
    /// Decoder blocks (upsampling path).
    /// </summary>
    private readonly UNetUpBlock<T>[] _decoderBlocks;

    /// <summary>
    /// Initial convolution.
    /// </summary>
    private readonly ConvolutionalLayer<T> _convFirst;

    /// <summary>
    /// Final convolution (1x1 to output channels).
    /// </summary>
    private readonly ConvolutionalLayer<T> _convLast;

    /// <summary>
    /// Number of encoder/decoder blocks.
    /// </summary>
    private readonly int _numBlocks;

    /// <summary>
    /// Base number of channels.
    /// </summary>
    private readonly int _numChannels;

    /// <summary>
    /// LeakyReLU activation.
    /// </summary>
    private readonly LeakyReLUActivation<T> _leakyReLU;

    /// <summary>
    /// Skip connections stored during forward pass for concatenation.
    /// </summary>
    private Tensor<T>[]? _skipConnections;

    /// <summary>
    /// Cached input for backpropagation.
    /// </summary>
    private Tensor<T>? _lastInput;

    #endregion

    #region Properties

    /// <summary>
    /// Gets the number of encoder/decoder blocks.
    /// </summary>
    public int NumBlocks => _numBlocks;

    /// <summary>
    /// Gets the base number of channels.
    /// </summary>
    public int NumChannels => _numChannels;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets a value indicating whether this layer supports GPU execution.
    /// </summary>
    protected override bool SupportsGpuExecution => false;

    /// <inheritdoc />
    public override bool SupportsJitCompilation
    {
        get
        {
            if (!_convFirst.SupportsJitCompilation) return false;
            if (!_convLast.SupportsJitCompilation) return false;

            foreach (var block in _encoderBlocks)
            {
                if (!block.SupportsJitCompilation) return false;
            }

            foreach (var block in _decoderBlocks)
            {
                if (!block.SupportsJitCompilation) return false;
            }

            return true;
        }
    }

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new U-Net discriminator.
    /// </summary>
    /// <param name="inputHeight">Height of input image.</param>
    /// <param name="inputWidth">Width of input image.</param>
    /// <param name="inputChannels">Number of input channels (3 for RGB).</param>
    /// <param name="numChannels">Base number of channels. Default: 64.</param>
    /// <param name="numBlocks">Number of encoder/decoder blocks. Default: 4.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Create a U-Net discriminator for Real-ESRGAN:
    /// <code>
    /// var discriminator = new UNetDiscriminator&lt;float&gt;(
    ///     inputHeight: 256,
    ///     inputWidth: 256,
    ///     inputChannels: 3,
    ///     numChannels: 64,
    ///     numBlocks: 4
    /// );
    /// </code>
    /// </para>
    /// </remarks>
    public UNetDiscriminator(
        int inputHeight,
        int inputWidth,
        int inputChannels = 3,
        int numChannels = 64,
        int numBlocks = 4)
        : base(
            [inputChannels, inputHeight, inputWidth],
            [1, inputHeight, inputWidth]) // Per-pixel output
    {
        if (numBlocks <= 0)
            throw new ArgumentOutOfRangeException(nameof(numBlocks), "Number of blocks must be positive.");
        if (numChannels <= 0)
            throw new ArgumentOutOfRangeException(nameof(numChannels), "Number of channels must be positive.");

        _numBlocks = numBlocks;
        _numChannels = numChannels;
        _leakyReLU = new LeakyReLUActivation<T>(0.2);

        // Initial convolution: inputChannels → numChannels
        _convFirst = new ConvolutionalLayer<T>(
            inputDepth: inputChannels,
            inputHeight: inputHeight,
            inputWidth: inputWidth,
            outputDepth: numChannels,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            activationFunction: null);

        // Encoder blocks (progressively downsample and increase channels)
        _encoderBlocks = new UNetConvBlock<T>[numBlocks];
        int currentChannels = numChannels;
        int currentHeight = inputHeight;
        int currentWidth = inputWidth;

        for (int i = 0; i < numBlocks; i++)
        {
            int outChannels = Math.Min(currentChannels * 2, numChannels * 8); // Cap at 8x base channels
            _encoderBlocks[i] = new UNetConvBlock<T>(
                currentChannels, outChannels, currentHeight, currentWidth, downsample: true);

            currentChannels = outChannels;
            currentHeight = (currentHeight + 1) / 2; // Account for stride=2
            currentWidth = (currentWidth + 1) / 2;
        }

        // Decoder blocks (progressively upsample, use skip connections)
        _decoderBlocks = new UNetUpBlock<T>[numBlocks];

        for (int i = numBlocks - 1; i >= 0; i--)
        {
            int skipChannels = i == 0 ? numChannels : Math.Min(numChannels * (1 << i), numChannels * 8);
            int outChannels = i == 0 ? numChannels : Math.Min(numChannels * (1 << (i - 1)), numChannels * 8);

            _decoderBlocks[numBlocks - 1 - i] = new UNetUpBlock<T>(
                currentChannels, skipChannels, outChannels,
                currentHeight, currentWidth);

            currentChannels = outChannels;
            currentHeight *= 2;
            currentWidth *= 2;
        }

        // Final convolution: numChannels → 1 (per-pixel prediction)
        _convLast = new ConvolutionalLayer<T>(
            inputDepth: numChannels,
            inputHeight: inputHeight,
            inputWidth: inputWidth,
            outputDepth: 1,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            activationFunction: null);
    }

    #endregion

    #region Forward Pass

    /// <inheritdoc />
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;

        // Initial conv + activation
        var x = _convFirst.Forward(input);
        x = ApplyLeakyReLU(x);

        // Encoder path - store skip connections
        _skipConnections = new Tensor<T>[_numBlocks];
        for (int i = 0; i < _numBlocks; i++)
        {
            _skipConnections[i] = x; // Store before downsampling
            x = _encoderBlocks[i].Forward(x);
        }

        // Decoder path - use skip connections
        for (int i = 0; i < _numBlocks; i++)
        {
            int skipIdx = _numBlocks - 1 - i;
            x = _decoderBlocks[i].Forward(x, _skipConnections[skipIdx]);
        }

        // Final conv
        x = _convLast.Forward(x);

        return x;
    }

    #endregion

    #region Backward Pass

    /// <inheritdoc />
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _skipConnections == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        var grad = outputGradient;

        // Backward through final conv
        grad = _convLast.Backward(grad);

        // Backward through decoder
        var skipGrads = new Tensor<T>[_numBlocks];
        for (int i = _numBlocks - 1; i >= 0; i--)
        {
            var (mainGrad, skipGrad) = _decoderBlocks[i].BackwardWithSkip(grad);
            grad = mainGrad;
            skipGrads[_numBlocks - 1 - i] = skipGrad;
        }

        // Backward through encoder (combine with skip gradients)
        for (int i = _numBlocks - 1; i >= 0; i--)
        {
            // Add skip gradient
            grad = AddTensors(grad, skipGrads[i]);
            grad = _encoderBlocks[i].Backward(grad);
        }

        // Backward through initial conv activation and conv
        grad = BackwardLeakyReLU(_convFirst.Forward(_lastInput), grad);
        grad = _convFirst.Backward(grad);

        return grad;
    }

    #endregion

    #region Helper Methods

    private Tensor<T> ApplyLeakyReLU(Tensor<T> input)
    {
        var output = new Tensor<T>(input.Shape);
        for (int i = 0; i < input.Length; i++)
        {
            output.Data[i] = _leakyReLU.Activate(input.Data[i]);
        }
        return output;
    }

    private Tensor<T> BackwardLeakyReLU(Tensor<T> forwardInput, Tensor<T> gradient)
    {
        var output = new Tensor<T>(gradient.Shape);
        for (int i = 0; i < gradient.Length; i++)
        {
            output.Data[i] = NumOps.Multiply(
                gradient.Data[i],
                _leakyReLU.Derivative(forwardInput.Data[i]));
        }
        return output;
    }

    private Tensor<T> AddTensors(Tensor<T> a, Tensor<T> b)
    {
        var output = new Tensor<T>(a.Shape);
        for (int i = 0; i < a.Length; i++)
        {
            output.Data[i] = NumOps.Add(a.Data[i], b.Data[i]);
        }
        return output;
    }

    #endregion

    #region Parameter Management

    /// <inheritdoc />
    public override void UpdateParameters(T learningRate)
    {
        _convFirst.UpdateParameters(learningRate);
        foreach (var block in _encoderBlocks)
        {
            block.UpdateParameters(learningRate);
        }
        foreach (var block in _decoderBlocks)
        {
            block.UpdateParameters(learningRate);
        }
        _convLast.UpdateParameters(learningRate);
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var allParams = new List<T>();

        AddParamsToList(allParams, _convFirst.GetParameters());
        foreach (var block in _encoderBlocks)
        {
            AddParamsToList(allParams, block.GetParameters());
        }
        foreach (var block in _decoderBlocks)
        {
            AddParamsToList(allParams, block.GetParameters());
        }
        AddParamsToList(allParams, _convLast.GetParameters());

        return new Vector<T>([.. allParams]);
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        int offset = 0;

        offset = SetLayerParams(_convFirst, parameters, offset);
        foreach (var block in _encoderBlocks)
        {
            offset = SetLayerParams(block, parameters, offset);
        }
        foreach (var block in _decoderBlocks)
        {
            offset = SetLayerParams(block, parameters, offset);
        }
        SetLayerParams(_convLast, parameters, offset);
    }

    private static void AddParamsToList(List<T> list, Vector<T> parameters)
    {
        for (int i = 0; i < parameters.Length; i++)
        {
            list.Add(parameters[i]);
        }
    }

    private static int SetLayerParams(ILayer<T> layer, Vector<T> parameters, int offset)
    {
        int count = layer.GetParameters().Length;
        layer.SetParameters(parameters.SubVector(offset, count));
        return offset + count;
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        _lastInput = null;
        _skipConnections = null;

        _convFirst.ResetState();
        foreach (var block in _encoderBlocks)
        {
            block.ResetState();
        }
        foreach (var block in _decoderBlocks)
        {
            block.ResetState();
        }
        _convLast.ResetState();
    }

    #endregion

    #region JIT Compilation

    /// <inheritdoc />
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes is null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (InputShape is null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured.");

        var symbolicInput = new Tensor<T>(new int[] { 1 }.Concat(InputShape).ToArray());
        var inputNode = TensorOperations<T>.Variable(symbolicInput, "input");
        inputNodes.Add(inputNode);

        return BuildComputationGraph(inputNode, "");
    }

    /// <inheritdoc />
    public ComputationNode<T> BuildComputationGraph(ComputationNode<T> inputNode, string namePrefix)
    {
        // Initial conv + LeakyReLU
        var x = BuildConvNode(_convFirst, inputNode, $"{namePrefix}conv_first_");
        x = TensorOperations<T>.LeakyReLU(x, 0.2);

        // Encoder path - store skip connections
        var skipConnections = new ComputationNode<T>[_numBlocks];
        for (int i = 0; i < _numBlocks; i++)
        {
            skipConnections[i] = x;
            x = _encoderBlocks[i].BuildComputationGraph(x, $"{namePrefix}enc{i}_");
        }

        // Decoder path - use skip connections
        for (int i = 0; i < _numBlocks; i++)
        {
            int skipIdx = _numBlocks - 1 - i;
            x = _decoderBlocks[i].BuildComputationGraph(x, skipConnections[skipIdx], $"{namePrefix}dec{i}_");
        }

        // Final conv
        x = BuildConvNode(_convLast, x, $"{namePrefix}conv_last_");

        return x;
    }

    private static ComputationNode<T> BuildConvNode(ConvolutionalLayer<T> conv, ComputationNode<T> input, string namePrefix)
    {
        var biases = conv.GetBiases();
        return TensorOperations<T>.Conv2D(
            input,
            TensorOperations<T>.Constant(conv.GetFilters(), $"{namePrefix}kernel"),
            biases is not null ? TensorOperations<T>.Constant(biases, $"{namePrefix}bias") : null,
            stride: new int[] { conv.Stride, conv.Stride },
            padding: new int[] { conv.Padding, conv.Padding });
    }

    #endregion
}

#region Helper Blocks

/// <summary>
/// Convolutional block for U-Net encoder with optional downsampling.
/// </summary>
internal class UNetConvBlock<T> : LayerBase<T>, IChainableComputationGraph<T>
{
    private readonly ConvolutionalLayer<T> _conv1;
    private readonly ConvolutionalLayer<T> _conv2;
    private readonly LeakyReLUActivation<T> _leakyReLU;
    private readonly bool _downsample;

    private Tensor<T>? _lastInput;
    private Tensor<T>? _conv1Output;      // After LeakyReLU (input to conv2)
    private Tensor<T>? _conv1RawOutput;   // Before LeakyReLU (for backward)
    private Tensor<T>? _conv2RawOutput;   // Before LeakyReLU (for backward)

    public UNetConvBlock(int inChannels, int outChannels, int height, int width, bool downsample)
        : base(
            [inChannels, height, width],
            downsample ? [outChannels, (height + 1) / 2, (width + 1) / 2] : [outChannels, height, width])
    {
        _downsample = downsample;
        _leakyReLU = new LeakyReLUActivation<T>(0.2);

        int outHeight = downsample ? (height + 1) / 2 : height;
        int outWidth = downsample ? (width + 1) / 2 : width;

        // First conv (with optional stride for downsampling)
        _conv1 = new ConvolutionalLayer<T>(
            inputDepth: inChannels,
            inputHeight: height,
            inputWidth: width,
            outputDepth: outChannels,
            kernelSize: 3,
            stride: downsample ? 2 : 1,
            padding: 1,
            activationFunction: null);

        // Second conv (always stride 1)
        _conv2 = new ConvolutionalLayer<T>(
            inputDepth: outChannels,
            inputHeight: outHeight,
            inputWidth: outWidth,
            outputDepth: outChannels,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            activationFunction: null);
    }

    public override bool SupportsTraining => true;

    public override bool SupportsJitCompilation =>
        _conv1.SupportsJitCompilation && _conv2.SupportsJitCompilation;

    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;

        // Conv1 + LeakyReLU
        _conv1RawOutput = _conv1.Forward(input);
        _conv1Output = ApplyLeakyReLU(_conv1RawOutput);

        // Conv2 + LeakyReLU
        _conv2RawOutput = _conv2.Forward(_conv1Output);
        var output = ApplyLeakyReLU(_conv2RawOutput);

        return output;
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _conv1Output == null || _conv1RawOutput == null || _conv2RawOutput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // Backward through LeakyReLU after conv2 (use cached raw output)
        var grad = BackwardLeakyReLU(_conv2RawOutput, outputGradient);
        grad = _conv2.Backward(grad);

        // Backward through LeakyReLU after conv1 (use cached raw output)
        grad = BackwardLeakyReLU(_conv1RawOutput, grad);
        grad = _conv1.Backward(grad);

        return grad;
    }

    private Tensor<T> ApplyLeakyReLU(Tensor<T> input)
    {
        var output = new Tensor<T>(input.Shape);
        for (int i = 0; i < input.Length; i++)
        {
            output.Data[i] = _leakyReLU.Activate(input.Data[i]);
        }
        return output;
    }

    private Tensor<T> BackwardLeakyReLU(Tensor<T> forwardInput, Tensor<T> gradient)
    {
        var output = new Tensor<T>(gradient.Shape);
        for (int i = 0; i < gradient.Length; i++)
        {
            output.Data[i] = NumOps.Multiply(
                gradient.Data[i],
                _leakyReLU.Derivative(forwardInput.Data[i]));
        }
        return output;
    }

    public override void UpdateParameters(T learningRate)
    {
        _conv1.UpdateParameters(learningRate);
        _conv2.UpdateParameters(learningRate);
    }

    public override Vector<T> GetParameters()
    {
        var params1 = _conv1.GetParameters();
        var params2 = _conv2.GetParameters();
        var result = new T[params1.Length + params2.Length];
        for (int i = 0; i < params1.Length; i++) result[i] = params1[i];
        for (int i = 0; i < params2.Length; i++) result[params1.Length + i] = params2[i];
        return new Vector<T>(result);
    }

    public override void SetParameters(Vector<T> parameters)
    {
        int count1 = _conv1.GetParameters().Length;
        _conv1.SetParameters(parameters.SubVector(0, count1));
        _conv2.SetParameters(parameters.SubVector(count1, parameters.Length - count1));
    }

    public override void ResetState()
    {
        _lastInput = null;
        _conv1Output = null;
        _conv1RawOutput = null;
        _conv2RawOutput = null;
        _conv1.ResetState();
        _conv2.ResetState();
    }

    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        var symbolicInput = new Tensor<T>(new int[] { 1 }.Concat(InputShape).ToArray());
        var inputNode = TensorOperations<T>.Variable(symbolicInput, "input");
        inputNodes.Add(inputNode);
        return BuildComputationGraph(inputNode, "");
    }

    public ComputationNode<T> BuildComputationGraph(ComputationNode<T> inputNode, string namePrefix)
    {
        var biases1 = _conv1.GetBiases();
        var x = TensorOperations<T>.Conv2D(
            inputNode,
            TensorOperations<T>.Constant(_conv1.GetFilters(), $"{namePrefix}conv1_kernel"),
            biases1 is not null ? TensorOperations<T>.Constant(biases1, $"{namePrefix}conv1_bias") : null,
            stride: new int[] { _downsample ? 2 : 1, _downsample ? 2 : 1 },
            padding: new int[] { 1, 1 });
        x = TensorOperations<T>.LeakyReLU(x, 0.2);

        var biases2 = _conv2.GetBiases();
        x = TensorOperations<T>.Conv2D(
            x,
            TensorOperations<T>.Constant(_conv2.GetFilters(), $"{namePrefix}conv2_kernel"),
            biases2 is not null ? TensorOperations<T>.Constant(biases2, $"{namePrefix}conv2_bias") : null,
            stride: new int[] { 1, 1 },
            padding: new int[] { 1, 1 });
        x = TensorOperations<T>.LeakyReLU(x, 0.2);

        return x;
    }
}

/// <summary>
/// Upsampling block for U-Net decoder with skip connection concatenation.
/// </summary>
internal class UNetUpBlock<T> : LayerBase<T>, IChainableComputationGraph<T>
{
    private readonly UpsamplingLayer<T> _upsample;
    private readonly ConvolutionalLayer<T> _conv1;
    private readonly ConvolutionalLayer<T> _conv2;
    private readonly LeakyReLUActivation<T> _leakyReLU;
    private readonly int _skipChannels;

    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastSkip;
    private Tensor<T>? _upsampledInput;
    private Tensor<T>? _concatenated;
    private Tensor<T>? _conv1Output;

    public UNetUpBlock(int inChannels, int skipChannels, int outChannels, int height, int width)
        : base(
            [inChannels, height, width],
            [outChannels, height * 2, width * 2])
    {
        _skipChannels = skipChannels;
        _leakyReLU = new LeakyReLUActivation<T>(0.2);

        int outHeight = height * 2;
        int outWidth = width * 2;

        // Bilinear upsampling
        _upsample = new UpsamplingLayer<T>(
            [inChannels, height, width],
            scaleFactor: 2);

        // Conv after concatenation (inChannels + skipChannels -> outChannels)
        _conv1 = new ConvolutionalLayer<T>(
            inputDepth: inChannels + skipChannels,
            inputHeight: outHeight,
            inputWidth: outWidth,
            outputDepth: outChannels,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            activationFunction: null);

        _conv2 = new ConvolutionalLayer<T>(
            inputDepth: outChannels,
            inputHeight: outHeight,
            inputWidth: outWidth,
            outputDepth: outChannels,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            activationFunction: null);
    }

    public override bool SupportsTraining => true;

    public override bool SupportsJitCompilation =>
        _upsample.SupportsJitCompilation && _conv1.SupportsJitCompilation && _conv2.SupportsJitCompilation;

    public override Tensor<T> Forward(Tensor<T> input)
    {
        return Forward(input, null);
    }

    public Tensor<T> Forward(Tensor<T> input, Tensor<T>? skip)
    {
        _lastInput = input;
        _lastSkip = skip;

        // Upsample
        _upsampledInput = _upsample.Forward(input);

        // Concatenate with skip connection
        Tensor<T> x;
        if (skip != null)
        {
            _concatenated = ConcatenateChannels(_upsampledInput, skip);
            x = _concatenated;
        }
        else
        {
            _concatenated = null;
            x = _upsampledInput;
        }

        // Conv1 + LeakyReLU
        x = _conv1.Forward(x);
        x = ApplyLeakyReLU(x);
        _conv1Output = x;

        // Conv2 + LeakyReLU
        x = _conv2.Forward(x);
        x = ApplyLeakyReLU(x);

        return x;
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        var (mainGrad, _) = BackwardWithSkip(outputGradient);
        return mainGrad;
    }

    public (Tensor<T> mainGrad, Tensor<T> skipGrad) BackwardWithSkip(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _upsampledInput == null || _conv1Output == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // Backward through conv2 + LeakyReLU
        var grad = BackwardLeakyReLU(_conv2.Forward(_conv1Output), outputGradient);
        grad = _conv2.Backward(grad);

        // Backward through conv1 + LeakyReLU
        grad = BackwardLeakyReLU(_conv1.Forward(_concatenated ?? _upsampledInput), grad);
        grad = _conv1.Backward(grad);

        Tensor<T> skipGrad;
        Tensor<T> upsampleGrad;

        // Split gradient if skip connection was used
        if (_concatenated != null && _lastSkip != null)
        {
            // Handle both 3D [C, H, W] and 4D [N, C, H, W] tensors
            bool has4D = _upsampledInput.Shape.Length == 4;
            int mainChannels = has4D ? _upsampledInput.Shape[1] : _upsampledInput.Shape[0];
            int skipChannels = has4D ? _lastSkip.Shape[1] : _lastSkip.Shape[0];
            (upsampleGrad, skipGrad) = SplitGradient(grad, mainChannels, skipChannels);
        }
        else
        {
            upsampleGrad = grad;
            skipGrad = new Tensor<T>(_lastSkip?.Shape ?? [1]);
        }

        // Backward through upsample
        var inputGrad = _upsample.Backward(upsampleGrad);

        return (inputGrad, skipGrad);
    }

    private Tensor<T> ConcatenateChannels(Tensor<T> a, Tensor<T> b)
    {
        // Handle both 3D [C, H, W] and 4D [N, C, H, W] tensors
        bool has4D = a.Shape.Length == 4;
        int batch = has4D ? a.Shape[0] : 1;
        int channelsA = has4D ? a.Shape[1] : a.Shape[0];
        int channelsB = has4D ? b.Shape[1] : b.Shape[0];
        int height = has4D ? a.Shape[2] : a.Shape[1];
        int width = has4D ? a.Shape[3] : a.Shape[2];
        int spatialSize = height * width;

        var resultShape = has4D
            ? new int[] { batch, channelsA + channelsB, height, width }
            : new int[] { channelsA + channelsB, height, width };
        var result = new Tensor<T>(resultShape);

        for (int n = 0; n < batch; n++)
        {
            int batchOffsetA = n * channelsA * spatialSize;
            int batchOffsetB = n * channelsB * spatialSize;
            int batchOffsetResult = n * (channelsA + channelsB) * spatialSize;

            for (int c = 0; c < channelsA; c++)
            {
                for (int hw = 0; hw < spatialSize; hw++)
                {
                    result.Data[batchOffsetResult + c * spatialSize + hw] = a.Data[batchOffsetA + c * spatialSize + hw];
                }
            }

            for (int c = 0; c < channelsB; c++)
            {
                for (int hw = 0; hw < spatialSize; hw++)
                {
                    result.Data[batchOffsetResult + (channelsA + c) * spatialSize + hw] = b.Data[batchOffsetB + c * spatialSize + hw];
                }
            }
        }

        return result;
    }

    private (Tensor<T> first, Tensor<T> second) SplitGradient(Tensor<T> grad, int firstChannels, int secondChannels)
    {
        // Handle both 3D [C, H, W] and 4D [N, C, H, W] tensors
        bool has4D = grad.Shape.Length == 4;
        int batch = has4D ? grad.Shape[0] : 1;
        int height = has4D ? grad.Shape[2] : grad.Shape[1];
        int width = has4D ? grad.Shape[3] : grad.Shape[2];
        int spatialSize = height * width;

        var firstShape = has4D
            ? new int[] { batch, firstChannels, height, width }
            : new int[] { firstChannels, height, width };
        var secondShape = has4D
            ? new int[] { batch, secondChannels, height, width }
            : new int[] { secondChannels, height, width };

        var first = new Tensor<T>(firstShape);
        var second = new Tensor<T>(secondShape);
        int totalChannels = firstChannels + secondChannels;

        for (int n = 0; n < batch; n++)
        {
            int batchOffsetGrad = n * totalChannels * spatialSize;
            int batchOffsetFirst = n * firstChannels * spatialSize;
            int batchOffsetSecond = n * secondChannels * spatialSize;

            for (int c = 0; c < firstChannels; c++)
            {
                for (int hw = 0; hw < spatialSize; hw++)
                {
                    first.Data[batchOffsetFirst + c * spatialSize + hw] = grad.Data[batchOffsetGrad + c * spatialSize + hw];
                }
            }

            for (int c = 0; c < secondChannels; c++)
            {
                for (int hw = 0; hw < spatialSize; hw++)
                {
                    second.Data[batchOffsetSecond + c * spatialSize + hw] = grad.Data[batchOffsetGrad + (firstChannels + c) * spatialSize + hw];
                }
            }
        }

        return (first, second);
    }

    private Tensor<T> ApplyLeakyReLU(Tensor<T> input)
    {
        var output = new Tensor<T>(input.Shape);
        for (int i = 0; i < input.Length; i++)
        {
            output.Data[i] = _leakyReLU.Activate(input.Data[i]);
        }
        return output;
    }

    private Tensor<T> BackwardLeakyReLU(Tensor<T> forwardInput, Tensor<T> gradient)
    {
        var output = new Tensor<T>(gradient.Shape);
        for (int i = 0; i < gradient.Length; i++)
        {
            output.Data[i] = NumOps.Multiply(
                gradient.Data[i],
                _leakyReLU.Derivative(forwardInput.Data[i]));
        }
        return output;
    }

    public override void UpdateParameters(T learningRate)
    {
        _conv1.UpdateParameters(learningRate);
        _conv2.UpdateParameters(learningRate);
    }

    public override Vector<T> GetParameters()
    {
        var params1 = _conv1.GetParameters();
        var params2 = _conv2.GetParameters();
        var result = new T[params1.Length + params2.Length];
        for (int i = 0; i < params1.Length; i++) result[i] = params1[i];
        for (int i = 0; i < params2.Length; i++) result[params1.Length + i] = params2[i];
        return new Vector<T>(result);
    }

    public override void SetParameters(Vector<T> parameters)
    {
        int count1 = _conv1.GetParameters().Length;
        _conv1.SetParameters(parameters.SubVector(0, count1));
        _conv2.SetParameters(parameters.SubVector(count1, parameters.Length - count1));
    }

    public override void ResetState()
    {
        _lastInput = null;
        _lastSkip = null;
        _upsampledInput = null;
        _concatenated = null;
        _conv1Output = null;
        _upsample.ResetState();
        _conv1.ResetState();
        _conv2.ResetState();
    }

    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        var symbolicInput = new Tensor<T>(new int[] { 1 }.Concat(InputShape).ToArray());
        var inputNode = TensorOperations<T>.Variable(symbolicInput, "input");
        inputNodes.Add(inputNode);
        return BuildComputationGraph(inputNode, null, "");
    }

    public ComputationNode<T> BuildComputationGraph(ComputationNode<T> inputNode, string namePrefix)
    {
        return BuildComputationGraph(inputNode, null, namePrefix);
    }

    public ComputationNode<T> BuildComputationGraph(ComputationNode<T> inputNode, ComputationNode<T>? skipNode, string namePrefix)
    {
        // Upsample (bilinear)
        var x = TensorOperations<T>.Upsample(inputNode, 2);

        // Concatenate with skip
        if (skipNode != null)
        {
            x = TensorOperations<T>.Concat([x, skipNode], axis: 1);
        }

        // Conv1 + LeakyReLU
        var biases1 = _conv1.GetBiases();
        x = TensorOperations<T>.Conv2D(
            x,
            TensorOperations<T>.Constant(_conv1.GetFilters(), $"{namePrefix}conv1_kernel"),
            biases1 is not null ? TensorOperations<T>.Constant(biases1, $"{namePrefix}conv1_bias") : null,
            stride: new int[] { 1, 1 },
            padding: new int[] { 1, 1 });
        x = TensorOperations<T>.LeakyReLU(x, 0.2);

        // Conv2 + LeakyReLU
        var biases2 = _conv2.GetBiases();
        x = TensorOperations<T>.Conv2D(
            x,
            TensorOperations<T>.Constant(_conv2.GetFilters(), $"{namePrefix}conv2_kernel"),
            biases2 is not null ? TensorOperations<T>.Constant(biases2, $"{namePrefix}conv2_bias") : null,
            stride: new int[] { 1, 1 },
            padding: new int[] { 1, 1 });
        x = TensorOperations<T>.LeakyReLU(x, 0.2);

        return x;
    }
}

#endregion
