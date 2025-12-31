using AiDotNet.ActivationFunctions;
using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// RRDBNet Generator - the full generator architecture from ESRGAN and Real-ESRGAN.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This implements the complete RRDBNet generator from the ESRGAN paper (Wang et al., 2018).
/// It combines multiple RRDB blocks with upsampling for image super-resolution.
/// </para>
/// <para>
/// The architecture is:
/// <code>
/// Input (3 channels, LR image)
///   ↓
/// Conv1: 3 → numFeatures (initial feature extraction)
///   ↓
/// RRDB × numRRDBBlocks (deep feature extraction)
///   ↓
/// Trunk Conv: numFeatures → numFeatures
///   ↓
/// + (global residual connection from Conv1 output)
///   ↓
/// Upsampling Blocks (PixelShuffle 2x each, repeated for scale)
///   ↓
/// HR Conv: numFeatures → numFeatures, LeakyReLU
///   ↓
/// Final Conv: numFeatures → 3 (output channels)
///   ↓
/// Output (3 channels, HR image)
/// </code>
/// </para>
/// <para>
/// <b>For Beginners:</b> This is the "brain" of Real-ESRGAN that transforms low-resolution
/// images into high-resolution ones.
///
/// Key components:
/// - **Initial Conv**: Extracts basic features from the input image
/// - **RRDB Blocks**: 23 deep blocks that learn how to enhance details
/// - **Trunk Conv + Residual**: Combines deep features with initial features
/// - **Upsampling**: Makes the image bigger (2x or 4x depending on scale)
/// - **Final Convs**: Produces the final RGB output image
///
/// The default parameters (64 features, 32 growth, 23 RRDBs, 4x scale) are from the
/// Real-ESRGAN paper and produce excellent results for general image super-resolution.
/// </para>
/// <para>
/// <b>Reference:</b> Wang et al., "Real-ESRGAN: Training Real-World Blind Super-Resolution
/// with Pure Synthetic Data", ICCV 2021. https://arxiv.org/abs/2107.10833
/// </para>
/// </remarks>
public class RRDBNetGenerator<T> : LayerBase<T>, IChainableComputationGraph<T>
{
    #region Fields

    /// <summary>
    /// Initial convolution: 3 → numFeatures.
    /// </summary>
    private readonly ConvolutionalLayer<T> _convFirst;

    /// <summary>
    /// The RRDB blocks for deep feature extraction.
    /// </summary>
    private readonly RRDBLayer<T>[] _rrdbBlocks;

    /// <summary>
    /// Trunk convolution after RRDB blocks.
    /// </summary>
    private readonly ConvolutionalLayer<T> _trunkConv;

    /// <summary>
    /// Upsampling convolutions (one before each PixelShuffle).
    /// For 4x upscaling: 2 convs (64 → 256 each for 2x PixelShuffle).
    /// </summary>
    private readonly ConvolutionalLayer<T>[] _upsampleConvs;

    /// <summary>
    /// PixelShuffle layers for upsampling.
    /// </summary>
    private readonly PixelShuffleLayer<T>[] _pixelShuffleLayers;

    /// <summary>
    /// High-resolution convolution (after upsampling).
    /// </summary>
    private readonly ConvolutionalLayer<T> _hrConv;

    /// <summary>
    /// Final convolution: numFeatures → outputChannels (typically 3 for RGB).
    /// </summary>
    private readonly ConvolutionalLayer<T> _convLast;

    /// <summary>
    /// LeakyReLU activation with negative slope 0.2.
    /// </summary>
    private readonly LeakyReLUActivation<T> _leakyReLU;

    /// <summary>
    /// Number of feature channels.
    /// </summary>
    private readonly int _numFeatures;

    /// <summary>
    /// Upscaling factor (2 or 4).
    /// </summary>
    private readonly int _scale;

    /// <summary>
    /// Cached input for backpropagation.
    /// </summary>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Cached conv1 output for trunk residual.
    /// </summary>
    private Tensor<T>? _conv1Output;

    /// <summary>
    /// Cached intermediate outputs for backpropagation.
    /// </summary>
    private Tensor<T>[]? _rrdbOutputs;

    /// <summary>
    /// Cached upsampling outputs for backpropagation.
    /// </summary>
    private Tensor<T>[]? _upsampleOutputs;

    #endregion

    #region Properties

    /// <summary>
    /// Gets the number of RRDB blocks.
    /// </summary>
    public int NumRRDBBlocks => _rrdbBlocks.Length;

    /// <summary>
    /// Gets the number of feature channels.
    /// </summary>
    public int NumFeatures => _numFeatures;

    /// <summary>
    /// Gets the upscaling factor.
    /// </summary>
    public int Scale => _scale;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <inheritdoc />
    public override bool SupportsJitCompilation
    {
        get
        {
            // Check all sub-components support JIT
            if (!_convFirst.SupportsJitCompilation) return false;
            if (!_trunkConv.SupportsJitCompilation) return false;
            if (!_hrConv.SupportsJitCompilation) return false;
            if (!_convLast.SupportsJitCompilation) return false;

            foreach (var rrdb in _rrdbBlocks)
            {
                if (!rrdb.SupportsJitCompilation) return false;
            }

            foreach (var conv in _upsampleConvs)
            {
                if (!conv.SupportsJitCompilation) return false;
            }

            foreach (var ps in _pixelShuffleLayers)
            {
                if (!ps.SupportsJitCompilation) return false;
            }

            return true;
        }
    }

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new RRDBNet generator.
    /// </summary>
    /// <param name="inputHeight">Height of input image.</param>
    /// <param name="inputWidth">Width of input image.</param>
    /// <param name="inputChannels">Number of input channels (3 for RGB).</param>
    /// <param name="outputChannels">Number of output channels (3 for RGB).</param>
    /// <param name="numFeatures">Number of feature channels. Default: 64 (from paper).</param>
    /// <param name="growthChannels">Growth channels for RDB. Default: 32 (from paper).</param>
    /// <param name="numRRDBBlocks">Number of RRDB blocks. Default: 23 (from paper).</param>
    /// <param name="scale">Upscaling factor (2 or 4). Default: 4.</param>
    /// <param name="residualScale">Residual scaling factor. Default: 0.2 (from paper).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Create a Real-ESRGAN generator for 4x super-resolution:
    /// <code>
    /// var generator = new RRDBNetGenerator&lt;float&gt;(
    ///     inputHeight: 64,
    ///     inputWidth: 64,
    ///     inputChannels: 3,      // RGB input
    ///     outputChannels: 3,     // RGB output
    ///     numFeatures: 64,       // Paper default
    ///     growthChannels: 32,    // Paper default
    ///     numRRDBBlocks: 23,     // Paper default (deep network)
    ///     scale: 4               // 4x upscaling
    /// );
    /// // Input: 64x64 → Output: 256x256
    /// </code>
    /// </para>
    /// </remarks>
    public RRDBNetGenerator(
        int inputHeight,
        int inputWidth,
        int inputChannels = 3,
        int outputChannels = 3,
        int numFeatures = 64,
        int growthChannels = 32,
        int numRRDBBlocks = 23,
        int scale = 4,
        double residualScale = 0.2)
        : base(
            [inputChannels, inputHeight, inputWidth],
            [outputChannels, inputHeight * scale, inputWidth * scale])
    {
        if (scale != 2 && scale != 4)
            throw new ArgumentOutOfRangeException(nameof(scale), "Scale must be 2 or 4.");
        if (numRRDBBlocks <= 0)
            throw new ArgumentOutOfRangeException(nameof(numRRDBBlocks), "Number of RRDB blocks must be positive.");
        if (numFeatures <= 0)
            throw new ArgumentOutOfRangeException(nameof(numFeatures), "Number of features must be positive.");

        _numFeatures = numFeatures;
        _scale = scale;
        _leakyReLU = new LeakyReLUActivation<T>(0.2);

        // Initial convolution: inputChannels → numFeatures
        _convFirst = new ConvolutionalLayer<T>(
            inputDepth: inputChannels,
            inputHeight: inputHeight,
            inputWidth: inputWidth,
            outputDepth: numFeatures,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            activationFunction: null);

        // RRDB blocks
        _rrdbBlocks = new RRDBLayer<T>[numRRDBBlocks];
        for (int i = 0; i < numRRDBBlocks; i++)
        {
            _rrdbBlocks[i] = new RRDBLayer<T>(
                numFeatures: numFeatures,
                growthChannels: growthChannels,
                inputHeight: inputHeight,
                inputWidth: inputWidth,
                residualScale: residualScale);
        }

        // Trunk convolution (after RRDB blocks)
        _trunkConv = new ConvolutionalLayer<T>(
            inputDepth: numFeatures,
            inputHeight: inputHeight,
            inputWidth: inputWidth,
            outputDepth: numFeatures,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            activationFunction: null);

        // Upsampling: for 4x scale, we need 2 stages of 2x upsampling
        // Each stage: Conv (numFeatures → numFeatures * 4) + PixelShuffle(2) + LeakyReLU
        int numUpsampleStages = scale == 4 ? 2 : 1;
        _upsampleConvs = new ConvolutionalLayer<T>[numUpsampleStages];
        _pixelShuffleLayers = new PixelShuffleLayer<T>[numUpsampleStages];

        int currentHeight = inputHeight;
        int currentWidth = inputWidth;

        for (int i = 0; i < numUpsampleStages; i++)
        {
            // Conv: numFeatures → numFeatures * 4 (for 2x PixelShuffle)
            _upsampleConvs[i] = new ConvolutionalLayer<T>(
                inputDepth: numFeatures,
                inputHeight: currentHeight,
                inputWidth: currentWidth,
                outputDepth: numFeatures * 4, // 4x channels for 2x spatial upscale
                kernelSize: 3,
                stride: 1,
                padding: 1,
                activationFunction: null);

            // PixelShuffle: 2x upscaling
            _pixelShuffleLayers[i] = new PixelShuffleLayer<T>(
                [numFeatures * 4, currentHeight, currentWidth],
                upscaleFactor: 2);

            currentHeight *= 2;
            currentWidth *= 2;
        }

        // HR convolution (after upsampling)
        _hrConv = new ConvolutionalLayer<T>(
            inputDepth: numFeatures,
            inputHeight: currentHeight,
            inputWidth: currentWidth,
            outputDepth: numFeatures,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            activationFunction: null);

        // Final convolution: numFeatures → outputChannels
        _convLast = new ConvolutionalLayer<T>(
            inputDepth: numFeatures,
            inputHeight: currentHeight,
            inputWidth: currentWidth,
            outputDepth: outputChannels,
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

        // Initial feature extraction
        _conv1Output = _convFirst.Forward(input);
        var x = _conv1Output;

        // Deep feature extraction through RRDB blocks
        _rrdbOutputs = new Tensor<T>[_rrdbBlocks.Length];
        for (int i = 0; i < _rrdbBlocks.Length; i++)
        {
            x = _rrdbBlocks[i].Forward(x);
            _rrdbOutputs[i] = x;
        }

        // Trunk convolution + global residual
        var trunk = _trunkConv.Forward(x);
        x = AddTensors(trunk, _conv1Output); // Global residual connection

        // Upsampling
        _upsampleOutputs = new Tensor<T>[_upsampleConvs.Length * 2]; // Conv output + PixelShuffle output
        for (int i = 0; i < _upsampleConvs.Length; i++)
        {
            x = _upsampleConvs[i].Forward(x);
            _upsampleOutputs[i * 2] = x;
            x = _pixelShuffleLayers[i].Forward(x);
            x = ApplyLeakyReLU(x);
            _upsampleOutputs[i * 2 + 1] = x;
        }

        // HR convolution + activation
        x = _hrConv.Forward(x);
        x = ApplyLeakyReLU(x);

        // Final convolution
        x = _convLast.Forward(x);

        return x;
    }

    #endregion

    #region Backward Pass

    /// <inheritdoc />
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _conv1Output == null || _rrdbOutputs == null || _upsampleOutputs == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        var grad = outputGradient;

        // Backward through final conv
        grad = _convLast.Backward(grad);

        // Backward through HR conv + activation
        grad = BackwardLeakyReLU(_hrConv.Forward(_upsampleOutputs[^1]), grad);
        grad = _hrConv.Backward(grad);

        // Backward through upsampling stages (in reverse)
        for (int i = _upsampleConvs.Length - 1; i >= 0; i--)
        {
            // Backward through LeakyReLU after PixelShuffle
            grad = BackwardLeakyReLU(_upsampleOutputs[i * 2], grad);

            // Backward through PixelShuffle
            grad = _pixelShuffleLayers[i].Backward(grad);

            // Backward through upsample conv
            grad = _upsampleConvs[i].Backward(grad);
        }

        // Backward through global residual: grad flows to both trunk and conv1
        var trunkGrad = grad;
        var conv1ResidualGrad = grad;

        // Backward through trunk conv
        var rrdbGrad = _trunkConv.Backward(trunkGrad);

        // Backward through RRDB blocks (in reverse)
        for (int i = _rrdbBlocks.Length - 1; i >= 0; i--)
        {
            rrdbGrad = _rrdbBlocks[i].Backward(rrdbGrad);
        }

        // Combine gradients at conv1 output
        var combinedGrad = AddTensors(rrdbGrad, conv1ResidualGrad);

        // Backward through first conv
        return _convFirst.Backward(combinedGrad);
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Applies LeakyReLU activation.
    /// </summary>
    private Tensor<T> ApplyLeakyReLU(Tensor<T> input)
    {
        var output = new Tensor<T>(input.Shape);
        for (int i = 0; i < input.Length; i++)
        {
            output.Data[i] = _leakyReLU.Activate(input.Data[i]);
        }
        return output;
    }

    /// <summary>
    /// Backward pass through LeakyReLU.
    /// </summary>
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

    /// <summary>
    /// Adds two tensors element-wise.
    /// </summary>
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

        foreach (var rrdb in _rrdbBlocks)
        {
            rrdb.UpdateParameters(learningRate);
        }

        _trunkConv.UpdateParameters(learningRate);

        foreach (var conv in _upsampleConvs)
        {
            conv.UpdateParameters(learningRate);
        }

        _hrConv.UpdateParameters(learningRate);
        _convLast.UpdateParameters(learningRate);
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var allParams = new List<T>();

        AddParametersToList(allParams, _convFirst.GetParameters());

        foreach (var rrdb in _rrdbBlocks)
        {
            AddParametersToList(allParams, rrdb.GetParameters());
        }

        AddParametersToList(allParams, _trunkConv.GetParameters());

        foreach (var conv in _upsampleConvs)
        {
            AddParametersToList(allParams, conv.GetParameters());
        }

        AddParametersToList(allParams, _hrConv.GetParameters());
        AddParametersToList(allParams, _convLast.GetParameters());

        return new Vector<T>([.. allParams]);
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        int offset = 0;

        offset = SetLayerParameters(_convFirst, parameters, offset);

        foreach (var rrdb in _rrdbBlocks)
        {
            offset = SetLayerParameters(rrdb, parameters, offset);
        }

        offset = SetLayerParameters(_trunkConv, parameters, offset);

        foreach (var conv in _upsampleConvs)
        {
            offset = SetLayerParameters(conv, parameters, offset);
        }

        offset = SetLayerParameters(_hrConv, parameters, offset);
        SetLayerParameters(_convLast, parameters, offset);
    }

    private static void AddParametersToList(List<T> list, Vector<T> parameters)
    {
        for (int i = 0; i < parameters.Length; i++)
        {
            list.Add(parameters[i]);
        }
    }

    private static int SetLayerParameters(ILayer<T> layer, Vector<T> parameters, int offset)
    {
        int count = layer.GetParameters().Length;
        layer.SetParameters(parameters.SubVector(offset, count));
        return offset + count;
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        _lastInput = null;
        _conv1Output = null;
        _rrdbOutputs = null;
        _upsampleOutputs = null;

        _convFirst.ResetState();
        foreach (var rrdb in _rrdbBlocks)
        {
            rrdb.ResetState();
        }
        _trunkConv.ResetState();
        foreach (var conv in _upsampleConvs)
        {
            conv.ResetState();
        }
        foreach (var ps in _pixelShuffleLayers)
        {
            ps.ResetState();
        }
        _hrConv.ResetState();
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

        // Create symbolic input node with batch dimension
        var symbolicInput = new Tensor<T>(new int[] { 1 }.Concat(InputShape).ToArray());
        var inputNode = TensorOperations<T>.Variable(symbolicInput, "input");
        inputNodes.Add(inputNode);

        return BuildComputationGraph(inputNode, "");
    }

    /// <inheritdoc />
    public ComputationNode<T> BuildComputationGraph(ComputationNode<T> inputNode, string namePrefix)
    {
        // Initial feature extraction
        var x = BuildConvNode(_convFirst, inputNode, $"{namePrefix}conv_first_");

        // Store for global residual
        var conv1Output = x;

        // RRDB blocks
        for (int i = 0; i < _rrdbBlocks.Length; i++)
        {
            x = _rrdbBlocks[i].BuildComputationGraph(x, $"{namePrefix}rrdb{i}_");
        }

        // Trunk conv
        x = BuildConvNode(_trunkConv, x, $"{namePrefix}trunk_conv_");

        // Global residual
        x = TensorOperations<T>.Add(x, conv1Output);

        // Upsampling stages
        for (int i = 0; i < _upsampleConvs.Length; i++)
        {
            x = BuildConvNode(_upsampleConvs[i], x, $"{namePrefix}upsample{i}_conv_");
            x = TensorOperations<T>.PixelShuffle(x, 2);
            x = TensorOperations<T>.LeakyReLU(x, 0.2);
        }

        // HR conv + LeakyReLU
        x = BuildConvNode(_hrConv, x, $"{namePrefix}hr_conv_");
        x = TensorOperations<T>.LeakyReLU(x, 0.2);

        // Final conv
        x = BuildConvNode(_convLast, x, $"{namePrefix}conv_last_");

        return x;
    }

    /// <summary>
    /// Builds a Conv2D computation node from a ConvolutionalLayer.
    /// </summary>
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
