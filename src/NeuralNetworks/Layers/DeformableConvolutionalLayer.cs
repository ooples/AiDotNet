using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Deformable Convolutional Layer that learns spatial sampling offsets.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Deformable convolution augments standard convolution with learnable 2D offsets for each
/// sampling location. This allows the convolution to adapt its receptive field to the
/// geometric structure of the input, making it particularly effective for video alignment.
/// </para>
/// <para>
/// <b>For Beginners:</b> Regular convolutions look at fixed grid positions around each pixel.
/// Deformable convolutions can look at shifted positions learned from the data itself.
///
/// This is useful for:
/// 1. Aligning features between video frames (different objects may have moved differently)
/// 2. Handling geometric transformations (rotation, scale, perspective)
/// 3. Object detection with varying shapes and poses
/// </para>
/// <para>
/// <b>Reference:</b> Dai et al., "Deformable Convolutional Networks", ICCV 2017.
/// https://arxiv.org/abs/1703.06211
/// </para>
/// </remarks>
public class DeformableConvolutionalLayer<T> : LayerBase<T>, IChainableComputationGraph<T>
{
    #region Fields

    private readonly IEngine _engine;
    private readonly int _inputHeight;
    private readonly int _inputWidth;
    private readonly int _inputChannels;
    private readonly int _outputChannels;
    private readonly int _kernelSize;
    private readonly int _stride;
    private readonly int _padding;
    private readonly int _groups;
    private readonly int _deformGroups;

    // Main convolution weights and bias
    private Tensor<T> _weights;
    private Tensor<T> _bias;

    // Offset prediction convolution
    private Tensor<T> _offsetWeights;
    private Tensor<T> _offsetBias;

    // Modulation mask prediction (for Deformable Conv v2)
    private readonly bool _useModulation;
    private Tensor<T>? _maskWeights;
    private Tensor<T>? _maskBias;

    // Gradients
    private Tensor<T>? _weightGradients;
    private Tensor<T>? _biasGradients;
    private Tensor<T>? _offsetWeightGradients;
    private Tensor<T>? _offsetBiasGradients;
    private Tensor<T>? _maskWeightGradients;
    private Tensor<T>? _maskBiasGradients;

    // Cached values for backward pass
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOffsets;
    private Tensor<T>? _lastMask;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a new Deformable Convolutional Layer.
    /// </summary>
    /// <param name="inputHeight">Height of input feature map.</param>
    /// <param name="inputWidth">Width of input feature map.</param>
    /// <param name="inputChannels">Number of input channels.</param>
    /// <param name="outputChannels">Number of output channels.</param>
    /// <param name="kernelSize">Size of the convolution kernel (default: 3).</param>
    /// <param name="stride">Convolution stride (default: 1).</param>
    /// <param name="padding">Padding size (default: 1).</param>
    /// <param name="groups">Number of convolution groups (default: 1).</param>
    /// <param name="deformGroups">Number of deformable groups (default: 1).</param>
    /// <param name="useModulation">Whether to use modulation mask (DCNv2, default: true).</param>
    /// <param name="engine">Optional computation engine (CPU or GPU). If null, uses default CPU engine.</param>
    public DeformableConvolutionalLayer(
        int inputHeight,
        int inputWidth,
        int inputChannels,
        int outputChannels,
        int kernelSize = 3,
        int stride = 1,
        int padding = 1,
        int groups = 1,
        int deformGroups = 1,
        bool useModulation = true,
        IEngine? engine = null)
        : base(
            [inputChannels, inputHeight, inputWidth],
            [outputChannels, (inputHeight + 2 * padding - kernelSize) / stride + 1, (inputWidth + 2 * padding - kernelSize) / stride + 1])
    {
        // Validate parameters
        if (inputHeight <= 0) throw new ArgumentOutOfRangeException(nameof(inputHeight), "Input height must be positive.");
        if (inputWidth <= 0) throw new ArgumentOutOfRangeException(nameof(inputWidth), "Input width must be positive.");
        if (inputChannels <= 0) throw new ArgumentOutOfRangeException(nameof(inputChannels), "Input channels must be positive.");
        if (outputChannels <= 0) throw new ArgumentOutOfRangeException(nameof(outputChannels), "Output channels must be positive.");
        if (kernelSize <= 0) throw new ArgumentOutOfRangeException(nameof(kernelSize), "Kernel size must be positive.");
        if (stride <= 0) throw new ArgumentOutOfRangeException(nameof(stride), "Stride must be positive.");
        if (padding < 0) throw new ArgumentOutOfRangeException(nameof(padding), "Padding must be non-negative.");
        if (groups < 1) throw new ArgumentOutOfRangeException(nameof(groups), "Groups must be at least 1.");
        if (deformGroups < 1) throw new ArgumentOutOfRangeException(nameof(deformGroups), "Deformable groups must be at least 1.");
        if (inputChannels % groups != 0) throw new ArgumentException($"Input channels ({inputChannels}) must be divisible by groups ({groups}).", nameof(groups));
        
        // Validate output dimensions are positive
        int outputHeight = (inputHeight + 2 * padding - kernelSize) / stride + 1;
        int outputWidth = (inputWidth + 2 * padding - kernelSize) / stride + 1;
        if (outputHeight <= 0 || outputWidth <= 0)
            throw new ArgumentException(
                $"Invalid layer parameters: output size would be {outputHeight}x{outputWidth} " +
                $"for input {inputHeight}x{inputWidth}, kernel {kernelSize}, stride {stride}, padding {padding}.");

        _engine = engine ?? new CpuEngine();
        _inputHeight = inputHeight;
        _inputWidth = inputWidth;
        _inputChannels = inputChannels;
        _outputChannels = outputChannels;
        _kernelSize = kernelSize;
        _stride = stride;
        _padding = padding;
        _groups = groups;
        _deformGroups = deformGroups;
        _useModulation = useModulation;

        // Initialize main convolution weights [outC, inC/groups, kH, kW]
        int inChannelsPerGroup = inputChannels / groups;
        _weights = InitializeWeights(outputChannels, inChannelsPerGroup, kernelSize, kernelSize);
        _bias = new Tensor<T>([outputChannels]);

        // Initialize offset prediction weights
        // Offsets: 2 (x, y) * kernelSize^2 per deform group
        int offsetChannels = 2 * kernelSize * kernelSize * deformGroups;
        _offsetWeights = InitializeWeights(offsetChannels, inputChannels, kernelSize, kernelSize);
        _offsetBias = new Tensor<T>([offsetChannels]);

        // Initialize modulation mask weights (if using DCNv2)
        if (useModulation)
        {
            int maskChannels = kernelSize * kernelSize * deformGroups;
            _maskWeights = InitializeWeights(maskChannels, inputChannels, kernelSize, kernelSize);
            _maskBias = new Tensor<T>([maskChannels]);
        }
    }

    #endregion

    #region Forward Pass

    /// <inheritdoc/>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;

        // Ensure 4D input [batch, channels, height, width]
        var input4D = EnsureBatchDimension(input);
        int batch = input4D.Shape[0];
        int channels = input4D.Shape[1];
        int height = input4D.Shape[2];
        int width = input4D.Shape[3];

        // Predict offsets using standard convolution via IEngine
        var offsets = PredictOffsetsViaEngine(input4D);
        _lastOffsets = offsets;

        // Predict modulation mask if using DCNv2
        Tensor<T>? mask = null;
        if (_useModulation)
        {
            mask = PredictMaskViaEngine(input4D);
            _lastMask = mask;
        }

        // Perform deformable convolution using IEngine
        var output = _engine.DeformableConv2D(
            input4D,
            _weights,
            offsets,
            mask,
            new[] { _stride, _stride },
            new[] { _padding, _padding },
            new[] { 1, 1 });

        // Add bias
        output = AddBias(output);

        // Remove batch dimension if input didn't have one
        return input.Rank == 3 ? RemoveBatchDimension(output) : output;
    }

    private Tensor<T> PredictOffsetsViaEngine(Tensor<T> input)
    {
        // Use IEngine Conv2D for offset prediction
        var offsetOutput = _engine.Conv2D(
            input,
            _offsetWeights,
            new[] { _stride, _stride },
            new[] { _padding, _padding },
            new[] { 1, 1 });

        // Add bias
        return AddBiasToTensor(offsetOutput, _offsetBias);
    }

    private Tensor<T> PredictMaskViaEngine(Tensor<T> input)
    {
        // Use IEngine Conv2D for mask prediction
        var maskOutput = _engine.Conv2D(
            input,
            _maskWeights!,
            new[] { _stride, _stride },
            new[] { _padding, _padding },
            new[] { 1, 1 });

        // Add bias
        maskOutput = AddBiasToTensor(maskOutput, _maskBias!);

        // Apply sigmoid activation for modulation weights
        maskOutput = _engine.Sigmoid(maskOutput);

        return maskOutput;
    }

    private Tensor<T> AddBias(Tensor<T> output)
    {
        return AddBiasToTensor(output, _bias);
    }

    private Tensor<T> AddBiasToTensor(Tensor<T> output, Tensor<T> bias)
    {
        int batch = output.Shape[0];
        int channels = output.Shape[1];
        int height = output.Shape[2];
        int width = output.Shape[3];

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                T biasVal = bias.Data[c];
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        int idx = b * channels * height * width + c * height * width + h * width + w;
                        output.Data[idx] = NumOps.Add(output.Data[idx], biasVal);
                    }
                }
            }
        }
        return output;
    }

    private static Tensor<T> EnsureBatchDimension(Tensor<T> tensor)
    {
        if (tensor.Rank == 4) return tensor;

        // Add batch dimension for 3D tensor [C, H, W] -> [1, C, H, W]
        var newShape = new int[4];
        newShape[0] = 1;
        for (int i = 0; i < tensor.Shape.Length; i++)
            newShape[i + 1] = tensor.Shape[i];

        return tensor.Reshape(newShape);
    }

    private static Tensor<T> RemoveBatchDimension(Tensor<T> tensor)
    {
        if (tensor.Rank != 4 || tensor.Shape[0] != 1) return tensor;

        // Remove batch dimension [1, C, H, W] -> [C, H, W]
        var newShape = new int[3];
        for (int i = 0; i < 3; i++)
            newShape[i] = tensor.Shape[i + 1];

        return tensor.Reshape(newShape);
    }

    #endregion

    #region Backward Pass

    /// <inheritdoc/>
    public override Tensor<T> Backward(Tensor<T> gradOutput)
    {
        if (_lastInput == null || _lastOffsets == null)
            throw new InvalidOperationException("Forward must be called before Backward.");

        // Ensure 4D tensors
        var input4D = EnsureBatchDimension(_lastInput);
        var gradOutput4D = EnsureBatchDimension(gradOutput);

        int batch = input4D.Shape[0];
        int inChannels = input4D.Shape[1];
        int height = input4D.Shape[2];
        int width = input4D.Shape[3];

        int outChannels = gradOutput4D.Shape[1];
        int outH = gradOutput4D.Shape[2];
        int outW = gradOutput4D.Shape[3];

        // Initialize gradients
        _weightGradients = new Tensor<T>(_weights.Shape);
        _biasGradients = new Tensor<T>(_bias.Shape);
        _offsetWeightGradients = new Tensor<T>(_offsetWeights.Shape);
        _offsetBiasGradients = new Tensor<T>(_offsetBias.Shape);

        if (_useModulation)
        {
            _maskWeightGradients = new Tensor<T>(_maskWeights!.Shape);
            _maskBiasGradients = new Tensor<T>(_maskBias!.Shape);
        }

        // 1. Compute bias gradients (sum over batch and spatial dimensions)
        ComputeBiasGradients(gradOutput4D);

        // 2. Compute kernel gradients using IEngine
        _weightGradients = _engine.DeformableConv2DBackwardKernel(
            gradOutput4D,
            input4D,
            _lastOffsets,
            _lastMask,
            kernelShape: _weights.Shape,
            stride: new[] { _stride, _stride },
            padding: new[] { _padding, _padding },
            dilation: new[] { 1, 1 });

        // 3. Compute input gradients from deformable conv using IEngine
        var gradInputFromDeform = _engine.DeformableConv2DBackwardInput(
            gradOutput4D,
            input4D,
            _weights,
            _lastOffsets,
            _lastMask,
            inputShape: input4D.Shape,
            stride: new[] { _stride, _stride },
            padding: new[] { _padding, _padding },
            dilation: new[] { 1, 1 });

        // 4. Compute offset gradients using IEngine
        var gradOffsets = _engine.DeformableConv2DBackwardOffset(
            gradOutput4D,
            input4D,
            _weights,
            _lastOffsets,
            _lastMask,
            stride: new[] { _stride, _stride },
            padding: new[] { _padding, _padding },
            dilation: new[] { 1, 1 });

        // 5. Compute mask gradients if using modulation
        Tensor<T>? gradMask = null;
        if (_useModulation && _lastMask != null)
        {
            gradMask = _engine.DeformableConv2DBackwardMask(
                gradOutput4D,
                input4D,
                _weights,
                _lastOffsets,
                _lastMask,
                stride: new[] { _stride, _stride },
                padding: new[] { _padding, _padding },
                dilation: new[] { 1, 1 });

            // Backprop through sigmoid: gradMask * sigmoid(x) * (1 - sigmoid(x))
            // _lastMask is already sigmoid output, so grad = gradMask * mask * (1 - mask)
            for (int i = 0; i < gradMask.Length; i++)
            {
                T m = _lastMask.Data[i];
                T oneMinusM = NumOps.Subtract(NumOps.One, m);
                gradMask.Data[i] = NumOps.Multiply(NumOps.Multiply(gradMask.Data[i], m), oneMinusM);
            }
        }

        // 6. Backprop through offset prediction conv to get offset weight/bias gradients
        // and input gradient contribution
        var gradInputFromOffset = BackpropConvolution(
            input4D, _offsetWeights, gradOffsets, _offsetWeightGradients, _offsetBiasGradients);

        // 7. Backprop through mask prediction conv if using modulation
        Tensor<T>? gradInputFromMask = null;
        if (_useModulation && gradMask != null)
        {
            gradInputFromMask = BackpropConvolution(
                input4D, _maskWeights!, gradMask, _maskWeightGradients!, _maskBiasGradients!);
        }

        // 8. Sum all input gradient contributions
        var totalInputGrad = new Tensor<T>(input4D.Shape);
        for (int i = 0; i < totalInputGrad.Length; i++)
        {
            totalInputGrad.Data[i] = gradInputFromDeform.Data[i];
            totalInputGrad.Data[i] = NumOps.Add(totalInputGrad.Data[i], gradInputFromOffset.Data[i]);
            if (gradInputFromMask != null)
            {
                totalInputGrad.Data[i] = NumOps.Add(totalInputGrad.Data[i], gradInputFromMask.Data[i]);
            }
        }

        // Remove batch dimension if original input didn't have one
        return _lastInput.Rank == 3 ? RemoveBatchDimension(totalInputGrad) : totalInputGrad;
    }

    private void ComputeBiasGradients(Tensor<T> gradOutput)
    {
        int batch = gradOutput.Shape[0];
        int channels = gradOutput.Shape[1];
        int outH = gradOutput.Shape[2];
        int outW = gradOutput.Shape[3];

        for (int c = 0; c < channels; c++)
        {
            T sum = NumOps.Zero;
            for (int b = 0; b < batch; b++)
            {
                for (int h = 0; h < outH; h++)
                {
                    for (int w = 0; w < outW; w++)
                    {
                        int idx = b * channels * outH * outW + c * outH * outW + h * outW + w;
                        sum = NumOps.Add(sum, gradOutput.Data[idx]);
                    }
                }
            }
            _biasGradients!.Data[c] = sum;
        }
    }

    private Tensor<T> BackpropConvolution(
        Tensor<T> input,
        Tensor<T> weights,
        Tensor<T> gradOutput,
        Tensor<T> weightGrad,
        Tensor<T> biasGrad)
    {
        // Use IEngine for backward convolution
        var gradInput = _engine.Conv2DBackwardInput(
            gradOutput,
            weights,
            inputShape: input.Shape,
            stride: new[] { _stride, _stride },
            padding: new[] { _padding, _padding },
            dilation: new[] { 1, 1 });

        // Compute weight gradients
        var computedWeightGrad = _engine.Conv2DBackwardKernel(
            gradOutput,
            input,
            kernelShape: weights.Shape,
            stride: new[] { _stride, _stride },
            padding: new[] { _padding, _padding },
            dilation: new[] { 1, 1 });

        // Copy to weight gradient tensor
        for (int i = 0; i < weightGrad.Length; i++)
        {
            weightGrad.Data[i] = computedWeightGrad.Data[i];
        }

        // Compute bias gradients (sum over batch and spatial)
        int batch = gradOutput.Shape[0];
        int channels = gradOutput.Shape[1];
        int outH = gradOutput.Shape[2];
        int outW = gradOutput.Shape[3];

        for (int c = 0; c < channels; c++)
        {
            T sum = NumOps.Zero;
            for (int b = 0; b < batch; b++)
            {
                for (int h = 0; h < outH; h++)
                {
                    for (int w = 0; w < outW; w++)
                    {
                        int idx = b * channels * outH * outW + c * outH * outW + h * outW + w;
                        sum = NumOps.Add(sum, gradOutput.Data[idx]);
                    }
                }
            }
            biasGrad.Data[c] = sum;
        }

        return gradInput;
    }

    #endregion

    #region Helper Methods

    private Tensor<T> InitializeWeights(int outC, int inC, int kH, int kW)
    {
        var weights = new Tensor<T>([outC, inC, kH, kW]);
        double fan_in = inC * kH * kW;
        double std = Math.Sqrt(2.0 / fan_in); // He initialization

        var random = new Random(42);
        for (int i = 0; i < weights.Length; i++)
        {
            weights.Data[i] = NumOps.FromDouble(random.NextDouble() * 2 * std - std);
        }

        return weights;
    }

    #endregion

    #region Layer Properties

    /// <inheritdoc/>
    /// <remarks>
    /// This layer supports full training with gradient computation for all parameters:
    /// - Main convolution weights and bias
    /// - Offset prediction weights and bias
    /// - Modulation mask weights and bias (if using DCNv2)
    /// All gradients are computed using IEngine backward operations with bilinear interpolation support.
    /// </remarks>
    public override bool SupportsTraining => true;

    /// <inheritdoc/>
    public override bool SupportsJitCompilation =>
        _weights != null && _bias != null && _offsetWeights != null && _offsetBias != null;

    #endregion

    #region IChainableComputationGraph Implementation

    /// <inheritdoc/>
    public override int[] GetInputShape() => [_inputChannels, _inputHeight, _inputWidth];

    /// <summary>
    /// Gets the output shape for this layer.
    /// </summary>
    public new int[] GetOutputShape()
    {
        int outH = (_inputHeight + 2 * _padding - _kernelSize) / _stride + 1;
        int outW = (_inputWidth + 2 * _padding - _kernelSize) / _stride + 1;
        return [_outputChannels, outH, outW];
    }

    /// <inheritdoc/>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null || inputNodes.Count == 0)
            throw new ArgumentException("Input nodes cannot be null or empty.", nameof(inputNodes));

        return BuildComputationGraph(inputNodes[0], "");
    }

    /// <inheritdoc/>
    public ComputationNode<T> BuildComputationGraph(ComputationNode<T> inputNode, string namePrefix)
    {
        if (!SupportsJitCompilation)
            throw new InvalidOperationException("Layer weights not initialized. Cannot build computation graph.");

        // Create constant nodes for weights
        var kernelNode = TensorOperations<T>.Constant(_weights, $"{namePrefix}kernel");
        var biasNode = TensorOperations<T>.Constant(_bias, $"{namePrefix}bias");
        var offsetWeightsNode = TensorOperations<T>.Constant(_offsetWeights, $"{namePrefix}offset_weights");
        var offsetBiasNode = TensorOperations<T>.Constant(_offsetBias, $"{namePrefix}offset_bias");

        // First compute offsets using standard convolution
        var offsetsNode = TensorOperations<T>.Conv2D(
            inputNode,
            offsetWeightsNode,
            offsetBiasNode,
            stride: new int[] { _stride, _stride },
            padding: new int[] { _padding, _padding });

        // Optionally compute modulation mask
        ComputationNode<T>? maskNode = null;
        if (_useModulation && _maskWeights != null && _maskBias != null)
        {
            var maskWeightsNode = TensorOperations<T>.Constant(_maskWeights, $"{namePrefix}mask_weights");
            var maskBiasNode = TensorOperations<T>.Constant(_maskBias, $"{namePrefix}mask_bias");

            var rawMaskNode = TensorOperations<T>.Conv2D(
                inputNode,
                maskWeightsNode,
                maskBiasNode,
                stride: new int[] { _stride, _stride },
                padding: new int[] { _padding, _padding });

            // Apply sigmoid activation to mask
            maskNode = TensorOperations<T>.Sigmoid(rawMaskNode);
        }

        // Apply deformable convolution with computed offsets and mask
        var deformConvNode = TensorOperations<T>.DeformableConv2D(
            inputNode,
            kernelNode,
            offsetsNode,
            maskNode,
            biasNode,
            stride: new int[] { _stride, _stride },
            padding: new int[] { _padding, _padding },
            dilation: new int[] { 1, 1 });

        return deformConvNode;
    }

    #endregion

    #region Parameter Management

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        var allParams = new List<T>();

        // Main conv weights and bias
        for (int i = 0; i < _weights.Length; i++)
            allParams.Add(_weights.Data[i]);
        for (int i = 0; i < _bias.Length; i++)
            allParams.Add(_bias.Data[i]);

        // Offset weights and bias
        for (int i = 0; i < _offsetWeights.Length; i++)
            allParams.Add(_offsetWeights.Data[i]);
        for (int i = 0; i < _offsetBias.Length; i++)
            allParams.Add(_offsetBias.Data[i]);

        // Mask weights and bias (if using modulation)
        if (_useModulation && _maskWeights != null && _maskBias != null)
        {
            for (int i = 0; i < _maskWeights.Length; i++)
                allParams.Add(_maskWeights.Data[i]);
            for (int i = 0; i < _maskBias.Length; i++)
                allParams.Add(_maskBias.Data[i]);
        }

        return new Vector<T>([.. allParams]);
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        int offset = 0;

        // Main conv weights
        for (int i = 0; i < _weights.Length; i++)
            _weights.Data[i] = parameters[offset++];

        // Main conv bias
        for (int i = 0; i < _bias.Length; i++)
            _bias.Data[i] = parameters[offset++];

        // Offset weights
        for (int i = 0; i < _offsetWeights.Length; i++)
            _offsetWeights.Data[i] = parameters[offset++];

        // Offset bias
        for (int i = 0; i < _offsetBias.Length; i++)
            _offsetBias.Data[i] = parameters[offset++];

        // Mask weights and bias
        if (_useModulation && _maskWeights != null && _maskBias != null)
        {
            for (int i = 0; i < _maskWeights.Length; i++)
                _maskWeights.Data[i] = parameters[offset++];
            for (int i = 0; i < _maskBias.Length; i++)
                _maskBias.Data[i] = parameters[offset++];
        }
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        if (_weightGradients != null)
        {
            for (int i = 0; i < _weights.Length; i++)
            {
                _weights.Data[i] = NumOps.Subtract(_weights.Data[i],
                    NumOps.Multiply(learningRate, _weightGradients.Data[i]));
            }
        }

        if (_biasGradients != null)
        {
            for (int i = 0; i < _bias.Length; i++)
            {
                _bias.Data[i] = NumOps.Subtract(_bias.Data[i],
                    NumOps.Multiply(learningRate, _biasGradients.Data[i]));
            }
        }
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameterGradients()
    {
        var allGrads = new List<T>();

        if (_weightGradients != null)
        {
            for (int i = 0; i < _weightGradients.Length; i++)
                allGrads.Add(_weightGradients.Data[i]);
        }

        if (_biasGradients != null)
        {
            for (int i = 0; i < _biasGradients.Length; i++)
                allGrads.Add(_biasGradients.Data[i]);
        }

        if (_offsetWeightGradients != null)
        {
            for (int i = 0; i < _offsetWeightGradients.Length; i++)
                allGrads.Add(_offsetWeightGradients.Data[i]);
        }

        if (_offsetBiasGradients != null)
        {
            for (int i = 0; i < _offsetBiasGradients.Length; i++)
                allGrads.Add(_offsetBiasGradients.Data[i]);
        }

        if (_useModulation)
        {
            if (_maskWeightGradients != null)
            {
                for (int i = 0; i < _maskWeightGradients.Length; i++)
                    allGrads.Add(_maskWeightGradients.Data[i]);
            }

            if (_maskBiasGradients != null)
            {
                for (int i = 0; i < _maskBiasGradients.Length; i++)
                    allGrads.Add(_maskBiasGradients.Data[i]);
            }
        }

        return new Vector<T>([.. allGrads]);
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        _lastInput = null;
        _lastOffsets = null;
        _lastMask = null;
        _weightGradients = null;
        _biasGradients = null;
        _offsetWeightGradients = null;
        _offsetBiasGradients = null;
        _maskWeightGradients = null;
        _maskBiasGradients = null;
    }

    #endregion
}
