#pragma warning disable CS0649, CS0414, CS0169
using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.Helpers;

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
[LayerCategory(LayerCategory.Convolution)]
[LayerTask(LayerTask.SpatialProcessing)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, ChangesShape = true, ExpectedInputRank = 3, Cost = ComputeCost.High, TestInputShape = "1, 8, 8", TestConstructorArgs = "8, 8, 1, 2, 3")]
public partial class DeformableConvolutionalLayer<T> : LayerBase<T>
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
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _weights;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

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

    // GPU caching for backward pass
    private Tensor<T>? _gpuInput;
    private int[]? _gpuInputShape;
    private Tensor<T>? _gpuOffsets;
    private Tensor<T>? _gpuMask;

    #endregion

    #region GPU Weight Storage Fields

    // Main conv weights - GPU tensors for GPU-resident training
    private Tensor<T>? _gpuWeights;
    private Tensor<T>? _gpuBias;
    private Tensor<T>? _gpuWeightGradient;
    private Tensor<T>? _gpuBiasGradient;
    private Tensor<T>? _gpuWeightVelocity;
    private Tensor<T>? _gpuBiasVelocity;
    private Tensor<T>? _gpuWeightM;
    private Tensor<T>? _gpuWeightV;
    private Tensor<T>? _gpuBiasM;
    private Tensor<T>? _gpuBiasV;

    // Offset weights - GPU tensors
    private Tensor<T>? _gpuOffsetWeights;
    private Tensor<T>? _gpuOffsetBias;
    private Tensor<T>? _gpuOffsetWeightGradient;
    private Tensor<T>? _gpuOffsetBiasGradient;
    private Tensor<T>? _gpuOffsetWeightVelocity;
    private Tensor<T>? _gpuOffsetBiasVelocity;
    private Tensor<T>? _gpuOffsetWeightM;
    private Tensor<T>? _gpuOffsetWeightV;
    private Tensor<T>? _gpuOffsetBiasM;
    private Tensor<T>? _gpuOffsetBiasV;

    // Mask weights - GPU tensors (only used if _useModulation)
    private Tensor<T>? _gpuMaskWeights;
    private Tensor<T>? _gpuMaskBias;
    private Tensor<T>? _gpuMaskWeightGradient;
    private Tensor<T>? _gpuMaskBiasGradient;
    private Tensor<T>? _gpuMaskWeightVelocity;
    private Tensor<T>? _gpuMaskBiasVelocity;
    private Tensor<T>? _gpuMaskWeightM;
    private Tensor<T>? _gpuMaskWeightV;
    private Tensor<T>? _gpuMaskBiasM;
    private Tensor<T>? _gpuMaskBiasV;

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
    /// <param name="groups">Number of convolution groups. Currently only groups=1 is supported (default: 1).</param>
    /// <exception cref="NotSupportedException">Thrown when groups is not 1 (grouped deformable convolution is not yet supported).</exception>
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
        if (groups != 1) throw new NotSupportedException("Grouped deformable convolution is not supported; set groups to 1 or implement engine support.");
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

        // Register trainable parameters for GPU memory persistence
        RegisterTrainableParameter(_weights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_bias, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_offsetWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_offsetBias, PersistentTensorRole.Biases);
        if (useModulation && _maskWeights != null && _maskBias != null)
        {
            RegisterTrainableParameter(_maskWeights, PersistentTensorRole.Weights);
            RegisterTrainableParameter(_maskBias, PersistentTensorRole.Biases);
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

    /// <summary>
    /// Performs the forward pass using GPU-resident tensors, keeping all data on GPU.
    /// </summary>
    /// <param name="inputs">GPU-resident input tensor [batch, inChannels, inHeight, inWidth] in NCHW format.</param>
    /// <returns>GPU-resident output tensor [batch, outChannels, outHeight, outWidth] in NCHW format.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the GPU-optimized version of the Forward method.
    /// The main convolution operation stays on GPU, though offset and mask prediction
    /// requires a brief CPU round-trip due to the current DeformableConv2D GPU implementation.</para>
    /// </remarks>
    public override Tensor<T> ForwardGpu(params Tensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));

        if (_engine is not DirectGpuTensorEngine gpuEngine)
        {
            throw new InvalidOperationException(
                "ForwardGpu requires a DirectGpuTensorEngine. Use Forward() for CPU execution.");
        }

        var input = inputs[0];

        // Validate input shape - GPU uses NCHW format [batch, channels, height, width]
        if (input.Shape.Length < 3)
        {
            throw new ArgumentException(
                $"Deformable Conv2D input requires at least 3D tensor [C, H, W]. Got rank {input.Shape.Length}.");
        }

        int rank = input.Shape.Length;

        // Reshape input to 4D NCHW [B, C, H, W] for convolution
        Tensor<T> input4D;
        bool addedBatchDimension = false;
        if (rank == 3)
        {
            // 3D [C, H, W] -> 4D [1, C, H, W]
            addedBatchDimension = true;
            input4D = input.Reshape([1, input.Shape[0], input.Shape[1], input.Shape[2]]);
        }
        else if (rank == 4)
        {
            // 4D [B, C, H, W] - no reshaping needed
            input4D = input;
        }
        else
        {
            // Higher rank: flatten leading dimensions into batch
            int flatBatch = 1;
            for (int d = 0; d < rank - 3; d++)
            {
                flatBatch *= input.Shape[d];
            }
            input4D = input.Reshape([flatBatch, input.Shape[rank - 3], input.Shape[rank - 2], input.Shape[rank - 1]]);
        }

        // Validate input channels
        int actualInputChannels = input4D.Shape[1];
        if (actualInputChannels != _inputChannels)
        {
            throw new ArgumentException(
                $"Expected input channels {_inputChannels}, but got {actualInputChannels}.");
        }

        // Predict offsets using GPU-accelerated Conv2D
        var offsetsGpu = gpuEngine.FusedConv2DGpu(
            input4D,
            _offsetWeights,
            _offsetBias,
            _stride, _stride,      // strideH, strideW
            _padding, _padding,    // padH, padW
            1, 1,                  // dilationH, dilationW
            FusedActivationType.None);
        var offsets = offsetsGpu;

        // Predict modulation mask if using DCNv2
        Tensor<T>? mask = null;
        Tensor<T>? maskGpu = null;
        if (_useModulation && _maskWeights != null && _maskBias != null)
        {
            // Use FusedConv2DGpu with Sigmoid activation for mask prediction
            maskGpu = gpuEngine.FusedConv2DGpu(
                input4D,
                _maskWeights,
                _maskBias,
                _stride, _stride,      // strideH, strideW
                _padding, _padding,    // padH, padW
                1, 1,                  // dilationH, dilationW
                FusedActivationType.Sigmoid);
            mask = maskGpu;
        }

        // Store for potential backward pass
        _lastOffsets = offsets;
        _lastMask = mask;

        // Cache GPU tensors for backward pass if in training mode
        if (IsTrainingMode)
        {
            _gpuInput = input4D;
            _gpuInputShape = input4D._shape.ToArray();
            _gpuOffsets = offsetsGpu;
            _gpuMask = maskGpu;
        }
        else
        {
            // Not training - dispose GPU tensors we don't need
            offsetsGpu.Dispose();
            maskGpu?.Dispose();
        }

        // Execute GPU-accelerated deformable convolution
        // DeformableConv2DGpu handles bias internally
        var result = gpuEngine.DeformableConv2DGpu(
            input4D,
            _weights,
            offsets,
            mask,
            _bias,
            _stride, _stride,      // strideH, strideW
            _padding, _padding,    // padH, padW
            1, 1,                  // dilationH, dilationW
            _groups, _deformGroups,
            FusedActivationType.None);

        // Restore original shape if needed
        if (addedBatchDimension)
        {
            // Input was 3D [C, H, W], output should also be 3D [OutC, OutH, OutW]
            return result.Reshape([_outputChannels, result.Shape[2], result.Shape[3]]);
        }

        return result;
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
            _maskWeights ?? throw new InvalidOperationException("_maskWeights has not been initialized."),
            new[] { _stride, _stride },
            new[] { _padding, _padding },
            new[] { 1, 1 });

        // Add bias
        var maskBiasVal = _maskBias ?? throw new InvalidOperationException("_maskBias has not been initialized.");
        maskOutput = AddBiasToTensor(maskOutput, maskBiasVal);

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
                T biasVal = bias.Data.Span[c];
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        int idx = b * channels * height * width + c * height * width + h * width + w;
                        output.Data.Span[idx] = NumOps.Add(output.Data.Span[idx], biasVal);
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
                        sum = NumOps.Add(sum, gradOutput.Data.Span[idx]);
                    }
                }
            }
            var biasGrad = _biasGradients ?? throw new InvalidOperationException("_biasGradients has not been initialized.");
            biasGrad.Data.Span[c] = sum;
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
            inputShape: input._shape,
            stride: new[] { _stride, _stride },
            padding: new[] { _padding, _padding },
            dilation: new[] { 1, 1 });

        // Compute weight gradients
        var computedWeightGrad = _engine.Conv2DBackwardKernel(
            gradOutput,
            input,
            kernelShape: weights._shape,
            stride: new[] { _stride, _stride },
            padding: new[] { _padding, _padding },
            dilation: new[] { 1, 1 });

        // Copy to weight gradient tensor
        for (int i = 0; i < weightGrad.Length; i++)
        {
            weightGrad.Data.Span[i] = computedWeightGrad.Data.Span[i];
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
                        sum = NumOps.Add(sum, gradOutput.Data.Span[idx]);
                    }
                }
            }
            biasGrad.Data.Span[c] = sum;
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

        var random = RandomHelper.CreateSeededRandom(42);
        for (int i = 0; i < weights.Length; i++)
        {
            weights.Data.Span[i] = NumOps.FromDouble(random.NextDouble() * 2 * std - std);
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
    public override int ParameterCount => GetParameters().Length;
    public override bool SupportsTraining => true;

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

    #endregion

    #region Parameter Management

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        var allParams = new List<T>();

        // Main conv weights and bias
        for (int i = 0; i < _weights.Length; i++)
            allParams.Add(_weights.Data.Span[i]);
        for (int i = 0; i < _bias.Length; i++)
            allParams.Add(_bias.Data.Span[i]);

        // Offset weights and bias
        for (int i = 0; i < _offsetWeights.Length; i++)
            allParams.Add(_offsetWeights.Data.Span[i]);
        for (int i = 0; i < _offsetBias.Length; i++)
            allParams.Add(_offsetBias.Data.Span[i]);

        // Mask weights and bias (if using modulation)
        if (_useModulation && _maskWeights != null && _maskBias != null)
        {
            for (int i = 0; i < _maskWeights.Length; i++)
                allParams.Add(_maskWeights.Data.Span[i]);
            for (int i = 0; i < _maskBias.Length; i++)
                allParams.Add(_maskBias.Data.Span[i]);
        }

        return new Vector<T>([.. allParams]);
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        int offset = 0;

        // Main conv weights
        for (int i = 0; i < _weights.Length; i++)
            _weights.Data.Span[i] = parameters[offset++];

        // Main conv bias
        for (int i = 0; i < _bias.Length; i++)
            _bias.Data.Span[i] = parameters[offset++];

        // Offset weights
        for (int i = 0; i < _offsetWeights.Length; i++)
            _offsetWeights.Data.Span[i] = parameters[offset++];

        // Offset bias
        for (int i = 0; i < _offsetBias.Length; i++)
            _offsetBias.Data.Span[i] = parameters[offset++];

        // Mask weights and bias
        if (_useModulation && _maskWeights != null && _maskBias != null)
        {
            for (int i = 0; i < _maskWeights.Length; i++)
                _maskWeights.Data.Span[i] = parameters[offset++];
            for (int i = 0; i < _maskBias.Length; i++)
                _maskBias.Data.Span[i] = parameters[offset++];
        }

        // Invalidate GPU cache after parameter update
        Engine.InvalidatePersistentTensor(_weights);
        Engine.InvalidatePersistentTensor(_bias);
        Engine.InvalidatePersistentTensor(_offsetWeights);
        Engine.InvalidatePersistentTensor(_offsetBias);
        if (_useModulation && _maskWeights != null && _maskBias != null)
        {
            Engine.InvalidatePersistentTensor(_maskWeights);
            Engine.InvalidatePersistentTensor(_maskBias);
        }
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        // Update main convolution weights using Engine tensor ops
        if (_weightGradients != null)
        {
            _weights = Engine.TensorSubtract(_weights, Engine.TensorMultiplyScalar(_weightGradients, learningRate));
        }

        if (_biasGradients != null)
        {
            _bias = Engine.TensorSubtract(_bias, Engine.TensorMultiplyScalar(_biasGradients, learningRate));
        }

        // Update offset prediction network weights using Engine tensor ops
        if (_offsetWeightGradients != null)
        {
            _offsetWeights = Engine.TensorSubtract(_offsetWeights, Engine.TensorMultiplyScalar(_offsetWeightGradients, learningRate));
        }

        if (_offsetBiasGradients != null)
        {
            _offsetBias = Engine.TensorSubtract(_offsetBias, Engine.TensorMultiplyScalar(_offsetBiasGradients, learningRate));
        }

        // Update mask prediction network weights (if using modulation) using Engine tensor ops
        if (_useModulation)
        {
            if (_maskWeightGradients != null && _maskWeights != null)
            {
                _maskWeights = Engine.TensorSubtract(_maskWeights, Engine.TensorMultiplyScalar(_maskWeightGradients, learningRate));
            }

            if (_maskBiasGradients != null && _maskBias != null)
            {
                _maskBias = Engine.TensorSubtract(_maskBias, Engine.TensorMultiplyScalar(_maskBiasGradients, learningRate));
            }
        }

        // Invalidate GPU cache after parameter update
        Engine.InvalidatePersistentTensor(_weights);
        Engine.InvalidatePersistentTensor(_bias);
        Engine.InvalidatePersistentTensor(_offsetWeights);
        Engine.InvalidatePersistentTensor(_offsetBias);
        if (_useModulation && _maskWeights != null && _maskBias != null)
        {
            Engine.InvalidatePersistentTensor(_maskWeights);
            Engine.InvalidatePersistentTensor(_maskBias);
        }
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameterGradients()
    {
        var allGrads = new List<T>();

        if (_weightGradients != null)
        {
            for (int i = 0; i < _weightGradients.Length; i++)
                allGrads.Add(_weightGradients.Data.Span[i]);
        }

        if (_biasGradients != null)
        {
            for (int i = 0; i < _biasGradients.Length; i++)
                allGrads.Add(_biasGradients.Data.Span[i]);
        }

        if (_offsetWeightGradients != null)
        {
            for (int i = 0; i < _offsetWeightGradients.Length; i++)
                allGrads.Add(_offsetWeightGradients.Data.Span[i]);
        }

        if (_offsetBiasGradients != null)
        {
            for (int i = 0; i < _offsetBiasGradients.Length; i++)
                allGrads.Add(_offsetBiasGradients.Data.Span[i]);
        }

        if (_useModulation)
        {
            if (_maskWeightGradients != null)
            {
                for (int i = 0; i < _maskWeightGradients.Length; i++)
                    allGrads.Add(_maskWeightGradients.Data.Span[i]);
            }

            if (_maskBiasGradients != null)
            {
                for (int i = 0; i < _maskBiasGradients.Length; i++)
                    allGrads.Add(_maskBiasGradients.Data.Span[i]);
            }
        }

        return new Vector<T>([.. allGrads]);
    }

    /// <summary>
    /// GPU-resident parameter update with polymorphic optimizer support.
    /// Updates all weight tensors directly on GPU using the specified optimizer configuration.
    /// </summary>
    /// <param name="config">GPU optimizer configuration specifying the optimizer type and hyperparameters.</param>
    public override void UpdateParametersGpu(IGpuOptimizerConfig config)
    {
        if (_engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("UpdateParametersGpu requires a DirectGpuTensorEngine.");

        var backend = gpuEngine.GetBackend();
        if (backend is null)
            throw new InvalidOperationException("GPU backend unavailable.");

        // Ensure GPU weights are initialized
        _gpuWeights ??= GpuTensorHelper.UploadToGpu<T>(backend, _weights, GpuTensorRole.Weight);
        _gpuBias ??= GpuTensorHelper.UploadToGpu<T>(backend, _bias, GpuTensorRole.Bias);
        _gpuOffsetWeights ??= GpuTensorHelper.UploadToGpu<T>(backend, _offsetWeights, GpuTensorRole.Weight);
        _gpuOffsetBias ??= GpuTensorHelper.UploadToGpu<T>(backend, _offsetBias, GpuTensorRole.Bias);
        if (_useModulation && _maskWeights != null && _maskBias != null)
        {
            _gpuMaskWeights ??= GpuTensorHelper.UploadToGpu<T>(backend, _maskWeights, GpuTensorRole.Weight);
            _gpuMaskBias ??= GpuTensorHelper.UploadToGpu<T>(backend, _maskBias, GpuTensorRole.Bias);
        }

        // Check for gradients
        if (_gpuWeightGradient is null || _gpuBiasGradient is null ||
            _gpuOffsetWeightGradient is null || _gpuOffsetBiasGradient is null)
            throw new InvalidOperationException("BackwardGpu must be called before UpdateParametersGpu.");

        // Ensure optimizer state buffers exist
        EnsureDeformableOptimizerState(backend, config.OptimizerType);

        // Apply updates using polymorphic optimizer dispatch
        config.ApplyUpdate(backend, _gpuWeights.Buffer, _gpuWeightGradient.Buffer, BuildDeformableOptimizerState("weights"), _weights.Length);
        config.ApplyUpdate(backend, _gpuBias.Buffer, _gpuBiasGradient.Buffer, BuildDeformableOptimizerState("bias"), _bias.Length);
        config.ApplyUpdate(backend, _gpuOffsetWeights.Buffer, _gpuOffsetWeightGradient.Buffer, BuildDeformableOptimizerState("offsetWeights"), _offsetWeights.Length);
        config.ApplyUpdate(backend, _gpuOffsetBias.Buffer, _gpuOffsetBiasGradient.Buffer, BuildDeformableOptimizerState("offsetBias"), _offsetBias.Length);

        // Mask weights and bias (if using modulation)
        if (_useModulation && _gpuMaskWeights != null && _gpuMaskBias != null &&
            _gpuMaskWeightGradient != null && _gpuMaskBiasGradient != null &&
            _maskWeights != null && _maskBias != null)
        {
            config.ApplyUpdate(backend, _gpuMaskWeights.Buffer, _gpuMaskWeightGradient.Buffer, BuildDeformableOptimizerState("maskWeights"), _maskWeights.Length);
            config.ApplyUpdate(backend, _gpuMaskBias.Buffer, _gpuMaskBiasGradient.Buffer, BuildDeformableOptimizerState("maskBias"), _maskBias.Length);
        }

        // Sync back to CPU tensors for compatibility
        _weights = _gpuWeights;
        _bias = _gpuBias;
        _offsetWeights = _gpuOffsetWeights;
        _offsetBias = _gpuOffsetBias;
        if (_useModulation && _gpuMaskWeights != null && _gpuMaskBias != null &&
            _maskWeights != null && _maskBias != null)
        {
            _maskWeights = _gpuMaskWeights;
            _maskBias = _gpuMaskBias;
        }
    }

    /// <summary>
    /// Ensures GPU optimizer state buffers exist for all deformable convolution parameters.
    /// </summary>
    private void EnsureDeformableOptimizerState(IDirectGpuBackend backend, GpuOptimizerType optimizerType)
    {
        int weightSize = _weights.Length;
        int biasSize = _bias.Length;
        int offsetWeightSize = _offsetWeights.Length;
        int offsetBiasSize = _offsetBias.Length;
        int maskWeightSize = _maskWeights?.Length ?? 0;
        int maskBiasSize = _maskBias?.Length ?? 0;

        switch (optimizerType)
        {
            case GpuOptimizerType.Sgd:
            case GpuOptimizerType.Nag:
            case GpuOptimizerType.Lars:
                // Velocity buffers
                _gpuWeightVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([weightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuOffsetWeightVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([offsetWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuOffsetBiasVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([offsetBiasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                if (_useModulation && maskWeightSize > 0 && maskBiasSize > 0)
                {
                    _gpuMaskWeightVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([maskWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                    _gpuMaskBiasVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([maskBiasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                }
                break;

            case GpuOptimizerType.Adam:
            case GpuOptimizerType.AdamW:
            case GpuOptimizerType.Lamb:
                // M and V buffers for Adam-family
                _gpuWeightM ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([weightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuWeightV ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([weightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasM ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasV ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuOffsetWeightM ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([offsetWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuOffsetWeightV ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([offsetWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuOffsetBiasM ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([offsetBiasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuOffsetBiasV ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([offsetBiasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                if (_useModulation && maskWeightSize > 0 && maskBiasSize > 0)
                {
                    _gpuMaskWeightM ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([maskWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                    _gpuMaskWeightV ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([maskWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                    _gpuMaskBiasM ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([maskBiasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                    _gpuMaskBiasV ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([maskBiasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                }
                break;

            case GpuOptimizerType.RmsProp:
            case GpuOptimizerType.Adagrad:
                // SquaredAvg/AccumulatedGrad buffers (reusing Velocity for these)
                _gpuWeightVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([weightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuBiasVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([biasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuOffsetWeightVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([offsetWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                _gpuOffsetBiasVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([offsetBiasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                if (_useModulation && maskWeightSize > 0 && maskBiasSize > 0)
                {
                    _gpuMaskWeightVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([maskWeightSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                    _gpuMaskBiasVelocity ??= GpuTensorHelper.UploadToGpu<T>(backend, Tensor<T>.CreateDefault([maskBiasSize], NumOps.Zero), GpuTensorRole.OptimizerState);
                }
                break;
        }
    }

    private GpuOptimizerState BuildDeformableOptimizerState(string paramName)
    {
        return paramName switch
        {
            "weights" => new GpuOptimizerState
            {
                Velocity = _gpuWeightVelocity?.Buffer,
                M = _gpuWeightM?.Buffer,
                V = _gpuWeightV?.Buffer,
                SquaredAvg = _gpuWeightVelocity?.Buffer,
                AccumulatedGrad = _gpuWeightVelocity?.Buffer
            },
            "bias" => new GpuOptimizerState
            {
                Velocity = _gpuBiasVelocity?.Buffer,
                M = _gpuBiasM?.Buffer,
                V = _gpuBiasV?.Buffer,
                SquaredAvg = _gpuBiasVelocity?.Buffer,
                AccumulatedGrad = _gpuBiasVelocity?.Buffer
            },
            "offsetWeights" => new GpuOptimizerState
            {
                Velocity = _gpuOffsetWeightVelocity?.Buffer,
                M = _gpuOffsetWeightM?.Buffer,
                V = _gpuOffsetWeightV?.Buffer,
                SquaredAvg = _gpuOffsetWeightVelocity?.Buffer,
                AccumulatedGrad = _gpuOffsetWeightVelocity?.Buffer
            },
            "offsetBias" => new GpuOptimizerState
            {
                Velocity = _gpuOffsetBiasVelocity?.Buffer,
                M = _gpuOffsetBiasM?.Buffer,
                V = _gpuOffsetBiasV?.Buffer,
                SquaredAvg = _gpuOffsetBiasVelocity?.Buffer,
                AccumulatedGrad = _gpuOffsetBiasVelocity?.Buffer
            },
            "maskWeights" => new GpuOptimizerState
            {
                Velocity = _gpuMaskWeightVelocity?.Buffer,
                M = _gpuMaskWeightM?.Buffer,
                V = _gpuMaskWeightV?.Buffer,
                SquaredAvg = _gpuMaskWeightVelocity?.Buffer,
                AccumulatedGrad = _gpuMaskWeightVelocity?.Buffer
            },
            "maskBias" => new GpuOptimizerState
            {
                Velocity = _gpuMaskBiasVelocity?.Buffer,
                M = _gpuMaskBiasM?.Buffer,
                V = _gpuMaskBiasV?.Buffer,
                SquaredAvg = _gpuMaskBiasVelocity?.Buffer,
                AccumulatedGrad = _gpuMaskBiasVelocity?.Buffer
            },
            _ => new GpuOptimizerState()
        };
    }

    /// <inheritdoc/>
    public override void ClearGradients()
    {
        base.ClearGradients();
        _weightGradients = null; _biasGradients = null;
        _offsetWeightGradients = null; _offsetBiasGradients = null;
        _maskWeightGradients = null; _maskBiasGradients = null;
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

        // Clear GPU caching fields
        _gpuInput = null;
        _gpuInputShape = null;
        _gpuOffsets?.Dispose();
        _gpuOffsets = null;
        _gpuMask?.Dispose();
        _gpuMask = null;
    }

    #endregion
}
