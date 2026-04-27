using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Engines;
using AiDotNet.Interfaces;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Residual Dense Block (RDB) as used in ESRGAN and Real-ESRGAN generators.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This implements the Residual Dense Block from the ESRGAN paper (Wang et al., 2018).
/// It differs from DenseNet's Dense Block by using LeakyReLU, no batch normalization,
/// and a local residual connection with scaling.
/// </para>
/// <para>
/// The architecture consists of 5 convolutional layers with dense connections:
/// <code>
/// x0 = input (numFeatures channels)
/// x1 = LeakyReLU(Conv(x0))                              [growthChannels]
/// x2 = LeakyReLU(Conv(concat(x0, x1)))                  [growthChannels]
/// x3 = LeakyReLU(Conv(concat(x0, x1, x2)))              [growthChannels]
/// x4 = LeakyReLU(Conv(concat(x0, x1, x2, x3)))          [growthChannels]
/// x5 = Conv(concat(x0, x1, x2, x3, x4))                 [numFeatures, no activation]
/// output = x5 * residualScale + x0                      [local residual learning]
/// </code>
/// </para>
/// <para>
/// <b>For Beginners:</b> This block is the building block of ESRGAN's generator.
/// It combines ideas from DenseNet (dense connections) and ResNet (residual learning):
///
/// 1. **Dense connections**: Each conv layer can see ALL previous features
/// 2. **Local residual**: The block's output is added to its input (helps training)
/// 3. **Residual scaling**: The residual is scaled by 0.2 (prevents instability)
/// 4. **LeakyReLU**: Uses LeakyReLU(0.2) instead of ReLU (better gradients)
/// 5. **No batch norm**: ESRGAN generator doesn't use batch normalization
///
/// The default parameters (64 features, 32 growth, 0.2 scale) are from the paper.
/// </para>
/// <para>
/// <b>Reference:</b> Wang et al., "ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks",
/// ECCV 2018 Workshops. https://arxiv.org/abs/1809.00219
/// </para>
/// </remarks>
[LayerCategory(LayerCategory.Residual)]
[LayerCategory(LayerCategory.Convolution)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, ExpectedInputRank = 3, Cost = ComputeCost.High, TestInputShape = "4, 8, 8", TestConstructorArgs = "4, 4, 3")]
public class ResidualDenseBlock<T> : LayerBase<T>
{
    #region Fields

    /// <summary>
    /// The 5 convolutional layers in the dense block.
    /// </summary>
    private readonly ConvolutionalLayer<T>[] _convLayers;

    /// <summary>
    /// LeakyReLU activation with negative slope 0.2 (from ESRGAN paper).
    /// </summary>
    private readonly LeakyReLUActivation<T> _activation;

    /// <summary>
    /// Residual scaling factor. Default: 0.2 from the paper.
    /// </summary>
    private readonly double _residualScale;

    /// <summary>
    /// Number of output channels for intermediate conv layers (growth rate).
    /// </summary>
    private readonly int _growthChannels;

    /// <summary>
    /// Number of input/output channels (feature channels).
    /// </summary>
    private readonly int _numFeatures;

    /// <summary>
    /// Input height.
    /// </summary>
    private readonly int _inputHeight;

    /// <summary>
    /// Input width.
    /// </summary>
    private readonly int _inputWidth;

    /// <summary>
    /// Cached input for backpropagation.
    /// </summary>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Cached intermediate conv outputs (before activation) for backpropagation.
    /// </summary>
    private Tensor<T>[]? _convOutputs;

    /// <summary>
    /// Cached intermediate activation outputs for backpropagation.
    /// </summary>
    private Tensor<T>[]? _activationOutputs;

    /// <summary>
    /// Cached concatenated inputs to each conv layer for backpropagation.
    /// </summary>
    private Tensor<T>[]? _concatInputs;

    // GPU cached tensors for backward pass
    private Tensor<T>? _gpuInput;
    private Tensor<T>? _gpuConv1Out;
    private Tensor<T>? _gpuX1Activated;
    private Tensor<T>? _gpuConcat1;
    private Tensor<T>? _gpuConv2Out;
    private Tensor<T>? _gpuX2Activated;
    private Tensor<T>? _gpuConcat2;
    private Tensor<T>? _gpuConv3Out;
    private Tensor<T>? _gpuX3Activated;
    private Tensor<T>? _gpuConcat3;
    private Tensor<T>? _gpuConv4Out;
    private Tensor<T>? _gpuX4Activated;
    private Tensor<T>? _gpuConcat4;
    private Tensor<T>? _gpuConv5Out;
    private int _gpuBatch;
    private int _gpuHeight;
    private int _gpuWidth;

    #endregion

    #region Properties

    /// <summary>
    /// Gets the residual scaling factor.
    /// </summary>
    public double ResidualScale => _residualScale;

    /// <summary>
    /// Gets the growth channels (intermediate conv output channels).
    /// </summary>
    public int GrowthChannels => _growthChannels;

    /// <summary>
    /// Gets the number of feature channels.
    /// </summary>
    public int NumFeatures => _numFeatures;

    /// <inheritdoc />
    public override int ParameterCount => GetParameters().Length;
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets a value indicating whether this layer supports GPU execution.
    /// </summary>
    protected override bool SupportsGpuExecution => true;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new Residual Dense Block.
    /// </summary>
    /// <param name="numFeatures">Number of input/output feature channels. Default: 64 (from paper).</param>
    /// <param name="growthChannels">Number of channels for intermediate convolutions. Default: 32 (from paper).</param>
    /// <param name="inputHeight">Height of input feature maps.</param>
    /// <param name="inputWidth">Width of input feature maps.</param>
    /// <param name="residualScale">Residual scaling factor. Default: 0.2 (from paper).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Create a Residual Dense Block for ESRGAN:
    /// <code>
    /// var rdb = new ResidualDenseBlock&lt;float&gt;(
    ///     numFeatures: 64,       // Main feature channels (paper default)
    ///     growthChannels: 32,    // Intermediate channels (paper default)
    ///     inputHeight: 128,
    ///     inputWidth: 128,
    ///     residualScale: 0.2     // Residual scaling (paper default)
    /// );
    /// </code>
    /// </para>
    /// </remarks>
    public ResidualDenseBlock(
        int numFeatures = 64,
        int growthChannels = 32,
        int inputHeight = 64,
        int inputWidth = 64,
        double residualScale = 0.2)
        : base(
            [numFeatures, inputHeight, inputWidth],
            [numFeatures, inputHeight, inputWidth])
    {
        if (numFeatures <= 0)
            throw new ArgumentOutOfRangeException(nameof(numFeatures), "Number of features must be positive.");
        if (growthChannels <= 0)
            throw new ArgumentOutOfRangeException(nameof(growthChannels), "Growth channels must be positive.");
        if (residualScale < 0 || residualScale > 1)
            throw new ArgumentOutOfRangeException(nameof(residualScale), "Residual scale must be between 0 and 1.");

        _numFeatures = numFeatures;
        _growthChannels = growthChannels;
        _inputHeight = inputHeight;
        _inputWidth = inputWidth;
        _residualScale = residualScale;
        _activation = new LeakyReLUActivation<T>(0.2); // ESRGAN uses 0.2 negative slope

        // Create 5 convolutional layers with increasing input channels due to concatenation
        // All use 3x3 kernels with padding=1 to preserve spatial dimensions
        _convLayers = new ConvolutionalLayer<T>[5];

        // Conv1: numFeatures -> growthChannels
        _convLayers[0] = new ConvolutionalLayer<T>(
            outputDepth: growthChannels,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            activationFunction: null);

        // Conv2: numFeatures + growthChannels -> growthChannels
        _convLayers[1] = new ConvolutionalLayer<T>(
            outputDepth: growthChannels,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            activationFunction: null);

        // Conv3: numFeatures + 2*growthChannels -> growthChannels
        _convLayers[2] = new ConvolutionalLayer<T>(
            outputDepth: growthChannels,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            activationFunction: null);

        // Conv4: numFeatures + 3*growthChannels -> growthChannels
        _convLayers[3] = new ConvolutionalLayer<T>(
            outputDepth: growthChannels,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            activationFunction: null);

        // Conv5: numFeatures + 4*growthChannels -> numFeatures (back to original channels)
        _convLayers[4] = new ConvolutionalLayer<T>(
            outputDepth: numFeatures,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            activationFunction: null);

        foreach (var conv in _convLayers)
            RegisterSubLayer(conv);
    }

    #endregion

    #region Forward Pass

    /// <inheritdoc />
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        _convOutputs = new Tensor<T>[5];
        _activationOutputs = new Tensor<T>[4]; // Only first 4 have activation
        _concatInputs = new Tensor<T>[5];

        // x0 = input
        var x0 = input;

        // Conv1: x1 = LeakyReLU(Conv(x0))
        _concatInputs[0] = x0;
        _convOutputs[0] = _convLayers[0].Forward(x0);
        _activationOutputs[0] = ApplyLeakyReLU(_convOutputs[0]);
        var x1 = _activationOutputs[0];

        // Conv2: x2 = LeakyReLU(Conv(concat(x0, x1)))
        var concat1 = ConcatenateChannels(x0, x1);
        _concatInputs[1] = concat1;
        _convOutputs[1] = _convLayers[1].Forward(concat1);
        _activationOutputs[1] = ApplyLeakyReLU(_convOutputs[1]);
        var x2 = _activationOutputs[1];

        // Conv3: x3 = LeakyReLU(Conv(concat(x0, x1, x2)))
        var concat2 = ConcatenateChannels(concat1, x2);
        _concatInputs[2] = concat2;
        _convOutputs[2] = _convLayers[2].Forward(concat2);
        _activationOutputs[2] = ApplyLeakyReLU(_convOutputs[2]);
        var x3 = _activationOutputs[2];

        // Conv4: x4 = LeakyReLU(Conv(concat(x0, x1, x2, x3)))
        var concat3 = ConcatenateChannels(concat2, x3);
        _concatInputs[3] = concat3;
        _convOutputs[3] = _convLayers[3].Forward(concat3);
        _activationOutputs[3] = ApplyLeakyReLU(_convOutputs[3]);
        var x4 = _activationOutputs[3];

        // Conv5: x5 = Conv(concat(x0, x1, x2, x3, x4)) - NO activation
        var concat4 = ConcatenateChannels(concat3, x4);
        _concatInputs[4] = concat4;
        _convOutputs[4] = _convLayers[4].Forward(concat4);
        var x5 = _convOutputs[4];

        // Local residual learning: output = x5 * residualScale + x0
        return AddResidual(x5, x0, _residualScale);
    }

    /// <summary>
    /// Performs the forward pass on GPU tensors.
    /// </summary>
    /// <param name="inputs">GPU tensor inputs.</param>
    /// <returns>GPU tensor output after residual dense block processing.</returns>
    public override Tensor<T> ForwardGpu(params Tensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));
        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("ForwardGpu requires a DirectGpuTensorEngine.");

        var backend = gpuEngine.GetBackend();
        if (backend is null)
            throw new InvalidOperationException("GPU backend unavailable.");

        var input = inputs[0];
        var originalShape = input._shape;

        // Support any rank >= 3: last 3 dims are [C, H, W], earlier dims are batch-like
        if (originalShape.Length < 3)
            throw new ArgumentException($"ResidualDenseBlock requires at least 3D tensor [C, H, W]. Got rank {originalShape.Length}.");

        int batch, channels, height, width;
        int[] shape;
        if (originalShape.Length == 3)
        {
            batch = 1;
            channels = originalShape[0];
            height = originalShape[1];
            width = originalShape[2];
            shape = originalShape;
        }
        else if (originalShape.Length == 4)
        {
            batch = originalShape[0];
            channels = originalShape[1];
            height = originalShape[2];
            width = originalShape[3];
            shape = originalShape;
        }
        else
        {
            // Higher rank: flatten leading dimensions into batch
            batch = 1;
            for (int d = 0; d < originalShape.Length - 3; d++)
                batch *= originalShape[d];
            channels = originalShape[originalShape.Length - 3];
            height = originalShape[originalShape.Length - 2];
            width = originalShape[originalShape.Length - 1];
            shape = new[] { batch, channels, height, width };
            input = gpuEngine.ReshapeGpu(input, shape);
        }

        int spatialSize = height * width;

        // Cache dimensions for backward pass
        if (IsTrainingMode)
        {
            _gpuBatch = batch;
            _gpuHeight = height;
            _gpuWidth = width;
            _gpuInput = input;
        }

        // x0 = input
        var x0Buffer = input.Buffer;

        // Conv1: x1 = LeakyReLU(Conv(x0))
        var conv1Out = _convLayers[0].ForwardGpu(input);
        int x1Size = batch * _growthChannels * spatialSize;
        var x1ActivatedBuffer = backend.AllocateBuffer(x1Size);
        backend.LeakyRelu(conv1Out.Buffer, x1ActivatedBuffer, 0.2f, x1Size);

        // Concatenate x0 and x1 for conv2 input: [channels + growthChannels]
        int concat1Channels = _numFeatures + _growthChannels;
        int concat1Size = batch * concat1Channels * spatialSize;
        var concat1Buffer = backend.AllocateBuffer(concat1Size);
        ConcatenateChannelsGpu(backend, x0Buffer, x1ActivatedBuffer, concat1Buffer,
            batch, _numFeatures, _growthChannels, spatialSize);
        var concat1Tensor = GpuTensorHelper.UploadToGpu<T>(backend, concat1Buffer,
            shape.Length == 3 ? [concat1Channels, height, width] : [batch, concat1Channels, height, width],
            GpuTensorRole.Activation, ownsBuffer: !IsTrainingMode);

        // Conv2: x2 = LeakyReLU(Conv(concat(x0, x1)))
        var conv2Out = _convLayers[1].ForwardGpu(concat1Tensor);
        int x2Size = batch * _growthChannels * spatialSize;
        var x2ActivatedBuffer = backend.AllocateBuffer(x2Size);
        backend.LeakyRelu(conv2Out.Buffer, x2ActivatedBuffer, 0.2f, x2Size);

        // Concatenate concat1 and x2 for conv3 input: [channels + 2*growthChannels]
        int concat2Channels = _numFeatures + 2 * _growthChannels;
        int concat2Size = batch * concat2Channels * spatialSize;
        var concat2Buffer = backend.AllocateBuffer(concat2Size);
        ConcatenateChannelsGpu(backend, concat1Buffer, x2ActivatedBuffer, concat2Buffer,
            batch, concat1Channels, _growthChannels, spatialSize);
        var concat2Tensor = GpuTensorHelper.UploadToGpu<T>(backend, concat2Buffer,
            shape.Length == 3 ? [concat2Channels, height, width] : [batch, concat2Channels, height, width],
            GpuTensorRole.Activation, ownsBuffer: !IsTrainingMode);

        // Conv3: x3 = LeakyReLU(Conv(concat(x0, x1, x2)))
        var conv3Out = _convLayers[2].ForwardGpu(concat2Tensor);
        int x3Size = batch * _growthChannels * spatialSize;
        var x3ActivatedBuffer = backend.AllocateBuffer(x3Size);
        backend.LeakyRelu(conv3Out.Buffer, x3ActivatedBuffer, 0.2f, x3Size);

        // Concatenate concat2 and x3 for conv4 input: [channels + 3*growthChannels]
        int concat3Channels = _numFeatures + 3 * _growthChannels;
        int concat3Size = batch * concat3Channels * spatialSize;
        var concat3Buffer = backend.AllocateBuffer(concat3Size);
        ConcatenateChannelsGpu(backend, concat2Buffer, x3ActivatedBuffer, concat3Buffer,
            batch, concat2Channels, _growthChannels, spatialSize);
        var concat3Tensor = GpuTensorHelper.UploadToGpu<T>(backend, concat3Buffer,
            shape.Length == 3 ? [concat3Channels, height, width] : [batch, concat3Channels, height, width],
            GpuTensorRole.Activation, ownsBuffer: !IsTrainingMode);

        // Conv4: x4 = LeakyReLU(Conv(concat(x0, x1, x2, x3)))
        var conv4Out = _convLayers[3].ForwardGpu(concat3Tensor);
        int x4Size = batch * _growthChannels * spatialSize;
        var x4ActivatedBuffer = backend.AllocateBuffer(x4Size);
        backend.LeakyRelu(conv4Out.Buffer, x4ActivatedBuffer, 0.2f, x4Size);

        // Concatenate concat3 and x4 for conv5 input: [channels + 4*growthChannels]
        int concat4Channels = _numFeatures + 4 * _growthChannels;
        int concat4Size = batch * concat4Channels * spatialSize;
        var concat4Buffer = backend.AllocateBuffer(concat4Size);
        ConcatenateChannelsGpu(backend, concat3Buffer, x4ActivatedBuffer, concat4Buffer,
            batch, concat3Channels, _growthChannels, spatialSize);
        var concat4Tensor = GpuTensorHelper.UploadToGpu<T>(backend, concat4Buffer,
            shape.Length == 3 ? [concat4Channels, height, width] : [batch, concat4Channels, height, width],
            GpuTensorRole.Activation, ownsBuffer: !IsTrainingMode);

        // Conv5: x5 = Conv(concat(x0, x1, x2, x3, x4)) - NO activation
        var x5 = _convLayers[4].ForwardGpu(concat4Tensor);
        int x5Size = batch * _numFeatures * spatialSize;

        // Local residual learning: output = x5 * residualScale + x0
        var scaledBuffer = backend.AllocateBuffer(x5Size);
        backend.Scale(x5.Buffer, scaledBuffer, (float)_residualScale, x5Size);

        var outputBuffer = backend.AllocateBuffer(x5Size);
        backend.Add(scaledBuffer, x0Buffer, outputBuffer, x5Size);
        scaledBuffer.Dispose();

        // Cache tensors for backward pass or dispose
        if (IsTrainingMode)
        {
            int[] activationShape = shape.Length == 3
                ? [_growthChannels, height, width]
                : [batch, _growthChannels, height, width];

            _gpuConv1Out = conv1Out;
            _gpuX1Activated = GpuTensorHelper.UploadToGpu<T>(backend, x1ActivatedBuffer, activationShape, GpuTensorRole.Intermediate, ownsBuffer: true);
            _gpuConcat1 = concat1Tensor;
            _gpuConv2Out = conv2Out;
            _gpuX2Activated = GpuTensorHelper.UploadToGpu<T>(backend, x2ActivatedBuffer, activationShape, GpuTensorRole.Intermediate, ownsBuffer: true);
            _gpuConcat2 = concat2Tensor;
            _gpuConv3Out = conv3Out;
            _gpuX3Activated = GpuTensorHelper.UploadToGpu<T>(backend, x3ActivatedBuffer, activationShape, GpuTensorRole.Intermediate, ownsBuffer: true);
            _gpuConcat3 = concat3Tensor;
            _gpuConv4Out = conv4Out;
            _gpuX4Activated = GpuTensorHelper.UploadToGpu<T>(backend, x4ActivatedBuffer, activationShape, GpuTensorRole.Intermediate, ownsBuffer: true);
            _gpuConcat4 = concat4Tensor;
            _gpuConv5Out = x5;
        }
        else
        {
            // Not training - dispose intermediate buffers
            x1ActivatedBuffer.Dispose();
            x2ActivatedBuffer.Dispose();
            x3ActivatedBuffer.Dispose();
            x4ActivatedBuffer.Dispose();
        }

        var result = GpuTensorHelper.UploadToGpu<T>(backend, outputBuffer, shape, GpuTensorRole.Activation, ownsBuffer: true);

        // Restore original tensor rank for higher-rank input
        if (originalShape.Length > 4)
        {
            return gpuEngine.ReshapeGpu(result, originalShape);
        }

        return result;
    }

    /// <summary>
    /// Concatenates two tensors along the channel dimension on GPU.
    /// For NCHW format: output[b, 0:channelsA, :, :] = A[b, :, :, :]
    ///                  output[b, channelsA:, :, :] = B[b, :, :, :]
    /// </summary>
    private static void ConcatenateChannelsGpu(
        Tensors.Engines.DirectGpu.IDirectGpuBackend backend,
        Tensors.Engines.DirectGpu.IGpuBuffer a,
        Tensors.Engines.DirectGpu.IGpuBuffer b,
        Tensors.Engines.DirectGpu.IGpuBuffer output,
        int batch, int channelsA, int channelsB, int spatialSize)
    {
        int totalChannels = channelsA + channelsB;
        int aChannelDataPerBatch = channelsA * spatialSize;
        int bChannelDataPerBatch = channelsB * spatialSize;
        int outChannelDataPerBatch = totalChannels * spatialSize;

        // Use Copy2DStrided for channel concatenation:
        // - numRows = batch (each row is one batch's worth of channel data)
        // - srcCols = channelsX * spatialSize (data per batch in source)
        // - destTotalCols = totalChannels * spatialSize (data per batch in output)
        // - destColOffset = offset within each batch's output data

        // Copy A's channels to output[b, 0:channelsA, :, :]
        backend.Copy2DStrided(a, output, batch, aChannelDataPerBatch, outChannelDataPerBatch, 0);

        // Copy B's channels to output[b, channelsA:, :, :]
        backend.Copy2DStrided(b, output, batch, bChannelDataPerBatch, outChannelDataPerBatch, aChannelDataPerBatch);
    }

    #endregion

    #region Backward Pass

    #endregion

    #region Helper Methods

    /// <summary>
    /// Concatenates two tensors along the channel dimension.
    /// </summary>
    private Tensor<T> ConcatenateChannels(Tensor<T> a, Tensor<T> b)
    {
        // Channel axis is 0 for 3D [C,H,W] and 1 for 4D [B,C,H,W]
        int channelAxis = a.Rank == 4 ? 1 : 0;
        return Engine.TensorConcatenate([a, b], axis: channelAxis);
    }

    /// <summary>
    /// Splits a tensor along the channel dimension.
    /// </summary>
    private (Tensor<T> first, Tensor<T> second) SplitGradient(Tensor<T> grad, int firstChannels, int secondChannels)
    {
        if (grad.Rank == 3)
        {
            int height = grad.Shape[1];
            int width = grad.Shape[2];
            int spatialSize = height * width;

            var first = new Tensor<T>([firstChannels, height, width]);
            var second = new Tensor<T>([secondChannels, height, width]);

            for (int c = 0; c < firstChannels; c++)
            {
                for (int hw = 0; hw < spatialSize; hw++)
                {
                    first.Data.Span[c * spatialSize + hw] = grad.Data.Span[c * spatialSize + hw];
                }
            }

            for (int c = 0; c < secondChannels; c++)
            {
                for (int hw = 0; hw < spatialSize; hw++)
                {
                    second.Data.Span[c * spatialSize + hw] = grad.Data.Span[(firstChannels + c) * spatialSize + hw];
                }
            }

            return (first, second);
        }
        else // 4D
        {
            int batch = grad.Shape[0];
            int height = grad.Shape[2];
            int width = grad.Shape[3];
            int spatialSize = height * width;
            int totalChannels = firstChannels + secondChannels;

            var first = new Tensor<T>([batch, firstChannels, height, width]);
            var second = new Tensor<T>([batch, secondChannels, height, width]);

            for (int n = 0; n < batch; n++)
            {
                for (int c = 0; c < firstChannels; c++)
                {
                    for (int hw = 0; hw < spatialSize; hw++)
                    {
                        int srcIdx = n * totalChannels * spatialSize + c * spatialSize + hw;
                        int dstIdx = n * firstChannels * spatialSize + c * spatialSize + hw;
                        first.Data.Span[dstIdx] = grad.Data.Span[srcIdx];
                    }
                }

                for (int c = 0; c < secondChannels; c++)
                {
                    for (int hw = 0; hw < spatialSize; hw++)
                    {
                        int srcIdx = n * totalChannels * spatialSize + (firstChannels + c) * spatialSize + hw;
                        int dstIdx = n * secondChannels * spatialSize + c * spatialSize + hw;
                        second.Data.Span[dstIdx] = grad.Data.Span[srcIdx];
                    }
                }
            }

            return (first, second);
        }
    }

    /// <summary>
    /// Applies LeakyReLU activation to the input tensor.
    /// </summary>
    private Tensor<T> ApplyLeakyReLU(Tensor<T> input)
    {
        var output = TensorAllocator.Rent<T>(input._shape);
        for (int i = 0; i < input.Length; i++)
        {
            output.Data.Span[i] = _activation.Activate(input.Data.Span[i]);
        }
        return output;
    }

    /// <summary>
    /// Backward pass through LeakyReLU activation.
    /// </summary>
    private Tensor<T> BackwardActivation(Tensor<T> activationOutput, Tensor<T> gradient)
    {
        var output = TensorAllocator.Rent<T>(gradient._shape);
        for (int i = 0; i < gradient.Length; i++)
        {
            output.Data.Span[i] = NumOps.Multiply(
                gradient.Data.Span[i],
                _activation.Derivative(activationOutput.Data.Span[i]));
        }
        return output;
    }

    /// <summary>
    /// Adds residual with scaling: output = a * scale + b.
    /// </summary>
    private Tensor<T> AddResidual(Tensor<T> a, Tensor<T> b, double scale)
    {
        var scaleT = NumOps.FromDouble(scale);
        var scaled = Engine.TensorMultiplyScalar(a, scaleT);
        return Engine.TensorAdd(scaled, b);
    }

    /// <summary>
    /// Scales a tensor by a factor.
    /// </summary>
    private Tensor<T> ScaleGradient(Tensor<T> gradient, double scale)
    {
        var scaleT = NumOps.FromDouble(scale);
        return Engine.TensorMultiplyScalar(gradient, scaleT);
    }

    /// <summary>
    /// Adds two tensors element-wise.
    /// </summary>
    private Tensor<T> AddTensors(Tensor<T> a, Tensor<T> b)
    {
        return Engine.TensorAdd(a, b);
    }

    #endregion

    #region Parameter Management

    /// <inheritdoc />
    public override void UpdateParameters(T learningRate)
    {
        foreach (var conv in _convLayers)
        {
            conv.UpdateParameters(learningRate);
        }
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var allParams = new List<T>();
        foreach (var conv in _convLayers)
        {
            var convParams = conv.GetParameters();
            for (int i = 0; i < convParams.Length; i++)
            {
                allParams.Add(convParams[i]);
            }
        }
        return new Vector<T>([.. allParams]);
    }

    public override Vector<T> GetParameterGradients()
    {
        var gradVectors = _convLayers
            .Select(c => c.GetParameterGradients())
            .ToArray();
        return Vector<T>.Concatenate(gradVectors);
    }

    public override void ClearGradients()
    {
        foreach (var conv in _convLayers)
            conv.ClearGradients();
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        int offset = 0;
        foreach (var conv in _convLayers)
        {
            int count = conv.GetParameters().Length;
            conv.SetParameters(parameters.SubVector(offset, count));
            offset += count;
        }
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        _lastInput = null;
        _convOutputs = null;
        _activationOutputs = null;
        _concatInputs = null;

        // Clear GPU cached tensors
        _gpuInput = null;
        _gpuConv1Out = null;
        _gpuX1Activated = null;
        _gpuConcat1 = null;
        _gpuConv2Out = null;
        _gpuX2Activated = null;
        _gpuConcat2 = null;
        _gpuConv3Out = null;
        _gpuX3Activated = null;
        _gpuConcat3 = null;
        _gpuConv4Out = null;
        _gpuX4Activated = null;
        _gpuConcat4 = null;
        _gpuConv5Out = null;
        _gpuBatch = 0;
        _gpuHeight = 0;
        _gpuWidth = 0;

        foreach (var conv in _convLayers)
        {
            conv.ResetState();
        }
    }


    /// <summary>
    /// Builds a Conv2D computation node from a ConvolutionalLayer.
    /// </summary>
    private ComputationNode<T> BuildConvNode(ConvolutionalLayer<T> conv, ComputationNode<T> input, string namePrefix)
    {
        var biases = conv.GetBiases();
        return TensorOperations<T>.Conv2D(
            input,
            TensorOperations<T>.Constant(conv.GetFilters(), $"{namePrefix}kernel"),
            biases is not null ? TensorOperations<T>.Constant(biases, $"{namePrefix}bias") : null,
            stride: new int[] { conv.Stride, conv.Stride },
            padding: new int[] { conv.Padding, conv.Padding });
    }

    /// <summary>
    /// Scales a computation node by a scalar value using element-wise multiplication.
    /// </summary>
    private static ComputationNode<T> ScaleNode(ComputationNode<T> node, double scale, string name)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var scaleValue = numOps.FromDouble(scale);

        // Create a constant tensor filled with the scale value matching the input shape
        var scaleTensor = new Tensor<T>(node.Value._shape);
        for (int i = 0; i < scaleTensor.Length; i++)
        {
            scaleTensor.Data.Span[i] = scaleValue;
        }

        var scaleNode = TensorOperations<T>.Constant(scaleTensor, name);
        return TensorOperations<T>.ElementwiseMultiply(node, scaleNode);
    }

    #endregion

}
