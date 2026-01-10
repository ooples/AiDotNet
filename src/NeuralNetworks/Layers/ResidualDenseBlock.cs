using AiDotNet.ActivationFunctions;
using AiDotNet.Autodiff;
using AiDotNet.Engines;
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
public class ResidualDenseBlock<T> : LayerBase<T>, IChainableComputationGraph<T>
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
    private IGpuTensor<T>? _gpuInput;
    private IGpuTensor<T>? _gpuConv1Out;
    private IGpuTensor<T>? _gpuX1Activated;
    private IGpuTensor<T>? _gpuConcat1;
    private IGpuTensor<T>? _gpuConv2Out;
    private IGpuTensor<T>? _gpuX2Activated;
    private IGpuTensor<T>? _gpuConcat2;
    private IGpuTensor<T>? _gpuConv3Out;
    private IGpuTensor<T>? _gpuX3Activated;
    private IGpuTensor<T>? _gpuConcat3;
    private IGpuTensor<T>? _gpuConv4Out;
    private IGpuTensor<T>? _gpuX4Activated;
    private IGpuTensor<T>? _gpuConcat4;
    private IGpuTensor<T>? _gpuConv5Out;
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
    public override bool SupportsTraining => true;

    /// <inheritdoc />
    public override bool SupportsJitCompilation
    {
        get
        {
            // Check all conv layers support JIT
            foreach (var conv in _convLayers)
            {
                if (!conv.SupportsJitCompilation)
                    return false;
            }
            return true;
        }
    }

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
            inputDepth: numFeatures,
            inputHeight: inputHeight,
            inputWidth: inputWidth,
            outputDepth: growthChannels,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            activationFunction: null);

        // Conv2: numFeatures + growthChannels -> growthChannels
        _convLayers[1] = new ConvolutionalLayer<T>(
            inputDepth: numFeatures + growthChannels,
            inputHeight: inputHeight,
            inputWidth: inputWidth,
            outputDepth: growthChannels,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            activationFunction: null);

        // Conv3: numFeatures + 2*growthChannels -> growthChannels
        _convLayers[2] = new ConvolutionalLayer<T>(
            inputDepth: numFeatures + 2 * growthChannels,
            inputHeight: inputHeight,
            inputWidth: inputWidth,
            outputDepth: growthChannels,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            activationFunction: null);

        // Conv4: numFeatures + 3*growthChannels -> growthChannels
        _convLayers[3] = new ConvolutionalLayer<T>(
            inputDepth: numFeatures + 3 * growthChannels,
            inputHeight: inputHeight,
            inputWidth: inputWidth,
            outputDepth: growthChannels,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            activationFunction: null);

        // Conv5: numFeatures + 4*growthChannels -> numFeatures (back to original channels)
        _convLayers[4] = new ConvolutionalLayer<T>(
            inputDepth: numFeatures + 4 * growthChannels,
            inputHeight: inputHeight,
            inputWidth: inputWidth,
            outputDepth: numFeatures,
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
    public override IGpuTensor<T> ForwardGpu(params IGpuTensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));
        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("ForwardGpu requires a DirectGpuTensorEngine.");

        var backend = gpuEngine.GetBackend();
        if (backend is null)
            throw new InvalidOperationException("GPU backend unavailable.");

        var input = inputs[0];
        var shape = input.Shape;

        // Handle 3D [C,H,W] or 4D [B,C,H,W] input
        int batch, channels, height, width;
        if (shape.Length == 3)
        {
            batch = 1;
            channels = shape[0];
            height = shape[1];
            width = shape[2];
        }
        else if (shape.Length == 4)
        {
            batch = shape[0];
            channels = shape[1];
            height = shape[2];
            width = shape[3];
        }
        else
        {
            throw new ArgumentException($"ResidualDenseBlock requires 3D or 4D input, got {shape.Length}D.");
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
        var concat1Tensor = new GpuTensor<T>(backend, concat1Buffer,
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
        var concat2Tensor = new GpuTensor<T>(backend, concat2Buffer,
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
        var concat3Tensor = new GpuTensor<T>(backend, concat3Buffer,
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
        var concat4Tensor = new GpuTensor<T>(backend, concat4Buffer,
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
            _gpuX1Activated = new GpuTensor<T>(backend, x1ActivatedBuffer, activationShape, GpuTensorRole.Intermediate, ownsBuffer: true);
            _gpuConcat1 = concat1Tensor;
            _gpuConv2Out = conv2Out;
            _gpuX2Activated = new GpuTensor<T>(backend, x2ActivatedBuffer, activationShape, GpuTensorRole.Intermediate, ownsBuffer: true);
            _gpuConcat2 = concat2Tensor;
            _gpuConv3Out = conv3Out;
            _gpuX3Activated = new GpuTensor<T>(backend, x3ActivatedBuffer, activationShape, GpuTensorRole.Intermediate, ownsBuffer: true);
            _gpuConcat3 = concat3Tensor;
            _gpuConv4Out = conv4Out;
            _gpuX4Activated = new GpuTensor<T>(backend, x4ActivatedBuffer, activationShape, GpuTensorRole.Intermediate, ownsBuffer: true);
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

        return new GpuTensor<T>(backend, outputBuffer, shape, GpuTensorRole.Activation, ownsBuffer: true);
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

    /// <inheritdoc />
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _convOutputs == null || _activationOutputs == null || _concatInputs == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        var x0 = _lastInput;

        // Gradient through residual: d(x5 * scale + x0)/d(x5) = scale, d(...)/d(x0) = 1
        var x5Gradient = ScaleGradient(outputGradient, _residualScale);
        var x0GradFromResidual = outputGradient; // Identity gradient

        // Backward through Conv5 (no activation)
        var concat4Gradient = _convLayers[4].Backward(x5Gradient);

        // Split concat4 gradient: [concat3, x4]
        var (concat3Grad, x4Grad) = SplitGradient(concat4Gradient,
            _numFeatures + 3 * _growthChannels, _growthChannels);

        // Backward through activation and Conv4
        var conv4PreGrad = BackwardActivation(_activationOutputs[3], x4Grad);
        var concat3GradFromConv4 = _convLayers[3].Backward(conv4PreGrad);
        concat3Grad = AddTensors(concat3Grad, concat3GradFromConv4);

        // Split concat3 gradient: [concat2, x3]
        var (concat2Grad, x3Grad) = SplitGradient(concat3Grad,
            _numFeatures + 2 * _growthChannels, _growthChannels);

        // Backward through activation and Conv3
        var conv3PreGrad = BackwardActivation(_activationOutputs[2], x3Grad);
        var concat2GradFromConv3 = _convLayers[2].Backward(conv3PreGrad);
        concat2Grad = AddTensors(concat2Grad, concat2GradFromConv3);

        // Split concat2 gradient: [concat1, x2]
        var (concat1Grad, x2Grad) = SplitGradient(concat2Grad,
            _numFeatures + _growthChannels, _growthChannels);

        // Backward through activation and Conv2
        var conv2PreGrad = BackwardActivation(_activationOutputs[1], x2Grad);
        var concat1GradFromConv2 = _convLayers[1].Backward(conv2PreGrad);
        concat1Grad = AddTensors(concat1Grad, concat1GradFromConv2);

        // Split concat1 gradient: [x0, x1]
        var (x0GradFromConcat1, x1Grad) = SplitGradient(concat1Grad,
            _numFeatures, _growthChannels);

        // Backward through activation and Conv1
        var conv1PreGrad = BackwardActivation(_activationOutputs[0], x1Grad);
        var x0GradFromConv1 = _convLayers[0].Backward(conv1PreGrad);

        // Combine all gradients flowing to x0
        var totalX0Grad = AddTensors(x0GradFromResidual, x0GradFromConcat1);
        totalX0Grad = AddTensors(totalX0Grad, x0GradFromConv1);

        return totalX0Grad;
    }

    /// <summary>
    /// Performs the backward pass on GPU tensors.
    /// </summary>
    /// <param name="outputGradient">GPU tensor representing the gradient from the next layer.</param>
    /// <returns>GPU tensor representing the gradient with respect to the input.</returns>
    public IGpuTensor<T> BackwardGpu(IGpuTensor<T> outputGradient)
    {
        if (_gpuInput == null || _gpuConv1Out == null || _gpuX1Activated == null ||
            _gpuConcat1 == null || _gpuConv2Out == null || _gpuX2Activated == null ||
            _gpuConcat2 == null || _gpuConv3Out == null || _gpuX3Activated == null ||
            _gpuConcat3 == null || _gpuConv4Out == null || _gpuX4Activated == null ||
            _gpuConcat4 == null || _gpuConv5Out == null)
        {
            throw new InvalidOperationException("ForwardGpu must be called in training mode before BackwardGpu.");
        }

        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("BackwardGpu requires a DirectGpuTensorEngine.");

        var backend = gpuEngine.GetBackend();
        if (backend is null)
            throw new InvalidOperationException("GPU backend unavailable.");

        int spatialSize = _gpuHeight * _gpuWidth;
        int outputSize = _gpuBatch * _numFeatures * spatialSize;

        // Gradient through residual: d(x5 * scale + x0)/d(x5) = scale, d(...)/d(x0) = 1
        // Scale outputGradient by residualScale for x5 gradient
        var x5GradBuffer = backend.AllocateBuffer(outputSize);
        backend.Scale(outputGradient.Buffer, x5GradBuffer, (float)_residualScale, outputSize);
        var x5Grad = new GpuTensor<T>(backend, x5GradBuffer, outputGradient.Shape, GpuTensorRole.Gradient, ownsBuffer: true);

        // Keep original outputGradient as x0GradFromResidual (will be accumulated later)

        // Backward through Conv5 (no activation) to get concat4 gradient
        var concat4Grad = _convLayers[4].BackwardGpu(x5Grad);

        // Split concat4 gradient into [concat3, x4] using SliceGpu (axis=1 for channels)
        int concat3Channels = _numFeatures + 3 * _growthChannels;
        var concat3Grad = gpuEngine.SliceGpu<T>(concat4Grad, 1, 0, concat3Channels);
        var x4Grad = gpuEngine.SliceGpu<T>(concat4Grad, 1, concat3Channels, concat3Channels + _growthChannels);

        // Backward through activation and Conv4
        var conv4PreGrad = gpuEngine.LeakyReluBackwardGpu<T>(x4Grad, _gpuConv4Out, 0.2f);
        var concat3GradFromConv4 = _convLayers[3].BackwardGpu(conv4PreGrad);
        concat3Grad = gpuEngine.AddGpu<T>(concat3Grad, concat3GradFromConv4);

        // Split concat3 gradient into [concat2, x3]
        int concat2Channels = _numFeatures + 2 * _growthChannels;
        var concat2Grad = gpuEngine.SliceGpu<T>(concat3Grad, 1, 0, concat2Channels);
        var x3Grad = gpuEngine.SliceGpu<T>(concat3Grad, 1, concat2Channels, concat2Channels + _growthChannels);

        // Backward through activation and Conv3
        var conv3PreGrad = gpuEngine.LeakyReluBackwardGpu<T>(x3Grad, _gpuConv3Out, 0.2f);
        var concat2GradFromConv3 = _convLayers[2].BackwardGpu(conv3PreGrad);
        concat2Grad = gpuEngine.AddGpu<T>(concat2Grad, concat2GradFromConv3);

        // Split concat2 gradient into [concat1, x2]
        int concat1Channels = _numFeatures + _growthChannels;
        var concat1Grad = gpuEngine.SliceGpu<T>(concat2Grad, 1, 0, concat1Channels);
        var x2Grad = gpuEngine.SliceGpu<T>(concat2Grad, 1, concat1Channels, concat1Channels + _growthChannels);

        // Backward through activation and Conv2
        var conv2PreGrad = gpuEngine.LeakyReluBackwardGpu<T>(x2Grad, _gpuConv2Out, 0.2f);
        var concat1GradFromConv2 = _convLayers[1].BackwardGpu(conv2PreGrad);
        concat1Grad = gpuEngine.AddGpu<T>(concat1Grad, concat1GradFromConv2);

        // Split concat1 gradient into [x0, x1]
        var x0GradFromConcat1 = gpuEngine.SliceGpu<T>(concat1Grad, 1, 0, _numFeatures);
        var x1Grad = gpuEngine.SliceGpu<T>(concat1Grad, 1, _numFeatures, _numFeatures + _growthChannels);

        // Backward through activation and Conv1
        var conv1PreGrad = gpuEngine.LeakyReluBackwardGpu<T>(x1Grad, _gpuConv1Out, 0.2f);
        var x0GradFromConv1 = _convLayers[0].BackwardGpu(conv1PreGrad);

        // Combine all gradients flowing to x0: residual + concat split + conv1 backward
        var totalX0Grad = gpuEngine.AddGpu<T>(outputGradient, x0GradFromConcat1);
        totalX0Grad = gpuEngine.AddGpu<T>(totalX0Grad, x0GradFromConv1);

        return totalX0Grad;
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Concatenates two tensors along the channel dimension.
    /// </summary>
    private Tensor<T> ConcatenateChannels(Tensor<T> a, Tensor<T> b)
    {
        // Handle both 3D [C,H,W] and 4D [B,C,H,W] tensors
        if (a.Rank == 3)
        {
            int channelsA = a.Shape[0];
            int channelsB = b.Shape[0];
            int height = a.Shape[1];
            int width = a.Shape[2];

            var result = new Tensor<T>([channelsA + channelsB, height, width]);
            int spatialSize = height * width;

            // Copy first tensor
            for (int c = 0; c < channelsA; c++)
            {
                for (int hw = 0; hw < spatialSize; hw++)
                {
                    result.Data[c * spatialSize + hw] = a.Data[c * spatialSize + hw];
                }
            }

            // Copy second tensor
            for (int c = 0; c < channelsB; c++)
            {
                for (int hw = 0; hw < spatialSize; hw++)
                {
                    result.Data[(channelsA + c) * spatialSize + hw] = b.Data[c * spatialSize + hw];
                }
            }

            return result;
        }
        else // 4D: [B,C,H,W]
        {
            int batch = a.Shape[0];
            int channelsA = a.Shape[1];
            int channelsB = b.Shape[1];
            int height = a.Shape[2];
            int width = a.Shape[3];

            var result = new Tensor<T>([batch, channelsA + channelsB, height, width]);
            int spatialSize = height * width;
            int totalChannels = channelsA + channelsB;

            for (int n = 0; n < batch; n++)
            {
                // Copy first tensor
                for (int c = 0; c < channelsA; c++)
                {
                    for (int hw = 0; hw < spatialSize; hw++)
                    {
                        int srcIdx = n * channelsA * spatialSize + c * spatialSize + hw;
                        int dstIdx = n * totalChannels * spatialSize + c * spatialSize + hw;
                        result.Data[dstIdx] = a.Data[srcIdx];
                    }
                }

                // Copy second tensor
                for (int c = 0; c < channelsB; c++)
                {
                    for (int hw = 0; hw < spatialSize; hw++)
                    {
                        int srcIdx = n * channelsB * spatialSize + c * spatialSize + hw;
                        int dstIdx = n * totalChannels * spatialSize + (channelsA + c) * spatialSize + hw;
                        result.Data[dstIdx] = b.Data[srcIdx];
                    }
                }
            }

            return result;
        }
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
                    first.Data[c * spatialSize + hw] = grad.Data[c * spatialSize + hw];
                }
            }

            for (int c = 0; c < secondChannels; c++)
            {
                for (int hw = 0; hw < spatialSize; hw++)
                {
                    second.Data[c * spatialSize + hw] = grad.Data[(firstChannels + c) * spatialSize + hw];
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
                        first.Data[dstIdx] = grad.Data[srcIdx];
                    }
                }

                for (int c = 0; c < secondChannels; c++)
                {
                    for (int hw = 0; hw < spatialSize; hw++)
                    {
                        int srcIdx = n * totalChannels * spatialSize + (firstChannels + c) * spatialSize + hw;
                        int dstIdx = n * secondChannels * spatialSize + c * spatialSize + hw;
                        second.Data[dstIdx] = grad.Data[srcIdx];
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
        var output = new Tensor<T>(input.Shape);
        for (int i = 0; i < input.Length; i++)
        {
            output.Data[i] = _activation.Activate(input.Data[i]);
        }
        return output;
    }

    /// <summary>
    /// Backward pass through LeakyReLU activation.
    /// </summary>
    private Tensor<T> BackwardActivation(Tensor<T> activationOutput, Tensor<T> gradient)
    {
        var output = new Tensor<T>(gradient.Shape);
        for (int i = 0; i < gradient.Length; i++)
        {
            output.Data[i] = NumOps.Multiply(
                gradient.Data[i],
                _activation.Derivative(activationOutput.Data[i]));
        }
        return output;
    }

    /// <summary>
    /// Adds residual with scaling: output = a * scale + b.
    /// </summary>
    private Tensor<T> AddResidual(Tensor<T> a, Tensor<T> b, double scale)
    {
        var output = new Tensor<T>(a.Shape);
        var scaleT = NumOps.FromDouble(scale);
        for (int i = 0; i < a.Length; i++)
        {
            output.Data[i] = NumOps.Add(
                NumOps.Multiply(a.Data[i], scaleT),
                b.Data[i]);
        }
        return output;
    }

    /// <summary>
    /// Scales a tensor by a factor.
    /// </summary>
    private Tensor<T> ScaleGradient(Tensor<T> gradient, double scale)
    {
        var output = new Tensor<T>(gradient.Shape);
        var scaleT = NumOps.FromDouble(scale);
        for (int i = 0; i < gradient.Length; i++)
        {
            output.Data[i] = NumOps.Multiply(gradient.Data[i], scaleT);
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

    /// <inheritdoc />
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes is null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (InputShape is null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured.");

        // Create symbolic input node with batch dimension [batch, channels, height, width]
        var symbolicInput = new Tensor<T>(new int[] { 1 }.Concat(InputShape).ToArray());
        var inputNode = TensorOperations<T>.Variable(symbolicInput, "input");
        inputNodes.Add(inputNode);

        return BuildComputationGraph(inputNode, "");
    }

    /// <inheritdoc />
    public ComputationNode<T> BuildComputationGraph(ComputationNode<T> inputNode, string namePrefix)
    {
        // x0 = input
        var x0 = inputNode;

        // Conv1: x1 = LeakyReLU(Conv(x0))
        var conv1Output = BuildConvNode(_convLayers[0], x0, $"{namePrefix}conv1_");
        var x1 = TensorOperations<T>.LeakyReLU(conv1Output, 0.2);

        // Conv2: x2 = LeakyReLU(Conv(concat(x0, x1)))
        var concat1 = TensorOperations<T>.Concat([x0, x1], axis: 1);
        var conv2Output = BuildConvNode(_convLayers[1], concat1, $"{namePrefix}conv2_");
        var x2 = TensorOperations<T>.LeakyReLU(conv2Output, 0.2);

        // Conv3: x3 = LeakyReLU(Conv(concat(x0, x1, x2)))
        var concat2 = TensorOperations<T>.Concat([concat1, x2], axis: 1);
        var conv3Output = BuildConvNode(_convLayers[2], concat2, $"{namePrefix}conv3_");
        var x3 = TensorOperations<T>.LeakyReLU(conv3Output, 0.2);

        // Conv4: x4 = LeakyReLU(Conv(concat(x0, x1, x2, x3)))
        var concat3 = TensorOperations<T>.Concat([concat2, x3], axis: 1);
        var conv4Output = BuildConvNode(_convLayers[3], concat3, $"{namePrefix}conv4_");
        var x4 = TensorOperations<T>.LeakyReLU(conv4Output, 0.2);

        // Conv5: x5 = Conv(concat(x0, x1, x2, x3, x4)) - NO activation
        var concat4 = TensorOperations<T>.Concat([concat3, x4], axis: 1);
        var x5 = BuildConvNode(_convLayers[4], concat4, $"{namePrefix}conv5_");

        // Local residual: output = x5 * residualScale + x0
        var scaledX5 = ScaleNode(x5, _residualScale, $"{namePrefix}residual_scale");
        return TensorOperations<T>.Add(scaledX5, x0);
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
        var scaleTensor = new Tensor<T>(node.Value.Shape);
        for (int i = 0; i < scaleTensor.Length; i++)
        {
            scaleTensor.Data[i] = scaleValue;
        }

        var scaleNode = TensorOperations<T>.Constant(scaleTensor, name);
        return TensorOperations<T>.ElementwiseMultiply(node, scaleNode);
    }

    #endregion
}
