using AiDotNet.ActivationFunctions;
using AiDotNet.Autodiff;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Implements a Transition Layer from the DenseNet architecture.
/// </summary>
/// <remarks>
/// <para>
/// A Transition Layer is placed between Dense Blocks to reduce the number of feature maps
/// and spatial dimensions. It performs:
/// 1. Batch Normalization
/// 2. 1x1 Convolution (channel reduction by compression factor)
/// 3. 2x2 Average Pooling with stride 2 (spatial dimension halving)
/// </para>
/// <para>
/// Architecture:
/// <code>
/// Input (C channels, H×W)
///   ↓
/// BN → ReLU → Conv1x1 (C × theta channels)
///   ↓
/// AvgPool 2×2, stride 2
///   ↓
/// Output (C × theta channels, H/2 × W/2)
/// </code>
/// Where theta is the compression factor (default: 0.5).
/// </para>
/// <para>
/// <b>For Beginners:</b> The transition layer acts as a "bottleneck" between dense blocks.
///
/// Its purposes:
/// - Reduce feature map channels (compression): Dense blocks produce many channels
/// - Reduce spatial size (pooling): Helps control computational cost
/// - Improve model compactness without sacrificing accuracy
///
/// The compression factor (theta) controls how much to reduce channels.
/// theta=0.5 means halving the channels at each transition.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class TransitionLayer<T> : LayerBase<T>, IChainableComputationGraph<T>
{
    private readonly BatchNormalizationLayer<T> _bn;
    private readonly ConvolutionalLayer<T> _conv;
    private readonly AveragePoolingLayer<T> _pool;
    private readonly IActivationFunction<T> _relu;

    private Tensor<T>? _lastInput;
    private Tensor<T>? _bnOut;
    private Tensor<T>? _reluOut;
    private Tensor<T>? _convOut;

    // GPU cached tensors for backward pass
    private IGpuTensor<T>? _gpuBnOut;
    private IGpuTensor<T>? _gpuConvOut;
    private bool _gpuAdded3DBatch;

    /// <summary>
    /// Stores the original input shape for any-rank tensor support.
    /// </summary>
    private int[]? _originalInputShape;

    /// <summary>
    /// Gets the number of output channels.
    /// </summary>
    public int OutputChannels { get; }

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets a value indicating whether this layer supports GPU execution.
    /// All sub-layers (BatchNorm, Conv, AvgPool) support GPU.
    /// </summary>
    protected override bool SupportsGpuExecution => true;

    /// <summary>
    /// Initializes a new instance of the <see cref="TransitionLayer{T}"/> class.
    /// </summary>
    /// <param name="inputChannels">The number of input channels.</param>
    /// <param name="inputHeight">The input feature map height.</param>
    /// <param name="inputWidth">The input feature map width.</param>
    /// <param name="compressionFactor">The channel compression factor (default: 0.5).</param>
    public TransitionLayer(
        int inputChannels,
        int inputHeight,
        int inputWidth,
        double compressionFactor = 0.5)
        : base(
            inputShape: [inputChannels, inputHeight, inputWidth],
            outputShape: [(int)(inputChannels * compressionFactor), inputHeight / 2, inputWidth / 2])
    {
        OutputChannels = (int)(inputChannels * compressionFactor);
        _relu = new ReLUActivation<T>();

        _bn = new BatchNormalizationLayer<T>(inputChannels);

        _conv = new ConvolutionalLayer<T>(
            inputDepth: inputChannels,
            outputDepth: OutputChannels,
            kernelSize: 1,
            inputHeight: inputHeight,
            inputWidth: inputWidth,
            stride: 1,
            padding: 0,
            activationFunction: new IdentityActivation<T>());

        // After 1x1 conv, dimensions are same
        // Then 2x2 avg pool with stride 2 halves dimensions
        _pool = new AveragePoolingLayer<T>(
            inputShape: [OutputChannels, inputHeight, inputWidth],
            poolSize: 2,
            strides: 2);
    }

    /// <summary>
    /// Performs the forward pass of the Transition Layer.
    /// </summary>
    /// <param name="input">The input tensor [B, C, H, W] or [C, H, W].</param>
    /// <returns>The output tensor with reduced channels and spatial dimensions.</returns>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Store original shape for any-rank tensor support
        _originalInputShape = input.Shape;
        int rank = input.Shape.Length;

        // Handle any-rank tensor: collapse leading dims for rank > 4
        Tensor<T> processInput;
        int flatBatch = 1;

        if (rank == 3)
        {
            // Standard 3D: [C, H, W]
            processInput = input;
        }
        else if (rank == 4)
        {
            // Standard 4D: [B, C, H, W]
            processInput = input;
        }
        else if (rank > 4)
        {
            // Higher-rank: collapse leading dims into batch
            for (int d = 0; d < rank - 3; d++)
                flatBatch *= input.Shape[d];
            int channels = input.Shape[rank - 3];
            int height = input.Shape[rank - 2];
            int width = input.Shape[rank - 1];
            processInput = input.Reshape(new[] { flatBatch, channels, height, width });
        }
        else
        {
            // Rank 2 or less - not typical for transition layer
            throw new ArgumentException($"TransitionLayer requires at least 3D input, got {rank}D");
        }

        _lastInput = processInput;

        // BN → ReLU → Conv1x1 → AvgPool
        _bnOut = _bn.Forward(processInput);
        _reluOut = _relu.Activate(_bnOut);
        _convOut = _conv.Forward(_reluOut);

        // Handle batched input (4D) - use Engine.AvgPool2D directly
        // AveragePoolingLayer expects 3D input, so we handle 4D separately
        // [B, C, H, W] uses Engine directly, [C, H, W] uses the AveragePoolingLayer
        Tensor<T> output = _convOut.Shape.Length == 4
            ? Engine.AvgPool2D(_convOut, poolSize: 2, stride: 2, padding: 0)
            : _pool.Forward(_convOut);

        // Restore original batch dimensions for any-rank support
        if (_originalInputShape != null && _originalInputShape.Length > 4)
        {
            // Output shape: [...leadingDims, outChannels, outH, outW]
            int outChannels = output.Shape[1];
            int outH = output.Shape[2];
            int outW = output.Shape[3];
            int[] newShape = new int[_originalInputShape.Length];
            for (int d = 0; d < _originalInputShape.Length - 3; d++)
                newShape[d] = _originalInputShape[d];
            newShape[_originalInputShape.Length - 3] = outChannels;
            newShape[_originalInputShape.Length - 2] = outH;
            newShape[_originalInputShape.Length - 1] = outW;
            output = output.Reshape(newShape);
        }

        return output;
    }

    /// <summary>
    /// Performs the forward pass using GPU-resident tensors.
    /// </summary>
    /// <param name="inputs">The GPU-resident input tensors.</param>
    /// <returns>A GPU-resident output tensor.</returns>
    /// <remarks>
    /// <para>
    /// Chains GPU operations through sub-layers: BN → ReLU → Conv1x1 → AvgPool.
    /// All intermediate results stay GPU-resident.
    /// </para>
    /// </remarks>
    public override IGpuTensor<T> ForwardGpu(params IGpuTensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));

        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("ForwardGpu requires DirectGpuTensorEngine.");

        var input = inputs[0];
        var shape = input.Shape;

        // Ensure 4D shape [B, C, H, W]
        IGpuTensor<T> processInput;
        bool added3DBatch = false;

        if (shape.Length == 4)
        {
            processInput = input;
        }
        else if (shape.Length == 3)
        {
            processInput = gpuEngine.ReshapeGpu(input, new[] { 1, shape[0], shape[1], shape[2] });
            added3DBatch = true;
        }
        else
        {
            throw new ArgumentException($"TransitionLayer requires 3D or 4D input, got {shape.Length}D");
        }

        // Cache for backward pass
        if (IsTrainingMode)
        {
            _lastInput = processInput.ToTensor();
            _originalInputShape = shape;
        }

        // Chain GPU operations: BN → ReLU → Conv1x1 → AvgPool
        var bnOutput = _bn.ForwardGpu(processInput);
        var reluOutput = gpuEngine.ActivationGpu(bnOutput, FusedActivationType.ReLU);
        var convOutput = _conv.ForwardGpu(reluOutput);
        var poolOutput = _pool.ForwardGpu(convOutput);

        // Cache intermediates for backward during training
        if (IsTrainingMode)
        {
            _gpuBnOut = bnOutput;
            _gpuConvOut = convOutput;
            _gpuAdded3DBatch = added3DBatch;
            _bnOut = bnOutput.ToTensor();
            _reluOut = reluOutput.ToTensor();
            _convOut = convOutput.ToTensor();
        }

        // Remove batch dimension if we added it
        if (added3DBatch)
        {
            var outShape = poolOutput.Shape;
            return gpuEngine.ReshapeGpu(poolOutput, new[] { outShape[1], outShape[2], outShape[3] });
        }

        return poolOutput;
    }

    /// <summary>
    /// Computes the gradient of the loss with respect to the input on the GPU.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    public override IGpuTensor<T> BackwardGpu(IGpuTensor<T> outputGradient)
    {
        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("BackwardGpu requires DirectGpuTensorEngine.");

        if (_gpuBnOut == null || _gpuConvOut == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // Handle 3D -> 4D shape conversion if we did it in forward
        IGpuTensor<T> grad = outputGradient;
        if (_gpuAdded3DBatch && outputGradient.Shape.Length == 3)
        {
            grad = gpuEngine.ReshapeGpu(outputGradient, new[] { 1, outputGradient.Shape[0], outputGradient.Shape[1], outputGradient.Shape[2] });
        }

        // Backward through pool
        var poolType = _pool.GetType();
        var poolBackwardGpu = poolType.GetMethod("BackwardGpu", new[] { typeof(IGpuTensor<T>) });
        if (poolBackwardGpu != null)
        {
            grad = (IGpuTensor<T>)poolBackwardGpu.Invoke(_pool, new object[] { grad })!;
        }
        else
        {
            var cpuGrad = grad.ToTensor();
            var cpuResult = _convOut != null ? AvgPool2DBackward(cpuGrad, _gpuConvOut.Shape) : _pool.Backward(cpuGrad);
            grad = gpuEngine.UploadToGpu<T>(cpuResult, GpuTensorRole.Gradient);
        }

        // Backward through conv
        var convType = _conv.GetType();
        var convBackwardGpu = convType.GetMethod("BackwardGpu", new[] { typeof(IGpuTensor<T>) });
        if (convBackwardGpu != null)
        {
            grad = (IGpuTensor<T>)convBackwardGpu.Invoke(_conv, new object[] { grad })!;
        }
        else
        {
            var cpuGrad = grad.ToTensor();
            var cpuResult = _conv.Backward(cpuGrad);
            grad = gpuEngine.UploadToGpu<T>(cpuResult, GpuTensorRole.Gradient);
        }

        // Backward through ReLU - uses BN output as input
        grad = gpuEngine.ReluBackwardGpu<T>(grad, _gpuBnOut);

        // Backward through BN
        var bnType = _bn.GetType();
        var bnBackwardGpu = bnType.GetMethod("BackwardGpu", new[] { typeof(IGpuTensor<T>) });
        if (bnBackwardGpu != null)
        {
            grad = (IGpuTensor<T>)bnBackwardGpu.Invoke(_bn, new object[] { grad })!;
        }
        else
        {
            var cpuGrad = grad.ToTensor();
            var cpuResult = _bn.Backward(cpuGrad);
            grad = gpuEngine.UploadToGpu<T>(cpuResult, GpuTensorRole.Gradient);
        }

        // Remove batch dimension if we added it
        if (_gpuAdded3DBatch && grad.Shape.Length == 4)
        {
            grad = gpuEngine.ReshapeGpu(grad, new[] { grad.Shape[1], grad.Shape[2], grad.Shape[3] });
        }

        return grad;
    }

    /// <summary>
    /// Performs the backward pass of the Transition Layer.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the output.</param>
    /// <returns>The gradient of the loss with respect to the input.</returns>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput is null || _bnOut is null || _reluOut is null || _convOut is null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // Backward through pool - handle 4D inputs
        // 4D: manual backward, 3D: use pooling layer
        Tensor<T> grad = outputGradient.Shape.Length == 4
            ? AvgPool2DBackward(outputGradient, _convOut.Shape)
            : _pool.Backward(outputGradient);

        // Backward through conv
        grad = _conv.Backward(grad);

        // Backward through ReLU
        grad = ApplyReluDerivative(_bnOut, grad);

        // Backward through BN
        grad = _bn.Backward(grad);

        return grad;
    }

    /// <summary>
    /// Backward pass for 4D average pooling.
    /// </summary>
    private Tensor<T> AvgPool2DBackward(Tensor<T> outputGrad, int[] inputShape)
    {
        int batch = outputGrad.Shape[0];
        int channels = outputGrad.Shape[1];
        int outH = outputGrad.Shape[2];
        int outW = outputGrad.Shape[3];
        int inH = inputShape[2];
        int inW = inputShape[3];
        int poolSize = 2;
        int stride = 2;

        var inputGrad = new Tensor<T>(inputShape);
        var divisor = NumOps.FromDouble((double)poolSize * (double)poolSize);

        for (int n = 0; n < batch; n++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int oh = 0; oh < outH; oh++)
                {
                    for (int ow = 0; ow < outW; ow++)
                    {
                        int outIdx = n * (channels * outH * outW) + c * (outH * outW) + oh * outW + ow;
                        var gradVal = NumOps.Divide(outputGrad.Data.Span[outIdx], divisor);

                        int hStart = oh * stride;
                        int wStart = ow * stride;

                        for (int kh = 0; kh < poolSize; kh++)
                        {
                            for (int kw = 0; kw < poolSize; kw++)
                            {
                                int ih = hStart + kh;
                                int iw = wStart + kw;
                                if (ih < inH && iw < inW)
                                {
                                    int inIdx = n * (channels * inH * inW) + c * (inH * inW) + ih * inW + iw;
                                    inputGrad.Data.Span[inIdx] = NumOps.Add(inputGrad.Data.Span[inIdx], gradVal);
                                }
                            }
                        }
                    }
                }
            }
        }

        return inputGrad;
    }

    private Tensor<T> ApplyReluDerivative(Tensor<T> input, Tensor<T> grad)
    {
        var result = new T[grad.Data.Length];
        for (int i = 0; i < grad.Data.Length; i++)
        {
            result[i] = NumOps.GreaterThan(input.Data.Span[i], NumOps.Zero)
                ? grad.Data.Span[i]
                : NumOps.Zero;
        }
        return new Tensor<T>(grad.Shape, new Vector<T>(result));
    }

    /// <summary>
    /// Updates the parameters of all sub-layers.
    /// </summary>
    /// <param name="learningRate">The learning rate for parameter updates.</param>
    public override void UpdateParameters(T learningRate)
    {
        _bn.UpdateParameters(learningRate);
        _conv.UpdateParameters(learningRate);
        // Pool has no parameters
    }

    /// <summary>
    /// Gets all trainable parameters from the layer.
    /// </summary>
    public override Vector<T> GetParameters()
    {
        var parameters = new List<T>();
        parameters.AddRange(_bn.GetParameters().ToArray());
        parameters.AddRange(_conv.GetParameters().ToArray());
        return new Vector<T>(parameters.ToArray());
    }

    /// <summary>
    /// Sets all trainable parameters from the given parameter vector.
    /// </summary>
    /// <param name="parameters">The parameter vector containing all layer parameters.</param>
    public override void SetParameters(Vector<T> parameters)
    {
        int offset = 0;

        int count = _bn.GetParameters().Length;
        _bn.SetParameters(parameters.SubVector(offset, count));
        offset += count;

        count = _conv.GetParameters().Length;
        _conv.SetParameters(parameters.SubVector(offset, count));
    }

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["InputChannels"] = InputShape[0].ToString();
        metadata["OutputChannels"] = OutputChannels.ToString();
        return metadata;
    }

    /// <summary>
    /// Resets the internal state of the layer.
    /// </summary>
    public override void ResetState()
    {
        _lastInput = null;
        _bnOut = null;
        _reluOut = null;
        _convOut = null;
        _gpuBnOut = null;
        _gpuConvOut = null;
        _gpuAdded3DBatch = false;

        _bn.ResetState();
        _conv.ResetState();
        _pool.ResetState();
    }

    /// <summary>
    /// Gets whether this layer supports JIT compilation.
    /// </summary>
    public override bool SupportsJitCompilation
    {
        get
        {
            return _bn.SupportsJitCompilation &&
                   _conv.SupportsJitCompilation &&
                   _pool.SupportsJitCompilation;
        }
    }

    /// <inheritdoc />
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
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
        var inputNode = TensorOperations<T>.Variable(symbolicInput, "input");
        inputNodes.Add(inputNode);

        return BuildComputationGraph(inputNode, "");
    }

    /// <inheritdoc />
    public ComputationNode<T> BuildComputationGraph(ComputationNode<T> inputNode, string namePrefix)
    {
        // BN
        var bnNode = TensorOperations<T>.BatchNorm(
            inputNode,
            gamma: TensorOperations<T>.Constant(_bn.GetGamma(), $"{namePrefix}bn_gamma"),
            beta: TensorOperations<T>.Constant(_bn.GetBeta(), $"{namePrefix}bn_beta"),
            runningMean: _bn.GetRunningMean(),
            runningVar: _bn.GetRunningVariance(),
            training: false,
            epsilon: NumOps.ToDouble(_bn.GetEpsilon()));

        // ReLU
        var reluNode = TensorOperations<T>.ReLU(bnNode);

        // Conv 1x1
        var convBiases = _conv.GetBiases();
        var convNode = TensorOperations<T>.Conv2D(
            reluNode,
            TensorOperations<T>.Constant(_conv.GetFilters(), $"{namePrefix}conv_kernel"),
            convBiases is not null ? TensorOperations<T>.Constant(convBiases, $"{namePrefix}conv_bias") : null,
            stride: new int[] { _conv.Stride, _conv.Stride },
            padding: new int[] { _conv.Padding, _conv.Padding });

        // Average Pooling 2x2, stride 2
        var poolNode = TensorOperations<T>.AvgPool2D(convNode, poolSize: new int[] { 2, 2 }, strides: new int[] { 2, 2 });

        return poolNode;
    }
}
