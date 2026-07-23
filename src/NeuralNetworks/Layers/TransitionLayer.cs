using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
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
[LayerCategory(LayerCategory.Convolution)]
[LayerCategory(LayerCategory.Pooling)]
[LayerTask(LayerTask.DownSampling)]
[LayerProperty(NormalizesInput = true, IsTrainable = true, ChangesShape = true, ExpectedInputRank = 3, TestInputShape = "4, 8, 8", TestConstructorArgs = "0.5")]
public class TransitionLayer<T> : LayerBase<T>, ILayerSerializationExtras<T>
{
    private readonly BatchNormalizationLayer<T> _bn;

    // Buffered parameters from a Deserialize → SetParameters that arrives
    // before the first Forward. Both _bn and _conv are still unresolved
    // at that point so we can't slice between them; stash the whole
    // vector and replay inside OnFirstForward.
    private Vector<T>? _pendingParameters;
    // Lazy ctor leaves _conv = null until OnFirstForward resolves
    // OutputChannels (= inputChannels × compressionFactor) and allocates
    // the 1×1 projection. Declared nullable so the compiler enforces
    // null-checks on every reference and the existing null-guards stop
    // looking like dead code.
    private ConvolutionalLayer<T>? _conv;
    private readonly AveragePoolingLayer<T> _pool;
    private readonly IActivationFunction<T> _relu;

    private Tensor<T>? _lastInput;
    private Tensor<T>? _bnOut;
    private Tensor<T>? _reluOut;
    private Tensor<T>? _convOut;

    // GPU cached tensors for backward pass
    private Tensor<T>? _gpuBnOut;
    private Tensor<T>? _gpuConvOut;
    private bool _gpuAdded3DBatch;

    /// <summary>
    /// Stores the original input shape for any-rank tensor support.
    /// </summary>
    private int[]? _originalInputShape;

    /// <summary>
    /// Gets the number of output channels. Resolved in
    /// <see cref="OnFirstForward"/> from <c>inputChannels ×
    /// compressionFactor</c>; before that point it returns the
    /// sentinel default (0).
    /// </summary>
    public int OutputChannels { get; private set; }

    /// <summary>
    /// Sum of trainable parameters across BN and the 1×1 projection.
    /// <c>_conv</c> stays null until <see cref="OnFirstForward"/> resolves
    /// the input channel count — return BN's count alone in that interim
    /// state.
    /// </summary>
    public override long ParameterCount =>
        _bn.ParameterCount + (_conv?.ParameterCount ?? 0L);

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    public override bool SupportsTraining => true;

    public override Vector<T> GetParameterGradients()
    {
        // _conv is null until OnFirstForward runs; return BN's gradients
        // alone in that interim state to avoid NullReferenceException.
        if (_conv is null) return _bn.GetParameterGradients();
        return Vector<T>.Concatenate(_bn.GetParameterGradients(), _conv.GetParameterGradients());
    }

    public override void ClearGradients()
    {
        base.ClearGradients();
        _bn.ClearGradients();
        _conv?.ClearGradients();
    }

    /// <summary>
    /// Gets a value indicating whether this layer supports GPU execution.
    /// All sub-layers (BatchNorm, Conv, AvgPool) support GPU.
    /// </summary>
    protected override bool SupportsGpuExecution => true;

    /// <summary>
    /// Lazy ctor — input depth/height/width come from the first
    /// <see cref="Forward"/> call (<see cref="OnFirstForward"/>); only
    /// the channel-compression factor is required at construction. The
    /// 1×1 projection's <c>outputDepth</c> is allocated against the
    /// resolved channel count in <see cref="OnFirstForward"/>.
    /// </summary>
    public TransitionLayer(double compressionFactor = 0.5)
        : base(inputShape: [-1, -1, -1], outputShape: [-1, -1, -1])
    {
        if (compressionFactor <= 0 || compressionFactor > 1)
            throw new ArgumentOutOfRangeException(nameof(compressionFactor),
                "compressionFactor must be in (0, 1].");

        _compressionFactor = compressionFactor;
        _relu = new ReLUActivation<T>();
        _bn = new BatchNormalizationLayer<T>();
        // _conv stays null until OnFirstForward observes the input channel
        // count (compiler tracks this via the field's nullability).

        // After 1×1 conv, spatial dims are same; 2×2 avg pool with
        // stride 2 halves them.
        _pool = new AveragePoolingLayer<T>(poolSize: 2, strides: 2);

        RegisterSubLayer(_bn);
        RegisterSubLayer(_pool);
    }

    private readonly double _compressionFactor;

    /// <inheritdoc/>
    /// <remarks>
    /// Resolves input channel count from <c>input.Shape</c>, allocates
    /// the 1×1 conv against the resolved <c>OutputChannels =
    /// inputChannels × compressionFactor</c>, then propagates shape to
    /// each sub-layer so <see cref="LayerBase{T}.ParameterCount"/>
    /// reflects the real weight count before any sub-layer's first
    /// Forward fires.
    /// </remarks>
    protected override void OnFirstForward(Tensor<T> input)
    {
        var s = input._shape;
        int inputChannels, inputHeight, inputWidth;
        if (s.Length == 3) { inputChannels = s[0]; inputHeight = s[1]; inputWidth = s[2]; }
        else if (s.Length == 4) { inputChannels = s[1]; inputHeight = s[2]; inputWidth = s[3]; }
        else
            throw new ArgumentException(
                $"TransitionLayer requires rank-3 [C,H,W] or rank-4 [B,C,H,W] input; got rank {s.Length}.",
                nameof(input));

        OutputChannels = (int)(inputChannels * _compressionFactor);

        // Allocate the 1×1 projection now that input channel count is known.
        _conv = new ConvolutionalLayer<T>(
            outputDepth: OutputChannels,
            kernelSize: 1,
            stride: 1,
            padding: 0,
            activationFunction: new IdentityActivation<T>());
        RegisterSubLayer(_conv);
        // Propagate parent's current training mode to the freshly-allocated
        // sub-layer; otherwise it stays in default training mode even after
        // the parent was toggled to eval before its first Forward.
        _conv.SetTrainingMode(IsTrainingMode);

        // Drive sub-layer shape resolution so ParameterCount predicts
        // correctly before each one's own Forward fires.
        // Use ResolveFromShape so weights are allocated up front — needed
        // for ApplyParameters below to slice correctly with the buffered
        // Deserialize parameter vector.
        _bn.ResolveFromShape(new[] { 1, inputChannels, inputHeight, inputWidth });
        _conv.ResolveFromShape(new[] { inputChannels, inputHeight, inputWidth });
        _pool.ResolveShapesOnly(new[] { OutputChannels, inputHeight, inputWidth });

        ResolveShapes(
            new[] { inputChannels, inputHeight, inputWidth },
            new[] { OutputChannels, inputHeight / 2, inputWidth / 2 });

        // Replay parameters that arrived before _bn / _conv shapes were
        // resolved. With both now allocated, slicing matches the
        // GetParameters() ordering above.
        if (_pendingParameters is not null)
        {
            var pending = _pendingParameters;
            _pendingParameters = null;
            int bnCount = _bn.GetParameters().Length;
            _bn.SetParameters(pending.SubVector(0, bnCount));
            int convCount = _conv.GetParameters().Length;
            _conv.SetParameters(pending.SubVector(bnCount, convCount));
        }
    }

    /// <summary>
    /// Performs the forward pass of the Transition Layer.
    /// </summary>
    /// <param name="input">The input tensor [B, C, H, W] or [C, H, W].</param>
    /// <returns>The output tensor with reduced channels and spatial dimensions.</returns>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Lazy ctor leaves _conv null until OnFirstForward observes the
        // input channel count and allocates it from compressionFactor.
        if (!IsShapeResolved) OnFirstForward(input);
        var conv = _conv ?? throw new InvalidOperationException("OnFirstForward did not allocate _conv.");

        // Store original shape for any-rank tensor support
        _originalInputShape = input._shape;
        int rank = input.Shape.Length;

        // Handle any-rank tensor: collapse leading dims for rank > 4
        Tensor<T> processInput;
        int flatBatch = 1;

        if (rank == 3)
        {
            // Standard 3D: [C, H, W] → add batch dim [1, C, H, W] for BN/Conv
            processInput = Engine.Reshape(input, [1, input.Shape[0], input.Shape[1], input.Shape[2]]);
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
            processInput = Engine.Reshape(input, new[] { flatBatch, channels, height, width });
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
        _convOut = conv.Forward(_reluOut);

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
            output = Engine.Reshape(output, newShape);
        }

        // Remove batch dim if we added it for 3D input
        if (rank == 3 && output.Shape.Length == 4 && output.Shape[0] == 1)
            output = Engine.Reshape(output, [output.Shape[1], output.Shape[2], output.Shape[3]]);

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
    public override Tensor<T> ForwardGpu(params Tensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));

        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("ForwardGpu requires DirectGpuTensorEngine.");

        var input = inputs[0];

        // Lazy gate — _conv is null until OnFirstForward observes the input
        // channel count. The CPU Forward path has the same gate; GPU path was
        // missing it (PR #1259 review).
        if (!IsShapeResolved) OnFirstForward(input);
        var conv = _conv ?? throw new InvalidOperationException("OnFirstForward did not allocate _conv.");

        var shape = input._shape;

        // Support any rank >= 3: last 3 dims are [C, H, W], earlier dims are batch-like
        if (shape.Length < 3)
            throw new ArgumentException($"TransitionLayer requires at least 3D tensor [C, H, W]. Got rank {shape.Length}.");

        Tensor<T> processInput;
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
            // Higher rank: flatten leading dimensions into batch
            int flatBatch = 1;
            for (int d = 0; d < shape.Length - 3; d++)
                flatBatch *= shape[d];
            processInput = gpuEngine.ReshapeGpu(input, new[] { flatBatch, shape[shape.Length - 3], shape[shape.Length - 2], shape[shape.Length - 1] });
        }

        // Cache for backward pass
        if (IsTrainingMode)
        {
            _lastInput = processInput;
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
            _bnOut = bnOutput;
            _reluOut = reluOutput;
            _convOut = convOutput;
        }

        // Restore original tensor rank
        if (shape.Length > 4)
        {
            var outShape = poolOutput._shape;
            var restoreShape = new int[shape.Length];
            for (int d = 0; d < shape.Length - 3; d++)
                restoreShape[d] = shape[d];
            restoreShape[shape.Length - 3] = outShape[1];
            restoreShape[shape.Length - 2] = outShape[2];
            restoreShape[shape.Length - 1] = outShape[3];
            return gpuEngine.ReshapeGpu(poolOutput, restoreShape);
        }
        if (added3DBatch)
        {
            var outShape = poolOutput._shape;
            return gpuEngine.ReshapeGpu(poolOutput, new[] { outShape[1], outShape[2], outShape[3] });
        }

        return poolOutput;
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

        var inputGrad = TensorAllocator.Rent<T>(inputShape);
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
        return new Tensor<T>(grad._shape, new Vector<T>(result));
    }

    /// <summary>
    /// Updates the parameters of all sub-layers.
    /// </summary>
    /// <param name="learningRate">The learning rate for parameter updates.</param>
    public override void UpdateParameters(T learningRate)
    {
        _bn.UpdateParameters(learningRate);
        _conv?.UpdateParameters(learningRate);
        // Pool has no parameters
    }

    /// <summary>
    /// Gets all trainable parameters from the layer. <c>_conv</c> contributes
    /// nothing until <see cref="OnFirstForward"/> resolves the input channel
    /// count and allocates the 1×1 projection.
    /// </summary>
    public override Vector<T> GetParameters()
    {
        var parameters = new List<T>();
        parameters.AddRange(_bn.GetParameters().ToArray());
        if (_conv is not null)
            parameters.AddRange(_conv.GetParameters().ToArray());
        return new Vector<T>(parameters.ToArray());
    }

    /// <summary>
    /// Sets all trainable parameters from the given parameter vector.
    /// </summary>
    /// <param name="parameters">The parameter vector containing all layer parameters.</param>
    public override void SetParameters(Vector<T> parameters)
    {
        // Pre-Forward: both _bn and _conv have unresolved shapes so
        // GetParameters().Length on either returns 0 — we can't slice
        // between them. Buffer the full vector and replay from
        // OnFirstForward once shapes are resolved.
        if (!IsShapeResolved)
        {
            _pendingParameters = parameters;
            return;
        }

        int offset = 0;
        int count = _bn.GetParameters().Length;
        _bn.SetParameters(parameters.SubVector(offset, count));
        offset += count;

        if (_conv is not null)
        {
            count = _conv.GetParameters().Length;
            _conv.SetParameters(parameters.SubVector(offset, count));
        }
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
        _conv?.ResetState();
        _pool.ResetState();
    }

    // --- ILayerSerializationExtras: propagate the inner BN's running stats.
    // Without this, DenseNet's serialize/deserialize round-trip loses each
    // transition's BN running mean/variance after training.

    int ILayerSerializationExtras<T>.ExtraParameterCount
        => _bn is ILayerSerializationExtras<T> e ? e.ExtraParameterCount : 0;

    Vector<T> ILayerSerializationExtras<T>.GetExtraParameters()
    {
        if (_bn is ILayerSerializationExtras<T> e)
            return e.GetExtraParameters();
        return new Vector<T>(0);
    }

    void ILayerSerializationExtras<T>.SetExtraParameters(Vector<T> extraParameters)
    {
        if (_bn is ILayerSerializationExtras<T> e)
            e.SetExtraParameters(extraParameters);
    }
}
