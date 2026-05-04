using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// A single layer within a DenseBlock: BN-ReLU-Conv1x1-BN-ReLU-Conv3x3.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[LayerCategory(LayerCategory.Convolution)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(NormalizesInput = true, IsTrainable = true, ChangesShape = true, ExpectedInputRank = 4, TestInputShape = "2, 4, 4, 4", TestConstructorArgs = "4")]
internal partial class DenseBlockLayer<T> : LayerBase<T>, ILayerSerializationExtras<T>
{
    private readonly BatchNormalizationLayer<T> _bn1;
    private readonly ConvolutionalLayer<T> _conv1x1;
    private readonly BatchNormalizationLayer<T> _bn2;
    private readonly ConvolutionalLayer<T> _conv3x3;
    private readonly IActivationFunction<T> _relu;

    private Tensor<T>? _lastInput;
    private Tensor<T>? _bn1Out;
    private Tensor<T>? _relu1Out;
    private Tensor<T>? _conv1Out;
    private Tensor<T>? _bn2Out;
    private Tensor<T>? _relu2Out;

    // GPU cached tensors for backward pass
    private Tensor<T>? _gpuBn1Out;
    private Tensor<T>? _gpuConv1Out;
    private Tensor<T>? _gpuBn2Out;

    public override long ParameterCount => _bn1.ParameterCount + _conv1x1.ParameterCount + _bn2.ParameterCount + _conv3x3.ParameterCount;
    public override bool SupportsTraining => true;

    public override Vector<T> GetParameterGradients()
    {
        return Vector<T>.Concatenate(
            Vector<T>.Concatenate(_bn1.GetParameterGradients(), _conv1x1.GetParameterGradients()),
            Vector<T>.Concatenate(_bn2.GetParameterGradients(), _conv3x3.GetParameterGradients()));
    }

    public override void ClearGradients()
    {
        base.ClearGradients();
        _bn1.ClearGradients(); _conv1x1.ClearGradients(); _bn2.ClearGradients(); _conv3x3.ClearGradients();
    }

    /// <summary>
    /// Gets a value indicating whether this layer supports GPU execution.
    /// </summary>
    protected override bool SupportsGpuExecution => true;

    /// <summary>
    /// Lazy ctor — input depth/height/width come from the first
    /// <see cref="Forward"/> call (<see cref="OnFirstForward"/>); only
    /// growthRate/bnMomentum are required at construction. The conv
    /// kernel shapes that depend on input channels (1×1 bottleneck) are
    /// allocated against the resolved <c>_inputChannels</c> in
    /// <see cref="OnFirstForward"/>.
    /// </summary>
    public DenseBlockLayer(int growthRate, double bnMomentum = 0.1)
        : base([-1, -1, -1], [growthRate, -1, -1])
    {
        if (growthRate <= 0) throw new ArgumentOutOfRangeException(nameof(growthRate));

        _inputChannels = -1; // resolved in OnFirstForward
        _growthRate = growthRate;
        _relu = new ReLUActivation<T>();

        int bottleneckChannels = 4 * growthRate;

        _bn1 = new BatchNormalizationLayer<T>();
        _conv1x1 = new ConvolutionalLayer<T>(
            outputDepth: bottleneckChannels,
            kernelSize: 1,
            stride: 1,
            padding: 0,
            activationFunction: new IdentityActivation<T>());
        _bn2 = new BatchNormalizationLayer<T>();
        _conv3x3 = new ConvolutionalLayer<T>(
            outputDepth: growthRate,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            activationFunction: new IdentityActivation<T>());

        RegisterSubLayer(_bn1);
        RegisterSubLayer(_conv1x1);
        RegisterSubLayer(_bn2);
        RegisterSubLayer(_conv3x3);
    }

    // Non-readonly: lazy ctor leaves _inputChannels = -1 until
    // OnFirstForward resolves it from the runtime input tensor.
    private int _inputChannels;
    private readonly int _growthRate;

    /// <inheritdoc/>
    /// <remarks>
    /// Resolves H/W from input.Shape and propagates to all sub-layers
    /// via ResolveShapesOnly so ParameterCount reports the real weight
    /// count before any sub-layer's first Forward fires.
    /// </remarks>
    protected override void OnFirstForward(Tensor<T> input)
    {
        var s = input._shape;
        int channels, height, width;
        if (s.Length == 3) { channels = s[0]; height = s[1]; width = s[2]; }
        else if (s.Length == 4) { channels = s[1]; height = s[2]; width = s[3]; }
        else
            throw new ArgumentException(
                $"DenseBlockLayer requires rank-3 [C,H,W] or rank-4 [B,C,H,W] input; got rank {s.Length}.",
                nameof(input));

        _inputChannels = channels;
        int bottleneckChannels = 4 * _growthRate;
        // Use ResolveFromShape (not ResolveShapesOnly) because we may
        // need to apply buffered Deserialize params below — that requires
        // weights already allocated so GetParameters().Length is correct.
        // RNG state cost is the same: weights get allocated either now
        // or on each sub-layer's first Forward; total draws are identical.
        _bn1.ResolveFromShape(new[] { 1, _inputChannels, height, width });
        _conv1x1.ResolveFromShape(new[] { _inputChannels, height, width });
        _bn2.ResolveFromShape(new[] { 1, bottleneckChannels, height, width });
        _conv3x3.ResolveFromShape(new[] { bottleneckChannels, height, width });

        ResolveShapes(
            new[] { _inputChannels, height, width },
            new[] { _growthRate, height, width });

        // Replay parameters that arrived via Deserialize → SetParameters
        // before sub-layer shapes were resolved.
        if (_pendingParameters is not null)
        {
            var pending = _pendingParameters;
            _pendingParameters = null;
            ApplyParameters(pending);
        }
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Lazy gate — OnFirstForward resolves _inputChannels and replays
        // any Deserialize-buffered parameters before sub-layer Forwards
        // run with their (possibly stale-from-init) weights.
        if (!IsShapeResolved) OnFirstForward(input);

        _lastInput = input;

        // BN/Conv expect [N, C, H, W] format. Add batch dim if 3D [C, H, W].
        var x = input.Shape.Length == 3 ? Engine.Reshape(input, [1, input.Shape[0], input.Shape[1], input.Shape[2]]) : input;

        // BN-ReLU-Conv1x1 (DenseNet paper: pre-activation bottleneck)
        _bn1Out = _bn1.Forward(x);
        _relu1Out = _relu.Activate(_bn1Out);
        _conv1Out = _conv1x1.Forward(_relu1Out);

        // BN-ReLU-Conv3x3
        _bn2Out = _bn2.Forward(_conv1Out);
        _relu2Out = _relu.Activate(_bn2Out);
        var output = _conv3x3.Forward(_relu2Out);

        // Remove batch dim if we added it
        if (input.Shape.Length == 3 && output.Shape.Length == 4 && output.Shape[0] == 1)
            output = Engine.Reshape(output, [output.Shape[1], output.Shape[2], output.Shape[3]]);

        return output;
    }

    /// <summary>
    /// Performs the GPU-resident forward pass of the dense block layer.
    /// </summary>
    /// <param name="inputs">The GPU input tensors.</param>
    /// <returns>The GPU output tensor.</returns>
    /// <remarks>
    /// Chains GPU operations: BN1 → ReLU → Conv1x1 → BN2 → ReLU → Conv3x3.
    /// All computations stay on GPU.
    /// </remarks>
    public override Tensor<T> ForwardGpu(params Tensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));

        var gpuEngine = Engine as DirectGpuTensorEngine;
        if (gpuEngine == null)
            throw new InvalidOperationException("ForwardGpu requires a DirectGpuTensorEngine.");

        var input = inputs[0];

        // BN1 → ReLU → Conv1x1
        var bn1Output = _bn1.ForwardGpu(input);
        var relu1Output = gpuEngine.ActivationGpu(bn1Output, FusedActivationType.ReLU);
        var conv1Output = _conv1x1.ForwardGpu(relu1Output);

        // BN2 → ReLU → Conv3x3
        var bn2Output = _bn2.ForwardGpu(conv1Output);
        var relu2Output = gpuEngine.ActivationGpu(bn2Output, FusedActivationType.ReLU);
        var output = _conv3x3.ForwardGpu(relu2Output);

        // Cache tensors for backward pass (need BN outputs for ReLU backward)
        if (IsTrainingMode)
        {
            _gpuBn1Out = bn1Output;
            _gpuConv1Out = conv1Output;
            _gpuBn2Out = bn2Output;
        }

        return output;
    }

    private Tensor<T> ApplyReluDerivative(Tensor<T> input, Tensor<T> grad)
    {
        var result = new T[grad.Data.Length];
        for (int i = 0; i < grad.Data.Length; i++)
        {
            // ReLU derivative: 1 if x > 0, else 0
            result[i] = NumOps.GreaterThan(input.Data.Span[i], NumOps.Zero)
                ? grad.Data.Span[i]
                : NumOps.Zero;
        }
        return new Tensor<T>(grad._shape, new Vector<T>(result));
    }

    public override void UpdateParameters(T learningRate)
    {
        _bn1.UpdateParameters(learningRate);
        _conv1x1.UpdateParameters(learningRate);
        _bn2.UpdateParameters(learningRate);
        _conv3x3.UpdateParameters(learningRate);
    }

    public override Vector<T> GetParameters()
    {
        var parameters = new List<T>();
        parameters.AddRange(_bn1.GetParameters().ToArray());
        parameters.AddRange(_conv1x1.GetParameters().ToArray());
        parameters.AddRange(_bn2.GetParameters().ToArray());
        parameters.AddRange(_conv3x3.GetParameters().ToArray());
        return new Vector<T>(parameters.ToArray());
    }

    public override void SetParameters(Vector<T> parameters)
    {
        // Pre-Forward: sub-layers' shapes haven't been resolved, so
        // their GetParameters().Length is wrong. Buffer and replay
        // from OnFirstForward.
        if (!IsShapeResolved)
        {
            _pendingParameters = parameters;
            return;
        }

        ApplyParameters(parameters);
    }

    private Vector<T>? _pendingParameters;

    private void ApplyParameters(Vector<T> parameters)
    {
        int offset = 0;

        int count = _bn1.GetParameters().Length;
        _bn1.SetParameters(parameters.SubVector(offset, count));
        offset += count;

        count = _conv1x1.GetParameters().Length;
        _conv1x1.SetParameters(parameters.SubVector(offset, count));
        offset += count;

        count = _bn2.GetParameters().Length;
        _bn2.SetParameters(parameters.SubVector(offset, count));
        offset += count;

        count = _conv3x3.GetParameters().Length;
        _conv3x3.SetParameters(parameters.SubVector(offset, count));
    }

    public override void ResetState()
    {
        _lastInput = null;
        _bn1Out = null;
        _relu1Out = null;
        _conv1Out = null;
        _bn2Out = null;
        _relu2Out = null;

        _gpuBn1Out = null;
        _gpuConv1Out = null;
        _gpuBn2Out = null;

        _bn1.ResetState();
        _conv1x1.ResetState();
        _bn2.ResetState();
        _conv3x3.ResetState();
    }

    // --- ILayerSerializationExtras: propagate internal BN running stats ---
    // Without this, DenseNet's serialize/deserialize round-trip for trained
    // models loses each block's BN running mean/variance, and the cloned
    // model uses default zero-mean/unit-variance for inference — producing
    // outputs that diverge from the original by orders of magnitude. Mirrors
    // InvertedResidualBlock.cs:463-510 (the precedent for nested BN in a
    // composite block).

    int ILayerSerializationExtras<T>.ExtraParameterCount
    {
        get
        {
            int count = 0;
            if (_bn1 is ILayerSerializationExtras<T> e1) count += e1.ExtraParameterCount;
            if (_bn2 is ILayerSerializationExtras<T> e2) count += e2.ExtraParameterCount;
            return count;
        }
    }

    Vector<T> ILayerSerializationExtras<T>.GetExtraParameters()
    {
        var parts = new List<T>();
        if (_bn1 is ILayerSerializationExtras<T> e1)
            parts.AddRange(e1.GetExtraParameters().ToArray());
        if (_bn2 is ILayerSerializationExtras<T> e2)
            parts.AddRange(e2.GetExtraParameters().ToArray());
        return new Vector<T>(parts.ToArray());
    }

    void ILayerSerializationExtras<T>.SetExtraParameters(Vector<T> extraParameters)
    {
        int offset = 0;
        if (_bn1 is ILayerSerializationExtras<T> e1)
        {
            int count = e1.ExtraParameterCount;
            if (offset + count > extraParameters.Length)
                throw new ArgumentException(
                    $"Truncated extra-parameters for _bn1: need {offset + count} but got {extraParameters.Length}.");
            e1.SetExtraParameters(extraParameters.SubVector(offset, count));
            offset += count;
        }
        if (_bn2 is ILayerSerializationExtras<T> e2)
        {
            int count = e2.ExtraParameterCount;
            if (offset + count > extraParameters.Length)
                throw new ArgumentException(
                    $"Truncated extra-parameters for _bn2: need {offset + count} but got {extraParameters.Length}.");
            e2.SetExtraParameters(extraParameters.SubVector(offset, count));
            offset += count;
        }
    }
}
