using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Implements a Dense Block from the DenseNet architecture.
/// </summary>
/// <remarks>
/// <para>
/// A Dense Block is the core building block of DenseNet. It contains multiple layers where
/// each layer receives feature maps from ALL preceding layers (dense connectivity).
/// This creates strong gradient flow and feature reuse throughout the network.
/// </para>
/// <para>
/// Architecture of a Dense Block with n layers:
/// <code>
/// Input (k0 channels)
///   ↓
/// Layer 1: BN → ReLU → Conv1x1 → BN → ReLU → Conv3x3 → Output1 (k channels)
///   ↓ concat
/// [Input, Output1] (k0 + k channels)
///   ↓
/// Layer 2: BN → ReLU → Conv1x1 → BN → ReLU → Conv3x3 → Output2 (k channels)
///   ↓ concat
/// [Input, Output1, Output2] (k0 + 2k channels)
///   ↓
/// ... (continues for n layers)
///   ↓
/// Final: [Input, Output1, ..., OutputN] (k0 + n*k channels)
/// </code>
/// Where k is the growth rate (number of channels added per layer).
/// </para>
/// <para>
/// <b>For Beginners:</b> Dense connectivity means each layer can directly access
/// features from all previous layers, promoting feature reuse and reducing
/// the need for redundant feature learning.
///
/// Key benefits:
/// - Strong gradient flow (helps with training very deep networks)
/// - Feature reuse (each layer can use features from all previous layers)
/// - Fewer parameters (layers can be narrow since they share features)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[LayerCategory(LayerCategory.Residual)]
[LayerCategory(LayerCategory.Convolution)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, ChangesShape = true, ExpectedInputRank = 3, Cost = ComputeCost.High, TestInputShape = "4, 8, 8", TestConstructorArgs = "4, 3")]
public class DenseBlock<T> : LayerBase<T>, ILayerSerializationExtras<T>
{
    private readonly List<DenseBlockLayer<T>> _layers;
    private readonly int _numLayers;
    private readonly int _growthRate;
    // Non-readonly: lazy ctor leaves _inputChannels = -1 until
    // OnFirstForward resolves it from the runtime input tensor.
    private int _inputChannels;
    private List<Tensor<T>>? _layerOutputs;

    // GPU cached tensors for backward pass
    private List<Tensor<T>>? _gpuFeatureMaps;

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    public override long ParameterCount => (int)_layers.Sum(l => l.ParameterCount);
    public override bool SupportsTraining => true;

    public override Vector<T> GetParameterGradients()
    {
        return new Vector<T>(_layers.SelectMany(l => l.GetParameterGradients().ToArray()).ToArray());
    }

    public override void ClearGradients()
    {
        base.ClearGradients();
        foreach (var l in _layers) l.ClearGradients();
    }

    /// <summary>
    /// Gets a value indicating whether this layer has a GPU implementation.
    /// </summary>
    protected override bool SupportsGpuExecution => true;

    /// <summary>
    /// Gets the number of layers in this dense block.
    /// </summary>
    public int NumLayers => _numLayers;

    /// <summary>
    /// Gets the growth rate (channels added per layer).
    /// </summary>
    public int GrowthRate => _growthRate;

    /// <summary>
    /// Gets the number of output channels (inputChannels + numLayers × growthRate).
    /// Returns the sentinel <c>-1</c> until <see cref="OnFirstForward"/> resolves
    /// the input channel count; downstream planning that needs this at
    /// construction time should compute it from the previous layer's
    /// output channel count instead.
    /// </summary>
    public int OutputChannels => _inputChannels < 0 ? -1 : _inputChannels + _numLayers * _growthRate;

    /// <summary>
    /// Initializes a new instance of the <see cref="DenseBlock{T}"/> class.
    /// </summary>
    /// <param name="inputChannels">The number of input channels.</param>
    /// <param name="numLayers">The number of layers in the dense block.</param>
    /// <param name="growthRate">The number of channels each layer adds (k in the paper).</param>
    /// <param name="inputHeight">The input feature map height.</param>
    /// <param name="inputWidth">The input feature map width.</param>
    /// <param name="bnMomentum">Batch normalization momentum (default: 0.1).</param>
    /// <summary>
    /// Lazy ctor — input depth/height/width come from the first
    /// <see cref="Forward"/> call (<see cref="OnFirstForward"/>); only
    /// numLayers/growthRate/bnMomentum are required at construction.
    /// Each inner <see cref="DenseBlockLayer{T}"/> resolves its own
    /// input channel count lazily on its own first Forward.
    /// </summary>
    public DenseBlock(
        int numLayers,
        int growthRate,
        double bnMomentum = 0.1)
        : base(
            inputShape: [-1, -1, -1],
            outputShape: [-1, -1, -1])
    {
        if (numLayers <= 0) throw new ArgumentOutOfRangeException(nameof(numLayers));
        if (growthRate <= 0) throw new ArgumentOutOfRangeException(nameof(growthRate));

        _inputChannels = -1; // resolved in OnFirstForward
        _numLayers = numLayers;
        _growthRate = growthRate;
        _layers = new List<DenseBlockLayer<T>>(numLayers);

        for (int i = 0; i < numLayers; i++)
        {
            var layer = new DenseBlockLayer<T>(growthRate, bnMomentum);
            _layers.Add(layer);
            RegisterSubLayer(layer);
        }
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Resolves the input channel count from <c>input.Shape</c> and locks
    /// the block's <see cref="OutputChannels"/> = inputChannels + numLayers
    /// × growthRate. Each inner <see cref="DenseBlockLayer{T}"/> resolves
    /// its own per-block input channel count (which grows by
    /// <c>growthRate</c> per layer) on its own first Forward.
    /// </remarks>
    protected override void OnFirstForward(Tensor<T> input)
    {
        // Use the public Shape property rather than the internal _shape
        // field so we go through Tensor<T>'s encapsulation boundary —
        // future changes that add validation, lazy-eval or proxy logic
        // to Shape get applied uniformly to every consumer.
        var s = input.Shape;
        int channels, height, width;
        if (s.Length == 3) { channels = s[0]; height = s[1]; width = s[2]; }
        else if (s.Length == 4) { channels = s[1]; height = s[2]; width = s[3]; }
        else
            throw new ArgumentException(
                $"DenseBlock requires rank-3 [C,H,W] or rank-4 [B,C,H,W] input; got rank {s.Length}.",
                nameof(input));

        _inputChannels = channels;

        // Drive each inner DenseBlockLayer's shape resolution so their
        // GetParameters() length is correct even before any inner Forward
        // has fired. Per DenseNet semantics, each inner layer receives the
        // accumulated [previous-features + concatenated-growth] tensor.
        int currentChannels = channels;
        foreach (var layer in _layers)
        {
            layer.ResolveShapesOnly(new[] { currentChannels, height, width });
            currentChannels += _growthRate;
        }

        ResolveShapes(
            new[] { channels, height, width },
            new[] { channels + _numLayers * _growthRate, height, width });

        // Replay parameters that arrived via Deserialize → SetParameters
        // before inner-layer shapes were resolved.
        if (_pendingParameters is not null)
        {
            var pending = _pendingParameters;
            _pendingParameters = null;
            ApplyParameters(pending);
        }
    }

    /// <summary>
    /// Performs the forward pass of the Dense Block.
    /// </summary>
    /// <param name="input">The input tensor [B, C, H, W].</param>
    /// <returns>The output tensor with all layer outputs concatenated.</returns>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Lazy gate — drives inner-layer shape resolution so any
        // Deserialize-buffered parameters get a chance to be replayed
        // before the inner-layer Forwards consume their weights.
        if (!IsShapeResolved) OnFirstForward(input);

        _layerOutputs = new List<Tensor<T>>(_numLayers + 1) { input };

        // Current feature maps (accumulated)
        var currentFeatures = input;

        foreach (var layer in _layers)
        {
            // Each layer takes ALL previous features as input
            var layerOutput = layer.Forward(currentFeatures);
            _layerOutputs.Add(layerOutput);

            // Concatenate new features with existing features along channel dimension
            currentFeatures = ConcatenateChannels(currentFeatures, layerOutput);
        }

        return currentFeatures;
    }

    /// <summary>
    /// Performs the forward pass on GPU, keeping data GPU-resident.
    /// </summary>
    /// <param name="inputs">The input tensors (expects single input).</param>
    /// <returns>The output tensor on GPU.</returns>
    public override Tensor<T> ForwardGpu(params Tensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));

        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("ForwardGpu requires a DirectGpuTensorEngine.");

        var currentFeatures = inputs[0];

        // Cache feature maps for backward pass during training
        if (IsTrainingMode)
        {
            _gpuFeatureMaps = new List<Tensor<T>>(_numLayers + 1) { currentFeatures };
        }

        foreach (var layer in _layers)
        {
            // Each layer takes ALL previous features as input
            var layerOutput = layer.ForwardGpu(currentFeatures);

            // Concatenate new features with existing features along channel dimension (axis 1)
            currentFeatures = gpuEngine.ConcatGpu(new[] { currentFeatures, layerOutput }, 1);

            // Cache for backward pass
            if (IsTrainingMode)
            {
                var gpuMaps = _gpuFeatureMaps ?? throw new InvalidOperationException("_gpuFeatureMaps has not been initialized.");
                gpuMaps.Add(currentFeatures);
            }
        }

        return currentFeatures;
    }

    /// <summary>
    /// Updates the parameters of all sub-layers.
    /// </summary>
    /// <param name="learningRate">The learning rate for parameter updates.</param>
    public override void UpdateParameters(T learningRate)
    {
        foreach (var layer in _layers)
        {
            layer.UpdateParameters(learningRate);
        }
    }

    /// <summary>
    /// Gets all trainable parameters from the block.
    /// </summary>
    public override Vector<T> GetParameters()
    {
        return new Vector<T>(_layers.SelectMany(l => l.GetParameters().ToArray()).ToArray());
    }

    /// <summary>
    /// Sets all trainable parameters from the given parameter vector.
    /// </summary>
    /// <param name="parameters">The parameter vector containing all layer parameters.</param>
    public override void SetParameters(Vector<T> parameters)
    {
        // Pre-Forward: inner layers' shapes haven't been resolved, so
        // their slice lengths are wrong. Buffer the full vector and
        // replay from OnFirstForward, after each inner DenseBlockLayer
        // has a chance to lock in its own input channel count.
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
        foreach (var layer in _layers)
        {
            // Each inner layer also buffers: distribute by handing over
            // its slice (still-unknown size) so its own SetParameters
            // can buffer/replay independently.
            int count = layer.GetParameters().Length;
            layer.SetParameters(parameters.SubVector(offset, count));
            offset += count;
        }
    }

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["InputChannels"] = _inputChannels.ToString();
        metadata["NumLayers"] = _numLayers.ToString();
        metadata["GrowthRate"] = _growthRate.ToString();
        return metadata;
    }

    /// <summary>
    /// Resets the internal state of the block.
    /// </summary>
    public override void ResetState()
    {
        _layerOutputs = null;
        _gpuFeatureMaps = null;
        foreach (var layer in _layers)
        {
            layer.ResetState();
        }
    }

    #region Helper Methods

    /// <summary>
    /// Concatenates two tensors along the channel dimension (dim=0 for CHW, dim=1 for NCHW).
    /// </summary>
    private Tensor<T> ConcatenateChannels(Tensor<T> a, Tensor<T> b)
    {
        // Channel axis is 0 for 3D (CHW) and 1 for 4D (NCHW)
        int channelAxis = a.Shape.Length == 4 ? 1 : 0;
        return Engine.TensorConcatenate([a, b], axis: channelAxis);
    }

    /// <summary>
    /// Splits a gradient tensor along the channel dimension (dim=0 for CHW, dim=1 for NCHW).
    /// </summary>
    private (Tensor<T> first, Tensor<T> second) SplitGradient(Tensor<T> grad, int firstChannels, int secondChannels)
    {
        // Handle 3D tensors (CHW format)
        if (grad.Shape.Length == 3)
        {
            int height = grad.Shape[1];
            int width = grad.Shape[2];

            var first = new Tensor<T>([firstChannels, height, width]);
            var second = new Tensor<T>([secondChannels, height, width]);

            int spatialSize = height * width;

            for (int c = 0; c < firstChannels; c++)
            {
                for (int hw = 0; hw < spatialSize; hw++)
                {
                    int srcIdx = c * spatialSize + hw;
                    int dstIdx = c * spatialSize + hw;
                    first.Data.Span[dstIdx] = grad.Data.Span[srcIdx];
                }
            }

            for (int c = 0; c < secondChannels; c++)
            {
                for (int hw = 0; hw < spatialSize; hw++)
                {
                    int srcIdx = (firstChannels + c) * spatialSize + hw;
                    int dstIdx = c * spatialSize + hw;
                    second.Data.Span[dstIdx] = grad.Data.Span[srcIdx];
                }
            }

            return (first, second);
        }

        // Handle 4D tensors (NCHW format)
        int batch = grad.Shape[0];
        int height4D = grad.Shape[2];
        int width4D = grad.Shape[3];

        var first4D = new Tensor<T>([batch, firstChannels, height4D, width4D]);
        var second4D = new Tensor<T>([batch, secondChannels, height4D, width4D]);

        int totalChannels4D = firstChannels + secondChannels;
        int spatialSize4D = height4D * width4D;

        for (int n = 0; n < batch; n++)
        {
            for (int c = 0; c < firstChannels; c++)
            {
                for (int hw = 0; hw < spatialSize4D; hw++)
                {
                    int srcIdx = n * (totalChannels4D * spatialSize4D) + c * spatialSize4D + hw;
                    int dstIdx = n * (firstChannels * spatialSize4D) + c * spatialSize4D + hw;
                    first4D.Data.Span[dstIdx] = grad.Data.Span[srcIdx];
                }
            }

            for (int c = 0; c < secondChannels; c++)
            {
                for (int hw = 0; hw < spatialSize4D; hw++)
                {
                    int srcIdx = n * (totalChannels4D * spatialSize4D) + (firstChannels + c) * spatialSize4D + hw;
                    int dstIdx = n * (secondChannels * spatialSize4D) + c * spatialSize4D + hw;
                    second4D.Data.Span[dstIdx] = grad.Data.Span[srcIdx];
                }
            }
        }

        return (first4D, second4D);
    }

    /// <summary>
    /// Adds two gradient tensors of the same shape element-wise.
    /// </summary>
    private Tensor<T> AddGradients(Tensor<T> a, Tensor<T> b)
    {
        return Engine.TensorAdd(a, b);
    }

    #endregion

    // --- ILayerSerializationExtras: propagate nested DenseBlockLayer BN
    // running stats through the block's serialization. Each child
    // DenseBlockLayer implements its own ExtraParameters covering its two
    // BN sub-layers; we just chain them. Without this, post-training Clone
    // loses every block's running stats and inference diverges by
    // orders of magnitude (Clone_AfterTraining_ShouldPreserveLearnedWeights).

    int ILayerSerializationExtras<T>.ExtraParameterCount
    {
        get
        {
            int count = 0;
            foreach (var layer in _layers)
            {
                if (layer is ILayerSerializationExtras<T> ex)
                    count += ex.ExtraParameterCount;
            }
            return count;
        }
    }

    Vector<T> ILayerSerializationExtras<T>.GetExtraParameters()
    {
        var parts = new List<T>();
        foreach (var layer in _layers)
        {
            if (layer is ILayerSerializationExtras<T> ex)
                parts.AddRange(ex.GetExtraParameters().ToArray());
        }
        return new Vector<T>(parts.ToArray());
    }

    void ILayerSerializationExtras<T>.SetExtraParameters(Vector<T> extraParameters)
    {
        int offset = 0;
        // Index loop instead of foreach + IndexOf: the original IndexOf call
        // inside the error message turned the loop O(n²) for no reason.
        // The error is only thrown on truncation, so we don't need IndexOf
        // for the happy path either — just track the index directly.
        for (int i = 0; i < _layers.Count; i++)
        {
            var layer = _layers[i];
            if (layer is ILayerSerializationExtras<T> ex)
            {
                int count = ex.ExtraParameterCount;
                if (offset + count > extraParameters.Length)
                    throw new ArgumentException(
                        $"Truncated extra-parameters at DenseBlockLayer #{i}: " +
                        $"need {offset + count} but got {extraParameters.Length}.");
                ex.SetExtraParameters(extraParameters.SubVector(offset, count));
                offset += count;
            }
        }

        // Reject surplus payload — silently dropping the tail would let
        // version-mismatched serialized blobs deserialize as if they
        // succeeded, masking schema drift between writer and reader.
        if (offset != extraParameters.Length)
        {
            throw new ArgumentException(
                $"DenseBlock extra-parameters payload had {extraParameters.Length} elements " +
                $"but only {offset} were consumed by sub-layer running stats. " +
                $"This usually means the serialized model was written with a different " +
                $"DenseBlock topology (numLayers / sub-layer composition) than the one " +
                $"being deserialized.");
        }
    }
}
