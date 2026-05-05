using System.IO;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.ComputerVision.Detection.Backbones;

/// <summary>
/// Thin convenience adapter around <see cref="ConvolutionalLayer{T}"/> used by detection /
/// OCR / segmentation models (DETR/DINO/RTDETR, FasterRCNN/CascadeRCNN, YOLO heads, CRAFT/
/// DBNet/EAST text detectors, MaskRCNN, CRNN/TrOCR, etc.). The 4 backbones (ResNet,
/// CSPDarknet, EfficientNet, SwinTransformer) consume <c>ConvolutionalLayer&lt;T&gt;</c>
/// directly — this shim only exists for the legacy detection-head call sites that were
/// written against the pre-lazy parallel-Conv2D contract. Post-#1209 it is a 30-line
/// adapter, not a parallel implementation.
/// </summary>
internal class Conv2D<T>
{
    private readonly ConvolutionalLayer<T> _layer;
    private readonly int _inChannels;
    private readonly int _outChannels;
    private readonly int _kernelSize;
    private readonly int _stride;
    private readonly int _padding;

    public Conv2D(int inChannels, int outChannels, int kernelSize, int stride = 1, int padding = 0)
    {
        if (inChannels <= 0) throw new ArgumentOutOfRangeException(nameof(inChannels));
        if (outChannels <= 0) throw new ArgumentOutOfRangeException(nameof(outChannels));
        if (kernelSize <= 0) throw new ArgumentOutOfRangeException(nameof(kernelSize));
        if (stride <= 0) throw new ArgumentOutOfRangeException(nameof(stride));
        if (padding < 0) throw new ArgumentOutOfRangeException(nameof(padding));

        _inChannels = inChannels;
        _outChannels = outChannels;
        _kernelSize = kernelSize;
        _stride = stride;
        _padding = padding;
        _layer = new ConvolutionalLayer<T>(outChannels, kernelSize, stride, padding,
            (Interfaces.IActivationFunction<T>?)null);
    }

    public Tensor<T> Forward(Tensor<T> input)
    {
        // Validate runtime input channel count matches the shim's
        // construction-time _inChannels. The underlying lazy
        // ConvolutionalLayer infers in-channels from input.Shape on
        // first Forward — if the caller passes a tensor whose channel
        // dim doesn't match what the shim was built for, the layer
        // resolves to the runtime count and the shim's later
        // Weights/Bias slicing (which uses the construction-time
        // _inChannels) silently misaligns. Throw loud rather than
        // ship corrupted weight inspections.
        // Conv2D backbone shim only supports NCHW (rank 4). Reject lower
        // ranks explicitly: a rank-3 [C, H, W] would let runtimeChannels
        // read input.Shape[1] = H and throw a misleading "channels" error
        // that leads the caller to fix the wrong axis. ConvolutionalLayer
        // itself accepts rank-3, but the BackboneBase contract is NCHW-
        // only — backbones produce batched feature maps.
        if (input.Shape.Length < 4)
        {
            throw new ArgumentException(
                $"Conv2D backbone shim requires NCHW (rank 4) input; got rank {input.Shape.Length} " +
                $"with shape [{string.Join(",", input.Shape)}]. Detection backbones operate on " +
                "batched feature maps — add a leading batch dimension before passing through.",
                nameof(input));
        }
        int runtimeChannels = input.Shape[1]; // NCHW
        if (runtimeChannels != _inChannels)
            throw new ArgumentException(
                $"Conv2D shim was constructed for inChannels={_inChannels} but input has " +
                $"{runtimeChannels} channels along axis 1 (NCHW). The shim's parameter slicing " +
                $"depends on the construction-time channel count; reconstruct the shim with the " +
                $"correct inChannels or reshape the input.",
                nameof(input));
        return _layer.Forward(input);
    }

    public long GetParameterCount() => _layer.ParameterCount;

    public void WriteParameters(BinaryWriter writer) =>
        BackboneSerialization.WriteLayerParameters(writer, _layer);

    public void ReadParameters(BinaryReader reader) =>
        BackboneSerialization.ReadLayerParameters(reader, _layer);

    public Tensor<T> Weights
    {
        get
        {
            if (!_layer.IsShapeResolved)
                throw new InvalidOperationException(
                    "Conv2D weights are unavailable until the layer's first Forward(): " +
                    "input depth is still the lazy sentinel.");
            var p = _layer.GetParameters();
            int weightLen = _outChannels * _inChannels * _kernelSize * _kernelSize;
            var arr = new T[weightLen];
            for (int i = 0; i < weightLen; i++) arr[i] = p[i];
            return new Tensor<T>(new[] { _outChannels, _inChannels, _kernelSize, _kernelSize }, new Vector<T>(arr));
        }
    }

    public Tensor<T> Bias
    {
        get
        {
            if (!_layer.IsShapeResolved)
                throw new InvalidOperationException(
                    "Conv2D bias is unavailable until the layer's first Forward().");
            var p = _layer.GetParameters();
            int weightLen = _outChannels * _inChannels * _kernelSize * _kernelSize;
            var arr = new T[_outChannels];
            for (int i = 0; i < _outChannels; i++) arr[i] = p[weightLen + i];
            return new Tensor<T>(new[] { _outChannels }, new Vector<T>(arr));
        }
    }
}

/// <summary>Thin adapter around <see cref="DenseLayer{T}"/> for legacy detection-head call sites.</summary>
internal class Dense<T>
{
    private readonly DenseLayer<T> _layer;
    private readonly int _inDim;
    private readonly int _outDim;

    public int InputSize => _inDim;
    public int OutputSize => _outDim;

    public Dense(int inDim, int outDim)
    {
        if (inDim <= 0) throw new ArgumentOutOfRangeException(nameof(inDim), "inDim must be positive.");
        if (outDim <= 0) throw new ArgumentOutOfRangeException(nameof(outDim), "outDim must be positive.");

        _inDim = inDim;
        _outDim = outDim;
        _layer = new DenseLayer<T>(outDim, (Interfaces.IActivationFunction<T>?)null);
    }

    public Tensor<T> Forward(Tensor<T> input)
    {
        // Validate runtime input feature size against the shim's
        // construction-time _inDim. DenseLayer<T> can resize its weight
        // matrix when the runtime feature dim differs (lazy resize via
        // shape resolution); the shim's Weights/Bias slicing uses the
        // construction-time _inDim, so a runtime-resize would silently
        // reshape the underlying matrix and break the shim's
        // serialization / inspection contract. Validate input.Shape[^1]
        // (last dim = features for any rank ≥ 1) and throw on mismatch.
        if (input.Shape.Length >= 1)
        {
            int runtimeFeatures = input.Shape[input.Shape.Length - 1];
            if (runtimeFeatures != _inDim)
                throw new ArgumentException(
                    $"Dense shim was constructed for inDim={_inDim} but input has " +
                    $"{runtimeFeatures} features along the last axis. The shim's parameter " +
                    $"slicing depends on the construction-time inDim; reconstruct the shim " +
                    $"with the correct inDim or reshape the input.",
                    nameof(input));
        }
        return _layer.Forward(input);
    }

    public long GetParameterCount() => _layer.ParameterCount;

    public void WriteParameters(BinaryWriter writer) =>
        BackboneSerialization.WriteLayerParameters(writer, _layer);

    public void ReadParameters(BinaryReader reader) =>
        BackboneSerialization.ReadLayerParameters(reader, _layer);

    public Tensor<T> Weights
    {
        get
        {
            if (!_layer.IsShapeResolved)
                throw new InvalidOperationException(
                    "Dense weights are unavailable until the layer's first Forward().");
            var p = _layer.GetParameters();
            int wlen = _inDim * _outDim;
            var arr = new T[wlen];
            for (int i = 0; i < wlen; i++) arr[i] = p[i];
            // Match DenseLayer's underlying weight shape [inputSize, outputSize]
            // (see DenseLayer.cs:434 — TensorAllocator.Rent<T>([inputSize, outputSize])).
            // Returning [_outDim, _inDim] without transposing the flat data
            // would mis-shape the matrix for any caller that reads .Weights
            // expecting the layer's native layout.
            return new Tensor<T>(new[] { _inDim, _outDim }, new Vector<T>(arr));
        }
    }

    public Tensor<T> Bias
    {
        get
        {
            if (!_layer.IsShapeResolved)
                throw new InvalidOperationException(
                    "Dense bias is unavailable until the layer's first Forward().");
            var p = _layer.GetParameters();
            int wlen = _inDim * _outDim;
            var arr = new T[_outDim];
            for (int i = 0; i < _outDim; i++) arr[i] = p[wlen + i];
            return new Tensor<T>(new[] { _outDim }, new Vector<T>(arr));
        }
    }
}

/// <summary>Thin adapter around <see cref="MultiHeadAttentionLayer{T}"/>.</summary>
internal class MultiHeadSelfAttention<T>
{
    private readonly MultiHeadAttentionLayer<T> _layer;
    private readonly int _dim;
    private readonly int _numHeads;

    public MultiHeadSelfAttention(int dim, int numHeads)
    {
        if (dim <= 0) throw new ArgumentOutOfRangeException(nameof(dim));
        if (numHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numHeads));
        if (dim % numHeads != 0)
            throw new ArgumentException(
                $"dim ({dim}) must be evenly divisible by numHeads ({numHeads}); got remainder {dim % numHeads}.",
                nameof(dim));

        _dim = dim;
        _numHeads = numHeads;
        _layer = new MultiHeadAttentionLayer<T>(numHeads, dim / numHeads,
            (Interfaces.IActivationFunction<T>?)null);
    }

    public Tensor<T> Forward(Tensor<T> input) => _layer.Forward(input);

    public long GetParameterCount() => _layer.ParameterCount;

    public void WriteParameters(BinaryWriter writer) =>
        BackboneSerialization.WriteLayerParameters(writer, _layer);

    public void ReadParameters(BinaryReader reader) =>
        BackboneSerialization.ReadLayerParameters(reader, _layer);
}
