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

    public Tensor<T> Forward(Tensor<T> input) => _layer.Forward(input);

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
        _inDim = inDim;
        _outDim = outDim;
        _layer = new DenseLayer<T>(outDim, (Interfaces.IActivationFunction<T>?)null);
    }

    public Tensor<T> Forward(Tensor<T> input) => _layer.Forward(input);

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
            return new Tensor<T>(new[] { _outDim, _inDim }, new Vector<T>(arr));
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
