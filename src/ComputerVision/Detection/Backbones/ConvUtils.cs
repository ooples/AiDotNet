using System.IO;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.ComputerVision.Detection.Backbones;

/// <summary>
/// 2D convolution layer used by detection backbones (ResNet, CSPDarknet, EfficientNet,
/// SwinTransformer). Thin wrapper around <see cref="ConvolutionalLayer{T}"/> that exposes
/// the legacy <c>Forward</c> / <c>GetParameterCount</c> / <c>WriteParameters</c> /
/// <c>ReadParameters</c> API the backbones were written against.
/// </summary>
/// <remarks>
/// The underlying <see cref="ConvolutionalLayer{T}"/> is lazy: it resolves input depth from
/// the input tensor on first forward, so <paramref name="inChannels"/> here is purely
/// informational (used by serialization-config validation).
/// </remarks>
internal class Conv2D<T>
{
    private readonly ConvolutionalLayer<T> _layer;
    private readonly int _inChannels;
    private readonly int _outChannels;
    private readonly int _kernelSize;
    private readonly int _stride;
    private readonly int _padding;

    public Tensor<T> Weights => _layer.GetParameters() is { Length: > 0 } v
        ? new Tensor<T>(new[] { _outChannels, _inChannels, _kernelSize, _kernelSize },
                        new Vector<T>(v.ToArray().AsSpan(0, _outChannels * _inChannels * _kernelSize * _kernelSize).ToArray()))
        : new Tensor<T>(new[] { _outChannels, _inChannels, _kernelSize, _kernelSize });

    /// <summary>
    /// Bias tensor extracted from the underlying <see cref="ConvolutionalLayer{T}"/>'s parameter
    /// vector. <see cref="ConvolutionalLayer{T}"/> always allocates a bias of length
    /// <c>OutputDepth</c>; before the layer has been forward-resolved the parameter vector is
    /// empty and we return an empty placeholder.
    /// </summary>
    public Tensor<T> Bias
    {
        get
        {
            var p = _layer.GetParameters();
            int weightLen = _outChannels * _inChannels * _kernelSize * _kernelSize;
            if (p.Length < weightLen + _outChannels)
                return new Tensor<T>(new[] { _outChannels });
            var arr = new T[_outChannels];
            for (int i = 0; i < _outChannels; i++) arr[i] = p[weightLen + i];
            return new Tensor<T>(new[] { _outChannels }, new Vector<T>(arr));
        }
    }

    /// <summary>
    /// Constructs a 2D convolution. The underlying <see cref="ConvolutionalLayer{T}"/> always
    /// allocates a bias term — there is no <c>useBias=false</c> mode at present, so callers
    /// that previously paired conv with a following BatchNorm just carry redundant bias
    /// parameters. This matches the legacy contract for the <c>useBias=true</c> case.
    /// </summary>
    public Conv2D(int inChannels, int outChannels, int kernelSize, int stride = 1, int padding = 0)
    {
        if (inChannels <= 0) throw new ArgumentOutOfRangeException(nameof(inChannels), "inChannels must be positive.");
        if (outChannels <= 0) throw new ArgumentOutOfRangeException(nameof(outChannels), "outChannels must be positive.");
        if (kernelSize <= 0) throw new ArgumentOutOfRangeException(nameof(kernelSize), "kernelSize must be positive.");
        if (stride <= 0) throw new ArgumentOutOfRangeException(nameof(stride), "stride must be positive.");
        if (padding < 0) throw new ArgumentOutOfRangeException(nameof(padding), "padding cannot be negative.");

        _inChannels = inChannels;
        _outChannels = outChannels;
        _kernelSize = kernelSize;
        _stride = stride;
        _padding = padding;
        _layer = new ConvolutionalLayer<T>(outChannels, kernelSize, stride, padding, (Interfaces.IActivationFunction<T>?)null);
    }

    public Tensor<T> Forward(Tensor<T> input) => _layer.Forward(input);

    public long GetParameterCount() => _layer.ParameterCount;

    public void WriteParameters(BinaryWriter writer)
    {
        writer.Write(_inChannels);
        writer.Write(_outChannels);
        writer.Write(_kernelSize);
        writer.Write(_stride);
        writer.Write(_padding);
        var p = _layer.GetParameters();
        writer.Write(p.Length);
        var numOps = MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < p.Length; i++)
            writer.Write(numOps.ToDouble(p[i]));
    }

    public void ReadParameters(BinaryReader reader)
    {
        int inC = reader.ReadInt32();
        int outC = reader.ReadInt32();
        int k = reader.ReadInt32();
        int s = reader.ReadInt32();
        int p = reader.ReadInt32();
        if (inC != _inChannels || outC != _outChannels || k != _kernelSize || s != _stride || p != _padding)
            throw new InvalidOperationException(
                $"Conv2D configuration mismatch during deserialization. " +
                $"Expected ({_inChannels},{_outChannels},{_kernelSize},{_stride},{_padding}); got ({inC},{outC},{k},{s},{p}).");
        int len = reader.ReadInt32();
        var values = new T[len];
        var numOps = MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < len; i++) values[i] = numOps.FromDouble(reader.ReadDouble());
        if (len > 0) _layer.SetParameters(new Vector<T>(values));
    }
}

/// <summary>
/// Fully-connected layer used by detection backbones. Thin wrapper around
/// <see cref="DenseLayer{T}"/> with the legacy parameter-serialization API.
/// </summary>
internal class Dense<T>
{
    private readonly DenseLayer<T> _layer;
    private readonly int _inDim;
    private readonly int _outDim;

    public int InputSize => _inDim;
    public int OutputSize => _outDim;

    public Tensor<T> Weights
    {
        get
        {
            var p = _layer.GetParameters();
            int wlen = _inDim * _outDim;
            if (p.Length < wlen)
                return new Tensor<T>(new[] { _outDim, _inDim });
            var arr = new T[wlen];
            for (int i = 0; i < wlen; i++) arr[i] = p[i];
            return new Tensor<T>(new[] { _outDim, _inDim }, new Vector<T>(arr));
        }
    }

    public Tensor<T> Bias
    {
        get
        {
            var p = _layer.GetParameters();
            int wlen = _inDim * _outDim;
            if (p.Length < wlen + _outDim)
                return new Tensor<T>(new[] { _outDim });
            var arr = new T[_outDim];
            for (int i = 0; i < _outDim; i++) arr[i] = p[wlen + i];
            return new Tensor<T>(new[] { _outDim }, new Vector<T>(arr));
        }
    }

    public Dense(int inDim, int outDim)
    {
        _inDim = inDim;
        _outDim = outDim;
        _layer = new DenseLayer<T>(outDim, (Interfaces.IActivationFunction<T>?)null);
    }

    public Tensor<T> Forward(Tensor<T> input) => _layer.Forward(input);

    public long GetParameterCount() => _layer.ParameterCount;

    public void WriteParameters(BinaryWriter writer)
    {
        writer.Write(_inDim);
        writer.Write(_outDim);
        var p = _layer.GetParameters();
        writer.Write(p.Length);
        var numOps = MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < p.Length; i++)
            writer.Write(numOps.ToDouble(p[i]));
    }

    public void ReadParameters(BinaryReader reader)
    {
        int inD = reader.ReadInt32();
        int outD = reader.ReadInt32();
        if (inD != _inDim || outD != _outDim)
            throw new InvalidOperationException(
                $"Dense configuration mismatch: expected ({_inDim},{_outDim}); got ({inD},{outD}).");
        int len = reader.ReadInt32();
        var values = new T[len];
        var numOps = MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < len; i++) values[i] = numOps.FromDouble(reader.ReadDouble());
        if (len > 0) _layer.SetParameters(new Vector<T>(values));
    }
}

/// <summary>
/// Multi-head self-attention used by SwinTransformer-style backbones. Thin wrapper around
/// <see cref="MultiHeadAttentionLayer{T}"/>.
/// </summary>
internal class MultiHeadSelfAttention<T>
{
    private readonly MultiHeadAttentionLayer<T> _layer;
    private readonly int _dim;
    private readonly int _numHeads;

    public MultiHeadSelfAttention(int dim, int numHeads)
    {
        if (dim <= 0) throw new ArgumentOutOfRangeException(nameof(dim), "dim must be positive.");
        if (numHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numHeads), "numHeads must be positive.");
        if (dim % numHeads != 0)
            throw new ArgumentException(
                $"dim ({dim}) must be evenly divisible by numHeads ({numHeads}); got remainder {dim % numHeads}.",
                nameof(dim));

        _dim = dim;
        _numHeads = numHeads;
        _layer = new MultiHeadAttentionLayer<T>(numHeads, dim / numHeads, (Interfaces.IActivationFunction<T>?)null);
    }

    public Tensor<T> Forward(Tensor<T> input) => _layer.Forward(input);

    public long GetParameterCount() => _layer.ParameterCount;

    public void WriteParameters(BinaryWriter writer)
    {
        writer.Write(_dim);
        writer.Write(_numHeads);
        var p = _layer.GetParameters();
        writer.Write(p.Length);
        var numOps = MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < p.Length; i++)
            writer.Write(numOps.ToDouble(p[i]));
    }

    public void ReadParameters(BinaryReader reader)
    {
        int dim = reader.ReadInt32();
        int heads = reader.ReadInt32();
        if (dim != _dim || heads != _numHeads)
            throw new InvalidOperationException(
                $"MultiHeadSelfAttention configuration mismatch: expected dim={_dim}, numHeads={_numHeads}, got dim={dim}, numHeads={heads}.");
        int len = reader.ReadInt32();
        var values = new T[len];
        var numOps = MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < len; i++) values[i] = numOps.FromDouble(reader.ReadDouble());
        if (len > 0) _layer.SetParameters(new Vector<T>(values));
    }
}
