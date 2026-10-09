using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.Enhancement;

/// <summary>
/// The Conv-TasNet network (Luo &amp; Mesgarani 2019, Sec. III), non-causal configuration: a learned 1-D
/// convolutional encoder, a temporal convolutional network (TCN) separator that estimates one sigmoid mask
/// per source, and a transposed-convolution decoder.
/// </summary>
/// <remarks>
/// <para>
/// Every weight lives in a library layer, so the base tape trains the whole network: encoder, every TCN
/// block, the mask head and the decoder. The previous model kept raw weight tensors with a hand-written
/// forward and a "simplified" update that only moved the decoder (#2298).
/// </para>
/// <para>
/// <b>Encoder:</b> Conv1d(1 → N, kernel L, stride L/2) + ReLU (Eq. 1-2).
/// <b>Separator:</b> global layer norm (gLN), a 1×1 bottleneck to B channels, then X blocks repeated R
/// times. Block x of each repeat: 1×1 conv B → H, PReLU, gLN, depthwise conv (kernel P, dilation 2^x,
/// "same" padding), PReLU, gLN, then a 1×1 residual conv H → B and a 1×1 skip conv H → Sc (Fig. 1C). The
/// skip outputs are summed, passed through PReLU and a 1×1 conv to C·N channels, and a sigmoid gives the
/// masks. The final block's residual output is never consumed, so, as in the reference implementation,
/// it has no residual conv.
/// <b>Decoder:</b> the masked encodings of each source through a transposed Conv1d(N → 1, L, stride L/2).
/// </para>
/// <para>
/// gLN normalizes each example over channels and time with a per-channel scale and shift, which is exactly
/// <c>GroupNorm(1, channels)</c>; it is fed <c>[B, C, T, 1]</c> because a rank-3 tensor would be read as one
/// unbatched <c>[C, H, W]</c> image. Sc equals B, as in the paper's configurations.
/// </para>
/// </remarks>
internal sealed class ConvTasNetNetwork<T>
{
    private readonly int _encoderDim;
    private readonly int _kernelSize;
    private readonly int _stride;
    private readonly int _bottleneckDim;
    private readonly int _hiddenDim;
    private readonly int _numBlocks;
    private readonly int _numRepeats;
    private readonly int _tcnKernelSize;
    private readonly int _numSources;
    private readonly List<ILayer<T>> _layers = new();
    private readonly List<TcnBlock> _blocks = new();
    private IReadOnlyList<ILayer<T>>? _bindSource;

    private Conv1DLayer<T> _encoder;
    private GroupNormalizationLayer<T> _inputNorm;
    private Conv1DLayer<T> _bottleneck;
    private PReLULayer<T> _outputPRelu;
    private Conv1DLayer<T> _maskConv;
    private Conv1DTransposeLayer<T> _decoder;

    public ConvTasNetNetwork(ConvTasNetOptions options)
    {
        if (options is null) throw new ArgumentNullException(nameof(options));
        RequirePositive(options.EncoderDim, nameof(options.EncoderDim));
        RequirePositive(options.KernelSize, nameof(options.KernelSize));
        RequirePositive(options.BottleneckDim, nameof(options.BottleneckDim));
        RequirePositive(options.HiddenDim, nameof(options.HiddenDim));
        RequirePositive(options.NumBlocks, nameof(options.NumBlocks));
        RequirePositive(options.NumRepeats, nameof(options.NumRepeats));
        RequirePositive(options.TcnKernelSize, nameof(options.TcnKernelSize));
        RequirePositive(options.NumSources, nameof(options.NumSources));
        // "Same" padding of dilation * (P - 1) / 2 keeps the frame count only for an odd kernel; an even
        // one shortens every block by one frame and the residual add fails on the first forward.
        if (options.TcnKernelSize % 2 == 0)
            throw new ArgumentOutOfRangeException(nameof(options.TcnKernelSize), options.TcnKernelSize,
                "TcnKernelSize must be odd so the depthwise convolution preserves the frame count (the paper uses 3).");
        _encoderDim = options.EncoderDim;
        _kernelSize = options.KernelSize;
        _stride = Math.Max(1, options.KernelSize / 2);
        _bottleneckDim = options.BottleneckDim;
        _hiddenDim = options.HiddenDim;
        _numBlocks = options.NumBlocks;
        _numRepeats = options.NumRepeats;
        _tcnKernelSize = options.TcnKernelSize;
        _numSources = options.NumSources;
        Build();
    }

    private static void RequirePositive(int value, string name)
    {
        if (value <= 0) throw new ArgumentOutOfRangeException(name, value, $"{name} must be positive.");
    }

    /// <summary>Every layer of the network, in a fixed order, for the owning model to publish.</summary>
    public IReadOnlyList<ILayer<T>> Layers => _layers;

    /// <summary>Encoder stride in samples (half the encoder kernel).</summary>
    public int Stride => _stride;

    /// <summary>
    /// Points every role at the model's current layers, position for position, after a deserialize or eager
    /// clone replaced the instances; does nothing while the graph is unchanged. A list that does not match
    /// the layout is refused and leaves the network as it was.
    /// </summary>
    public void BindTo(IReadOnlyList<ILayer<T>> layers)
    {
        if (layers is null) throw new ArgumentNullException(nameof(layers));
        bool same = layers.Count == _layers.Count;
        for (int i = 0; same && i < layers.Count; i++)
        {
            same = ReferenceEquals(layers[i], _layers[i]);
        }

        if (same) return;
        if (layers.Count != _layers.Count)
            throw new InvalidOperationException(
                $"The layer graph has {layers.Count} layers but the Conv-TasNet layout has {_layers.Count}.");

        var previous = (Layers: _layers.ToList(), Blocks: _blocks.ToList(), Encoder: _encoder, InputNorm: _inputNorm,
            Bottleneck: _bottleneck, OutputPRelu: _outputPRelu, MaskConv: _maskConv, Decoder: _decoder);
        _bindSource = layers;
        try
        {
            Build();
        }
        catch
        {
            _layers.Clear(); _layers.AddRange(previous.Layers);
            _blocks.Clear(); _blocks.AddRange(previous.Blocks);
            _encoder = previous.Encoder;
            _inputNorm = previous.InputNorm;
            _bottleneck = previous.Bottleneck;
            _outputPRelu = previous.OutputPRelu;
            _maskConv = previous.MaskConv;
            _decoder = previous.Decoder;
            throw;
        }
        finally
        {
            _bindSource = null;
        }
    }

    /// <summary>
    /// Separates a mixture <c>[B, samples]</c> into <c>[B, sources, samples]</c>, recording the named stages
    /// into <paramref name="stages"/> when it is given.
    /// </summary>
    public Tensor<T> Forward(Tensor<T> mixture, IDictionary<string, Tensor<T>>? stages = null)
    {
        if (mixture is null) throw new ArgumentNullException(nameof(mixture));
        if (mixture.Shape.Length != 2)
            throw new ArgumentException($"Conv-TasNet expects a mixture [batch, samples]; got rank {mixture.Shape.Length}.", nameof(mixture));

        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        int batch = mixture.Shape[0], samples = mixture.Shape[1];

        // Pad the end so the frames tile the signal: K = (padded - L) / stride + 1 is whole, and the decoder's
        // overlap-add, (K - 1) * stride + L, reproduces the padded length; the output is cropped back.
        int padded = Math.Max(samples, _kernelSize);
        int remainder = (padded - _kernelSize) % _stride;
        if (remainder != 0) padded += _stride - remainder;
        var waveform = engine.Reshape(mixture, new[] { batch, 1, samples });
        if (padded != samples)
        {
            var asImage = engine.Reshape(waveform, new[] { batch, 1, 1, samples });
            waveform = engine.Reshape(engine.Pad(asImage, 0, 0, 0, padded - samples, numOps.Zero), new[] { batch, 1, padded });
        }

        var encoded = _encoder.Forward(waveform);                                  // [B, N, K]
        Record(stages, "Encoder", encoded);
        int frames = encoded.Shape[2];

        var normalized = GlobalLayerNorm(engine, _inputNorm, encoded);
        Record(stages, "EncoderNormalization", normalized);
        var residual = _bottleneck.Forward(normalized);                            // [B, B, K]
        Record(stages, "BottleneckProjection", residual);

        Tensor<T>? skipSum = null;
        foreach (var block in _blocks)
        {
            var (next, skip) = block.Forward(engine, residual);
            skipSum = skipSum is null ? skip : engine.TensorAdd(skipSum, skip);
            if (next is not null) residual = next;
        }

        // Every block emits a skip and the constructor requires at least one block, so the mask head
        // always reads the skip sum, as in the paper. A layout that produced none would otherwise mask
        // from the bottleneck output and still run, silently.
        var separated = skipSum ?? throw new InvalidOperationException("The TCN separator produced no skip output.");
        Record(stages, "TemporalConvolutionalSeparator", separated);

        var masks = engine.Reshape(_maskConv.Forward(_outputPRelu.Forward(separated)),
            new[] { batch, _numSources, _encoderDim, frames });                    // sigmoid in the conv
        Record(stages, "SourceMasks", masks);

        var shape = new[] { batch, _numSources, _encoderDim, frames };
        var masked = engine.TensorMultiply(engine.TensorBroadcastTo(
            engine.Reshape(encoded, new[] { batch, 1, _encoderDim, frames }), shape), masks);
        Record(stages, "MaskedEncoderSources", masked);

        var decoded = _decoder.Forward(engine.Reshape(masked, new[] { batch * _numSources, _encoderDim, frames }));
        int decodedLength = decoded.Shape[decoded.Shape.Length - 1];
        var sources = engine.Reshape(decoded, new[] { batch, _numSources, decodedLength });
        if (decodedLength != samples)
        {
            sources = engine.TensorNarrow(sources, dim: 2, start: 0, length: samples);
        }

        Record(stages, "WaveformDecoder", sources);
        return sources;
    }

    private static void Record(IDictionary<string, Tensor<T>>? stages, string name, Tensor<T> value)
    {
        if (stages is not null) stages[name] = value.Clone();
    }

    /// <summary>gLN over channels and time per example: GroupNorm with one group on [B, C, T, 1].</summary>
    private static Tensor<T> GlobalLayerNorm(IEngine engine, GroupNormalizationLayer<T> norm, Tensor<T> x)
    {
        var image = engine.Reshape(x, new[] { x.Shape[0], x.Shape[1], x.Shape[2], 1 });
        return engine.Reshape(norm.Forward(image), x.Shape.ToArray());
    }

    [System.Diagnostics.CodeAnalysis.MemberNotNull(
        nameof(_encoder), nameof(_inputNorm), nameof(_bottleneck), nameof(_outputPRelu), nameof(_maskConv), nameof(_decoder))]
    private void Build()
    {
        _layers.Clear();
        _blocks.Clear();
        IActivationFunction<T> identity = new IdentityActivation<T>();

        _encoder = Add(new Conv1DLayer<T>(inputChannels: 1, outputChannels: _encoderDim, kernelSize: _kernelSize, dilation: 1, stride: _stride, padding: 0,
            activation: new ReLUActivation<T>()));
        _inputNorm = Add(new GroupNormalizationLayer<T>(1, _encoderDim));
        _bottleneck = Add(new Conv1DLayer<T>(inputChannels: _encoderDim, outputChannels: _bottleneckDim, kernelSize: 1, activation: identity));

        int total = _numRepeats * _numBlocks;
        for (int r = 0; r < _numRepeats; r++)
        {
            for (int x = 0; x < _numBlocks; x++)
            {
                int dilation = 1 << x;
                bool last = r * _numBlocks + x == total - 1;
                _blocks.Add(new TcnBlock(
                    Add(new Conv1DLayer<T>(inputChannels: _bottleneckDim, outputChannels: _hiddenDim, kernelSize: 1, activation: identity)),
                    Add(new PReLULayer<T>(1, channelAxis: 1)),
                    Add(new GroupNormalizationLayer<T>(1, _hiddenDim)),
                    Add(new Conv1DLayer<T>(inputChannels: _hiddenDim, outputChannels: _hiddenDim, kernelSize: _tcnKernelSize, dilation: dilation, stride: 1,
                        padding: dilation * (_tcnKernelSize - 1) / 2, activation: identity, groups: _hiddenDim)),
                    Add(new PReLULayer<T>(1, channelAxis: 1)),
                    Add(new GroupNormalizationLayer<T>(1, _hiddenDim)),
                    last ? null : Add(new Conv1DLayer<T>(inputChannels: _hiddenDim, outputChannels: _bottleneckDim, kernelSize: 1, activation: identity)),
                    Add(new Conv1DLayer<T>(inputChannels: _hiddenDim, outputChannels: _bottleneckDim, kernelSize: 1, activation: identity))));
            }
        }

        _outputPRelu = Add(new PReLULayer<T>(1, channelAxis: 1));
        _maskConv = Add(new Conv1DLayer<T>(inputChannels: _bottleneckDim, outputChannels: _numSources * _encoderDim, kernelSize: 1, activation: new SigmoidActivation<T>()));
        _decoder = Add(new Conv1DTransposeLayer<T>(_encoderDim, 1, _kernelSize, stride: _stride, padding: 0, activation: identity));
    }

    private TLayer Add<TLayer>(TLayer layer) where TLayer : class, ILayer<T>
    {
        if (_bindSource is not null)
        {
            int index = _layers.Count;
            if (index >= _bindSource.Count || _bindSource[index] is not TLayer existing)
            {
                throw new InvalidOperationException(
                    $"The layer graph does not match the Conv-TasNet layout at position {index}: expected " +
                    $"{typeof(TLayer).Name}, found {(index < _bindSource.Count ? _bindSource[index]?.GetType().Name ?? "null" : "the end")}.");
            }

            layer = existing;
        }

        _layers.Add(layer);
        return layer;
    }

    /// <summary>One 1-D convolutional block of the TCN (Fig. 1C).</summary>
    private sealed class TcnBlock
    {
        private readonly Conv1DLayer<T> _input;
        private readonly PReLULayer<T> _prelu1;
        private readonly GroupNormalizationLayer<T> _norm1;
        private readonly Conv1DLayer<T> _depthwise;
        private readonly PReLULayer<T> _prelu2;
        private readonly GroupNormalizationLayer<T> _norm2;
        private readonly Conv1DLayer<T>? _residual;
        private readonly Conv1DLayer<T> _skip;

        public TcnBlock(Conv1DLayer<T> input, PReLULayer<T> prelu1, GroupNormalizationLayer<T> norm1,
            Conv1DLayer<T> depthwise, PReLULayer<T> prelu2, GroupNormalizationLayer<T> norm2,
            Conv1DLayer<T>? residual, Conv1DLayer<T> skip)
        {
            _input = input;
            _prelu1 = prelu1;
            _norm1 = norm1;
            _depthwise = depthwise;
            _prelu2 = prelu2;
            _norm2 = norm2;
            _residual = residual;
            _skip = skip;
        }

        /// <summary>The next block's input (null for the last block) and this block's skip output.</summary>
        public (Tensor<T>? Next, Tensor<T> Skip) Forward(IEngine engine, Tensor<T> x)
        {
            var h = GlobalLayerNorm(engine, _norm1, _prelu1.Forward(_input.Forward(x)));
            h = GlobalLayerNorm(engine, _norm2, _prelu2.Forward(_depthwise.Forward(h)));
            var next = _residual is null ? null : engine.TensorAdd(x, _residual.Forward(h));
            return (next, _skip.Forward(h));
        }
    }
}