using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.Audio.Codecs;

/// <summary>
/// The shape of a SNAC codec (Siuzdak et al. 2024, "SNAC: Multi-Scale Neural Audio Codec"; reference <c>snac.SNAC</c>).
/// The defaults are the released 24 kHz speech model (<c>hubertsiuzdak/snac_24khz</c>), which has no local attention.
/// </summary>
internal sealed class SnacConfiguration
{
    /// <summary>The sampling rate (24000).</summary>
    public int SampleRate { get; init; } = 24000;

    /// <summary>The encoder's first width (48), doubled by each downsampling block.</summary>
    public int EncoderDim { get; init; } = 48;

    /// <summary>The encoder's strides (2, 4, 8, 8): a hop of 512 samples.</summary>
    public int[] EncoderRates { get; init; } = { 2, 4, 8, 8 };

    /// <summary>The decoder's first width (1024), halved by each upsampling block.</summary>
    public int DecoderDim { get; init; } = 1024;

    /// <summary>The decoder's upsampling rates (8, 8, 4, 2).</summary>
    public int[] DecoderRates { get; init; } = { 8, 8, 4, 2 };

    /// <summary>Codes per codebook (4096).</summary>
    public int CodebookSize { get; init; } = 4096;

    /// <summary>The factorized code dimension (8).</summary>
    public int CodebookDim { get; init; } = 8;

    /// <summary>Each codebook's temporal stride (4, 2, 1): the coarse codes cover 4 latent frames each.</summary>
    public int[] VqStrides { get; init; } = { 4, 2, 1 };

    /// <summary>Whether each decoder block adds input-dependent noise (true).</summary>
    public bool Noise { get; init; } = true;

    /// <summary>Whether the residual units' dilated convolutions are depthwise (true).</summary>
    public bool Depthwise { get; init; } = true;

    /// <summary>The latent width: the encoder's last width, <c>encoder_dim · 2^len(encoder_rates)</c>.</summary>
    public int LatentDim => EncoderDim << EncoderRates.Length;

    /// <summary>The hop in samples (the product of the encoder strides).</summary>
    public int HopLength => EncoderRates.Aggregate(1, (a, b) => a * b);
}

/// <summary>
/// SNAC's residual unit (reference <c>ResidualUnit</c>): Snake → a dilated k = 7 weight-normalized convolution (depthwise in
/// the released models) → Snake → a 1×1 convolution, added to the input.
/// </summary>
internal sealed class SnacResidualUnit<T>
{
    private readonly IEngine _engine;
    private readonly SnakeLayer<T> _snake1, _snake2;
    private readonly NormedConv1DLayer<T> _conv1, _conv2;

    public SnacResidualUnit(IEngine engine, int dim, int dilation, int groups, List<LayerBase<T>> owner)
    {
        _engine = engine;
        _snake1 = Snac.Add(owner, new SnakeLayer<T>(dim));
        _conv1 = Snac.Conv(owner, dim, dim, 7, 1, dilation, groups, 3 * dilation);
        _snake2 = Snac.Add(owner, new SnakeLayer<T>(dim));
        _conv2 = Snac.Conv(owner, dim, dim, 1, 1, 1, 1, 0);
    }

    internal IEnumerable<(string Name, LayerBase<T> Layer)> Named(string prefix)
    {
        yield return ($"{prefix}.block.0", _snake1);
        yield return ($"{prefix}.block.1", _conv1);
        yield return ($"{prefix}.block.2", _snake2);
        yield return ($"{prefix}.block.3", _conv2);
    }

    public Tensor<T> Forward(Tensor<T> x)
    {
        var y = _conv2.Forward(_snake2.Forward(_conv1.Forward(_snake1.Forward(x))));
        int pad = (x.Shape[2] - y.Shape[2]) / 2;
        if (pad > 0) x = _engine.TensorSlice(x, new[] { 0, 0, pad }, new[] { x.Shape[0], x.Shape[1], y.Shape[2] });
        return _engine.TensorAdd(x, y);
    }
}

/// <summary>
/// SNAC's encoder (reference <c>Encoder</c>): a k = 7 convolution to <c>encoder_dim</c>, then per stride three residual
/// units (dilations 1, 3, 9; depthwise) at the current width, Snake and a strided convolution (kernel 2s, padding ⌈s/2⌉)
/// doubling it, then a k = 7 convolution (depthwise) at the latent width.
/// </summary>
internal sealed class SnacEncoder<T>
{
    private readonly NormedConv1DLayer<T> _first, _last;
    private readonly List<(SnacResidualUnit<T>[] Units, SnakeLayer<T> Snake, NormedConv1DLayer<T> Down)> _blocks = new();

    public SnacEncoder(IEngine engine, SnacConfiguration c, List<LayerBase<T>> owner)
    {
        int d = c.EncoderDim;
        _first = Snac.Conv(owner, 1, d, 7, 1, 1, 1, 3);
        foreach (int s in c.EncoderRates)
        {
            int groups = c.Depthwise ? d : 1;
            var units = new[] { 1, 3, 9 }.Select(dilation => new SnacResidualUnit<T>(engine, d, dilation, groups, owner)).ToArray();
            var snake = Snac.Add(owner, new SnakeLayer<T>(d));
            var down = Snac.Conv(owner, d, 2 * d, 2 * s, s, 1, 1, (s + 1) / 2);
            _blocks.Add((units, snake, down));
            d *= 2;
        }
        _last = Snac.Conv(owner, d, d, 7, 1, 1, c.Depthwise ? d : 1, 3);
    }

    /// <summary>The modules under the checkpoint's names (<c>encoder.block.{i}</c>).</summary>
    internal IEnumerable<(string Name, LayerBase<T> Layer)> Named(string prefix)
    {
        yield return ($"{prefix}.block.0", _first);
        for (int i = 0; i < _blocks.Count; i++)
        {
            var (units, snake, down) = _blocks[i];
            for (int u = 0; u < units.Length; u++)
                foreach (var named in units[u].Named($"{prefix}.block.{i + 1}.block.{u}")) yield return named;
            yield return ($"{prefix}.block.{i + 1}.block.3", snake);
            yield return ($"{prefix}.block.{i + 1}.block.4", down);
        }
        yield return ($"{prefix}.block.{_blocks.Count + 1}", _last);
    }

    public Tensor<T> Forward(Tensor<T> x)
    {
        x = _first.Forward(x);
        foreach (var (units, snake, down) in _blocks)
        {
            foreach (var unit in units) x = unit.Forward(x);
            x = down.Forward(snake.Forward(x));
        }
        return _last.Forward(x);
    }
}

/// <summary>
/// SNAC's decoder (reference <c>Decoder</c>): a depthwise k = 7 convolution and a 1×1 convolution to <c>decoder_dim</c>,
/// then per rate Snake, a transposed convolution (kernel 2s, padding ⌈s/2⌉, output padding s mod 2) halving the width, a
/// noise block and three residual units (dilations 1, 3, 9; depthwise), then Snake, a k = 7 convolution to one channel and
/// tanh.
/// </summary>
/// <remarks>The noise block adds <c>noise · W x</c> with <c>noise ~ N(0, 1)</c> of shape <c>[B, 1, T]</c>, drawn anew on every
/// decode — in inference too, as the reference does.</remarks>
internal sealed class SnacDecoder<T>
{
    private readonly IEngine _engine;
    private readonly List<NormedConv1DLayer<T>> _input = new();
    private readonly List<(SnakeLayer<T> Snake, NormedConv1DLayer<T> Up, NormedConv1DLayer<T>? Noise, SnacResidualUnit<T>[] Units)> _blocks = new();
    private readonly SnakeLayer<T> _snake;
    private readonly NormedConv1DLayer<T> _last;

    public SnacDecoder(IEngine engine, SnacConfiguration c, List<LayerBase<T>> owner)
    {
        _engine = engine;
        int latent = c.LatentDim, channels = c.DecoderDim;
        if (c.Depthwise)
        {
            _input.Add(Snac.Conv(owner, latent, latent, 7, 1, 1, latent, 3));
            _input.Add(Snac.Conv(owner, latent, channels, 1, 1, 1, 1, 0));
        }
        else
        {
            _input.Add(Snac.Conv(owner, latent, channels, 7, 1, 1, 1, 3));
        }
        int output = channels;
        for (int i = 0; i < c.DecoderRates.Length; i++)
        {
            int input = channels >> i, s = c.DecoderRates[i];
            output = channels >> (i + 1);
            var snake = Snac.Add(owner, new SnakeLayer<T>(input));
            var up = Snac.Add(owner, new NormedConv1DLayer<T>(input, output, 2 * s, s, 1, 1, (s + 1) / 2, true, ConvolutionNormalization.Weight,
                outputPadding: s % 2));
            var noise = c.Noise ? Snac.Add(owner, new NormedConv1DLayer<T>(output, output, 1, 1, 1, 1, 0, false, ConvolutionNormalization.Weight, useBias: false)) : null;
            int groups = c.Depthwise ? output : 1;
            var units = new[] { 1, 3, 9 }.Select(dilation => new SnacResidualUnit<T>(engine, output, dilation, groups, owner)).ToArray();
            _blocks.Add((snake, up, noise, units));
        }
        _snake = Snac.Add(owner, new SnakeLayer<T>(output));
        _last = Snac.Conv(owner, output, 1, 7, 1, 1, 1, 3);
    }

    /// <summary>The modules under the checkpoint's names (<c>decoder.model.{i}</c>).</summary>
    internal IEnumerable<(string Name, LayerBase<T> Layer)> Named(string prefix)
    {
        for (int i = 0; i < _input.Count; i++) yield return ($"{prefix}.model.{i}", _input[i]);
        for (int b = 0; b < _blocks.Count; b++)
        {
            var (snake, up, noise, units) = _blocks[b];
            string block = $"{prefix}.model.{_input.Count + b}.block";
            yield return ($"{block}.0", snake);
            yield return ($"{block}.1", up);
            int next = 2;
            if (noise is not null) yield return ($"{block}.{next++}.linear", noise);
            for (int u = 0; u < units.Length; u++)
                foreach (var named in units[u].Named($"{block}.{next + u}")) yield return named;
        }
        yield return ($"{prefix}.model.{_input.Count + _blocks.Count}", _snake);
        yield return ($"{prefix}.model.{_input.Count + _blocks.Count + 1}", _last);
    }

    /// <summary>The waveform <c>[1, 1, samples]</c> of a latent <c>[1, latent, frames]</c>; <paramref name="noise"/> gives each
    /// noise block's draw of the given length, in order.</summary>
    public Tensor<T> Forward(Tensor<T> z, Func<int, double[]> noise)
    {
        var x = z;
        foreach (var conv in _input) x = conv.Forward(x);
        foreach (var (snake, up, noiseConv, units) in _blocks)
        {
            x = up.Forward(snake.Forward(x));
            if (noiseConv is not null)
            {
                int length = x.Shape[2];
                var draw = noise(length);
                if (draw.Length != length) throw new InvalidOperationException($"A noise draw has {draw.Length} values; the block needs {length}.");
                var values = new Tensor<T>(new[] { 1, 1, length }, new Vector<T>(draw.Select(v => MathHelper.GetNumericOperations<T>().FromDouble(v)).ToArray()));
                var h = noiseConv.Forward(x);
                x = _engine.TensorAdd(x, _engine.TensorMultiply(_engine.TensorBroadcastTo(values, h._shape), h));
            }
            foreach (var unit in units) x = unit.Forward(x);
        }
        return _engine.Tanh(_last.Forward(_snake.Forward(x)));
    }
}

/// <summary>
/// SNAC: the encoder, a residual vector quantizer whose codebooks run at different temporal resolutions (each a DAC-style
/// factorized, L2-normalized codebook applied to the residual average-pooled by its stride, its codes repeated back to
/// full rate), and the decoder. The released 24 kHz model makes 12, 23 and 47 codes per second.
/// </summary>
internal sealed class Snac<T>
{
    private readonly IEngine _engine;
    private readonly SnacEncoder<T> _encoder;
    private readonly FactorizedVectorQuantizerLayer<T>[] _quantizers;
    private readonly SnacDecoder<T> _decoder;
    private readonly List<LayerBase<T>> _layers;
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    public Snac(IEngine engine, SnacConfiguration configuration, List<LayerBase<T>> owner)
    {
        _engine = engine;
        Configuration = configuration ?? throw new ArgumentNullException(nameof(configuration));
        _layers = owner;
        _encoder = new SnacEncoder<T>(engine, configuration, owner);
        _quantizers = configuration.VqStrides
            .Select(_ => Snac.Add(owner, new FactorizedVectorQuantizerLayer<T>(configuration.LatentDim, 1, configuration.CodebookSize, configuration.CodebookDim)))
            .ToArray();
        _decoder = new SnacDecoder<T>(engine, configuration, owner);
    }

    public SnacConfiguration Configuration { get; }

    /// <summary>Samples per coarsest code (the hop times the first stride).</summary>
    public int SamplesPerFrame => Configuration.HopLength * Configuration.VqStrides[0];

    /// <summary>The codes of a waveform, one row per codebook (coarse to fine). The audio is right-padded with zeros to
    /// whole <see cref="SamplesPerFrame"/> windows first (<c>SNAC.preprocess</c>).</summary>
    public int[][] Encode(double[] audio)
    {
        if (audio is null) throw new ArgumentNullException(nameof(audio));
        using var _ = new NoGradScope<T>();
        foreach (var layer in _layers) layer.SetTrainingMode(false);
        int window = SamplesPerFrame;
        int padded = Math.Max(1, (audio.Length + window - 1) / window) * window;
        var input = new Tensor<T>(new[] { 1, 1, padded });
        for (int i = 0; i < audio.Length; i++) input[0, 0, i] = Ops.FromDouble(audio[i]);
        var residual = _encoder.Forward(input);
        var codes = new int[_quantizers.Length][];
        for (int q = 0; q < _quantizers.Length; q++)
        {
            int stride = Configuration.VqStrides[q];
            var row = _quantizers[q].Encode(Pool(residual, stride));
            codes[q] = Enumerable.Range(0, row.GetLength(1)).Select(t => row[0, t]).ToArray();
            residual = _engine.TensorSubtract(residual, Repeat(_quantizers[q].Decode(row), stride));
        }
        return codes;
    }

    /// <summary>The latent of codes (<c>ResidualVectorQuantize.from_codes</c>).</summary>
    public Tensor<T> FromCodes(IReadOnlyList<int[]> codes)
    {
        if (codes is null || codes.Count != _quantizers.Length)
            throw new ArgumentException($"SNAC takes {_quantizers.Length} rows of codes.", nameof(codes));
        int frames = codes[0].Length;
        if (frames == 0) throw new ArgumentException("There are no codes.", nameof(codes));
        Tensor<T>? latent = null;
        for (int q = 0; q < codes.Count; q++)
        {
            int stride = Configuration.VqStrides[q];
            if (codes[q].Length * stride != frames * Configuration.VqStrides[0])
                throw new ArgumentException($"Codebook {q} has {codes[q].Length} codes; {frames} coarse codes need {frames * Configuration.VqStrides[0] / stride}.", nameof(codes));
            var row = new int[1, codes[q].Length];
            for (int t = 0; t < codes[q].Length; t++)
            {
                int code = codes[q][t];
                if (code < 0 || code >= Configuration.CodebookSize)
                    throw new ArgumentOutOfRangeException(nameof(codes), code, $"Codes lie in [0, {Configuration.CodebookSize}).");
                row[0, t] = code;
            }
            var part = Repeat(_quantizers[q].Decode(row), stride);
            latent = latent is null ? part : _engine.TensorAdd(latent, part);
        }
        return latent!;
    }

    /// <summary>The waveform of codes (<c>SNAC.decode</c>), one row per codebook; <paramref name="noise"/> supplies each noise
    /// block's N(0, 1) draw of the given length.</summary>
    public double[] Decode(IReadOnlyList<int[]> codes, Func<int, double[]> noise)
    {
        using var _ = new NoGradScope<T>();
        foreach (var layer in _layers) layer.SetTrainingMode(false);
        var audio = _decoder.Forward(FromCodes(codes), noise);
        var result = new double[audio.Shape[2]];
        for (int i = 0; i < result.Length; i++) result[i] = Ops.ToDouble(audio[0, 0, i]);
        return result;
    }

    /// <summary>A noise source of standard normal draws from <paramref name="random"/> (Box–Muller).</summary>
    public static Func<int, double[]> GaussianNoise(Random random) => length =>
    {
        var values = new double[length];
        for (int i = 0; i < length; i++)
        {
            double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
            values[i] = Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
        }
        return values;
    };

    // avg_pool1d(z, s, s) over [1, D, T] (T a multiple of s).
    private Tensor<T> Pool(Tensor<T> z, int stride)
    {
        if (stride == 1) return z;
        int d = z.Shape[1], t = z.Shape[2];
        if (t % stride != 0) throw new ArgumentException($"{t} latent frames do not split into windows of {stride}.");
        return _engine.ReduceMean(_engine.Reshape(z, new[] { 1, d, t / stride, stride }), new[] { 3 }, keepDims: false);
    }

    // repeat_interleave(stride, dim=-1) over [1, D, T].
    private Tensor<T> Repeat(Tensor<T> z, int stride)
    {
        return stride == 1 ? z : _engine.TensorRepeatInterleave(z, stride, dim: 2);
    }

    /// <summary>Every module under the checkpoint's names.</summary>
    internal IEnumerable<(string Name, LayerBase<T> Layer)> Named()
    {
        foreach (var named in _encoder.Named("encoder")) yield return named;
        for (int q = 0; q < _quantizers.Length; q++)
        {
            yield return ($"quantizer.quantizers.{q}.in_proj", _quantizers[q].InProjection(0));
            yield return ($"quantizer.quantizers.{q}.out_proj", _quantizers[q].OutProjection(0));
        }
        foreach (var named in _decoder.Named("decoder")) yield return named;
    }

    /// <summary>
    /// Loads a SNAC state dict (the released <c>pytorch_model.bin</c>): weight-normalized convolutions as
    /// <c>parametrizations.weight.original0</c> (gain) and <c>original1</c> (direction) or the older <c>weight_g</c> and
    /// <c>weight_v</c>, Snake <c>alpha</c>, and each codebook's <c>codebook.weight</c>. Returns the names it read.
    /// </summary>
    public HashSet<string> Load(Func<string, double[]?> read)
    {
        var used = new HashSet<string>(StringComparer.Ordinal);
        double[]? Read(string name, int count, bool optional = false)
        {
            var values = read(name);
            if (values is null)
            {
                if (optional) return null;
                throw new InvalidDataException($"The checkpoint has no tensor '{name}'.");
            }
            if (values.Length != count) throw new InvalidDataException($"'{name}' has {values.Length} values; this model expects {count}.");
            used.Add(name);
            return values;
        }
        foreach (var (name, layer) in Named())
        {
            switch (layer)
            {
                case NormedConv1DLayer<T> conv:
                    var shape = conv.TorchWeightShape;
                    int count = shape.Aggregate(1, (a, b) => a * b);
                    var direction = Read($"{name}.parametrizations.weight.original1", count, optional: true);
                    var gain = direction is null ? Read($"{name}.weight_g", shape[0])! : Read($"{name}.parametrizations.weight.original0", shape[0])!;
                    direction ??= Read($"{name}.weight_v", count)!;
                    var bias = Read($"{name}.bias", conv.OutputChannels, optional: true) ?? new double[conv.OutputChannels];
                    conv.LoadTorchWeights(direction, gain, bias);
                    break;
                case SnakeLayer<T> snake:
                    snake.LoadAlpha(Read($"{name}.alpha", snake.Channels)!);
                    break;
                default:
                    throw new InvalidOperationException($"No loader for {layer.GetType().Name} at '{name}'.");
            }
        }
        for (int q = 0; q < _quantizers.Length; q++)
        {
            var codebook = Read($"quantizer.quantizers.{q}.codebook.weight", Configuration.CodebookSize * Configuration.CodebookDim)!;
            var target = _quantizers[q].Codebook(0);
            for (int i = 0; i < codebook.Length; i++) target[i] = Ops.FromDouble(codebook[i]);
            _engine.InvalidatePersistentTensor(target);
        }
        return used;
    }
}

/// <summary>Construction helpers for SNAC's modules.</summary>
internal static class Snac
{
    /// <summary>A weight-normalized convolution (reference <c>WNConv1d</c>) with PyTorch's default initialization (SNAC has
    /// no custom init).</summary>
    public static NormedConv1DLayer<T> Conv<T>(List<LayerBase<T>> owner, int input, int output, int kernel, int stride, int dilation, int groups,
        int padding) =>
        Add(owner, new NormedConv1DLayer<T>(input, output, kernel, stride, dilation, groups, padding, false, ConvolutionNormalization.Weight));

    public static TLayer Add<T, TLayer>(List<LayerBase<T>> owner, TLayer layer) where TLayer : LayerBase<T>
    {
        owner.Add(layer);
        return layer;
    }
}
