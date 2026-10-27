using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.Vocoders;

namespace AiDotNet.Audio.Codecs;

/// <summary>
/// The shape of Mimi (Défossez et al. 2024, "Moshi: a speech-text foundation model for real-time dialogue", §3.3; reference
/// <c>moshi</c> 0.2.2 <c>models/loaders.py</c>). The defaults are the released codec
/// (<c>kyutai/moshiko-pytorch-bf16</c>, <c>tokenizer-e351c8d8-checkpoint125.safetensors</c>): a causal SEANet at 24 kHz with
/// strides (4, 5, 6, 8), an 8-layer causal Transformer on each side of the bottleneck, a learnt stride-2 resampling from
/// 25 Hz to 12.5 Hz, and a split residual quantizer of one semantic and 31 acoustic codebooks of 2048 entries.
/// </summary>
internal sealed class MimiConfiguration
{
    /// <summary>The sampling rate (24000).</summary>
    public int SampleRate { get; init; } = 24000;

    /// <summary>The latent width shared by SEANet, the Transformers and the quantizer's projections (512).</summary>
    public int Dimension { get; init; } = 512;

    /// <summary>SEANet's base width (64).</summary>
    public int Filters { get; init; } = 64;

    /// <summary>SEANet's ratios as the reference lists them (8, 6, 5, 4; the encoder applies them in reverse).</summary>
    public int[] Ratios { get; init; } = { 8, 6, 5, 4 };

    /// <summary>Transformer layers on each side (8).</summary>
    public int TransformerLayers { get; init; } = 8;

    /// <summary>Attention heads (8).</summary>
    public int Heads { get; init; } = 8;

    /// <summary>The feed-forward width (2048).</summary>
    public int FeedForward { get; init; } = 2048;

    /// <summary>How many past frames (the current one included) each frame attends to (250).</summary>
    public int Context { get; init; } = 250;

    /// <summary>The layer scales' initial value (0.01).</summary>
    public double LayerScale { get; init; } = 0.01;

    /// <summary>The rotary embedding's maximum period (10000).</summary>
    public double MaxPeriod { get; init; } = 10000;

    /// <summary>The ratio of SEANet's frame rate to the codes' (2: 25 Hz to 12.5 Hz).</summary>
    public int ResampleStride { get; init; } = 2;

    /// <summary>All codebooks (32): <see cref="SemanticCodebooks"/> then the acoustic ones.</summary>
    public int Codebooks { get; init; } = 32;

    /// <summary>The semantic codebooks, quantized apart from the acoustic ones (1).</summary>
    public int SemanticCodebooks { get; init; } = 1;

    /// <summary>Entries per codebook (2048).</summary>
    public int CodebookSize { get; init; } = 2048;

    /// <summary>The quantizer's inner width (256).</summary>
    public int CodebookDim { get; init; } = 256;

    /// <summary>The floor on a cluster's usage when dividing its running sum (1e-5).</summary>
    public double CodebookEpsilon { get; init; } = 1e-5;

    /// <summary>Samples per SEANet frame (the product of the ratios: 1920).</summary>
    public int HopLength => Ratios.Aggregate(1, (a, b) => a * b);

    /// <summary>Samples per code frame (3840 with the defaults).</summary>
    public int SamplesPerFrame => HopLength * ResampleStride;

    internal SeanetConfig Seanet() => new()
    {
        Channels = 1,
        Dimension = Dimension,
        Filters = Filters,
        ResidualLayers = 1,
        Ratios = Ratios,
        EluAlpha = 1.0,
        Normalization = ConvolutionNormalization.None,
        KernelSize = 7,
        LastKernelSize = 3,
        DilationBase = 2,
        ResidualKernelSizes = new[] { 3, 1 },
        Causal = true,
        Reflect = false,
        TrueSkip = true,
        Compress = 2,
        LstmLayers = 0,
        TrimRightRatio = 1.0,
    };
}

/// <summary>
/// One layer of Mimi's Transformer (reference <c>StreamingTransformerLayer</c> with <c>norm="layer_norm"</c>,
/// <c>gating="none"</c>): x + γ₁ ⊙ attention(LayerNorm(x)), then x + γ₂ ⊙ W₂ GELU(W₁ LayerNorm(x)). Attention takes its
/// queries, keys and values from one bias-free projection laid out (part, head, dim), rotates queries and keys by the
/// interleaved rotary embedding (adjacent dimensions form each pair) and sees the last <c>context</c> frames causally.
/// </summary>
internal sealed class MimiTransformerLayer<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly int _dim, _heads;

    public MimiTransformerLayer(IEngine engine, MimiConfiguration c, List<LayerBase<T>> owner)
    {
        _engine = engine;
        _dim = c.Dimension;
        _heads = c.Heads;
        if (_dim % _heads != 0 || (_dim / _heads) % 2 != 0)
            throw new ArgumentException($"The width {_dim} must split into {_heads} heads of an even size.");
        owner.Add(Norm1 = new LayerNormalizationLayer<T>(_dim, 1e-5));
        owner.Add(InProjection = new BiasFreeLinearLayer<T>(_dim, 3 * _dim));
        owner.Add(OutProjection = new BiasFreeLinearLayer<T>(_dim, _dim));
        owner.Add(Scale1 = new ChannelScaleLayer<T>(_dim, c.LayerScale));
        owner.Add(Norm2 = new LayerNormalizationLayer<T>(_dim, 1e-5));
        owner.Add(Linear1 = new BiasFreeLinearLayer<T>(_dim, c.FeedForward));
        owner.Add(Linear2 = new BiasFreeLinearLayer<T>(c.FeedForward, _dim));
        owner.Add(Scale2 = new ChannelScaleLayer<T>(_dim, c.LayerScale));
    }

    public LayerNormalizationLayer<T> Norm1 { get; }
    public BiasFreeLinearLayer<T> InProjection { get; }
    public BiasFreeLinearLayer<T> OutProjection { get; }
    public ChannelScaleLayer<T> Scale1 { get; }
    public LayerNormalizationLayer<T> Norm2 { get; }
    public BiasFreeLinearLayer<T> Linear1 { get; }
    public BiasFreeLinearLayer<T> Linear2 { get; }
    public ChannelScaleLayer<T> Scale2 { get; }

    private Tensor<T> Scaled(ChannelScaleLayer<T> scale, Tensor<T> x) =>
        _engine.TensorMultiply(x, _engine.TensorTile(_engine.Reshape(scale.Scale, new[] { 1, _dim }), new[] { x.Shape[0], 1 }));

    // (x₀, x₁) pairs of adjacent dimensions rotated by the angle of their pair: x ⊙ cos + rot(x) ⊙ sin, rot(x₀, x₁) = (−x₁, x₀).
    private Tensor<T> Rotate(Tensor<T> x, Tensor<T> cos, Tensor<T> sin)
    {
        int heads = x.Shape[0], l = x.Shape[1], hd = x.Shape[2], half = hd / 2;
        var pairs = _engine.Reshape(x, new[] { heads, l, half, 2 });
        var even = _engine.TensorSlice(pairs, new[] { 0, 0, 0, 0 }, new[] { heads, l, half, 1 });
        var odd = _engine.TensorSlice(pairs, new[] { 0, 0, 0, 1 }, new[] { heads, l, half, 1 });
        var rotated = _engine.Reshape(_engine.TensorConcatenate(new[] { _engine.TensorNegate(odd), even }, 3), new[] { heads, l, hd });
        var shape = new[] { heads, l, hd };
        return _engine.TensorAdd(_engine.TensorMultiply(x, _engine.TensorBroadcastTo(cos, shape)), _engine.TensorMultiply(rotated, _engine.TensorBroadcastTo(sin, shape)));
    }

    /// <summary>The layer's output <c>[frames, dim]</c>; <paramref name="cos"/>, <paramref name="sin"/> are
    /// <c>[1, frames, head dim]</c> and <paramref name="bias"/> the additive causal-window mask <c>[1, frames, frames]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> x, Tensor<T> cos, Tensor<T> sin, Tensor<T> bias)
    {
        int l = x.Shape[0], hd = _dim / _heads;
        var qkv = InProjection.Forward(Norm1.Forward(x));
        Tensor<T> Part(int part) => _engine.TensorPermute(
            _engine.Reshape(_engine.TensorSlice(qkv, new[] { 0, part * _dim }, new[] { l, _dim }), new[] { l, _heads, hd }), new[] { 1, 0, 2 });
        var q = Rotate(Part(0), cos, sin);
        var k = Rotate(Part(1), cos, sin);
        var v = Part(2);
        var scores = _engine.TensorMultiplyScalar(_engine.BatchMatMul(q, _engine.TensorPermute(k, new[] { 0, 2, 1 })), NumOps.FromDouble(1.0 / Math.Sqrt(hd)));
        scores = _engine.TensorAdd(scores, _engine.TensorBroadcastTo(bias, new[] { _heads, l, l }));
        var context = _engine.BatchMatMul(_engine.Softmax(scores, axis: 2), v);
        var attended = OutProjection.Forward(_engine.Reshape(_engine.TensorPermute(context, new[] { 1, 0, 2 }), new[] { l, _dim }));
        x = _engine.TensorAdd(x, Scaled(Scale1, attended));
        var update = Linear2.Forward(ActivationHelper.ExactGelu(Linear1.Forward(Norm2.Forward(x)), _engine));
        return _engine.TensorAdd(x, Scaled(Scale2, update));
    }
}

/// <summary>
/// Mimi's Transformer (reference <c>ProjectedTransformer</c> over a <c>StreamingTransformer</c> with rotary positions, no
/// projections since its width equals SEANet's): <c>[1, dim, frames]</c> in and out, positions counted from 0.
/// </summary>
internal sealed class MimiTransformer<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly MimiConfiguration _c;

    public MimiTransformer(IEngine engine, MimiConfiguration c, List<LayerBase<T>> owner)
    {
        _engine = engine;
        _c = c;
        Layers = Enumerable.Range(0, c.TransformerLayers).Select(_ => new MimiTransformerLayer<T>(engine, c, owner)).ToArray();
    }

    public MimiTransformerLayer<T>[] Layers { get; }

    public Tensor<T> Forward(Tensor<T> x)
    {
        int l = x.Shape[2], hd = _c.Dimension / _c.Heads, half = hd / 2;
        var cos = new Tensor<T>(new[] { 1, l, hd });
        var sin = new Tensor<T>(new[] { 1, l, hd });
        for (int k = 0; k < half; k++)
        {
            double frequency = Math.Exp(k * (-Math.Log(_c.MaxPeriod) * 2 / hd));
            for (int j = 0; j < l; j++)
            {
                double angle = j * frequency;
                cos[0, j, 2 * k] = cos[0, j, 2 * k + 1] = NumOps.FromDouble(Math.Cos(angle));
                sin[0, j, 2 * k] = sin[0, j, 2 * k + 1] = NumOps.FromDouble(Math.Sin(angle));
            }
        }
        var bias = new Tensor<T>(new[] { 1, l, l });
        var blocked = NumOps.FromDouble(double.NegativeInfinity);
        for (int q = 0; q < l; q++)
            for (int k = 0; k < l; k++)
            {
                int delta = q - k;
                if (delta < 0 || delta >= _c.Context) bias[0, q, k] = blocked;
            }
        var h = _engine.TensorPermute(_engine.Reshape(x, new[] { x.Shape[1], l }), new[] { 1, 0 });
        foreach (var layer in Layers) h = layer.Forward(h, cos, sin, bias);
        return _engine.Reshape(_engine.TensorPermute(h, new[] { 1, 0 }), new[] { 1, _c.Dimension, l });
    }
}

/// <summary>
/// A codebook of Mimi's quantizer (reference <c>EuclideanCodebook</c>): the entries are the running sums over the
/// clusters' usage, <c>embedding_sum / max(cluster_usage, ε)</c>, and a vector's code is its nearest entry in Euclidean
/// distance.
/// </summary>
internal sealed class MimiCodebook<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly int _size, _dim;

    public MimiCodebook(IEngine engine, int size, int dim)
    {
        _engine = engine;
        _size = size;
        _dim = dim;
        Entries = new Tensor<T>(new[] { size, dim });
    }

    /// <summary>The entries <c>[size, dim]</c>.</summary>
    public Tensor<T> Entries { get; }

    /// <summary>Sets the entries from the running sums and usage (<c>embedding_sum</c> <c>[size, dim]</c>,
    /// <c>cluster_usage</c> <c>[size]</c>).</summary>
    public void Load(double[] sum, double[] usage, double epsilon)
    {
        for (int i = 0; i < _size; i++)
        {
            double count = Math.Max(usage[i], epsilon);
            for (int d = 0; d < _dim; d++) Entries[i, d] = NumOps.FromDouble(sum[i * _dim + d] / count);
        }
        _engine.InvalidatePersistentTensor(Entries);
    }

    /// <summary>The nearest entry's index for each row of <paramref name="x"/> <c>[frames, dim]</c>.</summary>
    public int[] Encode(Tensor<T> x)
    {
        int frames = x.Shape[0];
        var codes = new int[frames];
        var entryNorms = new double[_size];
        for (int i = 0; i < _size; i++)
        {
            double s = 0;
            for (int d = 0; d < _dim; d++)
            {
                double e = NumOps.ToDouble(Entries[i, d]);
                s += e * e;
            }
            entryNorms[i] = s;
        }
        var dots = _engine.TensorMatMul(x, _engine.TensorPermute(Entries, new[] { 1, 0 }));
        for (int t = 0; t < frames; t++)
        {
            double xs = 0;
            for (int d = 0; d < _dim; d++)
            {
                double v = NumOps.ToDouble(x[t, d]);
                xs += v * v;
            }
            double best = double.PositiveInfinity;
            int arg = 0;
            for (int i = 0; i < _size; i++)
            {
                double distance = xs - 2 * NumOps.ToDouble(dots[t, i]) + entryNorms[i];
                if (distance < best)
                {
                    best = distance;
                    arg = i;
                }
            }
            codes[t] = arg;
        }
        return codes;
    }

    /// <summary>The entries of <paramref name="codes"/> as <c>[frames, dim]</c>.</summary>
    public Tensor<T> Decode(int[] codes)
    {
        var result = new Tensor<T>(new[] { codes.Length, _dim });
        for (int t = 0; t < codes.Length; t++)
        {
            int code = codes[t];
            if (code < 0 || code >= _size) throw new ArgumentOutOfRangeException(nameof(codes), code, $"Codes lie in [0, {_size}).");
            for (int d = 0; d < _dim; d++) result[t, d] = Entries[code, d];
        }
        return result;
    }
}

/// <summary>
/// A residual vector quantizer of Mimi (reference <c>ResidualVectorQuantizer</c>): a bias-free 1×1 projection to the
/// codebook width, then each codebook quantizes what the previous ones left, and the summed entries are projected back.
/// </summary>
internal sealed class MimiResidualQuantizer<T>
{
    private readonly IEngine _engine;

    public MimiResidualQuantizer(IEngine engine, MimiConfiguration c, int codebooks, List<LayerBase<T>> owner)
    {
        _engine = engine;
        owner.Add(InputProjection = new NormedConv1DLayer<T>(c.Dimension, c.CodebookDim, 1, 1, 1, 1, 0, false, ConvolutionNormalization.None, useBias: false));
        owner.Add(OutputProjection = new NormedConv1DLayer<T>(c.CodebookDim, c.Dimension, 1, 1, 1, 1, 0, false, ConvolutionNormalization.None, useBias: false));
        Codebooks = Enumerable.Range(0, codebooks).Select(_ => new MimiCodebook<T>(engine, c.CodebookSize, c.CodebookDim)).ToArray();
    }

    public NormedConv1DLayer<T> InputProjection { get; }
    public NormedConv1DLayer<T> OutputProjection { get; }
    public MimiCodebook<T>[] Codebooks { get; }

    /// <summary>The codes <c>[codebooks][frames]</c> of a latent <c>[1, dim, frames]</c>, using the first
    /// <paramref name="count"/> codebooks.</summary>
    public int[][] Encode(Tensor<T> latent, int count)
    {
        var projected = InputProjection.Forward(latent);
        int frames = projected.Shape[2], dim = projected.Shape[1];
        var residual = _engine.TensorPermute(_engine.Reshape(projected, new[] { dim, frames }), new[] { 1, 0 });
        var codes = new int[count][];
        for (int q = 0; q < count; q++)
        {
            codes[q] = Codebooks[q].Encode(residual);
            residual = _engine.TensorSubtract(residual, Codebooks[q].Decode(codes[q]));
        }
        return codes;
    }

    /// <summary>The latent <c>[1, dim, frames]</c> of codes <c>[codebooks][frames]</c>.</summary>
    public Tensor<T> Decode(IReadOnlyList<int[]> codes)
    {
        Tensor<T> sum = Codebooks[0].Decode(codes[0]);
        for (int q = 1; q < codes.Count; q++) sum = _engine.TensorAdd(sum, Codebooks[q].Decode(codes[q]));
        int frames = sum.Shape[0], dim = sum.Shape[1];
        return OutputProjection.Forward(_engine.Reshape(_engine.TensorPermute(sum, new[] { 1, 0 }), new[] { 1, dim, frames }));
    }
}

/// <summary>
/// Mimi (Défossez et al. 2024, §3.3; reference <c>moshi</c> 0.2.2 <c>MimiModel</c>, run without streaming as Sesame's CSM
/// runs it): SEANet's causal encoder, the encoder Transformer, a learnt causal stride-2 convolution (replicate padding)
/// to 12.5 Hz, and the split quantizer — the first codebook alone on the latent, the acoustic codebooks as a second
/// residual quantizer on the same latent, their decodings summed. Decoding runs the transposed path: a depthwise causal
/// transposed convolution back to 25 Hz (the reference's <c>upsample_channel_wise_bug</c>), the decoder Transformer and
/// SEANet's decoder.
/// </summary>
internal sealed class Mimi<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly List<LayerBase<T>> _layers;
    private readonly SeanetEncoder<T> _encoder;
    private readonly SeanetDecoder<T> _decoder;
    private readonly MimiTransformer<T> _encoderTransformer, _decoderTransformer;
    private readonly NormedConv1DLayer<T> _downsample, _upsample;
    private readonly MimiResidualQuantizer<T> _semantic, _acoustic;

    public Mimi(IEngine engine, MimiConfiguration configuration, List<LayerBase<T>> owner)
    {
        _engine = engine;
        Configuration = configuration ?? throw new ArgumentNullException(nameof(configuration));
        if (configuration.SemanticCodebooks <= 0 || configuration.SemanticCodebooks >= configuration.Codebooks)
            throw new ArgumentException("Mimi needs at least one semantic and one acoustic codebook.", nameof(configuration));
        _layers = owner;
        var seanet = configuration.Seanet();
        _encoder = new SeanetEncoder<T>(engine, seanet, owner);
        _encoderTransformer = new MimiTransformer<T>(engine, configuration, owner);
        int d = configuration.Dimension, s = configuration.ResampleStride;
        owner.Add(_downsample = new NormedConv1DLayer<T>(d, d, 2 * s, s, 1, 1, 0, false, ConvolutionNormalization.None, useBias: false));
        _semantic = new MimiResidualQuantizer<T>(engine, configuration, configuration.SemanticCodebooks, owner);
        _acoustic = new MimiResidualQuantizer<T>(engine, configuration, configuration.Codebooks - configuration.SemanticCodebooks, owner);
        owner.Add(_upsample = new NormedConv1DLayer<T>(d, d, 2 * s, s, 1, d, 0, true, ConvolutionNormalization.None, useBias: false));
        _decoderTransformer = new MimiTransformer<T>(engine, configuration, owner);
        _decoder = new SeanetDecoder<T>(engine, seanet, owner);
    }

    public MimiConfiguration Configuration { get; }

    // Causal, replicate-padded stride-s convolution (reference ConvDownsample1d: StreamingConv1d, pad_mode="replicate").
    private Tensor<T> Downsample(Tensor<T> x)
    {
        int kernel = 2 * Configuration.ResampleStride, stride = Configuration.ResampleStride;
        int total = kernel - stride, extra = Seanet.ExtraPadding(x.Shape[2], kernel, stride, total);
        int channels = x.Shape[1], length = x.Shape[2];
        var parts = new List<Tensor<T>>();
        if (total > 0)
            parts.Add(_engine.TensorTile(_engine.TensorSlice(x, new[] { 0, 0, 0 }, new[] { 1, channels, 1 }), new[] { 1, 1, total }));
        parts.Add(x);
        if (extra > 0)
            parts.Add(_engine.TensorTile(_engine.TensorSlice(x, new[] { 0, 0, length - 1 }, new[] { 1, channels, 1 }), new[] { 1, 1, extra }));
        return _downsample.Forward(parts.Count == 1 ? x : _engine.TensorConcatenate(parts.ToArray(), 2));
    }

    // Causal depthwise transposed convolution (reference ConvTrUpsample1d): k − s samples trimmed on the right.
    private Tensor<T> Upsample(Tensor<T> x)
    {
        var y = _upsample.Forward(x);
        int trim = Configuration.ResampleStride;
        return _engine.TensorSlice(y, new[] { 0, 0, 0 }, new[] { 1, y.Shape[1], y.Shape[2] - trim });
    }

    /// <summary>The unquantized latent <c>[1, dim, frames]</c> at the code rate of audio <c>[1, 1, samples]</c>.</summary>
    public Tensor<T> Latent(Tensor<T> audio) => Downsample(_encoderTransformer.Forward(_encoder.Forward(audio)));

    /// <summary>The codes <c>[frames, codebooks]</c> of a waveform at <see cref="MimiConfiguration.SampleRate"/>, using the
    /// first <paramref name="codebooks"/> codebooks (all by default).</summary>
    public int[,] Encode(double[] audio, int? codebooks = null)
    {
        if (audio is null || audio.Length == 0) throw new ArgumentException("There is no audio.", nameof(audio));
        int count = codebooks ?? Configuration.Codebooks;
        if (count <= Configuration.SemanticCodebooks - 1 || count > Configuration.Codebooks)
            throw new ArgumentOutOfRangeException(nameof(codebooks), count, $"Mimi uses between {Configuration.SemanticCodebooks} and {Configuration.Codebooks} codebooks.");
        using var _ = new NoGradScope<T>();
        foreach (var layer in _layers) layer.SetTrainingMode(false);
        var input = new Tensor<T>(new[] { 1, 1, audio.Length });
        for (int i = 0; i < audio.Length; i++) input[0, 0, i] = NumOps.FromDouble(audio[i]);
        var latent = Latent(input);
        var semantic = _semantic.Encode(latent, Configuration.SemanticCodebooks);
        var acoustic = count > Configuration.SemanticCodebooks ? _acoustic.Encode(latent, count - Configuration.SemanticCodebooks) : Array.Empty<int[]>();
        var rows = semantic.Concat(acoustic).ToArray();
        int frames = rows[0].Length;
        var codes = new int[frames, count];
        for (int q = 0; q < count; q++)
            for (int t = 0; t < frames; t++) codes[t, q] = rows[q][t];
        return codes;
    }

    /// <summary>The quantized latent <c>[1, dim, frames]</c> of codes <c>[frames, codebooks]</c> (reference
    /// <c>decode_latent</c>).</summary>
    public Tensor<T> QuantizedLatent(int[,] codes)
    {
        if (codes is null) throw new ArgumentNullException(nameof(codes));
        int frames = codes.GetLength(0), count = codes.GetLength(1);
        if (frames == 0) throw new ArgumentException("There are no codes.", nameof(codes));
        if (count < Configuration.SemanticCodebooks || count > Configuration.Codebooks)
            throw new ArgumentException($"Mimi decodes between {Configuration.SemanticCodebooks} and {Configuration.Codebooks} codebooks; got {count}.", nameof(codes));
        int[] Row(int q) => Enumerable.Range(0, frames).Select(t => codes[t, q]).ToArray();
        var latent = _semantic.Decode(Enumerable.Range(0, Configuration.SemanticCodebooks).Select(Row).ToArray());
        if (count > Configuration.SemanticCodebooks)
            latent = _engine.TensorAdd(latent, _acoustic.Decode(Enumerable.Range(Configuration.SemanticCodebooks, count - Configuration.SemanticCodebooks).Select(Row).ToArray()));
        return latent;
    }

    /// <summary>The waveform of codes <c>[frames, codebooks]</c>: <see cref="MimiConfiguration.SamplesPerFrame"/> samples a
    /// frame (reference <c>decode</c>, whose output keeps the causal padding's length).</summary>
    public double[] Decode(int[,] codes)
    {
        using var _ = new NoGradScope<T>();
        foreach (var layer in _layers) layer.SetTrainingMode(false);
        var audio = _decoder.Forward(_decoderTransformer.Forward(Upsample(QuantizedLatent(codes))));
        var result = new double[audio.Shape[2]];
        for (int i = 0; i < result.Length; i++) result[i] = NumOps.ToDouble(audio[0, 0, i]);
        return result;
    }

    /// <summary>
    /// Loads a Mimi state dict (the released <c>tokenizer-e351c8d8-checkpoint125.safetensors</c>, the reference's
    /// <c>MimiModel</c> names): plain convolutions (<c>conv.conv</c>, <c>convtr.convtr</c>), the Transformers'
    /// <c>in_proj_weight</c>, <c>out_proj</c>, layer norms, feed-forward and layer scales, the resampling convolutions and
    /// each codebook's <c>embedding_sum</c> and <c>cluster_usage</c>. Returns the names it read; the codebooks'
    /// <c>_initialized</c> flags are bookkeeping and read as well.
    /// </summary>
    public HashSet<string> Load(Func<string, double[]?> read)
    {
        var used = new HashSet<string>(StringComparer.Ordinal);
        double[] Read(string name, int count)
        {
            var values = read(name) ?? throw new InvalidDataException($"The checkpoint has no tensor '{name}'.");
            if (values.Length != count) throw new InvalidDataException($"'{name}' has {values.Length} values; this model expects {count}.");
            used.Add(name);
            return values;
        }
        void Conv(string name, NormedConv1DLayer<T> conv)
        {
            int count = conv.TorchWeightShape.Aggregate(1, (a, b) => a * b);
            var bias = conv.HasBias ? Read($"{name}.bias", conv.OutputChannels) : new double[conv.OutputChannels];
            conv.LoadTorchWeights(Read($"{name}.weight", count), null, bias);
        }
        foreach (var (name, conv) in _encoder.NamedConvolutions("encoder.model")) Conv($"{name}.conv.conv", conv.Conv);
        foreach (var (name, conv) in _decoder.NamedConvolutions("decoder.model")) Conv($"{name}.conv.conv", conv.Conv);
        foreach (var (name, conv) in _decoder.NamedTransposedConvolutions("decoder.model")) Conv($"{name}.convtr.convtr", conv.Conv);
        Conv("downsample.conv.conv.conv", _downsample);
        Conv("upsample.convtr.convtr.convtr", _upsample);
        int d = Configuration.Dimension, f = Configuration.FeedForward;
        foreach (var (prefix, transformer) in new[] { ("encoder_transformer", _encoderTransformer), ("decoder_transformer", _decoderTransformer) })
        {
            for (int i = 0; i < transformer.Layers.Length; i++)
            {
                var l = transformer.Layers[i];
                string p = $"{prefix}.transformer.layers.{i}";
                TorchParameters.LayerNorm(_engine, l.Norm1, Read($"{p}.norm1.weight", d), Read($"{p}.norm1.bias", d));
                TorchParameters.LayerNorm(_engine, l.Norm2, Read($"{p}.norm2.weight", d), Read($"{p}.norm2.bias", d));
                TorchParameters.Linear(l.InProjection, d, 3 * d, Read($"{p}.self_attn.in_proj_weight", 3 * d * d));
                TorchParameters.Linear(l.OutProjection, d, d, Read($"{p}.self_attn.out_proj.weight", d * d));
                TorchParameters.Linear(l.Linear1, d, f, Read($"{p}.linear1.weight", f * d));
                TorchParameters.Linear(l.Linear2, f, d, Read($"{p}.linear2.weight", d * f));
                foreach (var (scale, key) in new[] { (l.Scale1, "layer_scale_1"), (l.Scale2, "layer_scale_2") })
                {
                    var values = Read($"{p}.{key}.scale", d);
                    for (int c = 0; c < d; c++) scale.Scale[c] = NumOps.FromDouble(values[c]);
                    _engine.InvalidatePersistentTensor(scale.Scale);
                }
            }
        }
        foreach (var (prefix, quantizer) in new[] { ("quantizer.rvq_first", _semantic), ("quantizer.rvq_rest", _acoustic) })
        {
            Conv($"{prefix}.input_proj", quantizer.InputProjection);
            Conv($"{prefix}.output_proj", quantizer.OutputProjection);
            for (int q = 0; q < quantizer.Codebooks.Length; q++)
            {
                string p = $"{prefix}.vq.layers.{q}._codebook";
                Read($"{p}._initialized", 1);
                quantizer.Codebooks[q].Load(Read($"{p}.embedding_sum", Configuration.CodebookSize * Configuration.CodebookDim),
                    Read($"{p}.cluster_usage", Configuration.CodebookSize), Configuration.CodebookEpsilon);
            }
        }
        return used;
    }
}
