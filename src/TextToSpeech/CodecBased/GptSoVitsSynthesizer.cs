using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.EndToEnd;
using AiDotNet.TextToSpeech.Latest;
using AiDotNet.TextToSpeech.Vocoders;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>The sizes of GPT-SoVITS's SoVITS synthesizer (the s2 config's <c>model</c>; <c>SynthesizerTrn</c>).</summary>
internal sealed record GptSoVitsSynthesizerConfiguration
{
    /// <summary>v1, v2, v2Pro or v2ProPlus.</summary>
    public string Version { get; init; } = "v2ProPlus";

    public int SpecChannels { get; init; } = 1025;
    public int InterChannels { get; init; } = 192;
    public int Hidden { get; init; } = 192;
    public int Filter { get; init; } = 768;
    public int Heads { get; init; } = 2;
    public int Layers { get; init; } = 6;
    public int Kernel { get; init; } = 3;
    public double Dropout { get; init; } = 0.0;
    public int[] ResblockKernels { get; init; } = { 3, 7, 11 };
    public int[][] ResblockDilations { get; init; } = { new[] { 1, 3, 5 }, new[] { 1, 3, 5 }, new[] { 1, 3, 5 } };
    public int[] UpsampleRates { get; init; } = { 10, 8, 2, 2, 2 };
    public int UpsampleInitialChannels { get; init; } = 768;
    public int[] UpsampleKernels { get; init; } = { 20, 16, 8, 2, 2 };
    public int Gin { get; init; } = 1024;

    /// <summary>Whether the semantic tokens run at 25 Hz (the ssl projection a stride-2 convolution); else 50 Hz.</summary>
    public bool SemanticRate25Hz { get; init; } = true;

    /// <summary>The phone vocabulary (symbols.py's for v1, symbols2.py's after).</summary>
    public int Symbols { get; init; } = 732;

    /// <summary>The SSL features' width (chinese-hubert-base, 768).</summary>
    public int SslDim { get; init; } = 768;

    /// <summary>The semantic codebook size (1024).</summary>
    public int Bins { get; init; } = 1024;

    /// <summary>The speaker-verification embedding width (ERes2NetV2's 20480).</summary>
    public int SvDim { get; init; } = 20480;

    /// <summary>MRTE's width (512 in the release, where it equals the style vector's width for v1 and v2).</summary>
    public int MrteChannels { get; init; } = 512;

    /// <summary>The style encoder's hidden width (128).</summary>
    public int StyleHidden { get; init; } = 128;

    public bool IsV2Pro => Version is "v2Pro" or "v2ProPlus";

    /// <summary>The reference spectrogram rows the style encoder reads (all for v1, the first 704 after).</summary>
    public int ReferenceChannels => Version == "v1" ? SpecChannels : 704;

    /// <summary>The release's s2 configurations: s2.json for v1, v2 and v2Pro (the latter two with gin 512), and
    /// s2v2ProPlus.json.</summary>
    public static GptSoVitsSynthesizerConfiguration For(string version) => version switch
    {
        "v1" => new() { Version = "v1", Dropout = 0.1, UpsampleInitialChannels = 512, UpsampleKernels = new[] { 16, 16, 8, 2, 2 }, Gin = 512, Symbols = 322 },
        "v2" => new() { Version = "v2", Dropout = 0.1, UpsampleInitialChannels = 512, UpsampleKernels = new[] { 16, 16, 8, 2, 2 }, Gin = 512 },
        "v2Pro" => new() { Version = "v2Pro", Dropout = 0.1, UpsampleInitialChannels = 512, UpsampleKernels = new[] { 16, 16, 8, 2, 2 }, Gin = 512 },
        "v2ProPlus" => new(),
        _ => throw new ArgumentException($"SoVITS's GAN synthesizer covers v1, v2, v2Pro and v2ProPlus, not {version}.", nameof(version)),
    };
}

/// <summary>
/// GPT-SoVITS's residual vector quantizer (<c>module/quantize.py</c> and <c>core_vq.py</c>, after EnCodec), with one
/// quantizer as SoVITS uses it: a 1,024-entry Euclidean codebook kept by exponential moving averages (decay 0.99, ε 1e-5),
/// dead codes (EMA size below 2) replaced by random batch vectors, and a commitment loss with the straight-through
/// estimator in training.
/// </summary>
internal sealed class GptSoVitsQuantizer<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly int _dim, _bins;
    private const double Decay = 0.99, Epsilon = 1e-5;
    private const int DeadCodeThreshold = 2;

    public GptSoVitsQuantizer(IEngine engine, int dim, int bins, Random random)
    {
        _engine = engine;
        _dim = dim;
        _bins = bins;
        // EuclideanCodebook with kmeans_init=True starts at zero and is initialized by k-means on the first batch.
        Embed = new double[bins * dim];
        EmbedAverage = new double[bins * dim];
        ClusterSize = new double[bins];
        Initialized = false;
    }

    public double[] Embed { get; }
    public double[] EmbedAverage { get; }
    public double[] ClusterSize { get; }
    public bool Initialized { get; private set; }

    private Tensor<T> Codebook()
    {
        var t = new Tensor<T>(new[] { _bins, _dim });
        for (int i = 0; i < Embed.Length; i++) t.SetFlat(i, NumOps.FromDouble(Embed[i]));
        return t;
    }

    /// <summary>The nearest code of each frame of <paramref name="x"/> <c>[1, dim, T]</c> (<c>quantize</c>: the argmax of
    /// −‖x − e‖², the first on a tie).</summary>
    public int[] Encode(Tensor<T> x)
    {
        int t = x.Shape[2];
        var codes = new int[t];
        var norms = new double[_bins];
        for (int b = 0; b < _bins; b++)
            for (int d = 0; d < _dim; d++) norms[b] += Embed[b * _dim + d] * Embed[b * _dim + d];
        for (int i = 0; i < t; i++)
        {
            double xx = 0;
            for (int d = 0; d < _dim; d++) { double v = NumOps.ToDouble(x[0, d, i]); xx += v * v; }
            double best = double.NegativeInfinity;
            for (int b = 0; b < _bins; b++)
            {
                double dot = 0;
                for (int d = 0; d < _dim; d++) dot += NumOps.ToDouble(x[0, d, i]) * Embed[b * _dim + d];
                double score = -(xx - 2 * dot + norms[b]);
                if (score > best) { best = score; codes[i] = b; }
            }
        }
        return codes;
    }

    /// <summary>The codes' vectors <c>[1, dim, T]</c> (<c>decode</c>).</summary>
    public Tensor<T> Decode(IReadOnlyList<int> codes)
    {
        if (codes.Any(c => c < 0 || c >= _bins)) throw new ArgumentOutOfRangeException(nameof(codes), $"Codes lie in [0, {_bins}).");
        var index = new Tensor<int>(new[] { codes.Count }, new Vector<int>(codes.ToArray()));
        var rows = _engine.TensorIndexSelect(Codebook(), index, 0);                                    // [T, dim]
        return _engine.Reshape(_engine.TensorTranspose(rows), new[] { 1, _dim, codes.Count });
    }

    /// <summary>
    /// The quantizer's forward pass on <c>[1, dim, T]</c>: the quantized frames, the codes and the commitment loss. In
    /// training (not frozen) the codebook takes its EMA step and dead codes are replaced, the output passes the gradient
    /// straight through to the input and the loss is the MSE of the input to its (detached) quantization; frozen or in
    /// evaluation the output is the codebook's vectors and the loss zero, as the release's eval mode gives.
    /// </summary>
    public (Tensor<T> Quantized, int[] Codes, Tensor<T> Commitment) Forward(Tensor<T> x, bool training, Random random)
    {
        int t = x.Shape[2];
        if (training && !Initialized) KMeansInitialize(x, random);
        var codes = Encode(x);
        var quantized = Decode(codes);
        if (!training) return (quantized, codes, new Tensor<T>(new[] { 1 }));
        // EMA update.
        var counts = new double[_bins];
        var sums = new double[_bins * _dim];
        for (int i = 0; i < t; i++)
        {
            counts[codes[i]]++;
            for (int d = 0; d < _dim; d++) sums[codes[i] * _dim + d] += NumOps.ToDouble(x[0, d, i]);
        }
        double n = 0;
        for (int b = 0; b < _bins; b++)
        {
            ClusterSize[b] = ClusterSize[b] * Decay + counts[b] * (1 - Decay);
            n += ClusterSize[b];
        }
        for (int i = 0; i < EmbedAverage.Length; i++) EmbedAverage[i] = EmbedAverage[i] * Decay + sums[i] * (1 - Decay);
        for (int b = 0; b < _bins; b++)
        {
            double size = (ClusterSize[b] + Epsilon) / (n + _bins * Epsilon) * n;
            for (int d = 0; d < _dim; d++) Embed[b * _dim + d] = EmbedAverage[b * _dim + d] / size;
        }
        // expire_codes_: codes whose EMA size fell below the threshold take random frames of this batch.
        var expired = Enumerable.Range(0, _bins).Where(b => ClusterSize[b] < DeadCodeThreshold).ToList();
        if (expired.Count > 0)
        {
            var picks = SampleVectors(t, expired.Count, random);
            for (int k = 0; k < expired.Count; k++)
                for (int d = 0; d < _dim; d++) Embed[expired[k] * _dim + d] = NumOps.ToDouble(x[0, d, picks[k]]);
        }
        // quantize = x + (quantize - x).detach(); commit = mse(quantize.detach(), x)
        var straight = _engine.TensorAdd(x, MeloOps.Detach(_engine.TensorSubtract(quantized, x)));
        var difference = _engine.TensorSubtract(x, MeloOps.Detach(quantized));
        var commitment = _engine.TensorMultiplyScalar(_engine.ReduceSum(_engine.TensorMultiply(difference, difference), null),
            NumOps.FromDouble(1.0 / ((double)t * _dim)));
        return (straight, codes, commitment);
    }

    // sample_vectors: a permutation's first picks when there are enough frames, else picks with replacement.
    private static int[] SampleVectors(int count, int needed, Random random)
    {
        if (count >= needed)
        {
            var order = Enumerable.Range(0, count).ToArray();
            for (int i = count - 1; i > 0; i--) { int j = random.Next(i + 1); (order[i], order[j]) = (order[j], order[i]); }
            return order.Take(needed).ToArray();
        }
        return Enumerable.Range(0, needed).Select(_ => random.Next(count)).ToArray();
    }

    // GPT-SoVITS's kmeans: up to 10,000 frames, 50 iterations of nearest-centroid assignment (in batches of 64, by
    // Euclidean distance) and per-cluster means, empty clusters kept; the cluster sizes of the last assignment.
    private void KMeansInitialize(Tensor<T> x, Random random)
    {
        int t = x.Shape[2];
        var samples = new List<double[]>();
        var indices = t > 10_000 ? SampleVectors(t, 10_000, random) : Enumerable.Range(0, t).ToArray();
        foreach (int i in indices) samples.Add(Enumerable.Range(0, _dim).Select(d => NumOps.ToDouble(x[0, d, i])).ToArray());
        var means = SampleVectors(samples.Count, _bins, random).Select(i => (double[])samples[i].Clone()).ToArray();
        var buckets = new int[samples.Count];
        for (int iteration = 0; iteration < 50; iteration++)
        {
            for (int s = 0; s < samples.Count; s++)
            {
                double best = double.PositiveInfinity;
                for (int c = 0; c < _bins; c++)
                {
                    double distance = 0;
                    for (int d = 0; d < _dim; d++) { double diff = samples[s][d] - means[c][d]; distance += diff * diff; }
                    if (distance < best) { best = distance; buckets[s] = c; }
                }
            }
            for (int c = 0; c < _bins; c++)
            {
                var members = Enumerable.Range(0, samples.Count).Where(s => buckets[s] == c).ToList();
                if (members.Count == 0) continue;
                for (int d = 0; d < _dim; d++) means[c][d] = members.Average(s => samples[s][d]);
            }
        }
        for (int c = 0; c < _bins; c++)
        {
            for (int d = 0; d < _dim; d++)
            {
                Embed[c * _dim + d] = means[c][d];
                EmbedAverage[c * _dim + d] = means[c][d];
            }
            ClusterSize[c] = buckets.Count(b => b == c);
        }
        Initialized = true;
    }

    /// <summary>Loads <c>quantizer.vq.layers.0._codebook.*</c>.</summary>
    public void Load(Func<string, double[]?> read, string prefix)
    {
        var embed = read(prefix + "embed") ?? throw new KeyNotFoundException(prefix + "embed");
        Array.Copy(embed, Embed, Embed.Length);
        var average = read(prefix + "embed_avg");
        Array.Copy(average ?? embed, EmbedAverage, EmbedAverage.Length);
        var sizes = read(prefix + "cluster_size");
        if (sizes is not null) Array.Copy(sizes, ClusterSize, ClusterSize.Length);
        var inited = read(prefix + "inited");
        Initialized = inited is null || inited[0] != 0;
    }
}

/// <summary>
/// The multi-head attention of GPT-SoVITS's attentions.py without relative positions (MRTE's cross-attention): 1×1
/// convolutions for queries (from x), keys and values (from c), masked softmax (masked scores −1e4) and an output 1×1
/// convolution, over <c>[1, C, T]</c>.
/// </summary>
internal sealed class GptSoVitsCrossAttention<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly int _channels, _heads;

    public GptSoVitsCrossAttention(IEngine engine, List<LayerBase<T>> layers, int channels, int outChannels, int heads)
    {
        _engine = engine;
        _channels = channels;
        _heads = heads;
        Query = MeloOps.Conv(layers, channels, channels, 1);
        Key = MeloOps.Conv(layers, channels, channels, 1);
        Value = MeloOps.Conv(layers, channels, channels, 1);
        Output = MeloOps.Conv(layers, channels, outChannels, 1);
    }

    public NormedConv1DLayer<T> Query { get; }
    public NormedConv1DLayer<T> Key { get; }
    public NormedConv1DLayer<T> Value { get; }
    public NormedConv1DLayer<T> Output { get; }

    public Tensor<T> Forward(Tensor<T> x, Tensor<T> c)
    {
        int tq = x.Shape[2], tk = c.Shape[2], h = _heads, dk = _channels / h;
        Tensor<T> Heads(Tensor<T> y, int length) => _engine.TensorPermute(_engine.Reshape(y, new[] { h, dk, length }), new[] { 0, 2, 1 });
        var q = _engine.TensorMultiplyScalar(Heads(Query.Forward(x), tq), NumOps.FromDouble(1.0 / Math.Sqrt(dk)));
        var k = Heads(Key.Forward(c), tk);
        var v = Heads(Value.Forward(c), tk);
        var p = _engine.Softmax(_engine.BatchMatMul(q, _engine.TensorPermute(k, new[] { 0, 2, 1 })), axis: 2);
        var output = _engine.Reshape(_engine.TensorPermute(_engine.BatchMatMul(p, v), new[] { 0, 2, 1 }), new[] { 1, _channels, tq });
        return Output.Forward(output);
    }

    public void Load(Func<string, double[]> read, string prefix)
    {
        MeloOps.LoadConv(read, Query, prefix + "conv_q");
        MeloOps.LoadConv(read, Key, prefix + "conv_k");
        MeloOps.LoadConv(read, Value, prefix + "conv_v");
        MeloOps.LoadConv(read, Output, prefix + "conv_o");
    }
}

/// <summary>
/// GPT-SoVITS's MelStyleEncoder (<c>module/modules.py</c>, after Meta-StyleSpeech): two Mish-activated linear layers,
/// two residual gated 1D convolutions (kernel 5), residual two-head self-attention (scaled by √d_model), a linear
/// projection and the average over time: one style vector <c>[1, out, 1]</c> of a spectrogram <c>[1, in, T]</c>.
/// </summary>
internal sealed class GptSoVitsMelStyleEncoder<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly int _in, _hidden, _out, _heads = 2;
    private readonly double _dropout;

    public GptSoVitsMelStyleEncoder(IEngine engine, List<LayerBase<T>> layers, Random random, int inChannels, int outChannels,
        int hidden = 128, double dropout = 0.1)
    {
        _engine = engine;
        _in = inChannels;
        _hidden = hidden;
        _out = outChannels;
        _dropout = dropout;
        Spectral1 = IndexTtsOps.Linear(engine, layers, random, inChannels, hidden);
        Spectral2 = IndexTtsOps.Linear(engine, layers, random, hidden, hidden);
        Temporal1 = MeloOps.Conv(layers, hidden, 2 * hidden, 5, 1, 2);
        Temporal2 = MeloOps.Conv(layers, hidden, 2 * hidden, 5, 1, 2);
        Query = IndexTtsOps.Linear(engine, layers, random, hidden, hidden);
        Key = IndexTtsOps.Linear(engine, layers, random, hidden, hidden);
        Value = IndexTtsOps.Linear(engine, layers, random, hidden, hidden);
        AttentionOutput = IndexTtsOps.Linear(engine, layers, random, hidden, hidden);
        Projection = IndexTtsOps.Linear(engine, layers, random, hidden, outChannels);
    }

    public DenseLayer<T> Spectral1 { get; }
    public DenseLayer<T> Spectral2 { get; }
    public NormedConv1DLayer<T> Temporal1 { get; }
    public NormedConv1DLayer<T> Temporal2 { get; }
    public DenseLayer<T> Query { get; }
    public DenseLayer<T> Key { get; }
    public DenseLayer<T> Value { get; }
    public DenseLayer<T> AttentionOutput { get; }
    public DenseLayer<T> Projection { get; }

    private Tensor<T> Mish(Tensor<T> x) => _engine.TensorMultiply(x, _engine.Tanh(_engine.Softplus(x)));

    private Tensor<T> Drop(Tensor<T> x, bool training, Random random) =>
        training && _dropout > 0 ? T5Seq2Seq<T>.Dropout(_engine, x, _dropout, random) : x;

    // Conv1dGLU: x + dropout(a · σ(b)) with [a; b] = conv(x).
    private Tensor<T> Glu(NormedConv1DLayer<T> conv, Tensor<T> x, bool training, Random random)
    {
        var y = conv.Forward(x);
        var gated = _engine.TensorMultiply(MeloOps.Channels(_engine, y, 0, _hidden), _engine.Sigmoid(MeloOps.Channels(_engine, y, _hidden, _hidden)));
        return _engine.TensorAdd(x, Drop(gated, training, random));
    }

    /// <summary>The style vector <c>[1, out, 1]</c> of <paramref name="spectrogram"/> <c>[1, in, T]</c> (no padding).</summary>
    public Tensor<T> Forward(Tensor<T> spectrogram, bool training, Random random)
    {
        int c = spectrogram.Shape[1], t = spectrogram.Shape[2];
        var x = _engine.TensorTranspose(_engine.Reshape(spectrogram, new[] { c, t }));                           // [T, in]
        x = Drop(Mish(Spectral1.Forward(x)), training, random);
        x = Drop(Mish(Spectral2.Forward(x)), training, random);
        var channels = _engine.Reshape(_engine.TensorTranspose(x), new[] { 1, _hidden, t });
        channels = Glu(Temporal2, Glu(Temporal1, channels, training, random), training, random);
        x = _engine.TensorTranspose(_engine.Reshape(channels, new[] { _hidden, t }));                             // [T, hidden]
        // MultiHeadAttention: heads of d_k = hidden / 2, scores / √hidden (temperature d_model^0.5), residual.
        int dk = _hidden / _heads;
        Tensor<T> Heads(Tensor<T> y) => _engine.TensorPermute(_engine.Reshape(y, new[] { t, _heads, dk }), new[] { 1, 0, 2 });
        var scores = _engine.TensorMultiplyScalar(_engine.BatchMatMul(Heads(Query.Forward(x)), _engine.TensorPermute(Heads(Key.Forward(x)), new[] { 0, 2, 1 })),
            NumOps.FromDouble(1.0 / Math.Sqrt(_hidden)));
        var p = Drop(_engine.Softmax(scores, axis: 2), training, random);
        var attended = _engine.Reshape(_engine.TensorPermute(_engine.BatchMatMul(p, Heads(Value.Forward(x))), new[] { 1, 0, 2 }), new[] { t, _hidden });
        x = _engine.TensorAdd(Drop(AttentionOutput.Forward(attended), training, random), x);
        x = Projection.Forward(x);                                                                                 // [T, out]
        // temporal_avg_pool with a mask: each frame divided by the length, then summed.
        var pooled = _engine.ReduceSum(_engine.TensorMultiplyScalar(x, NumOps.FromDouble(1.0 / t)), new[] { 0 }, keepDims: false);
        return _engine.Reshape(pooled, new[] { 1, _out, 1 });
    }

    public void Load(IEngine engine, Func<string, double[]> read, string prefix)
    {
        MeloOps.LoadLinear(engine, read, Spectral1, _in, _hidden, prefix + "spectral.0.fc");
        MeloOps.LoadLinear(engine, read, Spectral2, _hidden, _hidden, prefix + "spectral.3.fc");
        MeloOps.LoadConv(read, Temporal1, prefix + "temporal.0.conv1.conv");
        MeloOps.LoadConv(read, Temporal2, prefix + "temporal.1.conv1.conv");
        MeloOps.LoadLinear(engine, read, Query, _hidden, _hidden, prefix + "slf_attn.w_qs");
        MeloOps.LoadLinear(engine, read, Key, _hidden, _hidden, prefix + "slf_attn.w_ks");
        MeloOps.LoadLinear(engine, read, Value, _hidden, _hidden, prefix + "slf_attn.w_vs");
        MeloOps.LoadLinear(engine, read, AttentionOutput, _hidden, _hidden, prefix + "slf_attn.fc");
        MeloOps.LoadLinear(engine, read, Projection, _hidden, _out, prefix + "fc.fc");
    }
}

/// <summary>
/// GPT-SoVITS's SoVITS synthesizer for v1, v2, v2Pro and v2ProPlus (<c>module/models.py</c> <c>SynthesizerTrn</c>, after
/// VITS): the SSL features through a stride-2 projection and the semantic quantizer (upsampled ×2), the text encoder
/// (SSL and phone Transformers joined by MRTE's cross-attention with the reference's style vector, then a third
/// Transformer), the prior's mean and log-scale, the mean-only coupling flow, the HiFi-GAN generator conditioned on the
/// style vector, and in training the posterior encoder over the linear spectrogram. v2Pro adds a speaker-verification
/// embedding to the style vector (then PReLU) and projects it to 512 channels for MRTE.
/// </summary>
internal sealed class GptSoVitsSynthesizer<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly List<(NormedConv1DLayer<T> Pre, MeloWaveNet<T> Net, NormedConv1DLayer<T> Post)> _couplings = new();

    public GptSoVitsSynthesizer(IEngine engine, List<LayerBase<T>> layers, GptSoVitsSynthesizerConfiguration configuration, Random random)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        int h = c.Hidden;
        SslProjection = new NormedConv1DLayer<T>(c.SslDim, c.SslDim, c.SemanticRate25Hz ? 2 : 1, c.SemanticRate25Hz ? 2 : 1, 1, 1, 0, false,
            ConvolutionNormalization.None);
        layers.Add(SslProjection);
        Quantizer = new GptSoVitsQuantizer<T>(engine, c.SslDim, c.Bins, random);
        // TextEncoder
        TextSslProjection = MeloOps.Conv(layers, c.SslDim, h, 1);
        EncoderSsl = new MeloAttentionEncoder<T>(engine, layers, h, c.Filter, c.Heads, c.Layers / 2, c.Kernel, c.Dropout, 0);
        EncoderText = new MeloAttentionEncoder<T>(engine, layers, h, c.Filter, c.Heads, c.Layers, c.Kernel, c.Dropout, 0);
        TextEmbedding = IndexTtsOps.Embedding(layers, random, c.Symbols, h, 1.0);
        int mrte = c.MrteChannels;
        if (!c.IsV2Pro && c.Gin != mrte) throw new ArgumentException("MRTE adds the style vector, so its width must equal gin before v2Pro.", nameof(configuration));
        MrteContent = MeloOps.Conv(layers, h, mrte, 1);
        MrteText = MeloOps.Conv(layers, h, mrte, 1);
        MrteAttention = new GptSoVitsCrossAttention<T>(engine, layers, mrte, mrte, 4);
        MrtePost = MeloOps.Conv(layers, mrte, h, 1);
        Encoder2 = new MeloAttentionEncoder<T>(engine, layers, h, c.Filter, c.Heads, c.Layers / 2, c.Kernel, c.Dropout, 0);
        PriorProjection = MeloOps.Conv(layers, h, 2 * c.InterChannels, 1);
        // Flow: four mean-only couplings (kernel 5, dilation 1, four WaveNet layers) with flips.
        int half = c.InterChannels / 2;
        for (int i = 0; i < 4; i++)
        {
            var pre = MeloOps.Conv(layers, half, h, 1);
            var net = new MeloWaveNet<T>(engine, layers, h, 5, 1, 4, c.Gin);
            var post = MeloOps.Conv(layers, h, half, 1);
            post.LoadTorchWeights(new double[half * h], null, new double[half]);
            _couplings.Add((pre, net, post));
        }
        Generator = new HiFiGanGenerator<T>(engine, c.InterChannels, c.UpsampleInitialChannels, c.UpsampleRates, c.UpsampleKernels,
            c.ResblockKernels, c.ResblockDilations, resblockType1: true, conditionChannels: c.Gin, plainEnds: true, initialization: random);
        layers.AddRange(Generator.Layers);
        Posterior = new MeloPosteriorEncoder<T>(engine, layers, c.SpecChannels, c.InterChannels, h, c.Gin);
        ReferenceEncoder = new GptSoVitsMelStyleEncoder<T>(engine, layers, random, c.ReferenceChannels, c.Gin, c.StyleHidden, c.Dropout > 0 ? 0.1 : 0.0);
        if (c.IsV2Pro)
        {
            SvProjection = IndexTtsOps.Linear(engine, layers, random, c.SvDim, c.Gin);
            Prelu = IndexTtsOps.Values(layers, Enumerable.Repeat(0.25, c.Gin).ToArray());
            GeTo512 = IndexTtsOps.Linear(engine, layers, random, c.Gin, c.MrteChannels);
        }
    }

    public GptSoVitsSynthesizerConfiguration Configuration { get; }
    public NormedConv1DLayer<T> SslProjection { get; }
    public GptSoVitsQuantizer<T> Quantizer { get; }
    public NormedConv1DLayer<T> TextSslProjection { get; }
    public MeloAttentionEncoder<T> EncoderSsl { get; }
    public MeloAttentionEncoder<T> EncoderText { get; }
    public TiedEmbeddingLayer<T> TextEmbedding { get; }
    public NormedConv1DLayer<T> MrteContent { get; }
    public NormedConv1DLayer<T> MrteText { get; }
    public GptSoVitsCrossAttention<T> MrteAttention { get; }
    public NormedConv1DLayer<T> MrtePost { get; }
    public MeloAttentionEncoder<T> Encoder2 { get; }
    public NormedConv1DLayer<T> PriorProjection { get; }
    public HiFiGanGenerator<T> Generator { get; }
    public MeloPosteriorEncoder<T> Posterior { get; }
    public GptSoVitsMelStyleEncoder<T> ReferenceEncoder { get; }
    public DenseLayer<T>? SvProjection { get; }
    public BiasFreeLinearLayer<T>? Prelu { get; }
    public DenseLayer<T>? GeTo512 { get; }

    /// <summary>The text encoder's parameters' layers (text_embedding, encoder_text, MRTE), which s2 trains at a lower
    /// learning rate (<c>text_low_lr_rate</c>).</summary>
    public IEnumerable<LayerBase<T>> LowRateLayers =>
        new LayerBase<T>[] { TextEmbedding, MrteContent, MrteText, MrteAttention.Query, MrteAttention.Key, MrteAttention.Value, MrteAttention.Output, MrtePost }
            .Concat(EncoderText.Blocks.SelectMany(b => new LayerBase<T>[] { b.Query, b.Key, b.Value, b.Output, b.RelativeKeys, b.RelativeValues, b.Norm1, b.Ffn1, b.Ffn2, b.Norm2 }));

    /// <summary>
    /// The style vector <c>[1, gin, 1]</c> of a reference spectrogram <c>[1, 1025, T]</c> (its first 704 rows after v1), plus
    /// for v2Pro the projected speaker-verification embedding <c>[20480]</c>, then PReLU.
    /// </summary>
    public Tensor<T> Style(Tensor<T> reference, Tensor<T>? sv, bool training = false, Random? random = null)
    {
        var c = Configuration;
        var input = c.Version == "v1" ? reference : _engine.TensorSlice(reference, new[] { 0, 0, 0 }, new[] { 1, c.ReferenceChannels, reference.Shape[2] });
        var ge = ReferenceEncoder.Forward(input, training, random ?? new Random(0));
        if (!c.IsV2Pro) return ge;
        if (sv is null) throw new ArgumentNullException(nameof(sv), $"{c.Version} needs the reference's speaker-verification embedding.");
        var projected = _engine.Reshape(SvProjection!.Forward(_engine.Reshape(sv, new[] { 1, c.SvDim })), new[] { 1, c.Gin, 1 });
        ge = _engine.TensorAdd(ge, projected);
        // PReLU: max(0, x) + a·min(0, x) per channel.
        var a = _engine.Reshape(IndexTtsOps.Value(Prelu!), new[] { 1, c.Gin, 1 });
        var negative = _engine.TensorSubtract(ge, _engine.ReLU(ge));
        return _engine.TensorAdd(_engine.ReLU(ge), _engine.TensorMultiply(a, negative));
    }

    // The style vector as MRTE adds it: projected to 512 channels for v2Pro.
    private Tensor<T> MrteStyle(Tensor<T> ge)
    {
        var c = Configuration;
        if (!c.IsV2Pro) return ge;
        return _engine.Reshape(GeTo512!.Forward(_engine.Reshape(ge, new[] { 1, c.Gin })), new[] { 1, c.MrteChannels, 1 });
    }

    /// <summary>The quantized SSL features <c>[1, 768, 2·T']</c> of codes (<c>quantizer.decode</c>, nearest ×2 at 25 Hz).</summary>
    public Tensor<T> Dequantize(IReadOnlyList<int> codes) => Upsample(Quantizer.Decode(codes));

    private Tensor<T> Upsample(Tensor<T> quantized)
    {
        if (!Configuration.SemanticRate25Hz) return quantized;
        int t = quantized.Shape[2];
        var index = new Tensor<int>(new[] { 2 * t });
        for (int i = 0; i < 2 * t; i++) index[i] = i / 2;
        return TimeSelect(quantized, index);
    }

    /// <summary>The semantic codes of SSL features <c>[1, 768, T]</c> (<c>extract_latent</c>).</summary>
    public int[] ExtractLatent(Tensor<T> ssl) => Quantizer.Encode(SslProjection.Forward(ssl));

    /// <summary>
    /// The text encoder (<c>TextEncoder.forward</c>) of quantized features <c>[1, 768, T]</c> and phone ids, conditioned
    /// on <paramref name="ge"/>: the prior's mean and log-scale <c>[1, inter, T']</c> (T' = T at speed 1, else
    /// <c>⌊T/speed⌋ + 1</c> by linear interpolation).
    /// </summary>
    public (Tensor<T> Mean, Tensor<T> LogScale) Encode(Tensor<T> quantized, IReadOnlyList<int> text, Tensor<T> ge, double speed = 1.0,
        bool training = false, Random? random = null)
    {
        var c = Configuration;
        random ??= new Random(0);
        if (text.Any(id => id < 0 || id >= c.Symbols)) throw new ArgumentOutOfRangeException(nameof(text), $"Phone ids lie in [0, {c.Symbols}).");
        var y = TextSslProjection.Forward(quantized);
        y = EncoderSsl.Forward(y, null, training, random);
        var t = _engine.Reshape(_engine.TensorTranspose(TextEmbedding.Forward(IndexTtsOps.Ids<T>(text))), new[] { 1, c.Hidden, text.Count });
        t = EncoderText.Forward(t, null, training, random);
        // MRTE: cross-attention of the SSL features to the text, plus the features and the style vector.
        var content = MrteContent.Forward(y);
        var mixed = _engine.TensorAdd(_engine.TensorAdd(MrteAttention.Forward(content, MrteText.Forward(t)), content),
            MeloOps.OverTime(_engine, MrteStyle(ge), content.Shape[2]));
        y = MrtePost.Forward(mixed);
        y = Encoder2.Forward(y, null, training, random);
        if (speed != 1.0) y = LinearResize(y, (int)(y.Shape[2] / speed) + 1);
        var stats = PriorProjection.Forward(y);
        return (MeloOps.Channels(_engine, stats, 0, c.InterChannels), MeloOps.Channels(_engine, stats, c.InterChannels, c.InterChannels));
    }

    // x[:, :, index] for [1, C, T] (the engine selects along an axis of a matrix).
    private Tensor<T> TimeSelect(Tensor<T> x, Tensor<int> index)
    {
        int channels = x.Shape[1];
        var selected = _engine.TensorIndexSelect(_engine.Reshape(x, new[] { channels, x.Shape[2] }), index, 1);
        return _engine.Reshape(selected, new[] { 1, channels, index.Length });
    }

    // F.interpolate(mode="linear", align_corners=False) along time.
    private Tensor<T> LinearResize(Tensor<T> x, int size)
    {
        int t = x.Shape[2];
        var lower = new Tensor<int>(new[] { size });
        var upper = new Tensor<int>(new[] { size });
        var weight = new Tensor<T>(new[] { 1, 1, size });
        double scale = (double)t / size;
        for (int i = 0; i < size; i++)
        {
            double source = Math.Max((i + 0.5) * scale - 0.5, 0);
            int l = Math.Min((int)Math.Floor(source), t - 1);
            int u = Math.Min(l + 1, t - 1);
            lower[i] = l;
            upper[i] = u;
            weight[0, 0, i] = NumOps.FromDouble(source - l);
        }
        var w = _engine.TensorBroadcastTo(weight, new[] { 1, x.Shape[1], size });
        var a = TimeSelect(x, lower);
        var b = TimeSelect(x, upper);
        return _engine.TensorAdd(a, _engine.TensorMultiply(_engine.TensorSubtract(b, a), w));
    }

    /// <summary>The flow (<c>ResidualCouplingBlock</c>) forward, or reversed.</summary>
    public Tensor<T> Flow(Tensor<T> x, Tensor<T> ge, bool reverse)
    {
        if (!reverse)
        {
            foreach (var coupling in _couplings) x = MeloOps.Flip(_engine, Couple(x, coupling, ge, false));
            return x;
        }
        for (int i = _couplings.Count - 1; i >= 0; i--) x = Couple(MeloOps.Flip(_engine, x), _couplings[i], ge, true);
        return x;
    }

    private Tensor<T> Couple(Tensor<T> x, (NormedConv1DLayer<T> Pre, MeloWaveNet<T> Net, NormedConv1DLayer<T> Post) c, Tensor<T> ge, bool reverse)
    {
        int half = Configuration.InterChannels / 2;
        var x0 = MeloOps.Channels(_engine, x, 0, half);
        var x1 = MeloOps.Channels(_engine, x, half, half);
        var m = c.Post.Forward(c.Net.Forward(c.Pre.Forward(x0), ge));
        return _engine.TensorConcatenate(new[] { x0, reverse ? _engine.TensorSubtract(x1, m) : _engine.TensorAdd(x1, m) }, 1);
    }

    /// <summary>
    /// <c>decode</c>: the audio <c>[samples]</c> of semantic codes and phone ids in the voice of the reference spectrograms
    /// (their style vectors averaged) and, for v2Pro, their speaker-verification embeddings: the prior sampled with
    /// <paramref name="noise"/> <c>[1, inter, T']</c> (standard normal) at <paramref name="noiseScale"/>, the flow reversed,
    /// the generator.
    /// </summary>
    public Tensor<T> Decode(IReadOnlyList<int> codes, IReadOnlyList<int> text, IReadOnlyList<Tensor<T>> references,
        IReadOnlyList<Tensor<T>?>? sv, double noiseScale, double speed, Func<int[], Tensor<T>> noise)
    {
        if (references.Count == 0) throw new ArgumentException("At least one reference spectrogram.", nameof(references));
        Tensor<T>? ge = null;
        for (int i = 0; i < references.Count; i++)
        {
            var style = Style(references[i], sv?[i]);
            ge = ge is null ? style : _engine.TensorAdd(ge, style);
        }
        ge = _engine.TensorMultiplyScalar(ge!, NumOps.FromDouble(1.0 / references.Count));
        var (m, logs) = Encode(Dequantize(codes), text, ge, speed);
        var eps = noise(m.Shape.ToArray());
        var zp = _engine.TensorAdd(m, _engine.TensorMultiplyScalar(_engine.TensorMultiply(eps, _engine.TensorExp(logs)), NumOps.FromDouble(noiseScale)));
        var z = Flow(zp, ge, reverse: true);
        var waveform = Generator.Forward(z, ge);
        return _engine.Reshape(waveform, new[] { waveform.Length });
    }

    /// <summary>One training pass (<c>forward</c>, batch 1): the generated audio of a random latent segment and what
    /// the losses need.</summary>
    internal sealed record TrainingPass(Tensor<T> Audio, int SliceStart, Tensor<T> Commitment, Tensor<T> Zp, Tensor<T> PriorMean,
        Tensor<T> PriorLogScale, Tensor<T> PosteriorLogScale);

    /// <summary>
    /// <c>forward(ssl, spec, ..., text, sv_emb)</c> for one example: the style of the spectrogram, the quantized SSL
    /// features (frozen quantizer when <paramref name="freezeQuantizer"/>), the prior from the text encoder, the posterior
    /// from the spectrogram with <paramref name="posteriorNoise"/>, its flow, and the generator over a random latent
    /// segment of <paramref name="segmentFrames"/> frames.
    /// </summary>
    internal TrainingPass Forward(Tensor<T> ssl, Tensor<T> spectrogram, IReadOnlyList<int> text, Tensor<T>? sv, int segmentFrames,
        bool freezeQuantizer, Tensor<T> posteriorNoise, Random random)
    {
        var c = Configuration;
        int frames = spectrogram.Shape[2];
        var ge = Style(spectrogram, sv, training: true, random);
        var projected = SslProjection.Forward(ssl);
        var (quantized, _, commitment) = Quantizer.Forward(freezeQuantizer ? MeloOps.Detach(projected) : projected, training: !freezeQuantizer, random);
        quantized = Upsample(quantized);
        if (quantized.Shape[2] != frames)
            throw new ArgumentException($"The SSL features give {quantized.Shape[2]} frames for a {frames}-frame spectrogram.", nameof(ssl));
        var (m, logs) = Encode(quantized, text, ge, 1.0, training: true, random);
        var (z, _, logsQ) = Posterior.Forward(spectrogram, MeloOps.Detach(ge), posteriorNoise);
        var zp = Flow(z, ge, reverse: false);
        int maxStart = Math.Max(frames - segmentFrames + 1, 1);
        int start = (int)(random.NextDouble() * maxStart);
        var segment = _engine.TensorSlice(z, new[] { 0, 0, start }, new[] { 1, c.InterChannels, Math.Min(segmentFrames, frames - start) });
        var generated = Generator.Forward(segment, ge);
        var audio = _engine.Reshape(generated, new[] { generated.Length });
        return new TrainingPass(audio, start, commitment, zp, m, logs, logsQ);
    }

    /// <summary><c>kl_loss(z_p, logs_q, m_p, logs_p)</c> over all frames.</summary>
    internal Tensor<T> KlLoss(TrainingPass pass)
    {
        var half = NumOps.FromDouble(0.5);
        var kl = _engine.TensorAddScalar(_engine.TensorSubtract(pass.PriorLogScale, pass.PosteriorLogScale), NumOps.FromDouble(-0.5));
        var diff = _engine.TensorSubtract(pass.Zp, pass.PriorMean);
        kl = _engine.TensorAdd(kl, _engine.TensorMultiplyScalar(_engine.TensorMultiply(_engine.TensorMultiply(diff, diff),
            _engine.TensorExp(_engine.TensorMultiplyScalar(pass.PriorLogScale, NumOps.FromDouble(-2.0)))), half));
        return _engine.TensorMultiplyScalar(_engine.ReduceSum(kl, null), NumOps.FromDouble(1.0 / pass.Zp.Shape[2]));
    }

    // ---------------------------------------------------------------- weights

    /// <summary>Loads the s2 generator's state (the checkpoint's <c>weight</c>).</summary>
    public void Load(Func<string, double[]?> read)
    {
        var c = Configuration;
        double[] Require(string name) => read(name) ?? throw new KeyNotFoundException($"The SoVITS checkpoint has no '{name}'.");
        MeloOps.LoadConv(Require, SslProjection, "ssl_proj");
        Quantizer.Load(read, "quantizer.vq.layers.0._codebook.");
        MeloOps.LoadConv(Require, TextSslProjection, "enc_p.ssl_proj");
        EncoderSsl.LoadTorch(_engine, Require, "enc_p.encoder_ssl.");
        EncoderText.LoadTorch(_engine, Require, "enc_p.encoder_text.");
        TextEmbedding.LoadLeadingRows(Require("enc_p.text_embedding.weight"));
        MeloOps.LoadConv(Require, MrteContent, "enc_p.mrte.c_pre");
        MeloOps.LoadConv(Require, MrteText, "enc_p.mrte.text_pre");
        MrteAttention.Load(Require, "enc_p.mrte.cross_attention.");
        MeloOps.LoadConv(Require, MrtePost, "enc_p.mrte.c_post");
        Encoder2.LoadTorch(_engine, Require, "enc_p.encoder2.");
        MeloOps.LoadConv(Require, PriorProjection, "enc_p.proj");
        for (int i = 0; i < _couplings.Count; i++)
        {
            var (pre, net, post) = _couplings[i];
            MeloOps.LoadConv(Require, pre, $"flow.flows.{2 * i}.pre");
            net.LoadTorch(Require, $"flow.flows.{2 * i}.enc.");
            MeloOps.LoadConv(Require, post, $"flow.flows.{2 * i}.post");
        }
        Generator.LoadTorch(read, "dec.");
        if (read("enc_q.pre.weight") is not null) Posterior.LoadTorch(Require, "enc_q.");
        ReferenceEncoder.Load(_engine, Require, "ref_enc.");
        if (c.IsV2Pro)
        {
            MeloOps.LoadLinear(_engine, Require, SvProjection!, c.SvDim, c.Gin, "sv_emb");
            TorchParameters.Linear(Prelu!, 1, c.Gin, Require("prelu.weight"));
            MeloOps.LoadLinear(_engine, Require, GeTo512!, c.Gin, c.MrteChannels, "ge_to512");
        }
    }
}
