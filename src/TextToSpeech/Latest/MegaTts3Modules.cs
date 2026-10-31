using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.TextToSpeech.Latest;

/// <summary>Shared operations of MegaTTS 3's modules, on <c>[B, T, C]</c> tensors.</summary>
internal static class MegaTts3Ops
{
    public static DenseLayer<T> Linear<T>(List<LayerBase<T>> layers, int outputs)
    {
        var layer = new DenseLayer<T>(outputs, (IActivationFunction<T>)new IdentityActivation<T>());
        layers.Add(layer);
        return layer;
    }

    public static NormedConv1DLayer<T> Conv<T>(List<LayerBase<T>> layers, int input, int output, int kernel, int stride = 1, int padding = 0)
    {
        var conv = new NormedConv1DLayer<T>(input, output, kernel, stride, 1, 1, padding, false, ConvolutionNormalization.None);
        layers.Add(conv);
        return conv;
    }

    /// <summary>A dense layer on the last axis of any rank.</summary>
    public static Tensor<T> Rows<T>(IEngine engine, DenseLayer<T> layer, Tensor<T> x)
    {
        var shape = x.Shape.ToArray();
        int last = shape[^1];
        var y = layer.Forward(engine.Reshape(x, new[] { x.Length / last, last }));
        shape[^1] = y.Shape[1];
        return engine.Reshape(y, shape);
    }

    /// <summary>A convolution over time of <c>[B, T, C]</c> (as the reference's <c>[B, C, T]</c> conv).</summary>
    public static Tensor<T> OverTime<T>(IEngine engine, NormedConv1DLayer<T> conv, Tensor<T> x)
    {
        var y = conv.Forward(engine.TensorPermute(x, new[] { 0, 2, 1 }));
        return engine.TensorPermute(y, new[] { 0, 2, 1 });
    }

    /// <summary>Layer norm over the last axis without affine parameters.</summary>
    public static Tensor<T> PlainLayerNorm<T>(IEngine engine, Tensor<T> x, double epsilon)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var shape = x.Shape.ToArray();
        int rank = shape.Length;
        var mean = engine.TensorBroadcastTo(engine.ReduceMean(x, new[] { rank - 1 }, true), shape);
        var centered = engine.TensorSubtract(x, mean);
        var variance = engine.ReduceMean(engine.TensorMultiply(centered, centered), new[] { rank - 1 }, true);
        var inverse = engine.TensorPower(engine.TensorAddScalar(variance, ops.FromDouble(epsilon)), ops.FromDouble(-0.5));
        return engine.TensorMultiply(centered, engine.TensorBroadcastTo(inverse, shape));
    }

    /// <summary>Sinusoids <c>[sin(x·e_k), cos(x·e_k)]</c> with <c>e_k = exp(−k ln 10000 / (half − 1))</c> (the release's
    /// <c>PosEmb</c>, and with a scale its <c>SinusPositionEmbedding</c>).</summary>
    public static double[] Sinusoid(double x, int dim, double scale = 1.0)
    {
        int half = dim / 2;
        double step = Math.Log(10000) / (half - 1);
        var e = new double[dim];
        for (int k = 0; k < half; k++)
        {
            double angle = scale * x * Math.Exp(k * -step);
            e[k] = Math.Sin(angle);
            e[half + k] = Math.Cos(angle);
        }
        return e;
    }

    public static void LoadDense<T>(IEngine engine, Func<string, double[]> read, DenseLayer<T> layer, string name, int inputs, int outputs) =>
        TorchParameters.Linear(engine, layer, inputs, outputs, read($"{name}.weight"), read($"{name}.bias"));

    public static void LoadConv<T>(Func<string, double[]> read, NormedConv1DLayer<T> conv, string name) =>
        conv.LoadTorchWeights(read($"{name}.weight"), null, read($"{name}.bias"));
}

/// <summary>
/// The release's <c>RelTransformerEncoder</c> (Glow-TTS's text encoder): token embeddings scaled by √hidden plus other
/// embeddings, a conv–ReLU–norm prenet, pre-norm layers of multi-head self-attention with windowed relative-position
/// keys and values (window 4) and a convolutional feed-forward (ReLU), and a last layer norm, all under the length mask.
/// Its layer norms are over channels with ε = 1e-4.
/// </summary>
internal sealed class MegaTts3RelEncoder<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly int _hidden, _heads, _window;

    internal sealed record Block(DenseLayer<T> Query, DenseLayer<T> Key, DenseLayer<T> Value, DenseLayer<T> Output,
        BiasFreeLinearLayer<T> RelativeKeys, BiasFreeLinearLayer<T> RelativeValues, LayerNormalizationLayer<T> Norm1,
        NormedConv1DLayer<T> Ffn1, NormedConv1DLayer<T> Ffn2, LayerNormalizationLayer<T> Norm2);

    public MegaTts3RelEncoder(IEngine engine, List<LayerBase<T>> layers, int vocabulary, int hidden, int filter, int heads, int blocks,
        int kernel, int window = 4)
    {
        _engine = engine;
        _hidden = hidden;
        _heads = heads;
        _window = window;
        Embedding = new TiedEmbeddingLayer<T>(vocabulary, hidden);
        layers.Add(Embedding);
        for (int i = 0; i < 3; i++)
        {
            PrenetConvs.Add(MegaTts3Ops.Conv(layers, hidden, hidden, 5, padding: 2));
            var norm = new LayerNormalizationLayer<T>(hidden, 1e-4);
            layers.Add(norm);
            PrenetNorms.Add(norm);
        }
        PrenetProjection = MegaTts3Ops.Conv(layers, hidden, hidden, 1);
        int k = hidden / heads;
        for (int i = 0; i < blocks; i++)
        {
            var q = MegaTts3Ops.Linear(layers, hidden);
            var key = MegaTts3Ops.Linear(layers, hidden);
            var v = MegaTts3Ops.Linear(layers, hidden);
            var o = MegaTts3Ops.Linear(layers, hidden);
            var rk = new BiasFreeLinearLayer<T>(1, (2 * window + 1) * k);
            var rv = new BiasFreeLinearLayer<T>(1, (2 * window + 1) * k);
            layers.Add(rk);
            layers.Add(rv);
            var n1 = new LayerNormalizationLayer<T>(hidden, 1e-4);
            layers.Add(n1);
            var f1 = MegaTts3Ops.Conv(layers, hidden, filter, kernel, padding: kernel / 2);
            var f2 = MegaTts3Ops.Conv(layers, filter, hidden, 1);
            var n2 = new LayerNormalizationLayer<T>(hidden, 1e-4);
            layers.Add(n2);
            Blocks.Add(new Block(q, key, v, o, rk, rv, n1, f1, f2, n2));
        }
        LastNorm = new LayerNormalizationLayer<T>(hidden, 1e-4);
        layers.Add(LastNorm);
    }

    public TiedEmbeddingLayer<T> Embedding { get; }
    public List<NormedConv1DLayer<T>> PrenetConvs { get; } = new();
    public List<LayerNormalizationLayer<T>> PrenetNorms { get; } = new();
    public NormedConv1DLayer<T> PrenetProjection { get; }
    public List<Block> Blocks { get; } = new();
    public LayerNormalizationLayer<T> LastNorm { get; }

    /// <summary>The encoding <c>[B, L, hidden]</c> of tokens <c>[B, L]</c> with added embeddings <c>[B, L, hidden]</c>;
    /// a row's mask covers its first (count of non-zero tokens) positions, as <c>sequence_mask</c> does.</summary>
    public Tensor<T> Forward(int[,] tokens, Tensor<T>? other)
    {
        int b = tokens.GetLength(0), l = tokens.GetLength(1), h = _hidden;
        var ids = new Tensor<T>(new[] { b * l });
        var maskValues = new Tensor<T>(new[] { b, l, 1 });
        for (int i = 0; i < b; i++)
        {
            int length = 0;
            for (int j = 0; j < l; j++)
            {
                ids[i * l + j] = NumOps.FromDouble(tokens[i, j]);
                if (tokens[i, j] > 0) length++;
            }
            for (int j = 0; j < length; j++) maskValues[i, j, 0] = NumOps.One;
        }
        var x = _engine.TensorMultiplyScalar(_engine.Reshape(Embedding.Forward(ids), new[] { b, l, h }), NumOps.FromDouble(Math.Sqrt(h)));
        if (other is not null) x = _engine.TensorAdd(x, other);
        var mask = _engine.TensorBroadcastTo(maskValues, new[] { b, l, h });
        // ConvReluNorm: three conv, layer norm, ReLU; a projection added to the input.
        var y = x;
        for (int i = 0; i < PrenetConvs.Count; i++)
            y = _engine.ReLU(PrenetNorms[i].Forward(MegaTts3Ops.OverTime(_engine, PrenetConvs[i], _engine.TensorMultiply(y, mask))));
        x = _engine.TensorMultiply(_engine.TensorAdd(x, MegaTts3Ops.OverTime(_engine, PrenetProjection, y)), mask);
        foreach (var block in Blocks)
        {
            x = _engine.TensorMultiply(x, mask);
            var attended = Attend(block, block.Norm1.Forward(x), maskValues);
            x = _engine.TensorAdd(x, attended);
            var ffn = MegaTts3Ops.OverTime(_engine, block.Ffn1, _engine.TensorMultiply(block.Norm2.Forward(x), mask));
            ffn = MegaTts3Ops.OverTime(_engine, block.Ffn2, _engine.TensorMultiply(_engine.ReLU(ffn), _engine.TensorBroadcastTo(maskValues, ffn.Shape.ToArray())));
            x = _engine.TensorAdd(x, _engine.TensorMultiply(ffn, mask));
        }
        return _engine.TensorMultiply(LastNorm.Forward(x), mask);
    }

    // Self-attention with relative keys and values: a key at distance d = j − i (|d| ≤ window) adds q·K_{d+window} to the
    // score and V_{d+window} to the output; masked scores are set to −1e4.
    private Tensor<T> Attend(Block block, Tensor<T> x, Tensor<T> maskValues)
    {
        int b = x.Shape[0], l = x.Shape[1], heads = _heads, k = _hidden / heads, w = _window, m = 2 * w + 1;
        Tensor<T> Heads(DenseLayer<T> projection) =>
            _engine.TensorPermute(_engine.Reshape(MegaTts3Ops.Rows(_engine, projection, x), new[] { b, l, heads, k }), new[] { 0, 2, 1, 3 });
        var q = Heads(block.Query);
        var key = Heads(block.Key);
        var v = Heads(block.Value);
        var scale = NumOps.FromDouble(1.0 / Math.Sqrt(k));
        var scores = _engine.TensorMultiplyScalar(_engine.BatchMatMul(_engine.Reshape(q, new[] { b * heads, l, k }),
            _engine.TensorPermute(_engine.Reshape(key, new[] { b * heads, l, k }), new[] { 0, 2, 1 })), scale);         // [bh, l, l]
        // Relative keys: q · K_m for every bucket m, then each (i, j) reads bucket j − i + w.
        var relKeys = _engine.Reshape(Lfm2NetworkPaged(block.RelativeKeys), new[] { m, k });
        var qRel = _engine.TensorMatMul(_engine.Reshape(q, new[] { b * heads * l, k }), _engine.TensorTranspose(relKeys)); // [bhl, m]
        var (onehot, _) = Buckets(l, w);                                                                                  // [l, l, m]
        // rel[bh, i, j] = Σ_m qRel[bh, i, m] onehot[i, j, m]: per query i a [bh, m] × [m, l] product.
        var qRelByQuery = _engine.TensorPermute(_engine.Reshape(qRel, new[] { b * heads, l, m }), new[] { 1, 0, 2 });       // [l, bh, m]
        var rel = _engine.BatchMatMul(qRelByQuery, _engine.TensorPermute(onehot, new[] { 0, 2, 1 }));                        // [l, bh, l]
        scores = _engine.TensorAdd(scores, _engine.TensorMultiplyScalar(_engine.TensorPermute(rel, new[] { 1, 0, 2 }), scale));
        // masked_fill(mask == 0, −1e4) with mask = x_mask_i · x_mask_j.
        var maskRow = _engine.Reshape(maskValues, new[] { b, 1, l, 1 });
        var maskCol = _engine.Reshape(maskValues, new[] { b, 1, 1, l });
        var pair = _engine.Reshape(_engine.TensorBroadcastTo(_engine.TensorMultiply(_engine.TensorBroadcastTo(maskRow, new[] { b, heads, l, l }),
            _engine.TensorBroadcastTo(maskCol, new[] { b, heads, l, l })), new[] { b, heads, l, l }), new[] { b * heads, l, l });
        var filled = _engine.TensorAdd(_engine.TensorMultiply(scores, pair),
            _engine.TensorMultiplyScalar(_engine.TensorAddScalar(_engine.TensorNegate(pair), NumOps.One), NumOps.FromDouble(-1e4)));
        var p = _engine.Softmax(filled, axis: 2);                                                                        // [bh, l, l]
        var output = _engine.BatchMatMul(p, _engine.Reshape(v, new[] { b * heads, l, k }));                               // [bh, l, k]
        // Relative values: per query, the weights gathered into buckets, then times V_m.
        var pByQuery = _engine.TensorPermute(p, new[] { 1, 0, 2 });                                                       // [l, bh, l]
        var bucketWeights = _engine.BatchMatMul(pByQuery, onehot);                                                        // [l, bh, m]
        var relValues = _engine.Reshape(Lfm2NetworkPaged(block.RelativeValues), new[] { m, k });
        var relOut = _engine.TensorMatMul(_engine.Reshape(_engine.TensorPermute(bucketWeights, new[] { 1, 0, 2 }), new[] { b * heads * l, m }), relValues);
        output = _engine.TensorAdd(output, _engine.Reshape(relOut, new[] { b * heads, l, k }));
        var merged = _engine.Reshape(_engine.TensorPermute(_engine.Reshape(output, new[] { b, heads, l, k }), new[] { 0, 2, 1, 3 }), new[] { b, l, _hidden });
        return MegaTts3Ops.Rows(_engine, block.Output, merged);
    }

    private static Tensor<T> Lfm2NetworkPaged(BiasFreeLinearLayer<T> holder) => AiDotNet.NeuralNetworks.Lfm2Network<T>.Paged(holder.Weights);

    // onehot[i, j, m] = 1 when j − i + w = m (|j − i| ≤ w).
    private static (Tensor<T> OneHot, int Buckets) Buckets(int length, int window)
    {
        int m = 2 * window + 1;
        var onehot = new Tensor<T>(new[] { length, length, m });
        for (int i = 0; i < length; i++)
            for (int j = Math.Max(0, i - window); j <= Math.Min(length - 1, i + window); j++) onehot[i, j, j - i + window] = NumOps.One;
        return (onehot, m);
    }

    public void LoadTorch(IEngine engine, Func<string, double[]> read, string prefix)
    {
        Embedding.LoadTable(read($"{prefix}.emb.weight"));
        for (int i = 0; i < PrenetConvs.Count; i++)
        {
            MegaTts3Ops.LoadConv(read, PrenetConvs[i], $"{prefix}.pre.conv_layers.{i}");
            TorchParameters.LayerNorm(engine, PrenetNorms[i], read($"{prefix}.pre.norm_layers.{i}.gamma"), read($"{prefix}.pre.norm_layers.{i}.beta"));
        }
        MegaTts3Ops.LoadConv(read, PrenetProjection, $"{prefix}.pre.proj");
        int h = _hidden;
        for (int i = 0; i < Blocks.Count; i++)
        {
            var bl = Blocks[i];
            string a = $"{prefix}.encoder.attn_layers.{i}";
            // 1×1 convolutions [out, in, 1] as dense layers.
            MegaTts3Ops.LoadDense(engine, read, bl.Query, $"{a}.conv_q", h, h);
            MegaTts3Ops.LoadDense(engine, read, bl.Key, $"{a}.conv_k", h, h);
            MegaTts3Ops.LoadDense(engine, read, bl.Value, $"{a}.conv_v", h, h);
            MegaTts3Ops.LoadDense(engine, read, bl.Output, $"{a}.conv_o", h, h);
            var rk = read($"{a}.emb_rel_k");
            var rv = read($"{a}.emb_rel_v");
            TorchParameters.Linear(bl.RelativeKeys, 1, rk.Length, rk);
            TorchParameters.Linear(bl.RelativeValues, 1, rv.Length, rv);
            TorchParameters.LayerNorm(engine, bl.Norm1, read($"{prefix}.encoder.norm_layers_1.{i}.gamma"), read($"{prefix}.encoder.norm_layers_1.{i}.beta"));
            MegaTts3Ops.LoadConv(read, bl.Ffn1, $"{prefix}.encoder.ffn_layers.{i}.conv_1");
            MegaTts3Ops.LoadConv(read, bl.Ffn2, $"{prefix}.encoder.ffn_layers.{i}.conv_2");
            TorchParameters.LayerNorm(engine, bl.Norm2, read($"{prefix}.encoder.norm_layers_2.{i}.gamma"), read($"{prefix}.encoder.norm_layers_2.{i}.beta"));
        }
        TorchParameters.LayerNorm(engine, LastNorm, read($"{prefix}.encoder.last_ln.gamma"), read($"{prefix}.encoder.last_ln.beta"));
    }
}

/// <summary>The shape of MegaTTS 3's diffusion transformer; the defaults are the release's (<c>dit.py</c>).</summary>
internal sealed record MegaTts3DitConfiguration
{
    public int Width { get; init; } = 1024;
    public int Layers { get; init; } = 24;
    public int Heads { get; init; } = 16;
    public int LatentChannels { get; init; } = 32;
    public int LocalConditionDim { get; init; } = 512;
    public int ContextMaskDim { get; init; } = 16;
    public int PhoneVocabulary { get; init; } = 302;
    public int ToneVocabulary { get; init; } = 32;
    public int PhoneEncoderLayers { get; init; } = 6;
    public int PhoneEncoderHeads { get; init; } = 4;
}

/// <summary>
/// MegaTTS 3's latent diffusion transformer (<c>Diffusion</c>): the phones encoded with their tones and positions,
/// expanded to frames and downsampled by 4 by two strided convolutions, added to the noisy latent's and the local
/// condition's (the prompt latent and its mask) projections; transformer blocks with AdaLN-Zero time conditioning,
/// interleaved complex rotary embeddings and SiLU feed-forward; the predicted velocity.
/// </summary>
internal sealed class MegaTts3Dit<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    internal sealed record Block(DenseLayer<T> Modulation, DenseLayer<T> Query, DenseLayer<T> Key, DenseLayer<T> Value, DenseLayer<T> Output,
        DenseLayer<T> W1, DenseLayer<T> W2);

    public MegaTts3Dit(IEngine engine, List<LayerBase<T>> layers, MegaTts3DitConfiguration configuration)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        if (c.Width % c.Heads != 0 || (c.Width / c.Heads) % 2 != 0)
            throw new ArgumentException("The width must divide into heads of even width.", nameof(configuration));
        ContextMaskProjection = MegaTts3Ops.Linear(layers, c.ContextMaskDim);
        LocalConditionProjection = MegaTts3Ops.Linear(layers, c.LocalConditionDim);
        int ff = 256 * ((2 * c.Width + 255) / 256);
        for (int i = 0; i < c.Layers; i++)
            Blocks.Add(new Block(MegaTts3Ops.Linear(layers, 6 * c.Width), MegaTts3Ops.Linear(layers, c.Width), MegaTts3Ops.Linear(layers, c.Width),
                MegaTts3Ops.Linear(layers, c.Width), MegaTts3Ops.Linear(layers, c.Width), MegaTts3Ops.Linear(layers, ff),
                MegaTts3Ops.Linear(layers, c.Width)));
        OutputModulation = MegaTts3Ops.Linear(layers, 2 * c.Width);
        OutputProjection = MegaTts3Ops.Linear(layers, c.Width);
        LatentPrenet = MegaTts3Ops.Linear(layers, c.Width);
        ConditionPrenet = MegaTts3Ops.Linear(layers, c.Width);
        Postnet = MegaTts3Ops.Linear(layers, c.LatentChannels);
        TimeMlp1 = MegaTts3Ops.Linear(layers, c.Width);
        TimeMlp2 = MegaTts3Ops.Linear(layers, c.Width);
        PhoneEncoder = new MegaTts3RelEncoder<T>(engine, layers, c.PhoneVocabulary, c.Width, 2 * c.Width, c.PhoneEncoderHeads, c.PhoneEncoderLayers, 3);
        ToneEmbedding = new TiedEmbeddingLayer<T>(c.ToneVocabulary, c.Width);
        layers.Add(ToneEmbedding);
        LingConvs.Add(MegaTts3Ops.Conv(layers, c.Width, c.Width, 4, stride: 2, padding: 1));
        LingConvs.Add(MegaTts3Ops.Conv(layers, c.Width, c.Width, 4, stride: 2, padding: 1));
    }

    public MegaTts3DitConfiguration Configuration { get; }
    public DenseLayer<T> ContextMaskProjection { get; }
    public DenseLayer<T> LocalConditionProjection { get; }
    public List<Block> Blocks { get; } = new();
    public DenseLayer<T> OutputModulation { get; }
    public DenseLayer<T> OutputProjection { get; }
    public DenseLayer<T> LatentPrenet { get; }
    public DenseLayer<T> ConditionPrenet { get; }
    public DenseLayer<T> Postnet { get; }
    public DenseLayer<T> TimeMlp1 { get; }
    public DenseLayer<T> TimeMlp2 { get; }
    public MegaTts3RelEncoder<T> PhoneEncoder { get; }
    public TiedEmbeddingLayer<T> ToneEmbedding { get; }
    public List<NormedConv1DLayer<T>> LingConvs { get; } = new();

    /// <summary><c>forward_ling_encoder</c>: the phone encoding <c>[B, L, width]</c> of phones and tones <c>[B, L]</c>
    /// (padding 0 masked out).</summary>
    public Tensor<T> EncodePhones(int[,] phones, int[,] tones)
    {
        int b = phones.GetLength(0), l = phones.GetLength(1), h = Configuration.Width;
        var toneIds = new Tensor<T>(new[] { b * l });
        var nonPadding = new Tensor<T>(new[] { b, l, 1 });
        for (int i = 0; i < b; i++)
            for (int j = 0; j < l; j++)
            {
                toneIds[i * l + j] = NumOps.FromDouble(tones[i, j]);
                if (phones[i, j] > 0) nonPadding[i, j, 0] = NumOps.One;
            }
        var positions = new Tensor<T>(new[] { b, l, h });
        for (int j = 0; j < l; j++)
        {
            var e = MegaTts3Ops.Sinusoid(j, h);
            for (int i = 0; i < b; i++)
                for (int d = 0; d < h; d++) positions[i, j, d] = NumOps.FromDouble(e[d]);
        }
        var mask = _engine.TensorBroadcastTo(nonPadding, new[] { b, l, h });
        var other = _engine.TensorMultiply(_engine.TensorAdd(_engine.Reshape(ToneEmbedding.Forward(toneIds), new[] { b, l, h }), positions), mask);
        return _engine.TensorMultiply(PhoneEncoder.Forward(phones, other), mask);
    }

    /// <summary>The phone encoding at the latent rate: expanded to frames by <paramref name="mel2ph"/> (1-based, 0 for
    /// none) and downsampled by the two strided convolutions.</summary>
    public Tensor<T> LinguisticFeatures(int[,] phones, int[,] tones, int[,] mel2ph)
    {
        var encoded = EncodePhones(phones, tones);
        int b = encoded.Shape[0], l = encoded.Shape[1], h = encoded.Shape[2], frames = mel2ph.GetLength(1);
        // expand_states: a zero row first, then gather by mel2ph.
        var padded = _engine.TensorConcatenate(new[] { new Tensor<T>(new[] { b, 1, h }), encoded }, 1);
        var index = new Tensor<int>(new[] { b * frames });
        for (int i = 0; i < b; i++)
            for (int f = 0; f < frames; f++) index[i * frames + f] = i * (l + 1) + mel2ph[i, f];
        var expanded = _engine.Reshape(_engine.TensorIndexSelect(_engine.Reshape(padded, new[] { b * (l + 1), h }), index, 0), new[] { b, frames, h });
        var x = expanded;
        foreach (var conv in LingConvs) x = MegaTts3Ops.OverTime(_engine, conv, x);
        return x;
    }

    /// <summary>The local condition <c>[B, frames, LocalConditionDim]</c> of the prompt latent (zero past the prompt) and
    /// its mask <c>[B, frames, 1]</c>.</summary>
    public Tensor<T> LocalCondition(Tensor<T> promptLatent, Tensor<T> contextMask) =>
        MegaTts3Ops.Rows(_engine, LocalConditionProjection,
            _engine.TensorConcatenate(new[] { promptLatent, MegaTts3Ops.Rows(_engine, ContextMaskProjection, contextMask) }, 2));

    /// <summary>The time embedding <c>[B, width]</c> (the release's <c>TimestepEmbedding</c>: 256 sinusoids of 1000·t, then a
    /// SiLU MLP).</summary>
    public Tensor<T> TimeEmbedding(IReadOnlyList<double> times)
    {
        var sinusoids = new Tensor<T>(new[] { times.Count, 256 });
        for (int i = 0; i < times.Count; i++)
        {
            var e = MegaTts3Ops.Sinusoid(times[i], 256, 1000);
            for (int d = 0; d < 256; d++) sinusoids[i, d] = NumOps.FromDouble(e[d]);
        }
        return TimeMlp2.Forward(_engine.Swish(TimeMlp1.Forward(sinusoids)));
    }

    /// <summary>The velocity <c>[B, frames, latent]</c> the transformer predicts for a noisy latent (masked out over the
    /// prompt), its local condition and linguistic features, at times <c>[B]</c> (one shared time is broadcast); a key
    /// mask <c>[B, frames]</c> of false hides padding.</summary>
    public Tensor<T> Velocity(Tensor<T> noisy, Tensor<T> contextMask, Tensor<T> localCondition, Tensor<T> linguistic, Tensor<T> time,
        bool[,]? keyMask = null)
    {
        var c = Configuration;
        int b = noisy.Shape[0], frames = noisy.Shape[1], h = c.Width;
        var keep = _engine.TensorAddScalar(_engine.TensorNegate(contextMask), NumOps.One);
        var x = _engine.TensorMultiply(noisy, _engine.TensorBroadcastTo(keep, noisy.Shape.ToArray()));
        x = _engine.TensorAdd(_engine.TensorAdd(MegaTts3Ops.Rows(_engine, LatentPrenet, x), MegaTts3Ops.Rows(_engine, ConditionPrenet, localCondition)), linguistic);
        if (time.Shape[0] != b) time = _engine.TensorBroadcastTo(time, new[] { b, h });
        var (cos, sin) = Rotary(frames, h / c.Heads);
        var bias = new Tensor<T>(new[] { b, 1, 1, frames });
        if (keyMask is not null)
            for (int i = 0; i < b; i++)
                for (int k = 0; k < frames; k++)
                    if (!keyMask[i, k]) bias[i, 0, 0, k] = NumOps.FromDouble(double.NegativeInfinity);
        foreach (var block in Blocks) x = RunBlock(block, x, time, cos, sin, bias);
        var modulation = OutputModulation.Forward(_engine.Swish(time));
        var scale = _engine.TensorSlice(modulation, new[] { 0, 0 }, new[] { b, h });
        var shift = _engine.TensorSlice(modulation, new[] { 0, h }, new[] { b, h });
        x = Modulate(MegaTts3Ops.PlainLayerNorm(_engine, x, 1e-6), scale, shift);
        x = MegaTts3Ops.Rows(_engine, OutputProjection, x);
        return MegaTts3Ops.Rows(_engine, Postnet, x);
    }

    // x · (1 + scale) + shift with scale and shift [B, H] broadcast over time.
    private Tensor<T> Modulate(Tensor<T> x, Tensor<T> scale, Tensor<T> shift)
    {
        var shape = x.Shape.ToArray();
        int b = shape[0], h = shape[2];
        var s = _engine.TensorBroadcastTo(_engine.Reshape(_engine.TensorAddScalar(scale, NumOps.One), new[] { b, 1, h }), shape);
        var t = _engine.TensorBroadcastTo(_engine.Reshape(shift, new[] { b, 1, h }), shape);
        return _engine.TensorAdd(_engine.TensorMultiply(x, s), t);
    }

    private Tensor<T> Gate(Tensor<T> x, Tensor<T> gate)
    {
        var shape = x.Shape.ToArray();
        return _engine.TensorMultiply(x, _engine.TensorBroadcastTo(_engine.Reshape(gate, new[] { shape[0], 1, shape[2] }), shape));
    }

    private Tensor<T> RunBlock(Block block, Tensor<T> x, Tensor<T> time, Tensor<T> cos, Tensor<T> sin, Tensor<T> bias)
    {
        int b = x.Shape[0], frames = x.Shape[1], h = Configuration.Width, heads = Configuration.Heads, hd = h / heads;
        var modulation = block.Modulation.Forward(_engine.Swish(time));
        Tensor<T> Chunk(int i) => _engine.TensorSlice(modulation, new[] { 0, i * h }, new[] { b, h });
        var normed = Modulate(MegaTts3Ops.PlainLayerNorm(_engine, x, 1e-6), Chunk(1), Chunk(0));
        Tensor<T> Heads(DenseLayer<T> projection, bool rotate)
        {
            var y = _engine.Reshape(MegaTts3Ops.Rows(_engine, projection, normed), new[] { b, frames, heads, hd });
            if (rotate) y = RotatePairs(y, cos, sin);
            return _engine.TensorPermute(y, new[] { 0, 2, 1, 3 });
        }
        var q = Heads(block.Query, true);
        var k = Heads(block.Key, true);
        var v = Heads(block.Value, false);
        var scores = _engine.TensorMultiplyScalar(_engine.BatchMatMul(_engine.Reshape(q, new[] { b * heads, frames, hd }),
            _engine.TensorPermute(_engine.Reshape(k, new[] { b * heads, frames, hd }), new[] { 0, 2, 1 })), NumOps.FromDouble(1.0 / Math.Sqrt(hd)));
        scores = _engine.TensorAdd(_engine.Reshape(scores, new[] { b, heads, frames, frames }), _engine.TensorBroadcastTo(bias, new[] { b, heads, frames, frames }));
        var p = _engine.Reshape(_engine.Softmax(scores, axis: 3), new[] { b * heads, frames, frames });
        var context = _engine.Reshape(_engine.TensorPermute(_engine.Reshape(_engine.BatchMatMul(p, _engine.Reshape(v, new[] { b * heads, frames, hd })),
            new[] { b, heads, frames, hd }), new[] { 0, 2, 1, 3 }), new[] { b, frames, h });
        var attended = MegaTts3Ops.Rows(_engine, block.Output, context);
        var hidden = _engine.TensorAdd(x, Gate(attended, Chunk(2)));
        var ffnIn = Modulate(MegaTts3Ops.PlainLayerNorm(_engine, hidden, 1e-6), Chunk(4), Chunk(3));
        var ffn = MegaTts3Ops.Rows(_engine, block.W2, _engine.Swish(MegaTts3Ops.Rows(_engine, block.W1, ffnIn)));
        return _engine.TensorAdd(hidden, Gate(ffn, Chunk(5)));
    }

    // freqs_cis for positions 0..frames−1: cos and sin of position · θ^(−2k/hd) per pair k, [frames, hd/2].
    private (Tensor<T> Cos, Tensor<T> Sin) Rotary(int frames, int hd)
    {
        int half = hd / 2;
        var cos = new Tensor<T>(new[] { 1, frames, 1, half });
        var sin = new Tensor<T>(new[] { 1, frames, 1, half });
        for (int t = 0; t < frames; t++)
            for (int k = 0; k < half; k++)
            {
                double angle = t * (1.0 / Math.Pow(10000.0, 2.0 * k / hd));
                cos[0, t, 0, k] = NumOps.FromDouble(Math.Cos(angle));
                sin[0, t, 0, k] = NumOps.FromDouble(Math.Sin(angle));
            }
        return (cos, sin);
    }

    // apply_rotary_emb on consecutive pairs (2k, 2k+1) as complex numbers: (a cos − b sin, a sin + b cos).
    private Tensor<T> RotatePairs(Tensor<T> x, Tensor<T> cos, Tensor<T> sin)
    {
        var shape = x.Shape.ToArray();
        int b = shape[0], frames = shape[1], heads = shape[2], hd = shape[3], half = hd / 2;
        var pairs = _engine.Reshape(x, new[] { b, frames, heads, half, 2 });
        var re = _engine.Reshape(_engine.TensorSlice(pairs, new[] { 0, 0, 0, 0, 0 }, new[] { b, frames, heads, half, 1 }), new[] { b, frames, heads, half });
        var im = _engine.Reshape(_engine.TensorSlice(pairs, new[] { 0, 0, 0, 0, 1 }, new[] { b, frames, heads, half, 1 }), new[] { b, frames, heads, half });
        var c = _engine.TensorBroadcastTo(cos, new[] { b, frames, heads, half });
        var s = _engine.TensorBroadcastTo(sin, new[] { b, frames, heads, half });
        var outRe = _engine.TensorSubtract(_engine.TensorMultiply(re, c), _engine.TensorMultiply(im, s));
        var outIm = _engine.TensorAdd(_engine.TensorMultiply(re, s), _engine.TensorMultiply(im, c));
        return _engine.Reshape(_engine.TensorConcatenate(new[] { _engine.Reshape(outRe, new[] { b, frames, heads, half, 1 }),
            _engine.Reshape(outIm, new[] { b, frames, heads, half, 1 }) }, 4), shape);
    }

    /// <summary><c>_forward</c>'s multi-condition CFG: the predictions for (speaker and text, text, neither) combined as
    /// <c>u + w_text (c_t − u) + w_speaker (c_st − c_t)</c>.</summary>
    public Tensor<T> GuidedVelocity(Tensor<T> noisy, Tensor<T> contextMask, Tensor<T> localCondition, Tensor<T> linguistic, double time,
        double textWeight, double speakerWeight)
    {
        int frames = noisy.Shape[1], channels = noisy.Shape[2];
        var batch = _engine.TensorConcatenate(new[] { noisy, noisy, noisy }, 0);
        var pred = Velocity(batch, contextMask, localCondition, linguistic, TimeEmbedding(new[] { time }));
        Tensor<T> Part(int i) => _engine.TensorSlice(pred, new[] { i, 0, 0 }, new[] { 1, frames, channels });
        var speakerText = Part(0);
        var text = Part(1);
        var none = Part(2);
        return _engine.TensorAdd(_engine.TensorAdd(none, _engine.TensorMultiplyScalar(_engine.TensorSubtract(text, none), NumOps.FromDouble(textWeight))),
            _engine.TensorMultiplyScalar(_engine.TensorSubtract(speakerText, text), NumOps.FromDouble(speakerWeight)));
    }

    /// <summary>
    /// <c>Diffusion.inference</c>: from Gaussian noise, <paramref name="steps"/> steps on the sway schedule
    /// <c>t = 1 − cos(πs/2)</c> (s uniform) of the release's amo sampler: predict at <c>o = min(t' + 3(t' − t), 1)</c>,
    /// then <c>x' = (t'/o) x̂_o + √((1 − t')² − ((t'/o)(1 − o))²) ε</c>. <paramref name="draw"/> gives each Gaussian draw.
    /// </summary>
    public Tensor<T> Sample(Tensor<T> contextMask, Tensor<T> localCondition, Tensor<T> linguistic, int steps, double textWeight,
        double speakerWeight, Func<int[], Tensor<T>> draw)
    {
        int frames = localCondition.Shape[1], channels = Configuration.LatentChannels;
        var schedule = new double[steps + 1];
        for (int i = 0; i <= steps; i++)
        {
            double s = i / (double)steps;
            schedule[i] = s - (Math.Cos(Math.PI / 2 * s) - 1 + s);
        }
        var x = draw(new[] { 1, frames, channels });
        for (int step = 0; step < steps; step++)
        {
            double t = schedule[step], next = schedule[step + 1];
            var v = GuidedVelocity(x, contextMask, localCondition, linguistic, t, textWeight, speakerWeight);
            double o = Math.Min(next + 3 * (next - t), 1);
            var predicted = _engine.TensorAdd(x, _engine.TensorMultiplyScalar(v, NumOps.FromDouble(o - t)));
            double a = next / o, bCoefficient = Math.Sqrt(Math.Pow(1 - next, 2) - Math.Pow(a * (1 - o), 2));
            var noise = draw(new[] { 1, frames, channels });
            x = _engine.TensorAdd(_engine.TensorMultiplyScalar(predicted, NumOps.FromDouble(a)), _engine.TensorMultiplyScalar(noise, NumOps.FromDouble(bCoefficient)));
        }
        return x;
    }

    public void LoadTorch(IEngine engine, Func<string, double[]> read)
    {
        var c = Configuration;
        int h = c.Width, ff = 256 * ((2 * c.Width + 255) / 256);
        void Dense(DenseLayer<T> layer, string name, int inputs, int outputs) => MegaTts3Ops.LoadDense(engine, read, layer, name, inputs, outputs);
        Dense(ContextMaskProjection, "ctx_mask_proj", 1, c.ContextMaskDim);
        Dense(LocalConditionProjection, "local_cond_project", c.LatentChannels + c.ContextMaskDim, c.LocalConditionDim);
        for (int i = 0; i < Blocks.Count; i++)
        {
            var b = Blocks[i];
            string p = $"encoder.layers.{i}";
            Dense(b.Modulation, $"{p}.attention_norm.linear", h, 6 * h);
            Dense(b.Query, $"{p}.attention.wq", h, h);
            Dense(b.Key, $"{p}.attention.wk", h, h);
            Dense(b.Value, $"{p}.attention.wv", h, h);
            Dense(b.Output, $"{p}.attention.wo", h, h);
            Dense(b.W1, $"{p}.feed_forward.w1", h, ff);
            Dense(b.W2, $"{p}.feed_forward.w2", ff, h);
        }
        Dense(OutputModulation, "encoder.norm.linear", h, 2 * h);
        Dense(OutputProjection, "encoder.out_proj", h, h);
        Dense(LatentPrenet, "x_prenet", c.LatentChannels, h);
        Dense(ConditionPrenet, "prenet", c.LocalConditionDim, h);
        Dense(Postnet, "postnet", h, c.LatentChannels);
        Dense(TimeMlp1, "f5_time_embed.time_mlp.0", 256, h);
        Dense(TimeMlp2, "f5_time_embed.time_mlp.2", h, h);
        PhoneEncoder.LoadTorch(engine, read, "ph_encoder");
        ToneEmbedding.LoadTable(read("tone_embed.weight"));
        for (int i = 0; i < LingConvs.Count; i++) MegaTts3Ops.LoadConv(read, LingConvs[i], $"ling_pre_net.{i}");
    }
}
