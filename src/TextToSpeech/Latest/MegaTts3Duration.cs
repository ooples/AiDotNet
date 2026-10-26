using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.TextToSpeech.Latest;

/// <summary>The shape of MegaTTS 3's duration predictor; the defaults are the release's (<c>duration_lm/config.yaml</c>).</summary>
internal sealed record MegaTts3DurationConfiguration
{
    public int PhoneVocabulary { get; init; } = 174;
    public int EncoderWidth { get; init; } = 512;
    public int EncoderFilter { get; init; } = 1024;
    public int EncoderHeads { get; init; } = 2;
    public int EncoderLayers { get; init; } = 4;
    public int EncoderKernel { get; init; } = 3;
    public int ToneVocabulary { get; init; } = 23;
    public int DecoderWidth { get; init; } = 512;
    public int DecoderLayers { get; init; } = 8;
    public int DecoderHeads { get; init; } = 8;
    public int CodeSize { get; init; } = 128;
}

/// <summary>The keys and values each decoder layer has read (before the rotary embedding), for incremental decoding.</summary>
internal sealed class MegaTts3DurationCache<T>
{
    internal readonly List<(Tensor<T> Keys, Tensor<T> Values)?> Layers = new();

    public MegaTts3DurationCache<T> Copy() => new MegaTts3DurationCache<T> { }.With(Layers);

    private MegaTts3DurationCache<T> With(List<(Tensor<T>, Tensor<T>)?> layers)
    {
        Layers.AddRange(layers);
        return this;
    }
}

/// <summary>
/// MegaTTS 3's autoregressive duration predictor (<c>ARDurPredictor</c> with <c>ar_mse</c>): the phones encoded by a
/// relative-attention encoder with their tone embeddings and positions, projected to the decoder; pre-norm decoder layers
/// of causal self-attention with rotary embeddings (fairseq's packed projections, no biases) and a GELU feed-forward; a
/// final layer norm and a softplus regression of each phone's duration, read back as the next step's input token.
/// </summary>
internal sealed class MegaTts3Duration<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    internal sealed record Layer(LayerNormalizationLayer<T> Norm1, BiasFreeLinearLayer<T> InProjection, BiasFreeLinearLayer<T> OutProjection,
        LayerNormalizationLayer<T> Norm2, DenseLayer<T> Ffn1, DenseLayer<T> Ffn2);

    public MegaTts3Duration(IEngine engine, List<LayerBase<T>> layers, MegaTts3DurationConfiguration configuration)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        if (c.DecoderWidth % c.DecoderHeads != 0 || (c.DecoderWidth / c.DecoderHeads) % 2 != 0)
            throw new ArgumentException("The decoder width must divide into heads of even width.", nameof(configuration));
        Encoder = new MegaTts3RelEncoder<T>(engine, layers, c.PhoneVocabulary, c.EncoderWidth, c.EncoderFilter, c.EncoderHeads, c.EncoderLayers, c.EncoderKernel);
        ToneEmbedding = new TiedEmbeddingLayer<T>(c.ToneVocabulary, c.EncoderWidth);
        layers.Add(ToneEmbedding);
        EncoderProjection = MegaTts3Ops.Linear(layers, c.DecoderWidth);
        CodeEmbedding = new TiedEmbeddingLayer<T>(c.CodeSize + 2, c.DecoderWidth);
        layers.Add(CodeEmbedding);
        for (int i = 0; i < c.DecoderLayers; i++)
        {
            var n1 = new LayerNormalizationLayer<T>(c.DecoderWidth, 1e-5);
            var inProjection = new BiasFreeLinearLayer<T>(c.DecoderWidth, 3 * c.DecoderWidth);
            var outProjection = new BiasFreeLinearLayer<T>(c.DecoderWidth, c.DecoderWidth);
            var n2 = new LayerNormalizationLayer<T>(c.DecoderWidth, 1e-5);
            layers.Add(n1);
            layers.Add(inProjection);
            layers.Add(outProjection);
            layers.Add(n2);
            Layers.Add(new Layer(n1, inProjection, outProjection, n2, MegaTts3Ops.Linear(layers, 4 * c.DecoderWidth), MegaTts3Ops.Linear(layers, c.DecoderWidth)));
        }
        FinalNorm = new LayerNormalizationLayer<T>(c.DecoderWidth, 1e-5);
        layers.Add(FinalNorm);
        Output = MegaTts3Ops.Linear(layers, 1);
    }

    public MegaTts3DurationConfiguration Configuration { get; }
    public MegaTts3RelEncoder<T> Encoder { get; }
    public TiedEmbeddingLayer<T> ToneEmbedding { get; }
    public DenseLayer<T> EncoderProjection { get; }
    public TiedEmbeddingLayer<T> CodeEmbedding { get; }
    public List<Layer> Layers { get; } = new();
    public LayerNormalizationLayer<T> FinalNorm { get; }
    public DenseLayer<T> Output { get; }

    /// <summary><c>forward_ling_encoder</c> for one utterance's phones and tones (all non-zero): <c>[L, decoder]</c>.</summary>
    public Tensor<T> EncodeLinguistic(IReadOnlyList<int> phones, IReadOnlyList<int> tones)
    {
        int l = phones.Count, h = Configuration.EncoderWidth;
        var toneIds = new Tensor<T>(new[] { l });
        var positions = new Tensor<T>(new[] { 1, l, h });
        for (int j = 0; j < l; j++)
        {
            toneIds[j] = NumOps.FromDouble(tones[j]);
            var e = MegaTts3Ops.Sinusoid(j, h);
            for (int d = 0; d < h; d++) positions[0, j, d] = NumOps.FromDouble(e[d]);
        }
        var other = _engine.TensorAdd(_engine.Reshape(ToneEmbedding.Forward(toneIds), new[] { 1, l, h }), positions);
        var row = new int[1, l];
        for (int j = 0; j < l; j++) row[0, j] = phones[j];
        var encoded = Encoder.Forward(row, other);
        return _engine.Reshape(MegaTts3Ops.Rows(_engine, EncoderProjection, encoded), new[] { l, Configuration.DecoderWidth });
    }

    private Tensor<T> Code(IReadOnlyList<int> codes)
    {
        var ids = new Tensor<T>(new[] { codes.Count });
        for (int i = 0; i < codes.Count; i++) ids[i] = NumOps.FromDouble(codes[i]);
        return CodeEmbedding.Forward(ids);
    }

    // RotaryEmbeddings: rotate-half with cos and sin of position · 10000^(−2k/width), the angles repeated for both halves.
    private Tensor<T> Rotate(Tensor<T> x, IReadOnlyList<int> positions)
    {
        int heads = x.Shape[0], l = x.Shape[1], hd = x.Shape[2], half = hd / 2;
        var cos = new Tensor<T>(new[] { 1, l, hd });
        var sin = new Tensor<T>(new[] { 1, l, hd });
        for (int j = 0; j < l; j++)
            for (int k = 0; k < half; k++)
            {
                double angle = positions[j] * Math.Pow(10000.0, -2.0 * k / hd);
                cos[0, j, k] = cos[0, j, k + half] = NumOps.FromDouble(Math.Cos(angle));
                sin[0, j, k] = sin[0, j, k + half] = NumOps.FromDouble(Math.Sin(angle));
            }
        var first = _engine.TensorSlice(x, new[] { 0, 0, 0 }, new[] { heads, l, half });
        var second = _engine.TensorSlice(x, new[] { 0, 0, half }, new[] { heads, l, half });
        var rotated = _engine.TensorConcatenate(new[] { _engine.TensorNegate(second), first }, 2);
        var shape = new[] { heads, l, hd };
        return _engine.TensorAdd(_engine.TensorMultiply(_engine.TensorBroadcastTo(cos, shape), x), _engine.TensorMultiply(_engine.TensorBroadcastTo(sin, shape), rotated));
    }

    // The decoder over inputs [L, decoder] with query positions; with a cache the keys take positions 0..cached+L−1 and
    // there is no mask, without one the keys take the query positions under a causal mask (−1e8).
    private Tensor<T> Decode(Tensor<T> x, IReadOnlyList<int> queryPositions, MegaTts3DurationCache<T>? cache)
    {
        var c = Configuration;
        int l = x.Shape[0], h = c.DecoderWidth, heads = c.DecoderHeads, hd = h / heads;
        if (cache is not null)
            while (cache.Layers.Count < Layers.Count) cache.Layers.Add(null);
        for (int i = 0; i < Layers.Count; i++)
        {
            var layer = Layers[i];
            var normed = layer.Norm1.Forward(x);
            var qkv = layer.InProjection.Forward(normed);                                                       // [L, 3h]
            Tensor<T> Part(int p) => _engine.TensorPermute(_engine.Reshape(_engine.TensorSlice(qkv, new[] { 0, p * h }, new[] { l, h }),
                new[] { l, heads, hd }), new[] { 1, 0, 2 });                                                       // [heads, L, hd]
            var q = _engine.TensorMultiplyScalar(Part(0), NumOps.FromDouble(1.0 / Math.Sqrt(hd)));
            var k = Part(1);
            var v = Part(2);
            q = Rotate(q, queryPositions);
            IReadOnlyList<int> keyPositions = queryPositions;
            if (cache is not null)
            {
                if (cache.Layers[i] is { } held)
                {
                    k = _engine.TensorConcatenate(new[] { held.Keys, k }, 1);
                    v = _engine.TensorConcatenate(new[] { held.Values, v }, 1);
                }
                cache.Layers[i] = (k, v);
                keyPositions = Enumerable.Range(0, k.Shape[1]).ToArray();
            }
            int keys = k.Shape[1];
            var scores = _engine.BatchMatMul(q, _engine.TensorPermute(Rotate(k, keyPositions), new[] { 0, 2, 1 }));       // [heads, L, keys]
            if (cache is null)
            {
                var mask = new Tensor<T>(new[] { 1, l, keys });
                for (int a = 0; a < l; a++)
                    for (int b = a + 1; b < keys; b++) mask[0, a, b] = NumOps.FromDouble(-1e8);
                scores = _engine.TensorAdd(scores, _engine.TensorBroadcastTo(mask, new[] { heads, l, keys }));
            }
            var context = _engine.BatchMatMul(_engine.Softmax(scores, axis: 2), v);                                        // [heads, L, hd]
            var merged = _engine.Reshape(_engine.TensorPermute(context, new[] { 1, 0, 2 }), new[] { l, h });
            x = _engine.TensorAdd(x, layer.OutProjection.Forward(merged));
            var ffn = layer.Ffn2.Forward(ActivationHelper.ExactGelu(layer.Ffn1.Forward(layer.Norm2.Forward(x)), _engine));
            x = _engine.TensorAdd(x, ffn);
        }
        var y = Output.Forward(FinalNorm.Forward(x));                                                                       // [L, 1]
        return _engine.Reshape(_engine.Softplus(y), new[] { l });
    }

    /// <summary>A training forward: the duration regression <c>[L]</c> for previous tokens (the first is
    /// <c>CodeSize + 1</c>) at query and key positions.</summary>
    public Tensor<T> Forward(Tensor<T> linguistic, IReadOnlyList<int> previous, IReadOnlyList<int> positions) =>
        Decode(_engine.TensorAdd(Code(previous), linguistic), positions, null);

    /// <summary>
    /// <c>ARDurPredictor.infer</c> with an incremental state: teacher-forced on <paramref name="context"/> while it lasts,
    /// otherwise each step's rounded prediction (at least 1, except the first step's) is the next input; the start token
    /// is <c>CodeSize + 1</c>, or <paramref name="firstInput"/> when continuing a state. Returns the durations and each
    /// step's prediction.
    /// </summary>
    public (int[] Durations, double[] Steps) Infer(Tensor<T> linguistic, MegaTts3DurationCache<T> cache, IReadOnlyList<int>? context, int? firstInput,
        IReadOnlyList<int> positions)
    {
        int l = linguistic.Shape[0], h = Configuration.DecoderWidth;
        var decoded = new int[l + 1];
        decoded[0] = firstInput ?? Configuration.CodeSize + 1;
        var steps = new double[l];
        for (int step = 0; step < l; step++)
        {
            var input = _engine.TensorAdd(Code(new[] { decoded[step] }), _engine.TensorSlice(linguistic, new[] { step, 0 }, new[] { 1, h }));
            var prediction = Decode(input, new[] { positions[step] }, cache);
            steps[step] = NumOps.ToDouble(prediction[0]);
            if (context is not null && step < context.Count) decoded[step + 1] = context[step];
            else
            {
                int rounded = (int)Math.Round(steps[step], MidpointRounding.ToEven);
                decoded[step + 1] = step == 0 ? Math.Max(rounded, 0) : Math.Max(rounded, 1);
            }
        }
        return (decoded.Skip(1).ToArray(), steps);
    }

    /// <summary>Loads a release checkpoint's <c>dur_model</c> state (the character encoder, unused by the release's
    /// inference, is not read).</summary>
    public void LoadTorch(IEngine engine, Func<string, double[]> read)
    {
        var c = Configuration;
        int h = c.DecoderWidth;
        Encoder.LoadTorch(engine, read, "encoder");
        ToneEmbedding.LoadTable(read("ling_label_embed_layers.tone.weight"));
        MegaTts3Ops.LoadDense(engine, read, EncoderProjection, "enc_proj", c.EncoderWidth, h);
        CodeEmbedding.LoadTable(read("code_emb.weight"));
        for (int i = 0; i < Layers.Count; i++)
        {
            var layer = Layers[i];
            string p = $"layers.{i}.op";
            TorchParameters.LayerNorm(engine, layer.Norm1, read($"{p}.layer_norm1.weight"), read($"{p}.layer_norm1.bias"));
            TorchParameters.Linear(layer.InProjection, h, 3 * h, read($"{p}.self_attn.in_proj_weight"));
            TorchParameters.Linear(layer.OutProjection, h, h, read($"{p}.self_attn.out_proj.weight"));
            TorchParameters.LayerNorm(engine, layer.Norm2, read($"{p}.layer_norm2.weight"), read($"{p}.layer_norm2.bias"));
            // ffn_1 is a kernel-1 convolution [4h, h, 1]: a dense layer.
            MegaTts3Ops.LoadDense(engine, read, layer.Ffn1, $"{p}.ffn.ffn_1.1", h, 4 * h);
            MegaTts3Ops.LoadDense(engine, read, layer.Ffn2, $"{p}.ffn.ffn_2", 4 * h, h);
        }
        TorchParameters.LayerNorm(engine, FinalNorm, read("layer_norm.weight"), read("layer_norm.bias"));
        MegaTts3Ops.LoadDense(engine, read, Output, "project_out_dim.0", h, 1);
    }
}
