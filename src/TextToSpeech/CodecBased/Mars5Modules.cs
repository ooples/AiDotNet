using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.EndToEnd;
using AiDotNet.TextToSpeech.Latest;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>The sizes of MARS5's autoregressive model (<c>CodecLM(n_vocab, dim=1536, dim_ff_scale=7/3)</c>). The head width is
/// Mistral's <c>ModelArgs</c> default of 64, which <c>CodecLM</c> never overrides (it equals dim/heads only at 1536/24).</summary>
internal sealed record Mars5ArConfiguration(int Vocab, int Dim = 1536, int Heads = 24, int Layers = 26, int SpeakerLayers = 2,
    int Hidden = 3584, int SlidingWindow = 3000, int Codebooks = 8, int Codes = 1024, int HeadDim = 64);

/// <summary>The sizes of MARS5's non-autoregressive model (<c>ResidualTransformer(n_text_vocab, n_quant=1025)</c>).</summary>
internal sealed record Mars5NarConfiguration(int TextVocab, int Classes = 1025, int Dim = 1024, int Heads = 16, int EncoderLayers = 8,
    int DecoderLayers = 16, int SpeakerLayers = 3, int Codebooks = 8, int TimeDim = 1024, double CondDrop = 0.1);

/// <summary>
/// MARS5's <c>SinePositionalEmbedding(dim, scale=False, alpha=True)</c>: <c>x + α · pe</c> with sines on the even and
/// cosines on the odd channels, the table computed in float32, α learned.
/// </summary>
internal sealed class Mars5SinePositions<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly int _dim;

    public Mars5SinePositions(IEngine engine, List<LayerBase<T>> layers, int dim)
    {
        _engine = engine;
        _dim = dim;
        Alpha = IndexTtsOps.Values(layers, new[] { 1.0 });
    }

    public BiasFreeLinearLayer<T> Alpha { get; }

    private Tensor<T> Table(int length)
    {
        var pe = new Tensor<T>(new[] { length, _dim });
        for (int i = 0; i < _dim / 2; i++)
        {
            float div = MathF.Exp(2 * i * -(float)(Math.Log(10000.0) / _dim));
            for (int p = 0; p < length; p++)
            {
                float a = p * div;
                pe[p, 2 * i] = NumOps.FromDouble(MathF.Sin(a));
                pe[p, 2 * i + 1] = NumOps.FromDouble(MathF.Cos(a));
            }
        }
        return pe;
    }

    /// <summary>The rows <c>[L, dim]</c> with their positions added.</summary>
    public Tensor<T> Forward(Tensor<T> rows)
    {
        int length = rows.Shape[0];
        var alpha = _engine.TensorBroadcastTo(_engine.Reshape(IndexTtsOps.Value(Alpha), new[] { 1, 1 }), new[] { length, _dim });
        return _engine.TensorAdd(rows, _engine.TensorMultiply(alpha, Table(length)));
    }

    public void Load(Func<string, double[]?> read, string name) =>
        TorchParameters.Linear(Alpha, 1, 1, read(name) ?? throw new KeyNotFoundException($"MARS5's checkpoint has no '{name}'."));
}

/// <summary>
/// A pre-norm <c>nn.TransformerEncoderLayer</c> or <c>nn.TransformerDecoderLayer</c> as MARS5 builds them: its
/// <c>linear1</c> replaced by the identity and its activation by <c>FNNSwiGLU</c> (<c>silu(xW) ⊙ xV</c>, bias-free), so the
/// feed-forward is <c>linear2(silu(xW) ⊙ xV)</c>; LayerNorm ε 4e-5; packed multi-head projections. The decoder layer
/// adds cross-attention to the memory and has no causal mask.
/// </summary>
internal sealed class Mars5SwiGluLayer<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly int _dim, _heads;

    public Mars5SwiGluLayer(IEngine engine, List<LayerBase<T>> layers, Random random, int dim, int heads, int ff, bool decoder)
    {
        _engine = engine;
        _dim = dim;
        _heads = heads;
        Norm1 = IndexTtsOps.LayerNorm(layers, dim, Mars5Ops.LayerNormEpsilon);
        SelfIn = IndexTtsOps.Linear(engine, layers, random, dim, 3 * dim);
        SelfOut = IndexTtsOps.Linear(engine, layers, random, dim, dim);
        if (decoder)
        {
            NormCross = IndexTtsOps.LayerNorm(layers, dim, Mars5Ops.LayerNormEpsilon);
            CrossIn = IndexTtsOps.Linear(engine, layers, random, dim, 3 * dim);
            CrossOut = IndexTtsOps.Linear(engine, layers, random, dim, dim);
        }
        NormFeedForward = IndexTtsOps.LayerNorm(layers, dim, Mars5Ops.LayerNormEpsilon);
        GateW = IndexTtsOps.BiasFree(layers, random, dim, ff);
        GateV = IndexTtsOps.BiasFree(layers, random, dim, ff);
        Linear2 = IndexTtsOps.Linear(engine, layers, random, ff, dim);
    }

    public LayerNormalizationLayer<T> Norm1 { get; }
    public DenseLayer<T> SelfIn { get; }
    public DenseLayer<T> SelfOut { get; }
    public LayerNormalizationLayer<T>? NormCross { get; }
    public DenseLayer<T>? CrossIn { get; }
    public DenseLayer<T>? CrossOut { get; }
    public LayerNormalizationLayer<T> NormFeedForward { get; }
    public BiasFreeLinearLayer<T> GateW { get; }
    public BiasFreeLinearLayer<T> GateV { get; }
    public DenseLayer<T> Linear2 { get; }

    // nn.MultiheadAttention with packed projections: q from the query rows, k and v from the key rows.
    private Tensor<T> Attend(DenseLayer<T> inProjection, DenseLayer<T> outProjection, Tensor<T> query, Tensor<T> keys, Tensor<T>? keyBias)
    {
        var q = _engine.TensorNarrow(inProjection.Forward(query), 1, 0, _dim);
        var kv = inProjection.Forward(keys);
        var k = _engine.TensorNarrow(kv, 1, _dim, _dim);
        var v = _engine.TensorNarrow(kv, 1, 2 * _dim, _dim);
        Tensor<T>? bias = keyBias is null ? null : _engine.TensorBroadcastTo(keyBias, new[] { query.Shape[0], keys.Shape[0] });
        var attended = IndexTtsOps.Attend(_engine, IndexTtsOps.SplitHeads(_engine, q, _heads), IndexTtsOps.SplitHeads(_engine, k, _heads),
            IndexTtsOps.SplitHeads(_engine, v, _heads), 1.0 / Math.Sqrt(_dim / _heads), bias);
        return outProjection.Forward(IndexTtsOps.MergeHeads(_engine, attended));
    }

    /// <summary>The layer on rows <c>[L, dim]</c>; <paramref name="keyBias"/> <c>[1, L]</c> masks padded keys (−∞), and a
    /// decoder layer attends to <paramref name="memory"/> <c>[M, dim]</c> with <paramref name="memoryBias"/>.</summary>
    public Tensor<T> Forward(Tensor<T> x, Tensor<T>? keyBias, Tensor<T>? memory = null, Tensor<T>? memoryBias = null)
    {
        var h = Norm1.Forward(x);
        x = _engine.TensorAdd(x, Attend(SelfIn, SelfOut, h, h, keyBias));
        if (CrossIn is not null)
        {
            if (memory is null) throw new ArgumentNullException(nameof(memory), "A decoder layer attends to the encoder's memory.");
            x = _engine.TensorAdd(x, Attend(CrossIn, CrossOut!, NormCross!.Forward(x), memory, memoryBias));
        }
        h = NormFeedForward.Forward(x);
        var gated = _engine.TensorMultiply(_engine.Swish(GateW.Forward(h)), GateV.Forward(h));
        return _engine.TensorAdd(x, Linear2.Forward(gated));
    }

    public void Load(Func<string, double[]?> read, string prefix)
    {
        double[] R(string n) => read(prefix + n) ?? throw new KeyNotFoundException($"MARS5's checkpoint has no '{prefix + n}'.");
        int ff = GateW.GetParameters().Length / _dim;
        TorchParameters.LayerNorm(_engine, Norm1, R("norm1.weight"), R("norm1.bias"));
        TorchParameters.Linear(_engine, SelfIn, _dim, 3 * _dim, R("self_attn.in_proj_weight"), R("self_attn.in_proj_bias"));
        TorchParameters.Linear(_engine, SelfOut, _dim, _dim, R("self_attn.out_proj.weight"), R("self_attn.out_proj.bias"));
        string ffNorm = "norm2";
        if (CrossIn is not null)
        {
            TorchParameters.LayerNorm(_engine, NormCross!, R("norm2.weight"), R("norm2.bias"));
            TorchParameters.Linear(_engine, CrossIn, _dim, 3 * _dim, R("multihead_attn.in_proj_weight"), R("multihead_attn.in_proj_bias"));
            TorchParameters.Linear(_engine, CrossOut!, _dim, _dim, R("multihead_attn.out_proj.weight"), R("multihead_attn.out_proj.bias"));
            ffNorm = "norm3";
        }
        TorchParameters.LayerNorm(_engine, NormFeedForward, R(ffNorm + ".weight"), R(ffNorm + ".bias"));
        TorchParameters.Linear(GateW, _dim, ff, R("activation.W.weight"));
        TorchParameters.Linear(GateV, _dim, ff, R("activation.V.weight"));
        TorchParameters.Linear(_engine, Linear2, ff, _dim, R("linear2.weight"), R("linear2.bias"));
    }
}

/// <summary>Shared pieces of MARS5's networks.</summary>
internal static class Mars5Ops
{
    /// <summary>MARS5's <c>LAYERNORM_EPS</c>.</summary>
    public const double LayerNormEpsilon = 4e-5;

    /// <summary>A stack of SwiGLU layers with a final LayerNorm (<c>nn.TransformerEncoder(..., norm=LayerNorm)</c>).</summary>
    public static List<Mars5SwiGluLayer<T>> Stack<T>(IEngine engine, List<LayerBase<T>> layers, Random random, int dim, int heads, int count, bool decoder,
        out LayerNormalizationLayer<T> norm)
    {
        int ff = (int)(dim * 4 * (3.0 / 4));
        var stack = new List<Mars5SwiGluLayer<T>>();
        for (int i = 0; i < count; i++) stack.Add(new Mars5SwiGluLayer<T>(engine, layers, random, dim, heads, ff, decoder));
        norm = IndexTtsOps.LayerNorm(layers, dim, LayerNormEpsilon);
        return stack;
    }

    /// <summary><c>ChunkedEmbedding</c>: each codebook's code through its own table of width dim / codebooks, the results
    /// concatenated: rows <c>[L, dim]</c> of codes <c>[L, codebooks]</c>.</summary>
    public static Tensor<T> Chunked<T>(IEngine engine, IReadOnlyList<TiedEmbeddingLayer<T>> tables, int[,] codes)
    {
        int length = codes.GetLength(0);
        var parts = new Tensor<T>[tables.Count];
        for (int q = 0; q < tables.Count; q++)
        {
            var ids = new int[length];
            for (int i = 0; i < length; i++) ids[i] = codes[i, q];
            parts[q] = tables[q].Forward(IndexTtsOps.Ids<T>(ids));
        }
        return parts.Length == 1 ? parts[0] : engine.TensorConcatenate(parts, 1);
    }

    /// <summary>A key-padding bias <c>[1, L]</c>: 0 for attended keys, a large negative number for masked ones.</summary>
    public static Tensor<T>? KeyBias<T>(bool[] masked)
    {
        if (!masked.Any(m => m)) return null;
        var ops = MathHelper.GetNumericOperations<T>();
        var bias = new Tensor<T>(new[] { 1, masked.Length });
        for (int i = 0; i < masked.Length; i++) bias[0, i] = masked[i] ? ops.FromDouble(-1e30) : ops.Zero;
        return bias;
    }

    /// <summary>The speaker encoders' implicit embedding <c>[1, dim]</c>: a learned identity token before the chunked
    /// embedding of the reference codes, positions, the SwiGLU encoder with padded keys masked, the identity token's row.</summary>
    public static Tensor<T> SpeakerEmbedding<T>(IEngine engine, IReadOnlyList<TiedEmbeddingLayer<T>> tables, BiasFreeLinearLayer<T> identity,
        Mars5SinePositions<T> positions, IReadOnlyList<Mars5SwiGluLayer<T>> encoder, LayerNormalizationLayer<T> norm, int[,] codes, bool[] masked)
    {
        int dim = identity.GetParameters().Length;
        var rows = engine.TensorConcatenate(new[] { engine.Reshape(IndexTtsOps.Value(identity), new[] { 1, dim }), Chunked(engine, tables, codes) }, 0);
        rows = positions.Forward(rows);
        var bias = KeyBias<T>(new[] { false }.Concat(masked).ToArray());
        foreach (var layer in encoder) rows = layer.Forward(rows, bias);
        return engine.TensorNarrow(norm.Forward(rows), 0, 0, 1);
    }

    public static void LoadTables<T>(Func<string, double[]?> read, IReadOnlyList<TiedEmbeddingLayer<T>> tables, string prefix)
    {
        for (int q = 0; q < tables.Count; q++)
            tables[q].LoadLeadingRows(read($"{prefix}embs.{q}.weight") ?? throw new KeyNotFoundException($"MARS5's checkpoint has no '{prefix}embs.{q}.weight'."));
    }

    public static void LoadStack<T>(IEngine engine, Func<string, double[]?> read, IReadOnlyList<Mars5SwiGluLayer<T>> stack, LayerNormalizationLayer<T> norm, string prefix)
    {
        for (int i = 0; i < stack.Count; i++) stack[i].Load(read, $"{prefix}layers.{i}.");
        TorchParameters.LayerNorm(engine, norm, read(prefix + "norm.weight") ?? throw new KeyNotFoundException(prefix + "norm.weight"),
            read(prefix + "norm.bias") ?? throw new KeyNotFoundException(prefix + "norm.bias"));
    }
}

/// <summary>
/// MARS5's autoregressive model (<c>CodecLM</c>): a speaker embedding from the reference's eight-codebook EnCodec codes
/// (a two-layer SwiGLU encoder over a learned identity token and the chunked code embeddings) placed before the token
/// embeddings of the text and BPE-merged first-codebook speech tokens, through a Mistral decoder (RMSNorm, rotary
/// positions, SwiGLU, sliding-window causal attention), the speaker position's output removed.
/// </summary>
internal sealed class Mars5CodecLm<T>
{
    private readonly IEngine _engine;

    public Mars5CodecLm(IEngine engine, List<LayerBase<T>> layers, Mars5ArConfiguration configuration, Random random)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        if (c.Dim % c.Codebooks != 0) throw new ArgumentException("The width must divide by the codebooks (ChunkedEmbedding).", nameof(configuration));
        Decoder = new LlamaNetwork<T>(engine, layers, new LlamaConfiguration
        {
            VocabSize = c.Vocab, Hidden = c.Dim, Layers = c.Layers, Heads = c.Heads, KeyValueHeads = c.Heads, Intermediate = c.Hidden, RmsEpsilon = 1e-5,
            HeadDim = c.HeadDim,
        });
        Positions = new Mars5SinePositions<T>(engine, layers, c.Dim);
        for (int q = 0; q < c.Codebooks; q++) ReferenceTables.Add(IndexTtsOps.Embedding(layers, random, c.Codes + 1, c.Dim / c.Codebooks, 1.0));
        Identity = IndexTtsOps.Values(layers, Enumerable.Range(0, c.Dim).Select(_ => IndexTtsOps.Gaussian(random)).ToArray());
        SpeakerEncoder = Mars5Ops.Stack(engine, layers, random, c.Dim, c.Heads, c.SpeakerLayers, decoder: false, out var norm);
        SpeakerNorm = norm;
    }

    public Mars5ArConfiguration Configuration { get; }

    /// <summary>The Mistral decoder (<c>ar</c>), whose token table is <c>embed</c>.</summary>
    public LlamaNetwork<T> Decoder { get; }

    public Mars5SinePositions<T> Positions { get; }
    public List<TiedEmbeddingLayer<T>> ReferenceTables { get; } = new();
    public BiasFreeLinearLayer<T> Identity { get; }
    public List<Mars5SwiGluLayer<T>> SpeakerEncoder { get; }
    public LayerNormalizationLayer<T> SpeakerNorm { get; }

    /// <summary>The implicit speaker embedding <c>[1, dim]</c> of reference codes <c>[L, codebooks]</c> (1024 pads; a frame
    /// is padding from the first pad of the first codebook on, <c>construct_padding_mask</c>).</summary>
    public Tensor<T> Speaker(int[,] codes)
    {
        int length = codes.GetLength(0);
        var masked = new bool[length];
        bool seen = false;
        for (int i = 0; i < length; i++) masked[i] = seen |= codes[i, 0] == Configuration.Codes;
        return Mars5Ops.SpeakerEmbedding(_engine, ReferenceTables, Identity, Positions, SpeakerEncoder, SpeakerNorm, codes, masked);
    }

    private Tensor<T> Sequence(IReadOnlyList<int> tokens, Tensor<T> speaker)
    {
        var ids = new int[1, tokens.Count];
        for (int i = 0; i < tokens.Count; i++) ids[0, i] = tokens[i];
        var embedded = _engine.Reshape(Decoder.Embed(ids), new[] { tokens.Count, Configuration.Dim });
        int length = tokens.Count + 1;
        if (length > Configuration.SlidingWindow + 1)
            throw new ArgumentException($"MARS5 attends over a {Configuration.SlidingWindow}-position window; the sequence has {length} positions.", nameof(tokens));
        return _engine.Reshape(_engine.TensorConcatenate(new[] { speaker, embedded }, 0), new[] { 1, length, Configuration.Dim });
    }

    /// <summary><c>forward(x, spk_reference=...)</c>: the logits <c>[L, vocab]</c> after each token (the speaker position's
    /// removed).</summary>
    public Tensor<T> Logits(IReadOnlyList<int> tokens, Tensor<T> speaker)
    {
        var (_, logits) = Decoder.Forward(Sequence(tokens, speaker));
        int length = tokens.Count + 1;
        return _engine.TensorNarrow(_engine.Reshape(logits!, new[] { length, Configuration.Vocab }), 0, 1, tokens.Count);
    }

    /// <summary>The next-token logits <c>[vocab]</c> of a sequence, read through <paramref name="cache"/>: the whole prefix
    /// on the first call, then the new token (the release's kv-cached decoding).</summary>
    public Tensor<T> NextLogits(IReadOnlyList<int> tokens, Tensor<T> speaker, LlamaCache<T> cache)
    {
        Tensor<T> input = cache.Length == 0
            ? Sequence(tokens, speaker)
            : _engine.Reshape(Decoder.Embed(new[,] { { tokens[tokens.Count - 1] } }), new[] { 1, 1, Configuration.Dim });
        var (_, logits) = Decoder.Forward(input, null, null, cache, lastLogitsOnly: true);
        return _engine.Reshape(logits!, new[] { Configuration.Vocab });
    }

    /// <summary>
    /// Loads the AR checkpoint (<c>mars5_ar.safetensors</c>): the Mistral weights renamed to Hugging Face's and their
    /// query and key rows permuted from Mistral's interleaved rotary pairs to the half-split layout the decoder rotates.
    /// </summary>
    public void Load(Func<string, double[]?> read)
    {
        var c = Configuration;
        int headDim = c.HeadDim;
        double[] Require(string name) => read(name) ?? throw new KeyNotFoundException($"MARS5's AR checkpoint has no '{name}'.");
        double[] Permute(double[] w)
        {
            // w.view(heads, hd/2, 2, dim).transpose(1, 2).reshape(heads·hd, dim)
            var p = new double[w.Length];
            int half = headDim / 2;
            for (int h = 0; h < c.Heads; h++)
                for (int i = 0; i < half; i++)
                    for (int two = 0; two < 2; two++)
                    {
                        int from = h * headDim + 2 * i + two, to = h * headDim + two * half + i;
                        Array.Copy(w, from * c.Dim, p, to * c.Dim, c.Dim);
                    }
            return p;
        }
        Decoder.LoadTorchWeights((name, _) =>
        {
            if (name == "model.embed_tokens.weight") return Require("embed.weight");
            if (name == "model.norm.weight") return Require("ar.norm.weight");
            if (name == "lm_head.weight") return Require("ar.output.weight");
            var parts = name.Split('.');                                                            // model.layers.{i}.…
            string n = $"ar.layers.{parts[2]}.";
            string rest = string.Join(".", parts.Skip(3));
            return rest switch
            {
                "input_layernorm.weight" => Require(n + "attention_norm.weight"),
                "post_attention_layernorm.weight" => Require(n + "ffn_norm.weight"),
                "self_attn.q_proj.weight" => Permute(Require(n + "attention.wq.weight")),
                "self_attn.k_proj.weight" => Permute(Require(n + "attention.wk.weight")),
                "self_attn.v_proj.weight" => Require(n + "attention.wv.weight"),
                "self_attn.o_proj.weight" => Require(n + "attention.wo.weight"),
                "mlp.gate_proj.weight" => Require(n + "feed_forward.w1.weight"),
                "mlp.up_proj.weight" => Require(n + "feed_forward.w3.weight"),
                "mlp.down_proj.weight" => Require(n + "feed_forward.w2.weight"),
                _ => throw new KeyNotFoundException($"MARS5's AR decoder has no counterpart of '{name}'."),
            };
        });
        Positions.Load(read, "pos_embedding.alpha");
        Mars5Ops.LoadTables(read, ReferenceTables, "ref_chunked_emb.");
        TorchParameters.Linear(Identity, 1, c.Dim, Require("spk_identity_emb.weight"));
        Mars5Ops.LoadStack(_engine, read, SpeakerEncoder, SpeakerNorm, "spk_encoder.");
    }
}

/// <summary>
/// MARS5's non-autoregressive model (<c>ResidualTransformer</c>): an encoder over the speaker embedding of the
/// reference's codes followed by the text's token embeddings, and a decoder over the chunked embedding of all eight
/// codebooks of the noisy codes, each side with sine positions and its own projection of the diffusion timestep's
/// sinusoidal embedding; one LayerNorm and linear head per codebook predicts x₀'s logits.
/// </summary>
internal sealed class Mars5ResidualTransformer<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    public Mars5ResidualTransformer(IEngine engine, List<LayerBase<T>> layers, Mars5NarConfiguration configuration, Random random)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        if (c.Dim % c.Codebooks != 0) throw new ArgumentException("The width must divide by the codebooks (ChunkedEmbedding).", nameof(configuration));
        ConditionPositions = new Mars5SinePositions<T>(engine, layers, c.Dim);
        Positions = new Mars5SinePositions<T>(engine, layers, c.Dim);
        Encoder = Mars5Ops.Stack(engine, layers, random, c.Dim, c.Heads, c.EncoderLayers, decoder: false, out var encoderNorm);
        EncoderNorm = encoderNorm;
        Decoder = Mars5Ops.Stack(engine, layers, random, c.Dim, c.Heads, c.DecoderLayers, decoder: true, out var decoderNorm);
        DecoderNorm = decoderNorm;
        TimeEncoderIn = IndexTtsOps.Linear(engine, layers, random, c.TimeDim, c.Dim);
        TimeEncoderOut = IndexTtsOps.Linear(engine, layers, random, c.Dim, c.Dim);
        TimeDecoderIn = IndexTtsOps.Linear(engine, layers, random, c.TimeDim, c.Dim);
        TimeDecoderOut = IndexTtsOps.Linear(engine, layers, random, c.Dim, c.Dim);
        TextEmbedding = IndexTtsOps.Embedding(layers, random, c.TextVocab, c.Dim, 1.0);
        for (int q = 0; q < c.Codebooks; q++) ReferenceTables.Add(IndexTtsOps.Embedding(layers, random, c.Classes, c.Dim / c.Codebooks, 1.0));
        ReferencePositions = new Mars5SinePositions<T>(engine, layers, c.Dim);
        Identity = IndexTtsOps.Values(layers, Enumerable.Range(0, c.Dim).Select(_ => IndexTtsOps.Gaussian(random)).ToArray());
        SpeakerEncoder = Mars5Ops.Stack(engine, layers, random, c.Dim, c.Heads, c.SpeakerLayers, decoder: false, out var speakerNorm);
        SpeakerNorm = speakerNorm;
        for (int q = 0; q < c.Codebooks; q++) ResidualTables.Add(IndexTtsOps.Embedding(layers, random, c.Classes, c.Dim / c.Codebooks, 1.0));
        for (int q = 0; q < c.Codebooks; q++)
            Heads.Add((IndexTtsOps.LayerNorm(layers, c.Dim, 1e-5), IndexTtsOps.Linear(engine, layers, random, c.Dim, c.Classes)));
    }

    public Mars5NarConfiguration Configuration { get; }
    public Mars5SinePositions<T> ConditionPositions { get; }
    public Mars5SinePositions<T> Positions { get; }
    public List<Mars5SwiGluLayer<T>> Encoder { get; }
    public LayerNormalizationLayer<T> EncoderNorm { get; }
    public List<Mars5SwiGluLayer<T>> Decoder { get; }
    public LayerNormalizationLayer<T> DecoderNorm { get; }
    public DenseLayer<T> TimeEncoderIn { get; }
    public DenseLayer<T> TimeEncoderOut { get; }
    public DenseLayer<T> TimeDecoderIn { get; }
    public DenseLayer<T> TimeDecoderOut { get; }
    public TiedEmbeddingLayer<T> TextEmbedding { get; }
    public List<TiedEmbeddingLayer<T>> ReferenceTables { get; } = new();
    public Mars5SinePositions<T> ReferencePositions { get; }
    public BiasFreeLinearLayer<T> Identity { get; }
    public List<Mars5SwiGluLayer<T>> SpeakerEncoder { get; }
    public LayerNormalizationLayer<T> SpeakerNorm { get; }
    public List<TiedEmbeddingLayer<T>> ResidualTables { get; } = new();
    public List<(LayerNormalizationLayer<T> Norm, DenseLayer<T> Linear)> Heads { get; } = new();

    /// <summary><c>timestep_embedding(t, dim)</c>: <c>[cos(t·f), sin(t·f)]</c> with f = exp(−ln(10⁴)·i/half), all in float32
    /// (the arange is integer, so torch computes the table, the product and cos/sin in its default float32).</summary>
    private Tensor<T> TimestepEmbedding(int t)
    {
        int dim = Configuration.TimeDim, half = dim / 2;
        var e = new Tensor<T>(new[] { 1, dim });
        float scale = (float)-Math.Log(10000);
        for (int i = 0; i < half; i++)
        {
            float f = MathF.Exp(scale * i / half);
            float a = t * f;
            e[0, i] = NumOps.FromDouble(MathF.Cos(a));
            e[0, half + i] = NumOps.FromDouble(MathF.Sin(a));
        }
        if (dim % 2 == 1) e[0, dim - 1] = NumOps.Zero;
        return e;
    }

    /// <summary>
    /// <c>forward(c_text, c_codes, ..., x, x_padding_mask, t, drop_cond)</c> for one example: the logits <c>[L, codebooks,
    /// classes]</c> of x₀ given the text tokens, the reference codes <c>[R, codebooks]</c> and the noisy codes
    /// <c>[L, codebooks]</c> at step <paramref name="t"/>. The conditioning is dropped (codes set to the pad class) when
    /// <paramref name="dropCondition"/>, or in training with the condition-drop probability.
    /// </summary>
    public Tensor<T> Forward(IReadOnlyList<int> text, int[,] reference, int[,] x, int t, bool dropCondition, bool training = false, Random? random = null)
    {
        var c = Configuration;
        bool drop = training ? (random ?? throw new ArgumentNullException(nameof(random))).NextDouble() < c.CondDrop : dropCondition;
        int referenceLength = reference.GetLength(0);
        var codes = reference;
        if (drop)
        {
            codes = new int[referenceLength, c.Codebooks];
            for (int i = 0; i < referenceLength; i++)
                for (int q = 0; q < c.Codebooks; q++) codes[i, q] = c.Classes - 1;
        }
        // length_to_mask(c_codes_length + 1): the identity token and the reference frames (none when dropped).
        var masked = Enumerable.Range(0, referenceLength).Select(_ => drop).ToArray();
        var speaker = Mars5Ops.SpeakerEmbedding(_engine, ReferenceTables, Identity, ReferencePositions, SpeakerEncoder, SpeakerNorm, codes, masked);
        var time = TimestepEmbedding(t);
        var timeEncoder = TimeEncoderOut.Forward(_engine.Swish(TimeEncoderIn.Forward(time)));
        var timeDecoder = TimeDecoderOut.Forward(_engine.Swish(TimeDecoderIn.Forward(time)));
        var condition = _engine.TensorConcatenate(new[] { speaker, TextEmbedding.Forward(IndexTtsOps.Ids<T>(text)) }, 0);
        condition = ConditionPositions.Forward(condition);
        condition = _engine.TensorAdd(condition, _engine.TensorBroadcastTo(timeEncoder, condition.Shape.ToArray()));
        var target = Positions.Forward(Mars5Ops.Chunked(_engine, ResidualTables, x));
        target = _engine.TensorAdd(target, _engine.TensorBroadcastTo(timeDecoder, target.Shape.ToArray()));
        var memory = condition;
        foreach (var layer in Encoder) memory = layer.Forward(memory, null);
        memory = EncoderNorm.Forward(memory);
        var h = target;
        foreach (var layer in Decoder) h = layer.Forward(h, null, memory, null);
        h = DecoderNorm.Forward(h);
        int length = x.GetLength(0);
        var heads = Heads.Select(head => _engine.Reshape(head.Linear.Forward(head.Norm.Forward(h)), new[] { length, 1, c.Classes })).ToArray();
        return heads.Length == 1 ? heads[0] : _engine.TensorConcatenate(heads, 1);
    }

    /// <summary>Loads the NAR checkpoint (<c>mars5_nar.safetensors</c>).</summary>
    public void Load(Func<string, double[]?> read)
    {
        var c = Configuration;
        double[] Require(string name) => read(name) ?? throw new KeyNotFoundException($"MARS5's NAR checkpoint has no '{name}'.");
        ConditionPositions.Load(read, "cond_pos_embedding.alpha");
        Positions.Load(read, "pos_embedding.alpha");
        Mars5Ops.LoadStack(_engine, read, Encoder, EncoderNorm, "tfm.encoder.");
        Mars5Ops.LoadStack(_engine, read, Decoder, DecoderNorm, "tfm.decoder.");
        TorchParameters.Linear(_engine, TimeEncoderIn, c.TimeDim, c.Dim, Require("timestep_encoder_emb.0.weight"), Require("timestep_encoder_emb.0.bias"));
        TorchParameters.Linear(_engine, TimeEncoderOut, c.Dim, c.Dim, Require("timestep_encoder_emb.2.weight"), Require("timestep_encoder_emb.2.bias"));
        TorchParameters.Linear(_engine, TimeDecoderIn, c.TimeDim, c.Dim, Require("timestep_decoder_emb.0.weight"), Require("timestep_decoder_emb.0.bias"));
        TorchParameters.Linear(_engine, TimeDecoderOut, c.Dim, c.Dim, Require("timestep_decoder_emb.2.weight"), Require("timestep_decoder_emb.2.bias"));
        TextEmbedding.LoadLeadingRows(Require("text_embed.weight"));
        Mars5Ops.LoadTables(read, ReferenceTables, "ref_embedder.");
        ReferencePositions.Load(read, "ref_pos_embedding.alpha");
        TorchParameters.Linear(Identity, 1, c.Dim, Require("spk_identity_emb.weight"));
        Mars5Ops.LoadStack(_engine, read, SpeakerEncoder, SpeakerNorm, "spk_encoder.");
        Mars5Ops.LoadTables(read, ResidualTables, "residual_encoder.");
        for (int q = 0; q < Heads.Count; q++)
        {
            TorchParameters.LayerNorm(_engine, Heads[q].Norm, Require($"residual_decoder.{q}.0.weight"), Require($"residual_decoder.{q}.0.bias"));
            TorchParameters.Linear(_engine, Heads[q].Linear, c.Dim, c.Classes, Require($"residual_decoder.{q}.1.weight"), Require($"residual_decoder.{q}.1.bias"));
        }
    }
}
