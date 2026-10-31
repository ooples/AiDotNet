using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.Latest;

namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>The sizes and regularization of a SpeechT5 model (the official <c>t5_transformer</c> arguments).</summary>
internal sealed record SpeechT5Configuration
{
    public int Hidden { get; init; } = 768;
    public int FeedForward { get; init; } = 3072;
    public int Heads { get; init; } = 12;
    public int EncoderLayers { get; init; } = 12;
    public int DecoderLayers { get; init; } = 6;
    public int MaxRelativePosition { get; init; } = 160;
    public int TextVocabulary { get; init; } = 81;
    public int PadId { get; init; } = 1;
    public int EosId { get; init; } = 2;
    public int MaxTextPositions { get; init; } = 600;
    public int MaxSpeechPositions { get; init; } = 1876;
    public int MelChannels { get; init; } = 80;
    public int ReductionFactor { get; init; } = 2;
    public int SpeakerDim { get; init; } = 512;
    public int PrenetUnits { get; init; } = 256;
    public int PrenetLayers { get; init; } = 2;
    public double PrenetDropout { get; init; } = 0.5;
    public int PostnetChannels { get; init; } = 256;
    public int PostnetLayers { get; init; } = 5;
    public int PostnetKernel { get; init; } = 5;
    public double PostnetDropout { get; init; } = 0.5;
    public (int Channels, int Kernel, int Stride)[] FeatureLayers { get; init; } =
        { (512, 10, 5), (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 2, 2), (512, 2, 2) };
    public int ConvPositionKernel { get; init; } = 128;
    public int ConvPositionGroups { get; init; } = 16;
    public int FinalDim { get; init; } = 256;
    public int[] HubertClasses { get; init; } = { 504 };
    public double LogitTemperature { get; init; } = 0.1;
    public int LatentVars { get; init; } = 100;
    public int LatentGroups { get; init; } = 2;
    public double Dropout { get; init; } = 0.1;
    public double AttentionDropout { get; init; } = 0.1;
    public double ActivationDropout { get; init; } = 0.1;
    public double EncoderLayerDrop { get; init; }
    public double DecoderLayerDrop { get; init; }
    public double PositionalDropout { get; init; } = 0.1;
}

/// <summary>Shared operations of the SpeechT5 modules (rows <c>[T, C]</c>).</summary>
internal static class SpeechT5Ops
{
    private static readonly double Log10000 = Math.Log(10000.0);

    /// <summary>ESPnet's <c>PositionalEncoding</c> table <c>[T, d]</c>: <c>sin</c> at even columns and <c>cos</c> at odd,
    /// frequencies <c>exp(−2k ln(10⁴)/d)</c>, positions from 0.</summary>
    public static Tensor<T> EspnetTable<T>(int length, int d, int start = 0)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var table = new Tensor<T>(new[] { length, d });
        for (int p = 0; p < length; p++)
            for (int k = 0; k < d; k += 2)
            {
                double f = Math.Exp(k * -(Log10000 / d));
                table[p, k] = ops.FromDouble(Math.Sin((start + p) * f));
                if (k + 1 < d) table[p, k + 1] = ops.FromDouble(Math.Cos((start + p) * f));
            }
        return table;
    }

    /// <summary>fairseq's <c>SinusoidalPositionalEmbedding</c> rows <c>[T, d]</c> of positions <paramref name="start"/> …
    /// (tensor2tensor's halves: <c>[sin(p·f), cos(p·f)]</c> with <c>f_k = exp(−k ln(10⁴)/(d/2 − 1))</c>).</summary>
    public static Tensor<T> FairseqTable<T>(int length, int d, int start)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int half = d / 2;
        var table = new Tensor<T>(new[] { length, d });
        for (int p = 0; p < length; p++)
            for (int k = 0; k < half; k++)
            {
                double angle = (start + p) * Math.Exp(k * -(Log10000 / (half - 1)));
                table[p, k] = ops.FromDouble(Math.Sin(angle));
                table[p, half + k] = ops.FromDouble(Math.Cos(angle));
            }
        return table;
    }

    public static Tensor<T> Drop<T>(IEngine engine, Tensor<T> x, double rate, bool training, Random? random) =>
        training && rate > 0 && random is not null ? T5Seq2Seq<T>.Dropout(engine, x, rate, random) : x;

    public static Tensor<T> Gelu<T>(IEngine engine, Tensor<T> x) => ActivationHelper.ExactGelu(x, engine);

    /// <summary><c>F.normalize</c> of a vector (L2, with the norm floored at 1e-12).</summary>
    public static Tensor<T> Normalize<T>(IEngine engine, Tensor<T> v)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var norm = engine.TensorMax(engine.TensorSqrt(engine.ReduceSum(engine.TensorMultiply(v, v), null, true)), ops.FromDouble(1e-12));
        return engine.TensorDivide(v, engine.TensorBroadcastTo(norm, v.Shape.ToArray()));
    }
}

/// <summary>
/// The fairseq multi-head attention SpeechT5 uses: biased projections, the query scaled by <c>d_k^-1/2</c> before the
/// scores, and, for the encoder's self-attention, a relative-position term <c>q_i · pe_k[i − j]</c> added to the scores.
/// </summary>
internal sealed class SpeechT5Attention<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly int _d, _heads;

    public SpeechT5Attention(IEngine engine, List<LayerBase<T>> layers, Random random, int d, int heads)
    {
        _engine = engine;
        _d = d;
        _heads = heads;
        // fairseq's MultiheadAttention: Xavier-uniform with gain 1/√2 for the q, k, v projections, plain for the output.
        DenseLayer<T> Xavier(double gain)
        {
            var layer = IndexTtsOps.Linear(engine, layers, random, d, d);
            double bound = gain * Math.Sqrt(6.0 / (d + d));
            TorchParameters.Linear(engine, layer, d, d, Enumerable.Range(0, d * d).Select(_ => bound * (2 * random.NextDouble() - 1)).ToArray(), new double[d]);
            return layer;
        }
        Key = Xavier(1 / Math.Sqrt(2));
        Value = Xavier(1 / Math.Sqrt(2));
        Query = Xavier(1 / Math.Sqrt(2));
        Output = Xavier(1.0);
    }

    public DenseLayer<T> Query { get; }
    public DenseLayer<T> Key { get; }
    public DenseLayer<T> Value { get; }
    public DenseLayer<T> Output { get; }

    /// <summary>
    /// Attention of queries <c>[Lq, d]</c> over keys and values <c>[Lk, d]</c> (or, with <paramref name="cache"/>, the cached
    /// ones followed by these). <paramref name="positions"/> <c>[Lq, Lk, d_k]</c> adds the relative term;
    /// <paramref name="causal"/> hides later keys. Returns the output and the per-head weights <c>[H, Lq, Lk]</c>.
    /// </summary>
    public (Tensor<T> Output, Tensor<T> Weights) Forward(Tensor<T> queries, Tensor<T> keys, Tensor<T>? positions, bool causal,
        (Tensor<T> K, Tensor<T> V)? staticKeys, List<(Tensor<T> K, Tensor<T> V)>? cache, int cacheIndex, double dropout, bool training, Random? random)
    {
        int dk = _d / _heads, lq = queries.Shape[0];
        var q = IndexTtsOps.SplitHeads(_engine, _engine.TensorMultiplyScalar(Query.Forward(queries), NumOps.FromDouble(1.0 / Math.Sqrt(dk))), _heads);
        Tensor<T> k, v;
        if (staticKeys is { } s)
        {
            k = s.K;
            v = s.V;
        }
        else
        {
            k = IndexTtsOps.SplitHeads(_engine, Key.Forward(keys), _heads);
            v = IndexTtsOps.SplitHeads(_engine, Value.Forward(keys), _heads);
            if (cache is not null)
            {
                if (cacheIndex < cache.Count)
                {
                    k = _engine.TensorConcatenate(new[] { cache[cacheIndex].K, k }, 1);
                    v = _engine.TensorConcatenate(new[] { cache[cacheIndex].V, v }, 1);
                    cache[cacheIndex] = (k, v);
                }
                else
                {
                    cache.Add((k, v));
                }
            }
        }
        int lk = k.Shape[1];
        var scores = _engine.BatchMatMul(q, _engine.TensorPermute(k, new[] { 0, 2, 1 }));                  // [H, Lq, Lk]
        if (positions is not null)
        {
            // B[h, i, j] = q[h, i] · pe[i, j]: per query position, [H, dk] × [dk, Lk].
            var qi = _engine.TensorPermute(q, new[] { 1, 0, 2 });                                           // [Lq, H, dk]
            var bias = _engine.BatchMatMul(qi, _engine.TensorPermute(positions, new[] { 0, 2, 1 }));        // [Lq, H, Lk]
            scores = _engine.TensorAdd(scores, _engine.TensorPermute(bias, new[] { 1, 0, 2 }));
        }
        if (causal && lq > 1)
        {
            var mask = new Tensor<T>(new[] { lq, lk });
            var blocked = NumOps.FromDouble(double.NegativeInfinity);
            for (int i = 0; i < lq; i++)
                for (int j = lk - lq + i + 1; j < lk; j++) mask[i, j] = blocked;
            scores = _engine.TensorAdd(scores, _engine.TensorBroadcastTo(_engine.Reshape(mask, new[] { 1, lq, lk }), new[] { _heads, lq, lk }));
        }
        var weights = _engine.Softmax(scores, axis: 2);
        var context = IndexTtsOps.MergeHeads(_engine, _engine.BatchMatMul(SpeechT5Ops.Drop(_engine, weights, dropout, training, random), v));
        return (Output.Forward(context), weights);
    }

    /// <summary>The projected keys and values <c>[H, Lk, dk]</c> of an encoder output, for cross-attention.</summary>
    public (Tensor<T> K, Tensor<T> V) StaticKeys(Tensor<T> memory) =>
        (IndexTtsOps.SplitHeads(_engine, Key.Forward(memory), _heads), IndexTtsOps.SplitHeads(_engine, Value.Forward(memory), _heads));

    public void Load(Func<string, int[], double[]> read, string prefix)
    {
        IndexTtsOps.LoadLinear(_engine, read, Query, prefix + "q_proj", _d, _d);
        IndexTtsOps.LoadLinear(_engine, read, Key, prefix + "k_proj", _d, _d);
        IndexTtsOps.LoadLinear(_engine, read, Value, prefix + "v_proj", _d, _d);
        IndexTtsOps.LoadLinear(_engine, read, Output, prefix + "out_proj", _d, _d);
    }
}

/// <summary>
/// SpeechT5's shared encoder (the official <c>TransformerEncoder</c> with <c>TransformerSentenceEncoderLayer</c>s): the
/// input layer-normalized and dropped out, then post-norm layers of self-attention with the relative-position term
/// (<c>pe_k</c> of <c>i − j</c> clipped to ±max) and a GELU feed-forward, each with layer drop in training.
/// </summary>
internal sealed class SpeechT5Encoder<T>
{
    private readonly IEngine _engine;
    private readonly SpeechT5Configuration _c;

    internal sealed record Layer(SpeechT5Attention<T> Attention, LayerNormalizationLayer<T> AttentionNorm, DenseLayer<T> Fc1, DenseLayer<T> Fc2,
        LayerNormalizationLayer<T> FinalNorm);

    public SpeechT5Encoder(IEngine engine, List<LayerBase<T>> layers, SpeechT5Configuration c, Random random)
    {
        _engine = engine;
        _c = c;
        for (int i = 0; i < c.EncoderLayers; i++)
            Layers.Add(new Layer(new SpeechT5Attention<T>(engine, layers, random, c.Hidden, c.Heads), IndexTtsOps.LayerNorm(layers, c.Hidden),
                IndexTtsOps.Linear(engine, layers, random, c.Hidden, c.FeedForward), IndexTtsOps.Linear(engine, layers, random, c.FeedForward, c.Hidden),
                IndexTtsOps.LayerNorm(layers, c.Hidden)));
        InputNorm = IndexTtsOps.LayerNorm(layers, c.Hidden);
        RelativePositions = IndexTtsOps.Embedding(layers, random, 2 * c.MaxRelativePosition, c.Hidden / c.Heads, 1.0);
    }

    public List<Layer> Layers { get; } = new();
    public LayerNormalizationLayer<T> InputNorm { get; }
    public TiedEmbeddingLayer<T> RelativePositions { get; }

    /// <summary>The relative-position keys <c>[L, L, d_k]</c>: <c>pe_k</c> of <c>clip(i − j, −max, max − 1) + max</c>.</summary>
    private Tensor<T> Positions(int length)
    {
        int max = _c.MaxRelativePosition;
        var ids = new int[length * length];
        for (int i = 0; i < length; i++)
            for (int j = 0; j < length; j++) ids[i * length + j] = Math.Max(-max, Math.Min(max - 1, i - j)) + max;
        return _engine.Reshape(RelativePositions.Forward(IndexTtsOps.Ids<T>(ids)), new[] { length, length, _c.Hidden / _c.Heads });
    }

    /// <summary>The encoded states <c>[L, d]</c> of pre-net outputs <c>[L, d]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> input, bool training = false, Random? random = null)
    {
        var x = SpeechT5Ops.Drop(_engine, InputNorm.Forward(input), _c.Dropout, training, random);
        var positions = Positions(x.Shape[0]);
        foreach (var layer in Layers)
        {
            if (training && _c.EncoderLayerDrop > 0 && random is not null && random.NextDouble() <= _c.EncoderLayerDrop) continue;
            var (a, _) = layer.Attention.Forward(x, x, positions, false, null, null, 0, _c.AttentionDropout, training, random);
            x = layer.AttentionNorm.Forward(_engine.TensorAdd(x, SpeechT5Ops.Drop(_engine, a, _c.Dropout, training, random)));
            var h = SpeechT5Ops.Drop(_engine, SpeechT5Ops.Gelu(_engine, layer.Fc1.Forward(x)), _c.ActivationDropout, training, random);
            x = layer.FinalNorm.Forward(_engine.TensorAdd(x, SpeechT5Ops.Drop(_engine, layer.Fc2.Forward(h), _c.Dropout, training, random)));
        }
        return x;
    }

    public void Load(Func<string, int[], double[]> read, string prefix)
    {
        int d = _c.Hidden;
        for (int i = 0; i < Layers.Count; i++)
        {
            var l = Layers[i];
            string p = $"{prefix}layers.{i}.";
            l.Attention.Load(read, p + "self_attn.");
            IndexTtsOps.LoadNorm(_engine, read, l.AttentionNorm, p + "self_attn_layer_norm", d);
            IndexTtsOps.LoadLinear(_engine, read, l.Fc1, p + "fc1", d, _c.FeedForward);
            IndexTtsOps.LoadLinear(_engine, read, l.Fc2, p + "fc2", _c.FeedForward, d);
            IndexTtsOps.LoadNorm(_engine, read, l.FinalNorm, p + "final_layer_norm", d);
        }
        IndexTtsOps.LoadNorm(_engine, read, InputNorm, prefix + "layer_norm", d);
        RelativePositions.LoadTable(read(prefix + "pos_emb.pe_k.weight", new[] { 2 * _c.MaxRelativePosition, d / _c.Heads }));
    }
}

/// <summary>SpeechT5's decoder: post-norm layers of causal self-attention, cross-attention over the encoder output and a
/// GELU feed-forward, with no final norm; the decoder's own relative-position table is built but never applied.</summary>
internal sealed class SpeechT5Decoder<T>
{
    private readonly IEngine _engine;
    private readonly SpeechT5Configuration _c;

    internal sealed record Layer(SpeechT5Attention<T> SelfAttention, LayerNormalizationLayer<T> SelfNorm, SpeechT5Attention<T> CrossAttention,
        LayerNormalizationLayer<T> CrossNorm, DenseLayer<T> Fc1, DenseLayer<T> Fc2, LayerNormalizationLayer<T> FinalNorm);

    /// <summary>A decode's cached self-attention keys and values and the encoder's projected keys and values.</summary>
    internal sealed class State
    {
        public readonly List<(Tensor<T> K, Tensor<T> V)> Self = new();
        public readonly List<(Tensor<T> K, Tensor<T> V)> Cross = new();
    }

    public SpeechT5Decoder(IEngine engine, List<LayerBase<T>> layers, SpeechT5Configuration c, Random random)
    {
        _engine = engine;
        _c = c;
        for (int i = 0; i < c.DecoderLayers; i++)
            Layers.Add(new Layer(new SpeechT5Attention<T>(engine, layers, random, c.Hidden, c.Heads), IndexTtsOps.LayerNorm(layers, c.Hidden),
                new SpeechT5Attention<T>(engine, layers, random, c.Hidden, c.Heads), IndexTtsOps.LayerNorm(layers, c.Hidden),
                IndexTtsOps.Linear(engine, layers, random, c.Hidden, c.FeedForward), IndexTtsOps.Linear(engine, layers, random, c.FeedForward, c.Hidden),
                IndexTtsOps.LayerNorm(layers, c.Hidden)));
    }

    public List<Layer> Layers { get; } = new();

    public State Begin(Tensor<T> memory)
    {
        var state = new State();
        foreach (var l in Layers) state.Cross.Add(l.CrossAttention.StaticKeys(memory));
        return state;
    }

    /// <summary>The decoder states <c>[L, d]</c> of pre-net outputs <c>[L, d]</c> (continuing <paramref name="state"/>'s
    /// cache when given) and each layer's cross-attention weights <c>[H, L, Lenc]</c>.</summary>
    public (Tensor<T> Output, List<Tensor<T>> CrossWeights) Forward(Tensor<T> input, Tensor<T> memory, State? state, bool training = false,
        Random? random = null)
    {
        var x = input;
        var weights = new List<Tensor<T>>();
        for (int i = 0; i < Layers.Count; i++)
        {
            var l = Layers[i];
            if (training && _c.DecoderLayerDrop > 0 && random is not null && random.NextDouble() <= _c.DecoderLayerDrop) continue;
            var (a, _) = l.SelfAttention.Forward(x, x, null, state is null, null, state?.Self, i, _c.AttentionDropout, training, random);
            x = l.SelfNorm.Forward(_engine.TensorAdd(x, SpeechT5Ops.Drop(_engine, a, _c.Dropout, training, random)));
            var (c, w) = l.CrossAttention.Forward(x, memory, null, false, state?.Cross[i], null, 0, _c.AttentionDropout, training, random);
            weights.Add(w);
            x = l.CrossNorm.Forward(_engine.TensorAdd(x, SpeechT5Ops.Drop(_engine, c, _c.Dropout, training, random)));
            var h = SpeechT5Ops.Drop(_engine, SpeechT5Ops.Gelu(_engine, l.Fc1.Forward(x)), _c.ActivationDropout, training, random);
            x = l.FinalNorm.Forward(_engine.TensorAdd(x, SpeechT5Ops.Drop(_engine, l.Fc2.Forward(h), _c.Dropout, training, random)));
        }
        return (x, weights);
    }

    public void Load(Func<string, int[], double[]> read, string prefix)
    {
        int d = _c.Hidden;
        for (int i = 0; i < Layers.Count; i++)
        {
            var l = Layers[i];
            string p = $"{prefix}layers.{i}.";
            l.SelfAttention.Load(read, p + "self_attn.");
            IndexTtsOps.LoadNorm(_engine, read, l.SelfNorm, p + "self_attn_layer_norm", d);
            l.CrossAttention.Load(read, p + "encoder_attn.");
            IndexTtsOps.LoadNorm(_engine, read, l.CrossNorm, p + "encoder_attn_layer_norm", d);
            IndexTtsOps.LoadLinear(_engine, read, l.Fc1, p + "fc1", d, _c.FeedForward);
            IndexTtsOps.LoadLinear(_engine, read, l.Fc2, p + "fc2", _c.FeedForward, d);
            IndexTtsOps.LoadNorm(_engine, read, l.FinalNorm, p + "final_layer_norm", d);
        }
    }
}

/// <summary>ESPnet's <c>ScaledPositionalEncoding</c>: <c>x + α · PE</c> with a learned α, then dropout.</summary>
internal sealed class SpeechT5ScaledPositions<T>
{
    private readonly IEngine _engine;
    private readonly int _d;

    public SpeechT5ScaledPositions(IEngine engine, List<LayerBase<T>> layers, int d)
    {
        _engine = engine;
        _d = d;
        Alpha = IndexTtsOps.Values(layers, new[] { 1.0 });
    }

    public BiasFreeLinearLayer<T> Alpha { get; }

    public Tensor<T> Forward(Tensor<T> x, int start, double dropout, bool training, Random? random)
    {
        int length = x.Shape[0];
        var alpha = _engine.TensorBroadcastTo(_engine.Reshape(IndexTtsOps.Value(Alpha), new[] { 1, 1 }), new[] { length, _d });
        var y = _engine.TensorAdd(x, _engine.TensorMultiply(alpha, SpeechT5Ops.EspnetTable<T>(length, _d, start)));
        return SpeechT5Ops.Drop(_engine, y, dropout, training, random);
    }

    public void Load(Func<string, int[], double[]> read, string name) => TorchParameters.Linear(Alpha, 1, 1, read(name, new[] { 1 }));
}

/// <summary>
/// The speech decoder pre-net (paper §2.1): Tacotron 2's pre-net (linear + ReLU layers whose dropout stays on at
/// inference), a linear map to the model width, the scaled positional encoding, and the x-vector concatenated (after L2
/// normalization) and projected back with ReLU.
/// </summary>
internal sealed class SpeechT5SpeechDecoderPrenet<T>
{
    private readonly IEngine _engine;
    private readonly SpeechT5Configuration _c;

    public SpeechT5SpeechDecoderPrenet(IEngine engine, List<LayerBase<T>> layers, SpeechT5Configuration c, Random random)
    {
        _engine = engine;
        _c = c;
        for (int i = 0; i < c.PrenetLayers; i++)
            Prenet.Add(IndexTtsOps.Linear(engine, layers, random, i == 0 ? c.MelChannels : c.PrenetUnits, c.PrenetUnits));
        Projection = IndexTtsOps.Linear(engine, layers, random, c.PrenetUnits, c.Hidden);
        Positions = new SpeechT5ScaledPositions<T>(engine, layers, c.Hidden);
        Speaker = IndexTtsOps.Linear(engine, layers, random, c.SpeakerDim + c.Hidden, c.Hidden);
    }

    public List<DenseLayer<T>> Prenet { get; } = new();
    public DenseLayer<T> Projection { get; }
    public SpeechT5ScaledPositions<T> Positions { get; }
    public DenseLayer<T> Speaker { get; }

    /// <summary>The decoder inputs <c>[L, d]</c> of frames <c>[L, mel]</c> at decoder positions <paramref name="start"/>
    /// …; the pre-net's dropout draws from <paramref name="random"/> in training and inference alike.</summary>
    public Tensor<T> Forward(Tensor<T> frames, Tensor<T> speaker, int start, bool training, Random? random)
    {
        var x = frames;
        foreach (var layer in Prenet) x = SpeechT5Ops.Drop(_engine, _engine.ReLU(layer.Forward(x)), _c.PrenetDropout, true, random);
        x = Positions.Forward(Projection.Forward(x), start, _c.PositionalDropout, training, random);
        int length = x.Shape[0];
        var s = _engine.TensorBroadcastTo(_engine.Reshape(SpeechT5Ops.Normalize(_engine, speaker), new[] { 1, _c.SpeakerDim }), new[] { length, _c.SpeakerDim });
        return _engine.ReLU(Speaker.Forward(_engine.TensorConcatenate(new[] { x, s }, 1)));
    }

    public void Load(Func<string, int[], double[]> read, string prefix)
    {
        for (int i = 0; i < Prenet.Count; i++)
            IndexTtsOps.LoadLinear(_engine, read, Prenet[i], $"{prefix}decoder_prenet.0.0.prenet.{i}.0", i == 0 ? _c.MelChannels : _c.PrenetUnits, _c.PrenetUnits);
        IndexTtsOps.LoadLinear(_engine, read, Projection, prefix + "decoder_prenet.0.1", _c.PrenetUnits, _c.Hidden);
        Positions.Load(read, prefix + "decoder_prenet.1.alpha");
        IndexTtsOps.LoadLinear(_engine, read, Speaker, prefix + "spkembs_layer.0", _c.SpeakerDim + _c.Hidden, _c.Hidden);
    }
}

/// <summary>
/// The speech decoder post-net (paper §2.1): a linear map to r frames of the log-mel filterbank, a scalar stop logit per
/// frame, and Tacotron 2's five-layer convolutional post-net (bias-free convolutions, BatchNorm, tanh but on the last,
/// dropout) adding a residual.
/// </summary>
internal sealed class SpeechT5SpeechDecoderPostnet<T>
{
    private readonly IEngine _engine;
    private readonly SpeechT5Configuration _c;

    public SpeechT5SpeechDecoderPostnet(IEngine engine, List<LayerBase<T>> layers, SpeechT5Configuration c, Random random)
    {
        _engine = engine;
        _c = c;
        FeatureOut = IndexTtsOps.Linear(engine, layers, random, c.Hidden, c.MelChannels * c.ReductionFactor);
        ProbabilityOut = IndexTtsOps.Linear(engine, layers, random, c.Hidden, c.ReductionFactor);
        for (int i = 0; i < c.PostnetLayers; i++)
        {
            int inputs = i == 0 ? c.MelChannels : c.PostnetChannels, outputs = i == c.PostnetLayers - 1 ? c.MelChannels : c.PostnetChannels;
            var conv = new NormedConv1DLayer<T>(inputs, outputs, c.PostnetKernel, 1, 1, 1, (c.PostnetKernel - 1) / 2, false, ConvolutionNormalization.None, false);
            layers.Add(conv);
            conv.LoadTorchWeights(IndexTtsOps.Uniform(random, outputs * inputs * c.PostnetKernel, inputs * c.PostnetKernel), null, new double[outputs]);
            var norm = new BatchNormalizationLayer<T>(outputs, epsilon: 1e-5, momentum: 0.9);
            layers.Add(norm);
            Postnet.Add((conv, norm));
        }
    }

    public DenseLayer<T> FeatureOut { get; }
    public DenseLayer<T> ProbabilityOut { get; }
    public List<(NormedConv1DLayer<T> Conv, BatchNormalizationLayer<T> Norm)> Postnet { get; } = new();

    /// <summary>The frames before the post-net <c>[L·r, mel]</c> and the stop logits <c>[L·r]</c> of decoder states.</summary>
    public (Tensor<T> Frames, Tensor<T> Logits) Project(Tensor<T> states)
    {
        int length = states.Shape[0], r = _c.ReductionFactor;
        return (_engine.Reshape(FeatureOut.Forward(states), new[] { length * r, _c.MelChannels }),
            _engine.Reshape(ProbabilityOut.Forward(states), new[] { length * r }));
    }

    /// <summary>The post-net's residual added to frames <c>[F, mel]</c>.</summary>
    public Tensor<T> Refine(Tensor<T> frames, bool training, Random? random)
    {
        int f = frames.Shape[0];
        var h = _engine.Reshape(_engine.TensorTranspose(frames), new[] { 1, _c.MelChannels, f });
        for (int i = 0; i < Postnet.Count; i++)
        {
            var (conv, norm) = Postnet[i];
            var y = conv.Forward(h);
            int channels = y.Shape[1];
            h = _engine.Reshape(norm.Forward(_engine.Reshape(y, new[] { 1, channels, f, 1 })), new[] { 1, channels, f });   // NCHW
            if (i < Postnet.Count - 1) h = _engine.Tanh(h);
            h = SpeechT5Ops.Drop(_engine, h, _c.PostnetDropout, training, random);
        }
        return _engine.TensorAdd(frames, _engine.TensorTranspose(_engine.Reshape(h, new[] { _c.MelChannels, f })));
    }

    public void Load(Func<string, int[], double[]> read, string prefix)
    {
        IndexTtsOps.LoadLinear(_engine, read, FeatureOut, prefix + "feat_out", _c.Hidden, _c.MelChannels * _c.ReductionFactor);
        IndexTtsOps.LoadLinear(_engine, read, ProbabilityOut, prefix + "prob_out", _c.Hidden, _c.ReductionFactor);
        for (int i = 0; i < Postnet.Count; i++)
        {
            var (conv, norm) = Postnet[i];
            int outputs = conv.OutputChannels;
            conv.LoadTorchWeights(read($"{prefix}postnet.postnet.{i}.0.weight", conv.TorchWeightShape), null, new double[outputs]);
            string n = $"{prefix}postnet.postnet.{i}.1.";
            TorchParameters.BatchNorm(norm, read(n + "weight", new[] { outputs }), read(n + "bias", new[] { outputs }),
                read(n + "running_mean", new[] { outputs }), read(n + "running_var", new[] { outputs }));
        }
    }
}
