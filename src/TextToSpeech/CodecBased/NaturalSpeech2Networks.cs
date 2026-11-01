using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.Latest;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>
/// NaturalSpeech 2's Transformer encoder (the phoneme encoder and the speech prompt encoder, §3.2-3.3): pre-norm blocks of
/// multi-head self-attention and a feed-forward network whose first layer is a 1-D convolution (FastSpeech's), then a
/// final norm. The norms are plain, or style-adaptive on the prompt (Amphion's phoneme encoder); the input is phoneme ids
/// through an embedding, or rows of hidden vectors.
/// </summary>
internal sealed class Ns2TransformerEncoder<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly int _dim;
    private readonly double _dropout;
    private readonly NaturalSpeech2Positions _positions;

    public Ns2TransformerEncoder(IEngine engine, List<LayerBase<T>> layers, Random random, int vocab, int dim, int heads, int filter, int kernel,
        int count, double dropout, bool styleAdaptive, NaturalSpeech2Positions positions, double embeddingStd = 1e-5)
    {
        if (kernel % 2 == 0) throw new ArgumentException("The feed-forward convolution's kernel must be odd (padding kernel / 2 keeps the length).");
        _engine = engine;
        _dim = dim;
        _dropout = dropout;
        _positions = positions;
        // Amphion's NaturalSpeech 2 draws the phoneme embedding with a standard deviation of 1e-5.
        if (vocab > 0) Embedding = IndexTtsOps.Embedding(layers, random, vocab, dim, embeddingStd);
        for (int i = 0; i < count; i++)
            Blocks.Add(new Block(
                new Ns2Norm<T>(engine, layers, random, dim, styleAdaptive),
                new Ns2Attention<T>(engine, layers, random, dim, heads),
                new Ns2Norm<T>(engine, layers, random, dim, styleAdaptive),
                Ns2Ops.Conv(layers, random, dim, filter, kernel, padding: kernel / 2, weight: Ns2Ops.Normal(random, filter * dim * kernel, 0.02)),
                Ns2Ops.Linear(engine, layers, random, filter, dim, Ns2Ops.Normal(random, dim * filter, 0.02))));
        LastNorm = new Ns2Norm<T>(engine, layers, random, dim, styleAdaptive);
    }

    internal sealed record Block(Ns2Norm<T> Norm1, Ns2Attention<T> Attention, Ns2Norm<T> Norm2, NormedConv1DLayer<T> FeedForward1, DenseLayer<T> FeedForward2);

    public TiedEmbeddingLayer<T>? Embedding { get; }
    public List<Block> Blocks { get; } = new();
    public Ns2Norm<T> LastNorm { get; }

    /// <summary>Phoneme ids to rows <c>[n, d]</c>.</summary>
    public Tensor<T> Forward(IReadOnlyList<int> ids, Tensor<T>? condition, bool training, Random? random) =>
        Forward(Embedding?.Forward(IndexTtsOps.Ids<T>(ids)) ?? throw new InvalidOperationException("This encoder reads hidden vectors."), condition, training, random);

    /// <summary>Rows <c>[n, d]</c> (conditioned on prompt rows <c>[T′, d]</c> when style-adaptive) to rows <c>[n, d]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> x, Tensor<T>? condition, bool training, Random? random)
    {
        int n = x.Shape[0];
        // Amphion adds pe[:x.size(0)] to a batch-first [B, T, d]: row b of the table to every frame of batch item b, which at
        // a batch of one is the position-0 row everywhere.
        var table = _positions == NaturalSpeech2Positions.Sinusoidal ? Ns2Ops.Sinusoids<T>(0, n, _dim)
            : _engine.TensorTile(Ns2Ops.Sinusoids<T>(0, 1, _dim), new[] { n, 1 });
        x = Ns2Ops.Dropout(_engine, _engine.TensorAdd(x, table), _dropout, training, random);
        foreach (var b in Blocks)
        {
            var h = b.Norm1.Forward(x, condition);
            x = _engine.TensorAdd(x, Ns2Ops.Dropout(_engine, b.Attention.Forward(h, h), _dropout, training, random));
            h = Ns2Ops.Rows(_engine, b.FeedForward1.Forward(Ns2Ops.Channels(_engine, b.Norm2.Forward(x, condition))));
            h = Ns2Ops.Dropout(_engine, _engine.ReLU(h), _dropout, training, random);
            x = _engine.TensorAdd(x, b.FeedForward2.Forward(h));
        }
        return LastNorm.Forward(x, condition);
    }

    public void Load(Func<string, double[]> read, string prefix)
    {
        if (Embedding is not null) Embedding.LoadLeadingRows(read(prefix + "enc_emb_tokens.weight"));
        int filter = Blocks.Count > 0 ? Blocks[0].FeedForward1.OutputChannels : 0;
        for (int i = 0; i < Blocks.Count; i++)
        {
            var b = Blocks[i];
            string p = $"{prefix}layers.{i}.";
            b.Norm1.Load(read, p + "ln_1.");
            b.Attention.Load(read, p + "self_attn.");
            b.Norm2.Load(read, p + "ln_2.");
            Ns2Ops.SetConv(b.FeedForward1, read(p + "ffn.ffn_1.weight"), read(p + "ffn.ffn_1.bias"));
            TorchParameters.Linear(_engine, b.FeedForward2, filter, _dim, read(p + "ffn.ffn_2.weight"), read(p + "ffn.ffn_2.bias"));
        }
        LastNorm.Load(read, prefix + "last_ln.");
    }
}

/// <summary>
/// NaturalSpeech 2's duration or pitch predictor (§3.2-3.3): 1-D convolutions with ReLU, LayerNorm and dropout, residual
/// after the first, with a Q-K-V attention to the prompt encoder's hidden sequence before every third convolution
/// (pre-norm, dropout 0.2, the sum scaled by 1/√2), then a linear output per position.
/// </summary>
internal sealed class Ns2VariancePredictor<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly int _filter, _every;
    private readonly double _dropout;

    public Ns2VariancePredictor(IEngine engine, List<LayerBase<T>> layers, Random random, int input, int filter, int kernel, int count, int every,
        int heads, double dropout)
    {
        if (kernel % 2 == 0) throw new ArgumentException("The predictor's kernel must be odd (padding kernel / 2 keeps the length).");
        _engine = engine;
        _filter = filter;
        _every = every;
        _dropout = dropout;
        for (int i = 0; i < count; i++)
        {
            Convs.Add(Ns2Ops.Conv(layers, random, i == 0 ? input : filter, filter, kernel, padding: kernel / 2));
            Norms.Add(IndexTtsOps.LayerNorm(layers, filter));
            if (i % every == 0)
            {
                Attentions.Add(new Ns2Attention<T>(engine, layers, random, filter, heads));
                AttentionNorms.Add(IndexTtsOps.LayerNorm(layers, filter));
            }
        }
        Output = Ns2Ops.Linear(engine, layers, random, filter, 1, Ns2Ops.Normal(random, filter, 0.02));
    }

    public List<NormedConv1DLayer<T>> Convs { get; } = new();
    public List<LayerNormalizationLayer<T>> Norms { get; } = new();
    public List<Ns2Attention<T>> Attentions { get; } = new();
    public List<LayerNormalizationLayer<T>> AttentionNorms { get; } = new();
    public DenseLayer<T> Output { get; }

    /// <summary>Rows <c>[n, d]</c> attending to prompt rows <c>[T′, d]</c>; returns <c>[n]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> rows, Tensor<T> prompt, bool training, Random? random)
    {
        var x = Ns2Ops.Channels(_engine, rows);
        double invSqrt2 = 1.0 / Math.Sqrt(2.0);
        for (int i = 0; i < Convs.Count; i++)
        {
            var residual = x;
            if (i % _every == 0)
            {
                var y = Ns2Ops.Rows(_engine, x);
                var attended = Attentions[i / _every].Forward(AttentionNorms[i / _every].Forward(y), prompt);
                attended = Ns2Ops.Dropout(_engine, attended, 0.2, training, random);
                x = Ns2Ops.Channels(_engine, _engine.TensorMultiplyScalar(_engine.TensorAdd(attended, y), NumOps.FromDouble(invSqrt2)));
            }
            var h = Norms[i].Forward(Ns2Ops.Rows(_engine, _engine.ReLU(Convs[i].Forward(x))));
            x = Ns2Ops.Channels(_engine, Ns2Ops.Dropout(_engine, h, _dropout, training, random));
            if (i != 0) x = _engine.TensorAdd(x, residual);
        }
        var output = Output.Forward(Ns2Ops.Rows(_engine, x));
        return _engine.Reshape(output, new[] { output.Shape[0] });
    }

    public void Load(Func<string, double[]> read, string prefix)
    {
        for (int i = 0; i < Convs.Count; i++)
        {
            Ns2Ops.SetConv(Convs[i], read($"{prefix}conv.{i}.0.weight"), read($"{prefix}conv.{i}.0.bias"));
            TorchParameters.LayerNorm(_engine, Norms[i], read($"{prefix}conv.{i}.2.weight"), read($"{prefix}conv.{i}.2.bias"));
        }
        for (int a = 0; a < Attentions.Count; a++)
        {
            Attentions[a].Load(read, $"{prefix}cattn.{a}.0.");
            TorchParameters.LayerNorm(_engine, AttentionNorms[a], read($"{prefix}cattn.{a}.1.weight"), read($"{prefix}cattn.{a}.1.bias"));
        }
        TorchParameters.Linear(_engine, Output, _filter, 1, read(prefix + "linear.weight"), read(prefix + "linear.bias"));
    }
}

/// <summary>
/// NaturalSpeech 2's denoiser s_θ(z_t, t, c) (§3.2, App. B): a WaveNet of gated dilated convolutions conditioned on the
/// frame-level prior c and the diffusion step, where some layers attend to the prompt's query tokens and use the result as
/// a FiLM layer's scale and bias; the skip outputs averaged (by 1/√layers), then ReLU, 1×1 convolution, ReLU and a
/// zero-initialized 1×1 output convolution predicting ẑ₀.
/// </summary>
internal sealed class Ns2WaveNet<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly int _input, _hidden, _output;
    private readonly double _attentionDropout;

    public Ns2WaveNet(IEngine engine, List<LayerBase<T>> layers, Random random, int input, int hidden, int output, int count, Func<int, int> dilation,
        Func<int, bool> attends, int heads, double attentionDropout)
    {
        _engine = engine;
        _input = input;
        _hidden = hidden;
        _output = output;
        _attentionDropout = attentionDropout;
        InProjection = Ns2Ops.Conv(layers, random, input, hidden, 1, weight: Ns2Ops.KaimingNormal(random, hidden * input, input));
        Mlp1 = Ns2Ops.Linear(engine, layers, random, hidden, 4 * hidden, Ns2Ops.Normal(random, 4 * hidden * hidden, 0.02));
        Mlp2 = Ns2Ops.Linear(engine, layers, random, 4 * hidden, hidden, Ns2Ops.Normal(random, 4 * hidden * hidden, 0.02));
        ConditionNorm = IndexTtsOps.LayerNorm(layers, hidden);
        for (int i = 0; i < count; i++)
        {
            int d = dilation(i);
            bool a = attends(i);
            Layers.Add(new Residual(
                Ns2Ops.Conv(layers, random, hidden, 2 * hidden, 3, d, d, Ns2Ops.KaimingNormal(random, 2 * hidden * hidden * 3, hidden * 3)),
                Ns2Ops.Linear(engine, layers, random, hidden, hidden, Ns2Ops.Normal(random, hidden * hidden, 0.02)),
                Ns2Ops.Conv(layers, random, hidden, 2 * hidden, 1, weight: Ns2Ops.KaimingNormal(random, 2 * hidden * hidden, hidden)),
                Ns2Ops.Conv(layers, random, hidden, 2 * hidden, 1, weight: Ns2Ops.KaimingNormal(random, 2 * hidden * hidden, hidden)),
                a ? new Ns2Attention<T>(engine, layers, random, hidden, heads, attentionDropout) : null,
                a ? Ns2Ops.Linear(engine, layers, random, hidden, 2 * hidden, Ns2Ops.Xavier(random, 2 * hidden * hidden, hidden, 2 * hidden), Ones(2 * hidden)) : null,
                a ? Ns2Ops.Linear(engine, layers, random, hidden, 2 * hidden, Ns2Ops.Xavier(random, 2 * hidden * hidden, hidden, 2 * hidden), new double[2 * hidden]) : null,
                a ? IndexTtsOps.LayerNorm(layers, hidden) : null));
        }
        SkipProjection = Ns2Ops.Conv(layers, random, hidden, hidden, 1, weight: Ns2Ops.KaimingNormal(random, hidden * hidden, hidden));
        OutProjection = Ns2Ops.Conv(layers, random, hidden, output, 1, weight: new double[output * hidden]);
    }

    private static double[] Ones(int n) => Enumerable.Repeat(1.0, n).ToArray();

    internal sealed record Residual(NormedConv1DLayer<T> DilatedConv, DenseLayer<T> StepProjection, NormedConv1DLayer<T> ConditionProjection,
        NormedConv1DLayer<T> OutProjection, Ns2Attention<T>? Attention, DenseLayer<T>? FilmGain, DenseLayer<T>? FilmBias, LayerNormalizationLayer<T>? Norm);

    public NormedConv1DLayer<T> InProjection { get; }
    public DenseLayer<T> Mlp1 { get; }
    public DenseLayer<T> Mlp2 { get; }
    public LayerNormalizationLayer<T> ConditionNorm { get; }
    public List<Residual> Layers { get; } = new();
    public NormedConv1DLayer<T> SkipProjection { get; }
    public NormedConv1DLayer<T> OutProjection { get; }

    /// <summary>
    /// ẑ₀ (or the flow) <c>[1, out, T]</c> for z_t <c>[1, in, T]</c> at step <paramref name="t"/>, with the prior's frame-level
    /// condition rows <c>[T, d]</c> and the query tokens <c>[m, d]</c>.
    /// </summary>
    public Tensor<T> Forward(Tensor<T> xt, double t, Tensor<T> condition, Tensor<T> query, bool training, Random? random)
    {
        var cond = Ns2Ops.Channels(_engine, ConditionNorm.Forward(condition));
        var x = _engine.ReLU(InProjection.Forward(xt));
        var step = Mlp2.Forward(_engine.Mish(Mlp1.Forward(Ns2Ops.StepEmbedding<T>(t, _hidden))));                // [1, d]
        int frames = xt.Shape[2];
        Tensor<T>? skips = null;
        foreach (var l in Layers)
        {
            var s = _engine.Reshape(l.StepProjection.Forward(step), new[] { 1, _hidden, 1 });
            var y = _engine.TensorAdd(x, _engine.TensorBroadcastTo(s, x._shape));
            Tensor<T>? attended = null;
            if (l.Attention is not null) attended = l.Attention.Forward(l.Norm!.Forward(Ns2Ops.Rows(_engine, y)), query, training, random);
            y = _engine.TensorAdd(l.DilatedConv.Forward(y), l.ConditionProjection.Forward(cond));
            if (attended is not null)
            {
                var rows = Ns2Ops.Rows(_engine, y);
                y = Ns2Ops.Channels(_engine, _engine.TensorAdd(_engine.TensorMultiply(rows, l.FilmGain!.Forward(attended)), l.FilmBias!.Forward(attended)));
            }
            var gate = _engine.TensorSlice(y, new[] { 0, 0, 0 }, new[] { 1, _hidden, frames });
            var filter = _engine.TensorSlice(y, new[] { 0, _hidden, 0 }, new[] { 1, _hidden, frames });
            y = l.OutProjection.Forward(_engine.TensorMultiply(_engine.Sigmoid(gate), _engine.Tanh(filter)));
            var residual = _engine.TensorSlice(y, new[] { 0, 0, 0 }, new[] { 1, _hidden, frames });
            var skip = _engine.TensorSlice(y, new[] { 0, _hidden, 0 }, new[] { 1, _hidden, frames });
            x = _engine.TensorMultiplyScalar(_engine.TensorAdd(x, residual), NumOps.FromDouble(1.0 / Math.Sqrt(2.0)));
            skips = skips is null ? skip : _engine.TensorAdd(skips, skip);
        }
        var h = _engine.TensorMultiplyScalar(skips!, NumOps.FromDouble(1.0 / Math.Sqrt(Layers.Count)));
        return OutProjection.Forward(_engine.ReLU(SkipProjection.Forward(h)));
    }

    public void Load(Func<string, double[]> read, string prefix)
    {
        Ns2Ops.SetConv(InProjection, read(prefix + "in_proj.weight"), read(prefix + "in_proj.bias"));
        TorchParameters.Linear(_engine, Mlp1, _hidden, 4 * _hidden, read(prefix + "mlp.0.weight"), read(prefix + "mlp.0.bias"));
        TorchParameters.Linear(_engine, Mlp2, 4 * _hidden, _hidden, read(prefix + "mlp.2.weight"), read(prefix + "mlp.2.bias"));
        TorchParameters.LayerNorm(_engine, ConditionNorm, read(prefix + "cond_ln.weight"), read(prefix + "cond_ln.bias"));
        for (int i = 0; i < Layers.Count; i++)
        {
            var l = Layers[i];
            string p = $"{prefix}layers.{i}.";
            Ns2Ops.SetConv(l.DilatedConv, read(p + "dilated_conv.weight"), read(p + "dilated_conv.bias"));
            TorchParameters.Linear(_engine, l.StepProjection, _hidden, _hidden, read(p + "diffusion_proj.weight"), read(p + "diffusion_proj.bias"));
            Ns2Ops.SetConv(l.ConditionProjection, read(p + "cond_proj.weight"), read(p + "cond_proj.bias"));
            Ns2Ops.SetConv(l.OutProjection, read(p + "out_proj.weight"), read(p + "out_proj.bias"));
            if (l.Attention is null) continue;
            l.Attention.Load(read, p + "attn.");
            TorchParameters.Linear(_engine, l.FilmGain!, _hidden, 2 * _hidden, read(p + "film.gain.weight"), read(p + "film.gain.bias"));
            TorchParameters.Linear(_engine, l.FilmBias!, _hidden, 2 * _hidden, read(p + "film.bias.weight"), read(p + "film.bias.bias"));
            TorchParameters.LayerNorm(_engine, l.Norm!, read(p + "ln.weight"), read(p + "ln.bias"));
        }
        Ns2Ops.SetConv(SkipProjection, read(prefix + "skip_proj.weight"), read(prefix + "skip_proj.bias"));
        Ns2Ops.SetConv(OutProjection, read(prefix + "out_proj.weight"), read(prefix + "out_proj.bias"));
    }
}
