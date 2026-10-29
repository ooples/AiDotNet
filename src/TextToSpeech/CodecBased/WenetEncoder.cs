using AiDotNet.ActivationFunctions;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>The input layer of a wenet encoder (<c>input_layer</c>).</summary>
internal enum WenetInputLayer
{
    /// <summary><c>linear</c> (<c>LinearNoSubsampling</c>): Linear, LayerNorm (ε 1e-5), dropout.</summary>
    Linear,

    /// <summary><c>linear_legacy</c> (<c>LegacyLinearNoSubsampling</c>): the same, then ReLU.</summary>
    LinearLegacy,
}

/// <summary>The shape of a wenet <c>TransformerEncoder</c> / <c>ConformerEncoder</c> (without its convolution module or
/// macaron feed-forward) with ESPnet relative positions, as CosyVoice configures them.</summary>
/// <param name="StaticChunkSize">The attention's chunk size: 0 attends everywhere, n &gt; 0 to the position's chunk of n
/// and every earlier chunk (1 is causal).</param>
/// <param name="Swish">The feed-forward activation: Swish (the Conformer's default) or ReLU (the Transformer's).</param>
/// <param name="Conformer">A <c>ConformerEncoderLayer</c> (norms <c>norm_mha</c>, <c>norm_ff</c>) rather than a
/// <c>TransformerEncoderLayer</c> (<c>norm1</c>, <c>norm2</c>); without its convolution and macaron modules they compute
/// the same block.</param>
/// <param name="BlockNormEpsilon">The blocks' LayerNorm ε: wenet's 1e-12, or 1e-5 in GLM-4-Voice's cosyvoice fork.</param>
internal sealed record WenetEncoderConfiguration(int InputSize, int OutputSize, int Heads, int LinearUnits, int Blocks,
    WenetInputLayer InputLayer, bool Swish, bool Conformer, int StaticChunkSize = 0, double Dropout = 0.1, double PositionalDropout = 0.1,
    double AttentionDropout = 0.0, bool AfterNorm = true, double BlockNormEpsilon = 1e-12);

/// <summary>
/// A wenet encoder with ESPnet relative positional attention (CosyVoice <c>cosyvoice/transformer</c>: <c>BaseEncoder</c>,
/// <c>EspnetRelPositionalEncoding</c>, <c>RelPositionMultiHeadedAttention</c>, the pre-norm encoder layers): an input
/// layer, the input scaled by √d with a relative position table, pre-norm blocks of relative-position self-attention
/// and a two-layer feed-forward network, and a final LayerNorm.
/// </summary>
/// <remarks>
/// <para>Attention (Dai et al. 2019, §3.3): scores <c>((q + u) kᵀ + rel_shift((q + v) pᵀ)) / √d_k</c> with p the
/// projected position table over relative positions L−1 … −(L−1); <c>rel_shift</c> keeps, for query i and key j, the
/// column of relative position i − j.</para>
/// <para>Layer norms inside the blocks use ε = 1e-12 (<see cref="WenetEncoderConfiguration.BlockNormEpsilon"/>), the input
/// layer's and the final one 1e-5, as wenet sets them.</para>
/// </remarks>
internal sealed class WenetEncoder<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    internal sealed record Block(LayerNormalizationLayer<T> NormAttention, DenseLayer<T> Query, DenseLayer<T> Key, DenseLayer<T> Value,
        DenseLayer<T> Output, BiasFreeLinearLayer<T> PositionProjection, BiasFreeLinearLayer<T> BiasU, BiasFreeLinearLayer<T> BiasV,
        LayerNormalizationLayer<T> NormFeedForward, DenseLayer<T> FeedForward1, DenseLayer<T> FeedForward2);

    public WenetEncoder(IEngine engine, List<LayerBase<T>> layers, WenetEncoderConfiguration configuration, Random initialization)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        if (c.OutputSize % c.Heads != 0) throw new ArgumentException("The width must divide by the heads.", nameof(configuration));
        TLayer Add<TLayer>(TLayer layer) where TLayer : LayerBase<T> { layers.Add(layer); return layer; }
        double Uniform(int fanIn) => (2 * initialization.NextDouble() - 1) / Math.Sqrt(fanIn);
        DenseLayer<T> Linear(int inputs, int outputs)
        {
            var layer = Add(new DenseLayer<T>(outputs, (IActivationFunction<T>)new IdentityActivation<T>()));
            // nn.Linear's initialization: weights and biases from U(±1/√fan_in).
            TorchParameters.Linear(engine, layer, inputs, outputs, Enumerable.Range(0, inputs * outputs).Select(_ => Uniform(inputs)).ToArray(),
                Enumerable.Range(0, outputs).Select(_ => Uniform(inputs)).ToArray());
            return layer;
        }
        BiasFreeLinearLayer<T> Values(int count, Func<double> sample)
        {
            var holder = Add(new BiasFreeLinearLayer<T>(1, count));
            TorchParameters.Linear(holder, 1, count, Enumerable.Range(0, count).Select(_ => sample()).ToArray());
            return holder;
        }
        LayerNormalizationLayer<T> Norm(double epsilon) => Add(new LayerNormalizationLayer<T>(c.OutputSize, epsilon));

        int d = c.OutputSize, dk = d / c.Heads;
        EmbedLinear = Linear(c.InputSize, d);
        EmbedNorm = Norm(1e-5);
        double xavier = Math.Sqrt(6.0 / (c.Heads + dk));
        for (int i = 0; i < c.Blocks; i++)
        {
            var normAttention = Norm(c.BlockNormEpsilon);
            var query = Linear(d, d);
            var key = Linear(d, d);
            var value = Linear(d, d);
            var output = Linear(d, d);
            var position = Add(new BiasFreeLinearLayer<T>(d, d));
            TorchParameters.Linear(position, d, d, Enumerable.Range(0, d * d).Select(_ => Uniform(d)).ToArray());
            // pos_bias_u / pos_bias_v: xavier_uniform over [heads, d_k].
            var biasU = Values(d, () => xavier * (2 * initialization.NextDouble() - 1));
            var biasV = Values(d, () => xavier * (2 * initialization.NextDouble() - 1));
            var normFeedForward = Norm(c.BlockNormEpsilon);
            Blocks.Add(new Block(normAttention, query, key, value, output, position, biasU, biasV, normFeedForward,
                Linear(d, c.LinearUnits), Linear(c.LinearUnits, d)));
        }
        AfterNorm = c.AfterNorm ? Norm(1e-5) : null;
    }

    public WenetEncoderConfiguration Configuration { get; }
    public DenseLayer<T> EmbedLinear { get; }
    public LayerNormalizationLayer<T> EmbedNorm { get; }
    public List<Block> Blocks { get; } = new();
    /// <summary>The final LayerNorm; null for a stack that another one continues (UpsampleConformerEncoder's first).</summary>
    public LayerNormalizationLayer<T>? AfterNorm { get; }

    private Tensor<T> Value(BiasFreeLinearLayer<T> holder) => holder.Forward(new Tensor<T>(new[] { 1, 1 }, new Vector<T>(new[] { NumOps.One })));

    private Tensor<T> Dropout(Tensor<T> x, double rate, bool training, Random random) =>
        training && rate > 0 ? T5Seq2Seq<T>.Dropout(_engine, x, rate, random) : x;

    /// <summary>The relative position table <c>[2L − 1, d]</c> over relative positions L−1 … −(L−1)
    /// (<c>EspnetRelPositionalEncoding</c>: sin at even columns, cos at odd, frequencies 10000^(−2k/d)).</summary>
    internal Tensor<T> PositionTable(int length)
    {
        int d = Configuration.OutputSize;
        var table = new Tensor<T>(new[] { 2 * length - 1, d });
        for (int r = 0; r < 2 * length - 1; r++)
        {
            double position = length - 1 - r;
            for (int k = 0; k < d; k += 2)
            {
                double frequency = Math.Exp(k * -(Math.Log(10000.0) / d));
                table[r, k] = NumOps.FromDouble(Math.Sin(position * frequency));
                if (k + 1 < d) table[r, k + 1] = NumOps.FromDouble(Math.Cos(position * frequency));
            }
        }
        return table;
    }

    /// <summary>The attention mask bias <c>[L, L]</c> for chunks of <paramref name="chunk"/> (0: none): 0 where query i
    /// may attend key j, −∞ elsewhere.</summary>
    private Tensor<T> MaskBias(int length, int chunk)
    {
        var bias = new Tensor<T>(new[] { length, length });
        if (chunk <= 0) return bias;
        var blocked = NumOps.FromDouble(double.NegativeInfinity);
        for (int i = 0; i < length; i++)
        {
            int end = Math.Min(length, (i / chunk + 1) * chunk);                      // subsequent_chunk_mask, all left chunks
            for (int j = end; j < length; j++) bias[i, j] = blocked;
        }
        return bias;
    }

    /// <summary>The encoded states <c>[L, OutputSize]</c> of inputs <c>[L, InputSize]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> input, bool training, Random random)
    {
        var (x, positions) = Embed(input, training, random);
        x = RunBlocks(x, positions, Configuration.StaticChunkSize, training, random);
        return AfterNorm is null ? x : AfterNorm.Forward(x);
    }

    /// <summary>The input layer and positional encoding: the states <c>[L, OutputSize]</c> (scaled by √d) and the relative
    /// position table <c>[2L − 1, OutputSize]</c>.</summary>
    internal (Tensor<T> States, Tensor<T> Positions) Embed(Tensor<T> input, bool training, Random random)
    {
        var c = Configuration;
        int length = input.Shape[0], d = c.OutputSize;
        var x = EmbedNorm.Forward(EmbedLinear.Forward(input));
        x = Dropout(x, c.Dropout, training, random);
        if (c.InputLayer == WenetInputLayer.LinearLegacy) x = _engine.ReLU(x);
        x = Dropout(_engine.TensorMultiplyScalar(x, NumOps.FromDouble(Math.Sqrt(d))), c.PositionalDropout, training, random);
        return (x, Dropout(PositionTable(length), c.PositionalDropout, training, random));
    }

    /// <summary>The blocks on embedded states, attending within chunks of <paramref name="chunk"/> and every earlier chunk
    /// (0: everywhere).</summary>
    internal Tensor<T> RunBlocks(Tensor<T> x, Tensor<T> positions, int chunk, bool training, Random random)
    {
        var c = Configuration;
        int length = x.Shape[0], d = c.OutputSize, heads = c.Heads, dk = d / heads;
        var mask = MaskBias(length, chunk);
        // rel_shift: query i, key j reads column (L − 1) − i + j of the [L, 2L − 1] position scores.
        var shift = new Tensor<int>(new[] { length * length });
        for (int i = 0; i < length; i++)
            for (int j = 0; j < length; j++) shift[i * length + j] = i * (2 * length - 1) + (length - 1 - i + j);

        Tensor<T> Heads(Tensor<T> t, int rows) => _engine.TensorPermute(_engine.Reshape(t, new[] { rows, heads, dk }), new[] { 1, 0, 2 });
        foreach (var block in Blocks)
        {
            var normed = block.NormAttention.Forward(x);
            var q = Heads(block.Query.Forward(normed), length);                                                // [H, L, dk]
            var k = Heads(block.Key.Forward(normed), length);
            var v = Heads(block.Value.Forward(normed), length);
            var p = Heads(block.PositionProjection.Forward(positions), 2 * length - 1);                         // [H, 2L−1, dk]
            var u = _engine.Reshape(Value(block.BiasU), new[] { heads, 1, dk });
            var w = _engine.Reshape(Value(block.BiasV), new[] { heads, 1, dk });
            var ac = _engine.BatchMatMul(_engine.TensorAdd(q, _engine.TensorBroadcastTo(u, new[] { heads, length, dk })),
                _engine.TensorPermute(k, new[] { 0, 2, 1 }));                                                 // [H, L, L]
            var bdFull = _engine.BatchMatMul(_engine.TensorAdd(q, _engine.TensorBroadcastTo(w, new[] { heads, length, dk })),
                _engine.TensorPermute(p, new[] { 0, 2, 1 }));                                                 // [H, L, 2L−1]
            var bd = _engine.Reshape(_engine.TensorIndexSelect(_engine.Reshape(bdFull, new[] { heads, length * (2 * length - 1) }), shift, 1),
                new[] { heads, length, length });
            var scores = _engine.TensorMultiplyScalar(_engine.TensorAdd(ac, bd), NumOps.FromDouble(1.0 / Math.Sqrt(dk)));
            scores = _engine.TensorAdd(scores, _engine.TensorBroadcastTo(_engine.Reshape(mask, new[] { 1, length, length }), new[] { heads, length, length }));
            var weights = Dropout(_engine.Softmax(scores, axis: 2), c.AttentionDropout, training, random);
            var context = _engine.Reshape(_engine.TensorPermute(_engine.BatchMatMul(weights, v), new[] { 1, 0, 2 }), new[] { length, d });
            x = _engine.TensorAdd(x, Dropout(block.Output.Forward(context), c.Dropout, training, random));

            var hidden = block.FeedForward1.Forward(block.NormFeedForward.Forward(x));
            hidden = c.Swish ? _engine.Swish(hidden) : _engine.ReLU(hidden);
            hidden = block.FeedForward2.Forward(Dropout(hidden, c.Dropout, training, random));
            x = _engine.TensorAdd(x, Dropout(hidden, c.Dropout, training, random));
        }
        return x;
    }

    /// <summary>Loads a wenet encoder's state (<c>{prefix}embed.out.0/1</c>, <c>{prefix}encoders.{i}.*</c>,
    /// <c>{prefix}after_norm</c>).</summary>
    public void LoadTorchWeights(Func<string, int[], double[]> read, string prefix, string embedName = "embed",
        string encodersName = "encoders")
    {
        var c = Configuration;
        int d = c.OutputSize;
        void Dense(DenseLayer<T> layer, string name, int inputs, int outputs) =>
            TorchParameters.Linear(_engine, layer, inputs, outputs, read(name + ".weight", new[] { outputs, inputs }), read(name + ".bias", new[] { outputs }));
        void Norm(LayerNormalizationLayer<T> norm, string name) =>
            TorchParameters.LayerNorm(_engine, norm, read(name + ".weight", new[] { d }), read(name + ".bias", new[] { d }));
        Dense(EmbedLinear, $"{prefix}{embedName}.out.0", c.InputSize, d);
        Norm(EmbedNorm, $"{prefix}{embedName}.out.1");
        for (int i = 0; i < Blocks.Count; i++)
        {
            var b = Blocks[i];
            string p = $"{prefix}{encodersName}.{i}";
            // The Transformer layer names its norms norm1/norm2, the Conformer layer norm_mha/norm_ff.
            bool conformer = c.Conformer;
            Norm(b.NormAttention, p + (conformer ? ".norm_mha" : ".norm1"));
            Dense(b.Query, p + ".self_attn.linear_q", d, d);
            Dense(b.Key, p + ".self_attn.linear_k", d, d);
            Dense(b.Value, p + ".self_attn.linear_v", d, d);
            Dense(b.Output, p + ".self_attn.linear_out", d, d);
            TorchParameters.Linear(b.PositionProjection, d, d, read(p + ".self_attn.linear_pos.weight", new[] { d, d }));
            TorchParameters.Linear(b.BiasU, 1, d, read(p + ".self_attn.pos_bias_u", new[] { c.Heads, d / c.Heads }));
            TorchParameters.Linear(b.BiasV, 1, d, read(p + ".self_attn.pos_bias_v", new[] { c.Heads, d / c.Heads }));
            Norm(b.NormFeedForward, p + (conformer ? ".norm_ff" : ".norm2"));
            Dense(b.FeedForward1, p + ".feed_forward.w_1", d, c.LinearUnits);
            Dense(b.FeedForward2, p + ".feed_forward.w_2", c.LinearUnits, d);
        }
        if (AfterNorm is not null) Norm(AfterNorm, prefix + "after_norm");
    }
}
