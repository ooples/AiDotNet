using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.FlowDiffusion;
using AiDotNet.TextToSpeech.Latest;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>The sizes of FireRedTTS's flow-matching Mel decoder (the release's <c>flow</c> config).</summary>
internal sealed record FireRedTtsFlowConfiguration(int MelChannels, int CodebookSize, int CodebookDim, int Dim, int Heads, int LinearUnits,
    int Blocks, int CrossAttentionBlocks, double Dropout, int RegulatorBlocks, int TokenMelRatio, int PromptHidden, int PromptReduction,
    int PromptLayers, int PromptBlocks, int PromptKernel, int[] DecoderChannels, int DecoderHeads, int DecoderHeadDim, int DecoderBlocksPerLevel,
    int DecoderMidBlocks, double InferenceCfgRate, double TrainingCfgRate, double SigmaMin);

/// <summary>
/// FireRedTTS's flow-matching Mel decoder (Guo et al. 2024, §3.3.1; the release's <c>CrossAttnFlowMatching</c>): semantic
/// tokens through the speech tokenizer's frozen product-quantizer codebooks (<c>HHGCodecEmbedding</c>: token mod 128 and
/// token div 128 looked up and concatenated, then projected), a Conformer whose first layers cross-attend to the prompt
/// mel encoded by <c>MelReduceEncoder</c> (the timbre, §3.3.1), stretched to four mel frames per token
/// (<c>InterpolateRegulator</c>: nearest, then conv–GroupNorm–Mish stages), projected to the mel's channels as the
/// condition μ, and a conditional flow-matching U-Net (<see cref="MatchaDecoder{T}"/>, CosyVoice's estimator) that turns
/// noise into the 16 kHz, 10 ms, 100-band mel.
/// </summary>
/// <remarks>
/// <para>Inference (<c>ConditionalCFM.inference</c>): z ~ N(0, I), Euler steps over t = 1 − cos(πs/2), each field
/// (1 + w)·v(x, μ) − w·v(x, 0) with w = 0.7 (eq. 7).</para>
/// <para>Training (eq. 6, no code released): the optimal-transport path x_t = (1 − (1 − σ)t) z + t x₁ with target
/// x₁ − (1 − σ) z, t ~ U(0, 1) through the same cosine schedule, μ dropped with probability 0.2 (§3.3.1), and the squared
/// error of the estimate averaged over frames and channels; σ = 1e-6 as CosyVoice's <c>compute_loss</c>, whose estimator
/// this is.</para>
/// </remarks>
internal sealed class FireRedTtsFlow<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly Func<int, Tensor<T>> _codebook;

    internal sealed record Block(LayerNormalizationLayer<T> NormAttention, DenseLayer<T> Query, DenseLayer<T> Key, DenseLayer<T> Value,
        DenseLayer<T> Output, BiasFreeLinearLayer<T> PositionProjection, BiasFreeLinearLayer<T> BiasU, BiasFreeLinearLayer<T> BiasV,
        LayerNormalizationLayer<T>? NormCross, DenseLayer<T>? CrossQuery, DenseLayer<T>? CrossKey, DenseLayer<T>? CrossValue,
        DenseLayer<T>? CrossOutput, LayerNormalizationLayer<T> NormFeedForward, DenseLayer<T> FeedForward1, DenseLayer<T> FeedForward2);

    internal sealed record ConvLayer(NormedConv1DLayer<T> Conv, LayerNormalizationLayer<T> Norm);

    /// <param name="codebook">The speech tokenizer's codebook <c>[CodebookSize, CodebookDim]</c> of a sub-quantizer (0 or
    /// 1), which the input embedding reads frozen.</param>
    public FireRedTtsFlow(IEngine engine, List<LayerBase<T>> layers, FireRedTtsFlowConfiguration configuration, Func<int, Tensor<T>> codebook,
        Random random)
    {
        _engine = engine;
        _codebook = codebook;
        Configuration = configuration;
        var c = configuration;
        if (c.Dim % c.Heads != 0) throw new ArgumentException("The width must divide by the heads.", nameof(configuration));
        DenseLayer<T> Linear(int inputs, int outputs) => IndexTtsOps.Linear(engine, layers, random, inputs, outputs);
        NormedConv1DLayer<T> Conv(int inputs, int outputs, int kernel, int stride, int padding)
        {
            var conv = new NormedConv1DLayer<T>(inputs, outputs, kernel, stride, 1, 1, padding, false, ConvolutionNormalization.None);
            layers.Add(conv);
            conv.LoadTorchWeights(IndexTtsOps.Uniform(random, outputs * inputs * kernel, inputs * kernel), null,
                IndexTtsOps.Uniform(random, outputs, inputs * kernel));
            return conv;
        }
        int d = c.Dim, dk = d / c.Heads;
        InputProjection = 2 * c.CodebookDim == d ? null : Linear(2 * c.CodebookDim, d);

        EmbedLinear = Linear(d, d);
        EmbedNorm = IndexTtsOps.LayerNorm(layers, d);
        double xavier = Math.Sqrt(6.0 / (c.Heads + dk));
        for (int i = 0; i < c.Blocks; i++)
        {
            var normAttention = IndexTtsOps.LayerNorm(layers, d);
            var q = Linear(d, d);
            var k = Linear(d, d);
            var v = Linear(d, d);
            var o = Linear(d, d);
            var position = IndexTtsOps.BiasFree(layers, random, d, d);
            var biasU = IndexTtsOps.Values(layers, Enumerable.Range(0, d).Select(_ => xavier * (2 * random.NextDouble() - 1)).ToArray());
            var biasV = IndexTtsOps.Values(layers, Enumerable.Range(0, d).Select(_ => xavier * (2 * random.NextDouble() - 1)).ToArray());
            bool cross = i < c.CrossAttentionBlocks;
            LayerNormalizationLayer<T>? normCross = null;
            DenseLayer<T>? cq = null, ck = null, cv = null, co = null;
            if (cross)
            {
                cq = Linear(d, d);
                ck = Linear(d, d);
                cv = Linear(d, d);
                co = Linear(d, d);
            }
            var normFeedForward = IndexTtsOps.LayerNorm(layers, d);
            var ff1 = Linear(d, c.LinearUnits);
            var ff2 = Linear(c.LinearUnits, d);
            if (cross) normCross = IndexTtsOps.LayerNorm(layers, d);
            Blocks.Add(new Block(normAttention, q, k, v, o, position, biasU, biasV, normCross, cq, ck, cv, co, normFeedForward, ff1, ff2));
        }
        AfterNorm = IndexTtsOps.LayerNorm(layers, d);

        for (int i = 0; i < c.RegulatorBlocks; i++)
        {
            var conv = Conv(d, d, 3, 1, 1);
            var norm = new GroupNormalizationLayer<T>(1, d, 1e-5);
            layers.Add(norm);
            Regulator.Add((conv, norm));
        }
        RegulatorOutput = Conv(d, d, 1, 1, 0);
        EncoderProjection = Linear(d, c.MelChannels);

        // MelReduceEncoder: in_proj, 5 residual conv layers, the strided reduction, 5 more, out_proj.
        int h = c.PromptHidden;
        PromptIn = c.MelChannels == h ? null : Conv(c.MelChannels, h, c.PromptKernel, 1, (c.PromptKernel - 1) / 2);
        List<ConvLayer[]> Stack() => Enumerable.Range(0, c.PromptBlocks).Select(_ => Enumerable.Range(0, c.PromptLayers)
            .Select(__ =>
            {
                var conv = Conv(h, h, c.PromptKernel, 1, (c.PromptKernel - 1) / 2);
                return new ConvLayer(conv, IndexTtsOps.LayerNorm(layers, h));
            }).ToArray()).ToList();
        PromptBefore = Stack();
        PromptReduce = Conv(h, h, c.PromptReduction, c.PromptReduction, 0);
        PromptAfter = Stack();
        PromptOut = d == h ? null : Conv(h, d, c.PromptKernel, 1, (c.PromptKernel - 1) / 2);

        Estimator = new MatchaDecoder<T>(engine, 2 * c.MelChannels, c.MelChannels, c.DecoderChannels, 0.0, c.DecoderHeadDim, c.DecoderHeads,
            c.DecoderMidBlocks, c.DecoderBlocksPerLevel, MatchaFeedForward.Gelu, biasMask: false);
        layers.AddRange(Estimator.Layers);
    }

    public FireRedTtsFlowConfiguration Configuration { get; }
    public DenseLayer<T>? InputProjection { get; }
    public DenseLayer<T> EmbedLinear { get; }
    public LayerNormalizationLayer<T> EmbedNorm { get; }
    public List<Block> Blocks { get; } = new();
    public LayerNormalizationLayer<T> AfterNorm { get; }
    public List<(NormedConv1DLayer<T> Conv, GroupNormalizationLayer<T> Norm)> Regulator { get; } = new();
    public NormedConv1DLayer<T> RegulatorOutput { get; }
    public DenseLayer<T> EncoderProjection { get; }
    public NormedConv1DLayer<T>? PromptIn { get; }
    public List<ConvLayer[]> PromptBefore { get; }
    public NormedConv1DLayer<T> PromptReduce { get; }
    public List<ConvLayer[]> PromptAfter { get; }
    public NormedConv1DLayer<T>? PromptOut { get; }
    public MatchaDecoder<T> Estimator { get; }

    private Tensor<T> Drop(Tensor<T> x, bool training, Random? random) =>
        training && Configuration.Dropout > 0 && random is not null ? T5Seq2Seq<T>.Dropout(_engine, x, Configuration.Dropout, random) : x;

    // [L, C] through a 1-D convolution over L.
    private Tensor<T> OverTime(NormedConv1DLayer<T> conv, Tensor<T> x)
    {
        int length = x.Shape[0], channels = x.Shape[1];
        var y = conv.Forward(_engine.Reshape(_engine.TensorTranspose(x), new[] { 1, channels, length }));
        return _engine.TensorTranspose(_engine.Reshape(y, new[] { y.Shape[1], y.Shape[2] }));
    }

    /// <summary><c>HHGCodecEmbedding</c>: the codebook rows of token mod K and token div K, concatenated, then projected
    /// <c>[n, Dim]</c>.</summary>
    public Tensor<T> Embed(IReadOnlyList<int> tokens)
    {
        int size = Configuration.CodebookSize;
        if (tokens.Any(t => t < 0 || t >= size * size)) throw new ArgumentOutOfRangeException(nameof(tokens), $"Semantic tokens lie in [0, {size * size}).");
        var low = new Tensor<int>(new[] { tokens.Count }, new Vector<int>(tokens.Select(t => t % size).ToArray()));
        var high = new Tensor<int>(new[] { tokens.Count }, new Vector<int>(tokens.Select(t => t / size).ToArray()));
        var x = _engine.TensorConcatenate(new[] { _engine.TensorIndexSelect(_codebook(0), low, 0), _engine.TensorIndexSelect(_codebook(1), high, 0) }, 1);
        return InputProjection is null ? x : InputProjection.Forward(x);
    }

    /// <summary><c>MelReduceEncoder</c>: the prompt features <c>[⌊F / 4⌋, Dim]</c> of a prompt mel <c>[F, mel]</c>.</summary>
    public Tensor<T> EncodePrompt(Tensor<T> mel)
    {
        var x = PromptIn is null ? mel : OverTime(PromptIn, mel);
        foreach (var layer in PromptBefore) x = ResidualConv(layer, x);
        x = OverTime(PromptReduce, x);
        foreach (var layer in PromptAfter) x = ResidualConv(layer, x);
        return PromptOut is null ? x : OverTime(PromptOut, x);
    }

    // ResidualConvLayer: x + (conv → dropout(0) → LayerNorm → GELU) twice.
    private Tensor<T> ResidualConv(ConvLayer[] layers, Tensor<T> x)
    {
        var h = x;
        foreach (var layer in layers) h = ActivationHelper.ExactGelu(layer.Norm.Forward(OverTime(layer.Conv, h)), _engine);
        return _engine.TensorAdd(x, h);
    }

    /// <summary>The ESPnet relative position table <c>[2L − 1, d]</c> (positions L−1 … −(L−1)).</summary>
    internal Tensor<T> PositionTable(int length)
    {
        int d = Configuration.Dim;
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

    /// <summary><c>ConformerDecoderV2</c>: the encoded tokens <c>[n, Dim]</c> of the embedded tokens, cross-attending to
    /// the prompt features <c>[m, Dim]</c>.</summary>
    public Tensor<T> Encode(Tensor<T> embedded, Tensor<T> memory, bool training = false, Random? random = null)
    {
        var c = Configuration;
        int length = embedded.Shape[0], d = c.Dim, heads = c.Heads, dk = d / heads, m = memory.Shape[0];
        var x = Drop(EmbedNorm.Forward(EmbedLinear.Forward(embedded)), training, random);
        x = Drop(_engine.TensorMultiplyScalar(x, NumOps.FromDouble(Math.Sqrt(d))), training, random);
        var positions = Drop(PositionTable(length), training, random);
        var shift = new Tensor<int>(new[] { length * length });
        for (int i = 0; i < length; i++)
            for (int j = 0; j < length; j++) shift[i * length + j] = i * (2 * length - 1) + (length - 1 - i + j);
        Tensor<T> Heads(Tensor<T> t, int rows) => _engine.TensorPermute(_engine.Reshape(t, new[] { rows, heads, dk }), new[] { 1, 0, 2 });
        Tensor<T> Merge(Tensor<T> t, int rows) => _engine.Reshape(_engine.TensorPermute(t, new[] { 1, 0, 2 }), new[] { rows, d });
        var scale = NumOps.FromDouble(1.0 / Math.Sqrt(dk));
        foreach (var b in Blocks)
        {
            var normed = b.NormAttention.Forward(x);
            var q = Heads(b.Query.Forward(normed), length);
            var k = Heads(b.Key.Forward(normed), length);
            var v = Heads(b.Value.Forward(normed), length);
            var p = Heads(b.PositionProjection.Forward(positions), 2 * length - 1);
            var u = _engine.Reshape(IndexTtsOps.Value(b.BiasU), new[] { heads, 1, dk });
            var w = _engine.Reshape(IndexTtsOps.Value(b.BiasV), new[] { heads, 1, dk });
            var ac = _engine.BatchMatMul(_engine.TensorAdd(q, _engine.TensorBroadcastTo(u, new[] { heads, length, dk })), _engine.TensorPermute(k, new[] { 0, 2, 1 }));
            var bdFull = _engine.BatchMatMul(_engine.TensorAdd(q, _engine.TensorBroadcastTo(w, new[] { heads, length, dk })), _engine.TensorPermute(p, new[] { 0, 2, 1 }));
            var bd = _engine.Reshape(_engine.TensorIndexSelect(_engine.Reshape(bdFull, new[] { heads, length * (2 * length - 1) }), shift, 1),
                new[] { heads, length, length });
            var weights = Drop(_engine.Softmax(_engine.TensorMultiplyScalar(_engine.TensorAdd(ac, bd), scale), axis: 2), training, random);
            x = _engine.TensorAdd(x, Drop(b.Output.Forward(Merge(_engine.BatchMatMul(weights, v), length)), training, random));

            if (b.CrossQuery is not null)
            {
                var n2 = (b.NormCross ?? throw new InvalidOperationException("A cross-attention block has its norm.")).Forward(x);
                var cq = Heads(b.CrossQuery.Forward(n2), length);
                var ck = Heads((b.CrossKey ?? throw new InvalidOperationException()).Forward(memory), m);
                var cv = Heads((b.CrossValue ?? throw new InvalidOperationException()).Forward(memory), m);
                var cw = Drop(_engine.Softmax(_engine.TensorMultiplyScalar(_engine.BatchMatMul(cq, _engine.TensorPermute(ck, new[] { 0, 2, 1 })), scale), axis: 2),
                    training, random);
                x = _engine.TensorAdd(x, Drop((b.CrossOutput ?? throw new InvalidOperationException()).Forward(Merge(_engine.BatchMatMul(cw, cv), length)),
                    training, random));
            }

            var hidden = Drop(_engine.Swish(b.FeedForward1.Forward(b.NormFeedForward.Forward(x))), training, random);
            x = _engine.TensorAdd(x, Drop(b.FeedForward2.Forward(hidden), training, random));
        }
        return AfterNorm.Forward(x);
    }

    /// <summary>The length regulator and projection: μ <c>[ratio · n, mel]</c> of the encoded tokens <c>[n, Dim]</c>
    /// (nearest interpolation to <c>ratio · n</c> frames, the conv–GroupNorm–Mish stages and a 1×1 convolution).</summary>
    public Tensor<T> Condition(Tensor<T> encoded)
    {
        var c = Configuration;
        int n = encoded.Shape[0], frames = c.TokenMelRatio * n, d = c.Dim;
        var index = new Tensor<int>(new[] { frames }, new Vector<int>(Enumerable.Range(0, frames).Select(i => (int)Math.Floor(i * (double)n / frames)).ToArray()));
        var x = _engine.Reshape(_engine.TensorTranspose(_engine.TensorIndexSelect(encoded, index, 0)), new[] { 1, d, frames });
        foreach (var (conv, norm) in Regulator)
            x = _engine.Mish(_engine.Reshape(norm.Forward(_engine.Reshape(conv.Forward(x), new[] { 1, d, 1, frames })), new[] { 1, d, frames }));
        x = RegulatorOutput.Forward(x);
        return EncoderProjection.Forward(_engine.TensorTranspose(_engine.Reshape(x, new[] { d, frames })));
    }

    /// <summary>μ <c>[ratio · n, mel]</c> of the tokens given the prompt mel.</summary>
    public Tensor<T> Mu(IReadOnlyList<int> tokens, Tensor<T> promptMel, bool training = false, Random? random = null) =>
        Condition(Encode(Embed(tokens), EncodePrompt(promptMel), training, random));

    /// <summary>The cosine schedule's times t_s = 1 − cos(π s / 2n), s = 0 … n.</summary>
    internal static double[] Schedule(int steps) =>
        Enumerable.Range(0, steps + 1).Select(s => 1 - Math.Cos((double)s / steps * 0.5 * Math.PI)).ToArray();

    /// <summary><c>CrossAttnFlowMatching.inference</c>: the mel <c>[ratio · n, mel]</c> from <paramref name="noise"/>
    /// (z, already scaled by the temperature) by <paramref name="steps"/> guided Euler steps.</summary>
    public Tensor<T> Inference(IReadOnlyList<int> tokens, Tensor<T> promptMel, int steps, Tensor<T> noise)
    {
        if (steps < 1) throw new ArgumentOutOfRangeException(nameof(steps));
        var mu = Mu(tokens, promptMel);
        if (noise.Shape[0] != mu.Shape[0] || noise.Shape[1] != mu.Shape[1]) throw new ArgumentException("The noise has the mel's shape.", nameof(noise));
        int frames = mu.Shape[0];
        var mask = new Tensor<T>(new[] { frames });
        mask.Fill(NumOps.One);
        var zero = new Tensor<T>(mu.Shape.ToArray());
        var span = Schedule(steps);
        double w = Configuration.InferenceCfgRate, t = span[0], dt = span[1] - span[0];
        var x = noise;
        for (int step = 1; step <= steps; step++)
        {
            var field = Estimator.Estimate(x, mu, t, mask);
            if (w > 0)
            {
                var free = Estimator.Estimate(x, zero, t, mask);
                field = _engine.TensorSubtract(_engine.TensorMultiplyScalar(field, NumOps.FromDouble(1 + w)), _engine.TensorMultiplyScalar(free, NumOps.FromDouble(w)));
            }
            x = _engine.TensorAdd(x, _engine.TensorMultiplyScalar(field, NumOps.FromDouble(dt)));
            t += dt;
            if (step < steps) dt = span[step + 1] - t;
        }
        return x;
    }

    /// <summary>
    /// The flow-matching loss (eq. 6) of the target mel <c>[ratio · n, mel]</c> of the tokens, with the timbre from the
    /// reference mel (another recording of the speaker, §3.3.1): one time, one noise draw, μ dropped with probability
    /// <see cref="FireRedTtsFlowConfiguration.TrainingCfgRate"/>.
    /// </summary>
    public Tensor<T> TrainingLoss(IReadOnlyList<int> tokens, Tensor<T> mel, Tensor<T> referenceMel, bool training, Random random)
    {
        var c = Configuration;
        var mu = Mu(tokens, referenceMel, training, random);
        double t = 1 - Math.Cos(random.NextDouble() * 0.5 * Math.PI);
        int frames = mu.Shape[0];
        var z = new Tensor<T>(new[] { frames, c.MelChannels }, TorchParameters.Gaussian<T>(random, frames * c.MelChannels, 1.0));
        bool drop = random.NextDouble() < c.TrainingCfgRate;
        return Loss(mu, mel, t, z, drop);
    }

    /// <summary>The flow-matching loss of μ <c>[F, mel]</c> against the target mel at time <paramref name="t"/> and noise
    /// <paramref name="z"/>, μ zeroed when <paramref name="dropCondition"/>.</summary>
    internal Tensor<T> Loss(Tensor<T> mu, Tensor<T> mel, double t, Tensor<T> z, bool dropCondition)
    {
        var c = Configuration;
        int frames = mu.Shape[0], channels = c.MelChannels;
        if (mel.Shape[0] != frames || mel.Shape[1] != channels)
            throw new ArgumentException($"The target mel has {c.TokenMelRatio} frames per token: [{frames}, {channels}].", nameof(mel));
        var xt = _engine.TensorAdd(_engine.TensorMultiplyScalar(z, NumOps.FromDouble(1 - (1 - c.SigmaMin) * t)), _engine.TensorMultiplyScalar(mel, NumOps.FromDouble(t)));
        var target = _engine.TensorSubtract(mel, _engine.TensorMultiplyScalar(z, NumOps.FromDouble(1 - c.SigmaMin)));
        if (dropCondition) mu = _engine.TensorMultiplyScalar(mu, NumOps.Zero);
        var mask = new Tensor<T>(new[] { frames });
        mask.Fill(NumOps.One);
        var error = _engine.TensorSubtract(Estimator.Estimate(xt, mu, t, mask), target);
        return _engine.TensorMultiplyScalar(_engine.ReduceSum(_engine.TensorMultiply(error, error), null), NumOps.FromDouble(1.0 / (frames * channels)));
    }

    /// <summary>Loads the release's <c>flow.*</c> names (the codebooks belong to the speech tokenizer).</summary>
    public void Load(Func<string, int[], double[]> read, string prefix)
    {
        var c = Configuration;
        int d = c.Dim, h = c.PromptHidden;
        void Lin(DenseLayer<T> layer, string name, int inputs, int outputs) => IndexTtsOps.LoadLinear(_engine, read, layer, prefix + name, inputs, outputs);
        void Norm(LayerNormalizationLayer<T> norm, string name, int dim) => IndexTtsOps.LoadNorm(_engine, read, norm, prefix + name, dim);
        void Conv(NormedConv1DLayer<T> conv, string name, int outputs) => IndexTtsOps.LoadConv(read, conv, prefix + name, outputs);
        if (InputProjection is not null) Lin(InputProjection, "input_embedding.proj", 2 * c.CodebookDim, d);
        Lin(EmbedLinear, "encoder.embed.out.0", d, d);
        Norm(EmbedNorm, "encoder.embed.out.1", d);
        for (int i = 0; i < Blocks.Count; i++)
        {
            var b = Blocks[i];
            string p = $"encoder.encoders.{i}.";
            Norm(b.NormAttention, p + "norm_mha", d);
            Lin(b.Query, p + "self_attn.linear_q", d, d);
            Lin(b.Key, p + "self_attn.linear_k", d, d);
            Lin(b.Value, p + "self_attn.linear_v", d, d);
            Lin(b.Output, p + "self_attn.linear_out", d, d);
            IndexTtsOps.LoadLinear(read, b.PositionProjection, prefix + p + "self_attn.linear_pos", d, d);
            TorchParameters.Linear(b.BiasU, 1, d, read(prefix + p + "self_attn.pos_bias_u", new[] { c.Heads, d / c.Heads }));
            TorchParameters.Linear(b.BiasV, 1, d, read(prefix + p + "self_attn.pos_bias_v", new[] { c.Heads, d / c.Heads }));
            if (b.CrossQuery is not null)
            {
                Norm(b.NormCross!, p + "norm_mha2", d);
                Lin(b.CrossQuery, p + "src_attn.linear_q", d, d);
                Lin(b.CrossKey!, p + "src_attn.linear_k", d, d);
                Lin(b.CrossValue!, p + "src_attn.linear_v", d, d);
                Lin(b.CrossOutput!, p + "src_attn.linear_out", d, d);
            }
            Norm(b.NormFeedForward, p + "norm_ff", d);
            Lin(b.FeedForward1, p + "feed_forward.w_1", d, c.LinearUnits);
            Lin(b.FeedForward2, p + "feed_forward.w_2", c.LinearUnits, d);
        }
        Norm(AfterNorm, "encoder.after_norm", d);
        for (int i = 0; i < Regulator.Count; i++)
        {
            Conv(Regulator[i].Conv, $"length_regulator.model.{3 * i}", d);
            Regulator[i].Norm.LoadAffine(read($"{prefix}length_regulator.model.{3 * i + 1}.weight", new[] { d }),
                read($"{prefix}length_regulator.model.{3 * i + 1}.bias", new[] { d }));
        }
        Conv(RegulatorOutput, $"length_regulator.model.{3 * Regulator.Count}", d);
        Lin(EncoderProjection, "encoder_proj", d, c.MelChannels);
        if (PromptIn is not null) Conv(PromptIn, "prompt_prenet.encoder.in_proj", h);
        void Stack(List<ConvLayer[]> stack, string name)
        {
            for (int i = 0; i < stack.Count; i++)
                for (int j = 0; j < stack[i].Length; j++)
                {
                    string p = $"prompt_prenet.encoder.{name}.{i}.layers.{j}.";
                    Conv(stack[i][j].Conv, p + "conv", h);
                    Norm(stack[i][j].Norm, p + "norm", h);
                }
        }
        Stack(PromptBefore, "conv1");
        Conv(PromptReduce, "prompt_prenet.encoder.middle_layer", h);
        Stack(PromptAfter, "conv2");
        if (PromptOut is not null) Conv(PromptOut, "prompt_prenet.encoder.out_proj", d);
        Estimator.LoadTorchWeights(read, prefix + "decoder.estimator.");
    }
}
