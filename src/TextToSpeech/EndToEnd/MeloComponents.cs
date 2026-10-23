using System.Collections.Generic;
using System.Linq;
using AiDotNet.ActivationFunctions;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.TextToSpeech.EndToEnd;

/// <summary>
/// MeloTTS's text encoder (<c>models.TextEncoder</c>): the symbol, tone and language embeddings plus the projected
/// 1024-dimensional and 768-dimensional BERT features (<c>bert_proj</c>, <c>ja_bert_proj</c>), scaled by √hidden, the
/// relative-attention encoder (speaker-conditioned), and a 1×1 projection to the prior's mean and log-scale.
/// </summary>
internal sealed class MeloTextEncoder<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly int _hidden, _out;

    public MeloTextEncoder(IEngine engine, List<LayerBase<T>> layers, int vocabulary, int tones, int languages, int outChannels,
        int hidden, int filter, int heads, int blocks, int kernel, double dropout, int speakerChannels, int bertDim = 1024, int jaBertDim = 768)
    {
        _engine = engine;
        _hidden = hidden;
        _out = outChannels;
        layers.Add(Symbols = new TiedEmbeddingLayer<T>(vocabulary, hidden));
        layers.Add(Tones = new TiedEmbeddingLayer<T>(tones, hidden));
        layers.Add(Languages = new TiedEmbeddingLayer<T>(languages, hidden));
        BertProjection = MeloOps.Conv(layers, bertDim, hidden, 1);
        JaBertProjection = MeloOps.Conv(layers, jaBertDim, hidden, 1);
        Encoder = new MeloAttentionEncoder<T>(engine, layers, hidden, filter, heads, blocks, kernel, dropout, speakerChannels);
        Projection = MeloOps.Conv(layers, hidden, 2 * outChannels, 1);
    }

    public TiedEmbeddingLayer<T> Symbols { get; }
    public TiedEmbeddingLayer<T> Tones { get; }
    public TiedEmbeddingLayer<T> Languages { get; }
    public NormedConv1DLayer<T> BertProjection { get; }
    public NormedConv1DLayer<T> JaBertProjection { get; }
    public MeloAttentionEncoder<T> Encoder { get; }
    public NormedConv1DLayer<T> Projection { get; }

    private static Tensor<T> Ids(IReadOnlyList<int> ids)
    {
        var t = new Tensor<T>(new[] { ids.Count });
        for (int i = 0; i < ids.Count; i++) t[i] = NumOps.FromDouble(ids[i]);
        return t;
    }

    /// <summary>
    /// The hidden states <c>[1, hidden, L]</c> and the prior's mean and log-scale <c>[1, out, L]</c> of a symbol
    /// sequence with its tones, languages and BERT features (<c>[1, 1024, L]</c>, <c>[1, 768, L]</c>).
    /// </summary>
    public (Tensor<T> Hidden, Tensor<T> Mean, Tensor<T> LogScale) Forward(IReadOnlyList<int> symbols, IReadOnlyList<int> tones,
        IReadOnlyList<int> languages, Tensor<T> bert, Tensor<T> jaBert, Tensor<T>? g, bool training, Random random)
    {
        int length = symbols.Count;
        var sum = _engine.TensorAdd(_engine.TensorAdd(Symbols.Forward(Ids(symbols)), Tones.Forward(Ids(tones))), Languages.Forward(Ids(languages)));
        var x = _engine.Reshape(_engine.TensorTranspose(sum), new[] { 1, _hidden, length });
        x = _engine.TensorAdd(x, _engine.TensorAdd(BertProjection.Forward(bert), JaBertProjection.Forward(jaBert)));
        x = _engine.TensorMultiplyScalar(x, NumOps.FromDouble(Math.Sqrt(_hidden)));
        x = Encoder.Forward(x, g, training, random);
        var stats = Projection.Forward(x);
        return (x, MeloOps.Channels(_engine, stats, 0, _out), MeloOps.Channels(_engine, stats, _out, _out));
    }

    public void LoadTorch(Func<string, double[]> read, string prefix)
    {
        Symbols.LoadTable(read(prefix + "emb.weight"));
        Tones.LoadTable(read(prefix + "tone_emb.weight"));
        Languages.LoadTable(read(prefix + "language_emb.weight"));
        MeloOps.LoadConv(read, BertProjection, prefix + "bert_proj");
        MeloOps.LoadConv(read, JaBertProjection, prefix + "ja_bert_proj");
        Encoder.LoadTorch(_engine, read, prefix + "encoder.");
        MeloOps.LoadConv(read, Projection, prefix + "proj");
    }
}

/// <summary>
/// The deterministic duration predictor (<c>models.DurationPredictor</c>): on the detached text states plus the
/// detached projected speaker, two conv–ReLU–LayerNorm–dropout blocks and a 1×1 projection to log-durations.
/// </summary>
internal sealed class MeloDurationPredictor<T>
{
    private readonly IEngine _engine;
    private readonly double _dropout;

    public MeloDurationPredictor(IEngine engine, List<LayerBase<T>> layers, int inChannels, int filter, int kernel, double dropout, int speakerChannels)
    {
        _engine = engine;
        _dropout = dropout;
        Conv1 = MeloOps.Conv(layers, inChannels, filter, kernel, padding: kernel / 2);
        Norm1 = MeloOps.Norm(layers, filter);
        Conv2 = MeloOps.Conv(layers, filter, filter, kernel, padding: kernel / 2);
        Norm2 = MeloOps.Norm(layers, filter);
        Projection = MeloOps.Conv(layers, filter, 1, 1);
        if (speakerChannels > 0) Condition = MeloOps.Conv(layers, speakerChannels, inChannels, 1);
    }

    public NormedConv1DLayer<T> Conv1 { get; }
    public LayerNormalizationLayer<T> Norm1 { get; }
    public NormedConv1DLayer<T> Conv2 { get; }
    public LayerNormalizationLayer<T> Norm2 { get; }
    public NormedConv1DLayer<T> Projection { get; }
    public NormedConv1DLayer<T>? Condition { get; }

    private Tensor<T> Drop(Tensor<T> x, bool training, Random random) =>
        training && _dropout > 0 ? AiDotNet.TextToSpeech.CodecBased.T5Seq2Seq<T>.Dropout(_engine, x, _dropout, random) : x;

    public Tensor<T> Forward(Tensor<T> x, Tensor<T>? g, bool training, Random random)
    {
        x = MeloOps.Detach(x);
        if (g is not null && Condition is not null) x = _engine.TensorAdd(x, MeloOps.OverTime(_engine, Condition.Forward(MeloOps.Detach(g)), x.Shape[2]));
        x = Drop(MeloOps.ChannelNorm(_engine, Norm1, _engine.ReLU(Conv1.Forward(x))), training, random);
        x = Drop(MeloOps.ChannelNorm(_engine, Norm2, _engine.ReLU(Conv2.Forward(x))), training, random);
        return Projection.Forward(x);
    }

    public void LoadTorch(Func<string, double[]> read, string prefix)
    {
        MeloOps.LoadConv(read, Conv1, prefix + "conv_1");
        MeloOps.LoadNorm(_engine, read, Norm1, prefix + "norm_1");
        MeloOps.LoadConv(read, Conv2, prefix + "conv_2");
        MeloOps.LoadNorm(_engine, read, Norm2, prefix + "norm_2");
        MeloOps.LoadConv(read, Projection, prefix + "proj");
        if (Condition is not null) MeloOps.LoadConv(read, Condition, prefix + "cond");
    }
}

/// <summary>
/// VITS2's duration discriminator as Bert-VITS2 and MeloTTS build it (<c>models.DurationDiscriminator</c>): the detached
/// text states (plus the speaker) through two conv–ReLU–LayerNorm blocks, then for each duration (real and predicted
/// log-durations) the projected duration concatenated on, two more blocks and a sigmoid-linear probability per token.
/// </summary>
internal sealed class MeloDurationDiscriminator<T>
{
    private readonly IEngine _engine;
    private readonly double _dropout;
    private readonly int _filter;

    public MeloDurationDiscriminator(IEngine engine, List<LayerBase<T>> layers, int inChannels, int filter, int kernel, double dropout, int speakerChannels)
    {
        _engine = engine;
        _dropout = dropout;
        _filter = filter;
        Conv1 = MeloOps.Conv(layers, inChannels, filter, kernel, padding: kernel / 2);
        Norm1 = MeloOps.Norm(layers, filter);
        Conv2 = MeloOps.Conv(layers, filter, filter, kernel, padding: kernel / 2);
        Norm2 = MeloOps.Norm(layers, filter);
        DurationProjection = MeloOps.Conv(layers, 1, filter, 1);
        PreOut1 = MeloOps.Conv(layers, 2 * filter, filter, kernel, padding: kernel / 2);
        PreOutNorm1 = MeloOps.Norm(layers, filter);
        PreOut2 = MeloOps.Conv(layers, filter, filter, kernel, padding: kernel / 2);
        PreOutNorm2 = MeloOps.Norm(layers, filter);
        if (speakerChannels > 0) Condition = MeloOps.Conv(layers, speakerChannels, inChannels, 1);
        Output = new DenseLayer<T>(1, (IActivationFunction<T>)new IdentityActivation<T>());
        layers.Add(Output);
    }

    public NormedConv1DLayer<T> Conv1 { get; }
    public LayerNormalizationLayer<T> Norm1 { get; }
    public NormedConv1DLayer<T> Conv2 { get; }
    public LayerNormalizationLayer<T> Norm2 { get; }
    public NormedConv1DLayer<T> DurationProjection { get; }
    public NormedConv1DLayer<T> PreOut1 { get; }
    public LayerNormalizationLayer<T> PreOutNorm1 { get; }
    public NormedConv1DLayer<T> PreOut2 { get; }
    public LayerNormalizationLayer<T> PreOutNorm2 { get; }
    public NormedConv1DLayer<T>? Condition { get; }
    public DenseLayer<T> Output { get; }

    private Tensor<T> Drop(Tensor<T> x, bool training, Random random) =>
        training && _dropout > 0 ? AiDotNet.TextToSpeech.CodecBased.T5Seq2Seq<T>.Dropout(_engine, x, _dropout, random) : x;

    /// <summary>The probabilities <c>[L, 1]</c> that the real and the predicted log-durations <c>[1, 1, L]</c> are real.</summary>
    public (Tensor<T> Real, Tensor<T> Generated) Forward(Tensor<T> x, Tensor<T>? g, Tensor<T> realDurations, Tensor<T> generatedDurations,
        bool training, Random random)
    {
        x = MeloOps.Detach(x);
        if (g is not null && Condition is not null) x = _engine.TensorAdd(x, MeloOps.OverTime(_engine, Condition.Forward(MeloOps.Detach(g)), x.Shape[2]));
        x = Drop(MeloOps.ChannelNorm(_engine, Norm1, _engine.ReLU(Conv1.Forward(x))), training, random);
        x = Drop(MeloOps.ChannelNorm(_engine, Norm2, _engine.ReLU(Conv2.Forward(x))), training, random);
        Tensor<T> Probability(Tensor<T> durations)
        {
            var h = _engine.TensorConcatenate(new[] { x, DurationProjection.Forward(durations) }, 1);
            h = Drop(MeloOps.ChannelNorm(_engine, PreOutNorm1, _engine.ReLU(PreOut1.Forward(h))), training, random);
            h = Drop(MeloOps.ChannelNorm(_engine, PreOutNorm2, _engine.ReLU(PreOut2.Forward(h))), training, random);
            var rows = _engine.TensorTranspose(_engine.Reshape(h, new[] { _filter, h.Shape[2] }));
            return _engine.Sigmoid(Output.Forward(rows));
        }
        return (Probability(realDurations), Probability(generatedDurations));
    }

    public void LoadTorch(Func<string, double[]> read, string prefix)
    {
        MeloOps.LoadConv(read, Conv1, prefix + "conv_1");
        MeloOps.LoadNorm(_engine, read, Norm1, prefix + "norm_1");
        MeloOps.LoadConv(read, Conv2, prefix + "conv_2");
        MeloOps.LoadNorm(_engine, read, Norm2, prefix + "norm_2");
        MeloOps.LoadConv(read, DurationProjection, prefix + "dur_proj");
        MeloOps.LoadConv(read, PreOut1, prefix + "pre_out_conv_1");
        MeloOps.LoadNorm(_engine, read, PreOutNorm1, prefix + "pre_out_norm_1");
        MeloOps.LoadConv(read, PreOut2, prefix + "pre_out_conv_2");
        MeloOps.LoadNorm(_engine, read, PreOutNorm2, prefix + "pre_out_norm_2");
        if (Condition is not null) MeloOps.LoadConv(read, Condition, prefix + "cond");
        MeloOps.LoadLinear(_engine, read, Output, _filter, 1, prefix + "output_layer.0");
    }
}

/// <summary>
/// VITS's stochastic duration predictor (<c>models.StochasticDurationPredictor</c>; its filter width is the input's, as the
/// code overrides it): a flow-based variational model of the durations. Training returns the negative variational lower
/// bound; generation inverts the flows from noise. The noise draws are parameters, so a test can supply them.
/// </summary>
internal sealed class MeloStochasticDurationPredictor<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly int _filter;

    public MeloStochasticDurationPredictor(IEngine engine, List<LayerBase<T>> layers, int inChannels, int kernel, double dropout,
        int flows, int speakerChannels)
    {
        _engine = engine;
        _filter = inChannels;
        Affine = new MeloAffine<T>(engine, layers, 2);
        for (int i = 0; i < flows; i++) Flows.Add(new MeloConvFlow<T>(engine, layers, _filter, kernel));
        PostPre = MeloOps.Conv(layers, 1, _filter, 1);
        PostProjection = MeloOps.Conv(layers, _filter, _filter, 1);
        PostConvs = new MeloDdsConv<T>(engine, layers, _filter, kernel, 3, dropout);
        PostAffine = new MeloAffine<T>(engine, layers, 2);
        for (int i = 0; i < 4; i++) PostFlows.Add(new MeloConvFlow<T>(engine, layers, _filter, kernel));
        Pre = MeloOps.Conv(layers, inChannels, _filter, 1);
        Projection = MeloOps.Conv(layers, _filter, _filter, 1);
        Convs = new MeloDdsConv<T>(engine, layers, _filter, kernel, 3, dropout);
        if (speakerChannels > 0) Condition = MeloOps.Conv(layers, speakerChannels, _filter, 1);
    }

    public MeloAffine<T> Affine { get; }
    public List<MeloConvFlow<T>> Flows { get; } = new();
    public NormedConv1DLayer<T> PostPre { get; }
    public NormedConv1DLayer<T> PostProjection { get; }
    public MeloDdsConv<T> PostConvs { get; }
    public MeloAffine<T> PostAffine { get; }
    public List<MeloConvFlow<T>> PostFlows { get; } = new();
    public NormedConv1DLayer<T> Pre { get; }
    public NormedConv1DLayer<T> Projection { get; }
    public MeloDdsConv<T> Convs { get; }
    public NormedConv1DLayer<T>? Condition { get; }

    private Tensor<T> Conditioning(Tensor<T> x, Tensor<T>? g, bool training, Random random)
    {
        x = Pre.Forward(MeloOps.Detach(x));
        if (g is not null && Condition is not null) x = _engine.TensorAdd(x, MeloOps.OverTime(_engine, Condition.Forward(MeloOps.Detach(g)), x.Shape[2]));
        return Projection.Forward(Convs.Forward(x, null, training, random));
    }

    private static double Gaussian(Random random)
    {
        double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
        return Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
    }

    private static Tensor<T> Noise(int length, Random random, double scale)
    {
        var z = new Tensor<T>(new[] { 1, 2, length });
        for (int i = 0; i < z.Length; i++) z[i] = NumOps.FromDouble(scale * Gaussian(random));
        return z;
    }

    /// <summary>The negative variational lower bound of the durations <paramref name="w"/> <c>[1, 1, L]</c>, given the
    /// posterior noise <paramref name="eq"/> <c>[1, 2, L]</c> (drawn when null).</summary>
    public Tensor<T> NegativeLogLikelihood(Tensor<T> x, Tensor<T> w, Tensor<T>? g, bool training, Random random, Tensor<T>? eq = null)
    {
        int length = x.Shape[2];
        var c = Conditioning(x, g, training, random);
        var hw = PostProjection.Forward(PostConvs.Forward(PostPre.Forward(w), null, training, random));
        eq ??= Noise(length, random, 1.0);
        var postCondition = _engine.TensorAdd(c, hw);
        var (zq, logDetQ) = PostAffine.Transform(eq, false);
        foreach (var flow in PostFlows)
        {
            var (z, d) = flow.Transform(zq, postCondition, false);
            zq = MeloOps.Flip(_engine, z);
            logDetQ = _engine.TensorAdd(logDetQ!, d!);
        }
        var zu = MeloOps.Channels(_engine, zq, 0, 1);
        var z1 = MeloOps.Channels(_engine, zq, 1, 1);
        var u = _engine.Sigmoid(zu);
        var z0 = _engine.TensorSubtract(w, u);
        // logsigmoid(z) + logsigmoid(−z) = −softplus(−z) − softplus(z).
        var logSigmoids = _engine.TensorNegate(_engine.TensorAdd(_engine.Softplus(zu), _engine.Softplus(_engine.TensorNegate(zu))));
        logDetQ = _engine.TensorAdd(logDetQ!, _engine.ReduceSum(logSigmoids, new[] { 0, 1, 2 }, keepDims: false));
        var logQ = _engine.TensorSubtract(
            _engine.TensorMultiplyScalar(_engine.ReduceSum(_engine.TensorAddScalar(_engine.TensorMultiply(eq, eq), NumOps.FromDouble(Math.Log(2 * Math.PI))),
                new[] { 0, 1, 2 }, keepDims: false), NumOps.FromDouble(-0.5)),
            logDetQ);

        // Log flow: log(max(z0, 1e-5)), log-determinant −Σ log.
        var logZ0 = _engine.TensorLog(_engine.TensorClamp(z0, NumOps.FromDouble(1e-5), NumOps.FromDouble(double.PositiveInfinity)));
        Tensor<T> logDet = _engine.TensorNegate(_engine.ReduceSum(logZ0, new[] { 0, 1, 2 }, keepDims: false));
        var zp = _engine.TensorConcatenate(new[] { logZ0, z1 }, 1);
        var (affined, affineDet) = Affine.Transform(zp, false);
        zp = affined;
        logDet = _engine.TensorAdd(logDet, affineDet!);
        foreach (var flow in Flows)
        {
            var (z, d) = flow.Transform(zp, c, false);
            zp = MeloOps.Flip(_engine, z);
            logDet = _engine.TensorAdd(logDet, d!);
        }
        var nll = _engine.TensorSubtract(
            _engine.TensorMultiplyScalar(_engine.ReduceSum(_engine.TensorAddScalar(_engine.TensorMultiply(zp, zp), NumOps.FromDouble(Math.Log(2 * Math.PI))),
                new[] { 0, 1, 2 }, keepDims: false), NumOps.FromDouble(0.5)),
            logDet);
        return _engine.TensorAdd(nll, logQ);
    }

    /// <summary>Log-durations <c>[1, 1, L]</c> sampled by inverting the flows from noise <paramref name="z"/>
    /// <c>[1, 2, L]</c> (drawn with σ <paramref name="noiseScale"/> when null).</summary>
    public Tensor<T> Sample(Tensor<T> x, Tensor<T>? g, Random random, double noiseScale, Tensor<T>? z = null)
    {
        int length = x.Shape[2];
        var c = Conditioning(x, g, false, random);
        z ??= Noise(length, random, noiseScale);
        // The reversed flows [flip, convflow_n, …, flip, convflow_1, affine] without convflow_1's preceding pair member
        // (flows[:-2] + [flows[-1]]): every flip, the flows n … 2, then the affine.
        for (int i = Flows.Count - 1; i >= 1; i--)
        {
            z = MeloOps.Flip(_engine, z);
            z = Flows[i].Transform(z, c, true).Output;
        }
        z = MeloOps.Flip(_engine, z);
        z = Affine.Transform(z, true).Output;
        return MeloOps.Channels(_engine, z, 0, 1);
    }

    public void LoadTorch(Func<string, double[]> read, string prefix)
    {
        Affine.LoadTorch(read, prefix + "flows.0.");
        for (int i = 0; i < Flows.Count; i++) Flows[i].LoadTorch(_engine, read, $"{prefix}flows.{1 + 2 * i}.");
        MeloOps.LoadConv(read, PostPre, prefix + "post_pre");
        MeloOps.LoadConv(read, PostProjection, prefix + "post_proj");
        PostConvs.LoadTorch(_engine, read, prefix + "post_convs.");
        PostAffine.LoadTorch(read, prefix + "post_flows.0.");
        for (int i = 0; i < PostFlows.Count; i++) PostFlows[i].LoadTorch(_engine, read, $"{prefix}post_flows.{1 + 2 * i}.");
        MeloOps.LoadConv(read, Pre, prefix + "pre");
        MeloOps.LoadConv(read, Projection, prefix + "proj");
        Convs.LoadTorch(_engine, read, prefix + "convs.");
        if (Condition is not null) MeloOps.LoadConv(read, Condition, prefix + "cond");
    }
}

/// <summary>
/// VITS's WaveNet (<c>modules.WN</c>): weight-normalized dilated convolutions with gated tanh·sigmoid activations, a
/// speaker condition sliced per layer from one weight-normalized 1×1 convolution, and residual and skip outputs.
/// </summary>
internal sealed class MeloWaveNet<T>
{
    private readonly IEngine _engine;
    private readonly int _hidden;
    private readonly List<NormedConv1DLayer<T>> _in = new(), _resSkip = new();
    private readonly NormedConv1DLayer<T>? _condition;

    public MeloWaveNet(IEngine engine, List<LayerBase<T>> layers, int hidden, int kernel, int dilationRate, int count, int speakerChannels)
    {
        _engine = engine;
        _hidden = hidden;
        if (speakerChannels > 0) _condition = MeloOps.WeightNormConv(layers, speakerChannels, 2 * hidden * count, 1);
        for (int i = 0; i < count; i++)
        {
            int dilation = (int)Math.Pow(dilationRate, i);
            _in.Add(MeloOps.WeightNormConv(layers, hidden, 2 * hidden, kernel, dilation, (kernel * dilation - dilation) / 2));
            _resSkip.Add(MeloOps.WeightNormConv(layers, hidden, i < count - 1 ? 2 * hidden : hidden, 1));
        }
    }

    public Tensor<T> Forward(Tensor<T> x, Tensor<T>? g)
    {
        int t = x.Shape[2];
        Tensor<T>? output = null;
        var conditions = g is not null && _condition is not null ? _condition.Forward(g) : null;
        for (int i = 0; i < _in.Count; i++)
        {
            var xin = _in[i].Forward(x);
            if (conditions is not null)
                xin = _engine.TensorAdd(xin, MeloOps.OverTime(_engine, MeloOps.Channels(_engine, conditions, i * 2 * _hidden, 2 * _hidden), t));
            var acts = _engine.TensorMultiply(_engine.Tanh(MeloOps.Channels(_engine, xin, 0, _hidden)), _engine.Sigmoid(MeloOps.Channels(_engine, xin, _hidden, _hidden)));
            var rs = _resSkip[i].Forward(acts);
            if (i < _in.Count - 1)
            {
                x = _engine.TensorAdd(x, MeloOps.Channels(_engine, rs, 0, _hidden));
                var skip = MeloOps.Channels(_engine, rs, _hidden, _hidden);
                output = output is null ? skip : _engine.TensorAdd(output, skip);
            }
            else
            {
                output = output is null ? rs : _engine.TensorAdd(output, rs);
            }
        }
        return output!;
    }

    public void LoadTorch(Func<string, double[]> read, string prefix)
    {
        if (_condition is not null) MeloOps.LoadConv(read, _condition, prefix + "cond_layer");
        for (int i = 0; i < _in.Count; i++)
        {
            MeloOps.LoadConv(read, _in[i], $"{prefix}in_layers.{i}");
            MeloOps.LoadConv(read, _resSkip[i], $"{prefix}res_skip_layers.{i}");
        }
    }
}

/// <summary>VITS's posterior encoder (<c>models.PosteriorEncoder</c>): a 1×1 convolution, WaveNet (kernel 5, 16 layers)
/// and a projection to the posterior's mean and log-scale; z = m + ε·e^{logs}.</summary>
internal sealed class MeloPosteriorEncoder<T>
{
    private readonly IEngine _engine;
    private readonly int _out;

    public MeloPosteriorEncoder(IEngine engine, List<LayerBase<T>> layers, int inChannels, int outChannels, int hidden, int speakerChannels)
    {
        _engine = engine;
        _out = outChannels;
        Pre = MeloOps.Conv(layers, inChannels, hidden, 1);
        Net = new MeloWaveNet<T>(engine, layers, hidden, 5, 1, 16, speakerChannels);
        Projection = MeloOps.Conv(layers, hidden, 2 * outChannels, 1);
    }

    public NormedConv1DLayer<T> Pre { get; }
    public MeloWaveNet<T> Net { get; }
    public NormedConv1DLayer<T> Projection { get; }

    public (Tensor<T> Z, Tensor<T> Mean, Tensor<T> LogScale) Forward(Tensor<T> spectrogram, Tensor<T>? g, Tensor<T> noise)
    {
        var stats = Projection.Forward(Net.Forward(Pre.Forward(spectrogram), g));
        var m = MeloOps.Channels(_engine, stats, 0, _out);
        var logs = MeloOps.Channels(_engine, stats, _out, _out);
        return (_engine.TensorAdd(m, _engine.TensorMultiply(noise, _engine.TensorExp(logs))), m, logs);
    }

    public void LoadTorch(Func<string, double[]> read, string prefix)
    {
        MeloOps.LoadConv(read, Pre, prefix + "pre");
        Net.LoadTorch(read, prefix + "enc.");
        MeloOps.LoadConv(read, Projection, prefix + "proj");
    }
}

/// <summary>
/// Bert-VITS2's transformer coupling flow as MeloTTS uses it (<c>models.TransformerCouplingBlock</c> of
/// <c>modules.TransformerCouplingLayer</c>, mean-only): per coupling, the first half's 1×1 projection through a
/// three-block relative-attention encoder (speaker-conditioned), a zero-initialized projection to a shift of the second
/// half, then a channel flip.
/// </summary>
internal sealed class MeloTransformerFlow<T>
{
    private readonly IEngine _engine;
    private readonly int _half;
    private readonly List<(NormedConv1DLayer<T> Pre, MeloAttentionEncoder<T> Encoder, NormedConv1DLayer<T> Post)> _couplings = new();

    public MeloTransformerFlow(IEngine engine, List<LayerBase<T>> layers, int channels, int hidden, int filter, int heads, int blocks,
        int kernel, double dropout, int flows, int speakerChannels)
    {
        _engine = engine;
        _half = channels / 2;
        for (int i = 0; i < flows; i++)
        {
            var pre = MeloOps.Conv(layers, _half, hidden, 1);
            var encoder = new MeloAttentionEncoder<T>(engine, layers, hidden, filter, heads, blocks, kernel, dropout, speakerChannels);
            var post = MeloOps.Conv(layers, hidden, _half, 1);
            post.LoadTorchWeights(new double[_half * hidden], null, new double[_half]);       // zero-initialized
            _couplings.Add((pre, encoder, post));
        }
    }

    private Tensor<T> Shift(int i, Tensor<T> x0, Tensor<T>? g, bool training, Random random)
    {
        var (pre, encoder, post) = _couplings[i];
        return post.Forward(encoder.Forward(pre.Forward(x0), g, training, random));
    }

    /// <summary>z → z_p (training) or, with <paramref name="reverse"/>, z_p → z.</summary>
    public Tensor<T> Forward(Tensor<T> x, Tensor<T>? g, bool reverse, bool training, Random random)
    {
        if (!reverse)
        {
            for (int i = 0; i < _couplings.Count; i++)
            {
                var x0 = MeloOps.Channels(_engine, x, 0, _half);
                var x1 = _engine.TensorAdd(Shift(i, x0, g, training, random), MeloOps.Channels(_engine, x, _half, _half));
                x = MeloOps.Flip(_engine, _engine.TensorConcatenate(new[] { x0, x1 }, 1));
            }
            return x;
        }
        for (int i = _couplings.Count - 1; i >= 0; i--)
        {
            x = MeloOps.Flip(_engine, x);
            var x0 = MeloOps.Channels(_engine, x, 0, _half);
            var x1 = _engine.TensorSubtract(MeloOps.Channels(_engine, x, _half, _half), Shift(i, x0, g, training, random));
            x = _engine.TensorConcatenate(new[] { x0, x1 }, 1);
        }
        return x;
    }

    public void LoadTorch(Func<string, double[]> read, string prefix)
    {
        for (int i = 0; i < _couplings.Count; i++)
        {
            string p = $"{prefix}flows.{2 * i}.";
            MeloOps.LoadConv(read, _couplings[i].Pre, p + "pre");
            _couplings[i].Encoder.LoadTorch(_engine, read, p + "enc.");
            MeloOps.LoadConv(read, _couplings[i].Post, p + "post");
        }
    }
}
