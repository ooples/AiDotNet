using System.Collections.Generic;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.Classic;

namespace AiDotNet.TextToSpeech.EndToEnd;

/// <summary>A per-channel affine map <c>y = m + e^{logs} x</c> with m and logs starting at zero (VITS
/// <c>ElementwiseAffine</c>). Input <c>[1, channels, time]</c>.</summary>
[LayerCategory(LayerCategory.Structural)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, TestInputShape = "1, 2, 5", TestConstructorArgs = "2")]
[ElementWiseShape(Note = "A per-channel affine map; the shape is carried through.")]
[AutoParameters]
internal sealed partial class ChannelAffineLayer<T> : LayerBase<T>
{
    private readonly int _channels;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]
    private Tensor<T> _shift;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _logScale;

    public override bool SupportsTraining => true;

    public ChannelAffineLayer([LayerState] int channels)
        : base(new[] { channels }, new[] { channels })
    {
        _channels = channels;
        _shift = new Tensor<T>(new[] { channels });
        _logScale = new Tensor<T>(new[] { channels });
        RegisterTrainableParameter(_shift, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_logScale, PersistentTensorRole.Weights);
    }

    private Tensor<T> Rows(Tensor<T> v, int time) => Engine.TensorTile(Engine.Reshape(v, new[] { 1, _channels, 1 }), new[] { 1, 1, time });

    /// <summary>The map or its inverse, and the log-determinant <c>T Σ logs</c> (forward only).</summary>
    public (Tensor<T> Output, Tensor<T>? LogDeterminant) Transform(Tensor<T> x, bool reverse)
    {
        int time = x.Shape[2];
        if (reverse)
            return (Engine.TensorMultiply(Engine.TensorSubtract(x, Rows(_shift, time)), Rows(Engine.TensorExp(Engine.TensorNegate(_logScale)), time)), null);
        var y = Engine.TensorAdd(Rows(_shift, time), Engine.TensorMultiply(Rows(Engine.TensorExp(_logScale), time), x));
        return (y, Engine.TensorMultiplyScalar(Engine.ReduceSum(_logScale, new[] { 0 }, keepDims: false), NumOps.FromDouble(time)));
    }

    protected override Tensor<T> ForwardTraced(Tensor<T> input) => Transform(input, false).Output;

    public override void ResetState()
    {
    }

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["Channels"] = _channels.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }
}

/// <summary>Channel-wise LayerNorm of <c>[1, C, T]</c> (VITS <c>modules.LayerNorm</c>, ε = 1e-5).</summary>
internal static class VitsOps
{
    public static Tensor<T> ChannelNorm<T>(IEngine engine, LayerNormalizationLayer<T> norm, Tensor<T> x)
    {
        int c = x.Shape[1], t = x.Shape[2];
        var rows = norm.Forward(engine.TensorTranspose(engine.Reshape(x, new[] { c, t })));
        return engine.Reshape(engine.TensorTranspose(rows), new[] { 1, c, t });
    }

    public static Tensor<T> Flip<T>(IEngine engine, Tensor<T> x)
    {
        int c = x.Shape[1], t = x.Shape[2];
        var index = new Tensor<int>(new[] { c });
        for (int i = 0; i < c; i++) index[i] = c - 1 - i;
        return engine.Reshape(engine.TensorIndexSelect(engine.Reshape(x, new[] { c, t }), index, 0), new[] { 1, c, t });
    }

    public static Tensor<T> Slice<T>(IEngine engine, Tensor<T> x, int start, int count)
        => engine.TensorSlice(x, new[] { 0, start, 0 }, new[] { 1, count, x.Shape[2] });
}

/// <summary>Dilated depth-separable convolutions (VITS <c>DDSConv</c>): per layer a depthwise convolution with dilation
/// k^i, LayerNorm, GELU, a 1×1 convolution, LayerNorm, GELU, dropout and a residual.</summary>
internal sealed class VitsDdsConv<T>
{
    private readonly IEngine _engine;
    private readonly List<(GroupedConv1DLayer<T> Separable, LayerNormalizationLayer<T> Norm1, Conv1DLayer<T> Pointwise,
        LayerNormalizationLayer<T> Norm2)> _layers = new();
    private readonly DropoutLayer<T>? _dropout;

    public VitsDdsConv(IEngine engine, List<LayerBase<T>> owned, int channels, int kernelSize, int layers, double dropout)
    {
        _engine = engine;
        for (int i = 0; i < layers; i++)
        {
            int dilation = (int)Math.Pow(kernelSize, i);
            var sep = new GroupedConv1DLayer<T>(channels, channels, kernelSize, channels, (kernelSize * dilation - dilation) / 2, dilation);
            var n1 = new LayerNormalizationLayer<T>(channels, 1e-5);
            var pw = new Conv1DLayer<T>(channels, channels, 1, 1, 1, 0);
            var n2 = new LayerNormalizationLayer<T>(channels, 1e-5);
            owned.AddRange(new LayerBase<T>[] { sep, n1, pw, n2 });
            _layers.Add((sep, n1, pw, n2));
        }
        if (dropout > 0)
        {
            _dropout = new DropoutLayer<T>(dropout);
            owned.Add(_dropout);
        }
    }

    public Tensor<T> Forward(Tensor<T> x, Tensor<T>? condition)
    {
        if (condition is not null) x = _engine.TensorAdd(x, condition);
        foreach (var (sep, n1, pw, n2) in _layers)
        {
            var y = _engine.GELU(VitsOps.ChannelNorm(_engine, n1, sep.Forward(x)));
            y = _engine.GELU(VitsOps.ChannelNorm(_engine, n2, pw.Forward(y)));
            if (_dropout is not null) y = _dropout.Forward(y);
            x = _engine.TensorAdd(x, y);
        }
        return x;
    }
}

/// <summary>A spline coupling layer (VITS <c>ConvFlow</c>): the first half of 2 channels conditions, through a 1×1
/// convolution, DDSConv and a zero-initialized projection, a rational-quadratic spline (10 bins, tail bound 5) on the
/// second half.</summary>
internal sealed class VitsConvFlow<T>
{
    private const int Bins = 10;
    private const double TailBound = 5.0;
    private readonly IEngine _engine;
    private readonly Conv1DLayer<T> _pre;
    private readonly VitsDdsConv<T> _convs;
    private readonly Conv1DLayer<T> _proj;
    private readonly int _filters;

    public VitsConvFlow(IEngine engine, List<LayerBase<T>> owned, int filters, int kernelSize, int layers)
    {
        _engine = engine;
        _filters = filters;
        _pre = new Conv1DLayer<T>(1, filters, 1, 1, 1, 0);
        owned.Add(_pre);
        _convs = new VitsDdsConv<T>(engine, owned, filters, kernelSize, layers, 0.0);
        _proj = new Conv1DLayer<T>(filters, 3 * Bins - 1, 1, 1, 1, 0, null, new AiDotNet.Initialization.ZeroInitializationStrategy<T>());
        owned.Add(_proj);
    }

    public (Tensor<T> Output, Tensor<T>? LogDeterminant) Transform(Tensor<T> x, Tensor<T> condition, bool reverse)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int t = x.Shape[2];
        var x0 = VitsOps.Slice(_engine, x, 0, 1);
        var x1 = VitsOps.Slice(_engine, x, 1, 1);
        var h = _proj.Forward(_convs.Forward(_pre.Forward(x0), condition));                       // [1, 29, T]
        var rows = _engine.TensorTranspose(_engine.Reshape(h, new[] { 3 * Bins - 1, t }));        // [T, 29]
        double scale = 1.0 / Math.Sqrt(_filters);
        var widths = _engine.TensorMultiplyScalar(_engine.TensorSlice(rows, new[] { 0, 0 }, new[] { t, Bins }), ops.FromDouble(scale));
        var heights = _engine.TensorMultiplyScalar(_engine.TensorSlice(rows, new[] { 0, Bins }, new[] { t, Bins }), ops.FromDouble(scale));
        var derivatives = _engine.TensorSlice(rows, new[] { 0, 2 * Bins }, new[] { t, Bins - 1 });
        var (y1, logAbsDet) = RationalQuadraticSpline.Apply(_engine, _engine.Reshape(x1, new[] { t }), widths, heights, derivatives, reverse, TailBound);
        var output = _engine.TensorConcatenate(new[] { x0, _engine.Reshape(y1, new[] { 1, 1, t }) }, 1);
        return (output, reverse ? null : _engine.ReduceSum(logAbsDet, new[] { 0 }, keepDims: false));
    }
}

/// <summary>
/// VITS's stochastic duration predictor (Kim et al. 2021 §2.2.2, App. A.3; reference <c>StochasticDurationPredictor</c>):
/// a flow over (d − u, ν) conditioned on the stop-gradient text encoding, trained with variational dequantization and
/// data augmentation through a posterior flow, giving a lower bound on log p(d | c).
/// </summary>
internal sealed class VitsStochasticDurationPredictor<T>
{
    private readonly IEngine _engine;
    private readonly List<LayerBase<T>> _layers = new();
    private readonly Conv1DLayer<T> _pre;
    private readonly VitsDdsConv<T> _convs;
    private readonly Conv1DLayer<T> _proj;
    private readonly Conv1DLayer<T> _postPre;
    private readonly VitsDdsConv<T> _postConvs;
    private readonly Conv1DLayer<T> _postProj;
    private readonly ChannelAffineLayer<T> _affine;
    private readonly List<VitsConvFlow<T>> _flows = new();
    private readonly ChannelAffineLayer<T> _postAffine;
    private readonly List<VitsConvFlow<T>> _postFlows = new();
    private readonly Conv1DLayer<T>? _condition;
    private readonly Conv1DLayer<T>? _languageCondition;

    /// <param name="engine">The engine.</param>
    /// <param name="inChannels">The text encoding's width (the hidden width, plus the language embedding in YourTTS).</param>
    /// <param name="filters">The convolutions' width (the hidden width: VITS overrides filter_channels with in_channels,
    /// Coqui passes 192 with a 196-wide multilingual encoding).</param>
    /// <param name="kernelSize">The DDS convolutions' kernel.</param>
    /// <param name="dropout">Dropout.</param>
    /// <param name="flows">Spline flows.</param>
    /// <param name="conditionChannels">Speaker embedding width, or 0.</param>
    /// <param name="languageChannels">Language embedding width, or 0.</param>
    public VitsStochasticDurationPredictor(IEngine engine, int inChannels, int filters, int kernelSize, double dropout, int flows,
        int conditionChannels = 0, int languageChannels = 0)
    {
        _engine = engine;
        _layers.Add(_affine = new ChannelAffineLayer<T>(2));
        for (int i = 0; i < flows; i++) _flows.Add(new VitsConvFlow<T>(engine, _layers, filters, kernelSize, 3));
        _layers.Add(_postPre = new Conv1DLayer<T>(1, filters, 1, 1, 1, 0));
        _layers.Add(_postProj = new Conv1DLayer<T>(filters, filters, 1, 1, 1, 0));
        _postConvs = new VitsDdsConv<T>(engine, _layers, filters, kernelSize, 3, dropout);
        _layers.Add(_postAffine = new ChannelAffineLayer<T>(2));
        for (int i = 0; i < 4; i++) _postFlows.Add(new VitsConvFlow<T>(engine, _layers, filters, kernelSize, 3));
        _layers.Add(_pre = new Conv1DLayer<T>(inChannels, filters, 1, 1, 1, 0));
        _layers.Add(_proj = new Conv1DLayer<T>(filters, filters, 1, 1, 1, 0));
        _convs = new VitsDdsConv<T>(engine, _layers, filters, kernelSize, 3, dropout);
        if (conditionChannels > 0) _layers.Add(_condition = new Conv1DLayer<T>(conditionChannels, filters, 1, 1, 1, 0));
        if (languageChannels > 0) _layers.Add(_languageCondition = new Conv1DLayer<T>(languageChannels, filters, 1, 1, 1, 0));
    }

    public IReadOnlyList<LayerBase<T>> Layers => _layers;

    // The stop-gradient text encoding [1, hidden, T] through pre, plus the stop-gradient speaker and language
    // embeddings [1, c, 1] projected and added over time (reference: x = pre(x.detach()); x += cond(g.detach());
    // x += cond_lang(lang_emb.detach())), then the DDS convolutions.
    private Tensor<T> Condition(Tensor<T> text, Tensor<T>? speaker, Tensor<T>? language)
    {
        var x = _pre.Forward(new Tensor<T>(text._shape, text.ToVector()));
        if (speaker is not null && _condition is not null)
            x = _engine.TensorAdd(x, _engine.TensorTile(_condition.Forward(new Tensor<T>(speaker._shape, speaker.ToVector())), new[] { 1, 1, x.Shape[2] }));
        if (language is not null && _languageCondition is not null)
            x = _engine.TensorAdd(x, _engine.TensorTile(_languageCondition.Forward(new Tensor<T>(language._shape, language.ToVector())), new[] { 1, 1, x.Shape[2] }));
        return _proj.Forward(_convs.Forward(x, null));
    }

    /// <summary>The negative variational lower bound <c>−log q + −log p</c> (summed over tokens) of the durations
    /// <paramref name="durations"/> <c>[1, 1, tokens]</c> given the encoding <c>[1, hidden, tokens]</c>.</summary>
    public Tensor<T> NegativeLogLikelihood(Tensor<T> text, Tensor<T>? speaker, Tensor<T>? language, Tensor<T> durations, Random random)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int t = text.Shape[2];
        var x = Condition(text, speaker, language);
        var hw = _postProj.Forward(_postConvs.Forward(_postPre.Forward(durations), null));
        var postCondition = _engine.TensorAdd(x, hw);

        // Posterior q(u, ν | d, c): e_q ~ N(0, I) through the posterior flows.
        var eq = new Tensor<T>(new[] { 1, 2, t });
        for (int i = 0; i < eq.Length; i++)
        {
            double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
            eq[i] = ops.FromDouble(Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2));
        }
        var (zq, logDetQ) = _postAffine.Transform(eq, false);
        foreach (var flow in _postFlows)
        {
            var (z, d) = flow.Transform(zq, postCondition, false);
            zq = VitsOps.Flip(_engine, z);
            logDetQ = _engine.TensorAdd(logDetQ!, d!);
        }
        var zu = VitsOps.Slice(_engine, zq, 0, 1);
        var z1 = VitsOps.Slice(_engine, zq, 1, 1);
        var u = _engine.Sigmoid(zu);
        var z0 = _engine.TensorSubtract(durations, u);
        // log σ(z) + log σ(−z) = −softplus(−z) − softplus(z)
        var logSigmoids = _engine.TensorNegate(_engine.TensorAdd(_engine.Softplus(zu), _engine.Softplus(_engine.TensorNegate(zu))));
        logDetQ = _engine.TensorAdd(logDetQ!, _engine.ReduceSum(logSigmoids, new[] { 0, 1, 2 }, keepDims: false));
        var logQ = _engine.TensorSubtract(
            _engine.TensorMultiplyScalar(_engine.ReduceSum(_engine.TensorAddScalar(_engine.TensorMultiply(eq, eq), ops.FromDouble(Math.Log(2 * Math.PI))),
                new[] { 0, 1, 2 }, keepDims: false), ops.FromDouble(-0.5)),
            logDetQ);

        // Prior p: log flow on d − u, then the affine and conv flows.
        var logZ0 = _engine.TensorLog(_engine.TensorAddScalar(_engine.ReLU(_engine.TensorAddScalar(z0, ops.FromDouble(-1e-5))), ops.FromDouble(1e-5)));
        Tensor<T> logDet = _engine.TensorNegate(_engine.ReduceSum(logZ0, new[] { 0, 1, 2 }, keepDims: false));
        var zp = _engine.TensorConcatenate(new[] { logZ0, z1 }, 1);
        var (affined, affineDet) = _affine.Transform(zp, false);
        zp = affined;
        logDet = _engine.TensorAdd(logDet, affineDet!);
        foreach (var flow in _flows)
        {
            var (z, d) = flow.Transform(zp, x, false);
            zp = VitsOps.Flip(_engine, z);
            logDet = _engine.TensorAdd(logDet, d!);
        }
        var nll = _engine.TensorSubtract(
            _engine.TensorMultiplyScalar(_engine.ReduceSum(_engine.TensorAddScalar(_engine.TensorMultiply(zp, zp), ops.FromDouble(Math.Log(2 * Math.PI))),
                new[] { 0, 1, 2 }, keepDims: false), ops.FromDouble(0.5)),
            logDet);
        return _engine.TensorAdd(nll, logQ);
    }

    /// <summary>Sampled log-durations <c>[tokens]</c>: noise scaled by <paramref name="noiseScale"/> through the reversed
    /// prior flows (the reference drops the last, unused flow pair when reversing).</summary>
    public Tensor<T> SampleLogDurations(Tensor<T> text, Tensor<T>? speaker, Tensor<T>? language, Random random, double noiseScale)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int t = text.Shape[2];
        var x = Condition(text, speaker, language);
        var z = new Tensor<T>(new[] { 1, 2, t });
        for (int i = 0; i < z.Length; i++)
        {
            double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
            z[i] = ops.FromDouble(noiseScale * Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2));
        }
        // Reversed list: [flip, flow_n, ..., flip, flow_1, affine]; the reference removes the first flip/flow pair's
        // flow (flows[:-2] + [flows[-1]] on the reversed list drops the second-to-last module, the first ConvFlow).
        for (int i = _flows.Count - 1; i >= 1; i--)
        {
            z = VitsOps.Flip(_engine, z);
            z = _flows[i].Transform(z, x, true).Output;
        }
        z = VitsOps.Flip(_engine, z);
        z = _affine.Transform(z, true).Output;
        return _engine.Reshape(VitsOps.Slice(_engine, z, 0, 1), new[] { t });
    }
}

/// <summary>VITS's posterior encoder (§2.1, App. A.2): a 1×1 convolution from the linear spectrogram, a non-causal
/// WaveNet (16 layers, kernel 5) and a projection to mean and log standard deviation; z = m + ε e^{logs}.</summary>
internal sealed class VitsPosteriorEncoder<T>
{
    private readonly IEngine _engine;
    private readonly Conv1DLayer<T> _pre;
    private readonly PortaSpeechWaveNet<T> _net;
    private readonly Conv1DLayer<T> _proj;
    private readonly int _channels;

    public VitsPosteriorEncoder(IEngine engine, List<LayerBase<T>> owned, int inChannels, int channels, int hidden, int kernelSize, int layers, int conditionChannels)
    {
        _engine = engine;
        _channels = channels;
        owned.Add(_pre = new Conv1DLayer<T>(inChannels, hidden, 1, 1, 1, 0));
        _net = new PortaSpeechWaveNet<T>(engine, owned, hidden, kernelSize, layers, conditionChannels);
        owned.Add(_proj = new Conv1DLayer<T>(hidden, 2 * channels, 1, 1, 1, 0));
    }

    public (Tensor<T> Z, Tensor<T> Mean, Tensor<T> LogScale) Forward(Tensor<T> spectrogram, Tensor<T>? condition, Tensor<T> noise)
    {
        var stats = _proj.Forward(_net.Forward(_pre.Forward(spectrogram), condition));
        var m = VitsOps.Slice(_engine, stats, 0, _channels);
        var logs = VitsOps.Slice(_engine, stats, _channels, _channels);
        return (_engine.TensorAdd(m, _engine.TensorMultiply(noise, _engine.TensorExp(logs))), m, logs);
    }
}

/// <summary>VITS's flow f_θ (§2.1, App. A.2): residual coupling layers (1×1 pre, WaveNet, zero-initialized mean-only
/// projection) alternating with channel flips; volume-preserving.</summary>
/// <remarks>With <c>transformerLayers</c> above zero each coupling first passes its conditioning half through a small
/// Transformer with a residual connection, <c>x₀' = x₀ + Transformer(x₀)</c>, before the 1×1 pre-convolution (VITS2 §2.3,
/// Fig. 1b; reference <c>ResidualCouplingTransformersLayer</c>, the <c>pre_conv</c> flow of its LJSpeech configuration:
/// width = the half channels, filter = the half channels, no positional encoding). The half that is transformed is left
/// unchanged, so the coupling stays invertible.</remarks>
internal sealed class VitsFlow<T>
{
    private readonly IEngine _engine;
    private readonly List<(List<RelativePositionTransformerBlock<T>> Transformer, Conv1DLayer<T> Pre, PortaSpeechWaveNet<T> Net, Conv1DLayer<T> Post)> _couplings = new();
    private readonly int _half;

    public VitsFlow(IEngine engine, List<LayerBase<T>> owned, int channels, int hidden, int kernelSize, int layers, int steps, int conditionChannels,
        int transformerLayers = 0, int transformerHeads = 2, int transformerKernelSize = 3, double transformerDropout = 0.1)
    {
        _engine = engine;
        _half = channels / 2;
        for (int i = 0; i < steps; i++)
        {
            var transformer = new List<RelativePositionTransformerBlock<T>>();
            for (int l = 0; l < transformerLayers; l++)
            {
                var block = new RelativePositionTransformerBlock<T>(_half, transformerHeads, _half, transformerKernelSize, transformerDropout, 0, positional: false);
                owned.Add(block);
                transformer.Add(block);
            }
            var pre = new Conv1DLayer<T>(_half, hidden, 1, 1, 1, 0);
            owned.Add(pre);
            var net = new PortaSpeechWaveNet<T>(engine, owned, hidden, kernelSize, layers, conditionChannels);
            var post = new Conv1DLayer<T>(hidden, _half, 1, 1, 1, 0, null, new AiDotNet.Initialization.ZeroInitializationStrategy<T>());
            owned.Add(post);
            _couplings.Add((transformer, pre, net, post));
        }
    }

    public Tensor<T> Forward(Tensor<T> x, Tensor<T>? condition, bool reverse)
    {
        if (!reverse)
        {
            foreach (var c in _couplings) x = VitsOps.Flip(_engine, Couple(x, c, condition, false));
            return x;
        }
        for (int i = _couplings.Count - 1; i >= 0; i--) x = Couple(VitsOps.Flip(_engine, x), _couplings[i], condition, true);
        return x;
    }

    private Tensor<T> Couple(Tensor<T> x, (List<RelativePositionTransformerBlock<T>> Transformer, Conv1DLayer<T> Pre, PortaSpeechWaveNet<T> Net, Conv1DLayer<T> Post) c,
        Tensor<T>? condition, bool reverse)
    {
        var x0 = VitsOps.Slice(_engine, x, 0, _half);
        var x1 = VitsOps.Slice(_engine, x, _half, x.Shape[1] - _half);
        var source = x0;
        if (c.Transformer.Count > 0)
        {
            int time = x0.Shape[2];
            var rows = _engine.TensorTranspose(_engine.Reshape(x0, new[] { _half, time }));          // [T, half]
            foreach (var block in c.Transformer) rows = block.Forward(rows);
            source = _engine.TensorAdd(x0, _engine.Reshape(_engine.TensorTranspose(rows), new[] { 1, _half, time }));
        }
        var m = c.Post.Forward(c.Net.Forward(c.Pre.Forward(source), condition));
        return _engine.TensorConcatenate(new[] { x0, reverse ? _engine.TensorSubtract(x1, m) : _engine.TensorAdd(x1, m) }, 1);
    }
}

// ---- VITS2 duration modules

/// <summary>
/// VITS2's duration generator G(z_d, h_text) (Kong et al. 2023 §2.1, Fig. 1a): two convolutions with ReLU, channel
/// LayerNorm and dropout and a 1×1 projection to the log-duration of every token, reading the stop-gradient text
/// encoding and one standard-normal noise value per token.
/// </summary>
/// <remarks>The layer stack is the reference's <c>DurationPredictor(hidden, 256, 3, 0.5)</c>; the paper's noise input
/// z_d, which the reference leaves out, enters as one extra input channel of the first convolution. A speaker embedding is
/// projected and added to the (stop-gradient) text encoding first, as the reference does.</remarks>
internal sealed class Vits2DurationPredictor<T>
{
    private readonly IEngine _engine;
    private readonly List<LayerBase<T>> _layers = new();
    private readonly Conv1DLayer<T> _conv1;
    private readonly LayerNormalizationLayer<T> _norm1;
    private readonly Conv1DLayer<T> _conv2;
    private readonly LayerNormalizationLayer<T> _norm2;
    private readonly Conv1DLayer<T> _proj;
    private readonly DropoutLayer<T>? _dropout;
    private readonly Conv1DLayer<T>? _condition;
    private readonly Conv1DLayer<T>? _languageCondition;

    public Vits2DurationPredictor(IEngine engine, int inChannels, int filters, int kernelSize, double dropout, int conditionChannels,
        int languageChannels = 0)
    {
        _engine = engine;
        _layers.Add(_conv1 = new Conv1DLayer<T>(inChannels + 1, filters, kernelSize, 1, 1, kernelSize / 2));
        _layers.Add(_norm1 = new LayerNormalizationLayer<T>(filters, 1e-5));
        _layers.Add(_conv2 = new Conv1DLayer<T>(filters, filters, kernelSize, 1, 1, kernelSize / 2));
        _layers.Add(_norm2 = new LayerNormalizationLayer<T>(filters, 1e-5));
        _layers.Add(_proj = new Conv1DLayer<T>(filters, 1, 1, 1, 1, 0));
        if (dropout > 0) _layers.Add(_dropout = new DropoutLayer<T>(dropout));
        if (conditionChannels > 0) _layers.Add(_condition = new Conv1DLayer<T>(conditionChannels, inChannels, 1, 1, 1, 0));
        if (languageChannels > 0) _layers.Add(_languageCondition = new Conv1DLayer<T>(languageChannels, inChannels, 1, 1, 1, 0));
    }

    public IReadOnlyList<LayerBase<T>> Layers => _layers;

    /// <summary>Log-durations <c>[1, 1, tokens]</c> for the text encoding <c>[1, hidden, tokens]</c> and the noise
    /// <c>[1, 1, tokens]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> text, Tensor<T>? speaker, Tensor<T>? language, Tensor<T> noise)
    {
        var x = new Tensor<T>(text._shape, text.ToVector());
        if (speaker is not null && _condition is not null)
            x = _engine.TensorAdd(x, _engine.TensorTile(_condition.Forward(new Tensor<T>(speaker._shape, speaker.ToVector())), new[] { 1, 1, x.Shape[2] }));
        if (language is not null && _languageCondition is not null)
            x = _engine.TensorAdd(x, _engine.TensorTile(_languageCondition.Forward(new Tensor<T>(language._shape, language.ToVector())), new[] { 1, 1, x.Shape[2] }));
        x = _engine.TensorConcatenate(new[] { x, noise }, 1);
        x = VitsOps.ChannelNorm(_engine, _norm1, _engine.ReLU(_conv1.Forward(x)));
        if (_dropout is not null) x = _dropout.Forward(x);
        x = VitsOps.ChannelNorm(_engine, _norm2, _engine.ReLU(_conv2.Forward(x)));
        if (_dropout is not null) x = _dropout.Forward(x);
        return _proj.Forward(x);
    }
}

/// <summary>
/// VITS2's time-step-wise conditional duration discriminator D(d, h_text) (Kong et al. 2023 §2.1, Eq. 1–2): one
/// real/fake score per token from the stop-gradient text encoding and a log-duration.
/// </summary>
/// <remarks>The reference's <c>DurationDiscriminatorV2(hidden, hidden, 3)</c>: the text through two convolutions with ReLU
/// and channel LayerNorm, the duration through a 1×1 projection, both concatenated and through two more such
/// convolutions, then a per-token linear map and a sigmoid.</remarks>
internal sealed class Vits2DurationDiscriminator<T>
{
    private readonly IEngine _engine;
    private readonly List<LayerBase<T>> _layers = new();
    private readonly Conv1DLayer<T> _conv1;
    private readonly LayerNormalizationLayer<T> _norm1;
    private readonly Conv1DLayer<T> _conv2;
    private readonly LayerNormalizationLayer<T> _norm2;
    private readonly Conv1DLayer<T> _durationProjection;
    private readonly Conv1DLayer<T> _preOut1;
    private readonly LayerNormalizationLayer<T> _preOutNorm1;
    private readonly Conv1DLayer<T> _preOut2;
    private readonly LayerNormalizationLayer<T> _preOutNorm2;
    private readonly Conv1DLayer<T> _output;

    public Vits2DurationDiscriminator(IEngine engine, int inChannels, int filters, int kernelSize)
    {
        _engine = engine;
        int pad = kernelSize / 2;
        _layers.Add(_conv1 = new Conv1DLayer<T>(inChannels, filters, kernelSize, 1, 1, pad));
        _layers.Add(_norm1 = new LayerNormalizationLayer<T>(filters, 1e-5));
        _layers.Add(_conv2 = new Conv1DLayer<T>(filters, filters, kernelSize, 1, 1, pad));
        _layers.Add(_norm2 = new LayerNormalizationLayer<T>(filters, 1e-5));
        _layers.Add(_durationProjection = new Conv1DLayer<T>(1, filters, 1, 1, 1, 0));
        _layers.Add(_preOut1 = new Conv1DLayer<T>(2 * filters, filters, kernelSize, 1, 1, pad));
        _layers.Add(_preOutNorm1 = new LayerNormalizationLayer<T>(filters, 1e-5));
        _layers.Add(_preOut2 = new Conv1DLayer<T>(filters, filters, kernelSize, 1, 1, pad));
        _layers.Add(_preOutNorm2 = new LayerNormalizationLayer<T>(filters, 1e-5));
        // nn.Linear(filters, 1) applied per token is a 1x1 convolution.
        _layers.Add(_output = new Conv1DLayer<T>(filters, 1, 1, 1, 1, 0));
    }

    public IReadOnlyList<LayerBase<T>> Layers => _layers;

    /// <summary>The text features shared by the real and the generated duration (stop-gradient input).</summary>
    public Tensor<T> Encode(Tensor<T> text)
    {
        var x = new Tensor<T>(text._shape, text.ToVector());
        x = VitsOps.ChannelNorm(_engine, _norm1, _engine.ReLU(_conv1.Forward(x)));
        return VitsOps.ChannelNorm(_engine, _norm2, _engine.ReLU(_conv2.Forward(x)));
    }

    /// <summary>Per-token probabilities <c>[1, 1, tokens]</c> that <paramref name="logDurations"/> are real.</summary>
    public Tensor<T> Score(Tensor<T> features, Tensor<T> logDurations)
    {
        var x = _engine.TensorConcatenate(new[] { features, _durationProjection.Forward(logDurations) }, 1);
        x = VitsOps.ChannelNorm(_engine, _preOutNorm1, _engine.ReLU(_preOut1.Forward(x)));
        x = VitsOps.ChannelNorm(_engine, _preOutNorm2, _engine.ReLU(_preOut2.Forward(x)));
        return _engine.Sigmoid(_output.Forward(x));
    }
}
