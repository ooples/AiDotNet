using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.EndToEnd;
using AiDotNet.TextToSpeech.Latest;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>The sizes of GPT-SoVITS's v3/v4/V5 DiT (<c>DiT(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512,
/// conv_layers=4)</c>).</summary>
internal sealed record GptSoVitsDiTConfiguration(int Dim = 1024, int Depth = 22, int Heads = 16, int HeadDim = 64, int FfMult = 2,
    int MelDim = 100, int TextDim = 512, int ConvLayers = 4, double Dropout = 0.1, bool UseStepEmbedding = true);

/// <summary>
/// GPT-SoVITS's diffusion Transformer (<c>f5_tts/model/backbones/dit.py</c>, after F5-TTS) for one sequence: the
/// conditioning features with absolute positions refined by ConvNeXt V2 blocks, the noisy mel, the prompt mel and the
/// features projected together with a convolutional position embedding, adaLN-zero DiT blocks (rotary positions on the
/// first head) conditioned on the time embedding plus the step size's (shortcut models; not in V5), and a final modulated
/// projection to the velocity.
/// </summary>
internal sealed class GptSoVitsDiT<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly float[] _textPositions;
    private const int TextPositionRows = 4096;

    internal sealed record ConvNeXt(NormedConv1DLayer<T> Depthwise, LayerNormalizationLayer<T> Norm, DenseLayer<T> Expand,
        BiasFreeLinearLayer<T> GrnGamma, BiasFreeLinearLayer<T> GrnBeta, DenseLayer<T> Project);

    internal sealed record Block(DenseLayer<T> Modulation, DenseLayer<T> Query, DenseLayer<T> Key, DenseLayer<T> Value, DenseLayer<T> Out,
        DenseLayer<T> FfIn, DenseLayer<T> FfOut);

    public GptSoVitsDiT(IEngine engine, List<LayerBase<T>> layers, GptSoVitsDiTConfiguration configuration, Random random)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        int inner = c.Heads * c.HeadDim;
        TimeIn = IndexTtsOps.Linear(engine, layers, random, 256, c.Dim);
        TimeOut = IndexTtsOps.Linear(engine, layers, random, c.Dim, c.Dim);
        if (c.UseStepEmbedding)
        {
            StepIn = IndexTtsOps.Linear(engine, layers, random, 256, c.Dim);
            StepOut = IndexTtsOps.Linear(engine, layers, random, c.Dim, c.Dim);
        }
        for (int i = 0; i < c.ConvLayers; i++)
            TextBlocks.Add(new ConvNeXt(MeloOps.Conv(layers, c.TextDim, c.TextDim, 7, 1, 3, c.TextDim), IndexTtsOps.LayerNorm(layers, c.TextDim, 1e-6),
                IndexTtsOps.Linear(engine, layers, random, c.TextDim, 2 * c.TextDim), IndexTtsOps.Values(layers, new double[2 * c.TextDim]),
                IndexTtsOps.Values(layers, new double[2 * c.TextDim]), IndexTtsOps.Linear(engine, layers, random, 2 * c.TextDim, c.TextDim)));
        InputProjection = IndexTtsOps.Linear(engine, layers, random, 2 * c.MelDim + c.TextDim, c.Dim);
        PositionConv1 = MeloOps.Conv(layers, c.Dim, c.Dim, 31, 1, 15, 16);
        PositionConv2 = MeloOps.Conv(layers, c.Dim, c.Dim, 31, 1, 15, 16);
        for (int i = 0; i < c.Depth; i++)
            Blocks.Add(new Block(IndexTtsOps.Linear(engine, layers, random, c.Dim, 6 * c.Dim), IndexTtsOps.Linear(engine, layers, random, c.Dim, inner),
                IndexTtsOps.Linear(engine, layers, random, c.Dim, inner), IndexTtsOps.Linear(engine, layers, random, c.Dim, inner),
                IndexTtsOps.Linear(engine, layers, random, inner, c.Dim), IndexTtsOps.Linear(engine, layers, random, c.Dim, c.FfMult * c.Dim),
                IndexTtsOps.Linear(engine, layers, random, c.FfMult * c.Dim, c.Dim)));
        NormOut = IndexTtsOps.Linear(engine, layers, random, c.Dim, 2 * c.Dim);
        ProjectionOut = IndexTtsOps.Linear(engine, layers, random, c.Dim, c.MelDim);
        // precompute_freqs_cis(text_dim, 4096): [cos; sin] of t·θ_i, built in float32.
        int half = c.TextDim / 2;
        _textPositions = new float[TextPositionRows * c.TextDim];
        var freqs = new float[half];
        for (int i = 0; i < half; i++) freqs[i] = 1.0f / MathF.Pow(10000.0f, (2 * i) / (float)c.TextDim);
        for (int t = 0; t < TextPositionRows; t++)
            for (int i = 0; i < half; i++)
            {
                float a = t * freqs[i];
                _textPositions[t * c.TextDim + i] = MathF.Cos(a);
                _textPositions[t * c.TextDim + half + i] = MathF.Sin(a);
            }
    }

    public GptSoVitsDiTConfiguration Configuration { get; }
    public DenseLayer<T> TimeIn { get; }
    public DenseLayer<T> TimeOut { get; }
    public DenseLayer<T>? StepIn { get; }
    public DenseLayer<T>? StepOut { get; }
    public List<ConvNeXt> TextBlocks { get; } = new();
    public DenseLayer<T> InputProjection { get; }
    public NormedConv1DLayer<T> PositionConv1 { get; }
    public NormedConv1DLayer<T> PositionConv2 { get; }
    public List<Block> Blocks { get; } = new();
    public DenseLayer<T> NormOut { get; }
    public DenseLayer<T> ProjectionOut { get; }

    private Tensor<T> Drop(Tensor<T> x, bool training, Random? random) =>
        training && Configuration.Dropout > 0 ? T5Seq2Seq<T>.Dropout(_engine, x, Configuration.Dropout, random ?? throw new ArgumentNullException(nameof(random))) : x;

    // SinusPositionEmbedding(256)'s frequencies exp(−i·ln(10⁴)/127) exactly as torch computes them in float32 (its
    // vectorized expf is one ulp off the correctly rounded value at i = 4, which moves the angle 1000·t·f by up to 3e-5).
    private static readonly float[] TimeFrequencies =
    {
        1.0f, 0.930044949054718f, 0.8649836182594299f, 0.8044736385345459f, 0.7481966018676758f, 0.695856511592865f,
        0.6471778154373169f, 0.6019044518470764f, 0.5597981810569763f, 0.5206375122070312f, 0.48421624302864075f, 0.45034289360046387f,
        0.41883912682533264f, 0.3895392119884491f, 0.3622889518737793f, 0.3369450569152832f, 0.31337401270866394f, 0.29145193099975586f,
        0.2710634171962738f, 0.2521011531352997f, 0.23446539044380188f, 0.2180633693933487f, 0.20280872285366058f, 0.18862122297286987f,
        0.17542622983455658f, 0.16315427422523499f, 0.15174080431461334f, 0.14112575352191925f, 0.13125328719615936f, 0.12207148224115372f,
        0.11353196203708649f, 0.10558982193470001f, 0.09820327907800674f, 0.09133346378803253f, 0.0849442183971405f, 0.07900194078683853f,
        0.07347536832094193f, 0.0683353915810585f, 0.06355498731136322f, 0.05910899117588997f, 0.054974015802145004f, 0.05112830549478531f,
        0.047551631927490234f, 0.04422515258193016f, 0.04113137722015381f, 0.0382540263235569f, 0.035577964037656784f, 0.033089105039834976f,
        0.03077436052262783f, 0.02862153761088848f, 0.026619315147399902f, 0.024757158011198044f, 0.0230252705514431f, 0.021414535120129585f,
        0.019916478544473648f, 0.01852322369813919f, 0.017227426171302795f, 0.01602228358387947f, 0.014901447109878063f, 0.01385901216417551f,
        0.012889507226645947f, 0.011987817473709583f, 0.0111492108553648f, 0.010369264520704746f, 0.009643884375691414f, 0.008969247341156006f,
        0.008341801352798939f, 0.007758251391351223f, 0.007215520367026329f, 0.0067107598297297955f, 0.006241306662559509f, 0.005804696585983038f,
        0.005398630164563656f, 0.005020967219024897f, 0.0046697258949279785f, 0.004343053791671991f, 0.004039236344397068f, 0.0037566719111055136f,
        0.0034938727039843798f, 0.003249459434300661f, 0.0030221424531191587f, 0.002810728969052434f, 0.002614103490486741f, 0.002431234112009406f,
        0.002261157613247633f, 0.0021029775962233543f, 0.001955864019691944f, 0.0018190409755334258f, 0.0016917901812121272f, 0.001573440502397716f,
        0.001463370630517602f, 0.0013610008172690868f, 0.0012657915940508246f, 0.0011772433063015342f, 0.0010948888957500458f, 0.0010182959958910942f,
        0.0009470612858422101f, 0.0008808093261905015f, 0.0008191924425773323f, 0.0007618855452165008f, 0.000708587933331728f, 0.0006590184639208019f,
        0.0006129168905317783f, 0.0005700403708033264f, 0.0005301630590111017f, 0.0004930755821987987f, 0.000458582304418087f, 0.00042650222894735634f,
        0.0003966661461163312f, 0.0003689174191094935f, 0.000343109859386459f, 0.00031910763937048614f, 0.00029678421560674906f, 0.00027602273621596396f,
        0.0002567135961726308f, 0.00023875522310845554f, 0.00022205313143786043f, 0.00020651923841796815f, 0.00019207221339456737f, 0.0001786358334356919f,
        0.0001661393907852471f, 0.00015451697981916368f, 0.00014370775897987187f, 0.0001336547138635069f, 0.00012430490460246801f, 0.00011560917482711375f,
        0.00010752164962468669f, 9.99999901978299e-05f,
    };

    // SinusPositionEmbedding(256) of scale·t: the float32 frequencies, the product and sin/cos in the model's precision.
    private static Tensor<T> Sinusoidal(double t)
    {
        const int half = 128;
        var e = new Tensor<T>(new[] { 1, 2 * half });
        for (int i = 0; i < half; i++)
        {
            double a = 1000.0 * t * TimeFrequencies[i];
            e[0, i] = NumOps.FromDouble(Math.Sin(a));
            e[0, half + i] = NumOps.FromDouble(Math.Cos(a));
        }
        return e;
    }

    /// <summary>The time embedding <c>[1, dim]</c>, plus the step size's when the model has one.</summary>
    public Tensor<T> TimeEmbedding(double t, double? step)
    {
        var time = TimeOut.Forward(_engine.Swish(TimeIn.Forward(Sinusoidal(t))));
        if (StepIn is null || step is not { } d) return time;
        return _engine.TensorAdd(time, StepOut!.Forward(_engine.Swish(StepIn.Forward(Sinusoidal(d)))));
    }

    private Tensor<T> Mish(Tensor<T> x) => _engine.TensorMultiply(x, _engine.Tanh(_engine.Softplus(x)));

    // [T, C] through a [1, C, T] convolution.
    private Tensor<T> ConvRows(NormedConv1DLayer<T> conv, Tensor<T> rows)
    {
        int t = rows.Shape[0], ci = rows.Shape[1];
        var y = conv.Forward(_engine.Reshape(_engine.TensorTranspose(rows), new[] { 1, ci, t }));
        return _engine.TensorTranspose(_engine.Reshape(y, new[] { y.Shape[1], t }));
    }

    /// <summary>The conditioning features <c>[T, text_dim]</c> (zeros when dropped) with positions, through the
    /// ConvNeXt V2 blocks (<c>TextEmbedding</c>).</summary>
    public Tensor<T> TextEmbedding(Tensor<T> text, bool drop)
    {
        var c = Configuration;
        int t = text.Shape[0];
        if (drop) text = new Tensor<T>(new[] { t, c.TextDim });
        if (c.ConvLayers == 0) return text;
        var positions = new Tensor<T>(new[] { t, c.TextDim });
        for (int i = 0; i < t; i++)
        {
            int row = Math.Min(i, TextPositionRows - 1);
            for (int j = 0; j < c.TextDim; j++) positions[i, j] = NumOps.FromDouble(_textPositions[row * c.TextDim + j]);
        }
        var x = _engine.TensorAdd(text, positions);
        foreach (var b in TextBlocks)
        {
            var y = b.Norm.Forward(ConvRows(b.Depthwise, x));
            y = AiDotNet.Helpers.ActivationHelper.ExactGelu(b.Expand.Forward(y), _engine);
            // GRN: γ·(y·N) + β + y with N the time L2 norm over its channel mean.
            int width = y.Shape[1];
            var g = _engine.TensorSqrt(_engine.ReduceSum(_engine.TensorMultiply(y, y), new[] { 0 }, keepDims: true));          // [1, width]
            var mean = _engine.TensorMultiplyScalar(_engine.ReduceSum(g, new[] { 1 }, keepDims: true), NumOps.FromDouble(1.0 / width));
            var n = _engine.TensorDivide(g, _engine.TensorBroadcastTo(_engine.TensorAddScalar(mean, NumOps.FromDouble(1e-6)), new[] { 1, width }));
            var shape = new[] { t, width };
            var gamma = _engine.TensorBroadcastTo(_engine.Reshape(IndexTtsOps.Value(b.GrnGamma), new[] { 1, width }), shape);
            var beta = _engine.TensorBroadcastTo(_engine.Reshape(IndexTtsOps.Value(b.GrnBeta), new[] { 1, width }), shape);
            y = _engine.TensorAdd(_engine.TensorAdd(_engine.TensorMultiply(gamma, _engine.TensorMultiply(y, _engine.TensorBroadcastTo(n, shape))), beta), y);
            x = _engine.TensorAdd(x, b.Project.Forward(y));
        }
        return x;
    }

    /// <summary>
    /// The velocity <c>[T, mel]</c> of the noisy mel <paramref name="x"/> <c>[T, mel]</c> given the prompt mel
    /// <paramref name="cond"/> (zeros past the prompt; zeroed when <paramref name="dropAudioCond"/>), the embedded
    /// features <paramref name="textEmbedding"/> <c>[T, text_dim]</c> and the time embedding <c>[1, dim]</c>.
    /// </summary>
    public Tensor<T> Forward(Tensor<T> x, Tensor<T> cond, Tensor<T> textEmbedding, Tensor<T> time, bool dropAudioCond, bool training = false,
        Random? random = null)
    {
        var c = Configuration;
        int frames = x.Shape[0], inner = c.Heads * c.HeadDim;
        if (dropAudioCond) cond = new Tensor<T>(cond.Shape.ToArray());
        var h = InputProjection.Forward(_engine.TensorConcatenate(new[] { x, cond, textEmbedding }, 1));
        h = _engine.TensorAdd(Mish(ConvRows(PositionConv2, Mish(ConvRows(PositionConv1, h)))), h);
        var silu = _engine.Swish(time);
        foreach (var b in Blocks)
        {
            var m = b.Modulation.Forward(silu);                                                          // [1, 6 dim]
            Tensor<T> Chunk(int i) => _engine.TensorNarrow(m, 1, i * c.Dim, c.Dim);
            Tensor<T> Rows(Tensor<T> row) => _engine.TensorBroadcastTo(row, new[] { frames, c.Dim });
            var normed = _engine.TensorAdd(_engine.TensorMultiply(AiDotNet.TextToSpeech.FlowDiffusion.FlowMatchingTts.PlainLayerNorm(_engine, h),
                Rows(_engine.TensorAddScalar(Chunk(1), NumOps.One))), Rows(Chunk(0)));
            // Attention: q, k, v over all heads; the rotary table (head width) rotates the first head's dimensions only.
            var q = b.Query.Forward(normed);
            var k = b.Key.Forward(normed);
            var v = b.Value.Forward(normed);
            q = RotateFirstHead(q);
            k = RotateFirstHead(k);
            Tensor<T> Split(Tensor<T> y) => _engine.TensorPermute(_engine.Reshape(y, new[] { frames, c.Heads, c.HeadDim }), new[] { 1, 0, 2 });
            var scores = _engine.TensorMultiplyScalar(_engine.BatchMatMul(Split(q), _engine.TensorPermute(Split(k), new[] { 0, 2, 1 })),
                NumOps.FromDouble(1.0 / Math.Sqrt(c.HeadDim)));
            var attended = _engine.Reshape(_engine.TensorPermute(_engine.BatchMatMul(_engine.Softmax(scores, axis: 2), Split(v)), new[] { 1, 0, 2 }),
                new[] { frames, inner });
            var attention = Drop(b.Out.Forward(attended), training, random);
            h = _engine.TensorAdd(h, _engine.TensorMultiply(Rows(Chunk(2)), attention));
            normed = _engine.TensorAdd(_engine.TensorMultiply(AiDotNet.TextToSpeech.FlowDiffusion.FlowMatchingTts.PlainLayerNorm(_engine, h),
                Rows(_engine.TensorAddScalar(Chunk(4), NumOps.One))), Rows(Chunk(3)));
            var ff = b.FfOut.Forward(Drop(GeluTanh(b.FfIn.Forward(normed)), training, random));
            h = _engine.TensorAdd(h, _engine.TensorMultiply(Rows(Chunk(5)), ff));
        }
        var final = NormOut.Forward(silu);
        var scale = _engine.TensorBroadcastTo(_engine.TensorAddScalar(_engine.TensorNarrow(final, 1, 0, c.Dim), NumOps.One), new[] { frames, c.Dim });
        var shift = _engine.TensorBroadcastTo(_engine.TensorNarrow(final, 1, c.Dim, c.Dim), new[] { frames, c.Dim });
        h = _engine.TensorAdd(_engine.TensorMultiply(AiDotNet.TextToSpeech.FlowDiffusion.FlowMatchingTts.PlainLayerNorm(_engine, h), scale), shift);
        return ProjectionOut.Forward(h);
    }

    // GELU with the tanh approximation (FeedForward's approximate="tanh").
    private Tensor<T> GeluTanh(Tensor<T> x) => IndexTtsOps.NewGelu(_engine, x);

    // apply_rotary_pos_emb over the first head_dim columns of [T, inner].
    private Tensor<T> RotateFirstHead(Tensor<T> x)
    {
        int d = Configuration.HeadDim, frames = x.Shape[0], width = x.Shape[1];
        var head = AiDotNet.TextToSpeech.FlowDiffusion.FlowMatchingTts.Rotary(_engine, _engine.TensorNarrow(x, 1, 0, d));
        return width == d ? head : _engine.TensorConcatenate(new[] { head, _engine.TensorNarrow(x, 1, d, width - d) }, 1);
    }

    /// <summary>Loads <c>cfm.estimator.*</c> (names after the prefix).</summary>
    public void Load(Func<string, double[]> read, string prefix)
    {
        var c = Configuration;
        int inner = c.Heads * c.HeadDim;
        void Linear(DenseLayer<T> layer, string name, int inputs, int outputs) => MeloOps.LoadLinear(_engine, read, layer, inputs, outputs, prefix + name);
        Linear(TimeIn, "time_embed.time_mlp.0", 256, c.Dim);
        Linear(TimeOut, "time_embed.time_mlp.2", c.Dim, c.Dim);
        if (StepIn is not null)
        {
            Linear(StepIn, "d_embed.time_mlp.0", 256, c.Dim);
            Linear(StepOut!, "d_embed.time_mlp.2", c.Dim, c.Dim);
        }
        for (int i = 0; i < TextBlocks.Count; i++)
        {
            var b = TextBlocks[i];
            string n = $"text_embed.text_blocks.{i}";
            MeloOps.LoadConv(read, b.Depthwise, prefix + n + ".dwconv");
            MeloOps.LoadNorm(_engine, read, b.Norm, prefix + n + ".norm", "weight", "bias");
            Linear(b.Expand, n + ".pwconv1", c.TextDim, 2 * c.TextDim);
            TorchParameters.Linear(b.GrnGamma, 1, 2 * c.TextDim, read(prefix + n + ".grn.gamma"));
            TorchParameters.Linear(b.GrnBeta, 1, 2 * c.TextDim, read(prefix + n + ".grn.beta"));
            Linear(b.Project, n + ".pwconv2", 2 * c.TextDim, c.TextDim);
        }
        Linear(InputProjection, "input_embed.proj", 2 * c.MelDim + c.TextDim, c.Dim);
        MeloOps.LoadConv(read, PositionConv1, prefix + "input_embed.conv_pos_embed.conv1d.0");
        MeloOps.LoadConv(read, PositionConv2, prefix + "input_embed.conv_pos_embed.conv1d.2");
        for (int i = 0; i < Blocks.Count; i++)
        {
            var b = Blocks[i];
            string n = $"transformer_blocks.{i}";
            Linear(b.Modulation, n + ".attn_norm.linear", c.Dim, 6 * c.Dim);
            Linear(b.Query, n + ".attn.to_q", c.Dim, inner);
            Linear(b.Key, n + ".attn.to_k", c.Dim, inner);
            Linear(b.Value, n + ".attn.to_v", c.Dim, inner);
            Linear(b.Out, n + ".attn.to_out.0", inner, c.Dim);
            Linear(b.FfIn, n + ".ff.ff.0.0", c.Dim, c.FfMult * c.Dim);
            Linear(b.FfOut, n + ".ff.ff.2", c.FfMult * c.Dim, c.Dim);
        }
        Linear(NormOut, "norm_out.linear", c.Dim, 2 * c.Dim);
        Linear(ProjectionOut, "proj_out", c.Dim, c.MelDim);
    }
}

/// <summary>
/// GPT-SoVITS's conditional flow matching for v3, v4 and V5 (<c>module/models.py</c> <c>CFM</c> and
/// <c>models_v5.py</c> <c>CFMV5</c>): Euler integration of the DiT's velocity from noise at t = 0 with the prompt's frames
/// held at zero, optionally with classifier-free guidance; and the training loss with the shortcut models' bootstrap
/// (v3/v4) or plain flow matching (V5).
/// </summary>
internal sealed class GptSoVitsCfm<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    public GptSoVitsCfm(IEngine engine, GptSoVitsDiT<T> estimator, bool v5)
    {
        _engine = engine;
        Estimator = estimator;
        IsV5 = v5;
    }

    public GptSoVitsDiT<T> Estimator { get; }

    /// <summary>V5: no step embedding, CFG keeps the text, the noise temperature fixed at 0.875.</summary>
    public bool IsV5 { get; }

    /// <summary>V5's noise temperature (V5_TEMPERATURE).</summary>
    public const double V5Temperature = 0.875;

    /// <summary>
    /// <c>inference(mu, x_lens, prompt, n_timesteps, temperature, inference_cfg_rate)</c> for one sequence: the mel
    /// <c>[T, mel]</c> from the features <paramref name="mu"/> <c>[T, text_dim]</c> and the prompt mel <c>[P, mel]</c>,
    /// the starting noise <paramref name="noise"/> <c>[T, mel]</c> (standard normal; scaled by the temperature).
    /// </summary>
    public Tensor<T> Inference(Tensor<T> mu, Tensor<T> prompt, int steps, double temperature, double cfgRate, Tensor<T> noise)
    {
        if (IsV5) temperature = V5Temperature;
        int frames = mu.Shape[0], mel = Estimator.Configuration.MelDim, promptLength = prompt.Shape[0];
        var x = _engine.TensorMultiplyScalar(noise, NumOps.FromDouble(temperature));
        var promptX = new Tensor<T>(new[] { frames, mel });
        for (int i = 0; i < promptLength && i < frames; i++)
            for (int j = 0; j < mel; j++) promptX[i, j] = prompt[i, j];
        x = ZeroPrompt(x, promptLength);
        double t = 0, d = 1.0 / steps;
        var text = Estimator.TextEmbedding(mu, drop: false);
        Tensor<T>? negativeText = null;
        for (int j = 0; j < steps; j++)
        {
            var time = Estimator.TimeEmbedding(IsV5 ? j * d : t, IsV5 ? null : d);
            var v = Estimator.Forward(x, promptX, text, time, dropAudioCond: false);
            if (cfgRate > 1e-5)
            {
                // CFG drops the prompt, and for v3/v4 the features too (cfg_drop_text).
                var negText = IsV5 ? text : negativeText ??= Estimator.TextEmbedding(mu, drop: true);
                var negative = Estimator.Forward(x, promptX, negText, time, dropAudioCond: true);
                v = _engine.TensorAdd(v, _engine.TensorMultiplyScalar(_engine.TensorSubtract(v, negative), NumOps.FromDouble(cfgRate)));
            }
            x = _engine.TensorAdd(x, _engine.TensorMultiplyScalar(v, NumOps.FromDouble(d)));
            t += d;
            x = ZeroPrompt(x, promptLength);
        }
        return x;
    }

    private Tensor<T> ZeroPrompt(Tensor<T> x, int promptLength)
    {
        int frames = x.Shape[0];
        if (promptLength <= 0) return x;
        if (promptLength >= frames) return new Tensor<T>(x.Shape.ToArray());
        return _engine.TensorConcatenate(new[] { new Tensor<T>(new[] { promptLength, x.Shape[1] }), _engine.TensorNarrow(x, 0, promptLength, frames - promptLength) }, 0);
    }

    /// <summary>
    /// <c>forward(x1, x_lens, prompt_lens, mu)</c> for one sequence: the flow-matching loss of the mel <c>[T, mel]</c>
    /// given the features <c>[T, text_dim]</c>, over the frames after the prompt. v3/v4 bootstrap the shortcut model with
    /// probability 0.3 (a step of 1/2^k, k ∈ [2, 8), the target the mean of two detached half-step velocities, the step
    /// embedding 2d); V5 does not.
    /// </summary>
    public Tensor<T> Loss(Tensor<T> mel, Tensor<T> mu, int promptLength, Random random, bool training = true)
    {
        int frames = mel.Shape[0], channels = mel.Shape[1];
        double t = random.NextDouble();
        var x0 = new Tensor<T>(new[] { frames, channels });
        for (int i = 0; i < x0.Length; i++) x0.SetFlat(i, NumOps.FromDouble(IndexTtsOps.Gaussian(random)));
        int? exponent = !IsV5 && random.NextDouble() < 0.3 ? random.Next(2, 8) : null;
        return Loss(mel, mu, promptLength, t, x0, exponent, training, random);
    }

    /// <summary>The loss with its draws given: the time <paramref name="t"/>, the noise <paramref name="x0"/>
    /// <c>[T, mel]</c>, and the shortcut step's exponent k (step 1/2^k) when the bootstrap fires, else null.</summary>
    public Tensor<T> Loss(Tensor<T> mel, Tensor<T> mu, int promptLength, double t, Tensor<T> x0, int? exponent, bool training, Random random)
    {
        int frames = mel.Shape[0], channels = mel.Shape[1];
        var vt = _engine.TensorSubtract(mel, x0);
        var xt = _engine.TensorAdd(x0, _engine.TensorMultiplyScalar(vt, NumOps.FromDouble(t)));
        var prompt = new Tensor<T>(new[] { frames, channels });
        for (int i = 0; i < promptLength; i++)
            for (int j = 0; j < channels; j++) prompt[i, j] = mel[i, j];
        xt = ZeroPrompt(xt, promptLength);
        var text = Estimator.TextEmbedding(mu, drop: false);
        double? dt = IsV5 ? null : 0.0;
        if (!IsV5 && exponent is { } k)
        {
            double d = 1.0 / Math.Pow(2, k);
            double dInput = d < 1e-2 ? 0 : d;
            var v1 = MeloOps.Detach(Estimator.Forward(xt, prompt, text, Estimator.TimeEmbedding(t, dInput), false, training, random));
            var middle = _engine.TensorAdd(xt, _engine.TensorMultiplyScalar(v1, NumOps.FromDouble(d)));
            var v2 = MeloOps.Detach(Estimator.Forward(middle, prompt, text, Estimator.TimeEmbedding(t + d, dInput), false, training, random));
            vt = _engine.TensorMultiplyScalar(_engine.TensorAdd(v1, v2), NumOps.FromDouble(0.5));
            dt = 2 * d;
        }
        var predicted = Estimator.Forward(xt, prompt, text, Estimator.TimeEmbedding(t, dt), false, training, random);
        int count = frames - promptLength;
        if (count <= 0) throw new ArgumentException("The prompt covers every frame; nothing is left to learn.", nameof(promptLength));
        var difference = _engine.TensorSubtract(_engine.TensorNarrow(predicted, 0, promptLength, count), _engine.TensorNarrow(vt, 0, promptLength, count));
        return _engine.TensorMultiplyScalar(_engine.ReduceSum(_engine.TensorMultiply(difference, difference), null),
            NumOps.FromDouble(1.0 / ((double)count * channels)));
    }
}

/// <summary>
/// The WaveNet encoder of v3/v4/V5's mel path (<c>models.Encoder</c>: 1×1 pre-convolution, <c>modules.WN</c> with the
/// style condition, 1×1 projection), with its frame mask: frames at or past <c>length</c> are zeroed at every layer.
/// </summary>
internal sealed class GptSoVitsMaskedEncoder<T>
{
    private readonly IEngine _engine;
    private readonly int _hidden;
    private readonly List<NormedConv1DLayer<T>> _in = new(), _resSkip = new();

    public GptSoVitsMaskedEncoder(IEngine engine, List<LayerBase<T>> layers, int inChannels, int outChannels, int hidden, int kernel,
        int dilationRate, int count, int gin)
    {
        _engine = engine;
        _hidden = hidden;
        Pre = MeloOps.Conv(layers, inChannels, hidden, 1);
        Condition = MeloOps.WeightNormConv(layers, gin, 2 * hidden * count, 1);
        for (int i = 0; i < count; i++)
        {
            int dilation = (int)Math.Pow(dilationRate, i);
            _in.Add(MeloOps.WeightNormConv(layers, hidden, 2 * hidden, kernel, dilation, (kernel * dilation - dilation) / 2));
            _resSkip.Add(MeloOps.WeightNormConv(layers, hidden, i < count - 1 ? 2 * hidden : hidden, 1));
        }
        Projection = MeloOps.Conv(layers, hidden, outChannels, 1);
    }

    public NormedConv1DLayer<T> Pre { get; }
    public NormedConv1DLayer<T> Condition { get; }
    public NormedConv1DLayer<T> Projection { get; }

    private Tensor<T> Mask(Tensor<T> x, int length)
    {
        int t = x.Shape[2];
        if (length >= t) return x;
        if (length <= 0) return new Tensor<T>(x.Shape.ToArray());
        return _engine.TensorConcatenate(new[] { _engine.TensorNarrow(x, 2, 0, length), new Tensor<T>(new[] { 1, x.Shape[1], t - length }) }, 2);
    }

    /// <summary>The encoding <c>[1, out, T]</c> of <paramref name="x"/> <c>[1, in, T]</c> with frames past
    /// <paramref name="length"/> masked, conditioned on <paramref name="ge"/> <c>[1, gin, 1]</c> (detached).</summary>
    public Tensor<T> Forward(Tensor<T> x, int length, Tensor<T> ge)
    {
        int t = x.Shape[2];
        x = Mask(Pre.Forward(x), length);
        var conditions = Condition.Forward(MeloOps.Detach(ge));
        Tensor<T>? output = null;
        for (int i = 0; i < _in.Count; i++)
        {
            var xin = _engine.TensorAdd(_in[i].Forward(x), MeloOps.OverTime(_engine, MeloOps.Channels(_engine, conditions, i * 2 * _hidden, 2 * _hidden), t));
            var acts = _engine.TensorMultiply(_engine.Tanh(MeloOps.Channels(_engine, xin, 0, _hidden)), _engine.Sigmoid(MeloOps.Channels(_engine, xin, _hidden, _hidden)));
            var rs = _resSkip[i].Forward(acts);
            if (i < _in.Count - 1)
            {
                x = Mask(_engine.TensorAdd(x, MeloOps.Channels(_engine, rs, 0, _hidden)), length);
                var skip = MeloOps.Channels(_engine, rs, _hidden, _hidden);
                output = output is null ? skip : _engine.TensorAdd(output, skip);
            }
            else output = output is null ? rs : _engine.TensorAdd(output, rs);
        }
        return Mask(Projection.Forward(Mask(output!, length)), length);
    }

    public void Load(Func<string, double[]> read, string prefix)
    {
        MeloOps.LoadConv(read, Pre, prefix + "pre");
        MeloOps.LoadConv(read, Condition, prefix + "enc.cond_layer");
        for (int i = 0; i < _in.Count; i++)
        {
            MeloOps.LoadConv(read, _in[i], $"{prefix}enc.in_layers.{i}");
            MeloOps.LoadConv(read, _resSkip[i], $"{prefix}enc.res_skip_layers.{i}");
        }
        MeloOps.LoadConv(read, Projection, prefix + "proj");
    }
}
