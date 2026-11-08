using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.Latest;
using AiDotNet.TextToSpeech.MultiModal;

namespace AiDotNet.TextToSpeech.FlowDiffusion;

/// <summary>The frozen SpeechT5 text encoder used by the English DiTTo model.</summary>
internal sealed class DiTToTextEncoder<T>
{
    private readonly SpeechT5Configuration _configuration;

    public DiTToTextEncoder(IEngine engine, List<LayerBase<T>> layers, DiTToTTSOptions options, Random random)
    {
        _configuration = new SpeechT5Configuration
        {
            Hidden = options.TextHiddenDim, FeedForward = options.TextFeedForwardDim, Heads = options.TextHeads,
            EncoderLayers = options.TextEncoderLayers, DecoderLayers = 0, TextVocabulary = options.TextVocabulary,
            PadId = options.TextPadId, EosId = options.TextEosId, MaxTextPositions = options.MaxTextLength,
            Dropout = options.DropoutRate, AttentionDropout = options.DropoutRate,
            ActivationDropout = options.DropoutRate, PositionalDropout = options.DropoutRate
        };
        Embedding = IndexTtsOps.Embedding(layers, random, _configuration.TextVocabulary, _configuration.Hidden,
            Math.Pow(_configuration.Hidden, -0.5));
        Positions = new SpeechT5ScaledPositions<T>(engine, layers, _configuration.Hidden);
        Encoder = new SpeechT5Encoder<T>(engine, layers, _configuration, random);
    }

    public TiedEmbeddingLayer<T> Embedding { get; }
    public SpeechT5ScaledPositions<T> Positions { get; }
    public SpeechT5Encoder<T> Encoder { get; }

    public Tensor<T> Forward(IReadOnlyList<int> tokens)
    {
        var x = Positions.Forward(Embedding.Forward(IndexTtsOps.Ids<T>(tokens)), 0, 0, false, null);
        return Encoder.Forward(x);
    }

    /// <summary>Loads <c>microsoft/speecht5_tts</c>'s Hugging Face text-encoder names.</summary>
    public void LoadHuggingFace(Func<string, int[], double[]> read)
    {
        double[] Read(string native, int[] shape)
        {
            string name = SpeechT5Core<T>.HuggingFaceName(native)
                ?? throw new InvalidDataException($"SpeechT5 has no Hugging Face mapping for '{native}'.");
            return read(name, shape);
        }
        var c = _configuration;
        Embedding.LoadTable(Read("text_encoder_prenet.encoder_prenet.0.weight", new[] { c.TextVocabulary, c.Hidden }));
        Positions.Load(Read, "text_encoder_prenet.encoder_prenet.1.alpha");
        Encoder.Load(Read, "encoder.");
    }
}

/// <summary>
/// CLaM-TTS's probabilistic RVQ: 32 greedy residual codebooks and the published monotonically decreasing
/// codebook scale <c>exp(m) · sum(softmax(s)[d..D])</c>, with every raw codeword L2-normalized.
/// </summary>
internal sealed class DiTToProbabilisticRvq<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    public DiTToProbabilisticRvq(IEngine engine, List<LayerBase<T>> layers, Random random, int depth, int vocabulary, int dimension)
    {
        _engine = engine; Depth = depth; Vocabulary = vocabulary; Dimension = dimension;
        for (int i = 0; i < depth; i++) Codebooks.Add(IndexTtsOps.Embedding(layers, random, vocabulary, dimension, 1.0));
        MaximumScaleLogit = IndexTtsOps.Values<T>(layers, new[] { 0.0 });
        ScaleLogits = IndexTtsOps.Values<T>(layers, new double[depth]);
        LogVariance = IndexTtsOps.Values<T>(layers, new[] { 0.0 });
    }

    public int Depth { get; }
    public int Vocabulary { get; }
    public int Dimension { get; }
    public List<TiedEmbeddingLayer<T>> Codebooks { get; } = new();
    public BiasFreeLinearLayer<T> MaximumScaleLogit { get; }
    public BiasFreeLinearLayer<T> ScaleLogits { get; }
    public BiasFreeLinearLayer<T> LogVariance { get; }

    /// <summary>CLaM-TTS equations 3–6: detached variational posteriors train every codeword,
    /// while straight-through reconstruction and commitment train the encoder.</summary>
    internal (Tensor<T> Quantized, Tensor<T> VariationalLoss, Tensor<T> CommitmentLoss) TrainingForward(Tensor<T> latents)
    {
        Tensor<T> Detached(Tensor<T> x) => new(x.Shape.ToArray(), new Vector<T>(x.ToArray()));
        int frames = latents.Shape[0];
        var probabilities = _engine.Softmax(IndexTtsOps.Value(ScaleLogits), 1);
        var tail = new Tensor<T>(new[] { Depth, Depth });
        for (int source = 0; source < Depth; source++)
            for (int target = 0; target <= source; target++) tail[source, target] = NumOps.One;
        var scales = _engine.TensorMultiply(_engine.TensorMatMul(probabilities, tail),
            _engine.TensorTile(_engine.TensorExp(IndexTtsOps.Value(MaximumScaleLogit)), new[] { 1, Depth }));
        var tables = new List<Tensor<T>>();
        var selected = new List<Tensor<T>>();
        var residual = latents.ToArray().Select(NumOps.ToDouble).ToArray();
        var sum = new Tensor<T>(new[] { frames, Dimension });
        for (int d = 0; d < Depth; d++)
        {
            var raw = Codebooks[d].Forward(IndexTtsOps.Ids<T>(Enumerable.Range(0, Vocabulary).ToArray()));
            var norm = _engine.TensorClampMin(_engine.TensorSqrt(_engine.ReduceSum(
                _engine.TensorMultiply(raw, raw), new[] { 1 }, true)), NumOps.FromDouble(1e-12));
            var scale = _engine.TensorSlice(scales, new[] { 0, d }, new[] { 1, 1 });
            var table = _engine.TensorMultiply(_engine.TensorDivide(raw, _engine.TensorTile(norm, new[] { 1, Dimension })),
                _engine.TensorTile(scale, new[] { Vocabulary, Dimension }));
            tables.Add(table);
            var ids = new int[frames];
            for (int f = 0; f < frames; f++)
            {
                double best = double.PositiveInfinity;
                for (int k = 0; k < Vocabulary; k++)
                {
                    double distance = 0;
                    for (int j = 0; j < Dimension; j++)
                    {
                        double delta = residual[f * Dimension + j] - NumOps.ToDouble(table[k, j]);
                        distance += delta * delta;
                    }
                    if (distance < best) { best = distance; ids[f] = k; }
                }
                for (int j = 0; j < Dimension; j++) residual[f * Dimension + j] -= NumOps.ToDouble(table[ids[f], j]);
            }
            var chosen = _engine.TensorIndexSelect(table, new Tensor<int>(new[] { frames }, new Vector<int>(ids)), 0);
            selected.Add(chosen);
            sum = _engine.TensorAdd(sum, chosen);
        }
        var logVariance = IndexTtsOps.Value(LogVariance);
        var variance = _engine.TensorExp(logVariance);
        Tensor<T>? loss = null;
        for (int d = 0; d < Depth; d++)
        {
            var target = Detached(_engine.TensorSubtract(latents, _engine.TensorSubtract(sum, selected[d])));
            var delta = _engine.TensorSubtract(
                _engine.TensorTile(_engine.Reshape(target, new[] { frames, 1, Dimension }), new[] { 1, Vocabulary, 1 }),
                _engine.TensorTile(_engine.Reshape(tables[d], new[] { 1, Vocabulary, Dimension }), new[] { frames, 1, 1 }));
            var distance = _engine.ReduceSum(_engine.TensorMultiply(delta, delta), new[] { 2 }, false);
            var nll = _engine.TensorAdd(_engine.TensorMultiplyScalar(_engine.TensorDivide(distance,
                _engine.TensorTile(variance, new[] { frames, Vocabulary })), NumOps.FromDouble(0.5)),
                _engine.TensorTile(_engine.TensorMultiplyScalar(_engine.TensorAddScalar(logVariance,
                    NumOps.FromDouble(Math.Log(2 * Math.PI))), NumOps.FromDouble(0.5 * Dimension)), new[] { frames, Vocabulary }));
            var posterior = Detached(_engine.Softmax(_engine.TensorNegate(nll), 1));
            var stageLoss = _engine.TensorMultiplyScalar(_engine.ReduceSum(
                _engine.TensorMultiply(posterior, nll), new[] { 0, 1 }, true), NumOps.FromDouble(1.0 / frames));
            loss = loss is null ? stageLoss : _engine.TensorAdd(loss, stageLoss);
        }
        var commitment = _engine.TensorSubtract(latents, Detached(sum));
        commitment = _engine.ReduceMean(_engine.TensorMultiply(commitment, commitment), new[] { 0, 1 }, true);
        return (_engine.TensorAdd(latents, Detached(_engine.TensorSubtract(sum, latents))), loss!, commitment);
    }

    internal double[] Scales()
    {
        double maximum = Math.Exp(NumOps.ToDouble(IndexTtsOps.Value(MaximumScaleLogit)[0, 0]));
        var raw = IndexTtsOps.Value(ScaleLogits);
        var probabilities = new double[Depth];
        double peak = Enumerable.Range(0, Depth).Max(i => NumOps.ToDouble(raw[0, i])), sum = 0;
        for (int i = 0; i < Depth; i++) sum += probabilities[i] = Math.Exp(NumOps.ToDouble(raw[0, i]) - peak);
        for (int i = 0; i < Depth; i++) probabilities[i] /= sum;
        var scales = new double[Depth];
        double tail = 0;
        for (int i = Depth - 1; i >= 0; i--) { tail += probabilities[i]; scales[i] = maximum * tail; }
        return scales;
    }

    /// <summary>Greedy residual quantization. The codec is frozen in DiTTo, so the nearest-code search is intentionally non-differentiable.</summary>
    public Tensor<T> Quantize(Tensor<T> latents)
    {
        int frames = latents.Shape[0];
        var residual = latents.ToArray().Select(NumOps.ToDouble).ToArray();
        var output = new double[residual.Length];
        var ids = Enumerable.Range(0, Vocabulary).ToArray();
        var scales = Scales();
        for (int d = 0; d < Depth; d++)
        {
            var table = Codebooks[d].Forward(IndexTtsOps.Ids<T>(ids));
            var normalized = new double[Vocabulary * Dimension];
            for (int k = 0; k < Vocabulary; k++)
            {
                double norm = 0;
                for (int j = 0; j < Dimension; j++) { double v = NumOps.ToDouble(table[k, j]); norm += v * v; }
                norm = Math.Max(Math.Sqrt(norm), 1e-12);
                for (int j = 0; j < Dimension; j++) normalized[k * Dimension + j] = scales[d] * NumOps.ToDouble(table[k, j]) / norm;
            }
            for (int f = 0; f < frames; f++)
            {
                int best = 0; double bestDistance = double.PositiveInfinity;
                for (int k = 0; k < Vocabulary; k++)
                {
                    double distance = 0;
                    for (int j = 0; j < Dimension; j++) { double z = residual[f * Dimension + j] - normalized[k * Dimension + j]; distance += z * z; }
                    if (distance < bestDistance) { bestDistance = distance; best = k; }
                }
                for (int j = 0; j < Dimension; j++)
                {
                    double value = normalized[best * Dimension + j];
                    output[f * Dimension + j] += value; residual[f * Dimension + j] -= value;
                }
            }
        }
        return new Tensor<T>(new[] { frames, Dimension }, new Vector<T>(output.Select(NumOps.FromDouble).ToArray()));
    }

    public void Load(Func<string, int[], double[]> read, string prefix)
    {
        for (int i = 0; i < Codebooks.Count; i++) Codebooks[i].LoadTable(read($"{prefix}codebooks.{i}.weight", new[] { Vocabulary, Dimension }));
        TorchParameters.Linear(MaximumScaleLogit, 1, 1, read(prefix + "maximum_scale_logit", new[] { 1 }));
        TorchParameters.Linear(ScaleLogits, 1, Depth, read(prefix + "scale_logits", new[] { Depth }));
        TorchParameters.Linear(LogVariance, 1, 1, read(prefix + "log_variance", new[] { 1 }));
    }
}

/// <summary>
/// The pretrained CLaM-TTS Mel-VAE named by DiTTo: 256 base channels, multipliers [1,1,2,2], three 2x temporal
/// downsamples, a 512-wide latent and probabilistic RVQ. CLaM-TTS does not publish its residual-block micro-layout;
/// this implementation uses the standard two-convolution VQGAN residual block while preserving every published size.
/// </summary>
internal sealed class DiTToMelVae<T>
{
    private readonly IEngine _engine;
    private readonly Dictionary<NormedConv1DLayer<T>, (int TrimOrPad, bool Transpose)> _causalConvolutions = new();
    internal sealed record Residual(NormedConv1DLayer<T> First, NormedConv1DLayer<T> Second, NormedConv1DLayer<T>? Shortcut);

    public DiTToMelVae(IEngine engine, List<LayerBase<T>> layers, DiTToTTSOptions options, Random random)
    {
        _engine = engine; Options = options;
        NormedConv1DLayer<T> Conv(int input, int output, int kernel, int stride = 1, bool transpose = false)
        {
            // CLaM-TTS section 4.3 requires causal convolutions. A strided output ends at the
            // end of its input group; transposed convolutions trim only the trailing overlap.
            var layer = new NormedConv1DLayer<T>(input, output, kernel, stride, 1, 1, 0, transpose, ConvolutionNormalization.None);
            _causalConvolutions.Add(layer, (kernel - stride, transpose));
            layers.Add(layer); return layer;
        }
        Residual Block(int input, int output) => new(Conv(input, output, 3), Conv(output, output, 3), input == output ? null : Conv(input, output, 1));

        int[] widths = options.MelVaeMultipliers.Select(m => m * options.MelVaeChannels).ToArray();
        if (options.MelVaeDownsampleFactor < 1 || widths.Length < 1 || widths.Any(width => width < 1)
            || (widths.Length == 1 && options.MelVaeDownsampleFactor != 1))
            throw new ArgumentException("The mel VAE needs positive widths and a positive downsample factor, with at least two widths for downsampling.", nameof(options));
        var strides = new int[widths.Length - 1];
        int remainingFactor = options.MelVaeDownsampleFactor;
        for (int i = 0; i < strides.Length; i++)
        {
            strides[i] = i == strides.Length - 1 ? remainingFactor : remainingFactor % 2 == 0 ? 2 : 1;
            remainingFactor /= strides[i];
        }
        EncoderInput = Conv(options.MelChannels, widths[0], 7);
        int current = widths[0];
        for (int i = 0; i < widths.Length; i++)
        {
            EncoderBlocks.Add(Block(current, widths[i])); current = widths[i];
            if (i + 1 < widths.Length) { EncoderDown.Add(Conv(current, widths[i + 1], checked(2 * strides[i]), strides[i])); current = widths[i + 1]; }
        }
        ToLatent = Conv(current, options.LatentDim, 3);
        FromLatent = Conv(options.LatentDim, current, 3);
        for (int i = widths.Length - 1; i >= 0; i--)
        {
            DecoderBlocks.Add(Block(current, widths[i])); current = widths[i];
            if (i > 0) { DecoderUp.Add(Conv(current, widths[i - 1], checked(2 * strides[i - 1]), strides[i - 1], true)); current = widths[i - 1]; }
        }
        DecoderOutput = Conv(current, options.MelChannels, 7);
        ConvNeXtDepthwise = new NormedConv1DLayer<T>(options.MelChannels, options.MelChannels, 7, 1, 1,
            options.MelChannels, 0, false, ConvolutionNormalization.None); layers.Add(ConvNeXtDepthwise);
        _causalConvolutions.Add(ConvNeXtDepthwise, (6, false));
        ConvNeXtExpand = IndexTtsOps.Linear(engine, layers, random, options.MelChannels, 4 * options.MelChannels);
        ConvNeXtProject = IndexTtsOps.Linear(engine, layers, random, 4 * options.MelChannels, options.MelChannels);
        Quantizer = new DiTToProbabilisticRvq<T>(engine, layers, random, options.RvqDepth, options.RvqVocabulary, options.LatentDim);
    }

    public DiTToTTSOptions Options { get; }
    public NormedConv1DLayer<T> EncoderInput { get; }
    public List<Residual> EncoderBlocks { get; } = new();
    public List<NormedConv1DLayer<T>> EncoderDown { get; } = new();
    public NormedConv1DLayer<T> ToLatent { get; }
    public NormedConv1DLayer<T> FromLatent { get; }
    public List<Residual> DecoderBlocks { get; } = new();
    public List<NormedConv1DLayer<T>> DecoderUp { get; } = new();
    public NormedConv1DLayer<T> DecoderOutput { get; }
    public NormedConv1DLayer<T> ConvNeXtDepthwise { get; }
    public DenseLayer<T> ConvNeXtExpand { get; }
    public DenseLayer<T> ConvNeXtProject { get; }
    public DiTToProbabilisticRvq<T> Quantizer { get; }

    private Tensor<T> Channels(Tensor<T> rows) => _engine.TensorPermute(_engine.Reshape(rows, new[] { 1, rows.Shape[0], rows.Shape[1] }), new[] { 0, 2, 1 });
    private Tensor<T> Rows(Tensor<T> channels) => _engine.Reshape(_engine.TensorPermute(channels, new[] { 0, 2, 1 }), new[] { channels.Shape[2], channels.Shape[1] });
    private Tensor<T> Causal(NormedConv1DLayer<T> layer, Tensor<T> input)
    {
        var (amount, transpose) = _causalConvolutions[layer];
        if (!transpose && amount > 0)
            input = _engine.TensorConcatenate(new[] { new Tensor<T>(new[] { input.Shape[0], input.Shape[1], amount }), input }, 2);
        var output = layer.Forward(input);
        return transpose && amount > 0
            ? _engine.TensorSlice(output, new[] { 0, 0, 0 }, new[] { output.Shape[0], output.Shape[1], output.Shape[2] - amount })
            : output;
    }
    private Tensor<T> ResidualForward(Residual block, Tensor<T> x)
    {
        var h = Causal(block.Second, _engine.Swish(Causal(block.First, _engine.Swish(x))));
        return _engine.TensorAdd(h, block.Shortcut is null ? x : Causal(block.Shortcut, x));
    }

    public Tensor<T> Encode(Tensor<T> mel)
    {
        var x = Causal(EncoderInput, Channels(mel));
        for (int i = 0; i < EncoderBlocks.Count; i++) { x = ResidualForward(EncoderBlocks[i], x); if (i < EncoderDown.Count) x = Causal(EncoderDown[i], x); }
        return Rows(Causal(ToLatent, _engine.Swish(x)));
    }

    public Tensor<T> Decode(Tensor<T> latents, bool quantize = true)
    {
        var z = quantize ? Quantizer.Quantize(latents) : latents;
        var x = Causal(FromLatent, Channels(z));
        for (int i = 0; i < DecoderBlocks.Count; i++) { x = ResidualForward(DecoderBlocks[i], x); if (i < DecoderUp.Count) x = Causal(DecoderUp[i], x); }
        x = Causal(DecoderOutput, _engine.Swish(x));
        var residual = Rows(x);
        var refined = Rows(Causal(ConvNeXtDepthwise, Channels(residual)));
        refined = ConvNeXtProject.Forward(ActivationHelper.ExactGelu(ConvNeXtExpand.Forward(refined), _engine));
        return _engine.TensorAdd(residual, refined);
    }

    /// <summary>Loads AiDotNet's documented conversion of the private CLaM-TTS Mel-VAE checkpoint.</summary>
    public void Load(Func<string, int[], double[]> read, string prefix)
    {
        void Conv(NormedConv1DLayer<T> layer, string name) => IndexTtsOps.LoadConv(read, layer, prefix + name, layer.OutputChannels);
        void Block(Residual b, string name)
        {
            Conv(b.First, name + ".first"); Conv(b.Second, name + ".second"); if (b.Shortcut is not null) Conv(b.Shortcut, name + ".shortcut");
        }
        Conv(EncoderInput, "encoder.input");
        for (int i = 0; i < EncoderBlocks.Count; i++) { Block(EncoderBlocks[i], $"encoder.blocks.{i}"); if (i < EncoderDown.Count) Conv(EncoderDown[i], $"encoder.down.{i}"); }
        Conv(ToLatent, "encoder.to_latent"); Conv(FromLatent, "decoder.from_latent");
        for (int i = 0; i < DecoderBlocks.Count; i++) { Block(DecoderBlocks[i], $"decoder.blocks.{i}"); if (i < DecoderUp.Count) Conv(DecoderUp[i], $"decoder.up.{i}"); }
        Conv(DecoderOutput, "decoder.output"); Conv(ConvNeXtDepthwise, "decoder.convnext.depthwise");
        IndexTtsOps.LoadLinear(_engine, read, ConvNeXtExpand, prefix + "decoder.convnext.expand", Options.MelChannels, 4 * Options.MelChannels);
        IndexTtsOps.LoadLinear(_engine, read, ConvNeXtProject, prefix + "decoder.convnext.project", 4 * Options.MelChannels, Options.MelChannels);
        Quantizer.Load(read, prefix + "rvq.");
    }
}

/// <summary>
/// DiTTo's diffusion Transformer. One shared 9d modulation is conditioned on the sinusoidal timestep and mean-pooled
/// text; every block adds its own learned 9d vector and applies shift/scale/gate independently to rotary self-attention,
/// text cross-attention and the GEGLU MLP. A two-layer long-skip block joins the input and final states.
/// </summary>
internal sealed class DiTToDiffusionTransformer<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    internal sealed record Block(
        BiasFreeLinearLayer<T> ModulationOffset,
        DenseLayer<T> SelfQuery, DenseLayer<T> SelfKey, DenseLayer<T> SelfValue, DenseLayer<T> SelfOutput,
        DenseLayer<T> CrossQuery, DenseLayer<T> CrossKey, DenseLayer<T> CrossValue, DenseLayer<T> CrossOutput,
        DenseLayer<T> GegluIn, DenseLayer<T> GegluOut);

    public DiTToDiffusionTransformer(IEngine engine, List<LayerBase<T>> layers, DiTToTTSOptions options, Random random)
    {
        _engine = engine; Options = options;
        int d = options.DiTHiddenDim, ff = d * options.DiTFeedForwardMultiplier;
        if (options.DiTHeads <= 0 || d % options.DiTHeads != 0 || (d / options.DiTHeads) % 2 != 0)
            throw new ArgumentException("DiT requires an even rotary dimension within every attention head.", nameof(options));
        Input = IndexTtsOps.Linear(engine, layers, random, options.LatentDim + 1, d);
        TextKeys = IndexTtsOps.Linear(engine, layers, random, options.TextHiddenDim, d);
        TimeIn = IndexTtsOps.Linear(engine, layers, random, 256, d);
        TimeOut = IndexTtsOps.Linear(engine, layers, random, d, d);
        TextConditionIn = IndexTtsOps.Linear(engine, layers, random, options.TextHiddenDim, d);
        TextConditionOut = IndexTtsOps.Linear(engine, layers, random, d, d);
        SharedModulation = IndexTtsOps.Linear(engine, layers, random, d, 9 * d);
        NullText = IndexTtsOps.Values<T>(layers, new double[options.TextHiddenDim]);
        for (int i = 0; i < options.DiTLayers; i++)
            Blocks.Add(new Block(IndexTtsOps.Values<T>(layers, new double[9 * d]),
                IndexTtsOps.Linear(engine, layers, random, d, d), IndexTtsOps.Linear(engine, layers, random, d, d),
                IndexTtsOps.Linear(engine, layers, random, d, d), IndexTtsOps.Linear(engine, layers, random, d, d),
                IndexTtsOps.Linear(engine, layers, random, d, d), IndexTtsOps.Linear(engine, layers, random, d, d),
                IndexTtsOps.Linear(engine, layers, random, d, d), IndexTtsOps.Linear(engine, layers, random, d, d),
                IndexTtsOps.Linear(engine, layers, random, d, 2 * ff), IndexTtsOps.Linear(engine, layers, random, ff, d)));
        SkipIn = IndexTtsOps.Linear(engine, layers, random, 2 * d, d);
        SkipOut = IndexTtsOps.Linear(engine, layers, random, d, d);
        Output = IndexTtsOps.Linear(engine, layers, random, d, options.LatentDim);
    }

    public DiTToTTSOptions Options { get; }
    public DenseLayer<T> Input { get; }
    public DenseLayer<T> TextKeys { get; }
    public DenseLayer<T> TimeIn { get; }
    public DenseLayer<T> TimeOut { get; }
    public DenseLayer<T> TextConditionIn { get; }
    public DenseLayer<T> TextConditionOut { get; }
    public DenseLayer<T> SharedModulation { get; }
    public BiasFreeLinearLayer<T> NullText { get; }
    public List<Block> Blocks { get; } = new();
    public DenseLayer<T> SkipIn { get; }
    public DenseLayer<T> SkipOut { get; }
    public DenseLayer<T> Output { get; }
    public int ModulationWidth => 9 * Options.DiTHiddenDim;

    private Tensor<T> Mean(Tensor<T> x)
    {
        int length = x.Shape[0], width = x.Shape[1];
        return _engine.ReduceMean(x, new[] { 0 }, keepDims: true).Shape[1] == width
            ? _engine.ReduceMean(x, new[] { 0 }, keepDims: true)
            : _engine.Reshape(_engine.ReduceMean(x, new[] { 0 }, keepDims: false), new[] { 1, width });
    }

    internal static Tensor<TValue> TimestepEmbedding<TValue>(double time, int dimension)
    {
        var ops = MathHelper.GetNumericOperations<TValue>();
        var result = new Tensor<TValue>(new[] { 1, dimension });
        int half = dimension / 2;
        for (int i = 0; i < half; i++)
        {
            double frequency = Math.Exp(-Math.Log(10000.0) * i / Math.Max(half, 1));
            result[0, i] = ops.FromDouble(Math.Cos(time * 1000 * frequency));
            if (i + half < dimension) result[0, i + half] = ops.FromDouble(Math.Sin(time * 1000 * frequency));
        }
        return result;
    }

    private Tensor<T> Attention(Tensor<T> q, Tensor<T> k, Tensor<T> v, bool rotary)
    {
        int heads = Options.DiTHeads, dk = Options.DiTHiddenDim / heads;
        if (rotary)
        {
            Tensor<T> RotateHeads(Tensor<T> projected)
            {
                var rotated = new Tensor<T>[heads];
                for (int head = 0; head < heads; head++)
                    rotated[head] = FlowMatchingTts.Rotary(_engine,
                        _engine.TensorSlice(projected, new[] { 0, head * dk }, new[] { projected.Shape[0], dk }));
                return _engine.TensorConcatenate(rotated, 1);
            }
            q = RotateHeads(q);
            k = RotateHeads(k);
        }
        return IndexTtsOps.MergeHeads(_engine, IndexTtsOps.Attend(_engine,
            IndexTtsOps.SplitHeads(_engine, q, heads), IndexTtsOps.SplitHeads(_engine, k, heads),
            IndexTtsOps.SplitHeads(_engine, v, heads), 1.0 / Math.Sqrt(dk), null));
    }

    private Tensor<T> Modulate(Tensor<T> x, Tensor<T> modulation, int shiftChunk, int scaleChunk)
    {
        int d = Options.DiTHiddenDim;
        var shift = FlowMatchingTts.Chunk(_engine, modulation, shiftChunk, d);
        var scale = FlowMatchingTts.Chunk(_engine, modulation, scaleChunk, d);
        return FlowMatchingTts.Modulate(_engine, FlowMatchingTts.PlainLayerNorm(_engine, x), scale, shift);
    }

    private Tensor<T> Gate(Tensor<T> update, Tensor<T> modulation, int chunk)
    {
        var gate = FlowMatchingTts.Chunk(_engine, modulation, chunk, Options.DiTHiddenDim);
        return _engine.TensorMultiply(update, _engine.TensorTile(gate, new[] { update.Shape[0], 1 }));
    }

    /// <param name="noisy">Noisy speech latent <c>[L, 512]</c>.</param>
    /// <param name="mask">One binary column: one on the region the model must generate.</param>
    /// <param name="dropText">Classifier-free unconditional branch: only the learned null token remains and pooled text is zero.</param>
    public Tensor<T> Forward(Tensor<T> noisy, Tensor<T> mask, Tensor<T> text, double time, bool dropText)
    {
        int d = Options.DiTHiddenDim;
        var input = _engine.TensorConcatenate(new[] { noisy, mask }, 1);
        var x = Input.Forward(input); var first = x;
        Tensor<T> memory;
        Tensor<T> pooled;
        var nullRow = IndexTtsOps.Value(NullText);
        if (dropText) { memory = TextKeys.Forward(nullRow); pooled = new Tensor<T>(new[] { 1, Options.TextHiddenDim }); }
        else { memory = TextKeys.Forward(_engine.TensorConcatenate(new[] { nullRow, text }, 0)); pooled = Mean(text); }
        var condition = _engine.TensorAdd(
            TimeOut.Forward(_engine.Swish(TimeIn.Forward(TimestepEmbedding<T>(time, 256)))),
            TextConditionOut.Forward(_engine.Swish(TextConditionIn.Forward(pooled))));
        var shared = SharedModulation.Forward(_engine.Swish(condition));
        foreach (var block in Blocks)
        {
            var modulation = _engine.TensorAdd(shared, IndexTtsOps.Value(block.ModulationOffset));
            var h = Modulate(x, modulation, 0, 1);
            var self = block.SelfOutput.Forward(Attention(block.SelfQuery.Forward(h), block.SelfKey.Forward(h), block.SelfValue.Forward(h), true));
            x = _engine.TensorAdd(x, Gate(self, modulation, 2));
            h = Modulate(x, modulation, 3, 4);
            var cross = block.CrossOutput.Forward(Attention(block.CrossQuery.Forward(h), block.CrossKey.Forward(memory), block.CrossValue.Forward(memory), false));
            x = _engine.TensorAdd(x, Gate(cross, modulation, 5));
            h = Modulate(x, modulation, 6, 7);
            var geglu = block.GegluIn.Forward(h);
            int ff = geglu.Shape[1] / 2;
            var value = _engine.TensorNarrow(geglu, 1, 0, ff);
            var gate = _engine.TensorNarrow(geglu, 1, ff, ff);
            var feedForward = block.GegluOut.Forward(_engine.TensorMultiply(value, ActivationHelper.ExactGelu(gate, _engine)));
            x = _engine.TensorAdd(x, Gate(feedForward, modulation, 8));
        }
        x = SkipOut.Forward(_engine.Swish(SkipIn.Forward(_engine.TensorConcatenate(new[] { first, x }, 1))));
        return Output.Forward(FlowMatchingTts.PlainLayerNorm(_engine, x));
    }

    public void Load(Func<string, int[], double[]> read, string prefix)
    {
        int d = Options.DiTHiddenDim, ff = d * Options.DiTFeedForwardMultiplier;
        void Linear(DenseLayer<T> l, string n, int i, int o) => IndexTtsOps.LoadLinear(_engine, read, l, prefix + n, i, o);
        Linear(Input, "input", Options.LatentDim + 1, d); Linear(TextKeys, "text_keys", Options.TextHiddenDim, d);
        Linear(TimeIn, "time.0", 256, d); Linear(TimeOut, "time.2", d, d);
        Linear(TextConditionIn, "text_condition.0", Options.TextHiddenDim, d); Linear(TextConditionOut, "text_condition.2", d, d);
        Linear(SharedModulation, "shared_adaln", d, 9 * d);
        TorchParameters.Linear(NullText, 1, Options.TextHiddenDim, read(prefix + "null_text", new[] { Options.TextHiddenDim }));
        for (int i = 0; i < Blocks.Count; i++)
        {
            var b = Blocks[i]; string p = $"blocks.{i}.";
            TorchParameters.Linear(b.ModulationOffset, 1, 9 * d, read(prefix + p + "adaln_offset", new[] { 9 * d }));
            Linear(b.SelfQuery, p + "self.q", d, d); Linear(b.SelfKey, p + "self.k", d, d); Linear(b.SelfValue, p + "self.v", d, d); Linear(b.SelfOutput, p + "self.out", d, d);
            Linear(b.CrossQuery, p + "cross.q", d, d); Linear(b.CrossKey, p + "cross.k", d, d); Linear(b.CrossValue, p + "cross.v", d, d); Linear(b.CrossOutput, p + "cross.out", d, d);
            Linear(b.GegluIn, p + "geglu.in", d, 2 * ff); Linear(b.GegluOut, p + "geglu.out", ff, d);
        }
        Linear(SkipIn, "long_skip.0", 2 * d, d); Linear(SkipOut, "long_skip.2", d, d); Linear(Output, "output", d, Options.LatentDim);
    }
}

/// <summary>
/// DiTTo's independently trained speech-length predictor: a causal Transformer decoder over prompt Mel-VAE tokens,
/// cross-attending to bidirectional text states. At prompt position p the target is the total remaining length N-p;
/// inference uses the top-20 expected value (the deterministic setting used for the paper's evaluations).
/// </summary>
internal sealed class DiTToLengthPredictor<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    internal sealed record Block(
        DenseLayer<T> SelfQuery, DenseLayer<T> SelfKey, DenseLayer<T> SelfValue, DenseLayer<T> SelfOutput,
        DenseLayer<T> CrossQuery, DenseLayer<T> CrossKey, DenseLayer<T> CrossValue, DenseLayer<T> CrossOutput,
        LayerNormalizationLayer<T> SelfNorm, LayerNormalizationLayer<T> CrossNorm, LayerNormalizationLayer<T> FeedForwardNorm,
        DenseLayer<T> FeedForward1, DenseLayer<T> FeedForward2);

    public DiTToLengthPredictor(IEngine engine, List<LayerBase<T>> layers, DiTToTTSOptions options, Random random)
    {
        _engine = engine; Options = options;
        int d = options.LengthHiddenDim;
        if (d % options.LengthHeads != 0) throw new ArgumentException("Length-predictor width must divide by its heads.", nameof(options));
        AudioInput = IndexTtsOps.Linear(engine, layers, random, options.LatentDim, d);
        TextInput = IndexTtsOps.Linear(engine, layers, random, options.TextHiddenDim, d);
        Bos = IndexTtsOps.Values<T>(layers, new double[d]);
        for (int i = 0; i < options.LengthLayers; i++)
            Blocks.Add(new Block(
                IndexTtsOps.Linear(engine, layers, random, d, d), IndexTtsOps.Linear(engine, layers, random, d, d),
                IndexTtsOps.Linear(engine, layers, random, d, d), IndexTtsOps.Linear(engine, layers, random, d, d),
                IndexTtsOps.Linear(engine, layers, random, d, d), IndexTtsOps.Linear(engine, layers, random, d, d),
                IndexTtsOps.Linear(engine, layers, random, d, d), IndexTtsOps.Linear(engine, layers, random, d, d),
                IndexTtsOps.LayerNorm<T>(layers, d), IndexTtsOps.LayerNorm<T>(layers, d), IndexTtsOps.LayerNorm<T>(layers, d),
                IndexTtsOps.Linear(engine, layers, random, d, options.LengthFeedForwardDim),
                IndexTtsOps.Linear(engine, layers, random, options.LengthFeedForwardDim, d)));
        Classifier = IndexTtsOps.Linear(engine, layers, random, d, options.MaxLatentLength + 1);
    }

    public DiTToTTSOptions Options { get; }
    public DenseLayer<T> AudioInput { get; }
    public DenseLayer<T> TextInput { get; }
    public BiasFreeLinearLayer<T> Bos { get; }
    public List<Block> Blocks { get; } = new();
    public DenseLayer<T> Classifier { get; }

    private Tensor<T> Attention(Tensor<T> q, Tensor<T> k, Tensor<T> v, bool causal)
    {
        int heads = Options.LengthHeads, dk = Options.LengthHiddenDim / heads;
        Tensor<T>? bias = null;
        if (causal)
        {
            bias = new Tensor<T>(new[] { q.Shape[0], k.Shape[0] });
            for (int i = 0; i < q.Shape[0]; i++) for (int j = i + 1; j < k.Shape[0]; j++) bias[i, j] = NumOps.FromDouble(double.NegativeInfinity);
        }
        return IndexTtsOps.MergeHeads(_engine, IndexTtsOps.Attend(_engine, IndexTtsOps.SplitHeads(_engine, q, heads),
            IndexTtsOps.SplitHeads(_engine, k, heads), IndexTtsOps.SplitHeads(_engine, v, heads), 1.0 / Math.Sqrt(dk), bias));
    }

    /// <summary>All remaining-length logits, beginning with the no-prompt BOS position.</summary>
    public Tensor<T> Forward(Tensor<T> text, Tensor<T>? prompt)
    {
        var bos = IndexTtsOps.Value(Bos);
        var x = prompt is null || prompt.Shape[0] == 0 ? bos : _engine.TensorConcatenate(new[] { bos, AudioInput.Forward(prompt) }, 0);
        var memory = TextInput.Forward(text);
        foreach (var b in Blocks)
        {
            var h = b.SelfNorm.Forward(x);
            x = _engine.TensorAdd(x, b.SelfOutput.Forward(Attention(b.SelfQuery.Forward(h), b.SelfKey.Forward(h), b.SelfValue.Forward(h), true)));
            h = b.CrossNorm.Forward(x);
            x = _engine.TensorAdd(x, b.CrossOutput.Forward(Attention(b.CrossQuery.Forward(h), b.CrossKey.Forward(memory), b.CrossValue.Forward(memory), false)));
            h = b.FeedForwardNorm.Forward(x);
            x = _engine.TensorAdd(x, b.FeedForward2.Forward(ActivationHelper.ExactGelu(b.FeedForward1.Forward(h), _engine)));
        }
        return Classifier.Forward(x);
    }

    /// <summary>The expected remaining frames under the final position's top-k distribution.</summary>
    public int ExpectedRemaining(Tensor<T> text, Tensor<T>? prompt)
    {
        var logits = Forward(text, prompt); int row = logits.Shape[0] - 1;
        int[] top = Enumerable.Range(0, logits.Shape[1]).OrderByDescending(i => NumOps.ToDouble(logits[row, i]))
            .Take(Math.Min(Options.LengthTopK, logits.Shape[1])).ToArray();
        double peak = top.Max(i => NumOps.ToDouble(logits[row, i])), total = 0, weighted = 0;
        foreach (int i in top) { double p = Math.Exp(NumOps.ToDouble(logits[row, i]) - peak); total += p; weighted += i * p; }
        return MathPolyfill.Clamp((int)Math.Round(weighted / Math.Max(total, 1e-12)), 0, Options.MaxLatentLength);
    }

    /// <summary>Labels for a prompt prefix: total, total-1, ... as specified in §3.2.</summary>
    public static int[] RemainingLabels(int totalLength, int promptLength) =>
        Enumerable.Range(0, promptLength + 1).Select(position => Math.Max(0, totalLength - position)).ToArray();

    public Tensor<T> CrossEntropy(Tensor<T> logits, IReadOnlyList<int> labels)
    {
        if (logits.Shape[0] != labels.Count) throw new ArgumentException("One remaining-length label is required per decoder position.", nameof(labels));
        Tensor<T>? total = null;
        for (int row = 0; row < labels.Count; row++)
        {
            var one = _engine.TensorSlice(logits, new[] { row, 0 }, new[] { 1, logits.Shape[1] });
            var logProbabilities = _engine.TensorLogSoftmax(one, axis: 1);
            var selected = _engine.TensorSlice(logProbabilities, new[] { 0, MathPolyfill.Clamp(labels[row], 0, logits.Shape[1] - 1) }, new[] { 1, 1 });
            total = total is null ? selected : _engine.TensorAdd(total, selected);
        }
        return _engine.TensorMultiplyScalar(total!, NumOps.FromDouble(-1.0 / labels.Count));
    }

    public void Load(Func<string, int[], double[]> read, string prefix)
    {
        int d = Options.LengthHiddenDim;
        void Linear(DenseLayer<T> l, string n, int input, int output) => IndexTtsOps.LoadLinear(_engine, read, l, prefix + n, input, output);
        Linear(AudioInput, "audio_input", Options.LatentDim, d); Linear(TextInput, "text_input", Options.TextHiddenDim, d);
        TorchParameters.Linear(Bos, 1, d, read(prefix + "bos", new[] { d }));
        for (int i = 0; i < Blocks.Count; i++)
        {
            var b = Blocks[i]; string p = $"blocks.{i}.";
            Linear(b.SelfQuery, p + "self.q", d, d); Linear(b.SelfKey, p + "self.k", d, d); Linear(b.SelfValue, p + "self.v", d, d); Linear(b.SelfOutput, p + "self.out", d, d);
            Linear(b.CrossQuery, p + "cross.q", d, d); Linear(b.CrossKey, p + "cross.k", d, d); Linear(b.CrossValue, p + "cross.v", d, d); Linear(b.CrossOutput, p + "cross.out", d, d);
            IndexTtsOps.LoadNorm(_engine, read, b.SelfNorm, prefix + p + "self_norm", d);
            IndexTtsOps.LoadNorm(_engine, read, b.CrossNorm, prefix + p + "cross_norm", d);
            IndexTtsOps.LoadNorm(_engine, read, b.FeedForwardNorm, prefix + p + "ff_norm", d);
            Linear(b.FeedForward1, p + "ff.0", d, Options.LengthFeedForwardDim); Linear(b.FeedForward2, p + "ff.2", Options.LengthFeedForwardDim, d);
        }
        Linear(Classifier, "classifier", d, Options.MaxLatentLength + 1);
    }
}
