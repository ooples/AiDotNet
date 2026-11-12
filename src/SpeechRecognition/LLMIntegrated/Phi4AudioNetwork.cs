using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.Tokenization.Algorithms;

namespace AiDotNet.SpeechRecognition.LLMIntegrated;

/// <summary>The geometry of a <see cref="Phi4AudioNetwork{T}"/>: the cascades conformer encoder, the projector and the
/// Phi-4-Mini decoder (the release's <c>audio_processor</c> and text configuration).</summary>
internal sealed record Phi4AudioConfiguration
{
    public int SampleRate { get; init; } = 16000;
    public int Mels { get; init; } = 80;
    public int EncoderWidth { get; init; } = 1024;
    public int EncoderLayers { get; init; } = 24;
    public int EncoderHeads { get; init; } = 16;
    public int EncoderFeedForward { get; init; } = 1536;
    public int SubsamplingChannels { get; init; } = 1024;
    public int ConvKernel { get; init; } = 3;
    public int RelativeBiasMaxDistance { get; init; } = 500;
    public int AttentionChunk { get; init; } = 500;
    public LlamaConfiguration Llm { get; init; } = new();
}

/// <summary>
/// Phi-4-multimodal's speech path (Abdin et al. 2025 §2.2, and the release's <c>speech_conformer_encoder.py</c>,
/// <c>modeling_phi4mm.py</c> and <c>processing_phi4mm.py</c>): SpeechLib log filterbanks; NeMo's depthwise-striding
/// convolutions to an 80 ms rate; 24 causal-convolution conformer blocks with T5 relative attention biases; the two-layer
/// speech projector; and Phi-4-Mini, which reads the projected frames in place of the audio placeholder tokens.
/// </summary>
internal sealed class Phi4AudioNetwork<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private const string TokenizerResource = "AiDotNet.Tokenization.phi4-tokenizer.json.gz";
    private const int FftSize = 512, WindowLength = 400, HopLength = 160;
    // The audio placeholder (_AUDIO_SPECIAL_TOKEN_ID).
    private const string AudioToken = "<|endoftext11|>";
    private static readonly Lazy<ByteLevelBpeTokenizer> PhiText = new(LoadTokenizer);
    private readonly IEngine _engine;
    private readonly double[,] _melBank;

    internal sealed record FeedForward(LayerNormalizationLayer<T> Norm, DenseLayer<T> Up, DenseLayer<T> Down);

    internal sealed record Block(FeedForward FeedForwardIn, LayerNormalizationLayer<T> AttentionNorm, DenseLayer<T> Query, DenseLayer<T> Key,
        DenseLayer<T> Value, DenseLayer<T> Output, LayerNormalizationLayer<T> ConvNorm, DenseLayer<T> GluConv, BiasFreeLinearLayer<T> GluBias,
        BiasFreeLinearLayer<T> DepthwiseKernel, BiasFreeLinearLayer<T> DepthwiseBias, DenseLayer<T> PointwiseConv, DenseLayer<T> ExtPointwiseConv,
        FeedForward FeedForwardOut, LayerNormalizationLayer<T> Norm);

    public Phi4AudioNetwork(IEngine engine, Phi4AudioConfiguration configuration, List<LayerBase<T>> layers, Random random)
    {
        _engine = engine;
        var c = configuration;
        if (c.EncoderWidth % c.EncoderHeads != 0) throw new ArgumentException("The encoder width must divide by its heads.", nameof(configuration));
        Configuration = c;
        _melBank = SpeechLibMel(c.SampleRate, FftSize, c.Mels, 0.0, 7690.0);
        GlobalMean = new double[c.Mels];
        GlobalInverseStd = Enumerable.Repeat(1.0, c.Mels).ToArray();
        TLayer Own<TLayer>(List<LayerBase<T>> owner, TLayer layer) where TLayer : LayerBase<T> { owner.Add(layer); return layer; }
        double Uniform(double bound) => bound * (2 * random.NextDouble() - 1);
        // PyTorch's default initialization: uniform in ±1/√fan-in for weights and biases.
        DenseLayer<T> Linear(List<LayerBase<T>> owner, int inputs, int outputs)
        {
            var layer = Own(owner, new DenseLayer<T>(outputs, (IActivationFunction<T>)new IdentityActivation<T>()));
            double bound = 1.0 / Math.Sqrt(inputs);
            TorchParameters.Linear(engine, layer, inputs, outputs, Enumerable.Range(0, inputs * outputs).Select(_ => Uniform(bound)).ToArray(),
                Enumerable.Range(0, outputs).Select(_ => Uniform(bound)).ToArray());
            return layer;
        }
        BiasFreeLinearLayer<T> Table(List<LayerBase<T>> owner, int rows, int columns, Func<double> draw)
        {
            var layer = Own(owner, new BiasFreeLinearLayer<T>(rows, columns));
            var values = new double[rows * columns];
            for (int r = 0; r < rows; r++) for (int k = 0; k < columns; k++) values[k * rows + r] = draw();
            TorchParameters.Linear(layer, rows, columns, values);
            return layer;
        }
        double Normal() => Math.Sqrt(-2.0 * Math.Log(1.0 - random.NextDouble())) * Math.Cos(2.0 * Math.PI * random.NextDouble());
        LayerNormalizationLayer<T> Norm(List<LayerBase<T>> owner, int width) => Own(owner, new LayerNormalizationLayer<T>(width, 1e-5));

        int d = c.EncoderWidth, channels = c.SubsamplingChannels;
        FirstConv = Own(EncoderLayers, new Conv2DLayer<T>(1, channels, 3, 3, 2, 2, 1, 1));
        DepthwiseStages.Add(Own(EncoderLayers, new DepthwiseSeparableConvolutionalLayer<T>(channels, 3, stride: 2, padding: 1, activation: (IActivationFunction<T>)new ReLUActivation<T>())));
        DepthwiseStages.Add(Own(EncoderLayers, new DepthwiseSeparableConvolutionalLayer<T>(channels, 3, stride: 2, padding: 1, activation: (IActivationFunction<T>)new ReLUActivation<T>())));
        SubsampledBins = (((c.Mels - 1) / 2 + 1 - 1) / 2 + 1 - 1) / 2 + 1;
        SubsamplingOut = Linear(EncoderLayers, channels * SubsampledBins, d);
        // T5 relative biases without bucketing: one row per offset in [−max, max − 1]; nn.Embedding's N(0, 1).
        RelativeBias = Table(EncoderLayers, 2 * c.RelativeBiasMaxDistance, c.EncoderHeads, Normal);
        FeedForward EncoderFeedForward() => new(Norm(EncoderLayers, d), Linear(EncoderLayers, d, 2 * c.EncoderFeedForward), Linear(EncoderLayers, c.EncoderFeedForward, d));
        double kernelBound = 1.0 / Math.Sqrt(c.ConvKernel);
        for (int i = 0; i < c.EncoderLayers; i++)
        {
            var ffIn = EncoderFeedForward();
            var attentionNorm = Norm(EncoderLayers, d);
            var query = Linear(EncoderLayers, d, d);
            var key = Linear(EncoderLayers, d, d);
            var value = Linear(EncoderLayers, d, d);
            var output = Linear(EncoderLayers, d, d);
            var convNorm = Norm(EncoderLayers, d);
            var glu = Linear(EncoderLayers, d, 2 * d);
            // GLUPointWiseConv's b1 and b2 start at zero.
            var gluBias = Table(EncoderLayers, 2, d, () => 0.0);
            var depthwise = Table(EncoderLayers, c.ConvKernel, d, () => Uniform(kernelBound));
            var depthwiseBias = Table(EncoderLayers, 1, d, () => Uniform(kernelBound));
            var pointwise = Linear(EncoderLayers, d, d);
            var extPointwise = Linear(EncoderLayers, d, d);
            var ffOut = EncoderFeedForward();
            Blocks.Add(new Block(ffIn, attentionNorm, query, key, value, output, convNorm, glu, gluBias, depthwise, depthwiseBias, pointwise,
                extPointwise, ffOut, Norm(EncoderLayers, d)));
        }
        Projection1 = Linear(ProjectorLayers, d, c.Llm.Hidden);
        Projection2 = Linear(ProjectorLayers, c.Llm.Hidden, c.Llm.Hidden);
        Llm = new LlamaNetwork<T>(engine, LlmLayers, c.Llm);
        Llm.InitializeLikeReference(random);
        layers.AddRange(EncoderLayers);
        layers.AddRange(ProjectorLayers);
        layers.AddRange(LlmLayers);
    }

    public Phi4AudioConfiguration Configuration { get; }
    /// <summary>The encoder's input normalization (<c>encoder_embedding.global_mean</c>), zero until loaded.</summary>
    public double[] GlobalMean { get; }
    /// <summary>The encoder's input normalization (<c>encoder_embedding.global_invstd</c>), one until loaded.</summary>
    public double[] GlobalInverseStd { get; }
    public Conv2DLayer<T> FirstConv { get; }
    public List<DepthwiseSeparableConvolutionalLayer<T>> DepthwiseStages { get; } = new();
    public int SubsampledBins { get; }
    public DenseLayer<T> SubsamplingOut { get; }
    public BiasFreeLinearLayer<T> RelativeBias { get; }
    public List<Block> Blocks { get; } = new();
    public DenseLayer<T> Projection1 { get; }
    public DenseLayer<T> Projection2 { get; }
    public LlamaNetwork<T> Llm { get; }
    public List<LayerBase<T>> EncoderLayers { get; } = new();
    public List<LayerBase<T>> ProjectorLayers { get; } = new();
    public List<LayerBase<T>> LlmLayers { get; } = new();

    /// <summary>Whether the vocabulary covers the release tokenizer's, so the prompt is chat text; a smaller vocabulary
    /// has no prompt text, and its last token closes the answer.</summary>
    public bool UsesPhiVocabulary => Configuration.Llm.VocabSize >= PhiText.Value.VocabularySize;

    public ByteLevelBpeTokenizer Tokenizer => PhiText.Value;

    /// <summary>The tokens that close the answer: <c>&lt;|end|&gt;&lt;|endoftext|&gt;</c> (the release's
    /// <c>_EOS_TOKEN</c>), or the last token without Phi's vocabulary.</summary>
    public int[] EndTokens => UsesPhiVocabulary
        ? new[] { Tokenizer.TokenId("<|end|>"), Tokenizer.TokenId("<|endoftext|>") }
        : new[] { Configuration.Llm.VocabSize - 1 };

    private static ByteLevelBpeTokenizer LoadTokenizer()
    {
        using var stream = typeof(Phi4AudioNetwork<T>).Assembly.GetManifestResourceStream(TokenizerResource)
            ?? throw new InvalidDataException($"The embedded resource {TokenizerResource} is missing.");
        using var gzip = new System.IO.Compression.GZipStream(stream, System.IO.Compression.CompressionMode.Decompress);
        using var reader = new StreamReader(gzip, System.Text.Encoding.UTF8);
        return ByteLevelBpeTokenizer.FromJson(reader.ReadToEnd());
    }

    /// <summary>SpeechLib's mel filterbank (<c>speechlib_mel</c>): triangles in the 1127 · ln(1 + f / 700) mel space over
    /// FFT bins [round(fmin) + 1, round(fmax)), <c>[mels, bins]</c>.</summary>
    internal static double[,] SpeechLibMel(int sampleRate, int nFft, int mels, double fMin, double fMax)
    {
        int bins = nFft / 2 + 1;
        double Mel(double f) => 1127.0 * Math.Log(1.0 + f / 700.0);
        double BinMel(int bin) => 1127.0 * Math.Log(1.0 + bin * (double)sampleRate / (nFft * 700.0));
        int ToBin(double f) => (int)(f * nFft / sampleRate + 0.5);
        int low = ToBin(fMin) + 1, high = Math.Max(ToBin(fMax), low);
        double melLow = Mel(fMin), melHigh = Mel(fMax), spacing = (melHigh - melLow) / (mels + 1);
        var matrix = new double[mels, bins];
        for (int m = 0; m < mels; m++)
        {
            double left = melLow + m * spacing, center = melLow + (m + 1) * spacing, right = melLow + (m + 2) * spacing;
            for (int bin = low; bin < high; bin++)
            {
                double mel = BinMel(bin);
                if (left < mel && mel < right) matrix[m, bin] = (float)(1.0 - Math.Abs(center - mel) / spacing);
            }
        }
        return matrix;
    }

    /// <summary>
    /// The encoder's input <c>[frames, mels]</c> of a 16 kHz waveform (the release's <c>Phi4MMAudioFeatureExtractor</c>):
    /// ⌊(samples − 400) / 160⌋ + 1 frames of 400 samples, each pre-emphasized by 0.97 within the frame and scaled by
    /// 32,768, a symmetric Hamming window, a 512-point FFT, the power through SpeechLib's filterbank (fmax 7,690 Hz),
    /// clipped below at 1 and logged.
    /// </summary>
    public double[,] Features(IReadOnlyList<double> waveform)
    {
        int frames = (waveform.Count - WindowLength) / HopLength + 1, bins = FftSize / 2 + 1, mels = Configuration.Mels;
        if (frames <= 0) throw new ArgumentException($"The recording is shorter than {WindowLength} samples.", nameof(waveform));
        var window = new double[WindowLength];
        for (int n = 0; n < WindowLength; n++) window[n] = 0.54 - 0.46 * Math.Cos(2 * Math.PI * n / (WindowLength - 1));
        var cos = new double[bins * WindowLength];
        var sin = new double[bins * WindowLength];
        for (int k = 0; k < bins; k++)
            for (int n = 0; n < WindowLength; n++)
            {
                double angle = 2 * Math.PI * ((long)k * n % FftSize) / FftSize;
                cos[k * WindowLength + n] = Math.Cos(angle);
                sin[k * WindowLength + n] = Math.Sin(angle);
            }
        var features = new double[frames, mels];
        var frame = new double[WindowLength];
        var power = new double[bins];
        for (int f = 0; f < frames; f++)
        {
            int start = f * HopLength;
            for (int n = 0; n < WindowLength; n++)
            {
                double previous = waveform[start + Math.Max(0, n - 1)];
                frame[n] = (waveform[start + n] - 0.97 * previous) * 32768.0 * window[n];
            }
            for (int k = 0; k < bins; k++)
            {
                double re = 0, im = 0;
                for (int n = 0; n < WindowLength; n++)
                {
                    re += frame[n] * cos[k * WindowLength + n];
                    im -= frame[n] * sin[k * WindowLength + n];
                }
                power[k] = re * re + im * im;
            }
            for (int m = 0; m < mels; m++)
            {
                double sum = 0;
                for (int k = 0; k < bins; k++) sum += _melBank[m, k] * power[k];
                features[f, m] = Math.Log(Math.Max(sum, 1.0));
            }
        }
        return features;
    }

    private Tensor<T> Run(FeedForward ff, Tensor<T> x)
    {
        // GLULinear with Swish: the first half times Swish of the second.
        var up = ff.Up.Forward(ff.Norm.Forward(x));
        int rows = up.Shape[0], inner = up.Shape[1] / 2;
        var gated = _engine.TensorMultiply(_engine.TensorSlice(up, new[] { 0, 0 }, new[] { rows, inner }),
            _engine.Swish(_engine.TensorSlice(up, new[] { 0, inner }, new[] { rows, inner })));
        return ff.Down.Forward(gated);
    }

    /// <summary>
    /// The encoder's output <c>[⌈frames / 8⌉, width]</c> of features <c>[frames, mels]</c>: the global mean-variance
    /// normalization, the subsampling convolutions, then the conformer blocks over chunks of at most 500 frames (the
    /// release's unfolding for its absolute-position limit) with full attention within each.
    /// </summary>
    public Tensor<T> Encode(double[,] features)
    {
        var c = Configuration;
        int frames = features.GetLength(0), mels = features.GetLength(1);
        var mel = new Tensor<T>(new[] { 1, 1, frames, mels });
        for (int f = 0; f < frames; f++)
            for (int m = 0; m < mels; m++) mel[0, 0, f, m] = NumOps.FromDouble((features[f, m] - GlobalMean[m]) * GlobalInverseStd[m]);
        var x = _engine.ReLU(FirstConv.Forward(mel));
        foreach (var stage in DepthwiseStages) x = stage.Forward(x);
        int length = x.Shape[2];
        var steps = _engine.Reshape(_engine.TensorPermute(x, new[] { 0, 2, 1, 3 }), new[] { length, c.SubsamplingChannels * SubsampledBins });
        var h = SubsamplingOut.Forward(steps);
        var chunks = new List<Tensor<T>>();
        for (int start = 0; start < length; start += c.AttentionChunk)
        {
            int size = Math.Min(c.AttentionChunk, length - start);
            var chunk = size == length ? h : _engine.TensorSlice(h, new[] { start, 0 }, new[] { size, c.EncoderWidth });
            var bias = Bias(size);
            foreach (var block in Blocks) chunk = Run(block, chunk, bias);
            chunks.Add(chunk);
        }
        return chunks.Count == 1 ? chunks[0] : _engine.TensorConcatenate(chunks.ToArray(), 0);
    }

    // T5RelativeAttentionLogitBias: bias[h, i, j] = values[clamp(j − i, −max, max − 1) + max, h], [heads, L, L].
    private Tensor<T> Bias(int length)
    {
        int max = Configuration.RelativeBiasMaxDistance;
        var index = new Tensor<int>(new[] { length * length });
        for (int i = 0; i < length; i++)
            for (int j = 0; j < length; j++) index[i * length + j] = Math.Max(-max, Math.Min(max - 1, j - i)) + max;
        var rows = _engine.TensorIndexSelect(RelativeBias.Weights, index, 0);                                  // [L·L, heads]
        return _engine.TensorPermute(_engine.Reshape(rows, new[] { length, length, Configuration.EncoderHeads }), new[] { 2, 0, 1 });
    }

    private Tensor<T> Run(Block block, Tensor<T> x, Tensor<T> bias)
    {
        var half = NumOps.FromDouble(0.5);
        x = _engine.TensorAdd(x, _engine.TensorMultiplyScalar(Run(block.FeedForwardIn, x), half));
        x = _engine.TensorAdd(x, Attention(block, block.AttentionNorm.Forward(x), bias));
        x = _engine.TensorAdd(x, Convolution(block, x));
        x = _engine.TensorAdd(x, _engine.TensorMultiplyScalar(Run(block.FeedForwardOut, x), half));
        return block.Norm.Forward(x);
    }

    private Tensor<T> Attention(Block block, Tensor<T> h, Tensor<T> bias)
    {
        var c = Configuration;
        int length = h.Shape[0], heads = c.EncoderHeads, hd = c.EncoderWidth / heads;
        Tensor<T> Heads(Tensor<T> t) => _engine.TensorPermute(_engine.Reshape(t, new[] { length, heads, hd }), new[] { 1, 0, 2 });
        var q = _engine.TensorMultiplyScalar(Heads(block.Query.Forward(h)), NumOps.FromDouble(1.0 / Math.Sqrt(hd)));
        var k = Heads(block.Key.Forward(h));
        var v = Heads(block.Value.Forward(h));
        var scores = _engine.TensorAdd(_engine.BatchMatMul(q, _engine.TensorPermute(k, new[] { 0, 2, 1 })), bias);
        var mixed = _engine.BatchMatMul(_engine.Softmax(scores, axis: 2), v);
        return block.Output.Forward(_engine.Reshape(_engine.TensorPermute(mixed, new[] { 1, 0, 2 }), new[] { length, c.EncoderWidth }));
    }

    /// <summary>The convolution module: LayerNorm; the pointwise GLU, (a + b1) · Swish(g + b2); the causal depthwise
    /// convolution (kernel − 1 frames of left padding) and its pointwise convolution; Swish; the extra pointwise
    /// convolution.</summary>
    private Tensor<T> Convolution(Block block, Tensor<T> x)
    {
        int frames = x.Shape[0], d = Configuration.EncoderWidth, kernel = Configuration.ConvKernel;
        var up = block.GluConv.Forward(block.ConvNorm.Forward(x));
        var gluBias = block.GluBias.Weights;                                                                        // [2, d]
        Tensor<T> Row(Tensor<T> table, int row) => _engine.TensorBroadcastTo(_engine.TensorSlice(table, new[] { row, 0 }, new[] { 1, d }), new[] { frames, d });
        var gated = _engine.TensorMultiply(_engine.TensorAdd(_engine.TensorSlice(up, new[] { 0, 0 }, new[] { frames, d }), Row(gluBias, 0)),
            _engine.Swish(_engine.TensorAdd(_engine.TensorSlice(up, new[] { 0, d }, new[] { frames, d }), Row(gluBias, 1))));
        var padded = kernel > 1 ? _engine.TensorConcatenate(new[] { new Tensor<T>(new[] { kernel - 1, d }), gated }, 0) : gated;
        var weights = block.DepthwiseKernel.Weights;                                                                // [K, d]
        Tensor<T> convolved = Row(block.DepthwiseBias.Weights, 0);
        for (int t = 0; t < kernel; t++)
            convolved = _engine.TensorAdd(convolved, _engine.TensorMultiply(_engine.TensorSlice(padded, new[] { t, 0 }, new[] { frames, d }), Row(weights, t)));
        var pointwise = _engine.Swish(block.PointwiseConv.Forward(convolved));
        return block.ExtPointwiseConv.Forward(pointwise);
    }

    /// <summary>The speech projector's output <c>[frames, LLM width]</c>: linear, GELU, linear.</summary>
    public Tensor<T> Project(Tensor<T> encoded) => Projection2.Forward(ActivationHelper.ExactGelu(Projection1.Forward(encoded), _engine));

    /// <summary>The projected speech of a waveform.</summary>
    public Tensor<T> Speech(IReadOnlyList<double> waveform) => Project(Encode(Features(waveform)));

    /// <summary>
    /// The prompt embeddings <c>[L, LLM width]</c> (the model card's): <c>&lt;|user|&gt;</c>, the projected speech in place
    /// of the <c>&lt;|audio_1|&gt;</c> placeholders, then <c>{instruction}&lt;|end|&gt;&lt;|assistant|&gt;</c>. Without
    /// Phi's vocabulary the prompt is the speech alone.
    /// </summary>
    public Tensor<T> Prompt(Tensor<T> speech, string instruction)
    {
        if (!UsesPhiVocabulary) return speech;
        var head = Tokenizer.Encode("<|user|>");
        var tail = Tokenizer.Encode($"{instruction}<|end|><|assistant|>");
        return _engine.TensorConcatenate(new[] { TokenEmbeddings(head), speech, TokenEmbeddings(tail) }, 0);
    }

    /// <summary>The audio placeholder token (<c>&lt;|endoftext11|&gt;</c>, id 200,011).</summary>
    public int AudioTokenId => Tokenizer.TokenId(AudioToken);

    public Tensor<T> TokenEmbeddings(IReadOnlyList<int> ids)
    {
        var grid = new int[1, ids.Count];
        for (int i = 0; i < ids.Count; i++) grid[0, i] = ids[i];
        return _engine.Reshape(Llm.Embed(grid), new[] { ids.Count, Configuration.Llm.Hidden });
    }

    /// <summary>The logits <c>[1, vocabulary]</c> of the answer's first token.</summary>
    public Tensor<T> FirstLogits(Tensor<T> prompt)
    {
        int length = prompt.Shape[0], width = Configuration.Llm.Hidden, vocabulary = Configuration.Llm.VocabSize;
        var (_, logits) = Llm.Forward(_engine.Reshape(prompt, new[] { 1, length, width }));
        var all = logits ?? throw new InvalidOperationException("The decoder has no head.");
        return _engine.Reshape(_engine.TensorSlice(all, new[] { 0, length - 1, 0 }, new[] { 1, 1, vocabulary }), new[] { 1, vocabulary });
    }

    /// <summary>The training loss: next-token cross-entropy over the transcript and the end tokens only (the release's
    /// fine-tuning recipe masks the prompt).</summary>
    public Tensor<T> Loss(Tensor<T> prompt, IReadOnlyList<int> transcript)
    {
        var target = transcript.Concat(EndTokens).ToArray();
        var input = _engine.TensorConcatenate(new[] { prompt, TokenEmbeddings(target) }, 0);
        int length = input.Shape[0], promptLength = prompt.Shape[0];
        var (_, logits) = Llm.Forward(_engine.Reshape(input, new[] { 1, length, Configuration.Llm.Hidden }));
        var labels = new int[1, length];
        for (int i = 0; i < promptLength; i++) labels[0, i] = -100;
        for (int i = 0; i < target.Length; i++) labels[0, promptLength + i] = target[i];
        return Llm.CausalLmLoss(logits ?? throw new InvalidOperationException("The decoder has no head."), labels);
    }

    /// <summary>The transcript by greedy decoding after the prompt, up to the first end token.</summary>
    public int[] Greedy(Tensor<T> prompt, int maxTokens)
    {
        int width = Configuration.Llm.Hidden, vocabulary = Configuration.Llm.VocabSize, end = EndTokens[0];
        var cache = new LlamaCache<T>();
        var answer = new List<int>();
        var input = prompt;
        for (int step = 0; step < maxTokens; step++)
        {
            var (_, logits) = Llm.Forward(_engine.Reshape(input, new[] { 1, input.Shape[0], width }), null, null, cache, lastLogitsOnly: true);
            var row = logits ?? throw new InvalidOperationException("The decoder has no head.");
            int best = 0;
            for (int v = 1; v < vocabulary; v++) if (NumOps.GreaterThan(row.GetFlat(v), row.GetFlat(best))) best = v;
            if (best == end) break;
            answer.Add(best);
            input = TokenEmbeddings(new[] { best });
        }
        return answer.ToArray();
    }
}
