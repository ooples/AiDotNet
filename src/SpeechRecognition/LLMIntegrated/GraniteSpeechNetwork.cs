using AiDotNet.ActivationFunctions;
using AiDotNet.Audio.Features;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.Tokenization.Algorithms;

namespace AiDotNet.SpeechRecognition.LLMIntegrated;

/// <summary>The geometry of a <see cref="GraniteSpeechNetwork{T}"/>: the CTC encoder, the window Q-Former projector and
/// the Granite LLM (the release's <c>GraniteSpeechConfig</c>).</summary>
internal sealed record GraniteSpeechConfiguration
{
    public int SampleRate { get; init; } = 16000;
    public int Mels { get; init; } = 80;
    public int EncoderWidth { get; init; } = 1024;
    public int EncoderLayers { get; init; } = 16;
    public int EncoderHeads { get; init; } = 8;
    public int EncoderHeadDim { get; init; } = 128;
    public int FeedForwardMultiplier { get; init; } = 4;
    public int ContextSize { get; init; } = 200;
    public int MaxRelativePosition { get; init; } = 512;
    public int ConvKernel { get; init; } = 15;
    public int ConvExpansion { get; init; } = 2;
    public int CtcOutputs { get; init; } = 256;
    public int ProjectorWidth { get; init; } = 1024;
    public int ProjectorLayers { get; init; } = 2;
    public int ProjectorHeads { get; init; } = 16;
    public int ProjectorIntermediate { get; init; } = 4096;
    public int WindowSize { get; init; } = 15;
    public int DownsampleRate { get; init; } = 5;
    public LlamaConfiguration Llm { get; init; } = new();
}

/// <summary>
/// Granite-speech's networks (Saon et al. 2025, and the release's <c>modeling_granite_speech.py</c>): the conformer CTC
/// encoder over pairs of stacked log-mel frames, with block self-attention and Shaw relative positions and a
/// self-conditioned CTC branch at the middle layer; the window Q-Former that turns each window of encoder frames into a
/// few query vectors; and the Granite LLM that reads them in place of the <c>&lt;|audio|&gt;</c> tokens.
/// </summary>
internal sealed class GraniteSpeechNetwork<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private const string TokenizerResource = "AiDotNet.Tokenization.granite3-tokenizer.json.gz";
    private const int FftSize = 512, WindowLength = 400, HopLength = 160;
    // granite-3.3's tokenizer with the release's <|audio|> (id 49159).
    private static readonly Lazy<ByteLevelBpeTokenizer> GraniteText = new(LoadTokenizer);
    private readonly IEngine _engine;
    private readonly double[,] _melBank;

    internal sealed record FeedForward(LayerNormalizationLayer<T> Norm, DenseLayer<T> Up, DenseLayer<T> Down);

    internal sealed record Block(FeedForward FeedForward1, LayerNormalizationLayer<T> AttentionNorm, DenseLayer<T> Query,
        DenseLayer<T> KeyValue, DenseLayer<T> Output, BiasFreeLinearLayer<T> RelativePositions, LayerNormalizationLayer<T> ConvNorm,
        DenseLayer<T> ConvUp, BiasFreeLinearLayer<T> DepthwiseKernel, BatchNormalizationLayer<T> ConvBatchNorm, DenseLayer<T> ConvDown,
        FeedForward FeedForward2, LayerNormalizationLayer<T> PostNorm);

    internal sealed record QFormerAttention(DenseLayer<T> Query, DenseLayer<T> Key, DenseLayer<T> Value, DenseLayer<T> Output,
        LayerNormalizationLayer<T> Norm);

    internal sealed record QFormerLayer(QFormerAttention SelfAttention, QFormerAttention CrossAttention, DenseLayer<T> Intermediate,
        DenseLayer<T> Output, LayerNormalizationLayer<T> Norm);

    public GraniteSpeechNetwork(IEngine engine, GraniteSpeechConfiguration configuration, List<LayerBase<T>> layers, Random random)
    {
        _engine = engine;
        var c = configuration;
        if (c.ContextSize <= 0 || c.ContextSize > c.MaxRelativePosition)
            throw new ArgumentException("The attention block must be positive and within the relative positions.", nameof(configuration));
        if (c.WindowSize % c.DownsampleRate != 0)
            throw new ArgumentException("The projector's window must be a multiple of its downsampling rate.", nameof(configuration));
        Configuration = c;
        _melBank = MelFilterbanks.Htk(c.SampleRate, FftSize, c.Mels, 0.0, c.SampleRate / 2.0);
        TLayer Own<TLayer>(List<LayerBase<T>> owner, TLayer layer) where TLayer : LayerBase<T> { owner.Add(layer); return layer; }
        // PyTorch's default initialization: uniform in ±1/√fan-in for weights and biases.
        DenseLayer<T> Linear(List<LayerBase<T>> owner, int inputs, int outputs)
        {
            var layer = Own(owner, new DenseLayer<T>(outputs, (IActivationFunction<T>)new IdentityActivation<T>()));
            double bound = 1.0 / Math.Sqrt(inputs);
            TorchParameters.Linear(engine, layer, inputs, outputs, Enumerable.Range(0, inputs * outputs).Select(_ => bound * (2 * random.NextDouble() - 1)).ToArray(),
                Enumerable.Range(0, outputs).Select(_ => bound * (2 * random.NextDouble() - 1)).ToArray());
            return layer;
        }
        BiasFreeLinearLayer<T> Table(List<LayerBase<T>> owner, int rows, int columns, Func<double> draw)
        {
            var layer = Own(owner, new BiasFreeLinearLayer<T>(rows, columns));
            // TorchParameters reads [out, in]; the holder's weights are [rows, columns].
            var values = new double[rows * columns];
            for (int r = 0; r < rows; r++) for (int k = 0; k < columns; k++) values[k * rows + r] = draw();
            TorchParameters.Linear(layer, rows, columns, values);
            return layer;
        }
        double Normal() => Math.Sqrt(-2.0 * Math.Log(1.0 - random.NextDouble())) * Math.Cos(2.0 * Math.PI * random.NextDouble());
        LayerNormalizationLayer<T> Norm(List<LayerBase<T>> owner, int width, double epsilon) => Own(owner, new LayerNormalizationLayer<T>(width, epsilon));

        int d = c.EncoderWidth, inner = c.EncoderHeads * c.EncoderHeadDim, convInner = d * c.ConvExpansion;
        FeedForward EncoderFeedForward() => new(Norm(EncoderLayers, d, 1e-5), Linear(EncoderLayers, d, d * c.FeedForwardMultiplier),
            Linear(EncoderLayers, d * c.FeedForwardMultiplier, d));
        InputLinear = Linear(EncoderLayers, 2 * c.Mels, d);
        for (int i = 0; i < c.EncoderLayers; i++)
        {
            var ff1 = EncoderFeedForward();
            var attentionNorm = Norm(EncoderLayers, d, 1e-5);
            var query = Linear(EncoderLayers, d, inner);
            var keyValue = Linear(EncoderLayers, d, 2 * inner);
            var output = Linear(EncoderLayers, inner, d);
            // nn.Embedding: N(0, 1).
            var relative = Table(EncoderLayers, 2 * c.MaxRelativePosition + 1, c.EncoderHeadDim, Normal);
            var convNorm = Norm(EncoderLayers, d, 1e-5);
            var convUp = Linear(EncoderLayers, d, 2 * convInner);
            double kernelBound = 1.0 / Math.Sqrt(c.ConvKernel);
            var depthwise = Table(EncoderLayers, c.ConvKernel, convInner, () => kernelBound * (2 * random.NextDouble() - 1));
            var batchNorm = Own(EncoderLayers, new BatchNormalizationLayer<T>(convInner, 1e-5, 0.9) { Layout = BatchNormDataLayout.ChannelsFirst });
            var convDown = Linear(EncoderLayers, convInner, d);
            var ff2 = EncoderFeedForward();
            Blocks.Add(new Block(ff1, attentionNorm, query, keyValue, output, relative, convNorm, convUp, depthwise, batchNorm, convDown, ff2,
                Norm(EncoderLayers, d, 1e-5)));
        }
        CtcOut = Linear(EncoderLayers, d, c.CtcOutputs);
        CtcOutMid = Linear(EncoderLayers, c.CtcOutputs, d);

        int q = c.ProjectorWidth;
        Queries = Table(ProjectorLayers, c.WindowSize / c.DownsampleRate, q, Normal);
        QueryNorm = Norm(ProjectorLayers, q, 1e-12);
        QFormerAttention Attention(int keyWidth) => new(Linear(ProjectorLayers, q, q), Linear(ProjectorLayers, keyWidth, q),
            Linear(ProjectorLayers, keyWidth, q), Linear(ProjectorLayers, q, q), Norm(ProjectorLayers, q, 1e-12));
        for (int i = 0; i < c.ProjectorLayers; i++)
            QFormer.Add(new QFormerLayer(Attention(q), Attention(d), Linear(ProjectorLayers, q, c.ProjectorIntermediate),
                Linear(ProjectorLayers, c.ProjectorIntermediate, q), Norm(ProjectorLayers, q, 1e-12)));
        ProjectorLinear = Linear(ProjectorLayers, q, c.Llm.Hidden);

        Llm = new LlamaNetwork<T>(engine, LlmLayers, c.Llm);
        Llm.InitializeLikeReference(random);
        layers.AddRange(EncoderLayers);
        layers.AddRange(ProjectorLayers);
        layers.AddRange(LlmLayers);
    }

    public GraniteSpeechConfiguration Configuration { get; }
    public DenseLayer<T> InputLinear { get; }
    public List<Block> Blocks { get; } = new();
    public DenseLayer<T> CtcOut { get; }
    public DenseLayer<T> CtcOutMid { get; }
    public BiasFreeLinearLayer<T> Queries { get; }
    public LayerNormalizationLayer<T> QueryNorm { get; }
    public List<QFormerLayer> QFormer { get; } = new();
    public DenseLayer<T> ProjectorLinear { get; }
    public LlamaNetwork<T> Llm { get; }
    public List<LayerBase<T>> EncoderLayers { get; } = new();
    public List<LayerBase<T>> ProjectorLayers { get; } = new();
    public List<LayerBase<T>> LlmLayers { get; } = new();

    /// <summary>The CTC blank (index 0).</summary>
    public const int Blank = 0;

    /// <summary>Whether the vocabulary covers the release tokenizer's, so the prompt is chat text; a smaller vocabulary
    /// has no prompt text, and its last token closes the answer.</summary>
    public bool UsesGraniteVocabulary => Configuration.Llm.VocabSize >= GraniteText.Value.VocabularySize;

    public ByteLevelBpeTokenizer Tokenizer => GraniteText.Value;

    /// <summary>The token that closes the answer: <c>&lt;|end_of_text|&gt;</c> (id 0, the release's end of sequence), or
    /// the last token without Granite's vocabulary.</summary>
    public int EndToken => UsesGraniteVocabulary ? Tokenizer.TokenId("<|end_of_text|>") : Configuration.Llm.VocabSize - 1;

    private static ByteLevelBpeTokenizer LoadTokenizer()
    {
        using var stream = typeof(GraniteSpeechNetwork<T>).Assembly.GetManifestResourceStream(TokenizerResource)
            ?? throw new InvalidDataException($"The embedded resource {TokenizerResource} is missing.");
        using var gzip = new System.IO.Compression.GZipStream(stream, System.IO.Compression.CompressionMode.Decompress);
        using var reader = new StreamReader(gzip, System.Text.Encoding.UTF8);
        return ByteLevelBpeTokenizer.FromJson(reader.ReadToEnd());
    }

    /// <summary>
    /// The encoder's input <c>[frames / 2, 2 · mels]</c> of a 16 kHz waveform (the release's
    /// <c>GraniteSpeechFeatureExtractor</c>): torchaudio's <c>MelSpectrogram</c> (a centred, reflect-padded 512-point STFT
    /// of a 400-sample periodic Hann window, hop 160, power 2, HTK filters), log10 above 1e-10, floored 8 below the
    /// maximum, divided by 4 plus 1, the odd last frame dropped and each two frames stacked.
    /// </summary>
    public Tensor<T> Features(IReadOnlyList<double> waveform)
    {
        int length = waveform.Count, half = FftSize / 2;
        if (length <= half) throw new ArgumentException($"The recording is shorter than {half + 1} samples.", nameof(waveform));
        var padded = new double[length + 2 * half];
        for (int i = 0; i < length; i++) padded[half + i] = waveform[i];
        for (int i = 0; i < half; i++)
        {
            padded[half - 1 - i] = waveform[i + 1];
            padded[half + length + i] = waveform[length - 2 - i];
        }
        // torch.stft centres the shorter window in the FFT frame.
        var window = new double[FftSize];
        int offset = (FftSize - WindowLength) / 2;
        for (int n = 0; n < WindowLength; n++) window[offset + n] = 0.5 - 0.5 * Math.Cos(2 * Math.PI * n / WindowLength);
        int frames = 1 + length / HopLength, bins = FftSize / 2 + 1, mels = Configuration.Mels;
        var cos = new double[bins * FftSize];
        var sin = new double[bins * FftSize];
        for (int k = 0; k < bins; k++)
            for (int n = 0; n < FftSize; n++)
            {
                double angle = 2 * Math.PI * ((long)k * n % FftSize) / FftSize;
                cos[k * FftSize + n] = Math.Cos(angle);
                sin[k * FftSize + n] = Math.Sin(angle);
            }
        var log = new double[frames, mels];
        var power = new double[bins];
        double max = double.NegativeInfinity;
        for (int f = 0; f < frames; f++)
        {
            for (int k = 0; k < bins; k++)
            {
                double re = 0, im = 0;
                for (int n = offset; n < offset + WindowLength; n++)
                {
                    double sample = padded[f * HopLength + n] * window[n];
                    re += sample * cos[k * FftSize + n];
                    im -= sample * sin[k * FftSize + n];
                }
                power[k] = re * re + im * im;
            }
            for (int m = 0; m < mels; m++)
            {
                double sum = 0;
                for (int k = 0; k < bins; k++) sum += _melBank[m, k] * power[k];
                log[f, m] = Math.Log10(Math.Max(sum, 1e-10));
                if (log[f, m] > max) max = log[f, m];
            }
        }
        int pairs = frames / 2;
        if (pairs == 0) throw new ArgumentException("The recording is shorter than two feature frames.", nameof(waveform));
        var features = new Tensor<T>(new[] { pairs, 2 * mels });
        for (int f = 0; f < 2 * pairs; f++)
            for (int m = 0; m < mels; m++)
                features[f / 2, f % 2 * mels + m] = NumOps.FromDouble(Math.Max(log[f, m], max - 8.0) / 4.0 + 1.0);
        return features;
    }

    private Tensor<T> Run(FeedForward ff, Tensor<T> x) => ff.Down.Forward(_engine.Swish(ff.Up.Forward(ff.Norm.Forward(x))));

    /// <summary>
    /// The encoder's output <c>[frames, width]</c> of features <c>[frames, 2 · mels]</c>, and the middle layer's CTC
    /// logits (<c>out</c> of its hidden states) that condition the rest of the stack.
    /// </summary>
    public (Tensor<T> Hidden, Tensor<T> MiddleLogits) Encode(Tensor<T> features, bool training)
    {
        var half = NumOps.FromDouble(0.5);
        var x = InputLinear.Forward(features);
        Tensor<T>? middle = null;
        for (int i = 0; i < Blocks.Count; i++)
        {
            var block = Blocks[i];
            block.ConvBatchNorm.SetTrainingMode(training);
            x = _engine.TensorAdd(x, _engine.TensorMultiplyScalar(Run(block.FeedForward1, x), half));
            x = _engine.TensorAdd(x, Attention(block, x));
            x = _engine.TensorAdd(x, Convolution(block, x));
            x = _engine.TensorAdd(x, _engine.TensorMultiplyScalar(Run(block.FeedForward2, x), half));
            x = block.PostNorm.Forward(x);
            if (i + 1 == Blocks.Count / 2)
            {
                // Self-conditioned CTC (Nozaki & Komatsu 2021): the middle layer's posteriors feed back into the stack.
                middle = CtcOut.Forward(x);
                x = _engine.TensorAdd(x, CtcOutMid.Forward(_engine.Softmax(middle, axis: 1)));
            }
        }
        return (x, middle ?? CtcOut.Forward(x));
    }

    /// <summary>Block self-attention: blocks of <see cref="GraniteSpeechConfiguration.ContextSize"/> frames attend within
    /// themselves, scored by q·k plus Shaw's q·r with r the embedding of the clamped offset; the last block is
    /// zero-padded and its padding masked.</summary>
    private Tensor<T> Attention(Block block, Tensor<T> x)
    {
        var c = Configuration;
        int frames = x.Shape[0], width = x.Shape[1], context = c.ContextSize, heads = c.EncoderHeads, hd = c.EncoderHeadDim, inner = heads * hd;
        int blocks = (frames + context - 1) / context, remainder = frames % context;
        var h = block.AttentionNorm.Forward(x);
        if (remainder > 0) h = _engine.TensorConcatenate(new[] { h, new Tensor<T>(new[] { context - remainder, width }) }, 0);
        var query = block.Query.Forward(h);
        var keyValue = block.KeyValue.Forward(h);
        int rows = blocks * context;
        var key = _engine.TensorSlice(keyValue, new[] { 0, 0 }, new[] { rows, inner });
        var value = _engine.TensorSlice(keyValue, new[] { 0, inner }, new[] { rows, inner });
        Tensor<T> Heads(Tensor<T> t) => _engine.TensorPermute(_engine.Reshape(t, new[] { blocks, context, heads, hd }), new[] { 0, 2, 1, 3 });
        var q = Heads(query);
        var k = Heads(key);
        var v = Heads(value);
        var scale = NumOps.FromDouble(1.0 / Math.Sqrt(hd));
        int groups = blocks * heads;
        var scores = _engine.BatchMatMul(_engine.Reshape(q, new[] { groups, context, hd }),
            _engine.TensorPermute(_engine.Reshape(k, new[] { groups, context, hd }), new[] { 0, 2, 1 }));
        // r[i, j] = rel_pos_emb[clamp(i − j, −C, C) + max_pos_emb]; pos[g, i, j] = q[g, i] · r[i, j].
        var index = new Tensor<int>(new[] { context * context });
        for (int i = 0; i < context; i++)
            for (int j = 0; j < context; j++) index[i * context + j] = Math.Max(-context, Math.Min(context, i - j)) + c.MaxRelativePosition;
        var table = block.RelativePositions.Weights;
        var relative = _engine.TensorPermute(_engine.Reshape(_engine.TensorIndexSelect(table, index, 0), new[] { context, context, hd }), new[] { 0, 2, 1 });
        var perRow = _engine.TensorPermute(_engine.Reshape(q, new[] { groups, context, hd }), new[] { 1, 0, 2 });                 // [C, G, hd]
        var positional = _engine.TensorPermute(_engine.BatchMatMul(perRow, relative), new[] { 1, 0, 2 });                             // [G, C, C]
        var total = _engine.TensorMultiplyScalar(_engine.TensorAdd(scores, positional), scale);
        if (remainder > 0)
        {
            var mask = new Tensor<T>(new[] { blocks, heads, context, context });
            var blocked = NumOps.FromDouble(typeof(T) == typeof(float) ? float.MinValue : double.MinValue);
            for (int hh = 0; hh < heads; hh++)
                for (int i = 0; i < context; i++)
                    for (int j = 0; j < context; j++)
                        if (i >= remainder || j >= remainder) mask[blocks - 1, hh, i, j] = blocked;
            total = _engine.TensorAdd(total, _engine.Reshape(mask, new[] { groups, context, context }));
        }
        var probabilities = _engine.Softmax(total, axis: 2);
        var context3 = _engine.BatchMatMul(probabilities, _engine.Reshape(v, new[] { groups, context, hd }));
        var merged = _engine.Reshape(_engine.TensorPermute(_engine.Reshape(context3, new[] { blocks, heads, context, hd }), new[] { 0, 2, 1, 3 }), new[] { rows, inner });
        if (rows != frames) merged = _engine.TensorSlice(merged, new[] { 0, 0 }, new[] { frames, inner });
        return block.Output.Forward(merged);
    }

    /// <summary>The convolution module: LayerNorm, a pointwise convolution to twice the inner width, GLU, the bias-free
    /// depthwise convolution (padding k/2 left and k/2 − (k + 1) mod 2 right), BatchNorm, SiLU and a pointwise
    /// convolution back.</summary>
    private Tensor<T> Convolution(Block block, Tensor<T> x)
    {
        var c = Configuration;
        int frames = x.Shape[0], inner = c.EncoderWidth * c.ConvExpansion, kernel = c.ConvKernel;
        var up = block.ConvUp.Forward(block.ConvNorm.Forward(x));
        var gated = _engine.TensorMultiply(_engine.TensorSlice(up, new[] { 0, 0 }, new[] { frames, inner }),
            _engine.Sigmoid(_engine.TensorSlice(up, new[] { 0, inner }, new[] { frames, inner })));
        int left = kernel / 2, right = kernel / 2 - (kernel + 1) % 2;
        var pieces = new List<Tensor<T>>();
        if (left > 0) pieces.Add(new Tensor<T>(new[] { left, inner }));
        pieces.Add(gated);
        if (right > 0) pieces.Add(new Tensor<T>(new[] { right, inner }));
        var padded = pieces.Count == 1 ? gated : _engine.TensorConcatenate(pieces.ToArray(), 0);
        var weights = block.DepthwiseKernel.Weights;                                                                                 // [K, inner]
        Tensor<T>? convolved = null;
        for (int t = 0; t < kernel; t++)
        {
            var tap = _engine.TensorMultiply(_engine.TensorSlice(padded, new[] { t, 0 }, new[] { frames, inner }),
                _engine.TensorBroadcastTo(_engine.TensorSlice(weights, new[] { t, 0 }, new[] { 1, inner }), new[] { frames, inner }));
            convolved = convolved is null ? tap : _engine.TensorAdd(convolved, tap);
        }
        var depthwise = convolved ?? throw new InvalidOperationException("The depthwise kernel is empty.");
        var channelsFirst = _engine.Reshape(_engine.TensorTranspose(depthwise), new[] { 1, inner, frames });
        var normalized = _engine.TensorTranspose(_engine.Reshape(block.ConvBatchNorm.Forward(channelsFirst), new[] { inner, frames }));
        return block.ConvDown.Forward(_engine.Swish(normalized));
    }

    /// <summary>
    /// The projector's output <c>[⌈frames / K⌉ · N, LLM width]</c>: the encoder frames zero-padded to whole windows of
    /// K = <see cref="GraniteSpeechConfiguration.WindowSize"/>, each window read by the Q-Former's N = K / downsampling
    /// learned queries, then the linear map into the LLM's width.
    /// </summary>
    public Tensor<T> Project(Tensor<T> encoded)
    {
        var c = Configuration;
        int frames = encoded.Shape[0], d = encoded.Shape[1], window = c.WindowSize, queries = window / c.DownsampleRate, q = c.ProjectorWidth;
        int windows = (frames + window - 1) / window, pad = windows * window - frames;
        var padded = pad > 0 ? _engine.TensorConcatenate(new[] { encoded, new Tensor<T>(new[] { pad, d }) }, 0) : encoded;
        var memory = _engine.Reshape(padded, new[] { windows, window, d });
        // Blip2QFormerModel: LayerNorm of the query embeddings, shared by every window.
        var embedded = QueryNorm.Forward(_engine.Reshape(Queries.Weights, new[] { queries, q }));
        var h = _engine.TensorBroadcastTo(_engine.Reshape(embedded, new[] { 1, queries, q }), new[] { windows, queries, q });
        foreach (var layer in QFormer)
        {
            h = Attend(layer.SelfAttention, h, h);
            h = Attend(layer.CrossAttention, h, memory);
            var flat = _engine.Reshape(h, new[] { windows * queries, q });
            var ff = layer.Output.Forward(ActivationHelper.ExactGelu(layer.Intermediate.Forward(flat), _engine));
            h = _engine.Reshape(layer.Norm.Forward(_engine.TensorAdd(ff, flat)), new[] { windows, queries, q });
        }
        return ProjectorLinear.Forward(_engine.Reshape(h, new[] { windows * queries, q }));
    }

    // BERT attention with its post-norm residual: LayerNorm(dense(softmax(q·kᵀ/√d)·v) + h), keys and values from memory.
    private Tensor<T> Attend(QFormerAttention attention, Tensor<T> h, Tensor<T> memory)
    {
        var c = Configuration;
        int windows = h.Shape[0], queries = h.Shape[1], q = c.ProjectorWidth, keys = memory.Shape[1], width = memory.Shape[2];
        int heads = c.ProjectorHeads, hd = q / heads, groups = windows * heads;
        var flat = _engine.Reshape(h, new[] { windows * queries, q });
        var flatMemory = _engine.Reshape(memory, new[] { windows * keys, width });
        Tensor<T> Heads(Tensor<T> t, int length) =>
            _engine.Reshape(_engine.TensorPermute(_engine.Reshape(t, new[] { windows, length, heads, hd }), new[] { 0, 2, 1, 3 }), new[] { groups, length, hd });
        var query = Heads(attention.Query.Forward(flat), queries);
        var key = Heads(attention.Key.Forward(flatMemory), keys);
        var value = Heads(attention.Value.Forward(flatMemory), keys);
        var scores = _engine.TensorMultiplyScalar(_engine.BatchMatMul(query, _engine.TensorPermute(key, new[] { 0, 2, 1 })), NumOps.FromDouble(1.0 / Math.Sqrt(hd)));
        var mixed = _engine.BatchMatMul(_engine.Softmax(scores, axis: 2), value);
        var merged = _engine.Reshape(_engine.TensorPermute(_engine.Reshape(mixed, new[] { windows, heads, queries, hd }), new[] { 0, 2, 1, 3 }), new[] { windows * queries, q });
        return _engine.Reshape(attention.Norm.Forward(_engine.TensorAdd(attention.Output.Forward(merged), flat)), new[] { windows, queries, q });
    }

    /// <summary>The projector's output for a waveform: features, encoder, Q-Former and linear map.</summary>
    public Tensor<T> Speech(IReadOnlyList<double> waveform, bool training) => Project(Encode(Features(waveform), training).Hidden);

    /// <summary>
    /// The prompt embeddings <c>[L, LLM width]</c> (granite-3.3's chat template, as the model card builds it):
    /// <c>&lt;|start_of_role|&gt;system&lt;|end_of_role|&gt;{system}&lt;|end_of_text|&gt;\n&lt;|start_of_role|&gt;user
    /// &lt;|end_of_role|&gt;</c>, the projector's output in place of the <c>&lt;|audio|&gt;</c> tokens, then
    /// <c>{instruction}&lt;|end_of_text|&gt;\n&lt;|start_of_role|&gt;assistant&lt;|end_of_role|&gt;</c>. Without
    /// Granite's vocabulary the prompt is the speech alone.
    /// </summary>
    public Tensor<T> Prompt(Tensor<T> speech, string systemPrompt, string instruction)
    {
        if (!UsesGraniteVocabulary) return speech;
        var head = Tokenizer.Encode($"<|start_of_role|>system<|end_of_role|>{systemPrompt}<|end_of_text|>\n<|start_of_role|>user<|end_of_role|>");
        var tail = Tokenizer.Encode($"{instruction}<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>");
        return _engine.TensorConcatenate(new[] { TokenEmbeddings(head), speech, TokenEmbeddings(tail) }, 0);
    }

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

    /// <summary>The training loss: next-token cross-entropy over the transcript and the end token only (§II-E).</summary>
    public Tensor<T> Loss(Tensor<T> prompt, IReadOnlyList<int> transcript)
    {
        var target = transcript.Concat(new[] { EndToken }).ToArray();
        var input = _engine.TensorConcatenate(new[] { prompt, TokenEmbeddings(target) }, 0);
        int length = input.Shape[0], promptLength = prompt.Shape[0];
        var (_, logits) = Llm.Forward(_engine.Reshape(input, new[] { 1, length, Configuration.Llm.Hidden }));
        var labels = new int[1, length];
        for (int i = 0; i < promptLength; i++) labels[0, i] = -100;
        for (int i = 0; i < target.Length; i++) labels[0, promptLength + i] = target[i];
        return Llm.CausalLmLoss(logits ?? throw new InvalidOperationException("The decoder has no head."), labels);
    }

    /// <summary>
    /// The CTC encoder's training loss (§II-C): self-conditioned CTC, 0.8 times the CTC of the last layer's
    /// <c>out</c> logits plus 0.2 times that of the middle layer's, over CTC vocabulary labels (blank 0).
    /// </summary>
    public Tensor<T> CtcLoss(Tensor<T> features, IReadOnlyList<int> labels, bool training)
    {
        var (hidden, middle) = Encode(features, training);
        Tensor<T> Ctc(Tensor<T> logits)
        {
            int frames = logits.Shape[0], outputs = Configuration.CtcOutputs;
            var logProbabilities = _engine.TensorLogSoftmax(logits, axis: -1);
            var ctc = _engine.TensorCTCLoss(_engine.Reshape(logProbabilities, new[] { frames, 1, outputs }),
                new Tensor<int>(labels.ToArray(), new[] { labels.Count }), new[] { frames }, new[] { labels.Count }, Blank);
            return _engine.Reshape(ctc, new[] { 1 });
        }
        return _engine.TensorAdd(_engine.TensorMultiplyScalar(Ctc(CtcOut.Forward(hidden)), NumOps.FromDouble(0.8)),
            _engine.TensorMultiplyScalar(Ctc(middle), NumOps.FromDouble(0.2)));
    }

    /// <summary>The transcript by greedy decoding after the prompt, without the end token.</summary>
    public int[] Greedy(Tensor<T> prompt, int maxTokens)
    {
        int width = Configuration.Llm.Hidden, vocabulary = Configuration.Llm.VocabSize;
        var cache = new LlamaCache<T>();
        var answer = new List<int>();
        var input = prompt;
        for (int step = 0; step < maxTokens; step++)
        {
            var (_, logits) = Llm.Forward(_engine.Reshape(input, new[] { 1, input.Shape[0], width }), null, null, cache, lastLogitsOnly: true);
            var row = logits ?? throw new InvalidOperationException("The decoder has no head.");
            int best = 0;
            for (int v = 1; v < vocabulary; v++) if (NumOps.GreaterThan(row.GetFlat(v), row.GetFlat(best))) best = v;
            if (best == EndToken) break;
            answer.Add(best);
            input = TokenEmbeddings(new[] { best });
        }
        return answer.ToArray();
    }
}
