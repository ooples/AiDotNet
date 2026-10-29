using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.Latest;

namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>The shape of GLM-4-Voice's speech tokenizer; the defaults are the release's (whisper-large-v3 based).</summary>
internal sealed record GlmSpeechTokenizerConfiguration
{
    public int Mels { get; init; } = 128;
    public int Dim { get; init; } = 1280;
    public int Heads { get; init; } = 20;
    public int FeedForward { get; init; } = 5120;
    /// <summary>Encoder layers in all (32); inference runs only the first <see cref="QuantizePosition"/>.</summary>
    public int EncoderLayers { get; init; } = 32;
    public int MaxSourcePositions { get; init; } = 1500;
    public int PoolingKernel { get; init; } = 4;
    /// <summary>The layer after which the average pooling runs (16).</summary>
    public int PoolingPosition { get; init; } = 16;
    /// <summary>The layer after which the vector quantizer runs (16).</summary>
    public int QuantizePosition { get; init; } = 16;
    public int CodebookSize { get; init; } = 16384;
    /// <summary>Encoder frames per attention block before pooling (200; 50 after).</summary>
    public int CausalBlockSize { get; init; } = 200;
    public double EmaDecay { get; init; } = 0.99;
    public double CommitCoefficient { get; init; } = 0.25;
    public double LossScale { get; init; } = 10.0;
    public int RestartInterval { get; init; } = 100;
    /// <summary>Whisper decoder layers (32) and vocabulary (51866), for the ASR loss that trains the tokenizer.</summary>
    public int DecoderLayers { get; init; } = 32;
    public int Vocabulary { get; init; } = 51866;
    public int MaxTargetPositions { get; init; } = 448;
    public double Dropout { get; init; }
}

/// <summary>
/// GLM-4-Voice's supervised speech tokenizer (Zeng et al. 2024 section 3.1; THUDM's <c>WhisperVQEncoder</c>): whisper-large-v3's
/// encoder with causal convolutions, attention causal across 200-frame blocks, an average pooling (k = 4) and a
/// vector quantizer (16384 codes, EMA-updated) after layer 16, giving 12.5 tokens a second. For training, the encoder's
/// remaining layers and Whisper's decoder carry the ASR loss through the quantized bottleneck.
/// </summary>
internal sealed class GlmSpeechTokenizer<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly GlmSpeechTokenizerConfiguration _c;

    internal sealed class Attention
    {
        public DenseLayer<T> Query = null!, Value = null!, Output = null!;
        public BiasFreeLinearLayer<T> Key = null!;
    }

    internal sealed class EncoderLayer
    {
        public LayerNormalizationLayer<T> AttentionNorm = null!, FinalNorm = null!;
        public Attention SelfAttention = null!;
        public DenseLayer<T> Fc1 = null!, Fc2 = null!;
    }

    internal sealed class DecoderLayer
    {
        public LayerNormalizationLayer<T> SelfNorm = null!, CrossNorm = null!, FinalNorm = null!;
        public Attention SelfAttention = null!, CrossAttention = null!;
        public DenseLayer<T> Fc1 = null!, Fc2 = null!;
    }

    public GlmSpeechTokenizer(IEngine engine, List<LayerBase<T>> layers, GlmSpeechTokenizerConfiguration configuration, Random random,
        bool withTrainingStack)
    {
        _engine = engine;
        _c = configuration;
        var c = configuration;
        if (c.QuantizePosition < c.PoolingPosition) throw new ArgumentException("The quantizer runs at or after the pooling.", nameof(configuration));
        int d = c.Dim;
        Conv1 = new NormedConv1DLayer<T>(c.Mels, d, 3, 1, 1, 1, 0, false, ConvolutionNormalization.None);
        Conv2 = new NormedConv1DLayer<T>(d, d, 3, 2, 1, 1, 0, false, ConvolutionNormalization.None);
        layers.Add(Conv1);
        layers.Add(Conv2);
        Conv1.LoadTorchWeights(IndexTtsOps.Uniform(random, d * c.Mels * 3, c.Mels * 3), null, IndexTtsOps.Uniform(random, d, c.Mels * 3));
        Conv2.LoadTorchWeights(IndexTtsOps.Uniform(random, d * d * 3, d * 3), null, IndexTtsOps.Uniform(random, d, d * 3));
        Positions = IndexTtsOps.Values(layers, Sinusoids(c.MaxSourcePositions, d));
        int pooled = (c.MaxSourcePositions + c.PoolingKernel - 1) / c.PoolingKernel;
        Positions2 = IndexTtsOps.Values(layers, Sinusoids(c.MaxSourcePositions, d).Take(pooled * d).ToArray());
        int count = withTrainingStack ? c.EncoderLayers : c.QuantizePosition;
        for (int i = 0; i < count; i++)
            EncoderLayers.Add(new EncoderLayer
            {
                AttentionNorm = IndexTtsOps.LayerNorm(layers, d), SelfAttention = NewAttention(layers, random),
                FinalNorm = IndexTtsOps.LayerNorm(layers, d), Fc1 = IndexTtsOps.Linear(engine, layers, random, d, c.FeedForward),
                Fc2 = IndexTtsOps.Linear(engine, layers, random, c.FeedForward, d),
            });
        // The release initializes the codebook from k-means of the hidden states (quantize_load_codebook); without one,
        // nn.Embedding's N(0, 1).
        Codebook = IndexTtsOps.Values(layers, IndexTtsOps.Normal(random, c.CodebookSize * d, 1.0));
        EmaCount = Enumerable.Repeat(1.0, c.CodebookSize).ToArray();
        EmaWeight = IndexTtsOps.Value(Codebook).ToArray().Select(v => NumOps.ToDouble(v)).ToArray();
        if (!withTrainingStack) return;
        FinalNorm = IndexTtsOps.LayerNorm(layers, d);
        TokenEmbedding = new TiedEmbeddingLayer<T>(c.Vocabulary, d);
        layers.Add(TokenEmbedding);
        TokenEmbedding.Reinitialize(() => 0.02 * IndexTtsOps.Gaussian(random));
        DecoderPositions = IndexTtsOps.Values(layers, IndexTtsOps.Normal(random, c.MaxTargetPositions * d, 0.02));
        for (int i = 0; i < c.DecoderLayers; i++)
            DecoderLayers.Add(new DecoderLayer
            {
                SelfNorm = IndexTtsOps.LayerNorm(layers, d), SelfAttention = NewAttention(layers, random),
                CrossNorm = IndexTtsOps.LayerNorm(layers, d), CrossAttention = NewAttention(layers, random),
                FinalNorm = IndexTtsOps.LayerNorm(layers, d), Fc1 = IndexTtsOps.Linear(engine, layers, random, d, c.FeedForward),
                Fc2 = IndexTtsOps.Linear(engine, layers, random, c.FeedForward, d),
            });
        DecoderNorm = IndexTtsOps.LayerNorm(layers, d);
    }

    /// <summary>Whisper's sinusoid table <c>[length, channels]</c> (sin then cos, timescales 1 … 10000), with which
    /// <c>_init_weights</c> fills the encoder's positions.</summary>
    private static double[] Sinusoids(int length, int channels)
    {
        int half = channels / 2;
        double increment = Math.Log(10000) / (half - 1);
        var values = new double[length * channels];
        for (int t = 0; t < length; t++)
            for (int i = 0; i < half; i++)
            {
                double angle = t * Math.Exp(-increment * i);
                values[t * channels + i] = Math.Sin(angle);
                values[t * channels + half + i] = Math.Cos(angle);
            }
        return values;
    }

    // ---------------------------------------------------------------- features

    private const int FftSize = 400, Hop = 160, SampleRate = 16000, ChunkSeconds = 30;
    private double[,]? _melBasis;

    /// <summary>The samples the release pads a chunk to a multiple of (conv strides · pooling · hop: 1280).</summary>
    public int PadMultiple => 2 * _c.PoolingKernel * Hop;

    /// <summary>
    /// <c>WhisperFeatureExtractor</c>'s log-mel features of one chunk of 16 kHz audio as <c>extract_speech_token</c>
    /// asks for them: the audio zero-padded to a multiple of <see cref="PadMultiple"/> samples, a centred
    /// reflect-padded 400-point STFT (hop 160, periodic Hann), its power without the last frame, Slaney mel filters over
    /// 0 – 8 kHz, log10 above 1e-10, floored 8 below the maximum, (x + 4) / 4: <c>[mels, frames]</c> and which frames
    /// hold audio (the padding mask taken every 160 samples).
    /// </summary>
    public (Tensor<T> Features, bool[] Valid) Features(IReadOnlyList<double> audio)
    {
        int samples = audio.Count, multiple = PadMultiple;
        if (samples == 0) throw new ArgumentException("The audio is empty.", nameof(audio));
        int length = (samples + multiple - 1) / multiple * multiple, half = FftSize / 2, bins = FftSize / 2 + 1, mels = _c.Mels;
        var x = new double[length];
        for (int i = 0; i < samples; i++) x[i] = audio[i];
        var padded = new double[length + 2 * half];
        for (int i = 0; i < length; i++) padded[half + i] = x[i];
        for (int i = 0; i < half; i++)
        {
            padded[half - 1 - i] = x[i + 1];
            padded[half + length + i] = x[length - 2 - i];
        }
        _melBasis ??= MelFilters(bins, mels, 0, SampleRate / 2.0);
        int frames = length / Hop;
        var window = Enumerable.Range(0, FftSize).Select(n => 0.5 - 0.5 * Math.Cos(2 * Math.PI * n / FftSize)).ToArray();
        var cos = new double[bins, FftSize];
        var sin = new double[bins, FftSize];
        for (int k = 0; k < bins; k++)
            for (int n = 0; n < FftSize; n++)
            {
                double angle = 2 * Math.PI * ((long)k * n % FftSize) / FftSize;
                cos[k, n] = Math.Cos(angle);
                sin[k, n] = Math.Sin(angle);
            }
        var logMel = new double[mels, frames];
        var power = new double[bins];
        double max = double.NegativeInfinity;
        for (int f = 0; f < frames; f++)
        {
            for (int k = 0; k < bins; k++)
            {
                double re = 0, im = 0;
                for (int n = 0; n < FftSize; n++)
                {
                    double v = padded[f * Hop + n] * window[n];
                    re += v * cos[k, n];
                    im -= v * sin[k, n];
                }
                power[k] = re * re + im * im;
            }
            for (int m = 0; m < mels; m++)
            {
                double sum = 0;
                for (int k = 0; k < bins; k++) sum += _melBasis[m, k] * power[k];
                logMel[m, f] = Math.Log10(Math.Max(sum, 1e-10));
                max = Math.Max(max, logMel[m, f]);
            }
        }
        var features = new Tensor<T>(new[] { mels, frames });
        for (int m = 0; m < mels; m++)
            for (int f = 0; f < frames; f++) features[m, f] = NumOps.FromDouble((Math.Max(logMel[m, f], max - 8.0) + 4.0) / 4.0);
        var valid = Enumerable.Range(0, frames).Select(f => f * Hop < samples).ToArray();
        return (features, valid);
    }

    /// <summary>
    /// transformers' <c>mel_filter_bank</c> with Slaney's scale and normalization, as WhisperFeatureExtractor builds it:
    /// <c>[mels, bins]</c> triangles between mel-spaced frequencies (numpy's <c>linspace</c>), each scaled by 2 / its width.
    /// </summary>
    private static double[,] MelFilters(int bins, int mels, double minHz, double maxHz)
    {
        double logStep = 27.0 / Math.Log(6.4);
        double ToMel(double f) => f >= 1000.0 ? 15.0 + Math.Log(f / 1000.0) * logStep : 3.0 * f / 200.0;
        double ToHz(double m) => m >= 15.0 ? 1000.0 * Math.Exp(Math.Log(6.4) / 27.0 * (m - 15.0)) : 200.0 * m / 3.0;
        double[] Linspace(double start, double stop, int count)
        {
            var values = new double[count];
            double step = (stop - start) / (count - 1);
            for (int i = 0; i < count; i++) values[i] = i * step + start;
            values[count - 1] = stop;
            return values;
        }
        var filterHz = Linspace(ToMel(minHz), ToMel(maxHz), mels + 2).Select(ToHz).ToArray();
        var fftHz = Linspace(0, Math.Floor(maxHz), bins);
        var filters = new double[mels, bins];
        for (int m = 0; m < mels; m++)
        {
            double enorm = 2.0 / (filterHz[m + 2] - filterHz[m]);
            for (int b = 0; b < bins; b++)
            {
                double down = -(filterHz[m] - fftHz[b]) / (filterHz[m + 1] - filterHz[m]);
                double up = (filterHz[m + 2] - fftHz[b]) / (filterHz[m + 2] - filterHz[m + 1]);
                filters[m, b] = Math.Max(0, Math.Min(down, up)) * enorm;
            }
        }
        return filters;
    }

    /// <summary>
    /// The speech tokens of 16 kHz audio (<c>extract_speech_token</c>): the audio in 30-second chunks, each chunk's
    /// codes at its valid frames, concatenated.
    /// </summary>
    public int[] Tokenize(IReadOnlyList<double> audio)
    {
        var tokens = new List<int>();
        int chunk = ChunkSeconds * SampleRate;
        for (int start = 0; start < audio.Count; start += chunk)
        {
            var piece = audio.Skip(start).Take(chunk).ToArray();
            var (features, valid) = Features(piece);
            var output = Encode(features, valid);
            for (int i = 0; i < output.Codes.Length; i++)
                if (output.Valid[i]) tokens.Add(output.Codes[i]);
        }
        return tokens.ToArray();
    }

    private Attention NewAttention(List<LayerBase<T>> layers, Random random) => new()
    {
        Query = IndexTtsOps.Linear(_engine, layers, random, _c.Dim, _c.Dim),
        Key = IndexTtsOps.BiasFree(layers, random, _c.Dim, _c.Dim),
        Value = IndexTtsOps.Linear(_engine, layers, random, _c.Dim, _c.Dim),
        Output = IndexTtsOps.Linear(_engine, layers, random, _c.Dim, _c.Dim),
    };

    public GlmSpeechTokenizerConfiguration Configuration => _c;
    public NormedConv1DLayer<T> Conv1 { get; }
    public NormedConv1DLayer<T> Conv2 { get; }
    public BiasFreeLinearLayer<T> Positions { get; }
    public BiasFreeLinearLayer<T> Positions2 { get; }
    public List<EncoderLayer> EncoderLayers { get; } = new();
    public BiasFreeLinearLayer<T> Codebook { get; }
    /// <summary>The EMA statistics of the codebook (<c>ema_count</c>, <c>ema_weight</c>).</summary>
    public double[] EmaCount { get; private set; }
    public double[] EmaWeight { get; private set; }
    /// <summary>EMA updates so far (<c>quantize_ema_count</c>), which times the dead-code restarts.</summary>
    public int EmaUpdates { get; private set; }
    public LayerNormalizationLayer<T>? FinalNorm { get; }
    public TiedEmbeddingLayer<T>? TokenEmbedding { get; }
    public BiasFreeLinearLayer<T>? DecoderPositions { get; }
    public List<DecoderLayer> DecoderLayers { get; } = new();
    public LayerNormalizationLayer<T>? DecoderNorm { get; }

    // ---------------------------------------------------------------- pieces

    private Tensor<T> Drop(Tensor<T> x, bool training, Random? random) =>
        training && _c.Dropout > 0 && random is not null ? T5Seq2Seq<T>.Dropout(_engine, x, _c.Dropout, random) : x;

    private Tensor<T> CausalConv(NormedConv1DLayer<T> conv, Tensor<T> x)
    {
        int channels = x.Shape[1];
        var padded = _engine.TensorConcatenate(new[] { new Tensor<T>(new[] { 1, channels, 2 }), x }, 2);
        return conv.Forward(padded);
    }

    /// <summary>Multi-head attention of queries <c>[L, d]</c> over keys and values <c>[S, d]</c> with an additive mask
    /// <c>[L, S]</c> (Whisper's: the query scaled by head^−½, no key bias).</summary>
    private Tensor<T> Attend(Attention a, Tensor<T> query, Tensor<T> memory, Tensor<T>? mask)
    {
        int heads = _c.Heads, hd = _c.Dim / heads;
        var q = IndexTtsOps.SplitHeads(_engine, a.Query.Forward(query), heads);
        var k = IndexTtsOps.SplitHeads(_engine, a.Key.Forward(memory), heads);
        var v = IndexTtsOps.SplitHeads(_engine, a.Value.Forward(memory), heads);
        return a.Output.Forward(IndexTtsOps.MergeHeads(_engine, IndexTtsOps.Attend(_engine, q, k, v, 1.0 / Math.Sqrt(hd), mask)));
    }

    private Tensor<T> Gelu(Tensor<T> x) => AiDotNet.TextToSpeech.MultiModal.SpeechT5Ops.Gelu(_engine, x);

    /// <summary>The block-causal mask <c>[L, L]</c>: a frame sees every frame of its block and of earlier blocks, and no
    /// padded frame (<c>get_block_causal_attention_mask</c>).</summary>
    private Tensor<T> BlockMask(bool[] valid, int block)
    {
        int length = valid.Length;
        var mask = new Tensor<T>(new[] { length, length });
        var blocked = NumOps.FromDouble(double.NegativeInfinity);
        for (int i = 0; i < length; i++)
        {
            int blockEnd = Math.Min(length, (i / block + 1) * block);
            for (int j = 0; j < length; j++)
                if (j >= blockEnd || !valid[j]) mask[i, j] = blocked;
        }
        return mask;
    }

    private Tensor<T> EncoderBlock(EncoderLayer layer, Tensor<T> x, Tensor<T> mask, bool training, Random? random)
    {
        var normed = layer.AttentionNorm.Forward(x);
        x = _engine.TensorAdd(x, Drop(Attend(layer.SelfAttention, normed, normed, mask), training, random));
        var h = Gelu(layer.Fc1.Forward(layer.FinalNorm.Forward(x)));
        return _engine.TensorAdd(x, Drop(layer.Fc2.Forward(h), training, random));
    }

    // ---------------------------------------------------------------- the encoder

    internal sealed record EncoderOutput(Tensor<T> Hidden, int[] Codes, bool[] Valid, Tensor<T> PreQuantized, Tensor<T> Quantized);

    /// <summary>
    /// The encoder on log-mel features <c>[mels, frames]</c> whose valid frames are <paramref name="validFrames"/> (a
    /// multiple of 8 frames in all, as the release pads to): the codes, and — with the training stack — the states
    /// through the remaining layers and the final norm. <paramref name="training"/> passes the quantized states
    /// straight through to the pre-quantized ones' gradient.
    /// </summary>
    public EncoderOutput Encode(Tensor<T> features, bool[] validFrames, bool training = false, Random? random = null)
    {
        var c = _c;
        int frames = features.Shape[1], d = c.Dim;
        if (frames % (2 * c.PoolingKernel) != 0) throw new ArgumentException($"Pad the features to a multiple of {2 * c.PoolingKernel} frames.", nameof(features));
        var x = Gelu(CausalConv(Conv1, _engine.Reshape(features, new[] { 1, c.Mels, frames })));
        x = Gelu(CausalConv(Conv2, x));
        int length = x.Shape[2];
        if (length > c.MaxSourcePositions) throw new ArgumentException($"The features are longer than {c.MaxSourcePositions} encoder frames.", nameof(features));
        x = _engine.TensorAdd(_engine.TensorTranspose(_engine.Reshape(x, new[] { d, length })),
            _engine.TensorNarrow(_engine.Reshape(IndexTtsOps.Value(Positions), new[] { c.MaxSourcePositions, d }), 0, 0, length));
        x = Drop(x, training, random);
        var valid = Enumerable.Range(0, length).Select(t => validFrames[t * 2]).ToArray();
        var mask = BlockMask(valid, c.CausalBlockSize);
        int[] codes = Array.Empty<int>();
        Tensor<T> pre = x, quantized = x;
        for (int i = 0; i < EncoderLayers.Count; i++)
        {
            x = EncoderBlock(EncoderLayers[i], x, mask, training, random);
            if (i + 1 == c.PoolingPosition)
            {
                int pooled = (x.Shape[0] + c.PoolingKernel - 1) / c.PoolingKernel;
                if (x.Shape[0] % c.PoolingKernel != 0)
                    x = _engine.TensorConcatenate(new[] { x, new Tensor<T>(new[] { pooled * c.PoolingKernel - x.Shape[0], d }) }, 0);
                x = _engine.ReduceMean(_engine.Reshape(x, new[] { pooled, c.PoolingKernel, d }), new[] { 1 }, false);
                valid = Enumerable.Range(0, pooled).Select(t => valid[t * c.PoolingKernel]).ToArray();
                mask = BlockMask(valid, c.CausalBlockSize / c.PoolingKernel);
            }
            if (i + 1 == c.QuantizePosition)
            {
                pre = x;
                (codes, quantized) = Quantize(x);
                // Training: straight-through (h + sg(q − h)); inference: the codebook rows.
                x = training ? _engine.TensorAdd(x, Constant(_engine.TensorSubtract(quantized, x))) : quantized;
                int rows = x.Shape[0];
                x = _engine.TensorAdd(x, _engine.TensorNarrow(_engine.Reshape(IndexTtsOps.Value(Positions2), new[] { Positions2Rows, d }), 0, 0, rows));
            }
        }
        if (FinalNorm is not null && EncoderLayers.Count > c.QuantizePosition) x = FinalNorm.Forward(x);
        return new EncoderOutput(x, codes, valid, pre, quantized);
    }

    private int Positions2Rows => (_c.MaxSourcePositions + _c.PoolingKernel - 1) / _c.PoolingKernel;

    private static Tensor<T> Constant(Tensor<T> x) => new(x.Shape.ToArray(), x.ToVector());

    /// <summary><c>vector_quantize</c>: per row the code minimizing |c|² + |x|² − 2 x·c, and those rows of the codebook.</summary>
    private (int[] Codes, Tensor<T> Quantized) Quantize(Tensor<T> x)
    {
        int rows = x.Shape[0], d = _c.Dim, k = _c.CodebookSize;
        var codebook = _engine.Reshape(IndexTtsOps.Value(Codebook), new[] { k, d });
        var products = _engine.TensorMatMul(Constant(x), _engine.TensorTranspose(Constant(codebook)));
        var cb = codebook.ToArray();
        var xs = x.ToArray();
        var codeSquares = new double[k];
        for (int j = 0; j < k; j++)
            for (int i = 0; i < d; i++) codeSquares[j] += Math.Pow(NumOps.ToDouble(cb[j * d + i]), 2);
        var codes = new int[rows];
        for (int r = 0; r < rows; r++)
        {
            double inputSquare = 0;
            for (int i = 0; i < d; i++) inputSquare += Math.Pow(NumOps.ToDouble(xs[r * d + i]), 2);
            double best = double.PositiveInfinity;
            for (int j = 0; j < k; j++)
            {
                double distance = codeSquares[j] + inputSquare - 2 * NumOps.ToDouble(products[r, j]);
                if (distance < best)
                {
                    best = distance;
                    codes[r] = j;
                }
            }
        }
        var picks = new Tensor<int>(new[] { rows }, new Vector<int>(codes));
        return (codes, _engine.TensorIndexSelect(codebook, picks, 0));
    }

    // ---------------------------------------------------------------- training

    /// <summary><c>mse_loss_with_mask</c>: the per-frame mean squared error over the valid frames' mean.</summary>
    public Tensor<T> MaskedMse(Tensor<T> input, Tensor<T> target, bool[] valid)
    {
        int rows = input.Shape[0], d = input.Shape[1];
        var weights = new Tensor<T>(new[] { rows, d });
        int count = valid.Count(v => v);
        for (int r = 0; r < rows; r++)
            if (valid[r])
                for (int i = 0; i < d; i++) weights[r, i] = NumOps.FromDouble(1.0 / (d * count));
        var diff = _engine.TensorSubtract(input, target);
        return _engine.ReduceSum(_engine.TensorMultiply(_engine.TensorMultiply(diff, diff), weights), null, false);
    }

    /// <summary>The commitment loss of a training forward: LossScale · CommitCoefficient · mse(h, sg(q)) over valid frames.</summary>
    public Tensor<T> CommitmentLoss(EncoderOutput output) =>
        _engine.TensorMultiplyScalar(MaskedMse(output.PreQuantized, Constant(output.Quantized), output.Valid),
            NumOps.FromDouble(_c.LossScale * _c.CommitCoefficient));

    /// <summary>
    /// The EMA codebook update after a training forward (section 3.1, Dhariwal et al.): counts and summed states of each code's
    /// valid frames decayed into <see cref="EmaCount"/> and <see cref="EmaWeight"/> (the counts Laplace-smoothed), the codebook
    /// set to their ratio; every <c>RestartInterval</c> updates, codes whose count fell below decay^interval restart at
    /// valid states <paramref name="pickRestart"/> chooses (<c>random.sample</c> of the frames).
    /// </summary>
    public void UpdateCodebook(EncoderOutput output, Func<int, int, int[]> pickRestart)
    {
        var c = _c;
        int k = c.CodebookSize, d = c.Dim;
        var hidden = output.PreQuantized.ToArray();
        var counts = new double[k];
        var sums = new double[k * d];
        for (int r = 0; r < output.Codes.Length; r++)
        {
            if (!output.Valid[r]) continue;
            int code = output.Codes[r];
            counts[code]++;
            for (int i = 0; i < d; i++) sums[code * d + i] += NumOps.ToDouble(hidden[r * d + i]);
        }
        double decay = c.EmaDecay, total = 0;
        for (int j = 0; j < k; j++)
        {
            EmaCount[j] = EmaCount[j] * decay + (1 - decay) * counts[j];
            total += EmaCount[j];
        }
        for (int j = 0; j < k; j++) EmaCount[j] = (EmaCount[j] + 1e-5) / (total + k * 1e-5) * total;
        for (int i = 0; i < k * d; i++) EmaWeight[i] = EmaWeight[i] * decay + (1 - decay) * sums[i];
        EmaUpdates++;
        if (c.RestartInterval > 0 && EmaUpdates % c.RestartInterval == 0)
        {
            double threshold = Math.Pow(decay, c.RestartInterval);
            var dead = Enumerable.Range(0, k).Where(j => EmaCount[j] < threshold).ToArray();
            var validRows = Enumerable.Range(0, output.Codes.Length).Where(r => output.Valid[r]).ToArray();
            var chosen = pickRestart(validRows.Length, dead.Length);
            for (int n = 0; n < dead.Length; n++)
            {
                int row = validRows[chosen[n]];
                EmaCount[dead[n]] = 1;
                for (int i = 0; i < d; i++) EmaWeight[dead[n] * d + i] = NumOps.ToDouble(hidden[row * d + i]);
            }
        }
        var codebook = new double[k * d];
        for (int j = 0; j < k; j++)
            for (int i = 0; i < d; i++) codebook[j * d + i] = EmaWeight[j * d + i] / EmaCount[j];
        TorchParameters.Linear(Codebook, 1, k * d, codebook);
    }

    /// <summary>
    /// Whisper's decoder logits <c>[L, V]</c> of decoder input ids over the encoder states (learned positions, causal
    /// self-attention, cross-attention to the valid encoder frames <paramref name="encoderValid"/>, the final norm and the
    /// embedding-tied projection).
    /// </summary>
    public Tensor<T> DecoderLogits(IReadOnlyList<int> ids, Tensor<T> encoder, bool[] encoderValid, bool training = false, Random? random = null)
    {
        var embedding = TokenEmbedding ?? throw new InvalidOperationException("The decoder exists with the training stack only.");
        int length = ids.Count, d = _c.Dim;
        if (length > _c.MaxTargetPositions) throw new ArgumentException($"At most {_c.MaxTargetPositions} decoder tokens.", nameof(ids));
        var x = _engine.TensorAdd(embedding.Forward(IndexTtsOps.Ids<T>(ids)),
            _engine.TensorNarrow(_engine.Reshape(IndexTtsOps.Value(DecoderPositions!), new[] { _c.MaxTargetPositions, d }), 0, 0, length));
        x = Drop(x, training, random);
        var causal = new Tensor<T>(new[] { length, length });
        var blocked = NumOps.FromDouble(double.NegativeInfinity);
        for (int i = 0; i < length; i++)
            for (int j = i + 1; j < length; j++) causal[i, j] = blocked;
        // WhisperVQModel passes the encoder's padding mask, downsampled by the convolutions and the pooling, to the
        // cross-attention.
        int sources = encoder.Shape[0];
        var cross = new Tensor<T>(new[] { length, sources });
        for (int i = 0; i < length; i++)
            for (int j = 0; j < sources; j++)
                if (!encoderValid[j]) cross[i, j] = blocked;
        foreach (var layer in DecoderLayers)
        {
            var normed = layer.SelfNorm.Forward(x);
            x = _engine.TensorAdd(x, Drop(Attend(layer.SelfAttention, normed, normed, causal), training, random));
            x = _engine.TensorAdd(x, Drop(Attend(layer.CrossAttention, layer.CrossNorm.Forward(x), encoder, cross), training, random));
            var h = Gelu(layer.Fc1.Forward(layer.FinalNorm.Forward(x)));
            x = _engine.TensorAdd(x, Drop(layer.Fc2.Forward(h), training, random));
        }
        return embedding.Logits(DecoderNorm!.Forward(x));
    }

    /// <summary>The ASR cross-entropy (the mean over the label positions) of decoder input ids and their labels.</summary>
    public Tensor<T> AsrLoss(IReadOnlyList<int> decoderInput, IReadOnlyList<int> labels, Tensor<T> encoder, bool[] encoderValid, bool training,
        Random? random)
    {
        var logits = DecoderLogits(decoderInput, encoder, encoderValid, training, random);
        int length = labels.Count, v = _c.Vocabulary;
        var logp = _engine.Reshape(_engine.TensorLogSoftmax(logits, 1), new[] { length * v, 1 });
        var picks = Enumerable.Range(0, length).Where(i => labels[i] >= 0).Select(i => i * v + labels[i]).ToArray();
        var nll = _engine.TensorNegate(_engine.ReduceSum(_engine.TensorIndexSelect(logp, new Tensor<int>(new[] { picks.Length }, new Vector<int>(picks)), 0), null, false));
        return _engine.TensorMultiplyScalar(nll, NumOps.FromDouble(1.0 / picks.Length));
    }

    // ---------------------------------------------------------------- weights

    private static void LoadAttention(IEngine engine, Func<string, int[], double[]> read, Attention a, string p, int d)
    {
        IndexTtsOps.LoadLinear(engine, read, a.Query, p + "q_proj", d, d);
        IndexTtsOps.LoadLinear(read, a.Key, p + "k_proj", d, d);
        IndexTtsOps.LoadLinear(engine, read, a.Value, p + "v_proj", d, d);
        IndexTtsOps.LoadLinear(engine, read, a.Output, p + "out_proj", d, d);
    }

    /// <summary>
    /// Loads the release's <c>WhisperVQEncoder</c> state (<c>conv1</c>, <c>conv2</c>, <c>embed_positions</c>,
    /// <c>layers.N</c>, <c>codebook</c>, <c>embed_positions2</c>, and <c>ema_count</c> / <c>ema_weight</c> when present),
    /// under <paramref name="prefix"/>; the encoder layers built.
    /// </summary>
    public void LoadEncoder(Func<string, int[], double[]> read, Func<string, bool> has, string prefix = "")
    {
        var c = _c;
        int d = c.Dim;
        Conv1.LoadTorchWeights(read(prefix + "conv1.weight", new[] { d, c.Mels, 3 }), null, read(prefix + "conv1.bias", new[] { d }));
        Conv2.LoadTorchWeights(read(prefix + "conv2.weight", new[] { d, d, 3 }), null, read(prefix + "conv2.bias", new[] { d }));
        TorchParameters.Linear(Positions, 1, c.MaxSourcePositions * d, read(prefix + "embed_positions.weight", new[] { c.MaxSourcePositions, d }));
        for (int i = 0; i < EncoderLayers.Count; i++)
        {
            var l = EncoderLayers[i];
            string p = $"{prefix}layers.{i}.";
            IndexTtsOps.LoadNorm(_engine, read, l.AttentionNorm, p + "self_attn_layer_norm", d);
            LoadAttention(_engine, read, l.SelfAttention, p + "self_attn.", d);
            IndexTtsOps.LoadNorm(_engine, read, l.FinalNorm, p + "final_layer_norm", d);
            IndexTtsOps.LoadLinear(_engine, read, l.Fc1, p + "fc1", d, c.FeedForward);
            IndexTtsOps.LoadLinear(_engine, read, l.Fc2, p + "fc2", c.FeedForward, d);
        }
        if (FinalNorm is not null && has(prefix + "layer_norm.weight")) IndexTtsOps.LoadNorm(_engine, read, FinalNorm, prefix + "layer_norm", d);
        if (has(prefix + "codebook.weight"))
        {
            var codebook = read(prefix + "codebook.weight", new[] { c.CodebookSize, d });
            TorchParameters.Linear(Codebook, 1, codebook.Length, codebook);
            TorchParameters.Linear(Positions2, 1, Positions2Rows * d, read(prefix + "embed_positions2.weight", new[] { Positions2Rows, d }));
            EmaWeight = has(prefix + "ema_weight") ? read(prefix + "ema_weight", new[] { c.CodebookSize, d }) : (double[])codebook.Clone();
            EmaCount = has(prefix + "ema_count") ? read(prefix + "ema_count", new[] { c.CodebookSize }) : Enumerable.Repeat(1.0, c.CodebookSize).ToArray();
        }
    }

    /// <summary>Loads a Hugging Face Whisper decoder (<c>embed_tokens</c>, <c>embed_positions</c>, <c>layers.N</c>,
    /// <c>layer_norm</c>) under <paramref name="prefix"/>, such as whisper-large-v3's <c>model.decoder.</c>.</summary>
    public void LoadDecoder(Func<string, int[], double[]> read, string prefix)
    {
        var c = _c;
        int d = c.Dim;
        var embedding = TokenEmbedding ?? throw new InvalidOperationException("The decoder exists with the training stack only.");
        embedding.LoadTable(read(prefix + "embed_tokens.weight", new[] { c.Vocabulary, d }));
        TorchParameters.Linear(DecoderPositions!, 1, c.MaxTargetPositions * d, read(prefix + "embed_positions.weight", new[] { c.MaxTargetPositions, d }));
        for (int i = 0; i < DecoderLayers.Count; i++)
        {
            var l = DecoderLayers[i];
            string p = $"{prefix}layers.{i}.";
            IndexTtsOps.LoadNorm(_engine, read, l.SelfNorm, p + "self_attn_layer_norm", d);
            LoadAttention(_engine, read, l.SelfAttention, p + "self_attn.", d);
            IndexTtsOps.LoadNorm(_engine, read, l.CrossNorm, p + "encoder_attn_layer_norm", d);
            LoadAttention(_engine, read, l.CrossAttention, p + "encoder_attn.", d);
            IndexTtsOps.LoadNorm(_engine, read, l.FinalNorm, p + "final_layer_norm", d);
            IndexTtsOps.LoadLinear(_engine, read, l.Fc1, p + "fc1", d, c.FeedForward);
            IndexTtsOps.LoadLinear(_engine, read, l.Fc2, p + "fc2", c.FeedForward, d);
        }
        IndexTtsOps.LoadNorm(_engine, read, DecoderNorm!, prefix + "layer_norm", d);
    }
}
