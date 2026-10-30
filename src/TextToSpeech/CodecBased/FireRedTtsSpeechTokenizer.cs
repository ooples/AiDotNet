using AiDotNet.Audio.Codecs;
using AiDotNet.Audio.Foundations;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.Latest;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>The sizes of FireRedTTS's semantic-aware speech tokenizer (the 1S release's <c>semantic_tokenizer</c> config,
/// with the paper's decoders).</summary>
/// <param name="DecoderStrides">Each decoder's upsampling strides (the config's <c>upsample_scales</c>): the first decodes
/// the quantized sequence to the SSL features, the second its hidden state to the mel.</param>
/// <param name="EmaDecay">The codebooks' EMA decay (DPQ-VAE §4.1.2: 0.9 for PQ-VAE; the 1S code's default 0.99).</param>
/// <param name="DeadCodeThreshold">The cluster size below which a code is replaced by a sample of the batch
/// (<c>threshold_ema_dead_code</c>, the 1S code's default 1.0).</param>
internal sealed record FireRedTtsTokenizerConfiguration(int SslDim, int ModelDim, int[] DownsampleStrides, int CodebookSize, int CodebookDim,
    int Codebooks, int MelChannels, int[][] DecoderStrides, int SamplesPerToken, int SslHop, double EmaDecay, double DeadCodeThreshold,
    double CommitmentWeight, bool NormalizeWave);

/// <summary>The sizes of a product quantizer (the 1S release's and SoCodec's <c>VectorQuantization</c> with a
/// <c>MultiHeadEuclideanCodebook</c>).</summary>
/// <param name="Groups">Streams the codebooks form (<c>n_groups</c>): each group's token is its codebooks' indices as
/// digits (the first the low digit).</param>
/// <param name="Ordered">SoCodec's ordered group dropout in training (<c>ordered</c>): per sequence a rate r ~ U(0, 1), and
/// group g kept when g / groups ≤ r, so the first groups carry the most information.</param>
internal sealed record FireRedTtsQuantizerConfiguration(int ModelDim, int CodebookSize, int CodebookDim, int Codebooks, int Groups,
    double EmaDecay, double DeadCodeThreshold, double CommitmentWeight, bool Ordered = false);

/// <summary>
/// A product quantizer (DPQ-VAE, Guo et al. 2024 [24]; the 1S release's and SoCodec's <c>VectorQuantization</c> with a
/// <c>MultiHeadEuclideanCodebook</c>): a projection to the codebooks' concatenated width, split into one chunk per
/// codebook, each replaced by its nearest codeword, and projected back. Codebooks form groups (streams); a group's token
/// is Σⱼ indexⱼ · Kʲ over its codebooks. Codebooks learn by exponential moving averages, not gradients; codewords whose
/// average cluster size falls below the dead-code threshold are replaced by encodings of the batch.
/// </summary>
internal sealed class FireRedTtsProductQuantizer<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly double[][] _clusterSize;
    private readonly double[][] _embedAverage;

    public FireRedTtsProductQuantizer(IEngine engine, List<LayerBase<T>> layers, FireRedTtsQuantizerConfiguration configuration, Random random)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        if (c.Groups < 1 || c.Codebooks % c.Groups != 0) throw new ArgumentException("The groups must divide the codebooks.", nameof(configuration));
        int width = c.Codebooks * c.CodebookDim;
        ProjectIn = IndexTtsOps.Linear(engine, layers, random, c.ModelDim, width);
        ProjectOut = IndexTtsOps.Linear(engine, layers, random, width, c.ModelDim);
        // uniform_init: kaiming_uniform_ on [K, dim], bound √(6 / dim); embed_avg starts as the codebook, cluster sizes at 0.
        double bound = Math.Sqrt(6.0 / c.CodebookDim);
        Codebooks = new Tensor<T>[c.Codebooks];
        _clusterSize = new double[c.Codebooks][];
        _embedAverage = new double[c.Codebooks][];
        for (int b = 0; b < c.Codebooks; b++)
        {
            var values = Enumerable.Range(0, c.CodebookSize * c.CodebookDim).Select(_ => bound * (2 * random.NextDouble() - 1)).ToArray();
            Codebooks[b] = new Tensor<T>(new[] { c.CodebookSize, c.CodebookDim }, new Vector<T>(values.Select(v => NumOps.FromDouble(v)).ToArray()));
            _clusterSize[b] = new double[c.CodebookSize];
            _embedAverage[b] = values;
        }
    }

    public FireRedTtsQuantizerConfiguration Configuration { get; }
    public DenseLayer<T> ProjectIn { get; }
    public DenseLayer<T> ProjectOut { get; }

    /// <summary>The codebooks <c>[K, dim]</c> (the flow decoder's input embedding reads the tokenizer's too).</summary>
    public Tensor<T>[] Codebooks { get; }

    private int PerGroup => Configuration.Codebooks / Configuration.Groups;

    /// <summary>A group's token vocabulary: K to the power of its codebooks.</summary>
    public int GroupVocabulary => (int)Math.Pow(Configuration.CodebookSize, PerGroup);

    /// <summary>The projected encodings <c>[n, Codebooks · dim]</c>.</summary>
    public Tensor<T> Project(Tensor<T> x) => ProjectIn.Forward(x);

    /// <summary>The tokens <c>[n, Groups]</c> of projected encodings: each chunk's nearest codeword (the first on ties, as
    /// <c>max</c> of the negated distance).</summary>
    public int[,] GroupTokens(Tensor<T> projected)
    {
        var c = Configuration;
        int n = projected.Shape[0];
        var tokens = new int[n, c.Groups];
        for (int b = 0; b < c.Codebooks; b++)
        {
            int g = b / PerGroup, scale = (int)Math.Pow(c.CodebookSize, b % PerGroup);
            var index = Nearest(b, Chunk(projected, b));
            for (int t = 0; t < n; t++) tokens[t, g] += index[t] * scale;
        }
        return tokens;
    }

    /// <summary>The tokens of a one-group quantizer.</summary>
    public int[] Tokens(Tensor<T> projected)
    {
        if (Configuration.Groups != 1) throw new InvalidOperationException("This quantizer makes several streams; use GroupTokens.");
        var tokens = GroupTokens(projected);
        return Enumerable.Range(0, tokens.GetLength(0)).Select(t => tokens[t, 0]).ToArray();
    }

    private Tensor<T> Chunk(Tensor<T> projected, int b) => _engine.TensorNarrow(projected, 1, b * Configuration.CodebookDim, Configuration.CodebookDim);

    // -(|x|² − 2 x·e + |e|²) maximized: the nearest codeword of each row.
    private int[] Nearest(int b, Tensor<T> chunk)
    {
        var c = Configuration;
        int n = chunk.Shape[0], k = c.CodebookSize, dim = c.CodebookDim;
        var book = Codebooks[b];
        var index = new int[n];
        var e2 = new double[k];
        for (int j = 0; j < k; j++)
            for (int i = 0; i < dim; i++)
            {
                double v = NumOps.ToDouble(book[j, i]);
                e2[j] += v * v;
            }
        for (int t = 0; t < n; t++)
        {
            double x2 = 0;
            for (int i = 0; i < dim; i++)
            {
                double v = NumOps.ToDouble(chunk[t, i]);
                x2 += v * v;
            }
            double best = double.NegativeInfinity;
            for (int j = 0; j < k; j++)
            {
                double dot = 0;
                for (int i = 0; i < dim; i++) dot += NumOps.ToDouble(chunk[t, i]) * NumOps.ToDouble(book[j, i]);
                double score = -(x2 - 2 * dot + e2[j]);
                if (score > best)
                {
                    best = score;
                    index[t] = j;
                }
            }
        }
        return index;
    }

    private int Index(int[,] tokens, int t, int b) => tokens[t, b / PerGroup] / (int)Math.Pow(Configuration.CodebookSize, b % PerGroup) % Configuration.CodebookSize;

    /// <summary>The codewords <c>[n, Codebooks · dim]</c> of tokens <c>[n, Groups]</c>.</summary>
    public Tensor<T> Lookup(int[,] tokens)
    {
        var c = Configuration;
        int n = tokens.GetLength(0);
        var parts = new Tensor<T>[c.Codebooks];
        for (int b = 0; b < c.Codebooks; b++)
        {
            var index = new Tensor<int>(new[] { n }, new Vector<int>(Enumerable.Range(0, n).Select(t => Index(tokens, t, b)).ToArray()));
            parts[b] = _engine.TensorIndexSelect(Codebooks[b], index, 0);
        }
        return _engine.TensorConcatenate(parts, 1);
    }

    /// <summary>The codewords of a one-group quantizer's tokens.</summary>
    public Tensor<T> Lookup(IReadOnlyList<int> tokens)
    {
        var grid = new int[tokens.Count, 1];
        for (int t = 0; t < tokens.Count; t++) grid[t, 0] = tokens[t];
        return Lookup(grid);
    }

    /// <summary>The quantized sequence <c>[n, ModelDim]</c> of tokens <c>[n, Groups]</c> (<c>decode</c>).</summary>
    public Tensor<T> Decode(int[,] tokens) => ProjectOut.Forward(Lookup(tokens));

    /// <summary>
    /// The quantizer's training pass on encodings <c>[n, ModelDim]</c>: the tokens <c>[n, Groups]</c>; the quantized
    /// sequence projected back (straight-through, so gradients reach the encoder; with ordered dropout in training, the
    /// dropped groups zeroed); the pre-quantized sequence projected back (dual decoding's <c>return_pre_quant</c>); and the
    /// commitment loss <c>‖E − sg(Z)‖²</c> averaged over the codebooks. With <paramref name="update"/> the codebooks then
    /// take one EMA step (dead codes replaced first).
    /// </summary>
    public (int[,] Tokens, Tensor<T> Quantized, Tensor<T> PreQuantized, Tensor<T> Commitment) Forward(Tensor<T> encoded, bool update, Random random)
    {
        var c = Configuration;
        var projected = Project(encoded);
        int n = projected.Shape[0];
        var tokens = GroupTokens(projected);
        var codewords = Lookup(tokens);
        var straight = _engine.TensorAdd(projected, _engine.TensorSubtract(StopGradient(codewords), StopGradient(projected)));
        if (update && c.Ordered && c.Groups > 1)
        {
            double rate = random.NextDouble();
            int width = PerGroup * c.CodebookDim;
            var keep = new Tensor<T>(new[] { 1, c.Codebooks * c.CodebookDim });
            for (int g = 0; g < c.Groups; g++)
                if ((double)g / c.Groups <= rate)
                    for (int i = 0; i < width; i++) keep[0, g * width + i] = NumOps.One;
            straight = _engine.TensorMultiply(straight, _engine.TensorTile(keep, new[] { n, 1 }));
        }
        // F.mse_loss(quantize.detach(), x) per codebook, averaged over them: the same as one mean over every chunk.
        var difference = _engine.TensorSubtract(projected, StopGradient(codewords));
        var commitment = _engine.TensorMultiplyScalar(_engine.ReduceSum(_engine.TensorMultiply(difference, difference), null),
            NumOps.FromDouble(c.CommitmentWeight / ((double)n * c.Codebooks * c.CodebookDim)));
        if (update) Update(projected, tokens, random);
        return (tokens, ProjectOut.Forward(straight), ProjectOut.Forward(projected), commitment);
    }

    private Tensor<T> StopGradient(Tensor<T> x)
    {
        using var _ = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        return x.Clone();
    }

    // EuclideanCodebook.forward's update: expire_codes_ (codes below the threshold replaced by batch samples), then the EMA
    // of cluster sizes and sums, and the codebook as their Laplace-smoothed ratio.
    private void Update(Tensor<T> projected, int[,] tokens, Random random)
    {
        var c = Configuration;
        int n = projected.Shape[0], k = c.CodebookSize, dim = c.CodebookDim;
        for (int b = 0; b < c.Codebooks; b++)
        {
            var x = new double[n, dim];
            for (int t = 0; t < n; t++)
                for (int i = 0; i < dim; i++) x[t, i] = NumOps.ToDouble(projected[t, b * dim + i]);
            var index = Enumerable.Range(0, n).Select(t => Index(tokens, t, b)).ToArray();
            var size = _clusterSize[b];
            var average = _embedAverage[b];
            var book = Codebooks[b];
            if (c.DeadCodeThreshold > 0)
            {
                double threshold = c.DeadCodeThreshold < 1 ? size.Sum() * c.DeadCodeThreshold : c.DeadCodeThreshold;
                if (size.Any(s => s < threshold))
                {
                    double reset = Math.Min(c.DeadCodeThreshold + 1, c.DeadCodeThreshold * 1.1);
                    // sample_vectors: a random permutation's first K rows, or K draws with replacement from fewer rows.
                    int[] picks = n >= k
                        ? Enumerable.Range(0, n).OrderBy(_ => random.Next()).Take(k).ToArray()
                        : Enumerable.Range(0, k).Select(_ => random.Next(n)).ToArray();
                    for (int j = 0; j < k; j++)
                    {
                        if (size[j] >= threshold) continue;
                        for (int i = 0; i < dim; i++)
                        {
                            double v = x[picks[j], i];
                            book[j, i] = NumOps.FromDouble(v);
                            average[j * dim + i] = v * reset;
                        }
                        size[j] = reset;
                    }
                }
            }
            var count = new double[k];
            var sum = new double[k * dim];
            for (int t = 0; t < n; t++)
            {
                count[index[t]] += 1;
                for (int i = 0; i < dim; i++) sum[index[t] * dim + i] += x[t, i];
            }
            double decay = c.EmaDecay;
            for (int j = 0; j < k; j++) size[j] = size[j] * decay + count[j] * (1 - decay);
            for (int i = 0; i < average.Length; i++) average[i] = average[i] * decay + sum[i] * (1 - decay);
            const double epsilon = 1e-5;
            double total = size.Sum();
            for (int j = 0; j < k; j++)
            {
                double smoothed = (size[j] + epsilon) / (total + k * epsilon) * total;
                for (int i = 0; i < dim; i++) book[j, i] = NumOps.FromDouble(average[j * dim + i] / smoothed);
            }
            _engine.InvalidatePersistentTensor(book);
        }
    }

    /// <summary>Loads <c>{prefix}project_in/out</c> and each codebook's <c>embed</c>, <c>embed_avg</c> and
    /// <c>cluster_size</c> (<c>{prefix}_codebook.codebooks.{b}.*</c>).</summary>
    public void Load(Func<string, int[], double[]> read, Func<string, bool> has, string prefix)
    {
        var c = Configuration;
        int width = c.Codebooks * c.CodebookDim;
        IndexTtsOps.LoadLinear(_engine, read, ProjectIn, prefix + "project_in", c.ModelDim, width);
        IndexTtsOps.LoadLinear(_engine, read, ProjectOut, prefix + "project_out", width, c.ModelDim);
        for (int b = 0; b < c.Codebooks; b++)
        {
            string p = $"{prefix}_codebook.codebooks.{b}.";
            var embed = read(p + "embed", new[] { c.CodebookSize, c.CodebookDim });
            LoadCodebook(b, embed);
            // The running averages are training state; without them the codebook restarts its EMA from the codewords.
            Array.Copy(has(p + "embed_avg") ? read(p + "embed_avg", new[] { c.CodebookSize, c.CodebookDim }) : embed, _embedAverage[b], _embedAverage[b].Length);
            if (has(p + "cluster_size")) Array.Copy(read(p + "cluster_size", new[] { c.CodebookSize }), _clusterSize[b], c.CodebookSize);
        }
    }

    /// <summary>Sets a codebook's codewords (the release's frozen flow embedding carries the same values).</summary>
    public void LoadCodebook(int b, double[] values)
    {
        var book = Codebooks[b];
        if (values.Length != book.Length) throw new ArgumentException($"Expected {book.Length} values.", nameof(values));
        for (int i = 0; i < values.Length; i++) book.SetFlat(i, NumOps.FromDouble(values[i]));
        _engine.InvalidatePersistentTensor(book);
    }
}

/// <summary>
/// FireRedTTS's semantic-aware speech tokenizer (SAST, §3.1): HuBERT features of 16 kHz speech, a convolutional encoder
/// that halves their rate to 40 ms, and the product quantizer's 16,384 tokens (the 1S release's <c>SemanticVQVAE</c>,
/// whose codebooks are FireRedTTS-1's); for training, the paper's decoder: the quantized sequence plus the acoustic
/// encoder's global embedding (the shared ECAPA-TDNN on the "Clip&amp;Shuffle" mel), upsampled by ResNet blocks with
/// transposed convolutions to reconstruct the SSL features, and from that decoder's hidden state the mel.
/// </summary>
/// <remarks>
/// <para>The encoder is the release's (EnCodec's SEANet blocks: weight-normalized, reflect-padded streamable
/// convolutions, Swish, residual units, strided convolutions, a half-width bidirectional LSTM with a skip). HuBERT (the
/// release's <c>TencentGameMate/chinese-hubert-large</c>, frozen) reads the waveform zero-padded by 3,200 samples and
/// layer-normalized, masking the padded frames, as the release's <c>HuBERT._extract</c> does.</para>
/// <para>The decoders are not released; their layout is the release's unused <c>Decoder</c> class (a convolution and the
/// LSTM, then per stride a transposed convolution, Swish and two residual units, the speaker embedding added before
/// each, and an output convolution) with the release config's two stride lists. Training (eq. 1 with DPQ-VAE's dual
/// decoding, eq. 4): λ_s ‖SSL − SSL̂‖² + λ_a ‖mel − mel̂‖², the same for the pre-quantized sequence weighted by γ (1 to 0.1
/// between steps 20k and 80k), and λ_vq times the commitment loss; the codebooks learn by EMA.</para>
/// </remarks>
internal sealed class FireRedTtsSpeechTokenizer<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    internal sealed class Decoder
    {
        public StreamableConv1d<T> In = null!;
        public SeanetLstm<T> Lstm = null!;
        public List<(StreamableConvTranspose1d<T> Up, (StreamableConv1d<T> A, StreamableConv1d<T> B)[] Units)> Blocks = new();
        public StreamableConv1d<T> Out = null!;
    }

    public FireRedTtsSpeechTokenizer(IEngine engine, List<LayerBase<T>> hubertLayers, List<LayerBase<T>> layers,
        FireRedTtsTokenizerConfiguration configuration, HubertConfiguration hubert, Random random)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        Hubert = new HubertNetwork<T>(engine, hubertLayers, hubert);
        Hubert.InitializeLikeReference(random);
        StreamableConv1d<T> Conv(int inputs, int outputs, int kernel, int stride = 1)
        {
            var conv = new StreamableConv1d<T>(engine, inputs, outputs, kernel, stride, 1, false, true, ConvolutionNormalization.Weight, layers);
            conv.Conv.LoadTorchWeights(IndexTtsOps.Uniform(random, outputs * inputs * kernel, inputs * kernel), null,
                IndexTtsOps.Uniform(random, outputs, inputs * kernel));
            return conv;
        }
        (StreamableConv1d<T>, StreamableConv1d<T>) Unit(int dim) => (Conv(dim, dim / 2, 3), Conv(dim / 2, dim, 3));
        int d = c.ModelDim;
        EncoderIn = Conv(c.SslDim, d, 3);
        foreach (int stride in c.DownsampleStrides)
            EncoderBlocks.Add((new[] { Unit(d), Unit(d) }, Conv(d, d, 2 * stride, stride)));
        EncoderOut = Conv(d, d, 3);
        EncoderLstm = new SeanetLstm<T>(engine, d, 1, true, layers, halfWidth: true);
        Quantizer = new FireRedTtsProductQuantizer<T>(engine, layers,
            new FireRedTtsQuantizerConfiguration(d, c.CodebookSize, c.CodebookDim, c.Codebooks, 1, c.EmaDecay, c.DeadCodeThreshold, c.CommitmentWeight), random);
        if (c.DecoderStrides.Length != 2) throw new ArgumentException("SAST has an SSL decoder and a mel decoder.", nameof(configuration));
        Decoder Build(int outputs, int[] strides)
        {
            var decoder = new Decoder { In = Conv(d, d, 3), Lstm = new SeanetLstm<T>(engine, d, 1, true, layers, halfWidth: true) };
            foreach (int stride in strides)
            {
                var up = new StreamableConvTranspose1d<T>(engine, d, d, 2 * stride, stride, false, 1.0, ConvolutionNormalization.Weight, layers);
                int fan = d * 2 * stride;
                up.Conv.LoadTorchWeights(IndexTtsOps.Uniform(random, d * d * 2 * stride, fan), null, IndexTtsOps.Uniform(random, d, fan));
                decoder.Blocks.Add((up, new[] { Unit(d), Unit(d) }));
            }
            decoder.Out = Conv(d, outputs, 3);
            return decoder;
        }
        SslDecoder = Build(c.SslDim, c.DecoderStrides[0]);
        MelDecoder = Build(c.MelChannels, c.DecoderStrides[1]);
    }

    public FireRedTtsTokenizerConfiguration Configuration { get; }
    public HubertNetwork<T> Hubert { get; }
    public StreamableConv1d<T> EncoderIn { get; }
    public List<((StreamableConv1d<T> A, StreamableConv1d<T> B)[] Units, StreamableConv1d<T> Down)> EncoderBlocks { get; } = new();
    public StreamableConv1d<T> EncoderOut { get; }
    public SeanetLstm<T> EncoderLstm { get; }
    public FireRedTtsProductQuantizer<T> Quantizer { get; }
    public Decoder SslDecoder { get; }
    public Decoder MelDecoder { get; }

    /// <summary>The total downsampling of the encoder (the SSL frames per token).</summary>
    public int Downsampling => Configuration.DownsampleStrides.Aggregate(1, (a, b) => a * b);

    /// <summary>The waveform zero-padded to whole tokens (<c>n_samples_per_token</c>).</summary>
    public double[] PadToTokens(double[] wave)
    {
        int per = Configuration.SamplesPerToken, extra = wave.Length % per;
        if (extra == 0) return wave;
        var padded = new double[wave.Length + per - extra];
        Array.Copy(wave, padded, wave.Length);
        return padded;
    }

    /// <summary>
    /// The release's <c>HuBERT._extract</c>: the SSL features <c>[⌈S / 320⌉, SslDim]</c> of a 16 kHz waveform
    /// <c>[S]</c>: zero-padded by 3,200, layer-normalized as a whole (when the checkpoint's task normalizes), the last
    /// layer's output after the encoder's final norm with the padded frames masked, cut to ⌈S / 320⌉ frames.
    /// </summary>
    public Tensor<T> SslFeatures(double[] wave)
    {
        var c = Configuration;
        const int padding = 3200;
        var padded = new double[wave.Length + padding];
        Array.Copy(wave, padded, wave.Length);
        if (c.NormalizeWave)
        {
            double mean = padded.Average(), variance = padded.Sum(v => (v - mean) * (v - mean)) / padded.Length, std = Math.Sqrt(variance + 1e-5);
            for (int i = 0; i < padded.Length; i++) padded[i] = (padded[i] - mean) / std;
        }
        int frames = (int)Math.Ceiling(wave.Length / (double)c.SslHop);
        var mask = new bool[frames + 9];
        for (int i = frames; i < mask.Length; i++) mask[i] = true;
        var audio = new Tensor<T>(new[] { padded.Length }, new Vector<T>(padded.Select(v => NumOps.FromDouble(v)).ToArray()));
        var states = Hubert.Forward(audio, Hubert.EncoderLayers.Count, mask);
        var last = Hubert.EncoderNorm.Forward(states[states.Count - 1]);
        return _engine.TensorNarrow(last, 0, 0, Math.Min(frames, last.Shape[0]));
    }

    // [L, C] through a streamable convolution over L.
    private Tensor<T> Run(Func<Tensor<T>, Tensor<T>> conv, Tensor<T> x)
    {
        int length = x.Shape[0], channels = x.Shape[1];
        var y = conv(_engine.Reshape(_engine.TensorTranspose(x), new[] { 1, channels, length }));
        return _engine.TensorTranspose(_engine.Reshape(y, new[] { y.Shape[1], y.Shape[2] }));
    }

    private Tensor<T> ResidualUnit((StreamableConv1d<T> A, StreamableConv1d<T> B) unit, Tensor<T> x) =>
        _engine.TensorAdd(x, Run(unit.B.Forward, _engine.Swish(Run(unit.A.Forward, x))));

    /// <summary>The encoder: SSL features <c>[L, SslDim]</c> to encodings <c>[⌈L / downsampling⌉, ModelDim]</c>.</summary>
    public Tensor<T> Encode(Tensor<T> ssl)
    {
        var x = _engine.Swish(Run(EncoderIn.Forward, ssl));
        foreach (var (units, down) in EncoderBlocks)
        {
            x = _engine.Swish(ResidualUnit(units[0], x));
            x = ResidualUnit(units[1], x);
            x = _engine.Swish(Run(down.Forward, _engine.Swish(x)));
        }
        x = Run(EncoderOut.Forward, x);
        return Run(EncoderLstm.Forward, x);
    }

    /// <summary><c>extract_speech_tokens</c>: the tokens of a 16 kHz waveform (padded to whole tokens first).</summary>
    public int[] Tokenize(double[] wave)
    {
        var padded = PadToTokens(wave);
        var encoded = Encode(SslFeatures(padded));
        int tokens = (int)Math.Ceiling(padded.Length / (double)Configuration.SslHop / Downsampling);
        var ids = Quantizer.Tokens(Quantizer.Project(encoded));
        return ids.Take(Math.Min(tokens, ids.Length)).ToArray();
    }

    /// <summary>A decoder's output <c>[L·∏strides, outputs]</c> and hidden state <c>[L·∏strides, ModelDim]</c> of a sequence
    /// <c>[L, ModelDim]</c>, the speaker embedding <c>[ModelDim]</c> added before each upsampling block.</summary>
    public (Tensor<T> Output, Tensor<T> Hidden) Decode(Decoder decoder, Tensor<T> x, Tensor<T> speaker)
    {
        int d = Configuration.ModelDim;
        var h = Run(decoder.Lstm.Forward, Run(decoder.In.Forward, x));
        var g = _engine.Reshape(speaker, new[] { 1, d });
        foreach (var (up, units) in decoder.Blocks)
        {
            h = _engine.TensorAdd(h, _engine.TensorTile(g, new[] { h.Shape[0], 1 }));
            h = _engine.Swish(Run(up.Forward, h));
            h = _engine.Swish(ResidualUnit(units[0], h));
            h = _engine.Swish(ResidualUnit(units[1], h));
        }
        return (Run(decoder.Out.Forward, h), h);
    }

    /// <summary>The SSL and mel reconstructions of a sequence (quantized or pre-quantized) <c>[n, ModelDim]</c>.</summary>
    public (Tensor<T> Ssl, Tensor<T> Mel) Reconstruct(Tensor<T> sequence, Tensor<T> speaker)
    {
        var (ssl, hidden) = Decode(SslDecoder, sequence, speaker);
        var (mel, _) = Decode(MelDecoder, hidden, speaker);
        return (ssl, mel);
    }

    /// <summary>
    /// "Clip&amp;Shuffle" (§3.1): a random segment of 25 % to 75 % of the mel's frames, cut into slices of
    /// <paramref name="sliceFrames"/> (one second) and put back in a random order.
    /// </summary>
    public Tensor<T> ClipAndShuffle(Tensor<T> mel, int sliceFrames, Random random)
    {
        int frames = mel.Shape[0];
        int length = Math.Max(1, (int)Math.Round(frames * (0.25 + 0.5 * random.NextDouble())));
        int start = random.Next(frames - length + 1);
        var slices = new List<(int Start, int Length)>();
        for (int s = 0; s < length; s += sliceFrames) slices.Add((start + s, Math.Min(sliceFrames, length - s)));
        var order = slices.OrderBy(_ => random.Next()).ToList();
        return _engine.TensorConcatenate(order.Select(s => _engine.TensorNarrow(mel, 0, s.Start, s.Length)).ToArray(), 0);
    }

    /// <summary>Loads the 1S release's <c>codec.bin</c> names (<c>encoder.layers.*</c>, <c>quantizer.quantizers.0.*</c>);
    /// the speaker encoder and HuBERT load separately.</summary>
    public void Load(Func<string, int[], double[]> read, Func<string, bool> has)
    {
        void Conv(StreamableConv1d<T> conv, string name)
        {
            var shape = conv.Conv.TorchWeightShape;
            conv.Conv.LoadTorchWeights(read(name + ".conv.conv.weight_v", shape), read(name + ".conv.conv.weight_g", new[] { shape[0], 1, 1 }),
                read(name + ".conv.conv.bias", new[] { shape[0] }));
        }
        // encoder.layers: 0 conv, 1 swish, then per stride a block and a swish, then the conv and the LSTM.
        Conv(EncoderIn, "encoder.layers.0");
        for (int i = 0; i < EncoderBlocks.Count; i++)
        {
            string p = $"encoder.layers.{2 + 2 * i}.layers.";
            var (units, down) = EncoderBlocks[i];
            Conv(units[0].A, p + "0.layers.0");
            Conv(units[0].B, p + "0.layers.2");
            Conv(units[1].A, p + "2.layers.0");
            Conv(units[1].B, p + "2.layers.2");
            Conv(down, p + "4");
        }
        int last = 2 + 2 * EncoderBlocks.Count;
        Conv(EncoderOut, $"encoder.layers.{last}");
        string lstm = $"encoder.layers.{last + 1}.lstm.";
        foreach (var (cell, suffix) in new[] { (EncoderLstm.ForwardCells[0], ""), (EncoderLstm.BackwardCells[0]!, "_reverse") })
        {
            int hidden = cell.HiddenSize, input = cell.InputSize;
            cell.LoadTorchWeights(read($"{lstm}weight_ih_l0{suffix}", new[] { 4 * hidden, input }), read($"{lstm}weight_hh_l0{suffix}", new[] { 4 * hidden, hidden }),
                read($"{lstm}bias_ih_l0{suffix}", new[] { 4 * hidden }), read($"{lstm}bias_hh_l0{suffix}", new[] { 4 * hidden }));
        }
        Quantizer.Load(read, has, "quantizer.quantizers.0.");
    }
}
