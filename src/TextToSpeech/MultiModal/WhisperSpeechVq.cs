using AiDotNet.ActivationFunctions;
using AiDotNet.Audio.Whisper;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>The sizes and tunables of WhisperSpeech's semantic tokenizer (the checkpoint's <c>config</c> and
/// <c>tunables</c>; the defaults are whisper-vq-stoks-medium-en+pl's, trained as <c>make_model</c> builds it).</summary>
internal sealed record WhisperSpeechVqConfiguration
{
    /// <summary>The semantic codes (<c>vq_codes</c>); the codebook holds one more, the padding code, with mask_embs.</summary>
    public int Codes { get; init; } = 512;
    /// <summary>The residual quantizers (<c>q_depth</c>).</summary>
    public int QuantizerDepth { get; init; } = 1;
    public int Depth { get; init; } = 1;
    public int Heads { get; init; } = 16;
    public int HeadWidth { get; init; } = 64;
    public int FfnMult { get; init; } = 4;
    /// <summary>The codebook's width, reached through <c>project_in</c> (<c>codebook_dim</c>).</summary>
    public int CodebookWidth { get; init; } = 64;
    /// <summary>Whisper frames per semantic token (<c>downsample</c>).</summary>
    public int Downsample { get; init; } = 2;
    /// <summary>Downsampling by the mean of each group (<c>downsample_mean</c>) rather than by keeping its first frame.</summary>
    public bool MeanDownsampling { get; init; } = true;
    /// <summary>Codes whose moving-average usage falls below this take a sampled batch vector (<c>threshold_ema_dead_code</c>;
    /// <c>make_model</c> trains with 0, never replacing).</summary>
    public double DeadCodeThreshold { get; init; }
    /// <summary>Audio padding reads the padding code's embedding (<c>mask_embs</c>).</summary>
    public bool MaskEmbeddings { get; init; } = true;
    public bool Rope { get; init; } = true;
    public double QueryMult { get; init; } = 2;
    public double InitStd { get; init; } = 1.5;
    public double EmbeddingsStd { get; init; } = 0.045;
    public double CodebookDecay { get; init; } = 0.9;
    /// <summary>The KL-to-teacher term's weight (<c>kl_loss_mul</c>).</summary>
    public double KlLossMul { get; init; } = 1;

    public int Width => Heads * HeadWidth;
    public int BaseWidth => 3 * HeadWidth;
    public double QkScale => QueryMult * 8 / Math.Sqrt(HeadWidth);
    /// <summary>The codebook's rows: the codes plus the padding code under mask_embs.</summary>
    public int CodebookSize => Codes + (MaskEmbeddings ? 1 : 0);
}

/// <summary>
/// WhisperSpeech's semantic tokenizer (<c>RQBottleneckTransformer</c>, vq_stoks.py, with vector-quantize-pytorch 1.6's
/// <c>ResidualVQ</c> of <c>VectorQuantize</c> layers): the Whisper encoder's states, downsampled, through a residual MLP
/// into a cosine-similarity codebook (projected to its width, l2-normalized, the most similar code; moving-average codebook
/// updates); decoding repeats the codes back to the encoder's frame rate, adds learned positions and runs rotary
/// transformer blocks, so that the frozen Whisper decoder transcribes from them. Training is cross-entropy on the
/// transcript plus KL divergence to the Whisper teacher's logits plus the commitment loss.
/// </summary>
internal sealed class WhisperSpeechVq<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly Random _random;

    public WhisperSpeechVq(IEngine engine, List<LayerBase<T>> layers, WhisperSpeechVqConfiguration configuration, OpenAiWhisperNetwork<T> whisper,
        Random random)
    {
        _engine = engine;
        _random = random;
        Configuration = configuration;
        Whisper = whisper;
        var c = configuration;
        if (whisper.Dimensions.AudioState != c.Width)
            throw new ArgumentException($"The bottleneck is {c.Width} wide; the Whisper encoder is {whisper.Dimensions.AudioState}.", nameof(whisper));
        if (whisper.Dimensions.AudioContext % c.Downsample != 0)
            throw new ArgumentException("The downsampling must divide the audio context.", nameof(configuration));
        TLayer Add<TLayer>(TLayer layer) where TLayer : LayerBase<T>
        {
            layers.Add(layer);
            return layer;
        }
        DenseLayer<T> Dense(int outputs) => Add(new DenseLayer<T>(outputs, (IActivationFunction<T>)new IdentityActivation<T>()));
        MlpNorm = Add(new LayerNormalizationLayer<T>(c.Width, 1e-5));
        Mlp1 = Dense(c.Width * c.FfnMult);
        Mlp2 = Dense(c.Width);
        for (int q = 0; q < c.QuantizerDepth; q++)
        {
            ProjectIn.Add(Dense(c.CodebookWidth));
            ProjectOut.Add(Dense(c.Width));
            // The codebook is moving-average state, not trained by the optimizer.
            Embed.Add(Add(new BiasFreeLinearLayer<T>(1, c.CodebookSize * c.CodebookWidth)));
            EmbedAverage.Add(Add(new BiasFreeLinearLayer<T>(1, c.CodebookSize * c.CodebookWidth)));
            ClusterSize.Add(Add(new BiasFreeLinearLayer<T>(1, c.CodebookSize)));
        }
        Positions = Add(new BiasFreeLinearLayer<T>(1, whisper.Dimensions.AudioContext * c.Width));
        for (int i = 0; i < c.Depth; i++)
            Blocks.Add(new WhisperSpeechBlock<T>(engine, layers, c.Width, c.Heads, c.QkScale, c.FfnMult, cross: false, rope: c.Rope));
        PostNorm = Add(new LayerNormalizationLayer<T>(c.Width, 1e-5));
        Initialize();
    }

    public WhisperSpeechVqConfiguration Configuration { get; }
    /// <summary>The frozen Whisper whose encoder feeds the bottleneck and whose decoder reads it back.</summary>
    public OpenAiWhisperNetwork<T> Whisper { get; }
    public LayerNormalizationLayer<T> MlpNorm { get; }
    public DenseLayer<T> Mlp1 { get; }
    public DenseLayer<T> Mlp2 { get; }
    public List<DenseLayer<T>> ProjectIn { get; } = new();
    public List<DenseLayer<T>> ProjectOut { get; } = new();
    public List<BiasFreeLinearLayer<T>> Embed { get; } = new();
    public List<BiasFreeLinearLayer<T>> EmbedAverage { get; } = new();
    public List<BiasFreeLinearLayer<T>> ClusterSize { get; } = new();
    public BiasFreeLinearLayer<T> Positions { get; }
    public List<WhisperSpeechBlock<T>> Blocks { get; } = new();
    public LayerNormalizationLayer<T> PostNorm { get; }

    /// <summary>Semantic tokens per 30-second window (<c>stoks_len</c>).</summary>
    public int SemanticLength => Whisper.Dimensions.AudioContext / Configuration.Downsample;

    // init_transformer: linears (biases too) truncated-normal at init_std / fan_in, the positions at embeddings_std, the
    // query heads zero, layer norms at 1 and 0; the codebook l2-normalized Kaiming-uniform rows, its sums the same rows and
    // its usage zero (no k-means initialization).
    private void Initialize()
    {
        var c = Configuration;
        double[] Normal(int count, double std) => Enumerable.Range(0, count).Select(_ => WhisperSpeechOps.TruncatedNormal(_random, std)).ToArray();
        void Linear(DenseLayer<T> layer, int inputs, int outputs) =>
            TorchParameters.Linear(_engine, layer, inputs, outputs, Normal(inputs * outputs, c.InitStd / inputs), Normal(outputs, c.InitStd / inputs));
        int w = c.Width, hidden = w * c.FfnMult;
        Linear(Mlp1, w, hidden);
        Linear(Mlp2, hidden, w);
        double bound = Math.Sqrt(6.0 / (c.CodebookSize * c.CodebookWidth));
        for (int q = 0; q < c.QuantizerDepth; q++)
        {
            Linear(ProjectIn[q], w, c.CodebookWidth);
            Linear(ProjectOut[q], c.CodebookWidth, w);
            var rows = new double[c.CodebookSize * c.CodebookWidth];
            for (int i = 0; i < rows.Length; i++) rows[i] = bound * (2 * _random.NextDouble() - 1);
            Normalize(rows, c.CodebookWidth);
            TorchParameters.Linear(Embed[q], 1, rows.Length, rows);
            TorchParameters.Linear(EmbedAverage[q], 1, rows.Length, rows);
            TorchParameters.Linear(ClusterSize[q], 1, c.CodebookSize, new double[c.CodebookSize]);
        }
        TorchParameters.Linear(Positions, 1, Whisper.Dimensions.AudioContext * w, Normal(Whisper.Dimensions.AudioContext * w, c.EmbeddingsStd));
        foreach (var block in Blocks)
        {
            TorchParameters.Linear(_engine, block.Attention.Query, w, w, new double[w * w], new double[w]);
            TorchParameters.Linear(block.Attention.Key, w, w, Normal(w * w, c.InitStd / w));
            Linear(block.Attention.Value, w, w);
            Linear(block.Attention.Output, w, w);
            Linear(block.Mlp1, w, hidden);
            Linear(block.Mlp2, hidden, w);
        }
    }

    // F.normalize over rows of the given width: x / max(‖x‖, 1e-12).
    private static void Normalize(double[] rows, int width)
    {
        for (int r = 0; r < rows.Length / width; r++)
        {
            double norm = 0;
            for (int j = 0; j < width; j++) norm += rows[r * width + j] * rows[r * width + j];
            norm = Math.Max(Math.Sqrt(norm), 1e-12);
            for (int j = 0; j < width; j++) rows[r * width + j] /= norm;
        }
    }

    private double[] Values(BiasFreeLinearLayer<T> holder)
    {
        var w = holder.Weights;
        var values = new double[w.Length];
        for (int i = 0; i < values.Length; i++) values[i] = NumOps.ToDouble(w[i]);
        return values;
    }

    private void Store(BiasFreeLinearLayer<T> holder, double[] values)
    {
        var w = holder.Weights;
        for (int i = 0; i < values.Length; i++) w[i] = NumOps.FromDouble(values[i]);
        _engine.InvalidatePersistentTensor(w);
    }

    private Tensor<T> Codebook(int q) =>
        _engine.Reshape(Embed[q].Weights, new[] { Configuration.CodebookSize, Configuration.CodebookWidth });

    // The l2-normalized rows [n, d] of x (with gradient): x / max(‖x‖, 1e-12).
    private Tensor<T> L2Normalize(Tensor<T> x)
    {
        var norm = _engine.TensorSqrt(_engine.ReduceSum(_engine.TensorMultiply(x, x), new[] { 1 }, true));
        var floor = new Tensor<T>(norm._shape);
        for (int i = 0; i < floor.Length; i++) floor[i] = NumOps.FromDouble(NumOps.ToDouble(norm[i]) < 1e-12 ? 1e-12 - NumOps.ToDouble(norm[i]) : 0);
        return _engine.TensorDivide(x, _engine.TensorBroadcastTo(_engine.TensorAdd(norm, floor), x._shape));
    }

    /// <summary>The encoder states <c>[audio context, width]</c> downsampled and through the residual MLP
    /// (<c>downsample_embeddings</c>, then <c>x + mlp(mlp_ln(x))</c>).</summary>
    private Tensor<T> Bottleneck(Tensor<T> states)
    {
        var c = Configuration;
        int frames = states.Shape[0], w = c.Width, n = frames / c.Downsample;
        var x = c.MeanDownsampling
            ? _engine.ReduceMean(_engine.Reshape(states, new[] { n, c.Downsample, w }), new[] { 1 }, false)
            : _engine.Reshape(_engine.TensorSlice(_engine.Reshape(states, new[] { n, c.Downsample, w }), new[] { 0, 0, 0 }, new[] { n, 1, w }), new[] { n, w });
        return _engine.TensorAdd(x, Mlp2.Forward(ActivationHelper.ExactGelu(Mlp1.Forward(MlpNorm.Forward(x)), _engine)));
    }

    private int[] MostSimilar(Tensor<T> normalized, int q)
    {
        var scores = _engine.TensorMatMul(normalized, _engine.TensorTranspose(Codebook(q)));
        int n = scores.Shape[0], size = scores.Shape[1];
        var codes = new int[n];
        for (int i = 0; i < n; i++)
        {
            int best = 0;
            for (int k = 1; k < size; k++)
                if (NumOps.ToDouble(scores[i, k]) > NumOps.ToDouble(scores[i, best])) best = k;
            codes[i] = best;
        }
        return codes;
    }

    private Tensor<T> Rows(int q, IReadOnlyList<int> codes)
    {
        var index = new Tensor<int>(new[] { codes.Count }, new Vector<int>(codes.ToArray()));
        return _engine.TensorIndexSelect(Codebook(q), index, 0);
    }

    /// <summary>The semantic tokens <c>[quantizers][positions]</c> of Whisper encoder states <c>[1, audio context, width]</c>
    /// (<c>quantize</c>): each residual quantizer's most cosine-similar code.</summary>
    public int[][] Quantize(Tensor<T> states)
    {
        var c = Configuration;
        using var scope = new NoGradScope<T>();
        var residual = Bottleneck(_engine.Reshape(states, new[] { states.Shape[^2], c.Width }));
        var codes = new int[c.QuantizerDepth][];
        for (int q = 0; q < c.QuantizerDepth; q++)
        {
            codes[q] = MostSimilar(L2Normalize(ProjectIn[q].Forward(residual)), q);
            if (q + 1 < c.QuantizerDepth) residual = _engine.TensorSubtract(residual, ProjectOut[q].Forward(Rows(q, codes[q])));
        }
        return codes;
    }

    /// <summary>
    /// The semantic tokens of a log-mel spectrogram <c>[mels, frames]</c> (<c>encode_mel</c>): cut to, or padded with −1.5
    /// to, the 30-second window, and with mask_embs only the tokens of the given frames are kept.
    /// </summary>
    public int[] EncodeMel(Tensor<T> mel)
    {
        int mels = Whisper.Dimensions.Mels, window = 2 * Whisper.Dimensions.AudioContext, n = mel.Shape[^1];
        var padded = new Tensor<T>(new[] { mels, window });
        var fill = NumOps.FromDouble(-1.5);
        for (int m = 0; m < mels; m++)
            for (int t = 0; t < window; t++) padded[m, t] = t < n ? mel[m, t] : fill;
        Tensor<T> states;
        using (new NoGradScope<T>()) states = Whisper.Encode(padded);
        var codes = Quantize(states)[0];
        return Configuration.MaskEmbeddings ? codes.Take(Math.Min(n, window) / 2 / Configuration.Downsample).ToArray() : codes;
    }

    // The decoder side: codes' vectors at the encoder's frame rate [audio context, width] plus positions, through the blocks.
    private Tensor<T> Unbottleneck(Tensor<T> x)
    {
        int frames = x.Shape[0], w = Configuration.Width;
        var positions = Enumerable.Range(0, frames).ToArray();
        x = _engine.TensorAdd(x, _engine.TensorSlice(_engine.Reshape(Positions.Weights, new[] { Whisper.Dimensions.AudioContext, w }), new[] { 0, 0 }, new[] { frames, w }));
        foreach (var block in Blocks) x = block.Forward(x, positions, null, null, false);
        return PostNorm.Forward(x);
    }

    private Tensor<T> Repeat(Tensor<T> x)
    {
        int n = x.Shape[0], w = x.Shape[1], d = Configuration.Downsample;
        return _engine.Reshape(_engine.TensorBroadcastTo(_engine.Reshape(x, new[] { n, 1, w }), new[] { n, d, w }), new[] { n * d, w });
    }

    /// <summary>
    /// The Whisper-encoder-like states <c>[audio context, width]</c> of semantic tokens (<c>dequantize</c>, one quantizer):
    /// cut at the first padding code, padded to the window with the padding code (code 0 without mask_embs), each code's
    /// vector repeated for its frames and projected out, positions added, through the blocks.
    /// </summary>
    public Tensor<T> Dequantize(IReadOnlyList<int> tokens)
    {
        var c = Configuration;
        if (c.QuantizerDepth != 1) throw new NotSupportedException("The release dequantizes single-quantizer tokens only.");
        var codes = tokens.TakeWhile(t => t != c.Codes).ToList();
        if (codes.Count > SemanticLength) throw new ArgumentException($"At most {SemanticLength} tokens.", nameof(tokens));
        while (codes.Count < SemanticLength) codes.Add(c.MaskEmbeddings ? c.Codes : 0);
        return Unbottleneck(ProjectOut[0].Forward(Repeat(Rows(0, codes))));
    }

    /// <summary>
    /// One training step's loss (<c>forward</c> in training mode) for a log-mel window <c>[mels, 2 × audio context]</c>, the
    /// encoder frames holding audio (<paramref name="mask"/>, the rest padding), and the transcript's decoder input and
    /// targets (−100 ignored): cross-entropy, plus <c>kl_loss_mul</c> × the KL divergence (summed, per batch) from the
    /// teacher's logits — zeroed where the target is ignored, so those positions are pulled toward uniform, as there — plus
    /// the commitment loss (each quantizer's mean squared distance of the normalized projection to its code, averaged).
    /// The codebooks take their moving-average update unless <paramref name="updateCodebook"/> is false (evaluation).
    /// </summary>
    public Tensor<T> Loss(Tensor<T> mel, bool[] mask, IReadOnlyList<int> inputTokens, IReadOnlyList<int> outputTokens, bool updateCodebook = true)
    {
        var c = Configuration;
        int frames = Whisper.Dimensions.AudioContext, w = c.Width, vocabulary = Whisper.Dimensions.Vocabulary, n = inputTokens.Count;
        if (mask.Length != frames) throw new ArgumentException($"The mask covers the {frames} encoder frames.", nameof(mask));
        Tensor<T> states, teacher;
        using (new NoGradScope<T>())
        {
            states = Whisper.Encode(mel);
            teacher = Whisper.Logits(inputTokens, states);
        }
        var residual = Bottleneck(_engine.Reshape(states, new[] { frames, w }));
        Tensor<T>? quantized = null, commitment = null;
        for (int q = 0; q < c.QuantizerDepth; q++)
        {
            var normalized = L2Normalize(ProjectIn[q].Forward(residual));
            int[] codes;
            Tensor<T> code;
            using (new NoGradScope<T>())
            {
                codes = MostSimilar(normalized, q);
                code = new Tensor<T>(new[] { codes.Length, c.CodebookWidth }, Rows(q, codes).ToVector());
                if (updateCodebook) UpdateCodebook(q, normalized, codes);
            }
            var difference = _engine.TensorSubtract(code, normalized);
            var term = _engine.ReduceMean(_engine.TensorMultiply(difference, difference), new[] { 0, 1 }, false);
            commitment = commitment is null ? term : _engine.TensorAdd(commitment, term);
            // Straight-through: x + stop(q − x).
            var straight = _engine.TensorAdd(normalized, new Tensor<T>(difference._shape, difference.ToVector()));
            var output = ProjectOut[q].Forward(straight);
            quantized = quantized is null ? output : _engine.TensorAdd(quantized, output);
            if (q + 1 < c.QuantizerDepth) residual = _engine.TensorSubtract(residual, new Tensor<T>(output._shape, output.ToVector()));
        }
        var x = Repeat(quantized!);
        if (c.MaskEmbeddings)
        {
            // x[~mask] = project_out(embed[padding code]) — the codebook after this step's update.
            var padding = ProjectOut[0].Forward(Rows(0, new[] { c.Codes }));
            var keep = new Tensor<T>(new[] { frames, 1 });
            var drop = new Tensor<T>(new[] { frames, 1 });
            for (int i = 0; i < frames; i++) (mask[i] ? keep : drop)[i, 0] = NumOps.One;
            x = _engine.TensorAdd(_engine.TensorMultiply(x, _engine.TensorBroadcastTo(keep, x._shape)),
                _engine.TensorMultiply(_engine.TensorBroadcastTo(padding, x._shape), _engine.TensorBroadcastTo(drop, x._shape)));
        }
        x = Unbottleneck(x);
        var logits = Whisper.Logits(inputTokens, _engine.Reshape(x, new[] { 1, frames, w }));
        var log = _engine.TensorLogSoftmax(logits, axis: 1);
        int valid = outputTokens.Count(t => t != -100);
        // The teacher's distribution p (rows of ignored targets zeroed to uniform); KL = Σ p log p − Σ p log q.
        var weights = new Tensor<T>(log._shape);
        double entropy = 0;
        for (int i = 0; i < n; i++)
        {
            bool ignored = outputTokens[i] == -100;
            double max = double.NegativeInfinity;
            for (int v = 0; v < vocabulary; v++) max = Math.Max(max, ignored ? 0 : NumOps.ToDouble(teacher[i, v]));
            double sum = 0;
            for (int v = 0; v < vocabulary; v++) sum += Math.Exp((ignored ? 0 : NumOps.ToDouble(teacher[i, v])) - max);
            for (int v = 0; v < vocabulary; v++)
            {
                double lp = (ignored ? 0 : NumOps.ToDouble(teacher[i, v])) - max - Math.Log(sum), p = Math.Exp(lp);
                entropy += p * lp;
                double weight = -c.KlLossMul * p;
                if (!ignored && v == outputTokens[i]) weight -= 1.0 / valid;
                weights[i, v] = NumOps.FromDouble(weight);
            }
        }
        var loss = _engine.TensorAddScalar(_engine.ReduceSum(_engine.TensorMultiply(log, weights), new[] { 0, 1 }, false), NumOps.FromDouble(c.KlLossMul * entropy));
        return _engine.TensorAdd(loss, _engine.TensorMultiplyScalar(commitment!, NumOps.FromDouble(1.0 / c.QuantizerDepth)));
    }

    // CosineSimCodebook's training update: usage and sums by moving average (torch.lerp), the codes the sums over the
    // Laplace-smoothed usage, l2-normalized; then codes used less than the threshold take normalized batch vectors.
    private void UpdateCodebook(int q, Tensor<T> normalized, int[] codes)
    {
        var c = Configuration;
        int size = c.CodebookSize, d = c.CodebookWidth;
        double rate = 1 - c.CodebookDecay;
        var usage = Values(ClusterSize[q]);
        var sums = Values(EmbedAverage[q]);
        var counts = new double[size];
        var batch = new double[size * d];
        for (int i = 0; i < codes.Length; i++)
        {
            counts[codes[i]] += 1;
            for (int j = 0; j < d; j++) batch[codes[i] * d + j] += NumOps.ToDouble(normalized[i, j]);
        }
        static double Lerp(double from, double to, double weight) => weight < 0.5 ? from + weight * (to - from) : to - (to - from) * (1 - weight);
        for (int k = 0; k < size; k++) usage[k] = Lerp(usage[k], counts[k], rate);
        for (int i = 0; i < sums.Length; i++) sums[i] = Lerp(sums[i], batch[i], rate);
        double total = usage.Sum();
        var embed = new double[size * d];
        for (int k = 0; k < size; k++)
        {
            double smoothed = (usage[k] + 1e-5) / (total + size * 1e-5) * total;
            for (int j = 0; j < d; j++) embed[k * d + j] = sums[k * d + j] / smoothed;
        }
        Normalize(embed, d);
        Normalize(embed, d);
        if (c.DeadCodeThreshold > 0)
        {
            // replace(): sample_vectors over the batch's (normalized) vectors, a permutation when there are enough.
            var expired = Enumerable.Range(0, size).Where(k => usage[k] < c.DeadCodeThreshold).ToArray();
            if (expired.Length > 0)
            {
                int rows = codes.Length;
                int[] picks;
                if (rows >= expired.Length)
                {
                    var order = Enumerable.Range(0, rows).ToArray();
                    for (int i = rows - 1; i > 0; i--)
                    {
                        int j = _random.Next(i + 1);
                        (order[i], order[j]) = (order[j], order[i]);
                    }
                    picks = order.Take(expired.Length).ToArray();
                }
                else picks = Enumerable.Range(0, expired.Length).Select(_ => _random.Next(rows)).ToArray();
                for (int e = 0; e < expired.Length; e++)
                {
                    var row = new double[d];
                    for (int j = 0; j < d; j++) row[j] = NumOps.ToDouble(normalized[picks[e], j]);
                    Normalize(row, d);
                    for (int j = 0; j < d; j++)
                    {
                        embed[expired[e] * d + j] = row[j];
                        sums[expired[e] * d + j] = row[j] * c.DeadCodeThreshold;
                    }
                    usage[expired[e]] = c.DeadCodeThreshold;
                }
            }
        }
        Store(ClusterSize[q], usage);
        Store(EmbedAverage[q], sums);
        Store(Embed[q], embed);
    }

    /// <summary>Loads a checkpoint's <c>state_dict</c> (vector-quantize-pytorch 1.6's per-quantizer projections; the output
    /// blocks under <c>out_blocks</c>, as the released checkpoints name them, or <c>_out_blocks</c>, as the code does).</summary>
    public void LoadTorchWeights(Func<string, double[]?> read)
    {
        var c = Configuration;
        double[] Read(string name) => read(name) ?? throw new InvalidDataException($"The VQ checkpoint has no tensor '{name}'.");
        int w = c.Width, hidden = w * c.FfnMult;
        TorchParameters.LayerNorm(_engine, MlpNorm, Read("mlp_ln.weight"), Read("mlp_ln.bias"));
        TorchParameters.Linear(_engine, Mlp1, w, hidden, Read("mlp.0.weight"), Read("mlp.0.bias"));
        TorchParameters.Linear(_engine, Mlp2, hidden, w, Read("mlp.2.weight"), Read("mlp.2.bias"));
        for (int q = 0; q < c.QuantizerDepth; q++)
        {
            string p = $"rq.layers.{q}.";
            TorchParameters.Linear(_engine, ProjectIn[q], w, c.CodebookWidth, Read(p + "project_in.weight"), Read(p + "project_in.bias"));
            TorchParameters.Linear(_engine, ProjectOut[q], c.CodebookWidth, w, Read(p + "project_out.weight"), Read(p + "project_out.bias"));
            TorchParameters.Linear(Embed[q], 1, c.CodebookSize * c.CodebookWidth, Read(p + "_codebook.embed"));
            TorchParameters.Linear(EmbedAverage[q], 1, c.CodebookSize * c.CodebookWidth, Read(p + "_codebook.embed_avg"));
            TorchParameters.Linear(ClusterSize[q], 1, c.CodebookSize, Read(p + "_codebook.cluster_size"));
        }
        TorchParameters.Linear(Positions, 1, Whisper.Dimensions.AudioContext * w, Read("positional_embedding.weight"));
        string blocks = read("out_blocks.0.attn_ln.weight") is not null ? "out_blocks." : "_out_blocks.";
        for (int i = 0; i < Blocks.Count; i++) Blocks[i].LoadTorchWeights(read, $"{blocks}{i}.");
        TorchParameters.LayerNorm(_engine, PostNorm, Read("ln_post.weight"), Read("ln_post.bias"));
    }
}
