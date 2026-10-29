using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.Latest;
using AiDotNet.TextToSpeech.Vocoders;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>The sizes of IndexTTS's ECAPA-TDNN speaker encoder (SpeechBrain's <c>ECAPA_TDNN</c> arguments).</summary>
internal sealed record IndexTtsEcapaConfiguration(int InputSize, int EmbeddingSize, int[] Channels, int[] Kernels, int[] Dilations,
    int AttentionChannels, int SeChannels, int Scale);

/// <summary>
/// The ECAPA-TDNN (Desplanques et al. 2020) IndexTTS's BigVGAN takes the speaker from, exactly as SpeechBrain builds it
/// (the release's <c>indextts/BigVGAN/ECAPA_TDNN.py</c>): every convolution "same"-padded by reflection; TDNN blocks of
/// convolution, ReLU and BatchNorm; SE-Res2Net blocks (1×1 TDNN, Res2Net over <c>Scale</c> splits, 1×1 TDNN,
/// squeeze-excitation, residual); the aggregated block outputs through one more TDNN; attentive statistics pooling with
/// global context (standard deviations clamped at 1e-12 before the square root); BatchNorm and a 1×1 convolution to the
/// embedding. BatchNorm ε = 1e-5, evaluated on its running statistics.
/// </summary>
internal sealed class IndexTtsEcapa<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    internal sealed record Tdnn(NormedConv1DLayer<T> Conv, BatchNormalizationLayer<T> Norm, int Kernel, int Dilation);

    internal sealed record Block(Tdnn Reduce, Tdnn[] Branches, Tdnn Expand, NormedConv1DLayer<T> Squeeze, NormedConv1DLayer<T> Excite,
        NormedConv1DLayer<T>? Shortcut);

    public IndexTtsEcapa(IEngine engine, List<LayerBase<T>> layers, IndexTtsEcapaConfiguration configuration, Random random)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        int stages = c.Channels.Length;
        if (stages < 3 || c.Kernels.Length != stages || c.Dilations.Length != stages)
            throw new ArgumentException("ECAPA-TDNN takes one channel count, kernel and dilation per stage, at least three stages.", nameof(configuration));
        for (int i = 1; i < stages - 1; i++)
            if (c.Channels[i] % c.Scale != 0) throw new ArgumentException("Every SE-Res2Net width divides by the Res2Net scale.", nameof(configuration));

        // SpeechBrain's Conv1d keeps torch's default initialization: U(±1/√fan_in) for weights and biases.
        NormedConv1DLayer<T> Conv(int inputs, int outputs, int kernel, int dilation)
        {
            var conv = new NormedConv1DLayer<T>(inputs, outputs, kernel, 1, dilation, 1, 0, false, ConvolutionNormalization.None);
            layers.Add(conv);
            conv.LoadTorchWeights(IndexTtsOps.Uniform(random, outputs * inputs * kernel, inputs * kernel), null,
                IndexTtsOps.Uniform(random, outputs, inputs * kernel));
            return conv;
        }
        BatchNormalizationLayer<T> Norm(int channels)
        {
            var norm = new BatchNormalizationLayer<T>(channels, epsilon: 1e-5, momentum: 0.9);
            layers.Add(norm);
            return norm;
        }
        Tdnn TdnnBlock(int inputs, int outputs, int kernel, int dilation) => new(Conv(inputs, outputs, kernel, dilation), Norm(outputs), kernel, dilation);

        First = TdnnBlock(c.InputSize, c.Channels[0], c.Kernels[0], c.Dilations[0]);
        for (int i = 1; i < stages - 1; i++)
        {
            int inputs = c.Channels[i - 1], width = c.Channels[i], split = width / c.Scale;
            var reduce = TdnnBlock(inputs, width, 1, 1);
            var branches = Enumerable.Range(0, c.Scale - 1).Select(_ => TdnnBlock(split, split, c.Kernels[i], c.Dilations[i])).ToArray();
            var expand = TdnnBlock(width, width, 1, 1);
            var squeeze = Conv(width, c.SeChannels, 1, 1);
            var excite = Conv(c.SeChannels, width, 1, 1);
            var shortcut = inputs != width ? Conv(inputs, width, 1, 1) : null;
            Blocks.Add(new Block(reduce, branches, expand, squeeze, excite, shortcut));
        }
        int aggregated = c.Channels[stages - 2] * (stages - 2), last = c.Channels[stages - 1];
        Aggregation = TdnnBlock(aggregated, last, c.Kernels[stages - 1], c.Dilations[stages - 1]);
        Attention = TdnnBlock(3 * last, c.AttentionChannels, 1, 1);
        AttentionScore = Conv(c.AttentionChannels, last, 1, 1);
        PoolingNorm = Norm(2 * last);
        Embedding = Conv(2 * last, c.EmbeddingSize, 1, 1);
    }

    public IndexTtsEcapaConfiguration Configuration { get; }
    public Tdnn First { get; }
    public List<Block> Blocks { get; } = new();
    public Tdnn Aggregation { get; }
    public Tdnn Attention { get; }
    public NormedConv1DLayer<T> AttentionScore { get; }
    public BatchNormalizationLayer<T> PoolingNorm { get; }
    public NormedConv1DLayer<T> Embedding { get; }

    // BatchNorm1d over [1, C, L] as the 2-D layer's [N, C, H, W].
    private Tensor<T> Normalize(BatchNormalizationLayer<T> norm, Tensor<T> x)
    {
        int channels = x.Shape[1], length = x.Shape[2];
        return _engine.Reshape(norm.Forward(_engine.Reshape(x, new[] { 1, channels, length, 1 })), new[] { 1, channels, length });
    }

    private Tensor<T> Forward(Tdnn block, Tensor<T> x)
    {
        int pad = block.Dilation * (block.Kernel - 1) / 2;
        var padded = pad > 0 ? VocoderOps.ReflectPad(_engine, x, pad, pad) : x;
        return Normalize(block.Norm, _engine.ReLU(block.Conv.Forward(padded)));
    }

    private Tensor<T> Forward(Block block, Tensor<T> x)
    {
        int scale = Configuration.Scale;
        var residual = block.Shortcut is null ? x : block.Shortcut.Forward(x);
        var h = Forward(block.Reduce, x);
        int split = h.Shape[1] / scale;
        var outputs = new Tensor<T>[scale];
        for (int i = 0; i < scale; i++)
        {
            var part = _engine.TensorNarrow(h, 1, i * split, split);
            outputs[i] = i == 0 ? part : Forward(block.Branches[i - 1], i == 1 ? part : _engine.TensorAdd(part, outputs[i - 1]));
        }
        h = Forward(block.Expand, _engine.TensorConcatenate(outputs, 1));
        int channels = h.Shape[1], length = h.Shape[2];
        var squeezed = _engine.ReduceMean(h, new[] { 2 }, keepDims: true);                     // [1, C, 1]
        var gate = _engine.Sigmoid(block.Excite.Forward(_engine.ReLU(block.Squeeze.Forward(squeezed))));
        h = _engine.TensorMultiply(h, _engine.TensorBroadcastTo(gate, new[] { 1, channels, length }));
        return _engine.TensorAdd(h, residual);
    }

    // Weighted mean and clamped standard deviation over time of x [1, C, L] under weights [1, C, L] (null: uniform).
    private (Tensor<T> Mean, Tensor<T> Std) Statistics(Tensor<T> x, Tensor<T>? weights)
    {
        int channels = x.Shape[1], length = x.Shape[2];
        var full = new[] { 1, channels, length };
        Tensor<T> Weighted(Tensor<T> v) => weights is null
            ? _engine.ReduceMean(v, new[] { 2 }, keepDims: true)
            : _engine.ReduceSum(_engine.TensorMultiply(weights, v), new[] { 2 }, keepDims: true);
        var mean = Weighted(x);
        var centered = _engine.TensorSubtract(x, _engine.TensorBroadcastTo(mean, full));
        var variance = _engine.TensorMax(Weighted(_engine.TensorMultiply(centered, centered)), NumOps.FromDouble(1e-12));
        return (mean, _engine.TensorSqrt(variance));
    }

    /// <summary>The speaker embedding <c>[EmbeddingSize]</c> of a log mel <c>[frames, InputSize]</c> (all frames valid).</summary>
    public Tensor<T> Forward(Tensor<T> mel)
    {
        var c = Configuration;
        int frames = mel.Shape[0];
        if (mel.Shape[1] != c.InputSize) throw new ArgumentException($"Expected {c.InputSize} bands a frame.", nameof(mel));
        int widest = Math.Max(c.Dilations[0] * (c.Kernels[0] - 1) / 2,
            Enumerable.Range(1, c.Channels.Length - 2).Select(i => c.Dilations[i] * (c.Kernels[i] - 1) / 2).DefaultIfEmpty(0).Max());
        if (frames <= widest) throw new ArgumentException($"The reflection padding needs more than {widest} frames.", nameof(mel));
        var x = Forward(First, _engine.Reshape(_engine.TensorTranspose(mel), new[] { 1, c.InputSize, frames }));
        var outputs = new List<Tensor<T>>();
        foreach (var block in Blocks)
        {
            x = Forward(block, x);
            outputs.Add(x);
        }
        x = Forward(Aggregation, _engine.TensorConcatenate(outputs.ToArray(), 1));
        int channels = x.Shape[1];
        var full = new[] { 1, channels, frames };
        var (globalMean, globalStd) = Statistics(x, null);
        var context = _engine.TensorConcatenate(new[] { x, _engine.TensorBroadcastTo(globalMean, full), _engine.TensorBroadcastTo(globalStd, full) }, 1);
        var scores = AttentionScore.Forward(_engine.Tanh(Forward(Attention, context)));
        var weights = _engine.Softmax(scores, axis: 2);
        var (mean, std) = Statistics(x, weights);
        var pooled = Normalize(PoolingNorm, _engine.TensorConcatenate(new[] { mean, std }, 1));          // [1, 2C, 1]
        return _engine.Reshape(Embedding.Forward(pooled), new[] { c.EmbeddingSize });
    }

    /// <summary>Loads SpeechBrain's names under <paramref name="prefix"/>: <c>blocks.{i}</c>, <c>mfa</c>, <c>asp</c>,
    /// <c>asp_bn</c> and <c>fc</c>.</summary>
    public void Load(Func<string, int[], double[]> read, string prefix)
    {
        void Conv(NormedConv1DLayer<T> conv, string name)
        {
            int outputs = conv.OutputChannels;
            conv.LoadTorchWeights(read(name + ".weight", conv.TorchWeightShape), null, read(name + ".bias", new[] { outputs }));
        }
        void Norm(BatchNormalizationLayer<T> norm, string name, int channels)
        {
            var shape = new[] { channels };
            TorchParameters.BatchNorm(norm, read(name + ".weight", shape), read(name + ".bias", shape), read(name + ".running_mean", shape),
                read(name + ".running_var", shape));
        }
        void Block(Tdnn block, string name)
        {
            Conv(block.Conv, name + ".conv.conv");
            Norm(block.Norm, name + ".norm.norm", block.Conv.OutputChannels);
        }
        Block(First, prefix + "blocks.0");
        for (int i = 0; i < Blocks.Count; i++)
        {
            var b = Blocks[i];
            string p = $"{prefix}blocks.{i + 1}.";
            Block(b.Reduce, p + "tdnn1");
            for (int j = 0; j < b.Branches.Length; j++) Block(b.Branches[j], $"{p}res2net_block.blocks.{j}");
            Block(b.Expand, p + "tdnn2");
            Conv(b.Squeeze, p + "se_block.conv1.conv");
            Conv(b.Excite, p + "se_block.conv2.conv");
            if (b.Shortcut is not null) Conv(b.Shortcut, p + "shortcut.conv");
        }
        Block(Aggregation, prefix + "mfa");
        Block(Attention, prefix + "asp.tdnn");
        Conv(AttentionScore, prefix + "asp.conv.conv");
        Norm(PoolingNorm, prefix + "asp_bn.norm", 2 * AttentionScore.OutputChannels);
        Conv(Embedding, prefix + "fc.conv");
    }
}

/// <summary>The sizes of IndexTTS's DVAE (the release's <c>vqvae</c> config).</summary>
internal sealed record IndexTtsDvaeConfiguration(int Channels, int Hidden, int Layers, int ResBlocks, int CodebookDim, int Tokens, int Kernel);

/// <summary>
/// The encoder and codebook of IndexTTS's DVAE (the release's <c>indextts/vqvae/xtts_dvae.py</c>, Tortoise's discrete VAE),
/// which turns a recording's 100-band log mel into the mel codes the GPT learns to predict
/// (<c>get_codebook_indices</c>): <c>Layers</c> stride-2 convolutions with ReLU (widths <c>Hidden · 2^i</c>), residual
/// blocks (3-wide convolution, ReLU, 3-wide convolution, ReLU, 1×1 convolution, plus the input), a 1×1 convolution to the
/// codebook width, and the nearest codebook entry per frame. The decoder only trains the DVAE and is not part of
/// IndexTTS.
/// </summary>
internal sealed class IndexTtsDvae<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    public IndexTtsDvae(IEngine engine, List<LayerBase<T>> layers, IndexTtsDvaeConfiguration configuration, Random random)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        if (c.Layers < 1) throw new ArgumentException("The DVAE downsamples at least once.", nameof(configuration));
        NormedConv1DLayer<T> Conv(int inputs, int outputs, int kernel, int stride, int padding)
        {
            var conv = new NormedConv1DLayer<T>(inputs, outputs, kernel, stride, 1, 1, padding, false, ConvolutionNormalization.None);
            layers.Add(conv);
            conv.LoadTorchWeights(IndexTtsOps.Uniform(random, outputs * inputs * kernel, inputs * kernel), null,
                IndexTtsOps.Uniform(random, outputs, inputs * kernel));
            return conv;
        }
        int width = c.Channels;
        for (int i = 0; i < c.Layers; i++)
        {
            int next = c.Hidden << i;
            Down.Add(Conv(width, next, c.Kernel, 2, (c.Kernel - 1) / 2));
            width = next;
        }
        for (int i = 0; i < c.ResBlocks; i++)
            Residual.Add((Conv(width, width, 3, 1, 1), Conv(width, width, 3, 1, 1), Conv(width, width, 1, 1, 0)));
        Out = Conv(width, c.CodebookDim, 1, 1, 0);
        Codebook = IndexTtsOps.Values(layers, IndexTtsOps.Normal(random, c.CodebookDim * c.Tokens, 1.0));      // embed [dim, tokens]
    }

    public IndexTtsDvaeConfiguration Configuration { get; }
    public List<NormedConv1DLayer<T>> Down { get; } = new();
    public List<(NormedConv1DLayer<T> A, NormedConv1DLayer<T> B, NormedConv1DLayer<T> C)> Residual { get; } = new();
    public NormedConv1DLayer<T> Out { get; }
    public BiasFreeLinearLayer<T> Codebook { get; }

    /// <summary>The encoder's output <c>[codes, CodebookDim]</c> of a log mel <c>[frames, Channels]</c>.</summary>
    public Tensor<T> Encode(Tensor<T> mel)
    {
        var c = Configuration;
        int frames = mel.Shape[0];
        var x = _engine.Reshape(_engine.TensorTranspose(mel), new[] { 1, c.Channels, frames });
        foreach (var down in Down) x = _engine.ReLU(down.Forward(x));
        foreach (var (a, b, conv) in Residual)
            x = _engine.TensorAdd(conv.Forward(_engine.ReLU(b.Forward(_engine.ReLU(a.Forward(x))))), x);
        x = Out.Forward(x);
        return _engine.TensorTranspose(_engine.Reshape(x, new[] { c.CodebookDim, x.Shape[2] }));
    }

    /// <summary>The codes of a log mel <c>[frames, Channels]</c>: per encoded frame, the codebook entry at the least
    /// squared distance (the first on a tie).</summary>
    public int[] Codes(Tensor<T> mel)
    {
        var c = Configuration;
        var z = Encode(mel);                                                                     // [n, dim]
        var embed = _engine.Reshape(IndexTtsOps.Value(Codebook), new[] { c.CodebookDim, c.Tokens });
        var cross = _engine.TensorMatMul(z, embed);                                              // [n, tokens]
        var norms = _engine.ReduceSum(_engine.TensorMultiply(embed, embed), new[] { 0 }, keepDims: false);
        int n = z.Shape[0];
        var codes = new int[n];
        for (int i = 0; i < n; i++)
        {
            double best = double.PositiveInfinity;
            for (int t = 0; t < c.Tokens; t++)
            {
                // |z|² is the same for every entry, so the order of |e|² − 2 z·e decides.
                double distance = NumOps.ToDouble(norms[t]) - 2 * NumOps.ToDouble(cross[i, t]);
                if (distance < best)
                {
                    best = distance;
                    codes[i] = t;
                }
            }
        }
        return codes;
    }

    /// <summary>Loads the release's <c>{prefix}encoder.*</c> and <c>{prefix}codebook.embed</c>.</summary>
    public void Load(Func<string, int[], double[]> read, string prefix)
    {
        var c = Configuration;
        void Conv(NormedConv1DLayer<T> conv, string name) =>
            conv.LoadTorchWeights(read(name + ".weight", conv.TorchWeightShape), null, read(name + ".bias", new[] { conv.OutputChannels }));
        for (int i = 0; i < Down.Count; i++) Conv(Down[i], $"{prefix}encoder.{i}.0");
        for (int i = 0; i < Residual.Count; i++)
        {
            string p = $"{prefix}encoder.{Down.Count + i}.net.";
            Conv(Residual[i].A, p + "0");
            Conv(Residual[i].B, p + "2");
            Conv(Residual[i].C, p + "4");
        }
        Conv(Out, $"{prefix}encoder.{Down.Count + Residual.Count}");
        TorchParameters.Linear(Codebook, 1, c.CodebookDim * c.Tokens, read(prefix + "codebook.embed", new[] { c.CodebookDim, c.Tokens }));
    }
}

/// <summary>The sizes and special tokens of IndexTTS's text-to-mel-code model (the release's <c>gpt</c> config).</summary>
internal sealed record IndexTts1VoiceConfiguration(int ModelDim, int Heads, int Layers, int MaxTextTokens, int MaxMelTokens, int TextTokens,
    int MelCodes, int StartText, int StopText, int StartMel, int StopMel, int ConditionLatents, IndexTtsConformerConfiguration Condition,
    int PerceiverMultiple, double Dropout);

/// <summary>
/// IndexTTS's autoregressive text-to-mel-code model (the release's <c>UnifiedVoice</c>, v1): a conformer and perceiver
/// pool the prompt's 100-band log mel into 32 latents; the GPT-2 stack reads <c>[latents, BT text ET, BM codes EM]</c>
/// with learned text and code positions and predicts, through <c>final_norm</c>, each next text token (<c>text_head</c>)
/// and each next mel code (<c>mel_head</c>). The latents of the code rows drive the BigVGAN decoder.
/// </summary>
internal sealed class IndexTts1UnifiedVoice<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    public IndexTts1UnifiedVoice(IEngine engine, List<LayerBase<T>> layers, IndexTts1VoiceConfiguration configuration, Random random)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        int d = c.ModelDim;
        Conditioning = new IndexTtsConformerEncoder<T>(engine, layers, c.Condition, random);
        Perceiver = new IndexTtsPerceiver<T>(engine, layers, random, d, c.Condition.OutputSize, c.ConditionLatents, c.Condition.Heads,
            c.PerceiverMultiple);
        TextEmbedding = IndexTtsOps.Embedding(layers, random, c.TextTokens + 1, d, 0.02);
        MelEmbedding = IndexTtsOps.Embedding(layers, random, c.MelCodes, d, 0.02);
        Gpt = new IndexTtsGpt2<T>(engine, layers, random, d, c.Heads, c.Layers, c.Dropout);
        MelPositions = IndexTtsOps.Embedding(layers, random, c.MaxMelTokens + 2 + 1, d, 0.02);
        TextPositions = IndexTtsOps.Embedding(layers, random, c.MaxTextTokens + 2, d, 0.02);
        FinalNorm = IndexTtsOps.LayerNorm(layers, d);
        TextHead = IndexTtsOps.Linear(engine, layers, random, d, c.TextTokens + 1);
        MelHead = IndexTtsOps.Linear(engine, layers, random, d, c.MelCodes);
    }

    public IndexTts1VoiceConfiguration Configuration { get; }
    public IndexTtsConformerEncoder<T> Conditioning { get; }
    public IndexTtsPerceiver<T> Perceiver { get; }
    public TiedEmbeddingLayer<T> TextEmbedding { get; }
    public TiedEmbeddingLayer<T> MelEmbedding { get; }
    public IndexTtsGpt2<T> Gpt { get; }
    public TiedEmbeddingLayer<T> MelPositions { get; }
    public TiedEmbeddingLayer<T> TextPositions { get; }
    public LayerNormalizationLayer<T> FinalNorm { get; }
    public DenseLayer<T> TextHead { get; }
    public DenseLayer<T> MelHead { get; }

    /// <summary>The speaker latents <c>[latents, d]</c> of a log mel <c>[frames, 100]</c> (<c>get_conditioning</c>).</summary>
    public Tensor<T> SpeakerConditioning(Tensor<T> mel, bool training = false, Random? random = null) =>
        Perceiver.Forward(Conditioning.Forward(mel, training, random));

    /// <summary>The text rows <c>[L + 2, d]</c>: <c>[BT, text…, ET]</c> embedded with text positions 0 ….</summary>
    public Tensor<T> TextRows(IReadOnlyList<int> text)
    {
        var c = Configuration;
        var ids = new List<int>(text.Count + 2) { c.StartText };
        ids.AddRange(text);
        ids.Add(c.StopText);
        if (ids.Count > TextPositions.VocabularySize) throw new ArgumentException($"At most {c.MaxTextTokens} text tokens fit.", nameof(text));
        return _engine.TensorAdd(TextEmbedding.Forward(IndexTtsOps.Ids<T>(ids)),
            TextPositions.Forward(IndexTtsOps.Ids<T>(Enumerable.Range(0, ids.Count).ToArray())));
    }

    /// <summary>Mel-code rows <c>[n, d]</c> of tokens at code positions <paramref name="start"/> ….</summary>
    public Tensor<T> MelRows(IReadOnlyList<int> tokens, int start = 0)
    {
        if (start + tokens.Count > MelPositions.VocabularySize)
            throw new ArgumentException($"At most {Configuration.MaxMelTokens} mel codes fit.", nameof(tokens));
        return _engine.TensorAdd(MelEmbedding.Forward(IndexTtsOps.Ids<T>(tokens)),
            MelPositions.Forward(IndexTtsOps.Ids<T>(Enumerable.Range(start, tokens.Count).ToArray())));
    }

    /// <summary>
    /// The teacher-forced pass (<c>forward</c> with <c>text_first</c>): the hidden states after <c>final_norm</c> of the
    /// text rows <c>[BT, text, ET]</c> <c>[L + 2, d]</c> and of the code rows <c>[BM, codes, EM]</c> <c>[n + 2, d]</c>.
    /// </summary>
    public (Tensor<T> Text, Tensor<T> Mel) Forward(Tensor<T> conditions, IReadOnlyList<int> text, IReadOnlyList<int> codes, bool training = false,
        Random? random = null)
    {
        var c = Configuration;
        var inputs = new List<int>(codes.Count + 2) { c.StartMel };
        inputs.AddRange(codes);
        inputs.Add(c.StopMel);
        var textRows = TextRows(text);
        var x = _engine.TensorConcatenate(new[] { conditions, textRows, MelRows(inputs) }, 0);
        int length = x.Shape[0], d = c.ModelDim, textLength = textRows.Shape[0], offset = conditions.Shape[0];
        var hidden = _engine.Reshape(Gpt.Forward(_engine.Reshape(x, new[] { 1, length, d }), null, training, random), new[] { length, d });
        var states = FinalNorm.Forward(_engine.TensorNarrow(hidden, 0, offset, length - offset));
        return (_engine.TensorNarrow(states, 0, 0, textLength), _engine.TensorNarrow(states, 0, textLength, inputs.Count));
    }

    /// <summary>The latents <c>[n, d]</c> BigVGAN decodes (<c>forward(return_latent=True)</c>: the code rows' states but
    /// the last two).</summary>
    public Tensor<T> Latents(Tensor<T> conditions, IReadOnlyList<int> text, IReadOnlyList<int> codes)
    {
        var (_, mel) = Forward(conditions, text, codes);
        return _engine.TensorNarrow(mel, 0, 0, codes.Count);
    }

    /// <summary>
    /// Generates mel codes (<c>inference_speech</c> through Hugging Face <c>generate</c>): <c>prepare_gpt_inputs</c> drops
    /// the start and stop ids from the text, then beam search (<see cref="IndexTtsBeamSearch"/>). The result stops before
    /// EM.
    /// </summary>
    public int[] Generate(Tensor<T> conditions, IReadOnlyList<int> text, IndexTtsDecoding decoding, Random random)
    {
        var c = Configuration;
        int d = c.ModelDim, beams = decoding.Beams;
        var kept = text.Where(t => t != c.StartText && t != c.StopText).ToArray();
        var prefix = _engine.TensorConcatenate(new[] { conditions, TextRows(kept), MelRows(new[] { c.StartMel }) }, 0);
        int prompt = prefix.Shape[0];
        var cache = new IndexTtsGpt2Cache<T>();
        var hidden = Gpt.Forward(_engine.Reshape(prefix, new[] { 1, prompt, d }), cache);
        var last = _engine.TensorNarrow(_engine.Reshape(hidden, new[] { prompt, d }), 0, prompt - 1, 1);
        Gpt.Reorder(cache, Enumerable.Repeat(0, beams).ToArray());
        var lastRows = Enumerable.Repeat(last, beams).ToArray();
        double[][] LogitsOfLast()
        {
            var logits = MelHead.Forward(FinalNorm.Forward(_engine.TensorConcatenate(lastRows, 0)));
            return Enumerable.Range(0, beams).Select(b => Enumerable.Range(0, c.MelCodes).Select(t => NumOps.ToDouble(logits[b, t])).ToArray()).ToArray();
        }
        void Advance(int[] sourceBeams, int[] tokens, int generated)
        {
            Gpt.Reorder(cache, sourceBeams);
            // GPT2InferenceModel embeds the k-th generated code at position k + 1, one past the teacher-forced position.
            var rows = new List<Tensor<T>>();
            for (int b = 0; b < beams; b++) rows.Add(MelRows(new[] { tokens[b] }, generated + 1));
            var next = Gpt.Forward(_engine.Reshape(_engine.TensorConcatenate(rows.ToArray(), 0), new[] { beams, 1, d }), cache);
            lastRows = Enumerable.Range(0, beams).Select(b => _engine.Reshape(_engine.TensorNarrow(next, 0, b, 1), new[] { 1, d })).ToArray();
        }
        return IndexTtsBeamSearch.Generate(c.StartMel, c.StopMel, decoding, random, LogitsOfLast, Advance);
    }

    /// <summary>Loads the release's <c>gpt.pth</c> names.</summary>
    public void Load(Func<string, int[], double[]> read)
    {
        var c = Configuration;
        int d = c.ModelDim;
        Conditioning.Load(read, "conditioning_encoder.");
        Perceiver.Load(_engine, read, "perceiver_encoder.");
        TextEmbedding.LoadTable(read("text_embedding.weight", new[] { c.TextTokens + 1, d }));
        MelEmbedding.LoadTable(read("mel_embedding.weight", new[] { c.MelCodes, d }));
        Gpt.Load(read, "gpt.");
        MelPositions.LoadTable(read("mel_pos_embedding.emb.weight", new[] { MelPositions.VocabularySize, d }));
        TextPositions.LoadTable(read("text_pos_embedding.emb.weight", new[] { TextPositions.VocabularySize, d }));
        IndexTtsOps.LoadNorm(_engine, read, FinalNorm, "final_norm", d);
        IndexTtsOps.LoadLinear(_engine, read, TextHead, "text_head", d, c.TextTokens + 1);
        IndexTtsOps.LoadLinear(_engine, read, MelHead, "mel_head", d, c.MelCodes);
    }
}
