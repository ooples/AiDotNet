using AiDotNet.Attributes;
using AiDotNet.Audio;
using AiDotNet.Audio.Foundations;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;

namespace AiDotNet.SpeechRecognition.AlibabaASR;

/// <summary>
/// SeACo-Paraformer (Shi et al. 2023): Paraformer-large with a bias encoder, bias decoder and bias output layer that bias
/// recognition toward a hotword list given at inference.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> SeACo-Paraformer is a fast, non-autoregressive speech recognizer that you can hand a list
/// of names or terms when you transcribe. A small extra network reads the list, checks at every output position whether
/// one of those terms is being spoken, and if so steers the output toward it; elsewhere the recognizer's own output is
/// kept.</para>
/// <para>
/// The backbone is the released Paraformer-large (<see cref="ParaformerLarge{T}"/>): a 50-block SAN-M encoder of width
/// 512, the convolutional CIF predictor and a 16-block SAN-M decoder over 8,404 characters. The release's
/// <c>CifPredictorV3</c> adds an upsampled timestamp branch to the same CIF path; it is not built, since this model emits
/// no timestamps. The bias modules (section 3.1, with the sizes of section 4.2) are a two-layer unidirectional LSTM of
/// width 512 over the hotwords' ASR-decoder embeddings, a SAN-M bias decoder of four blocks with 1,024 hidden units (FSMN
/// kernel 21, the release's) reading the CIF acoustic embeddings and the ASR decoder's hidden states over the hotword
/// representations, and an output layer over the vocabulary plus the appended no-bias class '#'. (The release's
/// configuration declares four blocks, but its decoder class builds <c>att_layer_num</c>, default 6; the paper's four are
/// used.)
/// </para>
/// <para>
/// Training follows the paper's two stages (<see cref="SeACoOptions.TrainingStage"/>). The backbone stage minimizes
/// Paraformer-large's loss. The bias stage freezes the backbone, samples hotwords from the batch's transcripts (a share
/// r_b = 0.75 of batches, r_u = 0.75 of their utterances, 2 to 8 tokens, plus the default hotword, the start token) and
/// minimizes the length-normalized label-smoothed (0.1) loss of the bias logits against the transcript with every
/// non-hotword position replaced by '#'. At inference the merge of Equation 5 keeps the ASR probabilities where the bias
/// decoder predicts '#' and uses λ P_bias + (1 − λ) P_ASR elsewhere (λ = 1); with more than k = 50 hotwords, attention
/// score filtering keeps the k the bias decoder's last layer attends to most. The optimizer is Adam at 5e-4 with 30,000
/// warmup steps and gradient clipping at 5.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.SpeechRecognition,
///     inputSize: 16000, outputSize: 8404);
/// var model = new SeACo&lt;float&gt;(architecture);
/// var text = model.Transcribe(waveform, new[] { "达摩院", "魔搭" }).Text;
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.SpeechRecognition)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("SeACo-Paraformer: A Non-Autoregressive ASR System with Flexible and Effective Hotword Customization Ability", "https://arxiv.org/abs/2308.03266", Year = 2023, Authors = "Shi et al.")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 5e-4, WarmupSteps = 30000, Schedule = LearningRateSchedulerType.Noam, MaxGradientNorm = 5,
                Source = "FunASR's seaco config: Adam at 5e-4, warmuplr with 30000 warmup steps, grad_clip 5.")]
public partial class SeACo<T> : AudioNeuralNetworkBase<T>, ISpeechRecognizer<T>, ITrainingObjectiveProvider<T>
{
    /// <inheritdoc />
    /// <remarks>The output vocabulary, blank (0), start (1) and end (2) included.</remarks>
    protected override int OutputFeatureWidth => _options.VocabSize;

    private const int StartToken = 1;
    private const int EndToken = 2;
    // The hotword sampler's seed when the options set none.
    private const int HotwordSamplingSeed = 1337;

    private readonly SeACoOptions _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;
    // The paper network on the native path; null in ONNX mode or with custom layers.
    private SanmEncoder<T>? _encoder;
    private ParaformerNetwork<T>? _network;
    private SeACoBiasNetwork<T>? _bias;
    private Random _random = Tensors.Helpers.RandomHelper.CreateSeededRandom(43);
    // Created once and advanced across training steps, so successive steps sample different hotwords.
    private Random _hotwordRandom = Tensors.Helpers.RandomHelper.CreateSeededRandom(HotwordSamplingSeed);
    private int _backboneLayerCount;
    private int _biasLayerCount;
    internal ParaformerNetwork<T>? Network => _network;
    internal SeACoBiasNetwork<T>? Bias => _bias;
    internal SanmEncoder<T>? Encoder => _encoder;
    public IReadOnlyList<string> SupportedLanguages { get; }
    /// <inheritdoc />
    public bool SupportsStreaming => false;
    /// <inheritdoc />
    public bool SupportsWordTimestamps => false;

    public SeACo(NeuralNetworkArchitecture<T> architecture, string modelPath, SeACoOptions? options = null) : base(architecture)
    {
        _options = options ?? new SeACoOptions(); Options = _options; _useNativeMode = false;
        base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels;
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath; OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        SupportedLanguages = new[] { _options.Language }; InitializeLayersCore();
    }

    public SeACo(NeuralNetworkArchitecture<T> architecture, SeACoOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture)
    {
        _options = options ?? new SeACoOptions(); Options = _options; _useNativeMode = true;
        _optimizer = optimizer ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this, warmupStepsOverride: _options.WarmupSteps) ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels;
        SupportedLanguages = new[] { _options.Language }; InitializeLayersCore();
    }

    /// <inheritdoc />
    /// <remarks>The model's optimizer, unless a base training optimizer was set explicitly.</remarks>
    protected internal override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? TrainingOptimizer => _optimizer;

    /// <summary>The encoder's CMVN shift and scale (FunASR's <c>am.mvn</c>, read by
    /// <see cref="SanmEncoder{T}.ReadNormalization"/>); identity until set.</summary>
    public (double[] Shift, double[] Scale) Normalization
    {
        get => (_encoder ?? throw new InvalidOperationException("No native network.")).Normalization;
        set => (_encoder ?? throw new InvalidOperationException("No native network.")).Normalization = value;
    }

    /// <summary>Transcribes audio without hotwords: the backbone alone, as Equation 5 prescribes when none is given.</summary>
    public TranscriptionResult<T> Transcribe(Tensor<T> audio, string? language = null, bool? includeTimestamps = null) =>
        Transcribe(audio, null, language, includeTimestamps);

    /// <summary>
    /// Transcribes audio biased toward <paramref name="hotwords"/> (names, terms, product names): the bias decoder reads
    /// the hotword list, and Equation 5 merges its probabilities with the backbone's wherever it detects a hotword.
    /// </summary>
    /// <param name="audio">The waveform.</param>
    /// <param name="hotwords">The hotwords, tokenized with <see cref="SeACoOptions.Vocabulary"/> when set (otherwise each
    /// character's code point is its token); null or empty transcribes without biasing.</param>
    /// <param name="language">Optional language override.</param>
    /// <param name="includeTimestamps">Timestamps are not supported.</param>
    public TranscriptionResult<T> Transcribe(Tensor<T> audio, IReadOnlyList<string>? hotwords, string? language = null, bool? includeTimestamps = null)
    {
        ThrowIfDisposed();
        if (audio is null) throw new ArgumentNullException(nameof(audio));
        if (ResolveReturnTimestamps(includeTimestamps)) throw new NotSupportedException("Word-level timestamps are not supported for SeACo-Paraformer.");
        var waveform = PreprocessAudio(audio);
        List<int> tokens;
        using (new Tensors.Engines.Autodiff.NoGradScope<T>())
        {
            if (_network is null || _encoder is null) tokens = ConformerAsrNetwork<T>.GreedyCtc(Logits(waveform));
            else
            {
                var ids = HotwordIds(hotwords);
                var encoded = _encoder.Encode(waveform);
                tokens = ids.Count == 0 || _bias is null ? _network.Greedy(encoded, _options.MaxTextLength) : BiasedGreedy(_network, _bias, encoded, ids);
            }
        }
        return new TranscriptionResult<T>
        {
            Text = Text(tokens), Language = language ?? _options.Language, Confidence = NumOps.FromDouble(tokens.Count > 0 ? 1.0 : 0.0),
            DurationSeconds = audio.Length / (double)SampleRate, Segments = Array.Empty<TranscriptionSegment<T>>(),
        };
    }

    public Task<TranscriptionResult<T>> TranscribeAsync(Tensor<T> audio, string? language = null, bool? includeTimestamps = null, CancellationToken cancellationToken = default) => Task.Run(() => Transcribe(audio, language, includeTimestamps), cancellationToken);
    public string DetectLanguage(Tensor<T> audio) => _options.Language;
    public IReadOnlyDictionary<string, T> DetectLanguageProbabilities(Tensor<T> audio) => new Dictionary<string, T> { [_options.Language] = NumOps.One };
    public IStreamingTranscriptionSession<T> StartStreamingSession(string? language = null) => throw new NotSupportedException("SeACo-Paraformer does not support streaming.");

    // Greedy decoding with Equation 5's merge at each acoustic token, after attention score filtering.
    private List<int> BiasedGreedy(ParaformerNetwork<T> network, SeACoBiasNetwork<T> bias, Tensor<T> encoded, List<int[]> hotwords)
    {
        var tokens = new List<int>();
        if (BiasedProbabilities(network, bias, encoded, hotwords) is not { } probabilities) return tokens;
        foreach (int best in Merge(probabilities.Asr, probabilities.Biased))
            if (best != 0 && best != StartToken && best != EndToken && tokens.Count < _options.MaxTextLength) tokens.Add(best);
        return tokens;
    }

    /// <summary>
    /// The bias decoder's probabilities <c>[n, vocabulary + 1]</c> for <paramref name="hotwords"/> on the audio, after
    /// attention score filtering keeps <see cref="SeACoOptions.AttentionFilterTopK"/> of them; null when the backbone
    /// predicts no acoustic token or the model has no bias decoder.
    /// </summary>
    internal Tensor<T>? BiasProbabilities(Tensor<T> audio, IReadOnlyList<string> hotwords)
    {
        ThrowIfDisposed();
        if (_network is null || _encoder is null || _bias is null) return null;
        using (new Tensors.Engines.Autodiff.NoGradScope<T>())
            return BiasedProbabilities(_network, _bias, _encoder.Encode(PreprocessAudio(audio)), HotwordIds(hotwords))?.Biased;
    }

    private (Tensor<T> Asr, Tensor<T> Biased)? BiasedProbabilities(ParaformerNetwork<T> network, SeACoBiasNetwork<T> bias, Tensor<T> encoded,
        List<int[]> hotwords)
    {
        var acoustic = network.AcousticEmbeddings(encoded);
        if (acoustic is null) return null;
        var hidden = network.Hidden(encoded, acoustic);
        var asr = Engine.Softmax(network.Output.Forward(hidden), axis: -1);
        var all = new List<int[]>(hotwords) { new[] { StartToken } };
        var z = bias.Hotwords(all.Select(h => network.TargetEmbeddings(h)).ToList());
        int k = _options.AttentionFilterTopK;
        if (k > 0 && hotwords.Count > k)
        {
            var keep = bias.Filter(z, hidden, k);
            z = Engine.TensorConcatenate(keep.Select(i => Engine.TensorSlice(z, new[] { i, 0 }, new[] { 1, _options.EncoderDim })).ToArray(), 0);
        }
        return (asr, Engine.Softmax(bias.Logits(z, acoustic, hidden), axis: -1));
    }

    /// <summary>
    /// Equation 5 at each position: the most likely token of the ASR probabilities <c>[n, vocabulary]</c> where the bias
    /// probabilities <c>[n, vocabulary + 1]</c> are most likely the no-bias class, otherwise of
    /// λ P_bias + (1 − λ) P_ASR.
    /// </summary>
    internal int[] Merge(Tensor<T> asr, Tensor<T> biased)
    {
        int vocabulary = _options.VocabSize, noBias = _options.ResolveHotwordMaskTokenId();
        double lambda = _options.BiasMergeLambda;
        var tokens = new int[asr.Shape[0]];
        for (int n = 0; n < tokens.Length; n++)
        {
            int detected = 0;
            for (int v = 1; v <= vocabulary; v++) if (NumOps.GreaterThan(biased[n, v], biased[n, detected])) detected = v;
            double bestScore = double.NegativeInfinity;
            for (int v = 0; v < vocabulary; v++)
            {
                double p = NumOps.ToDouble(asr[n, v]);
                if (detected != noBias) p = lambda * NumOps.ToDouble(biased[n, v]) + (1 - lambda) * p;
                if (p > bestScore) { bestScore = p; tokens[n] = v; }
            }
        }
        return tokens;
    }

    // Each hotword's tokens: a whitespace-separated piece found in the vocabulary is one token, otherwise each character
    // is (its vocabulary entry, or its code point without a vocabulary). Blank, start, end and unknown tokens are dropped.
    private List<int[]> HotwordIds(IReadOnlyList<string>? hotwords)
    {
        var result = new List<int[]>();
        if (hotwords is null) return result;
        Dictionary<string, int>? lookup = null;
        if (_options.Vocabulary is { } vocabulary)
        {
            lookup = new Dictionary<string, int>(StringComparer.Ordinal);
            for (int i = 0; i < vocabulary.Length; i++) if (!lookup.ContainsKey(vocabulary[i])) lookup[vocabulary[i]] = i;
        }
        foreach (var word in hotwords)
        {
            if (string.IsNullOrWhiteSpace(word)) continue;
            var ids = new List<int>();
            void Add(int id) { if (id > EndToken && id < _options.VocabSize) ids.Add(id); }
            foreach (var piece in word.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries))
            {
                if (lookup is not null && lookup.TryGetValue(piece, out int whole)) { Add(whole); continue; }
                for (int c = 0; c < piece.Length; c++)
                {
                    string character = char.IsSurrogatePair(piece, c) ? piece.Substring(c++, 2) : piece.Substring(c, 1);
                    if (lookup is null) Add(char.ConvertToUtf32(character, 0));
                    else if (lookup.TryGetValue(character, out int id)) Add(id);
                }
            }
            if (ids.Count > 0) result.Add(ids.ToArray());
        }
        return result;
    }

    // Each token is its vocabulary entry when a vocabulary is configured, otherwise its Unicode code point.
    private string Text(List<int> tokens)
    {
        var vocabulary = _options.Vocabulary;
        var sb = new System.Text.StringBuilder();
        foreach (int t in tokens)
        {
            if (vocabulary is not null && t < vocabulary.Length) { string s = vocabulary[t]; if (!s.StartsWith("<", StringComparison.Ordinal)) sb.Append(s); }
            else if (t <= 0x10FFFF) sb.Append(char.ConvertFromUtf32(t));
        }
        return sb.ToString().Trim();
    }

    protected override void InitializeLayers() => InitializeLayersCore();

    private void InitializeLayersCore()
    {
        if (!_useNativeMode) return;
        _encoder = null; _network = null; _bias = null; _backboneLayerCount = 0; _biasLayerCount = 0;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            // A caller-supplied list has an unknown layout, so no backbone/bias split can be asserted.
            Layers.AddRange(Architecture.Layers);
            _backboneLayerCount = Layers.Count;
            return;
        }
        // The paper network (see the class remarks). It used to be a generic Paraformer template with a separate
        // embedding, one LSTM and two attentions as the bias branch.
        var o = _options;
        var backbone = new List<LayerBase<T>>();
        var random = Tensors.Helpers.RandomHelper.CreateSeededRandom(o.Seed ?? 42);
        _encoder = new SanmEncoder<T>(Engine, backbone, new SanmEncoderConfiguration(o.NumMels, o.LfrM, o.LfrN, o.EncoderDim, o.NumAttentionHeads,
            o.FeedForwardDim, o.NumEncoderLayers, 0, o.FsmnKernelSize, 1e-12), random);
        _network = new ParaformerNetwork<T>(Engine, backbone, new ParaformerConfiguration
        {
            Width = o.EncoderDim, Heads = o.NumAttentionHeads, FeedForward = o.FeedForwardDim, DecoderLayers = o.NumDecoderLayers,
            Decoder = ParaformerDecoderKind.Sanm, Predictor = CifPredictorKind.Convolution, FsmnKernel = o.FsmnKernelSize, VocabSize = o.VocabSize,
            TailThreshold = o.TailThreshold, SamplingRatio = o.SamplingRatio, LabelSmoothing = o.LabelSmoothing, CtcWeight = 0,
            PredictorWeight = o.PredictorWeight, AppendEndToken = true, LengthNormalizedLoss = true, StartToken = StartToken, EndToken = EndToken,
        });
        var bias = new List<LayerBase<T>>();
        _bias = new SeACoBiasNetwork<T>(Engine, bias, o.EncoderDim, o.NumAttentionHeads, o.BiasFeedForwardDim, o.NumBiasDecoderLayers,
            o.BiasFsmnKernelSize, o.NumBiasEncoderLayers, o.VocabSize + 1);
        _random = Tensors.Helpers.RandomHelper.CreateSeededRandom((o.Seed ?? 42) + 1);
        _hotwordRandom = Tensors.Helpers.RandomHelper.CreateSeededRandom(o.Seed ?? HotwordSamplingSeed);
        Layers.AddRange(backbone);
        Layers.AddRange(bias);
        _backboneLayerCount = backbone.Count;
        _biasLayerCount = bias.Count;
    }

    /// <summary>Whether the bias modules exist, so the backbone/bias partition of staged training is known.</summary>
    internal bool HasBiasBranch => _biasLayerCount > 0;

    /// <summary>Half-open range of <c>Layers</c> holding the bias modules.</summary>
    internal (int Start, int End) BiasLayerRange => (_backboneLayerCount, _backboneLayerCount + _biasLayerCount);

    /// <summary>Number of flattened parameters owned by the backbone, which precede the bias modules'.</summary>
    internal long BackboneParameterCount => Layers.Take(_backboneLayerCount).Sum(layer => layer.ParameterCount);

    /// <summary>Rejects bias-stage training when no bias modules exist.</summary>
    private void EnsureBiasStageIsSupported()
    {
        if (_options.TrainingStage == SeACoTrainingStage.Bias && !HasBiasBranch)
        {
            throw new InvalidOperationException(
                "Bias-stage training requires the native SeACo bias branch. " +
                "Caller-supplied Architecture.Layers does not define a validated backbone/bias partition.");
        }
    }

    /// <inheritdoc />
    /// <remarks>The bias stage trains the bias modules alone and the backbone stage the backbone alone (section 3.1:
    /// the bias modules train "with a well-trained Paraformer model freezing").</remarks>
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        if (!HasBiasBranch || _options.TrainingStage == SeACoTrainingStage.Joint) return parameters;
        var (start, end) = BiasLayerRange;
        var bias = new HashSet<Tensor<T>>(
            Training.TapeTrainingStep<T>.CollectParameters(Layers.Skip(start).Take(end - start).ToList(), -1),
            TensorReferenceComparer<Tensor<T>>.Instance);
        bool trainBias = _options.TrainingStage == SeACoTrainingStage.Bias;
        return parameters.Where(p => bias.Contains(p) == trainBias).ToList();
    }

    // The recognizer reads the raw waveform, truncated to the longest window a transcription covers.
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio)
    {
        if (_network is null) return base.PreprocessAudio(rawAudio);
        int length = Math.Min(rawAudio.Length, SampleRate * _options.MaxAudioLengthSeconds);
        var waveform = new Tensor<T>(new[] { length });
        rawAudio.Data.Span.Slice(0, length).CopyTo(waveform.Data.Span);
        return waveform;
    }

    private Tensor<T> Samples(Tensor<T> waveform, int offset, int count)
    {
        var samples = new Tensor<T>(new[] { count });
        waveform.Data.Span.Slice(offset, count).CopyTo(samples.Data.Span);
        return samples;
    }

    // The backbone decoder's logits [1, vocabulary] for the first acoustic token of a waveform [S] (at least one token is
    // integrated), or [batch, 1, vocabulary] for [batch, S].
    private Tensor<T> Logits(Tensor<T> input)
    {
        if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input);
        if (_network is null || _encoder is null) { var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
        int rows = input.Rank == 1 ? 1 : input.Shape[0], samples = input.Length / rows;
        var outputs = new Tensor<T>[rows];
        for (int r = 0; r < rows; r++)
        {
            var encoded = _encoder.Encode(Samples(input, r * samples, samples));
            var acoustic = _network.AcousticEmbeddings(encoded, minimumTokens: 1) ?? throw new InvalidOperationException("No acoustic token.");
            var logits = _network.Decode(encoded, acoustic);
            outputs[r] = Engine.TensorSlice(logits, new[] { 0, 0 }, new[] { 1, logits.Shape[1] });
        }
        return rows == 1 && input.Rank == 1 ? outputs[0] : Engine.TensorStack(outputs, 0);
    }

    protected override Tensor<T> PredictCore(Tensor<T> input) { ThrowIfDisposed(); return Logits(input); }

    public override Tensor<T> ForwardForTraining(Tensor<T> input) => _network is null ? base.ForwardForTraining(input) : Logits(input);

    // The labels a training target names: label ids [length], or one distribution per position [n, vocabulary] read as a
    // CTC alignment (the most likely symbol per row, repeats merged, blanks removed).
    private int[] Labels(Tensor<T> target)
    {
        if (target.Rank >= 2 && target.Shape[^1] == _options.VocabSize)
            return ConformerAsrNetwork<T>.GreedyCtc(Engine.Reshape(target, new[] { target.Length / _options.VocabSize, _options.VocabSize })).ToArray();
        var labels = new int[target.Length];
        var span = target.Data.Span;
        for (int i = 0; i < labels.Length; i++)
        {
            double value = Math.Round(NumOps.ToDouble(span[i]));
            if (value <= 0 || value >= _options.VocabSize) throw new ArgumentException($"Label {value} is the blank or outside the vocabulary.", nameof(target));
            labels[i] = (int)value;
        }
        return labels;
    }

    private int[][] RowLabels(Tensor<T> input, Tensor<T> target)
    {
        int rows = input.Rank == 1 ? 1 : input.Shape[0], perRow = target.Length / rows;
        var labels = new int[rows][];
        for (int r = 0; r < rows; r++)
        {
            var row = rows == 1 ? target : Engine.Reshape(Engine.TensorSlice(Engine.Reshape(target, new[] { rows, perRow }), new[] { r, 0 }, new[] { 1, perRow }),
                target.Rank >= 3 ? target.Shape.ToArray().Skip(1).ToArray() : new[] { perRow });
            labels[r] = Labels(row);
        }
        return labels;
    }

    /// <summary>
    /// Samples a batch's hotwords as the paper does (section 3.1): with probability r_b the batch samples, and each
    /// utterance then contributes, with probability r_u, a random span of l_min to l_max of its tokens; the default
    /// hotword (the start token) is always last. A hotword position is wherever a sampled hotword occurs in a transcript.
    /// </summary>
    internal (List<int[]> Hotwords, bool[][] Positions) SampleHotwords(int[][] labels, Random random)
    {
        var hotwords = new List<int[]>();
        var positions = labels.Select(l => new bool[l.Length]).ToArray();
        int shortest = Math.Max(1, _options.HotwordMinLength), longest = Math.Max(shortest, _options.HotwordMaxLength);
        if (random.NextDouble() < _options.HotwordBatchRatio)
        {
            foreach (var row in labels)
            {
                if (random.NextDouble() >= _options.HotwordUtteranceRatio || row.Length < shortest) continue;
                int length = random.Next(shortest, Math.Min(longest, row.Length) + 1);
                int start = random.Next(0, row.Length - length + 1);
                hotwords.Add(row.Skip(start).Take(length).ToArray());
            }
        }
        foreach (var hotword in hotwords)
            for (int r = 0; r < labels.Length; r++)
                for (int s = 0; s + hotword.Length <= labels[r].Length; s++)
                {
                    bool match = true;
                    for (int i = 0; i < hotword.Length && match; i++) match = labels[r][s + i] == hotword[i];
                    if (match) for (int i = 0; i < hotword.Length; i++) positions[r][s + i] = true;
                }
        hotwords.Add(new[] { StartToken });
        return (hotwords, positions);
    }

    /// <summary>
    /// Applies the hotword-position-aware criterion to a label tensor: a copy over the bias classes whose non-hotword
    /// rows name the no-bias class.
    /// </summary>
    /// <param name="labels">Target labels, one distribution per output position.</param>
    /// <param name="isHotwordPosition">Marks the positions of a sampled hotword.</param>
    /// <remarks>SeACo section 3.1: the bias parameters "can be updated with the hotword-position-aware criterion in which
    /// labels in non-hotword positions are replaced by #".</remarks>
    internal Tensor<T> CreateBiasTrainingTarget(Tensor<T> labels, Func<int, bool> isHotwordPosition)
    {
        if (labels is null) throw new ArgumentNullException(nameof(labels));
        if (isHotwordPosition is null) throw new ArgumentNullException(nameof(isHotwordPosition));
        int sourceCols = labels.Rank > 1 ? labels.Shape[^1] : labels.Length;
        int rows = labels.Length / Math.Max(1, sourceCols);
        int biasClasses = _options.VocabSize + 1;
        var targetShape = labels.Shape.ToArray();
        if (targetShape.Length == 0) targetShape = new[] { biasClasses };
        else targetShape[^1] = biasClasses;
        var target = new Tensor<T>(targetShape);
        var span = labels.Data.Span;
        for (int r = 0; r < rows; r++)
        {
            int best = 0;
            for (int c = 1; c < sourceCols; c++) if (NumOps.GreaterThan(span[r * sourceCols + c], span[r * sourceCols + best])) best = c;
            int targetClass = isHotwordPosition(r) ? Math.Min(best, _options.VocabSize - 1) : _options.ResolveHotwordMaskTokenId();
            target.Data.Span[(r * biasClasses) + targetClass] = NumOps.One;
        }
        return target;
    }

    // The bias loss of one utterance [1, 1]: the label-smoothed loss of the bias logits at the transcript's positions (the
    // end token's dropped, as the release does) against the transcript with '#' at non-hotword positions, averaged over
    // the transcript.
    private Tensor<T> BiasLoss(SeACoBiasNetwork<T> bias, Tensor<T> hotwords, Tensor<T> acoustic, Tensor<T> hidden, int[] labels, bool[] positions)
    {
        int length = labels.Length;
        if (length == 0) return new Tensor<T>(new[] { 1, 1 });
        var logits = bias.Logits(hotwords, acoustic, hidden);
        var kept = Engine.TensorSlice(logits, new[] { 0, 0 }, new[] { length, logits.Shape[1] });
        int noBias = _options.ResolveHotwordMaskTokenId();
        var targets = new int[length];
        for (int i = 0; i < length; i++) targets[i] = positions[i] ? labels[i] : noBias;
        return ParaformerNetwork<T>.LabelSmoothedLoss(Engine, Engine.TensorLogSoftmax(kept, axis: -1), targets, _options.BiasLabelSmoothing, length);
    }

    // The stage's loss [1], averaged over the batch: Paraformer-large's loss for the backbone stage, the bias loss over
    // the frozen backbone's outputs for the bias stage, and their sum for the joint stage.
    private Tensor<T> StageLoss(Tensor<T> input, int[][] labels, Random hotwordRandom, bool sample)
    {
        var encoder = _encoder ?? throw new InvalidOperationException("No native network.");
        var network = _network ?? throw new InvalidOperationException("No native network.");
        var stage = _options.TrainingStage;
        int rows = input.Rank == 1 ? 1 : input.Shape[0], samples = input.Length / rows;
        Tensor<T>? hotwords = null;
        bool[][] positions = Array.Empty<bool[]>();
        var bias = stage == SeACoTrainingStage.Backbone ? null : _bias;
        if (bias is not null)
        {
            var sampled = SampleHotwords(labels, hotwordRandom);
            positions = sampled.Positions;
            hotwords = bias.Hotwords(sampled.Hotwords.Select(h => network.TargetEmbeddings(h)).ToList());
        }
        Tensor<T>? total = null;
        for (int r = 0; r < rows; r++)
        {
            Tensor<T> loss;
            if (bias is not null && hotwords is not null && stage == SeACoTrainingStage.Bias)
            {
                Tensor<T> acoustic, hidden;
                using (new Tensors.Engines.Autodiff.NoGradScope<T>())
                {
                    var encoded = encoder.Encode(Samples(input, r * samples, samples));
                    acoustic = network.TrainingAcousticEmbeddings(encoded, network.Targets(labels[r]).Length);
                    hidden = network.Hidden(encoded, acoustic);
                }
                loss = Engine.Reshape(BiasLoss(bias, hotwords, acoustic, hidden, labels[r], positions[r]), new[] { 1 });
            }
            else
            {
                var encoded = encoder.Encode(Samples(input, r * samples, samples));
                loss = network.Loss(encoded, labels[r], _random, null, sample);
                if (bias is not null && hotwords is not null)
                {
                    var acoustic = network.TrainingAcousticEmbeddings(encoded, network.Targets(labels[r]).Length);
                    var hidden = network.Hidden(encoded, acoustic);
                    loss = Engine.TensorAdd(loss, Engine.Reshape(BiasLoss(bias, hotwords, acoustic, hidden, labels[r], positions[r]), new[] { 1 }));
                }
            }
            total = total is null ? loss : Engine.TensorAdd(total, loss);
        }
        return Engine.TensorMultiplyScalar(total ?? throw new ArgumentException("No audio.", nameof(input)), NumOps.FromDouble(1.0 / rows));
    }

    /// <inheritdoc />
    /// <remarks>The loss of <see cref="SeACoOptions.TrainingStage"/> over the parameters that stage trains.</remarks>
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> target)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training not supported in ONNX mode.");
        EnsureBiasStageIsSupported();
        if (_encoder is null || _network is null) return false;
        var labels = RowLabels(input, target);
        TrainWithCustomLoss(input, _ => StageLoss(input, labels, _hotwordRandom, sample: true));
        return true;
    }

    TrainingObjectiveKind ITrainingObjectiveProvider<T>.TrainingObjectiveKind => TrainingObjectiveKind.Supervised;

    Tensor<T> ITrainingObjectiveProvider<T>.ResolveTrainingTarget(Tensor<T> input, Tensor<T> proposedTarget) => proposedTarget;

    /// <inheritdoc />
    /// <remarks>The objective <see cref="TryApplyLearningRule"/> minimizes, without the glancing sampler and with the
    /// hotwords drawn from a generator seeded afresh, so repeated evaluations agree.</remarks>
    T ITrainingObjectiveProvider<T>.EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target)
    {
        EnsureBiasStageIsSupported();
        if (_encoder is null || _network is null) throw new NotSupportedException("The objective is defined on the native network.");
        using var _ = new Tensors.Engines.Autodiff.NoGradScope<T>();
        var random = Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.Seed ?? HotwordSamplingSeed);
        return StageLoss(input, RowLabels(input, target), random, sample: false)[0];
    }

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    /// <inheritdoc />
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;
    public override ModelMetadata<T> GetModelMetadata() => new() { Name = _useNativeMode ? "SeACo-Native" : "SeACo-ONNX", Description = "SeACo-Paraformer: Paraformer-large with hotword bias modules (Shi et al. 2023)", FeatureCount = 1, Complexity = _options.NumEncoderLayers + _options.NumDecoderLayers, AdditionalInfo = BaseAudioMetadataInfo() };

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(SeACo<T>)); }
    /// <inheritdoc />
    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }
}
