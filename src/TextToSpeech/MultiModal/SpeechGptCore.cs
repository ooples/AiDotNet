using AiDotNet.Audio.Foundations;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.Vocoders;

namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>
/// SpeechGPT's networks and procedures, shared by <see cref="SpeechGPT{T}"/> and the speech recognizer: LLaMA over text,
/// unit and special tokens; mHuBERT's unit extractor; the unit vocoder (and, to train it, HiFi-GAN's discriminators);
/// the release's prompts; generation; the losses.
/// </summary>
internal sealed class SpeechGptCore<T>
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly SpeechGPTOptions _o;
    private SpeechGptTokenizer? _tokenizer;

    public SpeechGptCore(IEngine engine, SpeechGPTOptions options, List<LayerBase<T>> lmLayers, List<LayerBase<T>> hubertLayers,
        List<LayerBase<T>> vocoderLayers, List<LayerBase<T>> discriminatorLayers, int seed, bool withVocoder)
    {
        _engine = engine;
        _o = options;
        Validate(options);
        var o = options;
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        LanguageModel = new LlamaNetwork<T>(engine, lmLayers, LanguageModelConfiguration(o.Stage == SpeechGptTrainingStage.ChainOfModality));
        // Eq. 2: the expanded rows start random; a loaded checkpoint overwrites the text rows.
        LanguageModel.InitializeLikeReference(random);
        var hubert = HubertUnitExtractor<T>.MHubertBase(o.HubertLayer) with
        {
            HiddenSize = o.HubertWidth, Heads = o.HubertHeads, IntermediateSize = o.HubertIntermediate,
            ConvDims = Enumerable.Repeat(o.HubertConvChannels, 7).ToArray(),
        };
        Units = new HubertUnitExtractor<T>(engine, hubertLayers, hubert, o.HubertLayer, o.Units);
        Units.Network.InitializeLikeReference(random);
        if (withVocoder)
        {
            Vocoder = new CodeHiFiGan<T>(engine, vocoderLayers, new CodeHiFiGanConfiguration
            {
                Units = o.Units, EmbeddingDim = o.VocoderEmbeddingDim, InitialChannels = o.VocoderChannels,
                UpsampleRates = (int[])o.VocoderUpsampleRates.Clone(), UpsampleKernels = (int[])o.VocoderUpsampleKernels.Clone(),
                DurationHidden = o.VocoderEmbeddingDim, Speakers = o.Vocoder == SpeechGptVocoder.MultiSpeaker ? o.VocoderSpeakers : 0,
            }, random);
            if (o.IncludeVocoderDiscriminators)
            {
                Discriminators = new HiFiGanDiscriminators<T>(engine, new[] { 2, 3, 5, 7, 11 }, 3);
                discriminatorLayers.AddRange(Discriminators.Layers);
                // speech-resynthesis' mel loss: n_fft 1024, hop 256, window 1024, 80 bands to the Nyquist frequency.
                LossMel = new DifferentiableMel<T>(engine, o.SampleRate, 1024, 256, 1024, 80, 0.0, o.SampleRate / 2.0);
            }
        }
    }

    public static void Validate(SpeechGPTOptions o)
    {
        if (o.HiddenDim <= 0 || o.Layers <= 0 || o.NumHeads <= 0 || o.HiddenDim % o.NumHeads != 0)
            throw new ArgumentException("The language model's width must divide into its heads.");
        if (o.Temperature < 0 || o.TopP <= 0 || o.TopP > 1 || o.TopK < 0) throw new ArgumentException("Sampling needs a temperature ≥ 0, top-k ≥ 0 and top-p in (0, 1].");
        if (o.MaxNewTokens <= 0 || o.MinNewTokens < 0) throw new ArgumentException("MaxNewTokens must be positive and MinNewTokens non-negative.");
        if (o.VocoderUpsampleRates.Length != o.VocoderUpsampleKernels.Length) throw new ArgumentException("The vocoder needs one kernel per upsampling rate.");
        if (o.Vocoder == SpeechGptVocoder.MultiSpeaker && (o.Speaker < 0 || o.Speaker >= o.VocoderSpeakers))
            throw new ArgumentException($"Speaker must lie in [0, {o.VocoderSpeakers}).");
    }

    internal LlamaConfiguration LanguageModelConfiguration(bool lora) => new()
    {
        VocabSize = _o.VocabSize, Hidden = _o.HiddenDim, Layers = _o.Layers, Heads = _o.NumHeads, KeyValueHeads = _o.NumHeads,
        Intermediate = _o.Intermediate, RmsEpsilon = _o.RmsEpsilon, RopeTheta = _o.RopeTheta,
        LoraRank = lora ? _o.LoraRank : 0, LoraAlpha = _o.LoraAlpha, LoraDropout = _o.LoraDropout,
    };

    public LlamaNetwork<T> LanguageModel { get; }
    public HubertUnitExtractor<T> Units { get; }
    public CodeHiFiGan<T>? Vocoder { get; }
    public HiFiGanDiscriminators<T>? Discriminators { get; }
    public DifferentiableMel<T>? LossMel { get; }

    // ---------------------------------------------------------------- tokens

    public SpeechGptTokenizer Tokenizer
    {
        get
        {
            if (_tokenizer is not null) return _tokenizer;
            string directory = _o.TokenizerDirectory ?? throw new InvalidOperationException("Text needs SpeechGPT's tokenizer: set SpeechGPTOptions.TokenizerDirectory.");
            _tokenizer = SpeechGptTokenizer.FromDirectory(directory);
            if (_tokenizer.VocabularySize > _o.VocabSize)
                throw new InvalidOperationException($"The tokenizer has {_tokenizer.VocabularySize} ids; the language model's vocabulary is {_o.VocabSize}.");
            return _tokenizer;
        }
    }

    public void ResetTokenizer() => _tokenizer = null;

    /// <summary>The id of unit 0, <c>&lt;0&gt;</c>; unit u is this plus u.</summary>
    public int UnitOffset => Tokenizer.TokenId("<0>");

    public bool IsUnit(int id) => id >= UnitOffset && id < UnitOffset + _o.Units;

    /// <summary>The token string of units: <c>&lt;sosp&gt;&lt;u₁&gt;…&lt;eosp&gt;</c> (<c>Speech2Unit</c>'s merged output).</summary>
    public static string UnitString(IEnumerable<int> units) => "<sosp>" + string.Concat(units.Select(u => $"<{u}>")) + "<eosp>";

    /// <summary>A recording's units, repeats merged (§3.1), as the release's <c>Speech2Unit(merged=True)</c>.</summary>
    public int[] UnitsOf(Tensor<T> recording, bool merged = true)
    {
        var audio = new double[recording.Length];
        for (int i = 0; i < audio.Length; i++) audio[i] = Ops.ToDouble(recording.GetFlat(i));
        using var _ = new NoGradScope<T>();
        return Units.Units(audio, merged);
    }

    /// <summary>The prompt of a question: the meta instruction, <c>[Human]: {question} &lt;eoh&gt;. [SpeechGPT]: </c>
    /// (the release's inference template), tokenized with <c>&lt;s&gt;</c>.</summary>
    public int[] Prompt(string question) => Tokenizer.Encode(_o.MetaInstruction + $"[Human]: {question} <eoh>. [SpeechGPT]: ");

    /// <summary>The text of generated ids, the units and special tokens left out.</summary>
    public string TextOf(IEnumerable<int> ids)
    {
        var tokenizer = Tokenizer;
        var specials = new HashSet<int>(new[] { "<sosp>", "<eosp>", "[Human]", "[SpeechGPT]", "<eoh>", "<eos>" }.Select(tokenizer.TokenId));
        return tokenizer.Decode(ids.Where(id => !IsUnit(id) && !specials.Contains(id))).Trim();
    }

    /// <summary>The units in generated ids (the release reads every <c>&lt;n&gt;</c> of the answer).</summary>
    public int[] UnitsIn(IEnumerable<int> ids) => ids.Where(IsUnit).Select(id => id - UnitOffset).ToArray();

    // ---------------------------------------------------------------- generation

    private double[] Row(Tensor<T> logits)
    {
        var values = new double[_o.VocabSize];
        int offset = logits.Length - values.Length;
        for (int i = 0; i < values.Length; i++) values[i] = Ops.ToDouble(logits.GetFlat(offset + i));
        return values;
    }

    private Tensor<T> Embed(IReadOnlyList<int> ids)
    {
        var table = new int[1, ids.Count];
        for (int i = 0; i < ids.Count; i++)
        {
            if (ids[i] < 0 || ids[i] >= _o.VocabSize) throw new ArgumentOutOfRangeException(nameof(ids), $"Token ids lie in [0, {_o.VocabSize}).");
            table[0, i] = ids[i];
        }
        return LanguageModel.Embed(table);
    }

    /// <summary>
    /// Continues a prompt until <c>&lt;/s&gt;</c> or <see cref="SpeechGPTOptions.MaxNewTokens"/> (and the context of
    /// <see cref="TtsModelOptions.MaxTextLength"/>), sampling as transformers does (temperature, top-k, top-p; greedy at
    /// temperature 0), the end forbidden before <see cref="SpeechGPTOptions.MinNewTokens"/>. Returns the new ids.
    /// </summary>
    public List<int> Generate(int[] prompt, Random random)
    {
        if (prompt.Length >= _o.MaxTextLength) throw new ArgumentException($"The prompt's {prompt.Length} tokens fill the {_o.MaxTextLength}-token context.");
        var cache = new LlamaCache<T>();
        var (_, logits) = LanguageModel.Forward(Embed(prompt), null, null, cache, lastLogitsOnly: true);
        int end = Tokenizer.EosId;
        var generated = new List<int>();
        while (generated.Count < _o.MaxNewTokens && cache.Length < _o.MaxTextLength)
        {
            var row = Row(logits!);
            // MinNewTokensLengthLogitsProcessor: the end is -inf until the minimum is reached.
            if (generated.Count < _o.MinNewTokens) row[end] = double.NegativeInfinity;
            int next = _o.Temperature <= 0 ? NucleusSampling.Greedy(row) : NucleusSampling.Sample(row, _o.Temperature, _o.TopK, _o.TopP, null, random);
            if (next == end) break;
            generated.Add(next);
            (_, logits) = LanguageModel.Forward(Embed(new[] { next }), null, null, cache, lastLogitsOnly: true);
        }
        return generated;
    }

    // ---------------------------------------------------------------- losses

    /// <summary>
    /// An example's ids as the release's <c>generate_and_tokenize_prompt</c> builds them: <c>prefix + text</c> tokenized
    /// together with <c>&lt;s&gt;</c> and <c>&lt;/s&gt;</c> appended, and the length of the prefix tokenized alone, which
    /// the loss masks.
    /// </summary>
    public (int[] Ids, int PrefixLength) Example(string prefix, string text) =>
        (Tokenizer.Encode(prefix + text).Concat(new[] { Tokenizer.EosId }).ToArray(), Tokenizer.Encode(prefix).Length);

    /// <summary>
    /// The next-token cross-entropy of <paramref name="ids"/>, averaged over the positions from
    /// <paramref name="firstTarget"/> on (eq. 4: the prefix masked; eq. 3, every token after the first, with 1).
    /// </summary>
    public Tensor<T> Loss(int[] ids, int firstTarget)
    {
        if (ids.Length > _o.MaxTextLength) throw new ArgumentException($"The example's {ids.Length} tokens exceed {_o.MaxTextLength}.");
        int first = Math.Max(1, firstTarget), count = ids.Length - first;
        if (count <= 0) throw new ArgumentException("The example has no target tokens.");
        var (_, logits) = LanguageModel.Forward(Embed(ids.Take(ids.Length - 1).ToArray()));
        int vocabulary = _o.VocabSize, length = ids.Length - 1;
        var rows = _engine.TensorSlice(_engine.Reshape(logits!, new[] { length, vocabulary }), new[] { first - 1, 0 }, new[] { count, vocabulary });
        var logProbabilities = _engine.Reshape(_engine.TensorLogSoftmax(rows, axis: 1), new[] { count * vocabulary, 1 });
        var picks = new Tensor<int>(new[] { count }, new Vector<int>(Enumerable.Range(0, count).Select(j => j * vocabulary + ids[first + j]).ToArray()));
        return _engine.TensorMultiplyScalar(_engine.TensorNegate(_engine.ReduceSum(_engine.TensorIndexSelect(logProbabilities, picks, 0), null, false)),
            Ops.FromDouble(1.0 / count));
    }

}
