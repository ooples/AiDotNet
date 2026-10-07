using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Audio.Generation;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.FrontEnd;
using AiDotNet.TextToSpeech.Interfaces;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Which of Pheme's two networks a training step updates.</summary>
public enum PhemeTrainingStage
{
    /// <summary>The T5 text-to-semantic model: phonemes to SpeechTokenizer semantic tokens.</summary>
    TextToSemantic,

    /// <summary>The SoundStorm-style semantic-to-acoustic model: semantic tokens to the acoustic codebooks.</summary>
    SemanticToAcoustic,
}

/// <summary>
/// Pheme: efficient and conversational speech generation in two stages over SpeechTokenizer codes, a T5 text-to-semantic
/// model and a non-autoregressive SoundStorm-style semantic-to-acoustic model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// References: "Pheme: Efficient and Conversational Speech Generation" (Budzianowski et al., PolyAI, 2024) and its
/// reference implementation (PolyAI-LDN/pheme) for what the paper leaves unstated.
/// </para>
/// <para>
/// <b>Text to semantic</b> (T2S): text becomes espeak-style phonemes (<see cref="EnglishG2P"/>), read by a T5
/// encoder–decoder (<c>T5ForConditionalGeneration</c> v1.0) over one vocabulary of the phoneme symbols and the 1024
/// semantic codes: the input is <c>&lt;bos&gt; spkr_1 phonemes &lt;eos&gt;</c>, the target <c>&lt;bos&gt; spkr_1
/// semantic tokens &lt;eos&gt;</c>, trained with the teacher-forced cross-entropy.
/// </para>
/// <para>
/// <b>Semantic to acoustic</b> (S2A, after SoundStorm, Borsos et al. 2023): each of the seven acoustic codebooks has
/// its own embedding; a training step picks one codebook level, keeps a random prompt prefix, masks a cosine-scheduled
/// fraction of the level's remaining tokens, sums the embeddings of the lower levels, the partly masked level and the
/// semantic tokens, adds the projected L2-normalized pyannote speaker embedding, runs a SoundStorm Conformer and
/// predicts the masked tokens with the level's head (cross-entropy over the masked positions).
/// </para>
/// <para>
/// <b>Synthesis</b> (reference <c>transformer_infer.py</c>): the voice prompt's transcript precedes the text and its
/// semantic tokens start the T5 decoder, which samples the continuation (temperature 0.7, top 210, resampled while a
/// token repeats more than 100 times in a row); the acoustic model fills the first codebook with 16 MaskGIT steps and
/// the other six with one greedy step each, after the prompt's acoustic tokens (the reference code's order; the
/// paper's text describes the reverse, <see cref="PhemeOptions.AcousticDecoding"/>); SpeechTokenizer decodes the codes.
/// Two reference behaviours are kept as written: the speaker-embedding dropout is applied at inference too (the code
/// calls <c>F.dropout</c> without the training flag), and the last prompt frame is regenerated and returned with the
/// synthesized frames (the prompt split counts the leading speaker token).
/// One is not: the reference's MaskGIT loop passes the initial, fully masked sequence to the model at every step
/// (<c>self.forward(inputs, ...)</c>, never the updated <c>state.cur_seqs</c>), so its later steps refine nothing; here
/// each step reads the tokens decoded so far, as MaskGIT (Chang et al. 2022) and SoundStorm decode and the paper
/// describes.
/// </para>
/// <para>
/// <b>Training data</b>: <see cref="TtsTrainingSample{T}.Tokens"/> are the phoneme ids (see
/// <see cref="EncodePhonemes"/>), <see cref="TtsTrainingSample{T}.CodecTokens"/> the SpeechTokenizer codes
/// <c>[frames, 8]</c> (or <see cref="TtsTrainingSample{T}.Audio"/>, which the model encodes), and with the speaker
/// embedding the recording it is computed from (<see cref="TtsTrainingSample{T}.SpeakerReference"/> or the audio).
/// <see cref="CurrentStage"/> selects the network a step trains.
/// </para>
/// <para><b>For Beginners:</b> Pheme first writes down "what is said" as a short sequence of speech tokens, then fills
/// in "how it sounds" for a given voice, and a codec turns those tokens into audio.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(InputType.OneDimensional,
///     NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);
/// var pheme = new Pheme&lt;float&gt;(architecture, new PhemeOptions());
/// pheme.Voice = pheme.CreateVoice(promptAudio16kHz, "the prompt's transcript");
/// var audio = pheme.Synthesize("Hello there.");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "Pheme: Efficient and Conversational Speech Generation",
    "https://arxiv.org/abs/2401.02839",
    Year = 2024,
    Authors = "Budzianowski et al."
)]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 5e-4, Beta1 = 0.9, Beta2 = 0.98, Epsilon = 1e-8, WeightDecay = 0.0,
    Schedule = LearningRateSchedulerType.LinearWarmup, WarmupSteps = 10_000,
    PostWarmupDecay = LinearWarmupScheduler.DecayMode.Linear,
    Component = "text-to-semantic", Provenance = RecipeProvenance.Stated,
    Source = "Section 4.1: both networks use AdamW (beta1 0.9, beta2 0.98) at 5e-4 with 10,000 warm-up steps and linear "
             + "decay to zero over 800k steps. The paper states no weight decay or clipping; the reference trains the T5 "
             + "with the Hugging Face Trainer's (no weight decay, gradient norm clipped to 1).")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 5e-4, Beta1 = 0.9, Beta2 = 0.98, WeightDecay = 0.01,
    Schedule = LearningRateSchedulerType.LinearWarmup, WarmupSteps = 10_000,
    PostWarmupDecay = LinearWarmupScheduler.DecayMode.Linear,
    Component = "semantic-to-acoustic", Provenance = RecipeProvenance.Stated,
    Source = "Section 4.1: both networks use AdamW (beta1 0.9, beta2 0.98) at 5e-4 with 10,000 warm-up steps and linear "
             + "decay to zero over 800k steps. The paper states no weight decay; the reference's torch.optim.AdamW has "
             + "PyTorch's default 0.01.")]
public partial class Pheme<T> : TtsModelBase<T>, ITtsModel<T>
{
    private const int PadToken = 0, BeginToken = 1, EndToken = 2, Speaker1Token = 3;
    private const int SpeakerEmbeddingDim = 512;

    private readonly PhemeOptions _options;
    private readonly bool _useNativeMode;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _suppliedOptimizer;
    private readonly Dictionary<PhemeTrainingStage, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>> _stageOptimizers = new();
    private readonly Random _trainingRandom;
    private bool _disposed;

    // Vocabulary: [<pad>, <bos>, <eos>, spkr_1, spkr_2] + sorted(USLM symbols ∪ "0" … "1023").
    private readonly Dictionary<string, int> _tokenIds = new(StringComparer.Ordinal);
    private readonly string[] _tokens;
    private readonly int[] _semanticIds;

    private T5Seq2Seq<T>? _textToSemantic;
    private readonly List<LayerBase<T>> _textLayers = new();
    private readonly List<LayerBase<T>> _acousticLayers = new();
    private readonly List<TiedEmbeddingLayer<T>> _levelEmbeddings = new();
    private TiedEmbeddingLayer<T>? _semanticEmbedding;
    private DenseLayer<T>? _speakerProjection;
    private SoundStormConformer<T>? _conformer;
    private readonly List<DenseLayer<T>> _heads = new();
    private readonly List<LayerBase<T>> _speakerLayers = new();
    private PyannoteXVector<T>? _speakerEncoder;
    private AudioCodecLayer<T>? _codec;

    /// <summary>USLM's <c>unique_text_tokens.k2symbols</c> (fnlp/USLM, USLM_libritts), the phoneme table Pheme's
    /// checkpoints were trained on: espeak-ng's American English symbols, word separator and punctuation.</summary>
    private static readonly string[] PhonemeSymbols =
    {
        "<eps>", "!", "\"", "(", ")", ",", ".", ":", ";", "?", "_", "aɪ", "aɪə", "aɪɚ", "aɪʊ", "aɪʊɹ", "aʊ", "b", "d",
        "dʒ", "e", "enus", "es", "eɪ", "f", "fr", "h", "i", "iə", "iː", "j", "k", "l", "m", "n", "nʲ", "oʊ", "oː", "oːɹ",
        "p", "r", "s", "t", "tʃ", "uː", "v", "w", "x", "z", "æ", "ç", "ð", "ø", "ŋ", "ɐ", "ɑ", "ɑː", "ɑːɹ", "ɔ", "ɔɪ",
        "ɔː", "ɔːɹ", "ə", "əl", "ɚ", "ɛ", "ɛɹ", "ɛː", "ɜː", "ɡ", "ɡʲ", "ɣ", "ɪ", "ɪɹ", "ɫ", "ɬ", "ɲ", "ɹ", "ɾ", "ʃ",
        "ʊ", "ʊɹ", "ʌ", "ʒ", "ʔ", "̃", "̩", "θ", "ᵻ", "—",
    };

    /// <inheritdoc />
    public override ModelOptions GetOptions() => _options;

    /// <summary>Creates an ONNX-backed Pheme for inference.</summary>
    public Pheme(NeuralNetworkArchitecture<T> architecture, string modelPath, PhemeOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new PhemeOptions();
        _useNativeMode = false;
        _trainingRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
        (_tokens, _semanticIds) = BuildVocabulary(_tokenIds, _options.SemanticCodes);
        ConfigureBase();
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path required.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new AiDotNet.Onnx.OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a native, trainable Pheme.</summary>
    public Pheme(NeuralNetworkArchitecture<T> architecture, PhemeOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new PhemeOptions();
        _useNativeMode = true;
        _suppliedOptimizer = optimizer;
        _trainingRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
        (_tokens, _semanticIds) = BuildVocabulary(_tokenIds, _options.SemanticCodes);
        ConfigureBase();
        InitializeLayers();
    }

    private void ConfigureBase()
    {
        base.SampleRate = _options.SampleRate;
        base.HopSize = _options.HopSize;
        base.HiddenDim = _options.HiddenDim;
        _options.VocabSize = 5 + PhonemeSymbols.Length + _options.SemanticCodes;
    }

    private static (string[] Tokens, int[] SemanticIds) BuildVocabulary(Dictionary<string, int> ids, int semanticCodes)
    {
        var symbols = new List<string>(PhonemeSymbols);
        for (int s = 0; s < semanticCodes; s++) symbols.Add(s.ToString(System.Globalization.CultureInfo.InvariantCulture));
        symbols.Sort(StringComparer.Ordinal);                      // Python's sorted(): code-point order
        var tokens = new List<string> { "<pad>", "<bos>", "<eos>", "spkr_1", "spkr_2" };
        tokens.AddRange(symbols);
        for (int i = 0; i < tokens.Count; i++) ids[tokens[i]] = i;
        var semantic = new int[semanticCodes];
        for (int s = 0; s < semanticCodes; s++) semantic[s] = ids[s.ToString(System.Globalization.CultureInfo.InvariantCulture)];
        return (tokens.ToArray(), semantic);
    }

    int ITtsModel<T>.SampleRate => _options.SampleRate;

    /// <inheritdoc />
    public int MaxTextLength => _options.MaxTextLength;

    /// <summary>The network a training step updates.</summary>
    public PhemeTrainingStage CurrentStage { get; set; } = PhemeTrainingStage.TextToSemantic;

    /// <summary>Training steps taken by each network (they set the learning-rate schedules).</summary>
    public int TextToSemanticUpdates { get; private set; }

    /// <summary>Training steps taken by the acoustic network.</summary>
    public int SemanticToAcousticUpdates { get; private set; }

    /// <summary>The SpeechTokenizer whose codes Pheme models.</summary>
    public SpeechTokenizer<T> Codec => (SpeechTokenizer<T>?)_codec?.Codec ?? throw new InvalidOperationException("Pheme's codec exists in native mode only.");

    /// <inheritdoc />
    protected override TtsSupervision RequiredSupervision =>
        TtsSupervision.CodecTokens | (_options.UseSpeakerEmbedding ? TtsSupervision.ReferenceRecording : TtsSupervision.None);

    /// <inheritdoc />
    protected override TtsSupervision RequiredVoice => TtsSupervision.ReferenceRecording;

    /// <inheritdoc />
    public override int CodecTokenCodebooks => 1 + _options.AcousticCodebooks;

    /// <inheritdoc />
    public override int CodecTokenVocabulary => Math.Min(_options.SemanticCodes, _options.CodebookSize);

    private bool HasPaperLayers => _textToSemantic is not null;

    // The networks' layers, for tests that check a stage updates its own network only.
    internal IReadOnlyList<LayerBase<T>> TextToSemanticLayers => _textLayers;
    internal IReadOnlyList<LayerBase<T>> SemanticToAcousticLayers => _acousticLayers;
    internal IReadOnlyList<LayerBase<T>> SpeakerEncoderLayers => _speakerLayers;
    internal LayerBase<T>? CodecLayer => _codec;

    /// <inheritdoc />
    protected override void InitializeLayers()
    {
        if (!_useNativeMode)
            return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            AddUnsplitLayers(Architecture.Layers);
            return;
        }

        var o = _options;
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(Architecture.RandomSeed ?? o.SamplingSeed);
        _textToSemantic = new T5Seq2Seq<T>(Engine, _textLayers, new T5Configuration(
            VocabularySize: o.VocabSize, ModelDim: o.TextModelDim, FeedForwardDim: o.TextFeedForwardDim,
            KeyValueDim: o.TextKeyValueDim, Heads: o.NumHeads, EncoderLayers: o.NumEncoderLayers,
            DecoderLayers: o.NumDecoderLayers, Dropout: o.DropoutRate, DecoderStartTokenId: PadToken, EndTokenId: EndToken));
        _textToSemantic.InitializeLikeHuggingFace(random);

        // Acoustic model: per-level embeddings of the codes plus PAD, SPKR_1 and SPKR_2 (padding row 1024).
        int codeRows = o.CodebookSize + 3, semanticRows = o.SemanticCodes + 3;
        double Normal002()
        {
            double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
            return 0.02 * Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
        }
        for (int l = 0; l < o.AcousticCodebooks; l++)
        {
            var embedding = Own(_acousticLayers, new TiedEmbeddingLayer<T>(codeRows, o.HiddenDim));
            embedding.Reinitialize(Normal002, o.CodebookSize);
            _levelEmbeddings.Add(embedding);
        }
        _semanticEmbedding = Own(_acousticLayers, new TiedEmbeddingLayer<T>(semanticRows, o.HiddenDim));
        _semanticEmbedding.Reinitialize(Normal002, o.SemanticCodes);
        if (o.UseSpeakerEmbedding)
            _speakerProjection = Own(_acousticLayers, new DenseLayer<T>(o.HiddenDim, new IdentityActivation<T>() as IActivationFunction<T>));
        _conformer = new SoundStormConformer<T>(Engine, _acousticLayers, new SoundStormConformerConfiguration(
            Dim: o.HiddenDim, Layers: o.AcousticLayers, Heads: o.AcousticHeads, HeadDim: o.AcousticHeadDim,
            FeedForwardMultiplier: o.AcousticFeedForwardMultiplier, ConvExpansion: o.AcousticConvExpansion,
            ConvKernel: o.AcousticConvKernel, Dropout: o.DropoutRate));
        for (int l = 0; l < o.AcousticCodebooks; l++)
            _heads.Add(Own(_acousticLayers, new DenseLayer<T>(codeRows, new IdentityActivation<T>() as IActivationFunction<T>)));
        InitializeAcousticWeights(Normal002);

        if (o.UseSpeakerEmbedding)
        {
            _speakerEncoder = new PyannoteXVector<T>(Engine, _speakerLayers);
            // Its dense layers draw their weights on first use; draw them now, so the parameters a caller saves, counts
            // or loads into are the ones the model runs with.
            // In inference mode, so the batch norms' running statistics are untouched.
            foreach (var layer in _speakerLayers) layer.SetTrainingMode(false);
            using (new NoGradScope<T>()) _speakerEncoder.Forward(new Tensor<T>(new[] { 1, 1, 8000 }));   // half a second, past the TDNNs' receptive field
        }
        // Pheme only encodes and decodes with its (pretrained, frozen) codec, so the codec's training discriminators are
        // never built.
        var codecOptions = (SpeechTokenizerOptions)AiDotNet.Models.CloneEngine.CopyConfiguration(o.SpeechTokenizer);
        codecOptions.IncludeDiscriminators = false;
        var codec = new SpeechTokenizer<T>(new NeuralNetworkArchitecture<T>(InputType.OneDimensional,
            NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1) { RandomSeed = Architecture.RandomSeed }, codecOptions);
        if (codec.CodebookSize != o.CodebookSize || codec.CodebookSize != o.SemanticCodes)
            throw new ArgumentException($"Pheme's {o.CodebookSize} acoustic and {o.SemanticCodes} semantic codes must equal the codec's {codec.CodebookSize}.");
        int codecRate = ((IAudioCodec<T>)codec).SampleRate;
        if (codec.HopLength != o.HopSize || codecRate != o.SampleRate)
            throw new ArgumentException($"Pheme's {o.SampleRate} Hz audio and {o.HopSize}-sample frames must be its codec's " +
                $"({codecRate} Hz, {codec.HopLength}-sample hop); set SampleRate and HopSize to match the SpeechTokenizer options.");
        int codebooks = codec.QuantizersForBandwidth(o.SpeechTokenizer.TargetBandwidthKbps);
        if (codebooks != 1 + o.AcousticCodebooks)
            throw new ArgumentException($"Pheme models {1 + o.AcousticCodebooks} codebooks; the codec's bandwidth gives {codebooks}.");
        _codec = new AudioCodecLayer<T>(codec);

        AddEncoderDecoderLayers(_textLayers.Cast<ILayer<T>>().ToList(), Array.Empty<ILayer<T>>());
        ComponentLayers.AddRange(_acousticLayers);
        ComponentLayers.AddRange(_speakerLayers);
        ComponentLayers.Add(_codec);
    }

    private static TLayer Own<TLayer>(List<LayerBase<T>> layers, TLayer layer) where TLayer : LayerBase<T>
    {
        layers.Add(layer);
        return layer;
    }

    // Pheme.init_weights: Linear and Conv1d weights from N(0, 0.02) with zero biases; LayerNorm at one and zero.
    private void InitializeAcousticWeights(Func<double> normal)
    {
        foreach (var layer in _acousticLayers)
        {
            switch (layer)
            {
                case DenseLayer<T> dense:
                {
                    using (new NoGradScope<T>()) dense.Forward(new Tensor<T>(new[] { 1, InputWidth(dense) }));
                    var w = dense.GetWeights();
                    for (int i = 0; i < w.Length; i++) w[i] = NumOps.FromDouble(normal());
                    Engine.InvalidatePersistentTensor(w);
                    var b = dense.GetBiases();
                    for (int i = 0; i < b.Length; i++) b[i] = NumOps.Zero;
                    Engine.InvalidatePersistentTensor(b);
                    break;
                }
                case BiasFreeLinearLayer<T> linear:
                {
                    var values = new Vector<T>(linear.InputSize * linear.OutputSize);
                    for (int i = 0; i < values.Length; i++) values[i] = NumOps.FromDouble(normal());
                    linear.SetParameters(values);
                    break;
                }
                case NormedConv1DLayer<T> conv:
                    conv.Reinitialize(normal);
                    break;
            }
        }
    }

    private int InputWidth(DenseLayer<T> dense) =>
        ReferenceEquals(dense, _speakerProjection) ? SpeakerEmbeddingDim
        : _heads.Contains(dense) ? _options.HiddenDim
        : InputWidthInConformer(dense);

    private int InputWidthInConformer(DenseLayer<T> dense)
    {
        var o = _options;
        foreach (var block in _conformer!.Blocks)
        {
            if (ReferenceEquals(dense, block.Ff1In) || ReferenceEquals(dense, block.Ff2In)) return o.HiddenDim;
            if (ReferenceEquals(dense, block.Ff1Out) || ReferenceEquals(dense, block.Ff2Out)) return o.HiddenDim * o.AcousticFeedForwardMultiplier;
            if (ReferenceEquals(dense, block.ToOut)) return o.AcousticHeads * o.AcousticHeadDim;
        }
        throw new InvalidOperationException("Unknown dense layer in Pheme's acoustic model.");
    }

    // ---------------------------------------------------------------- vocabulary and front end

    /// <summary>Pheme's token id of each phoneme symbol (symbols outside its table are dropped, as the reference
    /// drops them), without the <c>&lt;bos&gt; spkr_1 … &lt;eos&gt;</c> frame.</summary>
    public int[] EncodePhonemes(IEnumerable<string> symbols)
    {
        var ids = new List<int>();
        foreach (var symbol in symbols)
            if (_tokenIds.TryGetValue(symbol, out int id) && id >= 5) ids.Add(id);
        return ids.ToArray();
    }

    /// <summary>A voice for <see cref="TtsModelBase{T}.Voice"/>: a 16 kHz prompt recording and its transcript.</summary>
    public TtsVoice<T> CreateVoice(Tensor<T> recording, string transcript)
    {
        if (recording is null) throw new ArgumentNullException(nameof(recording));
        var ids = EncodePhonemes(EnglishG2P.Default.Phonemize(transcript ?? throw new ArgumentNullException(nameof(transcript))));
        var tokens = new Tensor<T>(new[] { ids.Length });
        for (int i = 0; i < ids.Length; i++) tokens[i] = NumOps.FromDouble(ids[i]);
        return new TtsVoice<T> { ReferenceAudio = recording, ReferenceTokens = tokens };
    }

    /// <inheritdoc />
    /// <remarks>The text's phonemes (<see cref="EnglishG2P"/>) as Pheme's token ids, framed
    /// <c>&lt;bos&gt; spkr_1 … &lt;eos&gt;</c>.</remarks>
    protected override Tensor<T> PreprocessText(string text)
    {
        if (text is null) throw new ArgumentNullException(nameof(text));
        var ids = EncodePhonemes(EnglishG2P.Default.Phonemize(text));
        if (ids.Length == 0) throw new ArgumentException("The text has no pronounceable content.", nameof(text));
        int length = Math.Min(ids.Length, _options.MaxTextLength);
        var tokens = new Tensor<T>(new[] { length + 3 });
        tokens[0] = NumOps.FromDouble(BeginToken);
        tokens[1] = NumOps.FromDouble(Speaker1Token);
        for (int i = 0; i < length; i++) tokens[i + 2] = NumOps.FromDouble(ids[i]);
        tokens[length + 2] = NumOps.FromDouble(EndToken);
        return tokens;
    }

    private int SemanticCode(int tokenId)
    {
        if (tokenId < 5 || tokenId >= _tokens.Length) return -1;
        var token = _tokens[tokenId];
        return token.Length > 0 && char.IsDigit(token[0]) && int.TryParse(token, System.Globalization.NumberStyles.None,
            System.Globalization.CultureInfo.InvariantCulture, out int code) ? code : -1;
    }

    // ---------------------------------------------------------------- codes and speaker

    private int[,] CodesOf(TtsTrainingSample<T> sample)
    {
        int codebooks = CodecTokenCodebooks;
        if (sample.CodecTokens is { } tokens)
        {
            if (tokens.Rank != 2 || tokens.Shape[1] != codebooks)
                throw new ArgumentException($"Expected codec tokens [frames, {codebooks}].", nameof(sample));
            int frames = tokens.Shape[0];
            var codes = new int[codebooks, frames];
            for (int f = 0; f < frames; f++)
                for (int q = 0; q < codebooks; q++) codes[q, f] = (int)Math.Round(NumOps.ToDouble(tokens[f, q]));
            return codes;
        }
        var audio = sample.Audio ?? throw new ArgumentException("Pheme trains on codec tokens or the recording.", nameof(sample));
        return Codec.Encode(audio);
    }

    /// <summary>The L2-normalized pyannote embedding <c>[512]</c> of a recording (audio normalized to a 0.95 peak, as
    /// the reference does before the speaker model).</summary>
    private Tensor<T> SpeakerEmbedding(Tensor<T> recording)
    {
        var encoder = _speakerEncoder ?? throw new InvalidOperationException("This Pheme has no speaker embedding.");
        int samples = recording.Length;
        double peak = 0;
        for (int i = 0; i < samples; i++) peak = Math.Max(peak, Math.Abs(NumOps.ToDouble(recording[i])));
        var waveform = new Tensor<T>(new[] { 1, 1, samples });
        double scale = peak > 0 ? 0.95 / peak : 0;
        for (int i = 0; i < samples; i++) waveform[0, 0, i] = NumOps.FromDouble(NumOps.ToDouble(recording[i]) * scale);
        bool wasTraining = IsTrainingMode;
        foreach (var layer in _speakerLayers) layer.SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            var embedding = encoder.Forward(waveform);                                                       // [1, 512]
            double norm = 0;
            for (int i = 0; i < embedding.Length; i++) norm += Math.Pow(NumOps.ToDouble(embedding[i]), 2);
            norm = Math.Max(Math.Sqrt(norm), 1e-12);
            var normalized = new Tensor<T>(new[] { SpeakerEmbeddingDim });
            for (int i = 0; i < SpeakerEmbeddingDim; i++) normalized[i] = NumOps.FromDouble(NumOps.ToDouble(embedding[i]) / norm);
            return normalized;
        }
        finally
        {
            foreach (var layer in _speakerLayers) layer.SetTrainingMode(wasTraining);
        }
    }

    // ---------------------------------------------------------------- acoustic network

    /// <summary>
    /// The acoustic network's logits <c>[frames, codes + 3]</c> for codebook <paramref name="level"/>
    /// (reference <c>TTSConformer.forward</c>): the embeddings of levels below it, plus level <paramref name="level"/>'s
    /// with the <paramref name="masked"/> positions zeroed, plus the semantic embedding and the projected speaker
    /// embedding.
    /// </summary>
    internal Tensor<T> AcousticLogits(int[,] acoustic, int[] semantic, int level, bool[]? masked, Tensor<T>? speaker,
        bool training, Random random)
    {
        int frames = semantic.Length;
        Tensor<T>? sum = null;
        for (int l = 0; l <= level; l++)
        {
            var ids = new Tensor<T>(new[] { frames });
            var keep = new Tensor<T>(new[] { frames, _options.HiddenDim });
            for (int t = 0; t < frames; t++)
            {
                ids[t] = NumOps.FromDouble(acoustic[l, t]);
                // The padding row (1024) contributes zero and gets no gradient; at the current level, so do masked positions.
                bool zero = acoustic[l, t] == _options.CodebookSize || (l == level && masked is not null && masked[t]);
                if (!zero)
                    for (int d = 0; d < _options.HiddenDim; d++) keep[t, d] = NumOps.One;
            }
            var embedded = Engine.TensorMultiply(_levelEmbeddings[l].Forward(ids), keep);
            sum = sum is null ? embedded : Engine.TensorAdd(sum, embedded);
        }
        var semanticIds = new Tensor<T>(new[] { frames });
        var semanticKeep = new Tensor<T>(new[] { frames, _options.HiddenDim });
        for (int t = 0; t < frames; t++)
        {
            semanticIds[t] = NumOps.FromDouble(semantic[t]);
            if (semantic[t] != _options.SemanticCodes)
                for (int d = 0; d < _options.HiddenDim; d++) semanticKeep[t, d] = NumOps.One;
        }
        var x = Engine.TensorAdd(sum!, Engine.TensorMultiply(_semanticEmbedding!.Forward(semanticIds), semanticKeep));
        if (_speakerProjection is not null && speaker is not null)
        {
            // F.dropout(spkr_emb, p) without the training flag: active at inference too, as in the reference.
            var dropped = T5Seq2Seq<T>.Dropout(Engine, Engine.Reshape(speaker, new[] { 1, SpeakerEmbeddingDim }),
                _options.SpeakerEmbeddingDropout, random);
            var projected = _speakerProjection.Forward(dropped);                                               // [1, hidden]
            x = Engine.TensorAdd(x, Engine.TensorTile(projected, new[] { frames, 1 }));
        }
        var hidden = _conformer!.Forward(x, training, random);
        return _heads[level].Forward(hidden);
    }

    // masking_logic.schedule("cosine"): cos(r · π / 2), clipped to [1e-6, 1].
    private static double CosineSchedule(double ratio) => Math.Min(1.0, Math.Max(1e-6, Math.Cos(ratio * Math.PI / 2)));

    // ---------------------------------------------------------------- training

    /// <inheritdoc />
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (IsOnnxMode)
            throw new NotSupportedException("Training is not supported in ONNX mode.");
        ThrowIfTokenMelTrainingUnsupported();
    }

    /// <inheritdoc />
    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        if (IsOnnxMode)
            throw new NotSupportedException("Training is not supported in ONNX mode.");
        var objective = BuildObjective(sample, _trainingRandom, training: true);
        var loss = TrainWithCustomObjective(sample.Tokens, sample.Tokens, objective, StageOptimizer());
        if (CurrentStage == PhemeTrainingStage.TextToSemantic) TextToSemanticUpdates++;
        else SemanticToAcousticUpdates++;
        return loss;
    }

    /// <inheritdoc />
    public override T EvaluateTrainingObjective(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        var objective = BuildObjective(sample, AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed), training: false);
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            return objective(sample.Tokens, sample.Tokens)[0];
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    private Func<Tensor<T>, Tensor<T>, Tensor<T>> BuildObjective(TtsTrainingSample<T> sample, Random random, bool training)
    {
        Guard.NotNull(sample);
        if (!HasPaperLayers)
            throw new NotSupportedException("The paper objective needs the paper's layers; this model was built from caller-supplied layers.");
        var codes = CodesOf(sample);
        int frames = codes.GetLength(1);
        var stage = CurrentStage;

        if (stage == PhemeTrainingStage.TextToSemantic)
        {
            // Labels <bos> spkr_1 semantic… <eos> (ConcatenateSemanticDataset).
            var labels = new int[frames + 3];
            labels[0] = BeginToken;
            labels[1] = Speaker1Token;
            for (int f = 0; f < frames; f++) labels[f + 2] = _semanticIds[codes[0, f]];
            labels[frames + 2] = EndToken;
            var framed = JoinPrompt(sample.Tokens, null);
            return (_, _) => _textToSemantic!.Loss(framed, labels, training, random);
        }

        if (frames < 3)
            throw new ArgumentException("Pheme's acoustic training needs at least three frames (a prompt and a masked span).", nameof(sample));
        var o = _options;
        int levels = o.AcousticCodebooks;
        var acoustic = new int[levels, frames];
        for (int l = 0; l < levels; l++)
            for (int f = 0; f < frames; f++) acoustic[l, f] = codes[l + 1, f];
        var semantic = new int[frames];
        for (int f = 0; f < frames; f++) semantic[f] = codes[0, f];
        Tensor<T>? speaker = null;
        if (o.UseSpeakerEmbedding)
            speaker = SpeakerEmbedding(sample.SpeakerReference ?? sample.Audio
                ?? throw new ArgumentException("Pheme's speaker embedding needs a recording.", nameof(sample)));

        // Pheme.training_step / TTSConformer.create_mask, for one utterance.
        int level = random.Next(levels);
        int start = 1 + random.Next(frames - 2);                                        // randint(1, min_len − 1)
        double ratio = CosineSchedule(random.NextDouble());
        var masked = new bool[frames];
        bool any = false;
        for (int t = start; t < frames; t++)
            any |= masked[t] = random.NextDouble() < ratio;
        if (!any) masked[start + random.Next(frames - start)] = true;
        var positions = Enumerable.Range(0, frames).Where(t => masked[t] && acoustic[level, t] != o.CodebookSize).ToArray();

        return (_, _) =>
        {
            var logits = AcousticLogits(acoustic, semantic, level, masked, speaker, training, random);         // [T, V]
            var logProbabilities = Engine.TensorLogSoftmax(logits, axis: -1);
            var target = new Tensor<T>(logits._shape);
            foreach (int t in positions) target[t, acoustic[level, t]] = NumOps.FromDouble(-1.0 / positions.Length);
            return Engine.ReduceSum(Engine.TensorMultiply(logProbabilities, target), new[] { 0, 1 }, keepDims: false);
        };
    }

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? StageOptimizer()
    {
        if (_suppliedOptimizer is not null)
            return _suppliedOptimizer;
        if (!_stageOptimizers.TryGetValue(CurrentStage, out var optimizer))
        {
            var o = _options;
            bool text = CurrentStage == PhemeTrainingStage.TextToSemantic;
            int warmup = text ? o.TextWarmupSteps : o.AcousticWarmupSteps;
            int total = text ? o.TextTrainingSteps : o.AcousticTrainingSteps;
            double peak = o.LearningRate;
            var schedule = new LambdaLRScheduler(peak, step =>
            {
                if (step < warmup) return (double)step / Math.Max(1, warmup);
                return Math.Max(0.0, (double)(total - step) / Math.Max(1, total - warmup));
            });
            var options = new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = peak,
                Beta1 = 0.9,
                Beta2 = 0.98,
                Epsilon = 1e-8,
                WeightDecay = text ? 0.0 : 0.01,
                EnableGradientClipping = text,
                MaxGradientNorm = 1.0,
                LearningRateScheduler = schedule,
            };
            optimizer = PaperOptimizerFactory.VerifyHandBuilt(this, new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this, options),
                text ? "text-to-semantic" : "semantic-to-acoustic");
            _stageOptimizers[CurrentStage] = optimizer;
        }
        return optimizer;
    }

    /// <inheritdoc />
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        if (!HasPaperLayers) return parameters;
        var stageLayers = CurrentStage == PhemeTrainingStage.TextToSemantic ? _textLayers : _acousticLayers;
        var stage = new HashSet<Tensor<T>>(Training.TapeTrainingStep<T>.CollectParameters(stageLayers.Cast<ILayer<T>>().ToList(), -1),
            Helpers.TensorReferenceComparer<Tensor<T>>.Instance);
        return parameters.Where(stage.Contains).ToList();
    }

    // ---------------------------------------------------------------- synthesis

    /// <inheritdoc />
    /// <remarks>
    /// <paramref name="input"/> is Pheme's framed token sequence (<see cref="PreprocessText"/>); the output is the
    /// 16 kHz waveform <c>[samples]</c>. With a voice's transcript (<see cref="TtsVoice{T}.ReferenceTokens"/>), its
    /// phonemes precede the text's, as the reference prepends the prompt's text.
    /// </remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        if (!HasPaperLayers) return base.PredictCore(input);
        var voice = RequireVoice();
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            var promptAudio = voice.ReferenceAudio!;
            var promptCodes = Codec.Encode(promptAudio);                                                     // [8, P]
            int promptFrames = promptCodes.GetLength(1);
            var textIds = JoinPrompt(input, voice.ReferenceTokens);
            var semantic = GenerateSemantic(textIds, promptCodes, random);
            var speaker = _options.UseSpeakerEmbedding ? SpeakerEmbedding(promptAudio) : null;
            var codes = GenerateAcoustic(promptCodes, semantic, speaker, random);                            // [8, frames]
            var audio = Codec.Decode(codes);
            return audio.Reshape(new[] { audio.Length });
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    // <bos> spkr_1 prompt-phonemes _ text-phonemes <eos>.
    private Tensor<T> JoinPrompt(Tensor<T> input, Tensor<T>? promptTokens)
    {
        var ids = Enumerable.Range(0, input.Length).Select(i => (int)Math.Round(NumOps.ToDouble(input[i]))).ToList();
        if (ids.Any(id => id < 0 || id >= _tokens.Length))
            throw new ArgumentOutOfRangeException(nameof(input), $"Pheme's token ids lie in [0, {_tokens.Length}).");
        if (ids.Count < 2 || ids[0] != BeginToken || ids[1] != Speaker1Token)
            ids.InsertRange(0, new[] { BeginToken, Speaker1Token });
        if (ids[ids.Count - 1] != EndToken) ids.Add(EndToken);
        if (promptTokens is not null && promptTokens.Length > 0)
        {
            var prompt = Enumerable.Range(0, promptTokens.Length).Select(i => (int)Math.Round(NumOps.ToDouble(promptTokens[i]))).ToList();
            prompt.Add(_tokenIds["_"]);
            ids.InsertRange(2, prompt);
        }
        var tensor = new Tensor<T>(new[] { ids.Count });
        for (int i = 0; i < ids.Count; i++) tensor[i] = NumOps.FromDouble(ids[i]);
        return tensor;
    }

    // PhemeClient.infer_text: continue the prompt's semantic tokens; resample while a token repeats > 100 times.
    private int[] GenerateSemantic(Tensor<T> textIds, int[,] promptCodes, Random random)
    {
        int promptFrames = promptCodes.GetLength(1);
        var prefix = new List<int> { BeginToken, Speaker1Token };
        for (int f = 0; f < promptFrames; f++) prefix.Add(_semanticIds[promptCodes[0, f]]);
        for (int attempt = 0; ; attempt++)
        {
            var output = _textToSemantic!.Generate(textIds, prefix, _options.Temperature, _options.TopK, _options.MaxNewSemanticTokens, random);
            int longest = 0, run = 0;
            for (int i = 0; i < output.Count; i++)
            {
                run = i > 0 && output[i] == output[i - 1] ? run + 1 : 1;
                longest = Math.Max(longest, run);
            }
            if (longest <= _options.MaxConsecutiveRepeats || attempt >= 16)
            {
                var codes = output.Select(SemanticCode).Where(c => c >= 0).ToList();
                return codes.Skip(promptFrames).ToArray();
            }
        }
    }

    // PhemeClient.infer_acoustic + TTSConformer.inference: returns the codes after the prompt split, [8, frames].
    private int[,] GenerateAcoustic(int[,] promptCodes, int[] generatedSemantic, Tensor<T>? speaker, Random random)
    {
        var o = _options;
        int levels = o.AcousticCodebooks, pad = o.CodebookSize, speakerToken = o.CodebookSize + 1;
        int promptFrames = promptCodes.GetLength(1), frames = 1 + promptFrames + generatedSemantic.Length;
        var acoustic = new int[levels, frames];
        var semantic = new int[frames];
        semantic[0] = speakerToken;
        for (int l = 0; l < levels; l++) acoustic[l, 0] = speakerToken;
        for (int f = 0; f < promptFrames; f++)
        {
            semantic[1 + f] = promptCodes[0, f];
            for (int l = 0; l < levels; l++) acoustic[l, 1 + f] = promptCodes[l + 1, f];
        }
        for (int f = 0; f < generatedSemantic.Length; f++)
        {
            semantic[1 + promptFrames + f] = generatedSemantic[f];
            for (int l = 0; l < levels; l++) acoustic[l, 1 + promptFrames + f] = pad;
        }
        int start = promptFrames;                                     // start_t = len(acoustic_prompt), before the SPKR_1 pad

        for (int level = 0; level < levels; level++)
        {
            bool maskGit = o.AcousticDecoding == PhemeAcousticDecoding.MaskGitFirstLevel ? level == 0 : level > 0;
            if (maskGit)
            {
                MaskGit(acoustic, semantic, speaker, start, level, random);
                continue;
            }
            // TTSConformer.one_step_inference: one greedy pass over the level.
            var logits = AcousticLogits(acoustic, semantic, level, null, speaker, training: false, random);
            for (int t = start; t < frames; t++)
            {
                int best = 0;
                double bestValue = double.NegativeInfinity;
                for (int v = 0; v < logits.Shape[1]; v++)
                {
                    double value = NumOps.ToDouble(logits[t, v]);
                    if (value > bestValue) { bestValue = value; best = v; }
                }
                acoustic[level, t] = best;
            }
        }

        // Special tokens become code 0; the result keeps the frames from start_t on, semantic codebook first.
        int kept = frames - start;
        var codes = new int[1 + levels, kept];
        for (int t = 0; t < kept; t++)
        {
            codes[0, t] = semantic[start + t] >= o.SemanticCodes ? 0 : semantic[start + t];
            for (int l = 0; l < levels; l++) codes[l + 1, t] = acoustic[l, start + t] >= pad ? 0 : acoustic[l, start + t];
        }
        return codes;
    }

    // TTSConformer.multi_step_inference (MaskGIT, Chang et al. 2022) on one acoustic codebook.
    private void MaskGit(int[,] acoustic, int[] semantic, Tensor<T>? speaker, int start, int level, Random random)
    {
        var o = _options;
        int frames = semantic.Length, mask = o.CodebookSize, steps = o.MaskGitSteps;
        var current = new int[frames];
        for (int t = 0; t < frames; t++) current[t] = t < start ? acoustic[level, t] : mask;
        int maskedAtStart = frames - start;
        var final = (int[])current.Clone();
        for (int step = 0; step < steps; step++)
        {
            for (int t = 0; t < frames; t++) acoustic[level, t] = current[t];
            var logits = AcousticLogits(acoustic, semantic, level, null, speaker, training: false, random);
            int vocabulary = logits.Shape[1];
            var sampled = new int[frames];
            var confidence = new double[frames];
            int unknown = 0;
            double ratio = (step + 1.0) / steps, temperature = 1.0 * (1.0 - ratio);
            for (int t = 0; t < frames; t++)
            {
                var row = new double[vocabulary];
                double max = double.NegativeInfinity;
                for (int v = 0; v < vocabulary; v++) max = Math.Max(max, row[v] = NumOps.ToDouble(logits[t, v]));
                double total = 0;
                for (int v = 0; v < vocabulary; v++) total += row[v] = Math.Exp(row[v] - max);
                // Categorical(logits).sample()
                double draw = random.NextDouble() * total;
                int choice = vocabulary - 1;
                for (int v = 0; v < vocabulary; v++)
                {
                    draw -= row[v];
                    if (draw <= 0) { choice = v; break; }
                }
                bool isUnknown = current[t] == mask;
                sampled[t] = isUnknown ? choice : current[t];
                if (isUnknown)
                {
                    unknown++;
                    double gumbel = -Math.Log(-Math.Log(Math.Max(1e-20, random.NextDouble())));
                    confidence[t] = Math.Log(row[sampled[t]] / total) + temperature * gumbel;
                }
                else
                {
                    confidence[t] = double.PositiveInfinity;
                }
            }
            final = (int[])sampled.Clone();
            int toMask = (int)Math.Floor(maskedAtStart * CosineSchedule(ratio));
            toMask = Math.Max(1, Math.Min(unknown - 1, toMask));
            // mask_by_random_topk: everything below the toMask-th smallest confidence is masked again.
            var order = Enumerable.Range(0, frames).OrderBy(t => confidence[t]).ToArray();
            double cutOff = confidence[order[Math.Min(toMask, frames - 1)]];
            for (int t = 0; t < frames; t++) current[t] = confidence[t] < cutOff ? mask : sampled[t];
        }
        for (int t = 0; t < frames; t++) acoustic[level, t] = final[t];
    }

    /// <inheritdoc />
    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;

    // ---------------------------------------------------------------- pretrained weights

    /// <summary>Loads the released text-to-semantic model: the state dictionary of <c>T5ForConditionalGeneration</c> as
    /// safetensors, or the released <c>t2s.bin</c> (a PyTorch archive) saved with a <c>.pt</c> extension. Build the
    /// model with the matching options (<see cref="PhemeOptions.OfficialSmallCheckpoint"/> or
    /// <see cref="PhemeOptions.OfficialLargeCheckpoint"/>).</summary>
    public void LoadTextToSemanticWeights(string path) =>
        _textToSemantic!.LoadHuggingFaceWeights(Reader(path, prefix: ""));

    /// <summary>Loads the released acoustic model: the <c>state_dict</c> of <c>s2a.ckpt</c> (its <c>model.*</c>
    /// tensors) saved as safetensors or <c>.pt</c>; the Lightning checkpoint itself also holds optimizer state and is
    /// not read directly.</summary>
    public void LoadSemanticToAcousticWeights(string path)
    {
        var read = Reader(path, prefix: "model.");
        var o = _options;
        int rows = o.CodebookSize + 3;
        for (int l = 0; l < _levelEmbeddings.Count; l++) _levelEmbeddings[l].LoadTable(read($"embedding.{l}.weight", new[] { rows, o.HiddenDim }));
        _semanticEmbedding!.LoadTable(read("semantic_embedding.weight", new[] { o.SemanticCodes + 3, o.HiddenDim }));
        if (_speakerProjection is not null)
            TorchParameters.Linear(Engine, _speakerProjection, SpeakerEmbeddingDim, o.HiddenDim,
                read("spkr_linear.weight", new[] { o.HiddenDim, SpeakerEmbeddingDim }), read("spkr_linear.bias", new[] { o.HiddenDim }));
        _conformer!.LoadTorchWeights(Engine, "conformer", read);
        for (int l = 0; l < _heads.Count; l++)
            TorchParameters.Linear(Engine, _heads[l], o.HiddenDim, rows, read($"heads.{l}.weight", new[] { rows, o.HiddenDim }),
                read($"heads.{l}.bias", new[] { rows }));
    }

    /// <summary>Loads the pyannote/embedding speaker model's state dictionary (safetensors or <c>.pt</c>).</summary>
    public void LoadSpeakerEncoderWeights(string path) =>
        (_speakerEncoder ?? throw new InvalidOperationException("This Pheme has no speaker embedding.")).LoadTorchWeights(Reader(path, prefix: ""));

    private static Func<string, int[], double[]> Reader(string path, string prefix)
    {
        var file = new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(path);
        return (name, shape) =>
        {
            if (!file.TryGetValue(prefix + name, out var tensor) && !file.TryGetValue(name, out tensor))
                throw new InvalidDataException($"The checkpoint has no tensor '{prefix + name}'.");
            var actual = tensor.Shape.ToArray();
            if (!actual.SequenceEqual(shape))
                throw new InvalidDataException($"'{name}' is [{string.Join(", ", actual)}] in the checkpoint but [{string.Join(", ", shape)}] " +
                    "in this model; build the model with the checkpoint's configuration.");
            return tensor.ToVector().Select(v => (double)v).ToArray();
        };
    }

    // ---------------------------------------------------------------- housekeeping

    /// <inheritdoc />
    protected override bool SupportsParameterMutation => _useNativeMode;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "Pheme-Native" : "Pheme-ONNX",
            Description = "Pheme: Efficient and Conversational Speech Generation (Budzianowski et al., 2024)",
            FeatureCount = _options.HiddenDim,
            Complexity = _options.NumEncoderLayers + _options.NumDecoderLayers + _options.AcousticLayers,
        };
        metadata.AdditionalInfo["Architecture"] = "Pheme";
        metadata.AdditionalInfo["SampleRate"] = _options.SampleRate.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(Pheme<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        if (disposing) (_codec?.Codec as IDisposable)?.Dispose();
        base.Dispose(disposing);
    }
}
