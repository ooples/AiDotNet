using AiDotNet.Attributes;
using AiDotNet.Audio.Codecs;
using AiDotNet.Audio.Foundations;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.Interfaces;
using AiDotNet.TextToSpeech.Latest;
using AiDotNet.Tokenization.Algorithms;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>
/// Llasa (Ye et al. 2025): text-to-speech as plain language modeling — a Llama model reads the text in a chat turn and
/// answers with X-codec2 speech tokens, one per 20 ms, which X-codec2's decoder turns into 16 kHz audio.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>The paper (§2.1–2.2, §3.2.1) fixes the design: a single Transformer initialized from Llama, the text tokens on the
/// left and the speech tokens on the right, the loss on the speech tokens, and X-codec2 (a w2v-BERT 2.0 semantic encoder
/// and a convolutional acoustic encoder fused and quantized by one FSQ codebook, decoded by a Transformer Vocos). The
/// official code fills the rest: the chat prompt and sampling of the HKUSTAudio/Llasa-1B model card, the training rows of
/// zhenye234/LLaSA_training (<c>train_tts.py</c>, <c>get_memmap_from_token.py</c>), and the codec of the xcodec2 0.1.5
/// package.</para>
/// <para>A turn is Llama 3.2's chat template: the system header with its dates, the user's
/// "Convert the text to speech:&lt;|TEXT_UNDERSTANDING_START|&gt;{text}&lt;|TEXT_UNDERSTANDING_END|&gt;", and the assistant's
/// "&lt;|SPEECH_GENERATION_START|&gt;" followed by speech tokens <c>&lt;|s_k|&gt;</c> (id 128264 + k) until
/// "&lt;|SPEECH_GENERATION_END|&gt;". Given a recording and its transcript as a prompt, the transcript goes before the text
/// and the recording's codes open the answer, and the model continues in that voice.</para>
/// <para><b>For Beginners:</b> Llasa treats speech as a language a chatbot can write: asked to "convert the text to
/// speech", it answers in sound tokens, and a neural codec turns those tokens into audio.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);
/// var model = new Llasa&lt;float&gt;(architecture, new LlasaOptions());
/// model.LoadReferenceWeights("Llasa-1B");               // model.safetensors (and tokenizer.json)
/// model.LoadCodecWeights("xcodec2/model.safetensors");
/// var audio = model.Synthesize("Dealing with family secrets is never easy.");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "Llasa: Scaling Train-Time and Inference-Time Compute for Llama-based Speech Synthesis",
    "https://arxiv.org/abs/2502.04128",
    Year = 2025,
    Authors = "Ye et al."
)]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 5e-5, Beta1 = 0.9, Beta2 = 0.95, Epsilon = 1e-8, WeightDecay = 0.01,
    Source = "Ye et al. 2025, §3.2.1: 3 epochs at a maximum learning rate of 5e-5, a cosine schedule with warm-up over 3% "
             + "of an epoch and a final rate of 10% of the peak, sequences cropped to 2048 tokens. The optimizer is the "
             + "official recipe's (zhenye234/LLaSA_training train_tts.py with config.json): Hugging Face's Trainer with fused "
             + "AdamW, beta2 0.95 and weight decay 0.01 (not on the RMSNorm weights), betas and eps otherwise the Trainer's "
             + "defaults, gradient norm clipped to 1.0. No batch size is declared: the paper gives it as 2 million tokens.")]
public partial class Llasa<T> : TtsModelBase<T>, ICodecTts<T>
{
    private readonly LlasaOptions _options;
    private readonly bool _useNativeMode;
    private bool _disposed;

    private LlamaNetwork<T>? _lm;
    private XCodec2<T>? _codec;
    private ByteLevelBpeTokenizer? _tokenizer;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;

    private readonly List<LayerBase<T>> _lmLayers = new();
    private readonly List<LayerBase<T>> _codecLayers = new();

    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    /// <summary>Creates an ONNX-backed Llasa for inference.</summary>
    public Llasa(NeuralNetworkArchitecture<T> architecture, string modelPath, LlasaOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new LlasaOptions();
        Options = _options;
        _useNativeMode = false;
        ConfigureBase();
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a native, trainable Llasa.</summary>
    public Llasa(NeuralNetworkArchitecture<T> architecture, LlasaOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new LlasaOptions();
        Options = _options;
        _useNativeMode = true;
        ConfigureBase();
        InitializeLayers();
    }

    private void ConfigureBase()
    {
        var o = _options;
        base.SampleRate = o.SampleRate;
        base.HopSize = o.CodecEncoderRates.Aggregate(1, (a, b) => a * b);
        base.HiddenDim = o.HiddenDim;
        if (o.VocabSize < LlasaTokens.VocabularySize)
            throw new ArgumentException($"Llasa's tokens need a vocabulary of at least {LlasaTokens.VocabularySize}.");
        if (o.ChatDate is not null && o.ChatDate.Length == 0) throw new ArgumentException("The chat date cannot be empty.");
        // Hugging Face's Trainer clips the gradient norm to 1.0 (max_grad_norm), which the optimizer does here.
        MaxGradNorm = NumOps.Zero;
    }

    int ITtsModel<T>.SampleRate => _options.SampleRate;

    /// <inheritdoc />
    public int MaxTextLength => _options.MaxTextLength;

    /// <inheritdoc />
    /// <remarks>X-codec2's one codebook.</remarks>
    public int NumCodebooks => 1;

    /// <inheritdoc />
    public int CodebookSize => LlasaTokens.CodebookSize;

    /// <summary>Codes per second (50 at 16 kHz).</summary>
    public int CodecFrameRate => (int)Math.Round(_options.SampleRate / (double)HopSize);

    /// <inheritdoc />
    /// <remarks>Llasa trains on recordings: the recording (16 kHz) or its codes (<see cref="TtsTrainingSample{T}.CodecTokens"/>,
    /// <c>[frames, 1]</c> as <see cref="EncodeToTokens"/> gives them), and the transcript as the language model's token ids
    /// (<see cref="TtsTrainingSample{T}.Tokens"/>, from <see cref="TtsModelBase{T}.TextToTokens"/>).</remarks>
    protected override TtsSupervision RequiredSupervision => TtsSupervision.CodecTokens;

    /// <inheritdoc />
    public override bool TrainsOnSampleObjective => true;

    /// <inheritdoc />
    public override int CodecTokenCodebooks => 1;

    /// <inheritdoc />
    public override int CodecTokenVocabulary => LlasaTokens.CodebookSize;

    private bool HasPaperLayers => _lm is not null;
    private InvalidOperationException NotNative() => new($"{GetType().Name}'s networks exist in native mode only.");

    internal LlamaNetwork<T> LanguageModel => _lm ?? throw NotNative();
    internal XCodec2<T> Codec => _codec ?? throw NotNative();
    internal IReadOnlyList<LayerBase<T>> LanguageModelLayers => _lmLayers;
    internal IReadOnlyList<LayerBase<T>> CodecLayers => _codecLayers;

    /// <summary>The language model's tokenizer: Llama 3's with Llasa's markers and speech tokens, or one read by
    /// <see cref="LoadTokenizer"/>.</summary>
    internal ByteLevelBpeTokenizer Tokenizer => _tokenizer ??= LlasaTokens.Tokenizer;

    /// <summary>The date the chat template writes (<see cref="LlasaOptions.ChatDate"/>, or today's).</summary>
    internal string ChatDate => _options.ChatDate ?? DateTime.Now.ToString("dd MMM yyyy", System.Globalization.CultureInfo.InvariantCulture);

    /// <summary>Reads the language model's tokenizer from a Hugging Face <c>tokenizer.json</c> (a fine-tune's own).</summary>
    public void LoadTokenizer(string path)
    {
        var tokenizer = ByteLevelBpeTokenizer.FromJson(File.ReadAllText(path));
        if (tokenizer.VocabularySize > _options.VocabSize)
            throw new InvalidDataException($"The tokenizer has {tokenizer.VocabularySize} tokens; the model's vocabulary is {_options.VocabSize}.");
        if (tokenizer.TokenId("<|SPEECH_GENERATION_START|>") != LlasaTokens.SpeechGenerationStart || tokenizer.TokenId("<|s_0|>") != LlasaTokens.FirstSpeechToken
            || tokenizer.TokenId("<|begin_of_text|>") != Llama3Tokenizers.BeginOfText)
            throw new InvalidDataException("The tokenizer does not place Llasa's tokens where the model expects them.");
        _tokenizer = tokenizer;
    }

    /// <inheritdoc />
    public override LayerInputDomain GetInputDomain(int[]? inputShape) =>
        HasPaperLayers ? LayerInputDomain.Indices(_options.VocabSize) : base.GetInputDomain(inputShape);

    /// <inheritdoc />
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            AddUnsplitLayers(Architecture.Layers);
            return;
        }
        var o = _options;
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(Architecture.RandomSeed ?? o.SamplingSeed);
        _lm = new LlamaNetwork<T>(Engine, _lmLayers, new LlamaConfiguration
        {
            VocabSize = o.VocabSize, Hidden = o.HiddenDim, Layers = o.Layers, Heads = o.Heads, KeyValueHeads = o.KeyValueHeads, HeadDim = o.HeadDim,
            Intermediate = o.Intermediate, RmsEpsilon = o.RmsEpsilon, RopeTheta = o.RopeTheta, RopeScalingFactor = o.RopeScalingFactor,
            RopeLowFrequencyFactor = o.RopeLowFrequencyFactor, RopeHighFrequencyFactor = o.RopeHighFrequencyFactor,
            RopeOriginalMaxPositions = o.RopeOriginalMaxPositions, TieWordEmbeddings = true,
        });
        _lm.InitializeLikeReference(random);
        _codec = new XCodec2<T>(Engine, new XCodec2Configuration
        {
            SampleRate = o.SampleRate, EncoderDim = o.CodecEncoderDim, EncoderRates = (int[])o.CodecEncoderRates.Clone(), AcousticDim = o.CodecAcousticDim,
            SemanticDim = o.CodecSemanticDim, SemanticLayer = o.SemanticLayer, DecoderDim = o.CodecDecoderDim, DecoderDepth = o.CodecDecoderDepth,
            DecoderHeads = o.CodecDecoderHeads,
            Semantic = new Wav2Vec2BertConfiguration
            {
                HiddenSize = o.SemanticHiddenSize, Heads = o.SemanticHeads, IntermediateSize = o.SemanticIntermediateSize,
                Layers = Math.Max(o.SemanticLayer, 1),
            },
        }, _codecLayers);
        AddUnsplitLayers(_lmLayers.Cast<ILayer<T>>().ToList());
        ComponentLayers.AddRange(_codecLayers);
    }

    // ---------------------------------------------------------------- the codec

    private static double[] Doubles(Tensor<T> audio) => audio.ToArray().Select(v => Ops.ToDouble(v)).ToArray();

    private static Tensor<T> Wave(double[] samples) => new(new[] { samples.Length }, new Vector<T>(samples.Select(v => Ops.FromDouble(v)).ToArray()));

    /// <inheritdoc />
    /// <remarks>The codes of a 16 kHz recording as X-codec2 makes them (<c>encode_code</c>): <c>[frames, 1]</c>, each in
    /// [0, 65536).</remarks>
    public Tensor<T> EncodeToTokens(Tensor<T> audio)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers) throw NotNative();
        var codes = Codec.Encode(Doubles(audio));
        var tokens = new Tensor<T>(new[] { codes.Length, 1 });
        for (int f = 0; f < codes.Length; f++) tokens[f, 0] = Ops.FromDouble(codes[f]);
        return tokens;
    }

    /// <inheritdoc />
    /// <remarks>The audio of codes <c>[frames, 1]</c> (<c>decode_code</c>).</remarks>
    public Tensor<T> DecodeFromTokens(Tensor<T> tokens)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers) throw NotNative();
        var codes = CodesOf(tokens);
        return Wave(codes.Length == 0 ? Array.Empty<double>() : Codec.Decode(codes));
    }

    private static int[] CodesOf(Tensor<T> tokens)
    {
        if (tokens.Rank != 2 || tokens.Shape[1] != 1) throw new ArgumentException("Llasa's codes are [frames, 1].", nameof(tokens));
        var codes = new int[tokens.Shape[0]];
        for (int f = 0; f < codes.Length; f++)
        {
            codes[f] = (int)Math.Round(Ops.ToDouble(tokens[f, 0]));
            if (codes[f] < 0 || codes[f] >= LlasaTokens.CodebookSize)
                throw new ArgumentException($"Codes lie in [0, {LlasaTokens.CodebookSize}).", nameof(tokens));
        }
        return codes;
    }

    // ---------------------------------------------------------------- synthesis

    /// <inheritdoc />
    /// <remarks>The text's token ids in the language model's tokenizer (synthesis builds the prompt from their text).</remarks>
    protected override Tensor<T> PreprocessText(string text)
    {
        if (!HasPaperLayers) return base.PreprocessText(text);
        var ids = Tokenizer.Encode(text);
        if (ids.Length == 0) throw new ArgumentException("The text has no tokens.", nameof(text));
        return new Tensor<T>(new[] { ids.Length }, new Vector<T>(ids.Select(i => NumOps.FromDouble(i)).ToArray()));
    }

    private string TextOf(Tensor<T> ids) =>
        Tokenizer.Decode(ids.ToArray().Select(v => (int)Math.Round(NumOps.ToDouble(v))), skipSpecial: true);

    /// <inheritdoc />
    /// <remarks>The model card's generation: the chat prompt of the text, sampled until the end of speech or
    /// <see cref="LlasaOptions.MaxLength"/> tokens in all, the speech tokens' codes decoded by X-codec2. Empty when the
    /// model writes no speech token.</remarks>
    public override Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed();
        if (text is null) throw new ArgumentNullException(nameof(text));
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(PreprocessText(text));
        if (!HasPaperLayers) return base.Synthesize(text);
        return Wave(Decode(LlasaTokens.ReadCodes(Generate(LlasaTokens.Prompt(Tokenizer, text, ChatDate, Array.Empty<int>())))));
    }

    /// <summary>
    /// Speaks <paramref name="text"/> in the voice of a 16 kHz recording whose transcript is <paramref name="promptText"/>,
    /// as the model card's prompted synthesis does: the transcript and the text joined by a space between the
    /// text-understanding markers, the recording's codes opening the answer, and the continuation decoded with them.
    /// </summary>
    /// <remarks>With <see cref="LlasaOptions.IncludePromptAudio"/> (the card's output) the audio starts with the decoded
    /// prompt; otherwise the prompt's samples are cut off, as the card's commented alternative does.</remarks>
    public Tensor<T> SynthesizeWithPrompt(string text, Tensor<T> promptAudio, string promptText)
    {
        ThrowIfDisposed();
        if (text is null) throw new ArgumentNullException(nameof(text));
        if (promptAudio is null) throw new ArgumentNullException(nameof(promptAudio));
        if (promptText is null) throw new ArgumentNullException(nameof(promptText));
        if (!HasPaperLayers) throw NotNative();
        var prefix = Codec.Encode(Doubles(promptAudio));
        var generated = LlasaTokens.ReadCodes(Generate(LlasaTokens.Prompt(Tokenizer, promptText + " " + text, ChatDate, prefix)));
        var audio = Decode(prefix.Concat(generated).ToArray());
        if (_options.IncludePromptAudio) return Wave(audio);
        return Wave(audio.Skip(Math.Min(promptAudio.Length, audio.Length)).ToArray());
    }

    private double[] Decode(IReadOnlyList<int> codes) => codes.Count == 0 ? Array.Empty<double>() : Codec.Decode(codes);

    /// <inheritdoc />
    /// <remarks><paramref name="input"/> is a text's token ids (<see cref="PreprocessText"/>); the waveform
    /// <see cref="Synthesize"/> makes of their text.</remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input);
        if (!HasPaperLayers) return base.PredictCore(input);
        if (input.Length == 0) throw new ArgumentException("There is no text.", nameof(input));
        if (input.ToArray().Any(v => NumOps.ToDouble(v) < 0 || NumOps.ToDouble(v) >= _options.VocabSize))
            throw new ArgumentOutOfRangeException(nameof(input), $"Token ids lie in [0, {_options.VocabSize}).");
        return Synthesize(TextOf(input));
    }

    /// <inheritdoc />
    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;

    // Hugging Face generate() as the card calls it: sampling at the temperature with top-p (1: no cut), until the end of
    // speech (not kept) or MaxLength tokens in all. The generated tokens. (The card then drops the last generated token,
    // which is the end of speech when the model writes one; here the end of speech is never kept, so no code is dropped
    // when generation stops at the length limit instead.)
    internal List<int> Generate(IReadOnlyList<int> prompt)
    {
        var o = _options;
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(o.SamplingSeed);
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            var lm = LanguageModel;
            var ids = new List<int>(prompt);
            var generated = new List<int>();
            if (ids.Count >= o.MaxLength) return generated;
            var cache = new LlamaCache<T>();
            var logits = lm.Forward(lm.Embed(Row(prompt)), null, null, cache, lastLogitsOnly: true).Logits!;
            while (ids.Count < o.MaxLength)
            {
                var values = logits.ToArray();
                var scores = new double[values.Length];
                for (int i = 0; i < scores.Length; i++) scores[i] = NumOps.ToDouble(values[i]);
                if (o.ConstrainToSpeechTokens) LlasaTokens.MaskToSpeech(scores);
                int next;
                if (o.Temperature > 0)
                    next = OuteTTSSampler.Sample(OuteTTSSampler.Process(scores, ids, 1.0, 0, o.Temperature, 0, o.TopP, 0), random);
                else
                    next = Array.IndexOf(scores, scores.Max());
                if (next == LlasaTokens.SpeechGenerationEnd) break;
                ids.Add(next);
                generated.Add(next);
                if (ids.Count >= o.MaxLength) break;
                logits = lm.Forward(lm.Embed(Row(new[] { next })), null, null, cache, lastLogitsOnly: true).Logits!;
            }
            return generated;
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    private static int[,] Row(IReadOnlyList<int> ids)
    {
        var row = new int[1, ids.Count];
        for (int i = 0; i < ids.Count; i++) row[0, i] = ids[i];
        return row;
    }

    // ---------------------------------------------------------------- weights

    /// <summary>
    /// Loads a released Llasa model directory: its <c>*.safetensors</c> (the Hugging Face <c>LlamaForCausalLM</c> state
    /// dict) and, when present, its <c>tokenizer.json</c>.
    /// </summary>
    public void LoadReferenceWeights(string directory)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers) throw NotNative();
        string tokenizer = Path.Combine(directory, "tokenizer.json");
        if (File.Exists(tokenizer)) LoadTokenizer(tokenizer);
        var weights = new Dictionary<string, double[]>(StringComparer.Ordinal);
        foreach (var file in Directory.GetFiles(directory, "*.safetensors"))
            foreach (var entry in new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(file))
                weights[entry.Key] = entry.Value.ToVector().Select(v => (double)v).ToArray();
        if (weights.Count == 0) throw new FileNotFoundException($"No .safetensors files in {directory}.");
        LanguageModel.LoadTorchWeights((name, shape) => weights.TryGetValue(name, out var v) ? v
            : throw new InvalidDataException($"The checkpoint has no tensor '{name}'."));
    }

    /// <summary>Loads X-codec2: HKUSTAudio/xcodec2's <c>model.safetensors</c> (or its <c>pytorch_model.bin</c>).</summary>
    public void LoadCodecWeights(string path)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers) throw NotNative();
        var state = new Dictionary<string, double[]>(StringComparer.Ordinal);
        foreach (var entry in new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(path))
            state[entry.Key] = entry.Value.ToVector().Select(v => (double)v).ToArray();
        LoadCodec(name => state.TryGetValue(name, out var v) ? v : null, state.Keys);
    }

    internal void LoadCodec(Func<string, double[]?> read, IEnumerable<string> names)
    {
        var used = Codec.Load(read);
        // The anti-aliasing filters are recomputed, w2v-BERT's layers past the one X-codec2 reads are never run, and its
        // masked-span embedding is for pretraining only.
        int layer = Codec.Configuration.SemanticLayer;
        var unused = names.Where(n => !used.Contains(n) && !n.EndsWith(".filter", StringComparison.Ordinal)
            && n != "semantic_model.masked_spec_embed" && !IsLaterSemanticLayer(n, layer)).ToList();
        if (unused.Count > 0)
            throw new InvalidDataException($"The codec checkpoint has {unused.Count} tensors this model has no place for (first: '{unused[0]}').");
    }

    private static bool IsLaterSemanticLayer(string name, int built)
    {
        const string prefix = "semantic_model.encoder.layers.";
        if (!name.StartsWith(prefix, StringComparison.Ordinal)) return false;
        int dot = name.IndexOf('.', prefix.Length);
        return dot > 0 && int.TryParse(name.Substring(prefix.Length, dot - prefix.Length), out int index) && index >= built;
    }

    // ---------------------------------------------------------------- training

    /// <inheritdoc />
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        Train(new TtsTrainingSample<T> { Tokens = input, Audio = expectedOutput });
        return true;
    }

    /// <inheritdoc />
    /// <remarks>The recipe's training row of the transcript and the recording's codes in the chat template, cropped to 2048
    /// tokens, and the causal language-model loss on the answer from the speech-generation marker on.</remarks>
    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        if (!HasPaperLayers) throw new NotSupportedException("The objective needs the release's networks; this model was built from caller-supplied layers.");
        var row = TrainingRow(sample);
        return TrainWithCustomObjective(sample.Tokens, sample.Tokens, (_, _) => Loss(row), Optimizer());
    }

    /// <inheritdoc />
    public override T EvaluateTrainingObjective(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        Validation.Guard.NotNull(sample);
        if (!HasPaperLayers) throw new NotSupportedException("The objective needs the release's networks; this model was built from caller-supplied layers.");
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            return Loss(TrainingRow(sample))[0];
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    // Recordings' codes (X-codec2 runs once per recording).
    private readonly Dictionary<Tensor<T>, int[]> _codes = new(Helpers.TensorReferenceComparer<Tensor<T>>.Instance);

    internal LlasaTrainingRow TrainingRow(TtsTrainingSample<T> sample)
    {
        Validation.Guard.NotNull(sample);
        int[] codes;
        if (sample.CodecTokens is not null)
        {
            codes = CodesOf(sample.CodecTokens);
        }
        else
        {
            var audio = sample.Audio ?? throw new ArgumentException($"{GetType().Name} trains on the recording or its codes; set Audio or CodecTokens.", nameof(sample));
            if (!_codes.TryGetValue(audio, out codes!))
            {
                codes = Codec.Encode(Doubles(audio));
                if (_codes.Count > 64) _codes.Clear();
                _codes[audio] = codes;
            }
        }
        return LlasaTokens.TrainingRow(Tokenizer, sample.Text ?? TextOf(sample.Tokens), ChatDate, codes, _options.MaxLength);
    }

    internal Tensor<T> Loss(LlasaTrainingRow row)
    {
        var lm = LanguageModel;
        var ids = Row(row.Ids);
        var mask = new bool[1, row.Ids.Length];
        var labels = new int[1, row.Ids.Length];
        for (int i = 0; i < row.Ids.Length; i++)
        {
            mask[0, i] = row.Mask[i];
            labels[0, i] = row.Labels[i];
        }
        var (_, logits) = lm.ForwardWithLogits(lm.Embed(ids), mask);
        return lm.CausalLmLoss(logits, labels);
    }

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> Optimizer()
    {
        if (_optimizer is not null) return _optimizer;
        var o = _options;
        var options = new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = o.LearningRate, Beta1 = o.Beta1, Beta2 = o.Beta2, Epsilon = o.Epsilon, WeightDecay = o.WeightDecay,
            EnableGradientClipping = o.MaxGradNorm > 0, MaxGradientNorm = o.MaxGradNorm,
        };
        if (o.TrainingSteps > 0)
        {
            int warmup = (int)Math.Ceiling(o.TrainingSteps / (double)Math.Max(1, o.Epochs) * o.WarmupEpochFraction);
            options.LearningRateScheduler = new LinearWarmupScheduler(o.LearningRate, warmup, Math.Max(o.TrainingSteps, warmup), 0.0,
                LinearWarmupScheduler.DecayMode.Cosine, o.LearningRate * o.FinalLearningRateFraction);
            options.SchedulerStepMode = SchedulerStepMode.StepPerBatch;
        }
        // Hugging Face's Trainer decays every parameter but biases and normalization weights (Llama has no biases; its
        // RMSNorm weights are exempt).
        var norms = new HashSet<Tensor<T>>(Helpers.TensorReferenceComparer<Tensor<T>>.Instance);
        var lm = LanguageModel;
        var normLayers = lm.Layers.SelectMany(l => new[] { l.InputNorm, l.PostNorm, l.QueryNorm, l.KeyNorm }).Append(lm.FinalNorm)
            .Where(l => l is not null).Cast<ILayer<T>>().ToList();
        foreach (var p in Training.TapeTrainingStep<T>.CollectParameters(normLayers, -1)) norms.Add(p);
        var parameters = Training.TapeTrainingStep<T>.CollectParameters(_lmLayers.Cast<ILayer<T>>().ToList(), -1).ToList();
        var decays = new List<T>();
        foreach (var p in parameters)
            for (int i = 0; i < p.Length; i++) decays.Add(norms.Contains(p) ? Ops.Zero : Ops.One);
        options.WeightDecayMask = new Vector<T>(decays.ToArray());
        options.MaskParameters = parameters;
        return _optimizer = PaperOptimizerFactory.VerifyHandBuilt(this, new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this, options));
    }

    /// <inheritdoc />
    /// <remarks>Training updates the language model; X-codec2 is frozen.</remarks>
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        if (!HasPaperLayers) return parameters;
        var trained = new HashSet<Tensor<T>>(Training.TapeTrainingStep<T>.CollectParameters(_lmLayers.Cast<ILayer<T>>().ToList(), -1),
            Helpers.TensorReferenceComparer<Tensor<T>>.Instance);
        return parameters.Where(trained.Contains).ToList();
    }

    // ---------------------------------------------------------------- housekeeping

    /// <inheritdoc />
    protected override bool SupportsParameterMutation => _useNativeMode;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "Llasa-Native" : "Llasa-ONNX",
            Description = "Llasa: text-to-speech as Llama language modeling over X-codec2 codes (Ye et al., 2025)",
            FeatureCount = _options.HiddenDim,
        };
        metadata.AdditionalInfo["Architecture"] = "Llasa";
        metadata.AdditionalInfo["SampleRate"] = _options.SampleRate.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(Llasa<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }
}

/// <summary>A training row: token ids, labels (-100 where not scored) and the attention mask over keys.</summary>
internal sealed record LlasaTrainingRow(int[] Ids, int[] Labels, bool[] Mask);

/// <summary>
/// Llasa's tokens (HKUSTAudio/Llasa-1B's tokenizer and zhenye234/LLaSA_training): Llama 3's tokenizer with eight task
/// markers at 128256–128263 and the speech tokens <c>&lt;|s_0|&gt;</c>…<c>&lt;|s_65535|&gt;</c> after them; the chat prompt,
/// the training row, and reading codes back.
/// </summary>
internal static class LlasaTokens
{
    public const int EndOfTurn = 128009;
    public const int TextUnderstandingStart = Llama3Tokenizers.FirstAddedId + 2;
    public const int TextUnderstandingEnd = Llama3Tokenizers.FirstAddedId + 3;
    public const int SpeechGenerationStart = Llama3Tokenizers.FirstAddedId + 4;
    public const int SpeechGenerationEnd = Llama3Tokenizers.FirstAddedId + 5;
    public const int FirstSpeechToken = Llama3Tokenizers.FirstAddedId + 8;
    public const int CodebookSize = 65536;

    /// <summary>The tokenizer's size, 193,800.</summary>
    public const int VocabularySize = FirstSpeechToken + CodebookSize;

    private static readonly string[] Markers =
    {
        "<|TEXT_GENERATION_START|>", "<|TEXT_GENERATION_END|>", "<|TEXT_UNDERSTANDING_START|>", "<|TEXT_UNDERSTANDING_END|>",
        "<|SPEECH_GENERATION_START|>", "<|SPEECH_GENERATION_END|>", "<|SPEECH_UNDERSTANDING_START|>", "<|SPEECH_UNDERSTANDING_END|>",
    };

    private static readonly Lazy<ByteLevelBpeTokenizer> TokenizerInstance = new(() => Llama3Tokenizers.Create(
        Markers.Select((m, i) => (m, Llama3Tokenizers.FirstAddedId + i, false))
            .Concat(Enumerable.Range(0, CodebookSize).Select(k => ($"<|s_{k}|>", FirstSpeechToken + k, false)))), isThreadSafe: true);

    /// <summary>Llasa's tokenizer.</summary>
    public static ByteLevelBpeTokenizer Tokenizer => TokenizerInstance.Value;

    // Llama 3.2's chat template up to the assistant's content: the system header (its knowledge-cutoff and current dates,
    // no system message) and the user's turn.
    private static string Chat(string text, string date) =>
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: " + date
        + "\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nConvert the text to speech:<|TEXT_UNDERSTANDING_START|>" + text
        + "<|TEXT_UNDERSTANDING_END|><|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n<|SPEECH_GENERATION_START|>";

    /// <summary>The model card's prompt: the chat with the assistant's turn continued after the speech-generation marker and
    /// <paramref name="prefix"/>'s speech tokens (<c>apply_chat_template(continue_final_message=True)</c>).</summary>
    public static int[] Prompt(ByteLevelBpeTokenizer tokenizer, string text, string date, IReadOnlyList<int> prefix) =>
        tokenizer.Encode(Chat(text, date)).Concat(prefix.Select(code => FirstSpeechToken + code)).ToArray();

    /// <summary>
    /// The recipe's training row (<c>process_audio_id</c> and <c>TTSDataset.__getitem__</c>): the chat with the answer
    /// "speech-generation start, the codes, speech-generation end" and the template's closing end of turn, cropped to
    /// <paramref name="maxLength"/>; labels from the speech-generation marker on; and the attention mask, which is false at
    /// every end of turn because the recipe pads with it. Ends of turn are never scored.
    /// </summary>
    public static LlasaTrainingRow TrainingRow(ByteLevelBpeTokenizer tokenizer, string text, string date, IReadOnlyList<int> codes, int maxLength)
    {
        var ids = tokenizer.Encode(Chat(text, date)).ToList();
        int start = ids.Count - 1;
        ids.AddRange(codes.Select(code => FirstSpeechToken + code));
        ids.Add(SpeechGenerationEnd);
        ids.Add(EndOfTurn);
        if (ids.Count > maxLength) ids.RemoveRange(maxLength, ids.Count - maxLength);
        var labels = ids.Select((id, i) => i >= start && id != EndOfTurn ? id : -100).ToArray();
        return new LlasaTrainingRow(ids.ToArray(), labels, ids.Select(id => id != EndOfTurn).ToArray());
    }

    /// <summary>The codes of generated tokens (the card's <c>extract_speech_ids</c>): each speech token's code, every
    /// other token skipped.</summary>
    public static List<int> ReadCodes(IEnumerable<int> tokens) =>
        tokens.Where(t => t >= FirstSpeechToken && t < FirstSpeechToken + CodebookSize).Select(t => t - FirstSpeechToken).ToList();

    /// <summary>Removes every token but the speech tokens and the end of speech.</summary>
    public static void MaskToSpeech(double[] scores)
    {
        for (int i = 0; i < scores.Length; i++)
            if ((i < FirstSpeechToken || i >= FirstSpeechToken + CodebookSize) && i != SpeechGenerationEnd) scores[i] = double.NegativeInfinity;
    }
}
