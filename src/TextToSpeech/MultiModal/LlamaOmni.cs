using AiDotNet.Attributes;
using AiDotNet.Audio.Foundations;
using AiDotNet.Audio.Whisper;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.Latest;
using AiDotNet.TextToSpeech.Vocoders;
using AiDotNet.Tokenization.Algorithms;

namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>A LLaMA-Omni response: its text, its speech units and its 16 kHz audio.</summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public sealed class LlamaOmniResponse<T>
{
    /// <summary>The text response.</summary>
    public string Text { get; init; } = string.Empty;

    /// <summary>The discrete units of the speech response.</summary>
    public int[] Units { get; init; } = Array.Empty<int>();

    /// <summary>The speech response; empty when there are no units.</summary>
    public Tensor<T> Audio { get; init; } = new(new[] { 0 });
}

/// <summary>
/// LLaMA-Omni (Fang et al., ICLR 2025): low-latency speech interaction with an LLM. A frozen Whisper encoder and a speech
/// adaptor feed speech to Llama-3.1-8B-Instruct, which writes the text response; a non-autoregressive streaming speech
/// decoder reads the LLM's hidden states as they come and writes discrete units by CTC, which a unit vocoder voices.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>Speech in</b> (§2.1–2.2): whisper-large-v3's encoder on the 30-second log-mel window, every 5 frames
/// concatenated and mapped by a 2-layer ReLU perceptron into the LLM's embeddings, in place of <c>&lt;speech&gt;</c> in
/// the llama_3 prompt.</para>
/// <para><b>Speech out</b> (§2.4): each hidden state c_i (the one that predicts response token y_i) is repeated λ = 25
/// times, projected, and passed through two causal Llama layers to logits over K = 1000 mHuBERT units and a blank;
/// the best alignment, collapsed, is the unit sequence; fairseq's unit HiFi-GAN with a duration predictor voices it,
/// chunk by chunk of Ω units while the text is still being written (Algorithm 1), or all at once (Ω = +∞).</para>
/// <para><b>Training</b> (§2.5, §4.1): stage 1 trains the adaptor and the LLM on the text response; stage 2 trains only
/// the speech decoder by CTC on the deduplicated units of the response's speech.</para>
/// <para><see cref="Synthesize"/> reads a text aloud: the text is the user's turn and is fed as the response, and the
/// speech decoder voices it. <see cref="Chat"/> answers 16 kHz speech.</para>
/// <para><b>For Beginners:</b> LLaMA-Omni is a talking chatbot built on Llama 3. It listens to your question, writes
/// its answer, and speaks it at the same time.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);
/// var model = new LlamaOmni&lt;float&gt;(architecture, new LlamaOmniOptions { TokenizerPath = "Llama-3.1-8B-Omni/tokenizer.json" });
/// model.LoadReleasedWeights("Llama-3.1-8B-Omni");          // speech encoder, adaptor, LLM and speech decoder
/// model.LoadVocoder("vocoder/g_00500000");
/// var response = model.Chat(question16kHz);                // response.Text, response.Audio
/// var audio = model.Synthesize("Hello world.");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "LLaMA-Omni: Seamless Speech Interaction with Large Language Models",
    "https://arxiv.org/abs/2409.06666",
    Year = 2025,
    Authors = "Qingkai Fang, Shoutao Guo, Yan Zhou, Zhengrui Ma, Shaolei Zhang, Yang Feng"
)]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 2e-4, WarmupFraction = 0.03, Schedule = LearningRateSchedulerType.CosineAnnealing,
                WeightDecay = 0.0, Beta1 = 0.9, Beta2 = 0.999, MaxGradientNorm = 1.0,
                Source = "Fang et al. 2025 §4.1: stage 2 trains the speech decoder with a cosine schedule warming up over the first "
                        + "3% of steps to a peak of 2e-4, batch 32 for 3 epochs (stage 1: the same at 2e-5). The optimizer is "
                        + "the release's Hugging Face Trainer default, AdamW (0.9, 0.999), no weight decay, clipping at 1.0.")]
public partial class LlamaOmni<T> : TtsModelBase<T>
{
    private readonly LlamaOmniOptions _options;
    private readonly bool _useNativeMode;
    private bool _disposed;
    private OpenAiWhisperNetwork<T>? _encoder;
    private LlamaOmniAdaptor<T>? _adaptor;
    private LlamaNetwork<T>? _lm;
    private LlamaOmniSpeechDecoder<T>? _speechDecoder;
    private CodeHiFiGan<T>? _vocoder;
    private HubertUnitExtractor<T>? _units;
    private ByteLevelBpeTokenizer? _tokenizer;
    private readonly List<LayerBase<T>> _encoderLayers = new();
    private readonly List<LayerBase<T>> _adaptorLayers = new();
    private readonly List<LayerBase<T>> _lmLayers = new();
    private readonly List<LayerBase<T>> _decoderLayers = new();
    private readonly List<LayerBase<T>> _vocoderLayers = new();
    private readonly List<LayerBase<T>> _hubertLayers = new();

    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    /// <summary>Creates an ONNX-backed LLaMA-Omni for inference.</summary>
    public LlamaOmni(NeuralNetworkArchitecture<T> architecture, string modelPath, LlamaOmniOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new LlamaOmniOptions();
        Options = _options;
        _useNativeMode = false;
        ConfigureBase();
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a native, trainable LLaMA-Omni.</summary>
    public LlamaOmni(NeuralNetworkArchitecture<T> architecture, LlamaOmniOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new LlamaOmniOptions();
        Options = _options;
        _useNativeMode = true;
        ConfigureBase();
        InitializeLayers();
    }

    private void ConfigureBase()
    {
        var o = _options;
        base.SampleRate = o.SampleRate;
        base.HiddenDim = o.HiddenDim;
        if (o.HiddenDim <= 0 || o.Layers <= 0 || o.NumHeads <= 0 || o.HiddenDim % o.NumHeads != 0 || o.KeyValueHeads <= 0 || o.NumHeads % o.KeyValueHeads != 0)
            throw new ArgumentException("The language model's width must divide into its heads, and the heads into the key-value heads.");
        if (o.UpsampleFactor <= 0 || o.AdaptorDownsample <= 0) throw new ArgumentException("The upsample factor and the adaptor's downsampling must be positive.");
        if (o.MaxNewTokens <= 0) throw new ArgumentException("MaxNewTokens must be positive.");
        if (o.Temperature < 0 || o.TopP <= 0 || o.TopP > 1) throw new ArgumentException("The temperature must be at least 0 and top-p in (0, 1].");
        if (o.UnitChunkSize < 0) throw new ArgumentException("UnitChunkSize must be 0 (offline) or positive.");
        if (o.VocoderUpsampleRates.Length != o.VocoderUpsampleKernels.Length) throw new ArgumentException("The vocoder needs one kernel per upsampling rate.");
    }

    /// <summary>The longest sequence the language model takes (the release's 2048).</summary>
    public int MaxTextLength => _options.MaxTextLength;

    /// <inheritdoc />
    /// <remarks>The release's vocoder speaks with one voice.</remarks>
    protected override TtsSupervision RequiredVoice => TtsSupervision.None;

    private bool HasPaperLayers => _lm is not null;
    private int SeedValue => _options.Seed ?? 42;
    private InvalidOperationException NotNative() => new($"{GetType().Name}'s networks exist in native mode only.");

    internal OpenAiWhisperNetwork<T> SpeechEncoder => _encoder ?? throw NotNative();
    internal LlamaOmniAdaptor<T> Adaptor => _adaptor ?? throw NotNative();
    internal LlamaNetwork<T> LanguageModel => _lm ?? throw NotNative();
    internal LlamaOmniSpeechDecoder<T> SpeechDecoder => _speechDecoder ?? throw NotNative();
    internal CodeHiFiGan<T> Vocoder => _vocoder ?? throw NotNative();
    internal HubertUnitExtractor<T> UnitExtractor => _units ?? throw NotNative();

    /// <inheritdoc />
    /// <remarks>Text token ids, those of Llama 3's vocabulary.</remarks>
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
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(Architecture.RandomSeed ?? SeedValue);
        _encoder = new OpenAiWhisperNetwork<T>(Engine, _encoderLayers,
            new WhisperDimensions(o.EncoderMels, o.EncoderContext, o.EncoderWidth, o.EncoderHeads, o.EncoderLayers, 1, 1, 4, 1, 0) { EncoderOnly = true }, random);
        _adaptor = new LlamaOmniAdaptor<T>(Engine, _adaptorLayers, o.EncoderWidth, o.AdaptorDownsample, o.AdaptorHidden, o.HiddenDim, random);
        _lm = new LlamaNetwork<T>(Engine, _lmLayers, new LlamaConfiguration
        {
            VocabSize = o.VocabSize, Hidden = o.HiddenDim, Layers = o.Layers, Heads = o.NumHeads, KeyValueHeads = o.KeyValueHeads,
            Intermediate = o.Intermediate, RmsEpsilon = o.RmsEpsilon, RopeTheta = o.RopeTheta, RopeScalingFactor = o.RopeScalingFactor,
            RopeLowFrequencyFactor = o.RopeLowFrequencyFactor, RopeHighFrequencyFactor = o.RopeHighFrequencyFactor,
            RopeOriginalMaxPositions = o.RopeOriginalMaxPositions,
        });
        _lm.InitializeLikeReference(random);
        _speechDecoder = new LlamaOmniSpeechDecoder<T>(Engine, _decoderLayers, new LlamaOmniDecoderConfiguration(o.HiddenDim, o.SpeechDecoderLayers,
            o.SpeechDecoderWidth, o.SpeechDecoderHeads, o.SpeechDecoderIntermediate, o.UpsampleFactor, o.Units, o.RmsEpsilon, o.RopeTheta,
            o.RopeScalingFactor, o.RopeLowFrequencyFactor, o.RopeHighFrequencyFactor, o.RopeOriginalMaxPositions), random);
        _vocoder = new CodeHiFiGan<T>(Engine, _vocoderLayers, new CodeHiFiGanConfiguration
        {
            Units = o.Units, EmbeddingDim = o.VocoderEmbeddingDim, InitialChannels = o.VocoderChannels, UpsampleRates = (int[])o.VocoderUpsampleRates.Clone(),
            UpsampleKernels = (int[])o.VocoderUpsampleKernels.Clone(), DurationHidden = o.VocoderEmbeddingDim,
        }, random);
        var hubert = HubertUnitExtractor<T>.MHubertBase(o.HubertLayer) with
        {
            HiddenSize = o.HubertWidth, Heads = o.HubertHeads, IntermediateSize = o.HubertIntermediate,
            ConvDims = Enumerable.Repeat(o.HubertConvChannels, 7).ToArray(),
        };
        _units = new HubertUnitExtractor<T>(Engine, _hubertLayers, hubert, o.HubertLayer, o.Units);
        _units.Network.InitializeLikeReference(random);
        AddUnsplitLayers(_adaptorLayers.Concat(_lmLayers).Cast<ILayer<T>>().ToList());
        // The frozen encoder, the speech decoder, the vocoder and the unit extractor, so clones and saved models keep them.
        ComponentLayers.AddRange(_encoderLayers);
        ComponentLayers.AddRange(_decoderLayers);
        ComponentLayers.AddRange(_vocoderLayers);
        ComponentLayers.AddRange(_hubertLayers);
    }

    // ---------------------------------------------------------------- tokens and prompts

    private ByteLevelBpeTokenizer Tokenizer
    {
        get
        {
            if (_tokenizer is not null) return _tokenizer;
            string path = _options.TokenizerPath ?? throw new InvalidOperationException("Text needs Llama 3's tokenizer: set LlamaOmniOptions.TokenizerPath.");
            _tokenizer = ByteLevelBpeTokenizer.FromJson(File.ReadAllText(path));
            if (_tokenizer.VocabularySize > _options.VocabSize)
                throw new InvalidOperationException($"The tokenizer has {_tokenizer.VocabularySize} ids; the language model's vocabulary is {_options.VocabSize}.");
            return _tokenizer;
        }
    }

    /// <summary>The text token ids of <paramref name="text"/> (Llama 3's tokenizer, no special tokens added).</summary>
    public int[] TokenizeText(string text)
    {
        if (text is null) throw new ArgumentNullException(nameof(text));
        return Tokenizer.Encode(text);
    }

    /// <summary>The text of token ids, special tokens dropped.</summary>
    public string DecodeText(IEnumerable<int> ids) => Tokenizer.Decode(ids, skipSpecial: true);

    private const string SpeechPlaceholder = "<speech>";

    // The llama_3 template (conversation.py SeparatorStyle.LLAMA_3), up to the assistant's header.
    private string PromptText(string userMessage) =>
        "<|begin_of_text|>" + (_options.SystemPrompt.Length > 0 ? $"<|start_header_id|>system<|end_header_id|>\n\n{_options.SystemPrompt}<|eot_id|>" : "")
        + $"<|start_header_id|>user<|end_header_id|>\n\n{userMessage.Trim()}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";

    /// <summary>
    /// A prompt's ids split at <c>&lt;speech&gt;</c> (<c>tokenizer_speech_token</c>): the ids before the speech and those
    /// after (none without speech), each chunk tokenized on its own.
    /// </summary>
    internal (int[] Before, int[] After, bool HasSpeech) PromptIds(string userMessage)
    {
        var chunks = PromptText(userMessage).Split(new[] { SpeechPlaceholder }, StringSplitOptions.None);
        if (chunks.Length > 2) throw new ArgumentException("A prompt holds one speech input.", nameof(userMessage));
        return (Tokenizer.Encode(chunks[0]), chunks.Length == 2 ? Tokenizer.Encode(chunks[1]) : Array.Empty<int>(), chunks.Length == 2);
    }

    private int EotId => Tokenizer.TokenId("<|eot_id|>");

    // Generation ends at any of the release's eos_token_id: <|end_of_text|>, <|eom_id|>, <|eot_id|>.
    private HashSet<int> Stops => new() { Tokenizer.TokenId("<|end_of_text|>"), Tokenizer.TokenId("<|eom_id|>"), EotId };

    // ---------------------------------------------------------------- speech in

    /// <summary>
    /// The speech embeddings <c>[frames / k, d]</c> of 16 kHz audio: <c>pad_or_trim</c> to the encoder's window (30 s),
    /// Whisper's log-mel, the encoder, the adaptor (§2.1–2.2). The release keeps every frame of the padded window.
    /// </summary>
    public Tensor<T> SpeechEmbeddings(Tensor<T> recording)
    {
        ThrowIfDisposed();
        if (recording is null) throw new ArgumentNullException(nameof(recording));
        var o = _options;
        int window = o.EncoderContext * 2 * 160;
        var audio = new double[window];
        for (int i = 0; i < Math.Min(window, recording.Length); i++) audio[i] = Ops.ToDouble(recording.GetFlat(i));
        var mel = OpenAiWhisperTranscriber<T>.LogMelSpectrogram(audio, o.EncoderMels);
        int mels = mel.GetLength(0), frames = mel.GetLength(1);
        var melTensor = new Tensor<T>(new[] { mels, frames });
        for (int m = 0; m < mels; m++)
            for (int f = 0; f < frames; f++) melTensor[m, f] = Ops.FromDouble(mel[m, f]);
        Tensor<T> encoded;
        using (new NoGradScope<T>()) encoded = SpeechEncoder.Encode(melTensor);                                    // frozen (§2.1)
        return Adaptor.Forward(Engine.Reshape(encoded, new[] { encoded.Shape[1], encoded.Shape[2] }));
    }

    // The input embeddings [1, L, d] of a prompt with its speech, then the extra ids.
    private Tensor<T> Inputs((int[] Before, int[] After, bool HasSpeech) prompt, Tensor<T>? speech, IReadOnlyList<int> extra)
    {
        var parts = new List<Tensor<T>> { EmbedIds(prompt.Before) };
        if (prompt.HasSpeech) parts.Add(speech ?? throw new ArgumentException("The prompt has a speech input but no speech was given."));
        var tail = prompt.After.Concat(extra).ToArray();
        if (tail.Length > 0) parts.Add(EmbedIds(tail));
        var all = parts.Count == 1 ? parts[0] : Engine.TensorConcatenate(parts.ToArray(), 0);
        if (all.Shape[0] > _options.MaxTextLength) throw new ArgumentException($"The sequence's {all.Shape[0]} positions exceed {_options.MaxTextLength}.");
        return Engine.Reshape(all, new[] { 1, all.Shape[0], _options.HiddenDim });
    }

    private Tensor<T> EmbedIds(IReadOnlyList<int> ids)
    {
        var table = new int[1, ids.Count];
        for (int i = 0; i < ids.Count; i++)
        {
            if (ids[i] < 0 || ids[i] >= _options.VocabSize) throw new ArgumentOutOfRangeException(nameof(ids), $"Token ids lie in [0, {_options.VocabSize}).");
            table[0, i] = ids[i];
        }
        var embedded = LanguageModel.Embed(table);
        return Engine.Reshape(embedded, new[] { ids.Count, _options.HiddenDim });
    }

    // ---------------------------------------------------------------- generation

    private Random? _samplingRandom;

    private TResult Inference<TResult>(Func<TResult> run)
    {
        ThrowIfDisposed();
        bool wasTraining = IsTrainingMode;
        var outer = _samplingRandom;
        SetTrainingMode(false);
        _samplingRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(SeedValue);
        try
        {
            using var _ = new NoGradScope<T>();
            return run();
        }
        finally
        {
            _samplingRandom = outer;
            SetTrainingMode(wasTraining);
        }
    }

    private double[] Row(Tensor<T> logits)
    {
        var values = new double[_options.VocabSize];
        int offset = logits.Length - values.Length;
        for (int i = 0; i < values.Length; i++) values[i] = Ops.ToDouble(logits.GetFlat(offset + i));
        return values;
    }

    private int Choose(double[] logits) => _options.Temperature <= 0
        ? NucleusSampling.Greedy(logits)
        : NucleusSampling.Sample(logits, _options.Temperature, _options.TopP, null,
            _samplingRandom ?? throw new InvalidOperationException("Sampling happens inside an inference call."));

    /// <summary>
    /// Generates the text response after a prompt (Algorithm 1): the generated ids (an end id not kept) and the hidden
    /// states c_i, each the one that predicts the i-th token (the end's included), <c>[M, d]</c>.
    /// </summary>
    internal (List<int> Ids, Tensor<T> States) Generate(Tensor<T> inputs)
    {
        var lm = LanguageModel;
        var cache = new LlamaCache<T>();
        var (hidden, logits) = lm.Forward(inputs, null, null, cache, lastLogitsOnly: true);
        int length = inputs.Shape[1], d = _options.HiddenDim;
        var states = new List<Tensor<T>> { Engine.TensorSlice(Engine.Reshape(hidden, new[] { length, d }), new[] { length - 1, 0 }, new[] { 1, d }) };
        var ids = new List<int>();
        var stops = Stops;
        while (true)
        {
            int next = Choose(Row(logits!));
            if (stops.Contains(next) || ids.Count + 1 >= _options.MaxNewTokens)
            {
                if (!stops.Contains(next)) ids.Add(next);
                break;
            }
            ids.Add(next);
            if (cache.Length + 1 > _options.MaxTextLength) break;
            (hidden, logits) = lm.Forward(Engine.Reshape(EmbedIds(new[] { next }), new[] { 1, 1, d }), null, null, cache, lastLogitsOnly: true);
            states.Add(Engine.Reshape(hidden, new[] { 1, d }));
        }
        return (ids, Engine.TensorConcatenate(states.ToArray(), 0));
    }

    /// <summary>The last <paramref name="count"/> hidden states <c>[count, d]</c> of a fed sequence: for a response fed
    /// rather than generated, the prompt's last and each response token's, what generation would have read.</summary>
    internal Tensor<T> LastStates(Tensor<T> inputs, int count)
    {
        var (hidden, _) = LanguageModel.Forward(inputs);
        int length = inputs.Shape[1], d = _options.HiddenDim;
        var rows = Engine.Reshape(hidden, new[] { length, d });
        return Engine.TensorSlice(rows, new[] { length - count, 0 }, new[] { count, d });
    }

    /// <summary>
    /// The units of the response read from its hidden states (the best alignment, collapsed) and the waveform the vocoder
    /// makes of them: all at once, or (with <see cref="LlamaOmniOptions.UnitChunkSize"/> Ω) in the chunks Algorithm 1
    /// voices as each further state arrives.
    /// </summary>
    internal (int[] Units, Tensor<T> Audio) Speak(Tensor<T> states)
    {
        var decoder = SpeechDecoder;
        int omega = _options.UnitChunkSize, count = states.Shape[0], d = states.Shape[1];
        if (omega <= 0)
        {
            var units = decoder.Collapse(decoder.Alignment(states));
            return (units, units.Length == 0 ? new Tensor<T>(new[] { 0 }) : Vocoder.Forward(units));
        }
        var pieces = new List<Tensor<T>>();
        int voiced = 0;
        int[] all = Array.Empty<int>();
        for (int i = 1; i <= count; i++)
        {
            all = decoder.Collapse(decoder.Alignment(Engine.TensorSlice(states, new[] { 0, 0 }, new[] { i, d })));
            if (all.Length - voiced >= omega)
            {
                pieces.Add(Vocoder.Forward(all.Skip(voiced).ToArray()));
                voiced = all.Length;
            }
        }
        if (voiced < all.Length) pieces.Add(Vocoder.Forward(all.Skip(voiced).ToArray()));
        return (all, pieces.Count == 0 ? new Tensor<T>(new[] { 0 }) : Engine.TensorConcatenate(pieces.ToArray(), 0));
    }

    /// <summary>LLaMA-Omni's response to 16 kHz speech: the text, the units and the speech.</summary>
    public LlamaOmniResponse<T> Chat(Tensor<T> recording) => Inference(() =>
    {
        var prompt = PromptIds(SpeechPlaceholder + "\n" + _options.SpeechInstruction);
        var (ids, states) = Generate(Inputs(prompt, SpeechEmbeddings(recording), Array.Empty<int>()));
        var (units, audio) = Speak(states);
        return new LlamaOmniResponse<T> { Text = DecodeText(ids).Trim(), Units = units, Audio = audio };
    });

    /// <summary>The units and speech of a text read aloud: the text is the user's turn and the response, fed rather than
    /// generated, its hidden states voiced by the speech decoder.</summary>
    internal (int[] Units, Tensor<T> Audio) Read(int[] text)
    {
        var prompt = PromptIds(DecodeText(text));
        // The states that predict each text token and the end of the turn.
        var states = LastStates(Inputs(prompt, null, text), text.Length + 1);
        return Speak(states);
    }

    // ---------------------------------------------------------------- TTS

    /// <inheritdoc />
    /// <remarks>The text's token ids (<see cref="TokenizeText"/>).</remarks>
    protected override Tensor<T> PreprocessText(string text)
    {
        if (!HasPaperLayers) return base.PreprocessText(text);
        var ids = TokenizeText(text);
        if (ids.Length == 0) throw new ArgumentException("The text has no tokens.", nameof(text));
        return IndexTtsOps.Ids<T>(ids);
    }

    /// <inheritdoc />
    public override Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed();
        if (text is null) throw new ArgumentNullException(nameof(text));
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(PreprocessText(text));
        if (!HasPaperLayers) return base.Synthesize(text);
        return PredictCore(PreprocessText(text));
    }

    private int[] TokenIds(Tensor<T> tensor, string name)
    {
        var ids = new int[tensor.Length];
        for (int i = 0; i < ids.Length; i++)
        {
            ids[i] = (int)Math.Round(Ops.ToDouble(tensor.GetFlat(i)));
            if (ids[i] < 0 || ids[i] >= _options.VocabSize) throw new ArgumentOutOfRangeException(name, $"Token ids lie in [0, {_options.VocabSize}).");
        }
        return ids;
    }

    /// <inheritdoc />
    /// <remarks><paramref name="input"/> is the text's token ids; the result is that text read aloud at 16 kHz.</remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input);
        if (!HasPaperLayers) return base.PredictCore(input);
        var text = TokenIds(input, nameof(input));
        if (text.Length == 0) throw new ArgumentException("There are no text tokens.", nameof(input));
        return Inference(() => Read(text).Audio);
    }

    /// <inheritdoc />
    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;

    // ---------------------------------------------------------------- weights

    private static Dictionary<string, Tensor<float>> ReadAll(IEnumerable<string> paths)
    {
        var tensors = new Dictionary<string, Tensor<float>>(StringComparer.Ordinal);
        foreach (var path in paths)
        {
            if (!File.Exists(path)) throw new FileNotFoundException($"Checkpoint not found: {path}", path);
            foreach (var pair in new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(path)) tensors[pair.Key] = pair.Value;
        }
        return tensors;
    }

    private static double[] Values(Dictionary<string, Tensor<float>> tensors, HashSet<string> used, string name)
    {
        if (!tensors.TryGetValue(name, out var tensor)) throw new InvalidDataException($"The checkpoint has no tensor '{name}'.");
        used.Add(name);
        var values = new double[tensor.Length];
        for (int i = 0; i < values.Length; i++) values[i] = tensor.GetFlat(i);
        return values;
    }

    /// <summary>
    /// Loads ICTNLP/Llama-3.1-8B-Omni (its safetensors shards): the speech encoder (<c>model.speech_encoder</c>), the
    /// adaptor (<c>model.speech_projector</c>), the LLM (<c>model.*</c>, <c>lm_head</c>) and the speech decoder
    /// (<c>speech_generator</c>), and, unless one is set, its <c>tokenizer.json</c>.
    /// </summary>
    public void LoadReleasedWeights(string directory)
    {
        ThrowIfDisposed();
        if (directory is null) throw new ArgumentNullException(nameof(directory));
        string index = Path.Combine(directory, "model.safetensors.index.json");
        var files = File.Exists(index)
            ? Newtonsoft.Json.Linq.JObject.Parse(File.ReadAllText(index))["weight_map"]!.Values<string>().Distinct().Select(f => Path.Combine(directory, f!)).ToList()
            : new List<string> { Path.Combine(directory, "model.safetensors") };
        var tensors = ReadAll(files);
        var used = new HashSet<string>(StringComparer.Ordinal);
        SpeechEncoder.LoadTorchWeights((name, _) => Values(tensors, used, "model.speech_" + name));
        Adaptor.Load((name, _) => Values(tensors, used, name), "model.speech_projector.");
        LanguageModel.LoadTorchWeights((name, _) => Values(tensors, used, name), "model.", "lm_head.weight");
        SpeechDecoder.Load((name, _) => Values(tensors, used, name), "speech_generator.");
        var left = tensors.Keys.Where(k => !used.Contains(k)).ToList();
        if (left.Count > 0) throw new InvalidDataException($"The checkpoint has {left.Count} tensors this model has no place for (first: '{left[0]}').");
        string tokenizer = Path.Combine(directory, "tokenizer.json");
        if (_options.TokenizerPath is null && File.Exists(tokenizer))
        {
            _options.TokenizerPath = tokenizer;
            _tokenizer = null;
        }
    }

    /// <summary>Loads fairseq's unit HiFi-GAN (<c>g_00500000</c> of mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj).</summary>
    public void LoadVocoder(string path)
    {
        ThrowIfDisposed();
        var tensors = ReadAll(new[] { path });
        var used = new HashSet<string>(StringComparer.Ordinal);
        Vocoder.LoadTorchWeights(name => tensors.ContainsKey(name) ? Values(tensors, used, name) : null, "generator.");
    }

    /// <summary>Loads the unit extractor that makes the decoder's training targets: fairseq's mHuBERT checkpoint and its
    /// k-means model (<c>mhubert_base_vp_en_es_fr_it3.pt</c>, <c>mhubert_base_vp_en_es_fr_it3_L11_km1000.bin</c>).</summary>
    public void LoadUnitExtractor(string hubertCheckpoint, string kmeansModel)
    {
        ThrowIfDisposed();
        var tensors = ReadAll(new[] { hubertCheckpoint });
        var used = new HashSet<string>(StringComparer.Ordinal);
        UnitExtractor.LoadFairseqWeights(name => tensors.ContainsKey("model." + name) ? Values(tensors, used, "model." + name) : null);
        UnitExtractor.Centroids = JoblibKMeans.ReadClusterCenters(kmeansModel);
    }

    // ---------------------------------------------------------------- housekeeping

    /// <inheritdoc />
    /// <remarks>The unit extractor's k-means centroids are fitted, not trained.</remarks>
    protected override void RegisterState(AiDotNet.Models.ModelStateRegistry<T> state)
    {
        base.RegisterState(state);
        _units?.DeclareCentroids(state, "UnitCentroids");
    }

    /// <inheritdoc />
    protected override bool SupportsParameterMutation => _useNativeMode;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "LlamaOmni-Native" : "LlamaOmni-ONNX",
            Description = "LLaMA-Omni: seamless speech interaction with large language models (Fang et al., 2025)",
            FeatureCount = _options.HiddenDim,
        };
        metadata.AdditionalInfo["Architecture"] = "LlamaOmni";
        metadata.AdditionalInfo["Stage"] = _options.Stage.ToString();
        metadata.AdditionalInfo["UnitChunkSize"] = _options.UnitChunkSize.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(LlamaOmni<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
