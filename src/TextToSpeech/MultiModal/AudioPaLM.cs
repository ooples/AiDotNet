using AiDotNet.Attributes;
using AiDotNet.Audio.Foundations;
using AiDotNet.Audio.Generation;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.Latest;
using AiDotNet.Tokenization.Algorithms;

namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>
/// AudioPaLM (Rubenstein et al. 2023): a text-pretrained decoder-only language model whose vocabulary is extended by
/// audio tokens, so one model reads and writes both text and speech — text to speech, speech recognition, speech
/// translation to text and speech-to-speech translation in the speaker's voice.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>Speech becomes audio tokens through a speech encoder's k-means codes (§3.1: w2v-BERT, 1024 tokens at 25 Hz);
/// they follow the text tokens in one vocabulary, the embedding table (shared with the output layer) grown by their rows,
/// initialized to zero (§3.2). A task is a text tag such as <c>[TTS English]</c> or <c>[S2ST English French]</c>
/// before the input (§3.4); the model continues with the output and an end token, and is trained with the loss on the
/// output only. Audio tokens become speech through SoundStorm or AudioLM's acoustic stages, continuing a 3-second voice
/// prompt (§3.3).</para>
/// <para>The decoder is PaLM's (Chowdhery et al. 2022 §2) — PaLM-2's architecture is not published — or the public
/// Gemma's (<see cref="AudioPaLMOptions.Gemma2B"/>), whose pretrained weights stand in for the text checkpoint the paper
/// starts from.</para>
/// <para><b>For Beginners:</b> AudioPaLM is a chatbot-style language model that has learned "audio words" alongside
/// ordinary words. Give it text and a task, and it writes audio words that a separate decoder turns into speech; give
/// it speech, and it can write the transcript or a translation.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);
/// var options = AudioPaLMOptions.Gemma2B();
/// options.SentencePieceModelPath = "gemma-2b/tokenizer.model";
/// var model = new AudioPaLM&lt;float&gt;(architecture, options);
/// model.LoadGemmaWeights("gemma-2b");
/// model.FitSpeechTokenizer(recordings);        // the audio tokens' k-means codebook
/// model.Train(new TtsTrainingSample&lt;float&gt; { Tokens = textIds, Audio = recording });
/// model.Voice = new TtsVoice&lt;float&gt; { ReferenceAudio = voiceRecording };
/// var audio = model.Synthesize("Hello world.");
/// string text = model.Transcribe(recording, "English");
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
    "AudioPaLM: A Large Language Model That Can Speak and Listen",
    "https://arxiv.org/abs/2306.12925",
    Year = 2023,
    Authors = "Paul K. Rubenstein, Chulayuth Asawaroengchai, Duc Dung Nguyen, Ankur Bapna, Zalán Borsos, Félix de Chaumont Quitry, Peter Chen, Dalia El Badawy, Wei Han, Eugene Kharitonov, Hannah Muckenhirn, Dirk Padfield, James Qin, Danny Rozenberg, Tara Sainath, Johan Schalkwyk, Matt Sharifi, Michelle Tadmor Ramanovich, Marco Tagliasacchi, Alexandru Tudor, Mihajlo Velimirović, Damien Vincent, Jiahui Yu, Yongqiang Wang, Vicky Zayats, Neil Zeghidour, Yu Zhang, Zhishuai Zhang, Lukas Zilka, Christian Frank"
)]
[PaperOptimizer(OptimizerKind.Adafactor, LearningRate = 5e-5,
                Source = "Rubenstein et al. 2023 §3.6: the finetuning setup of Chowdhery et al. 2022 §6.1.2, Adafactor with a "
                        + "constant learning rate of 5e-5, dropout 0.1, loss masking on the inputs. A stated rate stands "
                        + "Adafactor's relative step rule down.")]
public partial class AudioPaLM<T> : TtsModelBase<T>
{
    private readonly AudioPaLMOptions _options;
    private readonly bool _useNativeMode;
    private bool _disposed;
    private AudioPalmCore<T>? _core;
    private readonly List<LayerBase<T>> _decoderLayers = new();

    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    /// <summary>Creates an ONNX-backed AudioPaLM for inference.</summary>
    public AudioPaLM(NeuralNetworkArchitecture<T> architecture, string modelPath, AudioPaLMOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new AudioPaLMOptions();
        Options = _options;
        _useNativeMode = false;
        ConfigureBase();
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a native, trainable AudioPaLM.</summary>
    public AudioPaLM(NeuralNetworkArchitecture<T> architecture, AudioPaLMOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new AudioPaLMOptions();
        Options = _options;
        _useNativeMode = true;
        ConfigureBase();
        InitializeLayers();
    }

    private void ConfigureBase()
    {
        base.SampleRate = _options.SampleRate;
        base.HiddenDim = _options.HiddenDim;
        AudioPalmCore<T>.Validate(_options);
    }

    /// <inheritdoc />
    public int MaxTextLength => _options.MaxTextLength;

    /// <inheritdoc />
    /// <remarks>Speech is rendered continuing a voice prompt (§3.3); synthesis takes it from the voice's recording.</remarks>
    protected override TtsSupervision RequiredVoice => TtsSupervision.ReferenceRecording;

    private bool HasPaperLayers => _core is not null;
    private InvalidOperationException NotNative() => new($"{GetType().Name}'s networks exist in native mode only.");

    internal AudioPalmCore<T> Core => _core ?? throw NotNative();
    internal PalmDecoder<T> Decoder => Core.Decoder;

    /// <summary>The w2v-BERT model whose k-means codes are the audio tokens (with the w2v-BERT tokenizer).</summary>
    public W2vBert<T>? SpeechEncoder => _core?.SpeechEncoder;

    /// <summary>The SoundStorm acoustic model that renders audio tokens (with the SoundStorm detokenizer).</summary>
    public SoundStorm<T>? SoundStormDetokenizer => _core?.SoundStorm;

    /// <summary>The AudioLM whose acoustic stages render audio tokens (with the AudioLM detokenizer).</summary>
    public AudioLM<T>? AudioLMDetokenizer => _core?.AudioLM;

    /// <summary>The first audio token's id: the text vocabulary's size (§3.2).</summary>
    public int AudioOffset => _options.VocabSize;

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
        _core = new AudioPalmCore<T>(Engine, _options, _decoderLayers, Architecture.RandomSeed ?? _options.TrainingSeed, withDetokenizer: true);
        AddUnsplitLayers(_decoderLayers.Cast<ILayer<T>>().ToList());
        // The speech tokenizer (its encoder and fitted codebook) and the detokenizer, so clones and saved models keep them.
        ComponentLayers.AddRange(_core.ComponentLayers());
    }

    // ---------------------------------------------------------------- tokens

    /// <summary>The text token ids of <paramref name="text"/> (the decoder's SentencePiece model).</summary>
    public int[] TokenizeText(string text) => Core.TokenizeText(text);

    /// <summary>The text of text token ids (audio ids and the start and end tokens dropped).</summary>
    public string DecodeText(IEnumerable<int> ids) => Core.DecodeText(ids);

    /// <summary>The decoder ids of a task: the start token, the tag's text tokens, then the input's ids (§3.4).</summary>
    public int[] TaskPrefix(string tag, IReadOnlyList<int> input) => Core.TaskPrefix(tag, input);

    /// <summary>
    /// Fits the audio tokens' k-means codebook on 16 kHz recordings: w2v-BERT's semantic tokenizer on its embeddings as
    /// they are (§3.1), or k-means over w2v-BERT 2.0's <see cref="AudioPaLMOptions.W2vBert2Layer"/> embeddings.
    /// </summary>
    public void FitSpeechTokenizer(IEnumerable<Tensor<T>> recordings)
    {
        ThrowIfDisposed();
        Core.FitSpeechTokenizer(recordings);
    }

    /// <summary>Sets the w2v-BERT 2.0 tokenizer's k-means centroids <c>[AudioTokens, 1024]</c>.</summary>
    public void LoadSpeechTokenizerCodebook(double[,] centroids)
    {
        ThrowIfDisposed();
        Core.LoadSpeechTokenizerCodebook(centroids);
    }

    /// <summary>The audio tokens of a 16 kHz recording (codes in [0, <see cref="AudioPaLMOptions.AudioTokens"/>)).</summary>
    public int[] AudioTokensOf(Tensor<T> recording)
    {
        ThrowIfDisposed();
        return Core.AudioTokensOf(recording);
    }

    private TResult Inference<TResult>(Func<TResult> run)
    {
        ThrowIfDisposed();
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            return run();
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    /// <summary>
    /// Continues <paramref name="prefix"/> (decoder ids) until the end token or <see cref="AudioPaLMOptions.MaxNewTokens"/>,
    /// choosing among audio tokens (<paramref name="audioOutput"/>) or text tokens, and the end token after the first;
    /// greedy, or sampled at <see cref="AudioPaLMOptions.Temperature"/>. Returns the new ids without the end token.
    /// </summary>
    public int[] Generate(IReadOnlyList<int> prefix, bool audioOutput) => Inference(() => Core.Generate(prefix, audioOutput));

    /// <summary>Renders audio token codes to a 16 kHz waveform with the detokenizer, continuing the voice prompt taken
    /// from <paramref name="voiceRecording"/> (§3.3).</summary>
    public Tensor<T> Detokenize(IReadOnlyList<int> codes, Tensor<T> voiceRecording) => Inference(() => Core.Detokenize(codes, voiceRecording));

    // ---------------------------------------------------------------- tasks

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
            if (ids[i] < 0 || ids[i] >= _options.VocabSize + _options.AudioTokens)
                throw new ArgumentOutOfRangeException(name, $"Token ids lie in [0, {_options.VocabSize + _options.AudioTokens}).");
        }
        return ids;
    }

    /// <summary>The TTS task's tag (<c>[TTS English]</c>).</summary>
    private string TtsTag => $"[TTS {_options.Language}]";

    /// <inheritdoc />
    /// <remarks><paramref name="input"/> is the text's token ids; the result is the 16 kHz speech the model writes for
    /// <c>[TTS language] text</c>, rendered continuing the voice's recording.</remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input);
        if (!HasPaperLayers) return base.PredictCore(input);
        var voice = RequireVoice();
        var ids = TokenIds(input, nameof(input));
        if (ids.Length == 0) throw new ArgumentException("There are no text tokens.", nameof(input));
        var generated = Generate(TaskPrefix(TtsTag, ids), audioOutput: true);
        return Detokenize(Core.AudioCodes(generated), voice.ReferenceAudio!);
    }

    /// <summary>Speech recognition (<c>[ASR language]</c>, the language's English name): the transcript of a 16 kHz
    /// recording.</summary>
    public string Transcribe(Tensor<T> recording, string language) =>
        DecodeText(Generate(TaskPrefix($"[ASR {language}]", Core.AudioIds(AudioTokensOf(recording))), audioOutput: false));

    /// <summary>Speech translation to text (<c>[AST source target]</c>).</summary>
    public string TranslateSpeechToText(Tensor<T> recording, string sourceLanguage, string targetLanguage) =>
        DecodeText(Generate(TaskPrefix($"[AST {sourceLanguage} {targetLanguage}]", Core.AudioIds(AudioTokensOf(recording))), audioOutput: false));

    /// <summary>Speech-to-speech translation (<c>[S2ST source target]</c>): the translated speech in the voice of the source
    /// recording, whose first seconds are the voice prompt (§3.3).</summary>
    public Tensor<T> TranslateSpeech(Tensor<T> recording, string sourceLanguage, string targetLanguage)
    {
        var generated = Generate(TaskPrefix($"[S2ST {sourceLanguage} {targetLanguage}]", Core.AudioIds(AudioTokensOf(recording))), audioOutput: true);
        return Detokenize(Core.AudioCodes(generated), recording);
    }

    /// <inheritdoc />
    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;

    // ---------------------------------------------------------------- weights

    /// <summary>Loads a Hugging Face Gemma checkpoint directory into the decoder (its text rows; the audio rows zeroed,
    /// §3.2) and, unless a tokenizer is set, its <c>tokenizer.model</c>. Use <see cref="AudioPaLMOptions.Gemma2B"/>.</summary>
    public void LoadGemmaWeights(string directory)
    {
        ThrowIfDisposed();
        Core.LoadGemma(directory);
    }

    /// <summary>Loads a lucidrains/PaLM-pytorch state dict into the PaLM decoder (its text rows; the audio rows zeroed).</summary>
    public void LoadPalmPytorchWeights(string path)
    {
        ThrowIfDisposed();
        Core.LoadPalmPytorch(path);
    }

    /// <summary>Loads w2v-BERT 2.0 (<c>facebook/w2v-bert-2.0</c>) for the w2v-BERT 2.0 tokenizer.</summary>
    public void LoadW2vBert2Weights(string path)
    {
        ThrowIfDisposed();
        Core.LoadW2vBert2(path);
    }

    // ---------------------------------------------------------------- housekeeping

    /// <inheritdoc />
    protected override bool SupportsParameterMutation => _useNativeMode;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "AudioPaLM-Native" : "AudioPaLM-ONNX",
            Description = "AudioPaLM: a large language model that can speak and listen (Rubenstein et al., 2023)",
            FeatureCount = _options.HiddenDim,
        };
        metadata.AdditionalInfo["Architecture"] = "AudioPaLM";
        metadata.AdditionalInfo["Decoder"] = _options.Decoder.ToString();
        metadata.AdditionalInfo["SpeechTokenizer"] = _options.SpeechTokenizer.ToString();
        metadata.AdditionalInfo["Detokenizer"] = _options.Detokenizer.ToString();
        return metadata;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(AudioPaLM<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        if (disposing)
        {
            _core?.SpeechEncoder?.Dispose();
            _core?.SoundStorm?.Dispose();
            _core?.AudioLM?.Dispose();
        }
        base.Dispose(disposing);
    }
}
