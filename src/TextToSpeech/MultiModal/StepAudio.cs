using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.Latest;

namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>
/// Step-Audio (Huang et al. 2025): speech understanding and generation for voice interaction. A dual-codebook tokenizer
/// turns speech into linguistic and semantic tokens, the 130B Step-1 chat model answers in text (AQTA), and the 3B Step-1
/// TTS model writes the answer's dual-codebook tokens, which CosyVoice's flow and HiFT turn into 22.05 kHz speech.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>Built as the release (stepfun-ai/Step-Audio) runs it; see <see cref="StepAudioOptions"/> and
/// <c>docs/text-to-speech/paper-fidelity-decisions.md</c>. Speech goes in at any rate (resampled to 16 kHz for the
/// tokenizer) and comes out at 22.05 kHz. The TTS speaks as <see cref="SpeakerVoice"/>, a speaker made from a prompt
/// recording and its transcript (<see cref="CreateSpeaker"/>).</para>
/// <para><b>For Beginners:</b> Step-Audio listens, thinks in text with a very large language model, and answers aloud
/// in a voice you give it a sample of; tags like "(高兴1)" or "(RAP)" at the start of the text set the style.</para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "Step-Audio: Unified Understanding and Generation in Intelligent Speech Interaction",
    "https://arxiv.org/abs/2502.11946",
    Year = 2025,
    Authors = "Ailin Huang, Boyong Wu, Bruce Wang, Chao Yan, Chen Hu, Chengli Feng, Fei Tian, Feiyu Shen, Jingbei Li, Mingrui Chen, et al."
)]
[PaperOptimizer(OptimizerKind.Unspecified, LearningRate = 2e-5, MinLearningRate = 2e-6,
                Schedule = LearningRateSchedulerType.CosineAnnealing,
                Phase = TrainingPhase.FineTuning,
                Source = "Huang et al. 2025, Sec. 5.1.2: the 3B TTS model is fine-tuned for one epoch at 2e-5 with cosine decay to "
                        + "2e-6; Sec. 4.2 pre-trains at 2e-5 (embedding and head at five times that, then one cosine to 5e-6), "
                        + "Sec. 5.2 fine-tunes the chat model from 5.656e-5 to 5.656e-6 and the reward model from 1.24e-5 to 6e-6. "
                        + "The paper names no optimizer; AdamW is used.")]
[PaperOptimizer(OptimizerKind.Unspecified, LearningRate = 1e-6, MinLearningRate = 2e-7,
                Schedule = LearningRateSchedulerType.CosineAnnealing,
                Phase = TrainingPhase.Alignment,
                Source = "Huang et al. 2025, Sec. 5.2.6: PPO with a clip of 0.2, a KL penalty of 0.05, the critic warmed up for "
                        + "80 steps, at 1e-6 decaying by cosine to 2e-7.")]
public partial class StepAudio<T> : TtsModelBase<T>
{
    private readonly StepAudioOptions _options;
    private readonly bool _useNativeMode;
    private bool _disposed;
    private StepAudioCore<T>? _core;
    private readonly StepAudioLayers<T> _layers = new();

    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    /// <summary>Creates Step-Audio running an ONNX export.</summary>
    public StepAudio(NeuralNetworkArchitecture<T> architecture, string modelPath, StepAudioOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new StepAudioOptions();
        Options = _options;
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates Step-Audio with its networks (randomly initialized).</summary>
    public StepAudio(NeuralNetworkArchitecture<T> architecture, StepAudioOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new StepAudioOptions();
        Options = _options;
        _useNativeMode = true;
        base.SampleRate = _options.SampleRate;
        base.HiddenDim = _options.TtsWidth;
        InitializeLayers();
    }

    /// <summary>The longest answer, in tokens.</summary>
    public int MaxTextLength => _options.MaxTextLength;

    private bool HasPaperLayers => _core is not null;
    private int SeedValue => _options.Seed ?? 42;
    private InvalidOperationException NotNative() => new($"{GetType().Name}'s networks exist in native mode only.");
    internal StepAudioCore<T> Core => _core ?? throw NotNative();

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
        _core = new StepAudioCore<T>(Engine, _options, _layers, Architecture.RandomSeed ?? SeedValue);
        if (_options.IncludeRlhfModels) BuildRlhfModels();
        AddUnsplitLayers(_layers.Tts.Cast<ILayer<T>>().ToList());
        // The chat model, the tokenizers, the decoders and the RLHF models, so clones and saved models keep them.
        ComponentLayers.AddRange(_layers.Chat);
        ComponentLayers.AddRange(_layers.Tokenizer);
        ComponentLayers.AddRange(_layers.Decoder);
        ComponentLayers.AddRange(_layers.MusicDecoder);
        ComponentLayers.AddRange(_layers.Reward);
        ComponentLayers.AddRange(_layers.Critic);
        ComponentLayers.AddRange(_layers.Reference);
    }

    // ---------------------------------------------------------------- speakers and text

    /// <summary>The speaker the TTS speaks as; Synthesize needs one.</summary>
    public StepAudioSpeaker<T>? SpeakerVoice { get; set; }

    /// <summary>A speaker from a prompt recording (any rate) and its transcript, named as the release's system prompt
    /// names speakers (empty for the prompt without a name).</summary>
    public StepAudioSpeaker<T> CreateSpeaker(Tensor<T> recording, int sampleRate, string transcript, string name = "")
    {
        ThrowIfDisposed();
        if (recording is null) throw new ArgumentNullException(nameof(recording));
        if (transcript is null) throw new ArgumentNullException(nameof(transcript));
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            return Core.CreateSpeaker(recording, sampleRate, transcript, name ?? string.Empty);
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    /// <summary>Step-1's token ids of text (with &lt;s&gt;, as the release encodes).</summary>
    public int[] TokenizeText(string text)
    {
        ThrowIfDisposed();
        return Core.Tokens.Encode(text ?? throw new ArgumentNullException(nameof(text)));
    }

    /// <summary>The dual-codebook token string of a recording ("&lt;audio_N&gt;…", tokenizer.py's <c>__call__</c>).</summary>
    public string AudioTokenString(Tensor<T> recording, int sampleRate)
    {
        ThrowIfDisposed();
        if (recording is null) throw new ArgumentNullException(nameof(recording));
        var (_, linguistic, semantic) = Core.AudioTokens(recording, sampleRate);
        return StepAudioTokens.AudioString(linguistic, semantic);
    }

    // ---------------------------------------------------------------- tasks

    /// <summary>Speech for text in <see cref="SpeakerVoice"/> (the release's TTS), 22.05 kHz.</summary>
    public Tensor<T> Speak(string text)
    {
        ThrowIfDisposed();
        if (text is null) throw new ArgumentNullException(nameof(text));
        var speaker = SpeakerVoice ?? throw new InvalidOperationException("Step-Audio's TTS speaks as a prompt speaker: set SpeakerVoice (CreateSpeaker).");
        using var _ = Inference();
        return Core.Speak(text, speaker, SamplingRandom());
    }

    /// <summary>The chat model's text answer to a spoken question (AQTA), the question at <paramref name="sampleRate"/>,
    /// after optional earlier turns (role, text).</summary>
    public string Answer(Tensor<T> speech, int sampleRate, IReadOnlyList<(string Role, string Content)>? history = null)
    {
        ThrowIfDisposed();
        if (speech is null) throw new ArgumentNullException(nameof(speech));
        using var _ = Inference();
        var turns = new List<(string, string)>(history ?? Array.Empty<(string, string)>()) { ("human", AudioTokenString(speech, sampleRate)) };
        return Core.Answer(turns, SamplingRandom());
    }

    /// <summary>The chat model's text answer to a text question (TQTA).</summary>
    public string Answer(string question)
    {
        ThrowIfDisposed();
        if (question is null) throw new ArgumentNullException(nameof(question));
        using var _ = Inference();
        return Core.Answer(new[] { ("human", question) }, SamplingRandom());
    }

    /// <summary>The full voice interaction (stepaudio.py's <c>__call__</c>): the spoken answer to a spoken question, and its
    /// text.</summary>
    public (Tensor<T> Audio, string Text) Chat(Tensor<T> speech, int sampleRate)
    {
        string text = Answer(speech, sampleRate);
        return (text.Length == 0 ? new Tensor<T>(new[] { 0 }) : Speak(text), text);
    }

    /// <inheritdoc />
    /// <remarks>The TTS model in <see cref="SpeakerVoice"/> (<see cref="Speak"/>).</remarks>
    public override Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed();
        if (text is null) throw new ArgumentNullException(nameof(text));
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(PreprocessText(text));
        if (!HasPaperLayers) return base.Synthesize(text);
        return Speak(text);
    }

    /// <inheritdoc />
    /// <remarks>The text's ids as the TTS prompt's human turn holds them (encode of a newline and the text, after the
    /// newline's ids).</remarks>
    protected override Tensor<T> PreprocessText(string text)
    {
        if (!HasPaperLayers) return base.PreprocessText(text);
        var prefix = Core.Tokens.Encode("\n");
        var ids = Core.Tokens.Encode("\n" + text).Skip(prefix.Length).ToArray();
        return IndexTtsOps.Ids<T>(ids.Length == 0 ? new[] { 0 } : ids);
    }

    /// <inheritdoc />
    /// <remarks><paramref name="input"/> is the text's Step-1 token ids (no &lt;s&gt;), placed as they are in the TTS prompt's
    /// human turn; the result is their speech in <see cref="SpeakerVoice"/>.</remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input);
        if (!HasPaperLayers) return base.PredictCore(input);
        var speaker = SpeakerVoice ?? throw new InvalidOperationException("Step-Audio's TTS speaks as a prompt speaker: set SpeakerVoice (CreateSpeaker).");
        using var _ = Inference();
        return Core.Speak(IdsOf(input), speaker, SamplingRandom());
    }

    private int[] IdsOf(Tensor<T> input)
    {
        var ids = new int[input.Length];
        for (int i = 0; i < ids.Length; i++)
        {
            ids[i] = (int)Math.Round(Ops.ToDouble(input.GetFlat(i)));
            if (ids[i] < 0 || ids[i] >= _options.VocabSize) throw new ArgumentOutOfRangeException(nameof(input), $"Token ids lie in [0, {_options.VocabSize}).");
        }
        return ids;
    }

    /// <inheritdoc />
    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;

    // A fresh generator per call: the same input gives the same output.
    private Random SamplingRandom() => AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(SeedValue);

    private IDisposable Inference()
    {
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        var scope = new NoGradScope<T>();
        return new Restore(() =>
        {
            scope.Dispose();
            SetTrainingMode(wasTraining);
        });
    }

    private sealed class Restore : IDisposable
    {
        private Action? _action;
        public Restore(Action action) => _action = action;
        public void Dispose()
        {
            _action?.Invoke();
            _action = null;
        }
    }

    // ---------------------------------------------------------------- housekeeping

    /// <inheritdoc />
    /// <remarks>The linguistic tokenizer's k-means centroids and normalization are model state, not parameters.</remarks>
    protected override void RegisterState(AiDotNet.Models.ModelStateRegistry<T> state)
    {
        base.RegisterState(state);
        if (_core is null) return;
        var core = _core;
        state.DeclareDoubleVector("LinguisticCentroids", () => new Vector<double>(core.Centroids), v => { if (v is not null) core.Centroids = v.ToArray(); });
        state.DeclareDoubleVector("LinguisticShift", () => new Vector<double>(core.Linguistic.Encoder.Normalization.Shift),
            v => { if (v is not null) core.Linguistic.Encoder.Normalization = (v.ToArray(), core.Linguistic.Encoder.Normalization.Scale); });
        state.DeclareDoubleVector("LinguisticScale", () => new Vector<double>(core.Linguistic.Encoder.Normalization.Scale),
            v => { if (v is not null) core.Linguistic.Encoder.Normalization = (core.Linguistic.Encoder.Normalization.Shift, v.ToArray()); });
    }

    /// <inheritdoc />
    protected override bool SupportsParameterMutation => _useNativeMode;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "StepAudio-Native" : "StepAudio-ONNX",
            Description = "Step-Audio: unified understanding and generation in intelligent speech interaction (Huang et al., 2025)",
            FeatureCount = _options.TtsWidth,
        };
        metadata.AdditionalInfo["Architecture"] = "StepAudio";
        return metadata;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(StepAudio<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
