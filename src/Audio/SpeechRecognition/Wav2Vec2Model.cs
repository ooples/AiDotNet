using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.SpeechRecognition;

/// <summary>
/// Wav2Vec2 self-supervised speech recognition model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Wav2Vec2 is a self-supervised learning model for speech recognition developed by Meta AI.
/// It learns representations from raw audio through contrastive learning, then can be
/// fine-tuned for speech recognition tasks.
/// </para>
/// <para><b>For Beginners:</b> Wav2Vec2 works differently from traditional speech recognition:
///
/// 1. It processes raw audio directly (no mel spectrograms needed)
/// 2. It learns speech patterns from unlabeled audio data
/// 3. It can be fine-tuned with small amounts of labeled data
///
/// Architecture:
/// - Convolutional feature encoder: Processes raw audio into features
/// - Transformer encoder: Captures long-range dependencies in speech
/// - CTC head: Aligns speech to text (Connectionist Temporal Classification)
///
/// Two ways to use this class:
/// 1. ONNX Mode: Load pretrained Wav2Vec2 models for fast inference
/// 2. Native Mode: Train your own speech recognition model from scratch
///
/// ONNX Mode Example:
/// <code>
/// var wav2vec2 = new Wav2Vec2Model&lt;float&gt;(
///     architecture,
///     modelPath: "path/to/wav2vec2.onnx");
/// var result = wav2vec2.Transcribe(audioTensor);
/// // Result is available in the returned value
/// </code>
///
/// Training Mode Example:
/// <code>
/// var wav2vec2 = new Wav2Vec2Model&lt;float&gt;(architecture);
/// for (int epoch = 0; epoch &lt; 100; epoch++)
/// {
///     foreach (var (audio, tokens) in trainingData)
///     {
///         wav2vec2.Train(audio, tokens);
///     }
/// }
/// </code>
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.SpeechRecognition)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations", "https://arxiv.org/abs/2006.11477", Year = 2020, Authors = "Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli")]
[TaskVariantOf(typeof(AiDotNet.Audio.Foundations.Wav2Vec2<>), "speech recognition with a CTC head")]
public partial class Wav2Vec2Model<T> : AudioNeuralNetworkBase<T>, ISpeechRecognizer<T>
{
    private readonly Wav2Vec2ModelOptions _options;

    #region Execution Mode

    /// <summary>
    /// Indicates whether this network uses native layers (true) or ONNX models (false).
    /// </summary>
    private bool _useNativeMode;

    #endregion

    #region ONNX Mode Fields

    /// <summary>
    /// Path to the ONNX model file.
    /// </summary>
    private string? _modelPath;

    #endregion

    #region Native Mode Fields

    // The shared wav2vec 2.0 network and the CTC head (transformers Wav2Vec2ForCTC: dropout, then lm_head).
    private Audio.Foundations.Wav2Vec2Network<T>? _network;
    private DenseLayer<T>? _ctcHead;
    private readonly Random _random;

    #endregion

    #region Shared Fields

    /// <summary>
    /// Optimizer for training (unused in ONNX mode).
    /// </summary>
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;

    /// <summary>
    /// Loss function for training.
    /// </summary>
    private ILossFunction<T> _lossFunction;

    /// <summary>
    /// Target language for transcription (non-readonly for deserialization support).
    /// </summary>
    private string? _language;

    /// <summary>
    /// Maximum audio length in seconds (non-readonly for deserialization support).
    /// </summary>
    private int _maxAudioLengthSeconds;

    /// <summary>
    /// Hidden dimension for the transformer (non-readonly for deserialization support).
    /// </summary>
    private int _hiddenDim;

    /// <summary>
    /// Number of transformer layers (non-readonly for deserialization support).
    /// </summary>
    private int _numTransformerLayers;

    /// <summary>
    /// Number of attention heads (non-readonly for deserialization support).
    /// </summary>
    private int _numHeads;

    /// <summary>
    /// Feed-forward dimension (non-readonly for deserialization support).
    /// </summary>
    private int _ffDim;

    /// <summary>
    /// Vocabulary size for CTC output (non-readonly for deserialization support).
    /// </summary>
    private int _vocabSize;

    /// <summary>
    /// Vocabulary mapping for CTC decoding (non-readonly for deserialization support).
    /// </summary>
    private string[] _vocabulary;

    /// <summary>
    /// Disposed flag.
    /// </summary>
    private bool _disposed;

    #endregion

    #region ISpeechRecognizer Properties

    /// <summary>
    /// Gets the list of languages supported by this model.
    /// </summary>
    public IReadOnlyList<string> SupportedLanguages { get; }

    /// <summary>
    /// Gets whether this model supports real-time streaming transcription.
    /// </summary>
    public bool SupportsStreaming => false;

    /// <summary>
    /// Gets whether this model can identify timestamps for each word.
    /// </summary>
    public bool SupportsWordTimestamps => true;

    #endregion

    #region Public Properties

    /// <summary>
    /// Gets whether the model is ready for inference.
    /// </summary>
    public bool IsReady => _useNativeMode || OnnxModel?.IsLoaded == true;

    /// <summary>
    /// Gets the target language for transcription.
    /// </summary>
    public string? Language => _language;

    /// <summary>
    /// Gets the maximum audio length in seconds.
    /// </summary>
    public int MaxAudioLengthSeconds => _maxAudioLengthSeconds;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a Wav2Vec2 network using a pretrained ONNX model.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="modelPath">Path to the ONNX model file.</param>
    /// <param name="language">Target language code (e.g., "en", "es"). Default is "en".</param>
    /// <param name="sampleRate">Audio sample rate in Hz. Wav2Vec2 expects 16000.</param>
    /// <param name="maxAudioLengthSeconds">Maximum audio length to process. Default is 30 seconds.</param>
    /// <param name="vocabulary">CTC vocabulary for decoding. If null, uses default English alphabet.</param>
    /// <param name="onnxOptions">ONNX runtime options.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor when you have a pretrained Wav2Vec2 ONNX model.
    ///
    /// You can get ONNX models from:
    /// - HuggingFace: facebook/wav2vec2-base-960h, etc.
    /// - Convert from PyTorch using ONNX export tools
    ///
    /// Example:
    /// <code>
    /// var wav2vec2 = new Wav2Vec2Model&lt;float&gt;(
    ///     architecture,
    ///     modelPath: "wav2vec2-base.onnx",
    ///     language: "en");
    /// </code>
    /// </para>
    /// </remarks>
    public Wav2Vec2Model(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        Wav2Vec2ModelOptions? options = null,
        string[]? vocabulary = null,
        OnnxModelOptions? onnxOptions = null)
        : base(architecture: architecture)
    {
        _options = options ?? new Wav2Vec2ModelOptions();
        _options.Validate();
        Options = _options;
        if (architecture is null)
            throw new ArgumentNullException(nameof(architecture));
        if (modelPath is null)
            throw new ArgumentNullException(nameof(modelPath));

        _useNativeMode = false;
        _modelPath = modelPath;
        _random = Tensors.Helpers.RandomHelper.CreateSeededRandom((_options.Seed ?? 42) + 1);
        _language = _options.Language;
        _maxAudioLengthSeconds = _options.MaxAudioLengthSeconds;

        // Set audio properties
        SampleRate = _options.SampleRate;
        NumMels = 0; // Wav2Vec2 doesn't use mel spectrograms

        // Model dimensions (standard Wav2Vec2 Base)
        _hiddenDim = _options.HiddenDim;
        _numTransformerLayers = _options.NumTransformerLayers;
        _numHeads = _options.NumHeads;
        _ffDim = _options.FfDim;

        // Initialize vocabulary
        _vocabulary = vocabulary ?? GetDefaultVocabulary();
        _vocabSize = _vocabulary.Length;

        // Load ONNX model
        var onnxOpts = onnxOptions ?? new OnnxModelOptions();
        OnnxModel = new OnnxModel<T>(modelPath, onnxOpts);

        // Initialize supported languages
        SupportedLanguages = new[] { _options.Language ?? "en" };

        // Wav2Vec2 + CTC is the standard ASR training stack (Baevski et al.
        // 2020 §3.2): CTC handles the variable-length output-vs-input
        // alignment that plain cross-entropy cannot. CE-with-logits would
        // be silently wrong here — it forces a fixed-length 1:1 alignment
        // and the loss is computed per-frame, which is not the ASR
        // objective. PR #1404 review (CodeRabbit).
        _lossFunction = new CTCLoss<T>(numClasses: _vocabSize, blankIndex: 0);

        InitializeLayers();
    }

    /// <summary>
    /// Creates a Wav2Vec2 network for training from scratch using native layers.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="language">Target language code (e.g., "en", "es"). Default is "en".</param>
    /// <param name="sampleRate">Audio sample rate in Hz. Default is 16000.</param>
    /// <param name="maxAudioLengthSeconds">Maximum audio length to process. Default is 30 seconds.</param>
    /// <param name="hiddenDim">Hidden dimension for transformer. Default is 768.</param>
    /// <param name="numTransformerLayers">Number of transformer layers. Default is 12.</param>
    /// <param name="numHeads">Number of attention heads. Default is 12.</param>
    /// <param name="ffDim">Feed-forward dimension. Default is 3072.</param>
    /// <param name="vocabulary">CTC vocabulary for decoding. If null, uses default English alphabet.</param>
    /// <param name="optimizer">Optimizer for training. If null, uses Adam with default settings.</param>
    /// <param name="lossFunction">Loss function for training. If null, uses CTC loss.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor to train a speech recognition model from scratch.
    ///
    /// Training Wav2Vec2 typically involves:
    /// 1. Pre-training on unlabeled audio (self-supervised)
    /// 2. Fine-tuning on labeled transcription data
    ///
    /// Example:
    /// <code>
    /// var wav2vec2 = new Wav2Vec2Model&lt;float&gt;(
    ///     architecture,
    ///     language: "en",
    ///     hiddenDim: 768,
    ///     numTransformerLayers: 12);
    ///
    /// // Training loop
    /// for (int epoch = 0; epoch &lt; numEpochs; epoch++)
    /// {
    ///     foreach (var (audio, tokens) in trainingData)
    ///     {
    ///         wav2vec2.Train(audio, tokens);
    ///     }
    /// }
    /// </code>
    /// </para>
    /// </remarks>
    public Wav2Vec2Model(
        NeuralNetworkArchitecture<T> architecture,
        Wav2Vec2ModelOptions? options = null,
        string[]? vocabulary = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture: architecture)
    {
        _options = options ?? new Wav2Vec2ModelOptions();
        _options.Validate();
        Options = _options;
        if (architecture is null)
            throw new ArgumentNullException(nameof(architecture));

        _useNativeMode = true;
        _random = Tensors.Helpers.RandomHelper.CreateSeededRandom((_options.Seed ?? 42) + 1);
        _language = _options.Language;
        _maxAudioLengthSeconds = _options.MaxAudioLengthSeconds;
        _hiddenDim = _options.HiddenDim;
        _numTransformerLayers = _options.NumTransformerLayers;
        _numHeads = _options.NumHeads;
        _ffDim = _options.FfDim;

        // Set audio properties
        SampleRate = _options.SampleRate;
        NumMels = 0; // Wav2Vec2 doesn't use mel spectrograms

        // Initialize vocabulary
        _vocabulary = vocabulary ?? GetDefaultVocabulary();
        _vocabSize = _vocabulary.Length;

        // Initialize supported languages
        SupportedLanguages = new[] { _options.Language ?? "en" };

        // Initialize training components — CTC for ASR (see ONNX ctor for
        // rationale). Wav2Vec2's variable-length frame-vs-character alignment
        // can't be expressed by plain cross-entropy.
        // Paper-faithful LR per Baevski et al. 2020 NeurIPS §3.3 ("wav2vec 2.0"):
        // Adam with peak LR=5e-4 for pretraining, 5e-5 for ASR fine-tuning.
        // Framework default (LR=1e-3) is too aggressive for this BERT-base scale
        // model at random init and causes Training_ShouldReduceLoss to diverge.
        // Use the 5e-5 fine-tuning default since the test runs from random init
        // and supervised CTC; pretraining-scale 5e-4 also works.
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(
            this,
            new Models.Options.AdamOptimizerOptions<T, Tensor<T>, Tensor<T>> { InitialLearningRate = 5e-5 });
        _lossFunction = lossFunction ?? new CTCLoss<T>(numClasses: _vocabSize, blankIndex: 0);

        InitializeNativeLayers();
    }

    #endregion

    #region Initialization

    /// <summary>
    /// Initializes layers for ONNX inference mode.
    /// </summary>
    protected override void InitializeLayers()
    {
        // ONNX mode - layers are handled by ONNX runtime
    }

    /// <summary>
    /// Initializes native mode layers for training from scratch.
    /// </summary>
    private void InitializeNativeLayers()
    {
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
            throw new ArgumentException("Wav2Vec2Model builds the wav2vec 2.0 network; custom layers in the architecture are not supported.");
        var layers = new List<LayerBase<T>>();
        _network = new Audio.Foundations.Wav2Vec2Network<T>(Engine, layers, _options.ToConfiguration());
        _ctcHead = new DenseLayer<T>(_vocabSize, (IActivationFunction<T>)new ActivationFunctions.IdentityActivation<T>());
        layers.Add(_ctcHead);
        var seed = Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.Seed ?? 42);
        _network.InitializeLikeReference(seed);
        // Wav2Vec2ForCTC's lm_head: N(0, 0.02) weights, zero biases.
        var head = new double[_vocabSize * _options.HiddenDim];
        for (int i = 0; i < head.Length; i++)
            head[i] = 0.02 * Math.Sqrt(-2.0 * Math.Log(1.0 - seed.NextDouble())) * Math.Cos(2.0 * Math.PI * seed.NextDouble());
        TextToSpeech.CodecBased.TorchParameters.Linear(Engine, _ctcHead, _options.HiddenDim, _vocabSize, head, new double[_vocabSize]);
        Layers.Clear();
        Layers.AddRange(layers);
    }

    /// <summary>Loads a transformers <c>Wav2Vec2ForCTC</c> state dict (<c>wav2vec2.*</c> and <c>lm_head.*</c>).</summary>
    public void LoadTorchWeights(Func<string, int[], double[]> read)
    {
        if (_network is null || _ctcHead is null) throw new InvalidOperationException("Weights load into the native network.");
        _network.LoadTorchWeights((name, shape) => read($"wav2vec2.{name}", shape));
        TextToSpeech.CodecBased.TorchParameters.Linear(Engine, _ctcHead, _options.HiddenDim, _vocabSize,
            read("lm_head.weight", new[] { _vocabSize, _options.HiddenDim }), read("lm_head.bias", new[] { _vocabSize }));
    }

    private static string[] GetDefaultVocabulary()
    {
        // CTC vocabulary: blank token + space + letters
        return new[]
        {
            "<blank>", "<pad>", "<s>", "</s>", "<unk>", "|",
            "a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
            "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
            "u", "v", "w", "x", "y", "z", "'", " "
        };
    }

    #endregion

    #region ISpeechRecognizer Implementation

    /// <summary>
    /// Transcribes audio to text.
    /// </summary>
    public TranscriptionResult<T> Transcribe(Tensor<T> audio, string? language = null, bool? includeTimestamps = null)
    {
        ThrowIfDisposed();

        // Preprocess audio
        var features = PreprocessAudio(audio);

        // Get model output
        Tensor<T> logits;
        if (_useNativeMode)
        {
            logits = RunModel(features);
        }
        else
        {
            logits = RunOnnxInference(features);
        }

        // CTC decode
        var tokens = CTCGreedyDecode(logits);
        var text = TokensToText(tokens);

        return new TranscriptionResult<T>
        {
            Text = text,
            Language = language ?? _language ?? "en",
            Confidence = NumOps.FromDouble(1.0),
            DurationSeconds = (double)audio.Shape[0] / SampleRate,
            Segments = ResolveReturnTimestamps(includeTimestamps) ? ExtractSegments(tokens, text, audio.Shape[0]) : Array.Empty<TranscriptionSegment<T>>()
        };
    }

    /// <summary>
    /// Transcribes audio to text asynchronously.
    /// </summary>
    public Task<TranscriptionResult<T>> TranscribeAsync(
        Tensor<T> audio,
        string? language = null,
        bool? includeTimestamps = null,
        CancellationToken cancellationToken = default)
    {
        return Task.Run(() => Transcribe(audio, language, includeTimestamps), cancellationToken);
    }

    /// <summary>
    /// Detects the language spoken in the audio.
    /// </summary>
    public string DetectLanguage(Tensor<T> audio)
    {
        // Wav2Vec2 is typically monolingual, return the configured language
        return _language ?? "en";
    }

    /// <summary>
    /// Gets language detection probabilities for the audio.
    /// </summary>
    public IReadOnlyDictionary<string, T> DetectLanguageProbabilities(Tensor<T> audio)
    {
        // Wav2Vec2 is typically monolingual
        var result = new Dictionary<string, T>
        {
            [_language ?? "en"] = NumOps.FromDouble(1.0)
        };
        return result;
    }

    /// <summary>
    /// Starts a streaming transcription session.
    /// </summary>
    public IStreamingTranscriptionSession<T> StartStreamingSession(string? language = null)
    {
        throw new NotSupportedException("Wav2Vec2Model does not support streaming transcription.");
    }

    #endregion

    #region AudioNeuralNetworkBase Implementation

    /// <summary>
    /// Preprocesses raw audio for model input.
    /// </summary>
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio)
    {
        // Wav2Vec2 consumes the RAW waveform directly (Baevski et al. 2020 — the whole premise of
        // wav2vec2 is learning from raw audio, not mel features). Flatten the input to a 1-D sample
        // stream on the TOTAL element count (rawAudio.Length), not Shape[0]: callers (and the
        // model-family tests) may pass a shaped tensor like [1, frames, samples] whose Shape[0] is a
        // batch dim of 1 — keying off Shape[0] collapsed the clip to one sample.
        //
        // Do NOT peak-normalize (÷max) here: that was both non-paper-accurate (the paper uses
        // zero-mean/unit-variance, applied as EXTERNAL preprocessing on real waveforms) AND
        // information-destroying for the synthetic test inputs — ÷max maps every constant input to
        // all-ones and cancels any input scaling, collapsing the model's output (failing
        // DifferentInputs / ScaledInput). The sibling ASR models that pass these invariants
        // (UniSpeech, RobustConformer) likewise treat the given tensor as the prepared feature stream.
        int targetLength = SampleRate * _maxAudioLengthSeconds;
        int length = Math.Min(rawAudio.Length, targetLength);

        var waveform = new Tensor<T>([length]);
        rawAudio.Data.Span.Slice(0, length).CopyTo(waveform.Data.Span);
        return waveform;
    }

    /// <summary>
    /// Postprocesses model output.
    /// </summary>
    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput)
    {
        return modelOutput;
    }

    /// <summary>
    /// Makes a prediction using the model.
    /// </summary>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        var preprocessed = PreprocessAudio(input);

        if (!_useNativeMode)
        {
            return RunOnnxInference(preprocessed);
        }
        else
        {
            return RunModel(preprocessed);
        }
    }

    /// <summary>
    /// Runs the native wav2vec 2.0 forward on a 1-D waveform: Conv1D feature encoder ->
    /// transpose -> feature projection -> transformer -> CTC head. Used by BOTH inference
    /// (<see cref="PredictCore"/> / <see cref="Transcribe"/>) and training
    /// (<see cref="ForwardForTraining"/>) so the two agree, and every op records the autodiff tape.
    /// </summary>
    /// <remarks>
    /// A plain linear layer-walk cannot express this network: the conv encoder emits [B, C, T']
    /// (channels-major) but the transformer attends over frames [B, T', C], so a transpose is
    /// required between them — done here with <c>Engine.TensorPermute</c> (the same pattern SileroVad
    /// uses). The conv encoder also preserves the temporal sequence, which the earlier DenseLayer
    /// stack destroyed (collapsing every input to one vector -> input-independent output).
    /// </remarks>
    // Minimum waveform length fed to the Conv1D encoder. The 7 paper strides downsample by 320x, so a
    // clip shorter than this collapses to zero frames ("Invalid output dimensions (1x0)"). Short inputs
    // (e.g. minimal-input invariants) are zero-padded up to this length before the encoder.
    // CTC logits: [frames, vocab] for a waveform [samples], [batch, frames, vocab] for [batch, samples]. A waveform
    // shorter than the feature encoder's receptive field is zero-padded to it.
    private Tensor<T> RunModel(Tensor<T> waveform, bool training = false)
    {
        if (_network is null || _ctcHead is null) throw new InvalidOperationException("The native network is not initialized.");
        if (waveform.Rank > 2) waveform = Engine.Reshape(waveform, new[] { waveform.Shape[0], waveform.Length / waveform.Shape[0] });
        int field = ReceptiveField();
        if (waveform.Shape[^1] < field)
        {
            var padded = new Tensor<T>(waveform.Rank == 1 ? new[] { field } : new[] { waveform.Shape[0], field });
            int length = waveform.Shape[^1], rows = waveform.Rank == 1 ? 1 : waveform.Shape[0];
            for (int r = 0; r < rows; r++)
                for (int i = 0; i < length; i++) padded[r * field + i] = waveform[r * length + i];
            waveform = padded;
        }
        var output = _network.Forward(waveform, training, _random, freezeFeatureEncoder: training && _options.FreezeFeatureEncoder);
        var hidden = output.Hidden;
        if (training && _options.FinalDropout > 0)
        {
            var mask = new Tensor<T>(hidden._shape);
            var keep = NumOps.FromDouble(1.0 / (1.0 - _options.FinalDropout));
            for (int i = 0; i < mask.Length; i++) mask[i] = _random.NextDouble() < _options.FinalDropout ? NumOps.Zero : keep;
            hidden = Engine.TensorMultiply(hidden, mask);
        }
        int batch = hidden.Shape[0], frames = hidden.Shape[1];
        var logits = Engine.Reshape(_ctcHead.Forward(Engine.Reshape(hidden, new[] { batch * frames, hidden.Shape[2] })), new[] { batch, frames, _vocabSize });
        return waveform.Rank == 1 ? Engine.Reshape(logits, new[] { frames, _vocabSize }) : logits;
    }

    // The fewest samples that give one frame.
    private int ReceptiveField()
    {
        int field = 1;
        var c = _network!.Configuration;
        for (int i = c.ConvKernels.Length - 1; i >= 0; i--) field = (field - 1) * c.ConvStrides[i] + c.ConvKernels[i];
        return field;
    }

    public override Tensor<T> ForwardForTraining(Tensor<T> input)
        => RunModel(input, training: true);

    /// <inheritdoc />
    /// <remarks>With <see cref="Wav2Vec2ModelOptions.FreezeFeatureEncoder"/> the convolutional feature encoder's
    /// parameters are frozen: fine-tuning does not train them (and the forward computes no gradient for them).</remarks>
    protected override AiDotNet.Models.Parameters.ParameterSlotRole ResolveLayerParameterRole(int layerIndex, AiDotNet.Models.Parameters.ParameterSlotRole declared)
    {
        if (declared != AiDotNet.Models.Parameters.ParameterSlotRole.Trainable || !_options.FreezeFeatureEncoder || _network is null)
            return declared;
        var layer = Layers[layerIndex];
        return _network.ConvLayers.Any(c => ReferenceEquals(c.Conv, layer) || ReferenceEquals(c.Norm, layer))
            ? AiDotNet.Models.Parameters.ParameterSlotRole.Frozen
            : declared;
    }

    /// <inheritdoc />
    /// <remarks>The forward is not a fold of <c>Layers</c>; the activations are those the recognizer computes.</remarks>
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
        => _useNativeMode ? ObservedLayerActivations(input) : new Dictionary<string, Tensor<T>>();

    protected override bool SupportsParameterMutation => _useNativeMode;
    /// <summary>
    /// Gets metadata about the model.
    /// </summary>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "Wav2Vec2",
            Description = "Wav2Vec2 self-supervised speech recognition model",
            FeatureCount = SampleRate * _maxAudioLengthSeconds,
            Complexity = 3
        };
        metadata.AdditionalInfo["InputFormat"] = $"Raw audio ({SampleRate}Hz, {_maxAudioLengthSeconds}s max)";
        metadata.AdditionalInfo["OutputFormat"] = "Transcription";
        metadata.AdditionalInfo["Mode"] = _useNativeMode ? "Native" : "ONNX";
        metadata.AdditionalInfo["HiddenDim"] = _hiddenDim.ToString();
        metadata.AdditionalInfo["TransformerLayers"] = _numTransformerLayers.ToString();
        return metadata;
    }

    /// <summary>
    /// Serializes network-specific data.
    /// </summary>


    /// <summary>
    /// Deserializes network-specific data.
    /// </summary>


    #endregion

    #region Private Methods

    private List<int> CTCGreedyDecode(Tensor<T> logits)
    {
        var tokens = new List<int>();
        int prevToken = -1;
        int blankIdx = 0; // Blank token is first in vocabulary

        if (logits.Rank == 3)
        {
            if (logits.Shape[0] != 1) throw new ArgumentException("Greedy CTC decoding takes one utterance's logits.", nameof(logits));
            logits = Engine.Reshape(logits, new[] { logits.Shape[1], logits.Shape[2] });
        }
        int numFrames = logits.Rank >= 2 ? logits.Shape[0] : 1;
        int vocabSize = logits.Rank >= 2 ? logits.Shape[^1] : logits.Shape[0];

        for (int t = 0; t < numFrames; t++)
        {
            // Find argmax
            int maxIdx = 0;
            double maxVal = double.NegativeInfinity;

            for (int v = 0; v < vocabSize; v++)
            {
                double val = logits.Rank >= 2
                    ? NumOps.ToDouble(logits[t, v])
                    : NumOps.ToDouble(logits[v]);
                if (val > maxVal)
                {
                    maxVal = val;
                    maxIdx = v;
                }
            }

            // CTC decoding: collapse repeated tokens and remove blanks
            if (maxIdx != blankIdx && maxIdx != prevToken)
            {
                tokens.Add(maxIdx);
            }
            prevToken = maxIdx;
        }

        return tokens;
    }

    private string TokensToText(List<int> tokens)
    {
        var chars = new List<char>();

        foreach (var token in tokens)
        {
            if (token >= 0 && token < _vocabulary.Length)
            {
                var symbol = _vocabulary[token];
                if (symbol == "|" || symbol == " ")
                {
                    chars.Add(' ');
                }
                else if (symbol.Length == 1 && char.IsLetter(symbol[0]))
                {
                    chars.Add(symbol[0]);
                }
                else if (symbol == "'")
                {
                    chars.Add('\'');
                }
            }
        }

        return new string(chars.ToArray()).Trim();
    }

    private IReadOnlyList<TranscriptionSegment<T>> ExtractSegments(List<int> tokens, string text, int audioLength)
    {
        if (tokens.Count == 0 || string.IsNullOrWhiteSpace(text))
            return Array.Empty<TranscriptionSegment<T>>();

        double duration = (double)audioLength / SampleRate;

        return new[]
        {
            new TranscriptionSegment<T>
            {
                Text = text,
                StartTime = 0.0,
                EndTime = duration,
                Confidence = NumOps.FromDouble(1.0)
            }
        };
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName);
    }

    #endregion

    #region IDisposable

    /// <summary>
    /// Disposes the model and releases resources.
    /// </summary>
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;

        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
