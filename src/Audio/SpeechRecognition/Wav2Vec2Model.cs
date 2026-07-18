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
public class Wav2Vec2Model<T> : AudioNeuralNetworkBase<T>, ISpeechRecognizer<T>
{
    private readonly Wav2Vec2ModelOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

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

    /// <summary>
    /// Convolutional feature encoder layers.
    /// </summary>
    private List<ILayer<T>> _featureEncoderLayers = [];

    /// <summary>
    /// Transformer encoder layers.
    /// </summary>
    private List<ILayer<T>> _transformerLayers = [];

    /// <summary>
    /// CTC projection layer.
    /// </summary>
    private ILayer<T>? _ctcProjection;

    #endregion

    #region Shared Fields

    /// <summary>
    /// Optimizer for training (unused in ONNX mode).
    /// </summary>
    private IOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;

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
    /// Number of Conv1D layers in the feature encoder (wav2vec 2.0 uses 7). The first
    /// <c>FeatureConvCount</c> entries of <see cref="_featureEncoderLayers"/> are the temporal
    /// convolutions (run BEFORE the channels-&gt;time transpose in <see cref="RunModel"/>); any
    /// remaining entries are the post-transpose feature projection.
    /// </summary>
    private const int FeatureConvCount = 7;

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
        string? language = "en",
        int sampleRate = 16000,
        int maxAudioLengthSeconds = 30,
        string[]? vocabulary = null,
        OnnxModelOptions? onnxOptions = null,
        Wav2Vec2ModelOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new Wav2Vec2ModelOptions();
        Options = _options;
        if (architecture is null)
            throw new ArgumentNullException(nameof(architecture));
        if (modelPath is null)
            throw new ArgumentNullException(nameof(modelPath));

        _useNativeMode = false;
        _modelPath = modelPath;
        _language = language;
        _maxAudioLengthSeconds = maxAudioLengthSeconds;

        // Set audio properties
        SampleRate = sampleRate;
        NumMels = 0; // Wav2Vec2 doesn't use mel spectrograms

        // Model dimensions (standard Wav2Vec2 Base)
        _hiddenDim = 768;
        _numTransformerLayers = 12;
        _numHeads = 12;
        _ffDim = 3072;

        // Initialize vocabulary
        _vocabulary = vocabulary ?? GetDefaultVocabulary();
        _vocabSize = _vocabulary.Length;

        // Load ONNX model
        var onnxOpts = onnxOptions ?? new OnnxModelOptions();
        OnnxModel = new OnnxModel<T>(modelPath, onnxOpts);

        // Initialize supported languages
        SupportedLanguages = new[] { language ?? "en" };

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
        string? language = "en",
        int sampleRate = 16000,
        int maxAudioLengthSeconds = 30,
        int hiddenDim = 768,
        int numTransformerLayers = 12,
        int numHeads = 12,
        int ffDim = 3072,
        string[]? vocabulary = null,
        IOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        Wav2Vec2ModelOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new Wav2Vec2ModelOptions();
        Options = _options;
        if (architecture is null)
            throw new ArgumentNullException(nameof(architecture));

        _useNativeMode = true;
        _language = language;
        _maxAudioLengthSeconds = maxAudioLengthSeconds;
        _hiddenDim = hiddenDim;
        _numTransformerLayers = numTransformerLayers;
        _numHeads = numHeads;
        _ffDim = ffDim;

        // Set audio properties
        SampleRate = sampleRate;
        NumMels = 0; // Wav2Vec2 doesn't use mel spectrograms

        // Initialize vocabulary
        _vocabulary = vocabulary ?? GetDefaultVocabulary();
        _vocabSize = _vocabulary.Length;

        // Initialize supported languages
        SupportedLanguages = new[] { language ?? "en" };

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
        var layers = (Architecture.Layers != null && Architecture.Layers.Count > 0)
            ? Architecture.Layers.ToList()
            : LayerHelper<T>.CreateWav2Vec2Layers(
                hiddenDim: _hiddenDim, numTransformerLayers: _numTransformerLayers,
                numHeads: _numHeads, ffDim: _ffDim, vocabSize: _vocabSize,
                sampleRate: SampleRate, maxAudioLengthSeconds: _maxAudioLengthSeconds).ToList();

        Layers.Clear();
        Layers.AddRange(layers);

        // Feature encoder: 7 Conv1D + 1 feature-projection Dense + 1 feature-projection LayerNorm = 9
        // (see CreateWav2Vec2Layers — the LayerNorm was added for paper-faithful feature projection).
        int featureEncoderCount = 9;
        // Transformer layers: numTransformerLayers residual TransformerEncoderBlocks (one per layer;
        // each block internally does MHA + FFN + the two LayerNorms), + 1 CTC projection.
        int transformerCount = _numTransformerLayers;
        int expectedTotal = featureEncoderCount + transformerCount + 1;

        if (Architecture.Layers != null && layers.Count != expectedTotal)
        {
            System.Diagnostics.Debug.WriteLine(
                $"[Wav2Vec2] Warning: Expected {expectedTotal} layers (9 encoder + {transformerCount} transformer + 1 CTC), " +
                $"but got {layers.Count}. Layer distribution may be incorrect.");
        }

        DistributeLayersToSubLists();
    }

    // Re-links the typed forward-path sub-lists (_featureEncoderLayers / _transformerLayers /
    // _ctcProjection) to the CURRENT contents of Layers. The forward reads these fields, NOT Layers,
    // so they must be rebuilt whenever Layers is replaced — critically after deserialization, where the
    // base clears Layers and adds freshly-deserialized (trained) layers. Without this, a cloned/loaded
    // model keeps its ctor's random-init sub-list layers and predicts as if untrained (#1221 class:
    // Clone_AfterTraining). Distribution order matches CreateWav2Vec2Layers: [0..8]=feature encoder,
    // [9..9+N-1]=transformer (one residual TransformerEncoderBlock each), [^1]=CTC projection.
    private void DistributeLayersToSubLists()
    {
        _featureEncoderLayers.Clear();
        _transformerLayers.Clear();

        int featureEncoderCount = 9;
        int transformerCount = _numTransformerLayers;

        for (int i = 0; i < featureEncoderCount && i < Layers.Count; i++)
            _featureEncoderLayers.Add(Layers[i]);

        int transformerStart = featureEncoderCount;
        for (int i = 0; i < transformerCount && transformerStart + i < Layers.Count; i++)
            _transformerLayers.Add(Layers[transformerStart + i]);

        if (Layers.Count > 0)
            _ctcProjection = Layers[^1];
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
    public TranscriptionResult<T> Transcribe(Tensor<T> audio, string? language = null, bool includeTimestamps = false)
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
            Segments = includeTimestamps ? ExtractSegments(tokens, text, audio.Shape[0]) : Array.Empty<TranscriptionSegment<T>>()
        };
    }

    /// <summary>
    /// Transcribes audio to text asynchronously.
    /// </summary>
    public Task<TranscriptionResult<T>> TranscribeAsync(
        Tensor<T> audio,
        string? language = null,
        bool includeTimestamps = false,
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
    private const int MinEncoderSamples = 1024;

    private bool _shapesProbed;

    /// <inheritdoc/>
    protected override void ResolveLazyLayerShapes()
    {
        // The real forward (RunModel) reshapes the waveform to [1, 1, T] before the Conv1D encoder, so
        // the first conv must resolve to in_channels = 1. The base linear walk instead feeds the
        // architecture input shape ([1, 64, 32]) straight into the first conv, resolving it to
        // in_channels = 64 — after which the real forward throws "Input channels (1) must match kernel
        // in_channels (64)". Probe the real forward once on a tiny dummy waveform so every lazy layer
        // resolves to what RunModel actually feeds it.
        if (_shapesProbed || !_useNativeMode || _featureEncoderLayers.Count == 0) return;
        _shapesProbed = true;
        _ = RunModel(new Tensor<T>(new[] { MinEncoderSamples }));
    }

    private Tensor<T> RunModel(Tensor<T> waveform)
    {
        // Zero-pad clips shorter than the encoder's receptive field so the 320x downsample leaves at
        // least a few frames instead of collapsing to length 0.
        if (waveform.Length < MinEncoderSamples)
        {
            var padded = new Tensor<T>(new[] { MinEncoderSamples });
            waveform.Data.Span.CopyTo(padded.Data.Span);
            waveform = padded;
        }

        // Reshape the flat waveform [T] to [B=1, C=1, T] for the Conv1D feature encoder.
        var x = Engine.Reshape(waveform, new[] { 1, 1, waveform.Length });

        // Conv feature encoder: _featureEncoderLayers[0..FeatureConvCount-1] : [1,1,T] -> [1,512,T'].
        int convCount = Math.Min(FeatureConvCount, _featureEncoderLayers.Count);
        for (int i = 0; i < convCount; i++)
            x = _featureEncoderLayers[i].Forward(x);

        // Transpose channels<->time: [1,512,T'] -> [1,T',512] so the transformer attends over frames.
        x = Engine.TensorPermute(x, new[] { 0, 2, 1 });

        // Feature projection: remaining _featureEncoderLayers : [1,T',512] -> [1,T',hidden].
        for (int i = convCount; i < _featureEncoderLayers.Count; i++)
            x = _featureEncoderLayers[i].Forward(x);

        // Transformer encoder (MHA + FFN blocks, walked in order as they were emitted).
        foreach (var layer in _transformerLayers)
            x = layer.Forward(x);

        // CTC projection: [1,T',hidden] -> [1,T',vocab].
        if (_ctcProjection is not null)
            x = _ctcProjection.Forward(x);

        return x;
    }

    /// <inheritdoc/>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
        => RunModel(input);

    /// <inheritdoc/>
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
    {
        // The base walks Layers linearly feeding the raw [1, 64, 32] input, which the Conv1D encoder
        // rejects (it expects [B, 1, T] and would resolve to the wrong channel count / collapse to 0
        // frames). Replay the real custom forward (RunModel) instead, capturing each layer's output.
        var activations = new Dictionary<string, Tensor<T>>();
        if (!_useNativeMode)
            return activations;

        var waveform = PreprocessAudio(input);
        if (waveform.Length < MinEncoderSamples)
        {
            var padded = new Tensor<T>(new[] { MinEncoderSamples });
            waveform.Data.Span.CopyTo(padded.Data.Span);
            waveform = padded;
        }

        var x = Engine.Reshape(waveform, new[] { 1, 1, waveform.Length });
        int idx = 0;
        int convCount = Math.Min(FeatureConvCount, _featureEncoderLayers.Count);
        for (int i = 0; i < convCount; i++)
        {
            x = _featureEncoderLayers[i].Forward(x);
            activations[$"Layer_{idx++}_{_featureEncoderLayers[i].GetType().Name}"] = x.Clone();
        }
        x = Engine.TensorPermute(x, new[] { 0, 2, 1 });
        for (int i = convCount; i < _featureEncoderLayers.Count; i++)
        {
            x = _featureEncoderLayers[i].Forward(x);
            activations[$"Layer_{idx++}_{_featureEncoderLayers[i].GetType().Name}"] = x.Clone();
        }
        foreach (var layer in _transformerLayers)
        {
            x = layer.Forward(x);
            activations[$"Layer_{idx++}_{layer.GetType().Name}"] = x.Clone();
        }
        if (_ctcProjection is not null)
        {
            x = _ctcProjection.Forward(x);
            activations[$"Layer_{idx}_{_ctcProjection.GetType().Name}"] = x.Clone();
        }
        return activations;
    }

    /// <summary>
    /// Updates model parameters by applying gradient descent.
    /// </summary>
    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode)
        {
            throw new NotSupportedException("Cannot update parameters in ONNX inference mode.");
        }

        // NeuralNetworkBase.UpdateParameters contract: caller passes NEW
        // parameter values (post-optimizer-step), NOT raw gradients. The
        // previous body computed `current − lr · input` then SetParameters,
        // which on top of Adam's own update produced a double-step that
        // destabilised training. Forward straight to SetParameters per the
        // contract — Adam already produced the correct new values.
        SetParameters(parameters);
    }

    /// <summary>
    /// Trains the model on a single batch.
    /// </summary>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_useNativeMode)
        {
            throw new NotSupportedException("Cannot train in ONNX inference mode.");
        }

        SetTrainingMode(true);
        try
        {
            // Pass the model's own non-AMSGrad AdamOptimizer explicitly.
            // The optimizer-null branch would otherwise fall back to
            // GetOrCreateBaseOptimizer (which builds an AMSGrad Adam),
            // and the fused-Adam fast path bails out on AMSGrad — leaving
            // every step on the BERT-base-scale wav2vec2 encoder running
            // through the eager tape executor at multi-second cost per
            // iteration.
            //
            // The cast goes through `as ... ?? throw` rather than plain
            // `as` so a user passing a non-gradient optimizer fails loudly
            // instead of silently dropping into the default-optimizer
            // fallback (would mask intent and produce mysteriously-different
            // training trajectories). PR #1404 review (CodeRabbit).
            var gradientOptimizer = _optimizer as IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>
                ?? throw new InvalidOperationException(
                    "Wav2Vec2Model training requires an optimizer implementing IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>.");
            // Train on the SAME preprocessed feature stream inference runs on (PredictCore ->
            // PreprocessAudio -> Forward). Feeding raw input straight to the tape produced a
            // different-shaped forward than Predict ([1,64,34] vs the preprocessed [.,34]), so the
            // MSE loss shape-mismatched and every training-based invariant failed.
            TrainWithTape(PreprocessAudio(input), expectedOutput, gradientOptimizer);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

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
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_useNativeMode);
        writer.Write(SampleRate);
        writer.Write(_maxAudioLengthSeconds);
        writer.Write(_hiddenDim);
        writer.Write(_numTransformerLayers);
        writer.Write(_numHeads);
        writer.Write(_ffDim);
        writer.Write(_vocabSize);
        writer.Write(_language ?? string.Empty);
    }

    /// <summary>
    /// Deserializes network-specific data.
    /// </summary>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _useNativeMode = reader.ReadBoolean();
        SampleRate = reader.ReadInt32();
        _maxAudioLengthSeconds = reader.ReadInt32();
        _hiddenDim = reader.ReadInt32();
        _numTransformerLayers = reader.ReadInt32();
        _numHeads = reader.ReadInt32();
        _ffDim = reader.ReadInt32();
        _vocabSize = reader.ReadInt32();
        _language = reader.ReadString();

        // Reinitialize / re-link layers for native mode. The base deserialize has already populated
        // Layers with the trained layers; re-link the typed forward-path sub-lists to THEM (the ctor
        // populated them from fresh random layers, and the forward reads the sub-lists, not Layers —
        // so without this a cloned/loaded model predicts untrained: #1221 Clone_AfterTraining).
        if (_useNativeMode)
        {
            if (Layers.Count > 0)
                DistributeLayersToSubLists();
            else
                // Layers.Count == 0 (older/empty native payload): rebuild the default native layers.
                // InitializeLayers() is the ONNX no-op and would leave a native model with no
                // feature-encoder/transformer/CTC layers at all — a silently broken model.
                InitializeNativeLayers();
        }
    }

    /// <summary>
    /// Creates a new instance of this model for cloning.
    /// </summary>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (_useNativeMode)
        {
            return new Wav2Vec2Model<T>(
                Architecture,
                language: _language,
                sampleRate: SampleRate,
                maxAudioLengthSeconds: _maxAudioLengthSeconds,
                hiddenDim: _hiddenDim,
                numTransformerLayers: _numTransformerLayers,
                numHeads: _numHeads,
                ffDim: _ffDim,
                vocabulary: _vocabulary);
        }
        else
        {
            return new Wav2Vec2Model<T>(
                Architecture,
                modelPath: _modelPath!,
                language: _language,
                sampleRate: SampleRate,
                maxAudioLengthSeconds: _maxAudioLengthSeconds,
                vocabulary: _vocabulary);
        }
    }

    #endregion

    #region Private Methods

    private List<int> CTCGreedyDecode(Tensor<T> logits)
    {
        var tokens = new List<int>();
        int prevToken = -1;
        int blankIdx = 0; // Blank token is first in vocabulary

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
