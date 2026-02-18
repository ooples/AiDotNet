using AiDotNet.ActivationFunctions;
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
/// Console.WriteLine(result.Text);
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

        // Default loss function (cross-entropy is standard for ASR)
        _lossFunction = new CrossEntropyLoss<T>();

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

        // Initialize training components
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new CrossEntropyLoss<T>();

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
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            return;
        }

        var layers = LayerHelper<T>.CreateWav2Vec2Layers(
            hiddenDim: _hiddenDim, numTransformerLayers: _numTransformerLayers,
            numHeads: _numHeads, ffDim: _ffDim, vocabSize: _vocabSize,
            sampleRate: SampleRate, maxAudioLengthSeconds: _maxAudioLengthSeconds).ToList();
        Layers.AddRange(layers);

        // Distribute to internal sub-lists for forward pass
        // Feature encoder: 7 conv layers + 1 projection = 8
        int featureEncoderCount = 8;
        for (int i = 0; i < featureEncoderCount; i++)
            _featureEncoderLayers.Add(layers[i]);

        // Transformer layers: numTransformerLayers * 3 (selfAttn + ff + ffOut)
        int transformerStart = featureEncoderCount;
        int transformerCount = _numTransformerLayers * 3;
        for (int i = 0; i < transformerCount; i++)
            _transformerLayers.Add(layers[transformerStart + i]);

        // CTC projection: last layer
        _ctcProjection = layers[^1];
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
            logits = Forward(features);
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
        // Wav2Vec2 processes raw audio directly
        // Normalize to [-1, 1] range
        int targetLength = SampleRate * _maxAudioLengthSeconds;

        var normalized = new Tensor<T>([Math.Min(rawAudio.Shape[0], targetLength)]);

        // Find max for normalization
        double maxVal = 0;
        for (int i = 0; i < normalized.Shape[0]; i++)
        {
            double val = Math.Abs(NumOps.ToDouble(rawAudio[i]));
            if (val > maxVal) maxVal = val;
        }

        if (maxVal > 0)
        {
            for (int i = 0; i < normalized.Shape[0]; i++)
            {
                normalized[i] = NumOps.FromDouble(NumOps.ToDouble(rawAudio[i]) / maxVal);
            }
        }
        else
        {
            for (int i = 0; i < normalized.Shape[0]; i++)
            {
                normalized[i] = rawAudio[i];
            }
        }

        return normalized;
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
    public override Tensor<T> Predict(Tensor<T> input)
    {
        var preprocessed = PreprocessAudio(input);

        if (!_useNativeMode)
        {
            return RunOnnxInference(preprocessed);
        }
        else
        {
            return Forward(preprocessed);
        }
    }

    /// <summary>
    /// Updates model parameters by applying gradient descent.
    /// </summary>
    public override void UpdateParameters(Vector<T> gradients)
    {
        if (!_useNativeMode)
        {
            throw new NotSupportedException("Cannot update parameters in ONNX inference mode.");
        }

        var currentParams = GetParameters();
        T learningRate = NumOps.FromDouble(0.0001); // Wav2Vec2 uses smaller learning rate

        for (int i = 0; i < currentParams.Length; i++)
        {
            currentParams[i] = NumOps.Subtract(currentParams[i], NumOps.Multiply(learningRate, gradients[i]));
        }

        SetParameters(currentParams);
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

        // Forward pass
        var features = PreprocessAudio(input);
        var prediction = Forward(features);

        // Calculate loss
        var flatPrediction = prediction.ToVector();
        var flatExpected = expectedOutput.ToVector();
        LastLoss = _lossFunction.CalculateLoss(flatPrediction, flatExpected);

        // Backward pass
        var outputGradients = _lossFunction.CalculateDerivative(flatPrediction, flatExpected);
        Backpropagate(Tensor<T>.FromVector(outputGradients));

        // Update parameters
        var parameterGradients = GetParameterGradients();
        UpdateParameters(parameterGradients);

        SetTrainingMode(false);
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
            ModelType = ModelType.NeuralNetwork,
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

        // Reinitialize layers if needed for native mode
        if (_useNativeMode && (_featureEncoderLayers is null || _featureEncoderLayers.Count == 0))
        {
            InitializeLayers();
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
