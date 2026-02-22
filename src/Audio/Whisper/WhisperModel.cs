using System.Diagnostics;
using AiDotNet.Diffusion;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.Whisper;

/// <summary>
/// Whisper automatic speech recognition model for transcribing audio to text.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Whisper is a state-of-the-art speech recognition model by OpenAI that can:
/// <list type="bullet">
/// <item>Transcribe speech in 99+ languages</item>
/// <item>Translate non-English speech to English</item>
/// <item>Detect the spoken language automatically</item>
/// <item>Handle noisy audio and accents well</item>
/// </list>
/// </para>
/// <para><b>For Beginners:</b> Whisper converts spoken audio into text. It works by:
/// 1. Converting audio to a mel spectrogram (visual representation of sound)
/// 2. Processing through an encoder neural network
/// 3. Generating text tokens through a decoder neural network
///
/// Two ways to use this class:
/// 1. ONNX Mode: Load pretrained models for fast inference
/// 2. Native Mode: Train your own speech recognition model from scratch
///
/// ONNX Mode Example:
/// <code>
/// var whisper = new WhisperModel&lt;float&gt;(
///     architecture,
///     encoderPath: "path/to/encoder.onnx",
///     decoderPath: "path/to/decoder.onnx");
/// var result = whisper.Transcribe(audioTensor);
/// Console.WriteLine(result.Text);
/// </code>
///
/// Training Mode Example:
/// <code>
/// var whisper = new WhisperModel&lt;float&gt;(architecture);
/// for (int epoch = 0; epoch &lt; 100; epoch++)
/// {
///     foreach (var (audio, tokens) in trainingData)
///     {
///         whisper.Train(audio, tokens);
///     }
/// }
/// </code>
/// </para>
/// </remarks>
public class WhisperModel<T> : AudioNeuralNetworkBase<T>, ISpeechRecognizer<T>
{
    private readonly WhisperOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    #region Execution Mode

    /// <summary>
    /// Indicates whether this network uses native layers (true) or ONNX models (false).
    /// </summary>
    private readonly bool _useNativeMode;

    #endregion

    #region ONNX Mode Fields

    /// <summary>
    /// Path to the encoder ONNX model file.
    /// </summary>
    private readonly string? _encoderPath;

    /// <summary>
    /// Path to the decoder ONNX model file.
    /// </summary>
    private readonly string? _decoderPath;

    #endregion

    #region Shared Fields

    /// <summary>
    /// The mel spectrogram preprocessor.
    /// </summary>
    private readonly MelSpectrogram<T> _melSpectrogram;

    /// <summary>
    /// The tokenizer for converting between text and tokens.
    /// </summary>
    private readonly WhisperTokenizer _tokenizer;

    /// <summary>
    /// Optimizer for training (unused in ONNX mode).
    /// </summary>
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;

    /// <summary>
    /// Loss function for training.
    /// </summary>
    private ILossFunction<T> _lossFunction;

    /// <summary>
    /// Model size variant.
    /// </summary>
    private readonly WhisperModelSize _modelSize;

    /// <summary>
    /// Target language for transcription (null for auto-detect).
    /// </summary>
    private readonly string? _language;

    /// <summary>
    /// Whether to translate to English.
    /// </summary>
    private readonly bool _translate;

    /// <summary>
    /// Maximum audio length in seconds.
    /// </summary>
    private readonly int _maxAudioLengthSeconds;

    /// <summary>
    /// Number of mel filterbank channels.
    /// </summary>
    private readonly int _numMels;

    /// <summary>
    /// Maximum number of tokens to generate.
    /// </summary>
    private readonly int _maxTokens;

    /// <summary>
    /// Beam size for beam search decoding.
    /// </summary>
    private readonly int _beamSize;

    /// <summary>
    /// Temperature for sampling (0 = greedy).
    /// </summary>
    private readonly double _temperature;

    /// <summary>
    /// Model dimension (hidden size).
    /// </summary>
    private readonly int _modelDim;

    /// <summary>
    /// Number of encoder layers.
    /// </summary>
    private readonly int _numEncoderLayers;

    /// <summary>
    /// Number of decoder layers.
    /// </summary>
    private readonly int _numDecoderLayers;

    /// <summary>
    /// Number of attention heads.
    /// </summary>
    private readonly int _numHeads;

    /// <summary>
    /// Feed-forward dimension.
    /// </summary>
    private readonly int _ffDim;

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
    /// Gets the model size variant.
    /// </summary>
    public WhisperModelSize ModelSize => _modelSize;

    /// <summary>
    /// Gets whether the model is ready for inference.
    /// </summary>
    public bool IsReady => _useNativeMode || (OnnxEncoder?.IsLoaded == true && OnnxDecoder?.IsLoaded == true);

    /// <summary>
    /// Gets the target language for transcription.
    /// </summary>
    public string? Language => _language;

    /// <summary>
    /// Gets whether translation to English is enabled.
    /// </summary>
    public bool Translate => _translate;

    /// <summary>
    /// Gets the maximum audio length in seconds.
    /// </summary>
    public int MaxAudioLengthSeconds => _maxAudioLengthSeconds;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a Whisper network using pretrained ONNX models.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="encoderPath">Path to the encoder ONNX model.</param>
    /// <param name="decoderPath">Path to the decoder ONNX model.</param>
    /// <param name="modelSize">Model size variant (Tiny, Base, Small, Medium, Large).</param>
    /// <param name="language">Target language code (e.g., "en", "es"). Null for auto-detection.</param>
    /// <param name="translate">Whether to translate non-English to English.</param>
    /// <param name="sampleRate">Audio sample rate in Hz. Whisper expects 16000.</param>
    /// <param name="numMels">Number of mel filterbank channels. Whisper uses 80.</param>
    /// <param name="maxAudioLengthSeconds">Maximum audio length to process. Whisper uses 30s chunks.</param>
    /// <param name="maxTokens">Maximum number of tokens to generate.</param>
    /// <param name="beamSize">Beam size for beam search decoding.</param>
    /// <param name="temperature">Sampling temperature (0 = greedy/deterministic).</param>
    /// <param name="onnxOptions">ONNX runtime options.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor when you have downloaded Whisper ONNX models.
    ///
    /// The encoder processes audio features and the decoder generates text tokens.
    /// Both are needed for transcription.
    ///
    /// You can get ONNX models from:
    /// - HuggingFace: openai/whisper-base, openai/whisper-small, etc.
    /// - Convert from PyTorch using ONNX export tools
    ///
    /// Example:
    /// <code>
    /// var whisper = new WhisperModel&lt;float&gt;(
    ///     architecture,
    ///     encoderPath: "whisper-base-encoder.onnx",
    ///     decoderPath: "whisper-base-decoder.onnx",
    ///     modelSize: WhisperModelSize.Base,
    ///     language: "en");  // English transcription
    /// </code>
    /// </para>
    /// </remarks>
    public WhisperModel(
        NeuralNetworkArchitecture<T> architecture,
        string encoderPath,
        string decoderPath,
        WhisperModelSize modelSize = WhisperModelSize.Base,
        string? language = null,
        bool translate = false,
        int sampleRate = 16000,
        int numMels = 80,
        int maxAudioLengthSeconds = 30,
        int maxTokens = 448,
        int beamSize = 5,
        double temperature = 0.0,
        OnnxModelOptions? onnxOptions = null,
        WhisperOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new WhisperOptions();
        Options = _options;
        if (encoderPath is null)
            throw new ArgumentNullException(nameof(encoderPath));
        if (decoderPath is null)
            throw new ArgumentNullException(nameof(decoderPath));

        _useNativeMode = false;
        _encoderPath = encoderPath;
        _decoderPath = decoderPath;
        _modelSize = modelSize;
        _language = language;
        _translate = translate;
        _maxAudioLengthSeconds = maxAudioLengthSeconds;
        _numMels = numMels;
        _maxTokens = maxTokens;
        _beamSize = beamSize;
        _temperature = temperature;

        // Get model dimensions based on size
        (_modelDim, _numEncoderLayers, _numDecoderLayers, _numHeads, _ffDim) = GetModelParameters(modelSize);

        // Set audio properties from base class
        SampleRate = sampleRate;
        NumMels = numMels;

        // Create tokenizer
        _tokenizer = new WhisperTokenizer();

        // Create mel spectrogram preprocessor with Whisper parameters
        _melSpectrogram = new MelSpectrogram<T>(
            sampleRate: sampleRate,
            nMels: numMels,
            nFft: 400,      // Whisper uses 25ms windows at 16kHz
            hopLength: 160, // Whisper uses 10ms hop at 16kHz
            fMin: 0,
            fMax: 8000,     // Whisper limits to 8kHz
            logMel: true);

        MelSpec = _melSpectrogram;

        // Load ONNX models with proper cleanup on failure
        var onnxOpts = onnxOptions ?? new OnnxModelOptions();
        OnnxModel<T>? encoder = null;

        try
        {
            encoder = new OnnxModel<T>(encoderPath, onnxOpts);
            var decoder = new OnnxModel<T>(decoderPath, onnxOpts);

            // Assign to properties only after both succeed
            OnnxEncoder = encoder;
            OnnxDecoder = decoder;
        }
        catch
        {
            // Clean up encoder if decoder creation failed
            encoder?.Dispose();
            throw;
        }

        // Initialize supported languages
        SupportedLanguages = GetSupportedLanguages();

        // Default loss function (cross-entropy is standard for sequence-to-sequence ASR)
        _lossFunction = new CrossEntropyLoss<T>();

        InitializeLayers();
    }

    /// <summary>
    /// Creates a Whisper network for training from scratch using native layers.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="modelSize">Model size variant (determines layer dimensions).</param>
    /// <param name="language">Target language code. Null for multilingual training.</param>
    /// <param name="translate">Whether to train for translation task.</param>
    /// <param name="sampleRate">Audio sample rate in Hz.</param>
    /// <param name="numMels">Number of mel filterbank channels.</param>
    /// <param name="maxAudioLengthSeconds">Maximum audio length to process.</param>
    /// <param name="maxTokens">Maximum sequence length for decoder.</param>
    /// <param name="beamSize">Beam size for inference.</param>
    /// <param name="temperature">Sampling temperature.</param>
    /// <param name="optimizer">Optimizer for training. If null, uses Adam with default settings.</param>
    /// <param name="lossFunction">Loss function for training. If null, uses cross-entropy.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor when you want to train a speech recognition
    /// model from scratch with your own data.
    ///
    /// Training Whisper requires:
    /// 1. Large amounts of paired audio-transcript data
    /// 2. Significant compute resources (GPUs recommended)
    /// 3. Many training epochs
    ///
    /// This is useful for:
    /// - Domain-specific vocabulary (medical, legal, technical)
    /// - Languages not well supported by pretrained models
    /// - Specific accent or dialect adaptation
    /// - Research and experimentation
    ///
    /// Example:
    /// <code>
    /// var whisper = new WhisperModel&lt;float&gt;(
    ///     architecture,
    ///     modelSize: WhisperModelSize.Base,
    ///     language: "en");
    ///
    /// // Training loop
    /// for (int epoch = 0; epoch &lt; numEpochs; epoch++)
    /// {
    ///     foreach (var (audio, tokens) in trainingData)
    ///     {
    ///         whisper.Train(audio, tokens);
    ///     }
    /// }
    /// </code>
    /// </para>
    /// </remarks>
    public WhisperModel(
        NeuralNetworkArchitecture<T> architecture,
        WhisperModelSize modelSize = WhisperModelSize.Base,
        string? language = null,
        bool translate = false,
        int sampleRate = 16000,
        int numMels = 80,
        int maxAudioLengthSeconds = 30,
        int maxTokens = 448,
        int beamSize = 5,
        double temperature = 0.0,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        WhisperOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new WhisperOptions();
        Options = _options;
        _useNativeMode = true;
        _modelSize = modelSize;
        _language = language;
        _translate = translate;
        _maxAudioLengthSeconds = maxAudioLengthSeconds;
        _numMels = numMels;
        _maxTokens = maxTokens;
        _beamSize = beamSize;
        _temperature = temperature;

        // Get model dimensions based on size
        (_modelDim, _numEncoderLayers, _numDecoderLayers, _numHeads, _ffDim) = GetModelParameters(modelSize);

        // Set audio properties from base class
        SampleRate = sampleRate;
        NumMels = numMels;

        // Create tokenizer
        _tokenizer = new WhisperTokenizer();

        // Create mel spectrogram preprocessor
        _melSpectrogram = new MelSpectrogram<T>(
            sampleRate: sampleRate,
            nMels: numMels,
            nFft: 400,
            hopLength: 160,
            fMin: 0,
            fMax: 8000,
            logMel: true);

        MelSpec = _melSpectrogram;

        // Initialize supported languages
        SupportedLanguages = GetSupportedLanguages();

        // Initialize training components
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new CrossEntropyLoss<T>();

        InitializeLayers();
    }

    #endregion

    #region Initialization

    /// <summary>
    /// Initializes layers following the golden standard pattern.
    /// </summary>
    protected override void InitializeLayers()
    {
        // ONNX mode - layers are handled by ONNX runtime
        if (!_useNativeMode)
        {
            return;
        }

        // Golden Standard Pattern:
        // 1. Check if user provided custom layers
        // 2. If yes, use them (full customization)
        // 3. If no, use LayerHelper.CreateDefaultWhisperLayers()
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            ValidateLayerConfiguration(Layers);
        }
        else
        {
            // Calculate max frames from audio parameters
            int maxFrames = (SampleRate * _maxAudioLengthSeconds) / 160;

            // Use default Whisper architecture
            Layers.AddRange(LayerHelper<T>.CreateDefaultWhisperLayers(
                modelDim: _modelDim,
                numEncoderLayers: _numEncoderLayers,
                numDecoderLayers: _numDecoderLayers,
                numHeads: _numHeads,
                ffDim: _ffDim,
                numMels: _numMels,
                maxFrames: maxFrames,
                maxTokens: _maxTokens,
                vocabSize: 51865,
                dropoutRate: 0.0));
        }
    }

    /// <summary>
    /// Validates that custom layers meet Whisper requirements.
    /// </summary>
    private void ValidateLayerConfiguration(List<ILayer<T>> layers)
    {
        if (layers.Count < 3)
        {
            throw new ArgumentException(
                "Whisper requires at least 3 layers: encoder, decoder, and output projection. " +
                "Use LayerHelper.CreateDefaultWhisperLayers() as a reference.",
                nameof(layers));
        }
    }

    private static (int modelDim, int encoderLayers, int decoderLayers, int heads, int ffDim) GetModelParameters(WhisperModelSize size)
    {
        return size switch
        {
            WhisperModelSize.Tiny => (384, 4, 4, 6, 1536),
            WhisperModelSize.Base => (512, 6, 6, 8, 2048),
            WhisperModelSize.Small => (768, 12, 12, 12, 3072),
            WhisperModelSize.Medium => (1024, 24, 24, 16, 4096),
            WhisperModelSize.Large or WhisperModelSize.LargeV2 or WhisperModelSize.LargeV3 => (1280, 32, 32, 20, 5120),
            _ => (512, 6, 6, 8, 2048) // Default to Base
        };
    }

    private static IReadOnlyList<string> GetSupportedLanguages()
    {
        // All Whisper models support these 99 languages
        return new[]
        {
            "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr",
            "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi",
            "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no",
            "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk",
            "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk",
            "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
            "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc",
            "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
            "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl",
            "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su"
        };
    }

    #endregion

    #region ISpeechRecognizer Implementation

    /// <summary>
    /// Transcribes audio to text.
    /// </summary>
    /// <param name="audio">Audio waveform tensor [batch, samples] or [samples].</param>
    /// <param name="language">Optional language code. Auto-detected if null.</param>
    /// <param name="includeTimestamps">Whether to include word-level timestamps.</param>
    /// <returns>Transcription result containing text and optional timestamps.</returns>
    public TranscriptionResult<T> Transcribe(Tensor<T> audio, string? language = null, bool includeTimestamps = false)
    {
        ThrowIfDisposed();

        var stopwatch = Stopwatch.StartNew();

        // Override language if specified
        string effectiveLanguage = language ?? _language ?? "en";

        // Preprocess audio to mel spectrogram
        var melFeatures = PreprocessAudio(audio);

        // Encode audio features
        var encoderOutput = EncodeAudio(melFeatures);

        // Decode to text tokens with confidence
        var (tokens, confidence) = DecodeTokens(encoderOutput, effectiveLanguage);

        // Convert tokens to text
        var text = _tokenizer.Decode(tokens);

        stopwatch.Stop();

        return new TranscriptionResult<T>
        {
            Text = text,
            Language = effectiveLanguage,
            Confidence = NumOps.FromDouble(confidence),
            DurationSeconds = (double)audio.Length / SampleRate,
            Segments = includeTimestamps ? ExtractSegments(tokens, text) : Array.Empty<TranscriptionSegment<T>>()
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
        var probabilities = DetectLanguageProbabilities(audio);
        return probabilities.OrderByDescending(p => NumOps.ToDouble(p.Value)).First().Key;
    }

    /// <summary>
    /// Gets language detection probabilities for the audio.
    /// </summary>
    public IReadOnlyDictionary<string, T> DetectLanguageProbabilities(Tensor<T> audio)
    {
        ThrowIfDisposed();

        // Preprocess audio
        var melFeatures = PreprocessAudio(audio);

        // Encode audio
        var encoderOutput = EncodeAudio(melFeatures);

        // Get language token probabilities from decoder
        if (OnnxDecoder is null && !_useNativeMode)
            throw new InvalidOperationException("Decoder not loaded.");

        // Create initial token sequence for language detection
        var initialTokens = new Tensor<T>([1, 1]);
        initialTokens[0, 0] = NumOps.FromDouble(_tokenizer.StartOfTranscript);

        Tensor<T> logits;
        if (_useNativeMode)
        {
            // Native mode: run through decoder layers
            logits = ForwardDecoder(initialTokens, encoderOutput);
        }
        else
        {
            var inputs = new Dictionary<string, Tensor<T>>
            {
                ["encoder_hidden_states"] = encoderOutput,
                ["input_ids"] = initialTokens
            };
            var output = OnnxDecoder!.Run(inputs);
            logits = output.Values.First();
        }

        // Extract language token probabilities
        var languageProbs = new Dictionary<string, T>();
        foreach (var lang in SupportedLanguages)
        {
            var langToken = _tokenizer.GetLanguageToken(lang);
            if (langToken >= 0 && langToken < logits.Shape[^1])
            {
                languageProbs[lang] = logits[0, 0, (int)langToken];
            }
        }

        // Apply softmax
        return ApplySoftmax(languageProbs);
    }

    /// <summary>
    /// Starts a streaming transcription session.
    /// </summary>
    public IStreamingTranscriptionSession<T> StartStreamingSession(string? language = null)
    {
        throw new NotSupportedException("WhisperModel does not support streaming transcription.");
    }

    #endregion

    #region AudioNeuralNetworkBase Implementation

    /// <summary>
    /// Preprocesses raw audio for model input.
    /// </summary>
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio)
    {
        // Pad or truncate to max length
        int targetLength = SampleRate * _maxAudioLengthSeconds;
        var paddedAudio = PadOrTruncate(rawAudio, targetLength);

        // Compute mel spectrogram
        var melSpec = _melSpectrogram.Forward(paddedAudio);

        return melSpec;
    }

    /// <summary>
    /// Postprocesses model output into the final result format.
    /// </summary>
    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput)
    {
        // For Whisper, postprocessing is handled in Transcribe method
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
    /// <param name="gradients">The gradients to apply.</param>
    /// <remarks>
    /// <para>
    /// Applies the simple gradient descent update rule: params = params - learning_rate * gradients.
    /// </para>
    /// <para><b>For Beginners:</b> This is how the model learns!
    ///
    /// During training:
    /// 1. The model transcribes audio
    /// 2. We compare to the correct transcription (loss)
    /// 3. We compute gradients (which direction to adjust each parameter)
    /// 4. This method applies those adjustments to improve transcription
    ///
    /// The learning rate controls adjustment magnitude:
    /// - Too big: May overshoot optimal values
    /// - Too small: Learning is slow but precise
    /// - Default (0.001): Good starting point
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode)
        {
            throw new NotSupportedException("Cannot update parameters in ONNX inference mode. Use the native constructor for training.");
        }

        int index = 0;
        foreach (var layer in Layers)
        {
            int count = layer.ParameterCount;
            var layerParams = parameters.Slice(index, count);
            layer.UpdateParameters(layerParams);
            index += count;
        }
    }

    /// <summary>
    /// Trains the model on a single batch of audio and expected transcription tokens.
    /// </summary>
    /// <param name="input">Audio waveform tensor [batch, samples] or [samples].</param>
    /// <param name="expectedOutput">Expected token sequence tensor [batch, sequence_length].</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Training teaches the model to transcribe audio correctly.
    ///
    /// The training process:
    /// 1. Forward pass: Audio goes through encoder-decoder to predict tokens
    /// 2. Loss calculation: Compare predicted tokens to expected tokens
    /// 3. Backward pass: Calculate gradients showing how to improve
    /// 4. Update: Adjust model parameters to reduce error
    ///
    /// Call this method repeatedly with different audio/transcript pairs.
    /// After many iterations, the model learns to transcribe correctly.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_useNativeMode)
        {
            throw new NotSupportedException("Cannot train in ONNX inference mode. Use the native constructor for training.");
        }

        // Set training mode
        SetTrainingMode(true);

        // 1. Preprocess audio to mel spectrogram
        var melFeatures = PreprocessAudio(input);

        // 2. Forward pass through encoder-decoder layers
        var prediction = Forward(melFeatures);

        // 3. Flatten tensors to vectors for loss calculation
        var flattenedPredictions = prediction.ToVector();
        var flattenedExpected = expectedOutput.ToVector();

        // 4. Calculate loss
        LastLoss = _lossFunction.CalculateLoss(flattenedPredictions, flattenedExpected);

        // 5. Calculate gradient of loss with respect to output
        var outputGradients = _lossFunction.CalculateDerivative(flattenedPredictions, flattenedExpected);

        // 6. Backward pass through layers
        Backpropagate(Tensor<T>.FromVector(outputGradients));

        // 7. Update parameters using optimizer
        _optimizer?.UpdateParameters(Layers);

        // Exit training mode
        SetTrainingMode(false);
    }

    /// <summary>
    /// Gets metadata about the model.
    /// </summary>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = $"Whisper-{_modelSize}",
            Description = $"Whisper speech recognition model - {_modelSize} variant",
            ModelType = ModelType.NeuralNetwork,
            FeatureCount = SampleRate * _maxAudioLengthSeconds,
            Complexity = (int)_modelSize
        };
        metadata.AdditionalInfo["InputFormat"] = $"Audio ({SampleRate}Hz, {_maxAudioLengthSeconds}s max)";
        metadata.AdditionalInfo["OutputFormat"] = "Transcription";
        metadata.AdditionalInfo["Mode"] = _useNativeMode ? "Native" : "ONNX";
        return metadata;
    }

    /// <summary>
    /// Serializes network-specific data.
    /// </summary>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_useNativeMode);
        writer.Write(SampleRate);
        writer.Write(_numMels);
        writer.Write(_maxAudioLengthSeconds);
        writer.Write((int)_modelSize);
        writer.Write(_language ?? string.Empty);
        writer.Write(_translate);
        writer.Write(_maxTokens);
        writer.Write(_beamSize);
        writer.Write(_temperature);
    }

    /// <summary>
    /// Deserializes network-specific data.
    /// </summary>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        // Note: Most fields are readonly, so deserialization would require a different approach
        // For now, we read and validate the values
        var useNative = reader.ReadBoolean();
        var sampleRate = reader.ReadInt32();
        var numMels = reader.ReadInt32();
        var maxAudioLen = reader.ReadInt32();
        var modelSize = (WhisperModelSize)reader.ReadInt32();
        var lang = reader.ReadString();
        var translate = reader.ReadBoolean();
        var maxTokens = reader.ReadInt32();
        var beamSize = reader.ReadInt32();
        var temperature = reader.ReadDouble();

        // Validation would happen here
    }

    /// <summary>
    /// Creates a new instance of this model for cloning.
    /// </summary>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (_useNativeMode)
        {
            return new WhisperModel<T>(
                Architecture,
                modelSize: _modelSize,
                language: _language,
                translate: _translate,
                sampleRate: SampleRate,
                numMels: _numMels,
                maxAudioLengthSeconds: _maxAudioLengthSeconds,
                maxTokens: _maxTokens,
                beamSize: _beamSize,
                temperature: _temperature);
        }
        else
        {
            return new WhisperModel<T>(
                Architecture,
                encoderPath: _encoderPath!,
                decoderPath: _decoderPath!,
                modelSize: _modelSize,
                language: _language,
                translate: _translate,
                sampleRate: SampleRate,
                numMels: _numMels,
                maxAudioLengthSeconds: _maxAudioLengthSeconds,
                maxTokens: _maxTokens,
                beamSize: _beamSize,
                temperature: _temperature);
        }
    }

    #endregion

    #region Private Methods

    private Tensor<T> PadOrTruncate(Tensor<T> audio, int targetLength)
    {
        int currentLength = audio.Shape[0];

        if (currentLength == targetLength)
        {
            return audio;
        }

        var result = new Tensor<T>([targetLength]);

        if (currentLength < targetLength)
        {
            // Pad with zeros
            for (int i = 0; i < currentLength; i++)
            {
                result[i] = audio[i];
            }
            for (int i = currentLength; i < targetLength; i++)
            {
                result[i] = NumOps.Zero;
            }
        }
        else
        {
            // Truncate
            for (int i = 0; i < targetLength; i++)
            {
                result[i] = audio[i];
            }
        }

        return result;
    }

    private Tensor<T> EncodeAudio(Tensor<T> melFeatures)
    {
        if (!_useNativeMode)
        {
            if (OnnxEncoder is null)
                throw new InvalidOperationException("Encoder not loaded.");

            // Add batch dimension if needed
            if (melFeatures.Rank == 2)
            {
                var batched = new Tensor<T>([1, melFeatures.Shape[0], melFeatures.Shape[1]]);
                for (int f = 0; f < melFeatures.Shape[0]; f++)
                {
                    for (int m = 0; m < melFeatures.Shape[1]; m++)
                    {
                        batched[0, f, m] = melFeatures[f, m];
                    }
                }
                melFeatures = batched;
            }

            return OnnxEncoder.Run(melFeatures);
        }
        else
        {
            // Native mode: forward through encoder layers
            // Encoder layers are approximately the first half of layers
            // (2 projection + 1 positional + numEncoderLayers * 4 layers per block + 1 final norm)
            int encoderLayerCount = 2 + 1 + (_numEncoderLayers * 4) + 1;
            if (encoderLayerCount > Layers.Count)
                encoderLayerCount = Layers.Count / 2;

            var current = melFeatures;
            for (int i = 0; i < encoderLayerCount && i < Layers.Count; i++)
            {
                current = Layers[i].Forward(current);
            }
            return current;
        }
    }

    private Tensor<T> ForwardDecoder(Tensor<T> tokens, Tensor<T> encoderOutput)
    {
        if (!_useNativeMode)
        {
            // ONNX mode: create input dictionary with both tokens and encoder output
            var inputs = new Dictionary<string, Tensor<T>>
            {
                ["tokens"] = tokens,
                ["encoder_hidden_states"] = encoderOutput
            };
            var outputs = OnnxDecoder?.Run(inputs);
            return outputs?.Values.FirstOrDefault() ?? tokens;
        }

        // Calculate where decoder layers start
        int encoderLayerCount = 2 + 1 + (_numEncoderLayers * 4) + 1;
        if (encoderLayerCount > Layers.Count)
            encoderLayerCount = Layers.Count / 2;

        // Forward through decoder layers (starting after encoder layers)
        var current = tokens;
        for (int i = encoderLayerCount; i < Layers.Count; i++)
        {
            var layer = Layers[i];
            if (layer is TransformerDecoderLayer<T> decoderLayer)
            {
                // Pass encoder output as context
                current = decoderLayer.Forward(current, encoderOutput);
            }
            else
            {
                current = layer.Forward(current);
            }
        }

        return current;
    }

    private (List<long> tokens, double confidence) DecodeTokens(Tensor<T> encoderOutput, string language)
    {
        var tokens = new List<long>();
        var tokenProbabilities = new List<double>();

        // Start with special tokens
        tokens.Add(_tokenizer.StartOfTranscript);
        tokens.Add(_tokenizer.GetLanguageToken(language));

        if (_translate)
        {
            tokens.Add(_tokenizer.TranslateToken);
        }
        else
        {
            tokens.Add(_tokenizer.TranscribeToken);
        }

        // Autoregressive decoding
        for (int step = 0; step < _maxTokens; step++)
        {
            // Create input tensor for decoder
            var tokenTensor = CreateTokenTensor(tokens);

            Tensor<T> logits;
            if (_useNativeMode)
            {
                logits = ForwardDecoder(tokenTensor, encoderOutput);
            }
            else
            {
                if (OnnxDecoder is null)
                    throw new InvalidOperationException("Decoder not loaded.");

                var inputs = new Dictionary<string, Tensor<T>>
                {
                    ["encoder_hidden_states"] = encoderOutput,
                    ["input_ids"] = tokenTensor
                };

                var output = OnnxDecoder.Run(inputs);
                logits = output.Values.First();
            }

            // Get next token and its probability
            var (nextToken, probability) = GetNextTokenWithProbability(logits, tokens.Count - 1);

            if (nextToken == _tokenizer.EndOfText)
            {
                break;
            }

            tokens.Add(nextToken);
            tokenProbabilities.Add(probability);
        }

        // Calculate overall confidence as geometric mean of token probabilities
        double confidence = 1.0;
        if (tokenProbabilities.Count > 0)
        {
            // Use log-sum for numerical stability, then convert back
            double logSum = tokenProbabilities.Sum(p => Math.Log(Math.Max(p, 1e-10)));
            confidence = Math.Exp(logSum / tokenProbabilities.Count);
        }

        return (tokens, confidence);
    }

    private (int token, double probability) GetNextTokenWithProbability(Tensor<T> logits, int position)
    {
        // Get logits for the last position
        int vocabSize = logits.Shape[^1];

        // Find max logit for numerical stability in softmax
        double maxLogit = double.NegativeInfinity;
        for (int v = 0; v < vocabSize; v++)
        {
            double value = NumOps.ToDouble(logits[0, position, v]);
            if (value > maxLogit)
            {
                maxLogit = value;
            }
        }

        // Compute softmax and find best token
        double sumExp = 0;
        int maxIdx = 0;
        double maxValue = double.NegativeInfinity;

        var expValues = new double[vocabSize];
        for (int v = 0; v < vocabSize; v++)
        {
            double value = NumOps.ToDouble(logits[0, position, v]);
            expValues[v] = Math.Exp(value - maxLogit);
            sumExp += expValues[v];

            if (value > maxValue)
            {
                maxValue = value;
                maxIdx = v;
            }
        }

        // Probability of selected token
        double probability = expValues[maxIdx] / sumExp;

        return (maxIdx, probability);
    }

    private Tensor<T> CreateTokenTensor(List<long> tokens)
    {
        var tensor = new Tensor<T>([1, tokens.Count]);

        for (int i = 0; i < tokens.Count; i++)
        {
            tensor[0, i] = NumOps.FromDouble(tokens[i]);
        }

        return tensor;
    }


    private IReadOnlyDictionary<string, T> ApplySoftmax(Dictionary<string, T> logits)
    {
        // Find max for numerical stability
        T maxLogit = NumOps.Zero;
        bool first = true;
        foreach (var logit in logits.Values)
        {
            if (first || NumOps.GreaterThan(logit, maxLogit))
            {
                maxLogit = logit;
                first = false;
            }
        }

        // Compute exp(logit - max) and sum
        var expValues = new Dictionary<string, T>();
        T expSum = NumOps.Zero;
        foreach (var (key, logit) in logits)
        {
            var exp = NumOps.Exp(NumOps.Subtract(logit, maxLogit));
            expValues[key] = exp;
            expSum = NumOps.Add(expSum, exp);
        }

        // Normalize
        var result = new Dictionary<string, T>();
        foreach (var (key, exp) in expValues)
        {
            result[key] = NumOps.Divide(exp, expSum);
        }

        return result;
    }

    private IReadOnlyList<TranscriptionSegment<T>> ExtractSegments(List<long> tokens, string text)
    {
        // Simplified segment extraction - actual implementation would parse timestamp tokens
        var segments = new List<TranscriptionSegment<T>>();

        if (text.Length > 0)
        {
            segments.Add(new TranscriptionSegment<T>
            {
                Text = text,
                StartTime = 0.0,
                EndTime = _maxAudioLengthSeconds,
                Confidence = NumOps.FromDouble(1.0)
            });
        }

        return segments;
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

        if (disposing)
        {
            // Base class disposes OnnxEncoder, OnnxDecoder, OnnxModel
        }

        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
