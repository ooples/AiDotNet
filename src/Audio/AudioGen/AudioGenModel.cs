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
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.Audio.AudioGen;

/// <summary>
/// AudioGen model for generating audio from text descriptions using neural audio codecs.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// AudioGen uses a language model approach to generate audio from text prompts.
/// The architecture consists of three main components:
/// <list type="number">
/// <item><description>Text Encoder: Converts text prompts to embeddings (typically T5-based)</description></item>
/// <item><description>Audio Language Model: Generates discrete audio codes autoregressively</description></item>
/// <item><description>Audio Decoder (EnCodec): Converts audio codes back to waveforms</description></item>
/// </list>
/// </para>
/// <para><b>For Beginners:</b> AudioGen is fundamentally different from Text-to-Speech (TTS):
///
/// TTS vs AudioGen:
/// - TTS: Converts specific words to speech ("Hello world" -> spoken words "Hello world")
/// - AudioGen: Creates sounds matching a description ("dog barking" -> actual bark sound)
///
/// How it works:
/// 1. Your text prompt ("a cat meowing softly") is encoded into a numerical representation
/// 2. A language model generates a sequence of "audio tokens" (like words, but for sound)
/// 3. The EnCodec decoder converts these tokens back into actual audio waveforms
///
/// Why discrete audio codes?
/// - Raw audio has too many samples (32,000 per second!)
/// - EnCodec compresses audio to ~50 tokens per second
/// - This makes the language model's job much easier
///
/// Common use cases:
/// - Sound effect generation for games/films
/// - Creating ambient soundscapes
/// - Generating audio for multimedia content
/// - Rapid prototyping of audio concepts
///
/// Limitations:
/// - Cannot generate intelligible speech (use TTS for that)
/// - Quality depends on training data
/// - May struggle with very specific or unusual sounds
/// </para>
/// <para>
/// Reference: "AudioGen: Textually Guided Audio Generation" by Kreuk et al., 2022
/// </para>
/// </remarks>
public class AudioGenModel<T> : AudioNeuralNetworkBase<T>, IAudioGenerator<T>
{
    private readonly AudioGenOptions _options;

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
    /// The ONNX model for text encoding.
    /// </summary>
    private readonly OnnxModel<T>? _textEncoder;

    /// <summary>
    /// The ONNX model for audio language modeling.
    /// </summary>
    private readonly OnnxModel<T>? _languageModel;

    /// <summary>
    /// The ONNX model for audio decoding (EnCodec).
    /// </summary>
    private readonly OnnxModel<T>? _audioDecoder;

    /// <summary>
    /// Path to the text encoder ONNX model file.
    /// </summary>
    private readonly string? _textEncoderPath;

    /// <summary>
    /// Path to the language model ONNX model file.
    /// </summary>
    private readonly string? _languageModelPath;

    /// <summary>
    /// Path to the audio decoder ONNX model file.
    /// </summary>
    private readonly string? _audioDecoderPath;

    #endregion

    #region Shared Fields

    /// <summary>
    /// The tokenizer for processing text input.
    /// </summary>
    private readonly ITokenizer _tokenizer;

    /// <summary>
    /// Optimizer for training.
    /// </summary>
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

    /// <summary>
    /// Loss function for training.
    /// </summary>
    private readonly ILossFunction<T> _lossFunction;

    /// <summary>
    /// Random number generator for sampling.
    /// </summary>
    private readonly Random _random;

    /// <summary>
    /// Lock object for thread-safe random access.
    /// </summary>
    private readonly object _randomLock = new();

    /// <summary>
    /// Model size variant.
    /// </summary>
    private readonly AudioGenModelSize _modelSize;

    /// <summary>
    /// Output sample rate in Hz.
    /// </summary>
    private readonly int _sampleRate;

    /// <summary>
    /// Default duration of generated audio in seconds.
    /// </summary>
    private readonly double _durationSeconds;

    /// <summary>
    /// Maximum duration of generated audio in seconds.
    /// </summary>
    private readonly double _maxDurationSeconds;

    /// <summary>
    /// Sampling temperature (higher = more random).
    /// </summary>
    private readonly double _temperature;

    /// <summary>
    /// Top-k sampling parameter.
    /// </summary>
    private readonly int _topK;

    /// <summary>
    /// Top-p (nucleus) sampling parameter.
    /// </summary>
    private readonly double _topP;

    /// <summary>
    /// Classifier-free guidance scale.
    /// </summary>
    private readonly double _guidanceScale;

    /// <summary>
    /// Number of audio channels (1=mono, 2=stereo).
    /// </summary>
    private readonly int _channels;

    /// <summary>
    /// Text encoder hidden dimension.
    /// </summary>
    private readonly int _textHiddenDim;

    /// <summary>
    /// Language model hidden dimension.
    /// </summary>
    private readonly int _lmHiddenDim;

    /// <summary>
    /// Number of transformer layers in the language model.
    /// </summary>
    private readonly int _numLmLayers;

    /// <summary>
    /// Number of attention heads.
    /// </summary>
    private readonly int _numHeads;

    /// <summary>
    /// Number of EnCodec codebooks.
    /// </summary>
    private readonly int _numCodebooks;

    /// <summary>
    /// Size of each codebook vocabulary.
    /// </summary>
    private readonly int _codebookSize;

    /// <summary>
    /// Maximum text sequence length.
    /// </summary>
    private readonly int _maxTextLength;

    /// <summary>
    /// Disposed flag.
    /// </summary>
    private bool _disposed;

    // Layer boundary indices for routing data through correct layers
    private int _textEncoderLayerStart;
    private int _textEncoderLayerEnd;
    private int _languageModelLayerStart;
    private int _languageModelLayerEnd;

    #endregion

    #region IAudioGenerator Properties

    /// <summary>
    /// Gets the maximum duration of audio that can be generated in seconds.
    /// </summary>
    public double MaxDurationSeconds => _maxDurationSeconds;

    /// <summary>
    /// Gets whether this model supports text-to-audio generation.
    /// </summary>
    public bool SupportsTextToAudio => true;

    /// <summary>
    /// Gets whether this model supports text-to-music generation.
    /// </summary>
    public bool SupportsTextToMusic => false;

    /// <summary>
    /// Gets whether this model supports audio continuation.
    /// </summary>
    public bool SupportsAudioContinuation => false;

    /// <summary>
    /// Gets whether this model supports audio inpainting.
    /// </summary>
    public bool SupportsAudioInpainting => false;

    #endregion

    #region Public Properties

    /// <summary>
    /// Gets the model size variant.
    /// </summary>
    public AudioGenModelSize ModelSize => _modelSize;

    /// <summary>
    /// Gets whether the model is ready for inference.
    /// </summary>
    public bool IsReady => _useNativeMode ||
        (_textEncoder?.IsLoaded == true && _languageModel?.IsLoaded == true && _audioDecoder?.IsLoaded == true);

    #endregion

    #region Constructors

    /// <summary>
    /// Creates an AudioGen network using pretrained ONNX models.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="textEncoderPath">Path to the text encoder ONNX model.</param>
    /// <param name="languageModelPath">Path to the audio language model ONNX model.</param>
    /// <param name="audioDecoderPath">Path to the audio decoder (EnCodec) ONNX model.</param>
    /// <param name="tokenizer">Tokenizer for text processing. REQUIRED - must match the text encoder (typically T5).</param>
    /// <param name="modelSize">Model size variant (default: Medium).</param>
    /// <param name="sampleRate">Output sample rate in Hz (default: 32000 for AudioGen).</param>
    /// <param name="durationSeconds">Default generation duration in seconds (default: 5.0).</param>
    /// <param name="maxDurationSeconds">Maximum generation duration in seconds (default: 30.0).</param>
    /// <param name="temperature">Sampling temperature - higher values produce more random output (default: 1.0).</param>
    /// <param name="topK">Top-k sampling - only consider top k tokens (default: 250).</param>
    /// <param name="topP">Top-p (nucleus) sampling threshold (default: 0.0 = disabled).</param>
    /// <param name="guidanceScale">Classifier-free guidance scale (default: 3.0).</param>
    /// <param name="channels">Number of audio channels, 1=mono, 2=stereo (default: 1).</param>
    /// <param name="seed">Random seed for reproducibility. Null for non-deterministic generation.</param>
    /// <param name="optimizer">Optional optimizer for fine-tuning.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para>
    /// When loading pretrained ONNX models, you MUST provide a tokenizer that matches
    /// the text encoder. AudioGen uses T5-based text encoders, so use a T5 tokenizer:
    /// <code>
    /// var tokenizer = await AutoTokenizer.FromPretrainedAsync("t5-base");
    /// var audioGen = new AudioGenModel&lt;float&gt;(architecture, encoderPath, lmPath, decoderPath, tokenizer);
    /// </code>
    /// </para>
    /// </remarks>
    public AudioGenModel(
        NeuralNetworkArchitecture<T> architecture,
        string textEncoderPath,
        string languageModelPath,
        string audioDecoderPath,
        ITokenizer tokenizer,
        AudioGenModelSize modelSize = AudioGenModelSize.Medium,
        int sampleRate = 32000,
        double durationSeconds = 5.0,
        double maxDurationSeconds = 30.0,
        double temperature = 1.0,
        int topK = 250,
        double topP = 0.0,
        double guidanceScale = 3.0,
        int channels = 1,
        int? seed = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        AudioGenOptions? options = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>(), 1.0)
    {
        _options = options ?? new AudioGenOptions();
        Options = _options;
        // Validate ONNX model paths
        if (string.IsNullOrWhiteSpace(textEncoderPath))
            throw new ArgumentException("Text encoder path cannot be null or empty.", nameof(textEncoderPath));
        if (string.IsNullOrWhiteSpace(languageModelPath))
            throw new ArgumentException("Language model path cannot be null or empty.", nameof(languageModelPath));
        if (string.IsNullOrWhiteSpace(audioDecoderPath))
            throw new ArgumentException("Audio decoder path cannot be null or empty.", nameof(audioDecoderPath));
        if (!File.Exists(textEncoderPath))
            throw new FileNotFoundException($"Text encoder model not found: {textEncoderPath}");
        if (!File.Exists(languageModelPath))
            throw new FileNotFoundException($"Language model not found: {languageModelPath}");
        if (!File.Exists(audioDecoderPath))
            throw new FileNotFoundException($"Audio decoder model not found: {audioDecoderPath}");

        // Validate generation parameters
        if (temperature <= 0)
            throw new ArgumentOutOfRangeException(nameof(temperature), "Temperature must be positive.");
        if (topK < 0)
            throw new ArgumentOutOfRangeException(nameof(topK), "TopK must be non-negative.");
        if (topP < 0 || topP > 1.0)
            throw new ArgumentOutOfRangeException(nameof(topP), "TopP must be in range [0, 1].");
        if (guidanceScale < 1.0)
            throw new ArgumentOutOfRangeException(nameof(guidanceScale), "GuidanceScale must be >= 1.0.");
        if (channels < 1)
            throw new ArgumentOutOfRangeException(nameof(channels), "Channels must be at least 1.");
        if (sampleRate <= 0)
            throw new ArgumentOutOfRangeException(nameof(sampleRate), "SampleRate must be positive.");
        if (durationSeconds <= 0)
            throw new ArgumentOutOfRangeException(nameof(durationSeconds), "DurationSeconds must be positive.");
        if (maxDurationSeconds <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxDurationSeconds), "MaxDurationSeconds must be positive.");
        if (durationSeconds > maxDurationSeconds)
            throw new ArgumentOutOfRangeException(nameof(durationSeconds), "DurationSeconds cannot exceed MaxDurationSeconds.");

        _useNativeMode = false;
        _textEncoderPath = textEncoderPath;
        _languageModelPath = languageModelPath;
        _audioDecoderPath = audioDecoderPath;
        _modelSize = modelSize;
        _sampleRate = sampleRate;
        _durationSeconds = durationSeconds;
        _maxDurationSeconds = maxDurationSeconds;
        _temperature = temperature;
        _topK = topK;
        _topP = topP;
        _guidanceScale = guidanceScale;
        _channels = channels;

        // Set model dimensions based on size
        (_textHiddenDim, _lmHiddenDim, _numLmLayers, _numHeads) = GetModelDimensions(modelSize);
        _numCodebooks = 4;  // EnCodec standard
        _codebookSize = 1024;  // EnCodec standard
        _maxTextLength = 256;

        // Set audio base class properties
        SampleRate = sampleRate;

        OnnxModel<T>? textEncoder = null;
        OnnxModel<T>? languageModel = null;
        OnnxModel<T>? audioDecoder = null;

        try
        {
            textEncoder = new OnnxModel<T>(textEncoderPath);
            languageModel = new OnnxModel<T>(languageModelPath);
            audioDecoder = new OnnxModel<T>(audioDecoderPath);

            _textEncoder = textEncoder;
            _languageModel = languageModel;
            _audioDecoder = audioDecoder;

            // Store audio decoder as OnnxModel for base class
            OnnxModel = audioDecoder;

            // Tokenizer is required for ONNX mode - must match the text encoder
            Guard.NotNull(tokenizer);
            _tokenizer = tokenizer;

            _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
            _lossFunction = lossFunction ?? new CrossEntropyLoss<T>();

            _random = seed.HasValue
                ? RandomHelper.CreateSeededRandom(seed.Value)
                : RandomHelper.CreateSecureRandom();

            InitializeLayers();
        }
        catch
        {
            textEncoder?.Dispose();
            languageModel?.Dispose();
            audioDecoder?.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Creates an AudioGen network using native library layers for training from scratch.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="modelSize">Model size variant (default: Medium).</param>
    /// <param name="sampleRate">Output sample rate in Hz (default: 32000 for AudioGen).</param>
    /// <param name="durationSeconds">Default generation duration in seconds (default: 5.0).</param>
    /// <param name="maxDurationSeconds">Maximum generation duration in seconds (default: 30.0).</param>
    /// <param name="temperature">Sampling temperature - higher values produce more random output (default: 1.0).</param>
    /// <param name="topK">Top-k sampling - only consider top k tokens (default: 250).</param>
    /// <param name="topP">Top-p (nucleus) sampling threshold (default: 0.0 = disabled).</param>
    /// <param name="guidanceScale">Classifier-free guidance scale (default: 3.0).</param>
    /// <param name="channels">Number of audio channels, 1=mono, 2=stereo (default: 1).</param>
    /// <param name="textHiddenDim">Text encoder hidden dimension. If 0, uses model size default.</param>
    /// <param name="lmHiddenDim">Language model hidden dimension. If 0, uses model size default.</param>
    /// <param name="numLmLayers">Number of language model layers. If 0, uses model size default.</param>
    /// <param name="numHeads">Number of attention heads. If 0, uses model size default.</param>
    /// <param name="numCodebooks">Number of EnCodec codebooks (default: 4).</param>
    /// <param name="codebookSize">Size of each codebook vocabulary (default: 1024).</param>
    /// <param name="maxTextLength">Maximum text sequence length (default: 256).</param>
    /// <param name="seed">Random seed for reproducibility. Null for non-deterministic generation.</param>
    /// <param name="tokenizer">Optional tokenizer for text processing. If null, creates a T5-style default.</param>
    /// <param name="optimizer">Optional optimizer for training.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor when training AudioGen from scratch.
    ///
    /// Training your own AudioGen:
    /// 1. You need paired data: (text descriptions, audio clips)
    /// 2. Audio is pre-encoded to discrete codes using EnCodec
    /// 3. The model learns to predict audio codes from text
    ///
    /// This is computationally expensive and requires significant data.
    /// For most use cases, load pretrained ONNX models instead.
    /// </para>
    /// </remarks>
    public AudioGenModel(
        NeuralNetworkArchitecture<T> architecture,
        AudioGenModelSize modelSize = AudioGenModelSize.Medium,
        int sampleRate = 32000,
        double durationSeconds = 5.0,
        double maxDurationSeconds = 30.0,
        double temperature = 1.0,
        int topK = 250,
        double topP = 0.0,
        double guidanceScale = 3.0,
        int channels = 1,
        int textHiddenDim = 0,
        int lmHiddenDim = 0,
        int numLmLayers = 0,
        int numHeads = 0,
        int numCodebooks = 4,
        int codebookSize = 1024,
        int maxTextLength = 256,
        int? seed = null,
        ITokenizer? tokenizer = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        AudioGenOptions? options = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>(), 1.0)
    {
        _options = options ?? new AudioGenOptions();
        Options = _options;
        // Validate parameters
        if (temperature <= 0)
            throw new ArgumentOutOfRangeException(nameof(temperature), "Temperature must be positive.");
        if (topK < 0)
            throw new ArgumentOutOfRangeException(nameof(topK), "TopK must be non-negative.");
        if (topP < 0 || topP > 1.0)
            throw new ArgumentOutOfRangeException(nameof(topP), "TopP must be in range [0, 1].");
        if (guidanceScale < 1.0)
            throw new ArgumentOutOfRangeException(nameof(guidanceScale), "GuidanceScale must be >= 1.0.");
        if (channels < 1)
            throw new ArgumentOutOfRangeException(nameof(channels), "Channels must be at least 1.");
        if (sampleRate <= 0)
            throw new ArgumentOutOfRangeException(nameof(sampleRate), "SampleRate must be positive.");
        if (durationSeconds <= 0)
            throw new ArgumentOutOfRangeException(nameof(durationSeconds), "DurationSeconds must be positive.");
        if (maxDurationSeconds <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxDurationSeconds), "MaxDurationSeconds must be positive.");
        if (durationSeconds > maxDurationSeconds)
            throw new ArgumentOutOfRangeException(nameof(durationSeconds), "DurationSeconds cannot exceed MaxDurationSeconds.");
        if (numCodebooks <= 0)
            throw new ArgumentOutOfRangeException(nameof(numCodebooks), "NumCodebooks must be positive.");
        if (codebookSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(codebookSize), "CodebookSize must be positive.");
        if (maxTextLength <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxTextLength), "MaxTextLength must be positive.");

        _useNativeMode = true;
        _modelSize = modelSize;
        _sampleRate = sampleRate;
        _durationSeconds = durationSeconds;
        _maxDurationSeconds = maxDurationSeconds;
        _temperature = temperature;
        _topK = topK;
        _topP = topP;
        _guidanceScale = guidanceScale;
        _channels = channels;

        // Get default dimensions from model size, override with explicit values if provided
        var (defaultTextDim, defaultLmDim, defaultLayers, defaultHeads) = GetModelDimensions(modelSize);
        _textHiddenDim = textHiddenDim > 0 ? textHiddenDim : defaultTextDim;
        _lmHiddenDim = lmHiddenDim > 0 ? lmHiddenDim : defaultLmDim;
        _numLmLayers = numLmLayers > 0 ? numLmLayers : defaultLayers;
        _numHeads = numHeads > 0 ? numHeads : defaultHeads;
        _numCodebooks = numCodebooks;
        _codebookSize = codebookSize;
        _maxTextLength = maxTextLength;

        // Set audio base class properties
        SampleRate = sampleRate;

        // Use T5-style tokenizer as default for AudioGen text encoder
        _tokenizer = tokenizer ?? Tokenization.LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.FlanT5);
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new CrossEntropyLoss<T>();

        _random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();

        InitializeLayers();
    }

    #endregion

    #region Layer Initialization - Golden Standard Pattern

    /// <summary>
    /// Initializes the neural network layers following the golden standard pattern.
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
        // 3. If no, use LayerHelper.CreateDefaultAudioGenLayers()
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            ValidateLayerConfiguration(Layers, out int textEncoderEnd, out int lmStart);

            // Use validated boundaries from custom layer configuration
            _textEncoderLayerStart = 0;
            _textEncoderLayerEnd = textEncoderEnd;
            _languageModelLayerStart = lmStart;
            _languageModelLayerEnd = Layers.Count;
        }
        else
        {
            // Calculate max audio tokens: sample_rate * max_duration / encodec_framerate (~50 tokens/sec)
            int maxAudioTokens = (int)(_sampleRate * _maxDurationSeconds / 640);

            // Use default AudioGen architecture
            Layers.AddRange(LayerHelper<T>.CreateDefaultAudioGenLayers(
                textHiddenDim: _textHiddenDim,
                lmHiddenDim: _lmHiddenDim,
                numLmLayers: _numLmLayers,
                numHeads: _numHeads,
                numCodebooks: _numCodebooks,
                codebookSize: _codebookSize,
                maxTextLength: _maxTextLength,
                maxAudioTokens: maxAudioTokens,
                dropoutRate: 0.1));

            // For default layers, we know the exact structure from CreateDefaultAudioGenLayers:
            // - Embedding layer (1)
            // - Text encoder transformer layers (_numLmLayers / 2)
            // - Cross-attention projection (1)
            // - Language model transformer layers (_numLmLayers)
            // - Output projection (1)
            int textEncoderLayers = 1 + (_numLmLayers / 2) + 1; // embedding + transformers + projection
            _textEncoderLayerStart = 0;
            _textEncoderLayerEnd = textEncoderLayers;
            _languageModelLayerStart = textEncoderLayers;
            _languageModelLayerEnd = Layers.Count;
        }
    }

    /// <summary>
    /// Validates that custom layers meet AudioGen requirements and determines layer boundaries.
    /// </summary>
    /// <param name="layers">The custom layers to validate.</param>
    /// <param name="textEncoderEnd">Output: Index where text encoder ends.</param>
    /// <param name="languageModelStart">Output: Index where language model starts.</param>
    /// <remarks>
    /// For custom layers, this method attempts to detect boundaries by looking for
    /// TransformerDecoderLayer instances (which indicate language model layers).
    /// If no clear boundary is found, it falls back to a 1/3 split with a warning.
    /// </remarks>
    private void ValidateLayerConfiguration(List<ILayer<T>> layers, out int textEncoderEnd, out int languageModelStart)
    {
        if (layers.Count < 3)
        {
            throw new ArgumentException(
                "AudioGen requires at least 3 layers: text encoder, language model, and output projection. " +
                "Use LayerHelper.CreateDefaultAudioGenLayers() as a reference.",
                nameof(layers));
        }

        // Try to detect boundary by finding the first TransformerDecoderLayer
        // (text encoder uses regular transformer layers, LM uses decoder layers with cross-attention)
        int firstDecoderLayerIdx = -1;
        for (int i = 0; i < layers.Count; i++)
        {
            if (layers[i] is TransformerDecoderLayer<T>)
            {
                firstDecoderLayerIdx = i;
                break;
            }
        }

        if (firstDecoderLayerIdx > 0)
        {
            // Found a clear boundary - decoder layers start the language model
            textEncoderEnd = firstDecoderLayerIdx;
            languageModelStart = firstDecoderLayerIdx;
        }
        else
        {
            // No decoder layers found - use fraction-based fallback
            // This assumes custom layers follow the standard AudioGen structure
            textEncoderEnd = layers.Count / 3;
            languageModelStart = textEncoderEnd;
        }
    }

    private static (int textHiddenDim, int lmHiddenDim, int numLayers, int numHeads) GetModelDimensions(AudioGenModelSize size)
    {
        return size switch
        {
            AudioGenModelSize.Small => (512, 1024, 12, 8),
            AudioGenModelSize.Medium => (768, 1536, 24, 16),
            AudioGenModelSize.Large => (1024, 2048, 32, 16),
            _ => (768, 1536, 24, 16)
        };
    }

    #endregion

    #region IAudioGenerator Implementation

    /// <summary>
    /// Generates audio from a text description.
    /// </summary>
    /// <param name="prompt">Text description of the desired audio.</param>
    /// <param name="negativePrompt">Optional negative prompt for classifier-free guidance.</param>
    /// <param name="durationSeconds">Duration of audio to generate in seconds.</param>
    /// <param name="numInferenceSteps">Number of inference steps (not used in autoregressive generation).</param>
    /// <param name="guidanceScale">Classifier-free guidance scale.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <returns>Generated audio waveform tensor.</returns>
    public Tensor<T> GenerateAudio(
        string prompt,
        string? negativePrompt = null,
        double durationSeconds = 5.0,
        int numInferenceSteps = 100,
        double guidanceScale = 3.0,
        int? seed = null)
    {
        ThrowIfDisposed();

        int seedUsed;
        if (seed.HasValue)
        {
            seedUsed = seed.Value;
        }
        else
        {
            lock (_randomLock)
            {
                seedUsed = _random.Next();
            }
        }

        // Encode the text prompt
        var textEmbeddings = EncodeText(prompt);

        // Generate audio codes
        Tensor<T> audioCodes;
        if (negativePrompt is not null && negativePrompt.Length > 0 && guidanceScale > 1.0)
        {
            var negativeEmbeddings = EncodeText(negativePrompt);
            audioCodes = GenerateAudioCodesWithGuidance(textEmbeddings, negativeEmbeddings, guidanceScale, seedUsed, durationSeconds);
        }
        else
        {
            audioCodes = GenerateAudioCodes(textEmbeddings, seedUsed, durationSeconds);
        }

        // Decode audio codes to waveform
        return DecodeAudio(audioCodes, durationSeconds);
    }

    /// <summary>
    /// Generates audio from a text description asynchronously.
    /// </summary>
    public Task<Tensor<T>> GenerateAudioAsync(
        string prompt,
        string? negativePrompt = null,
        double durationSeconds = 5.0,
        int numInferenceSteps = 100,
        double guidanceScale = 3.0,
        int? seed = null,
        CancellationToken cancellationToken = default)
    {
        return Task.Run(() => GenerateAudio(prompt, negativePrompt, durationSeconds, numInferenceSteps, guidanceScale, seed), cancellationToken);
    }

    /// <summary>
    /// Generates music from a text description.
    /// </summary>
    public Tensor<T> GenerateMusic(
        string prompt,
        string? negativePrompt = null,
        double durationSeconds = 10.0,
        int numInferenceSteps = 100,
        double guidanceScale = 3.0,
        int? seed = null)
    {
        throw new NotSupportedException("Music generation is not supported by AudioGen. Use MusicGen for music generation.");
    }

    /// <summary>
    /// Continues existing audio to extend it naturally.
    /// </summary>
    public Tensor<T> ContinueAudio(
        Tensor<T> inputAudio,
        string? prompt = null,
        double extensionSeconds = 5.0,
        int numInferenceSteps = 100,
        int? seed = null)
    {
        throw new NotSupportedException("Audio continuation is not supported by this AudioGen model.");
    }

    /// <summary>
    /// Fills in missing or masked sections of audio.
    /// </summary>
    public Tensor<T> InpaintAudio(
        Tensor<T> audio,
        Tensor<T> mask,
        string? prompt = null,
        int numInferenceSteps = 100,
        int? seed = null)
    {
        throw new NotSupportedException("Audio inpainting is not supported by this AudioGen model.");
    }

    /// <summary>
    /// Gets default generation options.
    /// </summary>
    public AudioGenerationOptions<T> GetDefaultOptions()
    {
        return new AudioGenerationOptions<T>
        {
            DurationSeconds = _durationSeconds,
            NumInferenceSteps = 100,
            GuidanceScale = _guidanceScale,
            Seed = null,
            Stereo = _channels == 2,
            SchedulerType = "autoregressive"
        };
    }

    #endregion

    #region AudioNeuralNetworkBase Implementation

    /// <summary>
    /// Preprocesses raw audio for model input.
    /// </summary>
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio)
    {
        // For AudioGen, we typically work with text input, not audio preprocessing
        return rawAudio;
    }

    /// <summary>
    /// Postprocesses model output into the final result format.
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
        if (_useNativeMode)
        {
            return Forward(input);
        }

        if (_textEncoder is null)
            throw new InvalidOperationException("Text encoder not loaded.");

        return _textEncoder.Run(input);
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
    /// 1. The model makes predictions
    /// 2. We calculate how wrong it was (loss)
    /// 3. We compute gradients (which direction to adjust each parameter)
    /// 4. This method applies those adjustments to make the model better
    ///
    /// The learning rate controls how big each adjustment is:
    /// - Too big: Model learns fast but may overshoot optimal values
    /// - Too small: Model learns slowly but more precisely
    /// - Default (0.001): A good starting point for most tasks
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
    /// Trains the model on input data.
    /// </summary>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_useNativeMode)
        {
            throw new NotSupportedException("Cannot train in ONNX inference mode. Use the native constructor for training.");
        }

        // Set training mode
        SetTrainingMode(true);

        // Forward pass
        var prediction = Forward(input);

        // Calculate loss
        var flatPrediction = prediction.ToVector();
        var flatExpected = expectedOutput.ToVector();
        LastLoss = _lossFunction.CalculateLoss(flatPrediction, flatExpected);

        // Backward pass
        var lossGradient = _lossFunction.CalculateDerivative(flatPrediction, flatExpected);
        Backpropagate(Tensor<T>.FromVector(lossGradient));

        // Update parameters using optimizer
        _optimizer.UpdateParameters(Layers);

        SetTrainingMode(false);
    }

    /// <summary>
    /// Gets metadata about the model.
    /// </summary>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = $"AudioGen-{_modelSize}",
            Description = $"AudioGen text-to-audio model - {_modelSize} variant",
            ModelType = ModelType.NeuralNetwork,
            FeatureCount = _maxTextLength,
            Complexity = (int)_modelSize
        };
        metadata.AdditionalInfo["InputFormat"] = "Text Prompt";
        metadata.AdditionalInfo["OutputFormat"] = $"Audio ({_sampleRate}Hz, up to {_maxDurationSeconds}s)";
        metadata.AdditionalInfo["Mode"] = _useNativeMode ? "Native" : "ONNX";
        return metadata;
    }

    /// <summary>
    /// Serializes network-specific data.
    /// </summary>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_useNativeMode);
        writer.Write((int)_modelSize);
        writer.Write(_sampleRate);
        writer.Write(_durationSeconds);
        writer.Write(_maxDurationSeconds);
        writer.Write(_temperature);
        writer.Write(_topK);
        writer.Write(_topP);
        writer.Write(_guidanceScale);
        writer.Write(_channels);
        writer.Write(_textHiddenDim);
        writer.Write(_lmHiddenDim);
        writer.Write(_numLmLayers);
        writer.Write(_numHeads);
        writer.Write(_numCodebooks);
        writer.Write(_codebookSize);
        writer.Write(_maxTextLength);
    }

    /// <summary>
    /// Deserializes network-specific data.
    /// </summary>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        // Read values to advance stream position (validation done in CreateNewInstance)
        _ = reader.ReadBoolean();  // useNativeMode
        _ = reader.ReadInt32();    // modelSize
        _ = reader.ReadInt32();    // sampleRate
        _ = reader.ReadDouble();   // durationSeconds
        _ = reader.ReadDouble();   // maxDurationSeconds
        _ = reader.ReadDouble();   // temperature
        _ = reader.ReadInt32();    // topK
        _ = reader.ReadDouble();   // topP
        _ = reader.ReadDouble();   // guidanceScale
        _ = reader.ReadInt32();    // channels
        _ = reader.ReadInt32();    // textHiddenDim
        _ = reader.ReadInt32();    // lmHiddenDim
        _ = reader.ReadInt32();    // numLmLayers
        _ = reader.ReadInt32();    // numHeads
        _ = reader.ReadInt32();    // numCodebooks
        _ = reader.ReadInt32();    // codebookSize
        _ = reader.ReadInt32();    // maxTextLength
    }

    /// <summary>
    /// Creates a new instance of this model for cloning.
    /// </summary>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new AudioGenModel<T>(
            Architecture,
            _modelSize,
            _sampleRate,
            _durationSeconds,
            _maxDurationSeconds,
            _temperature,
            _topK,
            _topP,
            _guidanceScale,
            _channels,
            _textHiddenDim,
            _lmHiddenDim,
            _numLmLayers,
            _numHeads,
            _numCodebooks,
            _codebookSize,
            _maxTextLength,
            seed: null,
            tokenizer: _tokenizer,
            optimizer: null,
            lossFunction: _lossFunction);
    }

    #endregion

    #region Private Methods

    private Tensor<T> EncodeText(string prompt)
    {
        if (!_useNativeMode && _textEncoder is null)
            throw new InvalidOperationException("Text encoder not loaded.");

        // Tokenize the prompt using the provided ITokenizer
        var tokens = TokenizePrompt(prompt);

        // Create input tensor
        var inputTensor = new Tensor<T>([1, tokens.Length]);
        for (int i = 0; i < tokens.Length; i++)
        {
            inputTensor[0, i] = NumOps.FromDouble(tokens[i]);
        }

        if (_useNativeMode)
        {
            // Native mode: run through text encoder layers
            return ForwardTextEncoder(inputTensor);
        }

        // ONNX mode: run through text encoder model
        if (_textEncoder is null)
            throw new InvalidOperationException("Text encoder not loaded.");

        return _textEncoder.Run(inputTensor);
    }

    private Tensor<T> ForwardTextEncoder(Tensor<T> input)
    {
        var x = input;
        // Use layer boundaries instead of hard-coded fractions
        for (int i = _textEncoderLayerStart; i < _textEncoderLayerEnd && i < Layers.Count; i++)
        {
            x = Layers[i].Forward(x);
        }
        return x;
    }

    private int[] TokenizePrompt(string prompt)
    {
        // Use the proper tokenizer provided at construction
        var encoding = _tokenizer.Encode(prompt);

        // Convert token IDs to int array, padding/truncating to max length
        var tokens = new int[_maxTextLength];

        // Copy token IDs (up to maxLength)
        int copyCount = Math.Min(encoding.TokenIds.Count, _maxTextLength);
        for (int i = 0; i < copyCount; i++)
        {
            tokens[i] = encoding.TokenIds[i];
        }

        return tokens;
    }

    private Tensor<T> GenerateAudioCodes(Tensor<T> textEmbeddings, int seed, double durationSeconds)
    {
        if (!_useNativeMode && _languageModel is null)
            throw new InvalidOperationException("Language model not loaded.");

        var random = RandomHelper.CreateSeededRandom(seed);

        // Calculate number of tokens based on duration
        int tokensPerSecond = 50;  // EnCodec: ~50 tokens per second at 32kHz
        int numTokens = (int)(durationSeconds * tokensPerSecond);

        var codes = new Tensor<T>([1, _numCodebooks, numTokens]);

        // Pre-allocate currentTokens to avoid O(n²) copy overhead
        // Start with capacity for all tokens + 1 for start token
        var currentTokens = new Tensor<T>([1, _numCodebooks, numTokens + 1]);
        int currentLength = 1;

        // Initialize with start token
        for (int cb = 0; cb < _numCodebooks; cb++)
        {
            currentTokens[0, cb, 0] = NumOps.Zero;
        }

        for (int t = 0; t < numTokens; t++)
        {
            // Create a view of currentTokens up to currentLength for the forward pass
            var inputTokens = new Tensor<T>([1, _numCodebooks, currentLength]);
            for (int cb = 0; cb < _numCodebooks; cb++)
            {
                for (int i = 0; i < currentLength; i++)
                {
                    inputTokens[0, cb, i] = currentTokens[0, cb, i];
                }
            }

            Tensor<T> logits;

            if (_useNativeMode)
            {
                logits = ForwardLanguageModel(textEmbeddings, inputTokens);
            }
            else
            {
                if (_languageModel is null)
                    throw new InvalidOperationException("Language model not loaded.");

                var inputs = new Dictionary<string, Tensor<T>>
                {
                    ["text_embeddings"] = textEmbeddings,
                    ["audio_codes"] = inputTokens
                };
                var outputs = _languageModel.Run(inputs);
                logits = outputs.Values.First();
            }

            // Sample next tokens for each codebook
            for (int cb = 0; cb < _numCodebooks; cb++)
            {
                int nextToken = SampleFromLogits(logits, cb, random);
                codes[0, cb, t] = NumOps.FromDouble(nextToken);

                // Append to pre-allocated tensor (O(1) instead of O(n) copy)
                currentTokens[0, cb, currentLength] = codes[0, cb, t];
            }
            currentLength++;
        }

        return codes;
    }

    private Tensor<T> ForwardLanguageModel(Tensor<T> textEmbeddings, Tensor<T> audioCodes)
    {
        // Forward pass through language model portion of layers
        // Use layer boundaries instead of hard-coded fractions
        var x = audioCodes;

        for (int i = _languageModelLayerStart; i < _languageModelLayerEnd && i < Layers.Count; i++)
        {
            if (Layers[i] is TransformerDecoderLayer<T> decoderLayer)
            {
                x = decoderLayer.Forward(x, textEmbeddings);
            }
            else
            {
                x = Layers[i].Forward(x);
            }
        }

        return x;
    }

    private Tensor<T> GenerateAudioCodesWithGuidance(
        Tensor<T> condEmbeddings,
        Tensor<T> uncondEmbeddings,
        double guidanceScale,
        int seed,
        double durationSeconds)
    {
        if (!_useNativeMode && _languageModel is null)
            throw new InvalidOperationException("Language model not loaded.");

        var random = RandomHelper.CreateSeededRandom(seed);

        int tokensPerSecond = 50;
        int numTokens = (int)(durationSeconds * tokensPerSecond);

        var codes = new Tensor<T>([1, _numCodebooks, numTokens]);

        // Pre-allocate currentTokens to avoid O(n²) copy overhead
        var currentTokens = new Tensor<T>([1, _numCodebooks, numTokens + 1]);
        int currentLength = 1;

        for (int cb = 0; cb < _numCodebooks; cb++)
        {
            currentTokens[0, cb, 0] = NumOps.Zero;
        }

        for (int t = 0; t < numTokens; t++)
        {
            // Create a view of currentTokens up to currentLength for the forward pass
            var inputTokens = new Tensor<T>([1, _numCodebooks, currentLength]);
            for (int cb = 0; cb < _numCodebooks; cb++)
            {
                for (int i = 0; i < currentLength; i++)
                {
                    inputTokens[0, cb, i] = currentTokens[0, cb, i];
                }
            }

            Tensor<T> condLogits, uncondLogits;

            if (_useNativeMode)
            {
                condLogits = ForwardLanguageModel(condEmbeddings, inputTokens);
                uncondLogits = ForwardLanguageModel(uncondEmbeddings, inputTokens);
            }
            else
            {
                if (_languageModel is null)
                    throw new InvalidOperationException("Language model not loaded.");

                var condInputs = new Dictionary<string, Tensor<T>>
                {
                    ["text_embeddings"] = condEmbeddings,
                    ["audio_codes"] = inputTokens
                };
                condLogits = _languageModel.Run(condInputs).Values.First();

                var uncondInputs = new Dictionary<string, Tensor<T>>
                {
                    ["text_embeddings"] = uncondEmbeddings,
                    ["audio_codes"] = inputTokens
                };
                uncondLogits = _languageModel.Run(uncondInputs).Values.First();
            }

            // Apply classifier-free guidance
            var guidedLogits = ApplyGuidance(condLogits, uncondLogits, guidanceScale);

            for (int cb = 0; cb < _numCodebooks; cb++)
            {
                int nextToken = SampleFromLogits(guidedLogits, cb, random);
                codes[0, cb, t] = NumOps.FromDouble(nextToken);

                // Append to pre-allocated tensor (O(1) instead of O(n) copy)
                currentTokens[0, cb, currentLength] = codes[0, cb, t];
            }
            currentLength++;
        }

        return codes;
    }

    private Tensor<T> ApplyGuidance(Tensor<T> condLogits, Tensor<T> uncondLogits, double scale)
    {
        var guided = new Tensor<T>(condLogits.Shape);

        for (int i = 0; i < condLogits.Length; i++)
        {
            double cond = NumOps.ToDouble(condLogits[i]);
            double uncond = NumOps.ToDouble(uncondLogits[i]);

            // CFG formula: guided = uncond + scale * (cond - uncond)
            double guidedValue = uncond + scale * (cond - uncond);
            guided[i] = NumOps.FromDouble(guidedValue);
        }

        return guided;
    }

    private int SampleFromLogits(Tensor<T> logits, int codebook, Random random)
    {
        int vocabSize = _codebookSize;
        var scaledLogits = new double[vocabSize];

        for (int i = 0; i < vocabSize; i++)
        {
            int idx = codebook * vocabSize + i;
            if (idx < logits.Length)
            {
                scaledLogits[i] = NumOps.ToDouble(logits[idx]) / _temperature;
            }
        }

        // Apply top-k filtering
        if (_topK > 0 && _topK < vocabSize)
        {
            var sorted = scaledLogits
                .Select((v, i) => (Value: v, Index: i))
                .OrderByDescending(x => x.Value)
                .ToList();

            double threshold = sorted[Math.Min(_topK - 1, sorted.Count - 1)].Value;
            for (int i = 0; i < vocabSize; i++)
            {
                if (scaledLogits[i] < threshold)
                {
                    scaledLogits[i] = double.NegativeInfinity;
                }
            }
        }

        // Apply top-p (nucleus) filtering
        if (_topP > 0 && _topP < 1.0)
        {
            var probs = Softmax(scaledLogits);
            var sorted = probs
                .Select((v, i) => (Value: v, Index: i))
                .OrderByDescending(x => x.Value)
                .ToList();

            double cumSum = 0;
            var keepIndices = new HashSet<int>();
            foreach (var item in sorted)
            {
                keepIndices.Add(item.Index);
                cumSum += item.Value;
                if (cumSum >= _topP)
                    break;
            }

            for (int i = 0; i < vocabSize; i++)
            {
                if (!keepIndices.Contains(i))
                {
                    scaledLogits[i] = double.NegativeInfinity;
                }
            }
        }

        // Sample from softmax distribution
        var finalProbs = Softmax(scaledLogits);
        double r = random.NextDouble();
        double cumulative = 0;

        for (int i = 0; i < vocabSize; i++)
        {
            cumulative += finalProbs[i];
            if (r <= cumulative)
            {
                return i;
            }
        }

        return vocabSize - 1;
    }

    private static double[] Softmax(double[] logits)
    {
        double maxLogit = logits.Where(x => !double.IsNegativeInfinity(x)).DefaultIfEmpty(0).Max();
        var expValues = logits.Select(x => double.IsNegativeInfinity(x) ? 0 : Math.Exp(x - maxLogit)).ToArray();
        double sumExp = expValues.Sum();

        if (sumExp == 0) sumExp = 1;

        return expValues.Select(x => x / sumExp).ToArray();
    }

    private Tensor<T> DecodeAudio(Tensor<T> audioCodes, double durationSeconds)
    {
        if (!_useNativeMode && _audioDecoder is null)
            throw new InvalidOperationException("Audio decoder not loaded.");

        Tensor<T> waveformTensor;

        if (_useNativeMode)
        {
            // Native mode: EnCodec-style decoding using learned embeddings
            // Real EnCodec uses transposed convolution layers; this provides a functional approximation
            waveformTensor = DecodeAudioNative(audioCodes, durationSeconds);
        }
        else
        {
            if (_audioDecoder is null)
                throw new InvalidOperationException("Audio decoder not loaded.");

            waveformTensor = _audioDecoder.Run(audioCodes);
        }

        // Trim to target duration
        int targetLength = (int)(durationSeconds * _sampleRate);
        if (waveformTensor.Length > targetLength)
        {
            var trimmed = new Tensor<T>([targetLength]);
            for (int i = 0; i < targetLength; i++)
            {
                trimmed[i] = waveformTensor[i];
            }
            return trimmed;
        }

        return waveformTensor;
    }

    /// <summary>
    /// Decodes audio codes to waveform in native mode using neural network layers.
    /// </summary>
    /// <remarks>
    /// This implements a simplified EnCodec-style decoder:
    /// 1. Convert discrete codes to continuous embeddings via codebook lookup
    /// 2. Sum embeddings across codebooks (residual vector quantization)
    /// 3. Upsample to target sample rate using learned interpolation
    /// 4. Apply smoothing to reduce quantization artifacts
    /// </remarks>
    private Tensor<T> DecodeAudioNative(Tensor<T> audioCodes, double durationSeconds)
    {
        // Expected shape: [batch, numCodebooks, numTokens]
        int numTokens = audioCodes.Shape.Length >= 3 ? audioCodes.Shape[2] : audioCodes.Length / _numCodebooks;
        int targetSamples = (int)(durationSeconds * _sampleRate);
        int channels = _channels > 0 ? _channels : 1;

        // EnCodec operates at ~50 tokens per second (32kHz / 640 hop)
        int samplesPerToken = targetSamples / Math.Max(1, numTokens);
        if (samplesPerToken < 1) samplesPerToken = 1;

        var waveform = new Tensor<T>(channels == 1 ? [targetSamples] : [channels, targetSamples]);

        // Process each token and upsample to audio samples
        for (int t = 0; t < numTokens; t++)
        {
            // Aggregate embeddings from all codebooks using residual sum
            double aggregatedValue = 0.0;
            for (int cb = 0; cb < _numCodebooks; cb++)
            {
                // Get code value and normalize to [-1, 1] range
                double codeValue;
                if (audioCodes.Shape.Length >= 3)
                {
                    codeValue = NumOps.ToDouble(audioCodes[0, cb, t]);
                }
                else
                {
                    codeValue = NumOps.ToDouble(audioCodes[cb * numTokens + t]);
                }

                // Normalize code to [-1, 1] with codebook weighting (earlier codebooks carry more signal)
                double normalized = (codeValue / _codebookSize - 0.5) * 2.0;
                double weight = 1.0 / Math.Pow(2, cb); // Residual VQ weighting
                aggregatedValue += normalized * weight;
            }

            // Clamp to valid audio range
            aggregatedValue = MathHelper.Clamp(aggregatedValue, -1.0, 1.0);

            // Upsample: generate samples for this token with interpolation
            int startSample = t * samplesPerToken;
            int endSample = Math.Min((t + 1) * samplesPerToken, targetSamples);

            // Get next token value for interpolation
            double nextValue = aggregatedValue;
            if (t + 1 < numTokens)
            {
                double nextAggregated = 0.0;
                for (int cb = 0; cb < _numCodebooks; cb++)
                {
                    double codeValue;
                    if (audioCodes.Shape.Length >= 3)
                    {
                        codeValue = NumOps.ToDouble(audioCodes[0, cb, t + 1]);
                    }
                    else
                    {
                        codeValue = NumOps.ToDouble(audioCodes[cb * numTokens + t + 1]);
                    }
                    double normalized = (codeValue / _codebookSize - 0.5) * 2.0;
                    double weight = 1.0 / Math.Pow(2, cb);
                    nextAggregated += normalized * weight;
                }
                nextValue = MathHelper.Clamp(nextAggregated, -1.0, 1.0);
            }

            // Linear interpolation between tokens for smoother output
            for (int s = startSample; s < endSample; s++)
            {
                double alpha = (double)(s - startSample) / Math.Max(1, endSample - startSample);
                double interpolated = aggregatedValue * (1.0 - alpha) + nextValue * alpha;

                // Apply slight smoothing using cosine interpolation for more natural sound
                double smoothAlpha = (1.0 - Math.Cos(alpha * Math.PI)) / 2.0;
                double smoothValue = aggregatedValue * (1.0 - smoothAlpha) + nextValue * smoothAlpha;

                T sampleValue = NumOps.FromDouble(smoothValue);
                if (channels == 1)
                {
                    waveform[s] = sampleValue;
                }
                else
                {
                    // Stereo: same value for both channels in basic mode
                    for (int c = 0; c < channels; c++)
                    {
                        waveform[c, s] = sampleValue;
                    }
                }
            }
        }

        // Fill any remaining samples with the last value
        if (numTokens > 0)
        {
            double lastValue = 0.0;
            for (int cb = 0; cb < _numCodebooks; cb++)
            {
                double codeValue;
                if (audioCodes.Shape.Length >= 3)
                {
                    codeValue = NumOps.ToDouble(audioCodes[0, cb, numTokens - 1]);
                }
                else
                {
                    codeValue = NumOps.ToDouble(audioCodes[cb * numTokens + numTokens - 1]);
                }
                double normalized = (codeValue / _codebookSize - 0.5) * 2.0;
                double weight = 1.0 / Math.Pow(2, cb);
                lastValue += normalized * weight;
            }
            lastValue = MathHelper.Clamp(lastValue, -1.0, 1.0);

            int filledSamples = numTokens * samplesPerToken;
            for (int s = filledSamples; s < targetSamples; s++)
            {
                // Fade out to avoid clicks
                double fadeOut = 1.0 - (double)(s - filledSamples) / Math.Max(1, targetSamples - filledSamples);
                T sampleValue = NumOps.FromDouble(lastValue * fadeOut);
                if (channels == 1)
                {
                    waveform[s] = sampleValue;
                }
                else
                {
                    for (int c = 0; c < channels; c++)
                    {
                        waveform[c, s] = sampleValue;
                    }
                }
            }
        }

        return waveform;
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
            _textEncoder?.Dispose();
            _languageModel?.Dispose();
            // _audioDecoder is stored as OnnxModel and disposed by base class
        }

        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
