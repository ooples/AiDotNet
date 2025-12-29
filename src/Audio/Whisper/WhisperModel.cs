using System.Diagnostics;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
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
/// Usage:
/// <code>
/// var whisper = await WhisperModel&lt;float&gt;.CreateAsync(new WhisperOptions
/// {
///     ModelSize = WhisperModelSize.Base,
///     Language = "en"
/// });
///
/// var audio = LoadAudio("speech.wav");  // Your audio loading code
/// var result = whisper.Transcribe(audio);
/// Console.WriteLine(result.Text);
///
/// whisper.Dispose();
/// </code>
/// </para>
/// </remarks>
public class WhisperModel<T> : AudioNeuralNetworkBase<T>, ISpeechRecognizer<T>
{
    private readonly WhisperOptions _options;
    private readonly MelSpectrogram<T> _melSpectrogram;
    private readonly WhisperTokenizer _tokenizer;
    private bool _disposed;

    /// <summary>
    /// Gets the model options.
    /// </summary>
    public WhisperOptions Options => _options;

    /// <summary>
    /// Gets whether the model is ready for inference.
    /// </summary>
    public bool IsReady => OnnxEncoder?.IsLoaded == true && OnnxDecoder?.IsLoaded == true;

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

    /// <summary>
    /// Creates a new WhisperModel instance.
    /// </summary>
    /// <param name="options">Configuration options.</param>
    /// <param name="encoder">Pre-loaded encoder model.</param>
    /// <param name="decoder">Pre-loaded decoder model.</param>
    private WhisperModel(WhisperOptions options, OnnxModel<T>? encoder, OnnxModel<T>? decoder)
        : base(CreateMinimalArchitecture(options))
    {
        _options = options;
        OnnxEncoder = encoder;
        OnnxDecoder = decoder;
        _tokenizer = new WhisperTokenizer();

        // Set audio properties
        SampleRate = options.SampleRate;
        NumMels = options.NumMels;

        // Create mel spectrogram preprocessor with Whisper parameters
        _melSpectrogram = new MelSpectrogram<T>(
            sampleRate: options.SampleRate,
            nMels: options.NumMels,
            nFft: 400,      // Whisper uses 25ms windows at 16kHz
            hopLength: 160, // Whisper uses 10ms hop at 16kHz
            fMin: 0,
            fMax: 8000,     // Whisper limits to 8kHz
            logMel: true);

        MelSpec = _melSpectrogram;

        // Initialize supported languages based on model size
        SupportedLanguages = GetSupportedLanguages(options.ModelSize);
    }

    private static NeuralNetworkArchitecture<T> CreateMinimalArchitecture(WhisperOptions options)
    {
        // Create minimal architecture for ONNX-only model
        return new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.SpeechRecognition,
            complexity: NetworkComplexity.VeryDeep,
            inputSize: options.SampleRate * options.MaxAudioLengthSeconds, // 30 seconds of audio
            outputSize: 51865 // Whisper vocabulary size
        );
    }

    private static IReadOnlyList<string> GetSupportedLanguages(WhisperModelSize size)
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

    /// <summary>
    /// Creates a WhisperModel asynchronously, downloading model files if needed.
    /// </summary>
    /// <param name="options">Configuration options.</param>
    /// <param name="progress">Optional download progress reporter.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The initialized WhisperModel.</returns>
    public static async Task<WhisperModel<T>> CreateAsync(
        WhisperOptions? options = null,
        IProgress<double>? progress = null,
        CancellationToken cancellationToken = default)
    {
        options ??= new WhisperOptions();

        OnnxModel<T>? encoder = null;
        OnnxModel<T>? decoder = null;

        try
        {
            // Load or download encoder
            if (options.EncoderModelPath is not null && options.EncoderModelPath.Length > 0)
            {
                encoder = new OnnxModel<T>(options.EncoderModelPath, options.OnnxOptions);
            }
            else
            {
                var downloader = new OnnxModelDownloader();
                var modelId = GetModelId(options.ModelSize);
                var encoderPath = await downloader.DownloadAsync(
                    modelId,
                    "encoder.onnx",
                    progress: new Progress<double>(p => progress?.Report(p * 0.5)),
                    cancellationToken);
                encoder = new OnnxModel<T>(encoderPath, options.OnnxOptions);
            }

            // Load or download decoder
            if (options.DecoderModelPath is not null && options.DecoderModelPath.Length > 0)
            {
                decoder = new OnnxModel<T>(options.DecoderModelPath, options.OnnxOptions);
            }
            else
            {
                var downloader = new OnnxModelDownloader();
                var modelId = GetModelId(options.ModelSize);
                var decoderPath = await downloader.DownloadAsync(
                    modelId,
                    "decoder.onnx",
                    progress: new Progress<double>(p => progress?.Report(0.5 + p * 0.5)),
                    cancellationToken);
                decoder = new OnnxModel<T>(decoderPath, options.OnnxOptions);
            }

            return new WhisperModel<T>(options, encoder, decoder);
        }
        catch
        {
            encoder?.Dispose();
            decoder?.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Creates a WhisperModel from local model files.
    /// </summary>
    /// <param name="encoderPath">Path to the encoder ONNX model.</param>
    /// <param name="decoderPath">Path to the decoder ONNX model.</param>
    /// <param name="options">Configuration options.</param>
    /// <returns>The initialized WhisperModel.</returns>
    public static WhisperModel<T> FromFiles(
        string encoderPath,
        string decoderPath,
        WhisperOptions? options = null)
    {
        options ??= new WhisperOptions();
        options.EncoderModelPath = encoderPath;
        options.DecoderModelPath = decoderPath;

        var encoder = new OnnxModel<T>(encoderPath, options.OnnxOptions);
        var decoder = new OnnxModel<T>(decoderPath, options.OnnxOptions);

        return new WhisperModel<T>(options, encoder, decoder);
    }

    #region ISpeechRecognizer Implementation

    /// <summary>
    /// Transcribes audio to text.
    /// </summary>
    /// <param name="audio">Audio waveform tensor [batch, samples] or [samples].</param>
    /// <param name="language">Optional language code (e.g., "en", "es"). Auto-detected if null.</param>
    /// <param name="includeTimestamps">Whether to include word-level timestamps.</param>
    /// <returns>Transcription result containing text and optional timestamps.</returns>
    public TranscriptionResult<T> Transcribe(Tensor<T> audio, string? language = null, bool includeTimestamps = false)
    {
        ThrowIfDisposed();

        var stopwatch = Stopwatch.StartNew();

        // Override language if specified
        string effectiveLanguage = language ?? _options.Language ?? "en";

        // Preprocess audio to mel spectrogram
        var melFeatures = PreprocessAudio(audio);

        // Encode audio features
        var encoderOutput = EncodeAudio(melFeatures);

        // Decode to text tokens
        var tokens = DecodeTokens(encoderOutput, effectiveLanguage);

        // Convert tokens to text
        var text = _tokenizer.Decode(tokens);

        stopwatch.Stop();

        return new TranscriptionResult<T>
        {
            Text = text,
            Language = effectiveLanguage,
            Confidence = NumOps.FromDouble(1.0), // TODO: Extract from decoder
            DurationSeconds = NumOps.ToDouble(audio[audio.Length - 1]) / SampleRate,
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
        if (OnnxDecoder is null)
            throw new InvalidOperationException("Decoder not loaded.");

        // Create initial token sequence for language detection
        var initialTokens = new Tensor<T>([1, 1]);
        initialTokens[0, 0] = NumOps.FromDouble(_tokenizer.StartOfTranscript);

        var inputs = new Dictionary<string, Tensor<T>>
        {
            ["encoder_hidden_states"] = encoderOutput,
            ["input_ids"] = initialTokens
        };

        var output = OnnxDecoder.Run(inputs);
        var logits = output.Values.First();

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
                EndTime = _options.MaxAudioLengthSeconds,
                Confidence = NumOps.FromDouble(1.0)
            });
        }

        return segments;
    }

    #endregion

    #region AudioNeuralNetworkBase Implementation

    /// <summary>
    /// Preprocesses raw audio for model input.
    /// </summary>
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio)
    {
        // Pad or truncate to 30 seconds
        int targetLength = _options.SampleRate * _options.MaxAudioLengthSeconds;
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
    /// Initializes the neural network layers.
    /// </summary>
    protected override void InitializeLayers()
    {
        // ONNX-only model - no native layers to initialize
    }

    /// <summary>
    /// Makes a prediction using the model.
    /// </summary>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        var preprocessed = PreprocessAudio(input);

        if (IsOnnxMode)
        {
            return RunOnnxInference(preprocessed);
        }
        else
        {
            return Forward(preprocessed);
        }
    }

    /// <summary>
    /// Updates model parameters.
    /// </summary>
    public override void UpdateParameters(Vector<T> parameters)
    {
        if (IsOnnxMode)
        {
            throw new NotSupportedException("Cannot update parameters in ONNX inference mode.");
        }

        // Native training mode - update layer parameters from the parameter vector
        // This would be implemented when native training support is added
        throw new NotImplementedException("Native parameter updates for Whisper are not yet implemented.");
    }

    /// <summary>
    /// Trains the model on input data.
    /// </summary>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (IsOnnxMode)
        {
            throw new NotSupportedException("Cannot train in ONNX inference mode. Load the model in training mode instead.");
        }

        // Native training would be implemented here
        throw new NotImplementedException("Native training for Whisper is not yet implemented.");
    }

    /// <summary>
    /// Gets metadata about the model.
    /// </summary>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = $"Whisper-{_options.ModelSize}",
            Description = $"Whisper speech recognition model - {_options.ModelSize} variant",
            ModelType = ModelType.NeuralNetwork,
            FeatureCount = SampleRate * _options.MaxAudioLengthSeconds,
            Complexity = (int)_options.ModelSize
        };
        metadata.AdditionalInfo["InputFormat"] = $"Audio ({SampleRate}Hz, {_options.MaxAudioLengthSeconds}s max)";
        metadata.AdditionalInfo["OutputFormat"] = "Transcription";
        return metadata;
    }

    /// <summary>
    /// Serializes network-specific data.
    /// </summary>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_options.SampleRate);
        writer.Write(_options.NumMels);
        writer.Write(_options.MaxAudioLengthSeconds);
        writer.Write((int)_options.ModelSize);
        writer.Write(_options.Language ?? string.Empty);
    }

    /// <summary>
    /// Deserializes network-specific data.
    /// </summary>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _options.SampleRate = reader.ReadInt32();
        _options.NumMels = reader.ReadInt32();
        _options.MaxAudioLengthSeconds = reader.ReadInt32();
        _options.ModelSize = (WhisperModelSize)reader.ReadInt32();
        var lang = reader.ReadString();
        _options.Language = string.IsNullOrEmpty(lang) ? null : lang;
    }

    /// <summary>
    /// Creates a new instance of this model for cloning.
    /// </summary>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new WhisperModel<T>(_options, null, null);
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

    private List<long> DecodeTokens(Tensor<T> encoderOutput, string language)
    {
        if (OnnxDecoder is null)
            throw new InvalidOperationException("Decoder not loaded.");

        var tokens = new List<long>();

        // Start with special tokens
        tokens.Add(_tokenizer.StartOfTranscript);
        tokens.Add(_tokenizer.GetLanguageToken(language));

        if (_options.Translate)
        {
            tokens.Add(_tokenizer.TranslateToken);
        }
        else
        {
            tokens.Add(_tokenizer.TranscribeToken);
        }

        // Autoregressive decoding
        for (int step = 0; step < _options.MaxTokens; step++)
        {
            // Create input tensor for decoder
            var tokenTensor = CreateTokenTensor(tokens);

            // Run decoder
            var inputs = new Dictionary<string, Tensor<T>>
            {
                ["encoder_hidden_states"] = encoderOutput,
                ["input_ids"] = tokenTensor
            };

            var output = OnnxDecoder.Run(inputs);
            var logits = output.Values.First();

            // Get next token (greedy decoding for now)
            int nextToken = GetNextToken(logits, tokens.Count - 1);

            if (nextToken == _tokenizer.EndOfText)
            {
                break;
            }

            tokens.Add(nextToken);
        }

        return tokens;
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

    private int GetNextToken(Tensor<T> logits, int position)
    {
        // Get logits for the last position
        int vocabSize = logits.Shape[^1];
        int maxIdx = 0;
        double maxValue = double.NegativeInfinity;

        for (int v = 0; v < vocabSize; v++)
        {
            double value = NumOps.ToDouble(logits[0, position, v]);
            if (value > maxValue)
            {
                maxValue = value;
                maxIdx = v;
            }
        }

        return maxIdx;
    }

    private static string GetModelId(WhisperModelSize size)
    {
        return size switch
        {
            WhisperModelSize.Tiny => "openai/whisper-tiny",
            WhisperModelSize.Base => "openai/whisper-base",
            WhisperModelSize.Small => "openai/whisper-small",
            WhisperModelSize.Medium => "openai/whisper-medium",
            WhisperModelSize.Large => "openai/whisper-large",
            WhisperModelSize.LargeV2 => "openai/whisper-large-v2",
            WhisperModelSize.LargeV3 => "openai/whisper-large-v3",
            _ => "openai/whisper-base"
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

        if (disposing)
        {
            // Base class disposes OnnxEncoder, OnnxDecoder, OnnxModel
        }

        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
