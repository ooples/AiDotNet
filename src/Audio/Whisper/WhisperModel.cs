using System.Diagnostics;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Interfaces;
using AiDotNet.Onnx;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
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
/// var result = await whisper.TranscribeAsync(audio);
/// Console.WriteLine(result.Text);
///
/// whisper.Dispose();
/// </code>
/// </para>
/// </remarks>
public class WhisperModel<T> : IDisposable
{
    private readonly INumericOperations<T> _numOps;
    private readonly WhisperOptions _options;
    private readonly MelSpectrogram<T> _melSpectrogram;
    private readonly OnnxModel<T>? _encoder;
    private readonly OnnxModel<T>? _decoder;
    private readonly WhisperTokenizer _tokenizer;
    private bool _disposed;

    /// <summary>
    /// Gets the model options.
    /// </summary>
    public WhisperOptions Options => _options;

    /// <summary>
    /// Gets whether the model is ready for inference.
    /// </summary>
    public bool IsReady => _encoder?.IsLoaded == true && _decoder?.IsLoaded == true;

    /// <summary>
    /// Creates a new WhisperModel instance.
    /// </summary>
    /// <param name="options">Configuration options.</param>
    /// <param name="encoder">Pre-loaded encoder model.</param>
    /// <param name="decoder">Pre-loaded decoder model.</param>
    private WhisperModel(WhisperOptions options, OnnxModel<T>? encoder, OnnxModel<T>? decoder)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _options = options;
        _encoder = encoder;
        _decoder = decoder;
        _tokenizer = new WhisperTokenizer();

        // Create mel spectrogram preprocessor with Whisper parameters
        _melSpectrogram = new MelSpectrogram<T>(
            sampleRate: options.SampleRate,
            nMels: options.NumMels,
            nFft: 400,      // Whisper uses 25ms windows at 16kHz
            hopLength: 160, // Whisper uses 10ms hop at 16kHz
            fMin: 0,
            fMax: 8000,     // Whisper limits to 8kHz
            logMel: true);
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
            if (!string.IsNullOrEmpty(options.EncoderModelPath))
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
            if (!string.IsNullOrEmpty(options.DecoderModelPath))
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

    /// <summary>
    /// Transcribes audio to text.
    /// </summary>
    /// <param name="audio">Audio waveform at 16kHz sample rate.</param>
    /// <returns>Transcription result.</returns>
    public WhisperResult Transcribe(Tensor<T> audio)
    {
        ThrowIfDisposed();

        var stopwatch = Stopwatch.StartNew();

        // Preprocess audio to mel spectrogram
        var melFeatures = PreprocessAudio(audio);

        // Encode audio features
        var encoderOutput = EncodeAudio(melFeatures);

        // Decode to text tokens
        var tokens = DecodeTokens(encoderOutput);

        // Convert tokens to text
        var text = _tokenizer.Decode(tokens);

        stopwatch.Stop();

        return new WhisperResult
        {
            Text = text,
            DetectedLanguage = _options.Language,
            ProcessingTimeMs = stopwatch.ElapsedMilliseconds
        };
    }

    /// <summary>
    /// Transcribes audio to text asynchronously.
    /// </summary>
    /// <param name="audio">Audio waveform at 16kHz sample rate.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Transcription result.</returns>
    public Task<WhisperResult> TranscribeAsync(
        Tensor<T> audio,
        CancellationToken cancellationToken = default)
    {
        return Task.Run(() => Transcribe(audio), cancellationToken);
    }

    /// <summary>
    /// Transcribes multiple audio segments in batch.
    /// </summary>
    /// <param name="audioSegments">List of audio segments.</param>
    /// <returns>List of transcription results.</returns>
    public List<WhisperResult> TranscribeBatch(IEnumerable<Tensor<T>> audioSegments)
    {
        return audioSegments.Select(Transcribe).ToList();
    }

    private Tensor<T> PreprocessAudio(Tensor<T> audio)
    {
        // Pad or truncate to 30 seconds
        int targetLength = _options.SampleRate * _options.MaxAudioLengthSeconds;
        var paddedAudio = PadOrTruncate(audio, targetLength);

        // Compute mel spectrogram
        var melSpec = _melSpectrogram.Forward(paddedAudio);

        return melSpec;
    }

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
                result[i] = _numOps.Zero;
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
        if (_encoder is null)
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

        return _encoder.Run(melFeatures);
    }

    private List<long> DecodeTokens(Tensor<T> encoderOutput)
    {
        if (_decoder is null)
            throw new InvalidOperationException("Decoder not loaded.");

        var tokens = new List<long>();

        // Start with special tokens
        tokens.Add(_tokenizer.StartOfTranscript);

        if (!string.IsNullOrEmpty(_options.Language))
        {
            tokens.Add(_tokenizer.GetLanguageToken(_options.Language));
        }

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

            var output = _decoder.Run(inputs);
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
            tensor[0, i] = _numOps.FromDouble(tokens[i]);
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
            double value = _numOps.ToDouble(logits[0, position, v]);
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
        ObjectDisposedException.ThrowIf(_disposed, this);
    }

    /// <summary>
    /// Disposes the model and releases resources.
    /// </summary>
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Disposes managed and unmanaged resources.
    /// </summary>
    protected virtual void Dispose(bool disposing)
    {
        if (_disposed) return;

        if (disposing)
        {
            _encoder?.Dispose();
            _decoder?.Dispose();
        }

        _disposed = true;
    }
}
