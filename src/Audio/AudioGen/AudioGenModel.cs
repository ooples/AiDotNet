using System.Diagnostics;
using AiDotNet.Onnx;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.AudioGen;

/// <summary>
/// Audio generation model that creates audio from text descriptions.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// AudioGen uses a language model approach to generate audio from text prompts.
/// The pipeline consists of:
/// 1. Text Encoder: Converts text prompt to embeddings
/// 2. Audio Language Model: Generates discrete audio codes autoregressively
/// 3. Audio Decoder (EnCodec): Converts audio codes to waveform
/// </para>
/// <para><b>For Beginners:</b> AudioGen is different from TTS:
/// - TTS: Converts specific words to speech ("Hello" → spoken "Hello")
/// - AudioGen: Creates sounds matching a description ("dog barking" → bark sound)
///
/// Usage:
/// <code>
/// var audioGen = await AudioGenModel&lt;float&gt;.CreateAsync(new AudioGenOptions
/// {
///     DurationSeconds = 5.0,
///     Temperature = 1.0
/// });
///
/// var result = audioGen.Generate("A dog barking in the distance");
/// SaveAudio(result.Audio, result.SampleRate);  // Your audio saving code
///
/// audioGen.Dispose();
/// </code>
/// </para>
/// </remarks>
public class AudioGenModel<T> : IDisposable
{
    private readonly INumericOperations<T> _numOps;
    private readonly AudioGenOptions _options;
    private readonly OnnxModel<T>? _textEncoder;
    private readonly OnnxModel<T>? _languageModel;
    private readonly OnnxModel<T>? _audioDecoder;
    private readonly Random _random;
    private bool _disposed;

    /// <summary>
    /// Gets the model options.
    /// </summary>
    public AudioGenOptions Options => _options;

    /// <summary>
    /// Gets whether the model is ready for generation.
    /// </summary>
    public bool IsReady => _textEncoder?.IsLoaded == true &&
        _languageModel?.IsLoaded == true &&
        _audioDecoder?.IsLoaded == true;

    /// <summary>
    /// Creates a new AudioGenModel instance.
    /// </summary>
    private AudioGenModel(
        AudioGenOptions options,
        OnnxModel<T>? textEncoder,
        OnnxModel<T>? languageModel,
        OnnxModel<T>? audioDecoder)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _options = options;
        _textEncoder = textEncoder;
        _languageModel = languageModel;
        _audioDecoder = audioDecoder;
        _random = options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(options.Seed.Value)
            : RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Creates an AudioGenModel asynchronously, downloading models if needed.
    /// </summary>
    /// <param name="options">Configuration options.</param>
    /// <param name="progress">Optional download progress reporter.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The initialized AudioGenModel.</returns>
    public static async Task<AudioGenModel<T>> CreateAsync(
        AudioGenOptions? options = null,
        IProgress<double>? progress = null,
        CancellationToken cancellationToken = default)
    {
        options ??= new AudioGenOptions();

        OnnxModel<T>? textEncoder = null;
        OnnxModel<T>? languageModel = null;
        OnnxModel<T>? audioDecoder = null;

        try
        {
            var downloader = new OnnxModelDownloader();
            var modelRepo = GetModelRepository(options.ModelSize);

            // Load text encoder
            if (options.TextEncoderPath is not null && options.TextEncoderPath.Length > 0)
            {
                textEncoder = new OnnxModel<T>(options.TextEncoderPath, options.OnnxOptions);
            }
            else
            {
                var path = await downloader.DownloadAsync(
                    modelRepo,
                    "text_encoder.onnx",
                    progress: new Progress<double>(p => progress?.Report(p * 0.33)),
                    cancellationToken);
                textEncoder = new OnnxModel<T>(path, options.OnnxOptions);
            }

            // Load language model
            if (options.LanguageModelPath is not null && options.LanguageModelPath.Length > 0)
            {
                languageModel = new OnnxModel<T>(options.LanguageModelPath, options.OnnxOptions);
            }
            else
            {
                var path = await downloader.DownloadAsync(
                    modelRepo,
                    "language_model.onnx",
                    progress: new Progress<double>(p => progress?.Report(0.33 + p * 0.33)),
                    cancellationToken);
                languageModel = new OnnxModel<T>(path, options.OnnxOptions);
            }

            // Load audio decoder (EnCodec)
            if (options.AudioCodecPath is not null && options.AudioCodecPath.Length > 0)
            {
                audioDecoder = new OnnxModel<T>(options.AudioCodecPath, options.OnnxOptions);
            }
            else
            {
                var path = await downloader.DownloadAsync(
                    modelRepo,
                    "audio_decoder.onnx",
                    progress: new Progress<double>(p => progress?.Report(0.66 + p * 0.34)),
                    cancellationToken);
                audioDecoder = new OnnxModel<T>(path, options.OnnxOptions);
            }

            return new AudioGenModel<T>(options, textEncoder, languageModel, audioDecoder);
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
    /// Creates an AudioGenModel from local model files.
    /// </summary>
    public static AudioGenModel<T> FromFiles(
        string textEncoderPath,
        string languageModelPath,
        string audioDecoderPath,
        AudioGenOptions? options = null)
    {
        options ??= new AudioGenOptions();
        options.TextEncoderPath = textEncoderPath;
        options.LanguageModelPath = languageModelPath;
        options.AudioCodecPath = audioDecoderPath;

        var textEncoder = new OnnxModel<T>(textEncoderPath, options.OnnxOptions);
        var languageModel = new OnnxModel<T>(languageModelPath, options.OnnxOptions);
        var audioDecoder = new OnnxModel<T>(audioDecoderPath, options.OnnxOptions);

        return new AudioGenModel<T>(options, textEncoder, languageModel, audioDecoder);
    }

    /// <summary>
    /// Generates audio from a text description.
    /// </summary>
    /// <param name="prompt">The text description of the audio to generate.</param>
    /// <returns>The generated audio result.</returns>
    public AudioGenResult<T> Generate(string prompt)
    {
        ThrowIfDisposed();

        var stopwatch = Stopwatch.StartNew();
        int seedUsed = _options.Seed ?? _random.Next();

        // Encode the text prompt
        var textEmbeddings = EncodeText(prompt);

        // Generate audio codes using language model
        var audioCodes = GenerateAudioCodes(textEmbeddings, seedUsed);

        // Decode audio codes to waveform
        var audio = DecodeAudio(audioCodes);

        stopwatch.Stop();

        return new AudioGenResult<T>
        {
            Audio = audio,
            SampleRate = _options.SampleRate,
            Duration = (double)audio.Length / _options.SampleRate,
            Prompt = prompt,
            SeedUsed = seedUsed,
            ProcessingTimeMs = stopwatch.ElapsedMilliseconds
        };
    }

    /// <summary>
    /// Generates audio from a text description asynchronously.
    /// </summary>
    public Task<AudioGenResult<T>> GenerateAsync(
        string prompt,
        CancellationToken cancellationToken = default)
    {
        return Task.Run(() => Generate(prompt), cancellationToken);
    }

    /// <summary>
    /// Generates audio with classifier-free guidance.
    /// </summary>
    /// <param name="prompt">The text description.</param>
    /// <param name="negativePrompt">What to avoid in generation (optional).</param>
    /// <returns>The generated audio result.</returns>
    public AudioGenResult<T> GenerateWithGuidance(
        string prompt,
        string? negativePrompt = null)
    {
        ThrowIfDisposed();

        var stopwatch = Stopwatch.StartNew();
        int seedUsed = _options.Seed ?? _random.Next();

        // Encode prompts
        var textEmbeddings = EncodeText(prompt);
        var negativeEmbeddings = negativePrompt is null || negativePrompt.Length == 0
            ? CreateUnconditionalEmbeddings(textEmbeddings.Shape)
            : EncodeText(negativePrompt);

        // Generate with classifier-free guidance
        var audioCodes = GenerateAudioCodesWithGuidance(
            textEmbeddings,
            negativeEmbeddings,
            _options.GuidanceScale,
            seedUsed);

        // Decode audio codes to waveform
        var audio = DecodeAudio(audioCodes);

        stopwatch.Stop();

        return new AudioGenResult<T>
        {
            Audio = audio,
            SampleRate = _options.SampleRate,
            Duration = (double)audio.Length / _options.SampleRate,
            Prompt = prompt,
            SeedUsed = seedUsed,
            ProcessingTimeMs = stopwatch.ElapsedMilliseconds
        };
    }

    private Tensor<T> EncodeText(string prompt)
    {
        if (_textEncoder is null)
            throw new InvalidOperationException("Text encoder not loaded.");

        // Tokenize the prompt (simplified - real impl would use proper tokenizer)
        var tokens = TokenizePrompt(prompt);

        // Create input tensor
        var inputTensor = new Tensor<T>([1, tokens.Length]);
        for (int i = 0; i < tokens.Length; i++)
        {
            inputTensor[0, i] = _numOps.FromDouble(tokens[i]);
        }

        // Run text encoder
        return _textEncoder.Run(inputTensor);
    }

    private int[] TokenizePrompt(string prompt)
    {
        // Simplified tokenization - real implementation would use proper BPE tokenizer
        var tokens = new List<int>();

        // Add start token
        tokens.Add(1);

        // Simple character-level tokenization as placeholder
        foreach (char c in prompt.ToLowerInvariant())
        {
            if (char.IsLetterOrDigit(c))
            {
                tokens.Add(c - 'a' + 10);
            }
            else if (c == ' ')
            {
                tokens.Add(3);
            }
        }

        // Add end token
        tokens.Add(2);

        // Pad or truncate to fixed length
        int maxLength = 256;
        while (tokens.Count < maxLength)
        {
            tokens.Add(0); // Pad token
        }

        return tokens.Take(maxLength).ToArray();
    }

    private Tensor<T> CreateUnconditionalEmbeddings(int[] shape)
    {
        // Create zero embeddings for unconditional generation
        var embeddings = new Tensor<T>(shape);
        // Tensor is initialized to zeros by default
        return embeddings;
    }

    private Tensor<T> GenerateAudioCodes(Tensor<T> textEmbeddings, int seed)
    {
        if (_languageModel is null)
            throw new InvalidOperationException("Language model not loaded.");

        var random = RandomHelper.CreateSeededRandom(seed);

        // Calculate number of tokens based on duration
        int numCodebooks = 4; // EnCodec uses 4 codebooks typically
        int tokensPerSecond = 50; // 50 tokens per second at 32kHz
        int numTokens = (int)(_options.DurationSeconds * tokensPerSecond);

        var codes = new Tensor<T>([1, numCodebooks, numTokens]);

        // Autoregressive generation
        var currentTokens = new Tensor<T>([1, numCodebooks, 1]);

        // Initialize with start token
        for (int cb = 0; cb < numCodebooks; cb++)
        {
            currentTokens[0, cb, 0] = _numOps.FromDouble(0);
        }

        for (int t = 0; t < numTokens; t++)
        {
            // Prepare inputs
            var inputs = new Dictionary<string, Tensor<T>>
            {
                ["text_embeddings"] = textEmbeddings,
                ["audio_codes"] = currentTokens
            };

            // Get logits from language model
            var outputs = _languageModel.Run(inputs);
            var logits = outputs.Values.First();

            // Sample next tokens for each codebook
            for (int cb = 0; cb < numCodebooks; cb++)
            {
                int nextToken = SampleFromLogits(logits, cb, random);
                codes[0, cb, t] = _numOps.FromDouble(nextToken);

                // Update current tokens for next iteration
                if (t < numTokens - 1)
                {
                    var newCurrentTokens = new Tensor<T>([1, numCodebooks, currentTokens.Shape[2] + 1]);
                    for (int c = 0; c < numCodebooks; c++)
                    {
                        for (int i = 0; i < currentTokens.Shape[2]; i++)
                        {
                            newCurrentTokens[0, c, i] = currentTokens[0, c, i];
                        }
                        newCurrentTokens[0, c, currentTokens.Shape[2]] = codes[0, c, t];
                    }
                    currentTokens = newCurrentTokens;
                }
            }
        }

        return codes;
    }

    private Tensor<T> GenerateAudioCodesWithGuidance(
        Tensor<T> condEmbeddings,
        Tensor<T> uncondEmbeddings,
        double guidanceScale,
        int seed)
    {
        if (_languageModel is null)
            throw new InvalidOperationException("Language model not loaded.");

        var random = RandomHelper.CreateSeededRandom(seed);

        int numCodebooks = 4;
        int tokensPerSecond = 50;
        int numTokens = (int)(_options.DurationSeconds * tokensPerSecond);

        var codes = new Tensor<T>([1, numCodebooks, numTokens]);
        var currentTokens = new Tensor<T>([1, numCodebooks, 1]);

        for (int cb = 0; cb < numCodebooks; cb++)
        {
            currentTokens[0, cb, 0] = _numOps.FromDouble(0);
        }

        for (int t = 0; t < numTokens; t++)
        {
            // Get conditional logits
            var condInputs = new Dictionary<string, Tensor<T>>
            {
                ["text_embeddings"] = condEmbeddings,
                ["audio_codes"] = currentTokens
            };
            var condOutputs = _languageModel.Run(condInputs);
            var condLogits = condOutputs.Values.First();

            // Get unconditional logits
            var uncondInputs = new Dictionary<string, Tensor<T>>
            {
                ["text_embeddings"] = uncondEmbeddings,
                ["audio_codes"] = currentTokens
            };
            var uncondOutputs = _languageModel.Run(uncondInputs);
            var uncondLogits = uncondOutputs.Values.First();

            // Apply classifier-free guidance
            var guidedLogits = ApplyGuidance(condLogits, uncondLogits, guidanceScale);

            // Sample next tokens
            for (int cb = 0; cb < numCodebooks; cb++)
            {
                int nextToken = SampleFromLogits(guidedLogits, cb, random);
                codes[0, cb, t] = _numOps.FromDouble(nextToken);

                if (t < numTokens - 1)
                {
                    var newCurrentTokens = new Tensor<T>([1, numCodebooks, currentTokens.Shape[2] + 1]);
                    for (int c = 0; c < numCodebooks; c++)
                    {
                        for (int i = 0; i < currentTokens.Shape[2]; i++)
                        {
                            newCurrentTokens[0, c, i] = currentTokens[0, c, i];
                        }
                        newCurrentTokens[0, c, currentTokens.Shape[2]] = codes[0, c, t];
                    }
                    currentTokens = newCurrentTokens;
                }
            }
        }

        return codes;
    }

    private Tensor<T> ApplyGuidance(Tensor<T> condLogits, Tensor<T> uncondLogits, double scale)
    {
        var guided = new Tensor<T>(condLogits.Shape);

        for (int i = 0; i < condLogits.Length; i++)
        {
            double cond = _numOps.ToDouble(condLogits[i]);
            double uncond = _numOps.ToDouble(uncondLogits[i]);

            // CFG formula: guided = uncond + scale * (cond - uncond)
            double guidedValue = uncond + scale * (cond - uncond);
            guided[i] = _numOps.FromDouble(guidedValue);
        }

        return guided;
    }

    private int SampleFromLogits(Tensor<T> logits, int codebook, Random random)
    {
        // Get logits for this codebook (assuming shape [batch, codebooks, vocab] or similar)
        int vocabSize = 1024; // Typical EnCodec codebook size

        // Apply temperature
        var scaledLogits = new double[vocabSize];
        for (int i = 0; i < vocabSize; i++)
        {
            // Simplified: assume logits are at end dimension
            int idx = codebook * vocabSize + i;
            if (idx < logits.Length)
            {
                scaledLogits[i] = _numOps.ToDouble(logits[idx]) / _options.Temperature;
            }
        }

        // Apply top-k filtering
        if (_options.TopK > 0 && _options.TopK < vocabSize)
        {
            var sorted = scaledLogits
                .Select((v, i) => (Value: v, Index: i))
                .OrderByDescending(x => x.Value)
                .ToList();

            double threshold = sorted[Math.Min(_options.TopK - 1, sorted.Count - 1)].Value;
            for (int i = 0; i < vocabSize; i++)
            {
                if (scaledLogits[i] < threshold)
                {
                    scaledLogits[i] = double.NegativeInfinity;
                }
            }
        }

        // Apply top-p (nucleus) filtering
        if (_options.TopP > 0 && _options.TopP < 1.0)
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
                if (cumSum >= _options.TopP)
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

        return vocabSize - 1; // Fallback
    }

    private static double[] Softmax(double[] logits)
    {
        double maxLogit = logits.Where(x => !double.IsNegativeInfinity(x)).DefaultIfEmpty(0).Max();
        var expValues = logits.Select(x => double.IsNegativeInfinity(x) ? 0 : Math.Exp(x - maxLogit)).ToArray();
        double sumExp = expValues.Sum();

        if (sumExp == 0) sumExp = 1; // Avoid division by zero

        return expValues.Select(x => x / sumExp).ToArray();
    }

    private T[] DecodeAudio(Tensor<T> audioCodes)
    {
        if (_audioDecoder is null)
            throw new InvalidOperationException("Audio decoder not loaded.");

        // Run EnCodec decoder
        var waveformTensor = _audioDecoder.Run(audioCodes);

        // Extract audio samples
        var audio = waveformTensor.ToArray();

        // Trim to target duration
        int targetSamples = (int)(_options.DurationSeconds * _options.SampleRate);
        if (audio.Length > targetSamples)
        {
            audio = audio.Take(targetSamples).ToArray();
        }

        return audio;
    }

    private static string GetModelRepository(AudioGenModelSize modelSize)
    {
        return modelSize switch
        {
            AudioGenModelSize.Small => "facebook/audiogen-small",
            AudioGenModelSize.Medium => "facebook/audiogen-medium",
            AudioGenModelSize.Large => "facebook/audiogen-large",
            _ => "facebook/audiogen-medium"
        };
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName);
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
            _textEncoder?.Dispose();
            _languageModel?.Dispose();
            _audioDecoder?.Dispose();
        }

        _disposed = true;
    }
}
