using System.Diagnostics;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Tensors.Helpers;
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
/// var result = audioGen.GenerateAudio("A dog barking in the distance");
/// SaveAudio(result.ToArray(), audioGen.SampleRate);  // Your audio saving code
///
/// audioGen.Dispose();
/// </code>
/// </para>
/// </remarks>
public class AudioGenModel<T> : AudioNeuralNetworkBase<T>, IAudioGenerator<T>
{
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
    /// Gets the maximum duration of audio that can be generated in seconds.
    /// </summary>
    public double MaxDurationSeconds => _options.MaxDurationSeconds;

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

    /// <summary>
    /// Creates a new AudioGenModel instance.
    /// </summary>
    private AudioGenModel(
        AudioGenOptions options,
        OnnxModel<T>? textEncoder,
        OnnxModel<T>? languageModel,
        OnnxModel<T>? audioDecoder)
        : base(CreateMinimalArchitecture(options))
    {
        _options = options;
        _textEncoder = textEncoder;
        _languageModel = languageModel;
        _audioDecoder = audioDecoder;
        _random = options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(options.Seed.Value)
            : RandomHelper.CreateSecureRandom();

        // Store audio decoder as OnnxModel for base class
        OnnxModel = audioDecoder;

        // Set audio properties
        SampleRate = options.SampleRate;
    }

    private static NeuralNetworkArchitecture<T> CreateMinimalArchitecture(AudioGenOptions options)
    {
        // Create minimal architecture for ONNX-only model
        return new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Generative,
            complexity: NetworkComplexity.VeryDeep,
            inputSize: 256, // Max text token sequence length
            outputSize: options.SampleRate * (int)options.MaxDurationSeconds
        );
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

    #region IAudioGenerator Implementation

    /// <summary>
    /// Generates audio from a text description.
    /// </summary>
    public Tensor<T> GenerateAudio(
        string prompt,
        string? negativePrompt = null,
        double durationSeconds = 5.0,
        int numInferenceSteps = 100,
        double guidanceScale = 3.0,
        int? seed = null)
    {
        ThrowIfDisposed();

        int seedUsed = seed ?? _options.Seed ?? _random.Next();

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
        throw new NotSupportedException("Music generation is not supported by this AudioGen model.");
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
    /// Gets generation options for advanced control.
    /// </summary>
    public AudioGenerationOptions<T> GetDefaultOptions()
    {
        return new AudioGenerationOptions<T>
        {
            DurationSeconds = _options.DurationSeconds,
            NumInferenceSteps = 100,
            GuidanceScale = _options.GuidanceScale,
            Seed = _options.Seed,
            Stereo = false,
            SchedulerType = "ddpm"
        };
    }

    #endregion

    #region AudioNeuralNetworkBase Implementation

    /// <summary>
    /// Preprocesses raw audio for model input.
    /// </summary>
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio)
    {
        // For AudioGen, we typically work with text input
        // This method is used for audio continuation scenarios
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
        // For AudioGen, prediction requires text encoding first
        if (_textEncoder is null)
            throw new InvalidOperationException("Text encoder not loaded.");

        return _textEncoder.Run(input);
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
        throw new NotImplementedException("Native parameter updates for AudioGen are not yet implemented.");
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

        throw new NotImplementedException("Native training for AudioGen is not yet implemented.");
    }

    /// <summary>
    /// Gets metadata about the model.
    /// </summary>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = $"AudioGen-{_options.ModelSize}",
            Description = $"AudioGen model for text-to-audio generation - {_options.ModelSize} variant",
            ModelType = ModelType.NeuralNetwork,
            FeatureCount = 256,
            Complexity = (int)_options.ModelSize
        };
        metadata.AdditionalInfo["InputFormat"] = "Text Prompt";
        metadata.AdditionalInfo["OutputFormat"] = $"Audio ({SampleRate}Hz, up to {MaxDurationSeconds}s)";
        return metadata;
    }

    /// <summary>
    /// Serializes network-specific data.
    /// </summary>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_options.SampleRate);
        writer.Write(_options.DurationSeconds);
        writer.Write(_options.Temperature);
        writer.Write(_options.GuidanceScale);
        writer.Write((int)_options.ModelSize);
    }

    /// <summary>
    /// Deserializes network-specific data.
    /// </summary>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _options.SampleRate = reader.ReadInt32();
        _options.DurationSeconds = reader.ReadDouble();
        _options.Temperature = reader.ReadDouble();
        _options.GuidanceScale = reader.ReadDouble();
        _options.ModelSize = (AudioGenModelSize)reader.ReadInt32();
    }

    /// <summary>
    /// Creates a new instance of this model for cloning.
    /// </summary>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new AudioGenModel<T>(_options, null, null, null);
    }

    #endregion

    #region Private Methods

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
            inputTensor[0, i] = NumOps.FromDouble(tokens[i]);
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

    private Tensor<T> GenerateAudioCodes(Tensor<T> textEmbeddings, int seed, double durationSeconds)
    {
        if (_languageModel is null)
            throw new InvalidOperationException("Language model not loaded.");

        var random = RandomHelper.CreateSeededRandom(seed);

        // Calculate number of tokens based on duration
        int numCodebooks = 4; // EnCodec uses 4 codebooks typically
        int tokensPerSecond = 50; // 50 tokens per second at 32kHz
        int numTokens = (int)(durationSeconds * tokensPerSecond);

        var codes = new Tensor<T>([1, numCodebooks, numTokens]);

        // Autoregressive generation
        var currentTokens = new Tensor<T>([1, numCodebooks, 1]);

        // Initialize with start token
        for (int cb = 0; cb < numCodebooks; cb++)
        {
            currentTokens[0, cb, 0] = NumOps.FromDouble(0);
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
                codes[0, cb, t] = NumOps.FromDouble(nextToken);

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
        int seed,
        double durationSeconds)
    {
        if (_languageModel is null)
            throw new InvalidOperationException("Language model not loaded.");

        var random = RandomHelper.CreateSeededRandom(seed);

        int numCodebooks = 4;
        int tokensPerSecond = 50;
        int numTokens = (int)(durationSeconds * tokensPerSecond);

        var codes = new Tensor<T>([1, numCodebooks, numTokens]);
        var currentTokens = new Tensor<T>([1, numCodebooks, 1]);

        for (int cb = 0; cb < numCodebooks; cb++)
        {
            currentTokens[0, cb, 0] = NumOps.FromDouble(0);
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
                codes[0, cb, t] = NumOps.FromDouble(nextToken);

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
                scaledLogits[i] = NumOps.ToDouble(logits[idx]) / _options.Temperature;
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

    private Tensor<T> DecodeAudio(Tensor<T> audioCodes, double durationSeconds)
    {
        if (_audioDecoder is null)
            throw new InvalidOperationException("Audio decoder not loaded.");

        // Run EnCodec decoder
        var waveformTensor = _audioDecoder.Run(audioCodes);

        // Trim to target duration
        int targetSamples = (int)(durationSeconds * _options.SampleRate);
        if (waveformTensor.Length > targetSamples)
        {
            var trimmed = new Tensor<T>([targetSamples]);
            for (int i = 0; i < targetSamples; i++)
            {
                trimmed[i] = waveformTensor[i];
            }
            return trimmed;
        }

        return waveformTensor;
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
