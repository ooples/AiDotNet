using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.Audio.MusicGen;

/// <summary>
/// Meta's MusicGen model for generating music from text descriptions.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// MusicGen is a state-of-the-art text-to-music generation model from Meta AI Research.
/// It uses a single-stage transformer language model that operates directly on EnCodec
/// audio codes, generating high-quality music from text descriptions.
/// </para>
/// <para>
/// Architecture components:
/// <list type="number">
/// <item><description><b>Text Encoder:</b> T5-based encoder that converts text prompts to embeddings</description></item>
/// <item><description><b>Language Model:</b> Transformer decoder that generates audio codes autoregressively</description></item>
/// <item><description><b>EnCodec Decoder:</b> Neural audio codec that converts discrete codes to waveforms</description></item>
/// </list>
/// </para>
/// <para><b>For Beginners:</b> MusicGen creates original music from your descriptions:
///
/// How it works:
/// 1. You describe the music you want ("upbeat jazz piano")
/// 2. The text encoder understands your description
/// 3. The language model generates a sequence of "music tokens"
/// 4. The EnCodec decoder converts tokens to actual audio
///
/// Key features:
/// - 30 seconds of high-quality 32kHz audio
/// - Multiple genres and styles
/// - Control over instruments, tempo, mood
/// - Stereo output option
///
/// Usage:
/// <code>
/// var model = new MusicGenModel&lt;float&gt;(options);
/// var audio = model.GenerateMusic("Calm piano melody with soft strings");
/// </code>
/// </para>
/// <para>
/// Reference: "Simple and Controllable Music Generation" by Copet et al., Meta AI, 2023
/// </para>
/// </remarks>
public class MusicGenModel<T> : AudioNeuralNetworkBase<T>, IAudioGenerator<T>
{
    #region Fields

    private readonly MusicGenOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private readonly ITokenizer _tokenizer;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private ILossFunction<T> _lossFunction;
    private readonly Random _random;

    // Model dimensions based on size
    private readonly int _textHiddenDim;
    private readonly int _lmHiddenDim;
    private readonly int _numLmLayers;
    private readonly int _numHeads;

    // ONNX models for inference mode
    private readonly OnnxModel<T>? _textEncoder;
    private readonly OnnxModel<T>? _languageModel;
    private readonly OnnxModel<T>? _encodecDecoder;
    private readonly bool _useNativeMode;

    private bool _disposed;

    #endregion

    #region IAudioGenerator Properties

    /// <summary>
    /// Gets the sample rate of generated audio.
    /// </summary>
    public new int SampleRate => _options.SampleRate;

    /// <summary>
    /// Gets the maximum duration of audio that can be generated.
    /// </summary>
    public double MaxDurationSeconds => _options.MaxDurationSeconds;

    /// <summary>
    /// Gets whether this model supports text-to-audio generation.
    /// </summary>
    public bool SupportsTextToAudio => false; // MusicGen is specifically for music

    /// <summary>
    /// Gets whether this model supports text-to-music generation.
    /// </summary>
    public bool SupportsTextToMusic => true;

    /// <summary>
    /// Gets whether this model supports audio continuation.
    /// </summary>
    public bool SupportsAudioContinuation => true; // MusicGen supports melody conditioning

    /// <summary>
    /// Gets whether this model supports audio inpainting.
    /// </summary>
    public bool SupportsAudioInpainting => false;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a MusicGen model using pretrained ONNX models for inference.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="textEncoderPath">Path to the T5 text encoder ONNX model.</param>
    /// <param name="languageModelPath">Path to the transformer LM ONNX model.</param>
    /// <param name="encodecDecoderPath">Path to the EnCodec decoder ONNX model.</param>
    /// <param name="tokenizer">T5 tokenizer for text processing (REQUIRED).</param>
    /// <param name="options">MusicGen configuration options.</param>
    /// <param name="optimizer">Optional optimizer for fine-tuning.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <exception cref="ArgumentException">Thrown when required paths are empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown when model files don't exist.</exception>
    /// <exception cref="ArgumentNullException">Thrown when tokenizer is null.</exception>
    public MusicGenModel(
        NeuralNetworkArchitecture<T> architecture,
        string textEncoderPath,
        string languageModelPath,
        string encodecDecoderPath,
        ITokenizer tokenizer,
        MusicGenOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>(), 1.0)
    {
        // Validate paths
        if (string.IsNullOrWhiteSpace(textEncoderPath))
            throw new ArgumentException("Text encoder path is required.", nameof(textEncoderPath));
        if (string.IsNullOrWhiteSpace(languageModelPath))
            throw new ArgumentException("Language model path is required.", nameof(languageModelPath));
        if (string.IsNullOrWhiteSpace(encodecDecoderPath))
            throw new ArgumentException("EnCodec decoder path is required.", nameof(encodecDecoderPath));
        if (!File.Exists(textEncoderPath))
            throw new FileNotFoundException($"Text encoder not found: {textEncoderPath}");
        if (!File.Exists(languageModelPath))
            throw new FileNotFoundException($"Language model not found: {languageModelPath}");
        if (!File.Exists(encodecDecoderPath))
            throw new FileNotFoundException($"EnCodec decoder not found: {encodecDecoderPath}");

        _options = options ?? new MusicGenOptions();
        Options = _options;
        Guard.NotNull(tokenizer);
        _tokenizer = tokenizer;
        _useNativeMode = false;

        // Set dimensions based on model size
        (_textHiddenDim, _lmHiddenDim, _numLmLayers, _numHeads) = GetModelDimensions(_options.ModelSize);

        // Load ONNX models
        OnnxModel<T>? textEncoder = null;
        OnnxModel<T>? languageModel = null;
        OnnxModel<T>? encodecDecoder = null;

        try
        {
            textEncoder = new OnnxModel<T>(textEncoderPath);
            languageModel = new OnnxModel<T>(languageModelPath);
            encodecDecoder = new OnnxModel<T>(encodecDecoderPath);

            _textEncoder = textEncoder;
            _languageModel = languageModel;
            _encodecDecoder = encodecDecoder;

            // Set base class ONNX model
            OnnxModel = encodecDecoder;
        }
        catch
        {
            textEncoder?.Dispose();
            languageModel?.Dispose();
            encodecDecoder?.Dispose();
            throw;
        }

        _optimizer = optimizer;
        _lossFunction = lossFunction ?? new CrossEntropyLoss<T>();
        _random = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSecureRandom();

        base.SampleRate = _options.SampleRate;

        InitializeLayers();
    }

    /// <summary>
    /// Creates a MusicGen model using native layers for training from scratch.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="options">MusicGen configuration options.</param>
    /// <param name="tokenizer">Optional tokenizer. If null, creates T5-compatible tokenizer.</param>
    /// <param name="optimizer">Optional optimizer. Defaults to AdamW.</param>
    /// <param name="lossFunction">Optional loss function. Defaults to CrossEntropy.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor when:
    /// - Training MusicGen from scratch (requires significant data)
    /// - Fine-tuning on custom music styles
    /// - Research and experimentation
    ///
    /// For most use cases, load pretrained ONNX models instead.
    /// </para>
    /// </remarks>
    public MusicGenModel(
        NeuralNetworkArchitecture<T> architecture,
        MusicGenOptions? options = null,
        ITokenizer? tokenizer = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>(), 1.0)
    {
        _options = options ?? new MusicGenOptions();
        Options = _options;
        _useNativeMode = true;

        // Set dimensions based on model size
        (_textHiddenDim, _lmHiddenDim, _numLmLayers, _numHeads) = GetModelDimensions(_options.ModelSize);

        // Use T5-compatible tokenizer as default
        _tokenizer = tokenizer ?? Tokenization.LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.FlanT5);
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new CrossEntropyLoss<T>();
        _random = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSecureRandom();

        base.SampleRate = _options.SampleRate;

        InitializeLayers();
    }

    #endregion

    #region Layer Initialization - Golden Standard Pattern

    /// <summary>
    /// Initializes the neural network layers following the golden standard pattern.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method follows the AiDotNet golden standard pattern:
    /// 1. First, check if the user provided custom layers via Architecture.Layers
    /// 2. If custom layers exist, use them (allows full customization)
    /// 3. Otherwise, use LayerHelper.CreateDefaultMusicGenLayers() for standard architecture
    /// </para>
    /// <para><b>For Beginners:</b> This gives you flexibility:
    /// - Want standard MusicGen? Just create the model, it auto-configures.
    /// - Want custom architecture? Pass your own layers in the Architecture.
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        // ONNX mode - layers are handled by ONNX runtime
        if (!_useNativeMode)
        {
            return;
        }

        // Golden standard pattern: check for user-provided custom layers first
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            // Use custom layers provided by the user
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
        }
        else
        {
            // Use default MusicGen layer configuration
            Layers.AddRange(LayerHelper<T>.CreateDefaultMusicGenLayers(
                textHiddenDim: _textHiddenDim,
                lmHiddenDim: _lmHiddenDim,
                numLmLayers: _numLmLayers,
                numHeads: _numHeads,
                numCodebooks: _options.NumCodebooks,
                codebookSize: _options.CodebookSize,
                maxTextLength: _options.MaxTextLength,
                maxAudioTokens: (int)(_options.MaxDurationSeconds * 50), // ~50 tokens/sec
                dropoutRate: _options.DropoutRate));
        }
    }

    private static (int textHiddenDim, int lmHiddenDim, int numLmLayers, int numHeads) GetModelDimensions(MusicGenModelSize size)
    {
        return size switch
        {
            MusicGenModelSize.Small => (256, 1024, 24, 16),
            MusicGenModelSize.Medium => (768, 1536, 24, 16),
            MusicGenModelSize.Large => (1024, 2048, 48, 16),
            MusicGenModelSize.Melody => (768, 1536, 24, 16),
            MusicGenModelSize.Stereo => (768, 1536, 24, 16),
            _ => (768, 1536, 24, 16)
        };
    }

    #endregion

    #region IAudioGenerator Implementation

    /// <summary>
    /// Generates audio from a text description.
    /// </summary>
    /// <remarks>
    /// MusicGen is optimized for music, not general audio.
    /// For best results, use GenerateMusic instead.
    /// </remarks>
    public Tensor<T> GenerateAudio(
        string prompt,
        string? negativePrompt = null,
        double durationSeconds = 5.0,
        int numInferenceSteps = 100,
        double guidanceScale = 3.0,
        int? seed = null)
    {
        // Redirect to music generation
        return GenerateMusic(prompt, negativePrompt, durationSeconds, numInferenceSteps, guidanceScale, seed);
    }

    /// <summary>
    /// Generates audio asynchronously.
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
    /// <param name="prompt">Text description of the desired music.</param>
    /// <param name="negativePrompt">What to avoid in the generated music.</param>
    /// <param name="durationSeconds">Duration of music to generate (max 30s).</param>
    /// <param name="numInferenceSteps">Not used in autoregressive generation.</param>
    /// <param name="guidanceScale">How closely to follow the prompt.</param>
    /// <param name="seed">Random seed for reproducibility.</param>
    /// <returns>Generated music waveform tensor.</returns>
    public Tensor<T> GenerateMusic(
        string prompt,
        string? negativePrompt = null,
        double durationSeconds = 10.0,
        int numInferenceSteps = 100,
        double guidanceScale = 3.0,
        int? seed = null)
    {
        ThrowIfDisposed();

        // Clamp duration to maximum
        durationSeconds = Math.Min(durationSeconds, _options.MaxDurationSeconds);

        int seedUsed = seed ?? _random.Next();
        var random = RandomHelper.CreateSeededRandom(seedUsed);

        // Encode text prompt
        var textEmbeddings = EncodeText(prompt);

        // Generate audio codes with optional classifier-free guidance
        Tensor<T> audioCodes;
        if (negativePrompt is string negPrompt && !string.IsNullOrEmpty(negPrompt) && guidanceScale > 1.0)
        {
            var negativeEmbeddings = EncodeText(negPrompt);
            audioCodes = GenerateWithGuidance(textEmbeddings, negativeEmbeddings, guidanceScale, durationSeconds, random);
        }
        else
        {
            audioCodes = GenerateAudioCodes(textEmbeddings, durationSeconds, random);
        }

        // Decode to waveform
        return DecodeToWaveform(audioCodes, durationSeconds);
    }

    /// <summary>
    /// Continues existing audio by extending it.
    /// </summary>
    /// <param name="inputAudio">Audio to continue from.</param>
    /// <param name="prompt">Optional text guidance for continuation.</param>
    /// <param name="extensionSeconds">How many seconds to add.</param>
    /// <param name="numInferenceSteps">Not used.</param>
    /// <param name="seed">Random seed.</param>
    /// <returns>Extended audio (original + continuation).</returns>
    public Tensor<T> ContinueAudio(
        Tensor<T> inputAudio,
        string? prompt = null,
        double extensionSeconds = 5.0,
        int numInferenceSteps = 100,
        int? seed = null)
    {
        ThrowIfDisposed();

        int seedUsed = seed ?? _random.Next();
        var random = RandomHelper.CreateSeededRandom(seedUsed);

        // Encode the input audio to get conditioning
        var inputCodes = EncodeAudioToCodes(inputAudio);

        // Encode text prompt if provided
        Tensor<T>? textEmbeddings = null;
        if (prompt is string textPrompt && !string.IsNullOrEmpty(textPrompt))
        {
            textEmbeddings = EncodeText(textPrompt);
        }

        // Generate continuation codes
        var continuationCodes = GenerateContinuation(inputCodes, textEmbeddings, extensionSeconds, random);

        // Concatenate and decode
        var allCodes = ConcatenateCodes(inputCodes, continuationCodes);
        double totalDuration = (double)inputAudio.Length / _options.SampleRate + extensionSeconds;

        return DecodeToWaveform(allCodes, totalDuration);
    }

    /// <summary>
    /// Inpainting is not supported by MusicGen.
    /// </summary>
    public Tensor<T> InpaintAudio(
        Tensor<T> audio,
        Tensor<T> mask,
        string? prompt = null,
        int numInferenceSteps = 100,
        int? seed = null)
    {
        throw new NotSupportedException("MusicGen does not support audio inpainting.");
    }

    /// <summary>
    /// Gets default generation options.
    /// </summary>
    public AudioGenerationOptions<T> GetDefaultOptions()
    {
        return new AudioGenerationOptions<T>
        {
            DurationSeconds = _options.DurationSeconds,
            NumInferenceSteps = 100,
            GuidanceScale = _options.GuidanceScale,
            Seed = _options.Seed,
            Stereo = _options.Stereo,
            SchedulerType = "autoregressive"
        };
    }

    #endregion

    #region Core Generation Methods

    private Tensor<T> EncodeText(string prompt)
    {
        // Tokenize the prompt
        var encoding = _tokenizer.Encode(prompt);
        var tokens = new int[_options.MaxTextLength];
        int copyCount = Math.Min(encoding.TokenIds.Count, _options.MaxTextLength);
        for (int i = 0; i < copyCount; i++)
        {
            tokens[i] = encoding.TokenIds[i];
        }

        // Create input tensor
        var inputTensor = new Tensor<T>([1, _options.MaxTextLength]);
        for (int i = 0; i < _options.MaxTextLength; i++)
        {
            inputTensor[0, i] = NumOps.FromDouble(tokens[i]);
        }

        if (!_useNativeMode && _textEncoder is not null)
        {
            return _textEncoder.Run(inputTensor);
        }

        // Native mode: forward through text encoder layers
        return ForwardTextEncoder(inputTensor);
    }

    private Tensor<T> ForwardTextEncoder(Tensor<T> input)
    {
        // Text encoder uses the first portion of layers
        // This is a simplified approach - real implementation would track layer indices
        var output = input;
        int textEncoderLayerCount = 6 * 5 + 3; // 6 blocks * 5 layers each + embedding/position layers

        for (int i = 0; i < Math.Min(textEncoderLayerCount, Layers.Count); i++)
        {
            output = Layers[i].Forward(output);
        }

        return output;
    }

    private Tensor<T> GenerateAudioCodes(Tensor<T> textEmbeddings, double durationSeconds, Random random)
    {
        // Calculate number of tokens needed (~50 tokens/second for EnCodec at 32kHz)
        int tokensPerSecond = 50;
        int numTokens = (int)(durationSeconds * tokensPerSecond);

        var codes = new Tensor<T>([1, _options.NumCodebooks, numTokens]);

        // Initialize with start token
        var currentTokens = new Tensor<T>([1, _options.NumCodebooks, 1]);
        for (int cb = 0; cb < _options.NumCodebooks; cb++)
        {
            currentTokens[0, cb, 0] = NumOps.Zero; // Start token
        }

        // Generate tokens autoregressively with delay pattern
        for (int t = 0; t < numTokens; t++)
        {
            Tensor<T> logits;

            if (!_useNativeMode && _languageModel is not null)
            {
                var inputs = new Dictionary<string, Tensor<T>>
                {
                    ["text_embeddings"] = textEmbeddings,
                    ["audio_codes"] = currentTokens
                };
                var outputs = _languageModel.Run(inputs);
                logits = outputs.Values.First();
            }
            else
            {
                logits = ForwardLanguageModel(textEmbeddings, currentTokens);
            }

            // Sample next token for each codebook with delay pattern
            for (int cb = 0; cb < _options.NumCodebooks; cb++)
            {
                // Delay pattern: codebook i is delayed by i tokens
                int effectiveT = t - cb;
                if (effectiveT < 0) continue;
                if (effectiveT >= numTokens) continue;

                int nextToken = SampleFromLogits(logits, cb, random);
                codes[0, cb, effectiveT] = NumOps.FromDouble(nextToken);
            }

            // Extend current tokens
            if (t < numTokens - 1)
            {
                var newTokens = new Tensor<T>([1, _options.NumCodebooks, currentTokens.Shape[2] + 1]);
                for (int cb = 0; cb < _options.NumCodebooks; cb++)
                {
                    for (int i = 0; i < currentTokens.Shape[2]; i++)
                    {
                        newTokens[0, cb, i] = currentTokens[0, cb, i];
                    }
                    newTokens[0, cb, currentTokens.Shape[2]] = codes[0, cb, Math.Max(0, t - cb)];
                }
                currentTokens = newTokens;
            }
        }

        return codes;
    }

    private Tensor<T> GenerateWithGuidance(
        Tensor<T> condEmbeddings,
        Tensor<T> uncondEmbeddings,
        double guidanceScale,
        double durationSeconds,
        Random random)
    {
        int tokensPerSecond = 50;
        int numTokens = (int)(durationSeconds * tokensPerSecond);

        var codes = new Tensor<T>([1, _options.NumCodebooks, numTokens]);
        var currentTokens = new Tensor<T>([1, _options.NumCodebooks, 1]);

        for (int cb = 0; cb < _options.NumCodebooks; cb++)
        {
            currentTokens[0, cb, 0] = NumOps.Zero;
        }

        for (int t = 0; t < numTokens; t++)
        {
            Tensor<T> condLogits, uncondLogits;

            if (!_useNativeMode && _languageModel is not null)
            {
                var condInputs = new Dictionary<string, Tensor<T>>
                {
                    ["text_embeddings"] = condEmbeddings,
                    ["audio_codes"] = currentTokens
                };
                condLogits = _languageModel.Run(condInputs).Values.First();

                var uncondInputs = new Dictionary<string, Tensor<T>>
                {
                    ["text_embeddings"] = uncondEmbeddings,
                    ["audio_codes"] = currentTokens
                };
                uncondLogits = _languageModel.Run(uncondInputs).Values.First();
            }
            else
            {
                condLogits = ForwardLanguageModel(condEmbeddings, currentTokens);
                uncondLogits = ForwardLanguageModel(uncondEmbeddings, currentTokens);
            }

            // Apply classifier-free guidance
            var guidedLogits = ApplyGuidance(condLogits, uncondLogits, guidanceScale);

            for (int cb = 0; cb < _options.NumCodebooks; cb++)
            {
                int effectiveT = t - cb;
                if (effectiveT < 0 || effectiveT >= numTokens) continue;

                int nextToken = SampleFromLogits(guidedLogits, cb, random);
                codes[0, cb, effectiveT] = NumOps.FromDouble(nextToken);
            }

            if (t < numTokens - 1)
            {
                var newTokens = new Tensor<T>([1, _options.NumCodebooks, currentTokens.Shape[2] + 1]);
                for (int cb = 0; cb < _options.NumCodebooks; cb++)
                {
                    for (int i = 0; i < currentTokens.Shape[2]; i++)
                    {
                        newTokens[0, cb, i] = currentTokens[0, cb, i];
                    }
                    newTokens[0, cb, currentTokens.Shape[2]] = codes[0, cb, Math.Max(0, t - cb)];
                }
                currentTokens = newTokens;
            }
        }

        return codes;
    }

    private Tensor<T> ForwardLanguageModel(Tensor<T> textEmbeddings, Tensor<T> audioCodes)
    {
        // Forward through language model layers
        var output = audioCodes;

        // Skip text encoder layers and process through LM layers
        int textEncoderLayerCount = 6 * 5 + 3;
        for (int i = textEncoderLayerCount; i < Layers.Count; i++)
        {
            output = Layers[i].Forward(output);
        }

        return output;
    }

    private Tensor<T> ApplyGuidance(Tensor<T> condLogits, Tensor<T> uncondLogits, double scale)
    {
        var guided = new Tensor<T>(condLogits.Shape);

        for (int i = 0; i < condLogits.Length; i++)
        {
            double cond = NumOps.ToDouble(condLogits[i]);
            double uncond = NumOps.ToDouble(uncondLogits[i]);
            double guidedValue = uncond + scale * (cond - uncond);
            guided[i] = NumOps.FromDouble(guidedValue);
        }

        return guided;
    }

    private int SampleFromLogits(Tensor<T> logits, int codebook, Random random)
    {
        int vocabSize = _options.CodebookSize;
        var scaledLogits = new double[vocabSize];

        for (int i = 0; i < vocabSize; i++)
        {
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

        // Apply top-p filtering
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
                if (cumSum >= _options.TopP) break;
            }

            for (int i = 0; i < vocabSize; i++)
            {
                if (!keepIndices.Contains(i))
                {
                    scaledLogits[i] = double.NegativeInfinity;
                }
            }
        }

        // Sample from distribution
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

    private Tensor<T> DecodeToWaveform(Tensor<T> codes, double durationSeconds)
    {
        if (!_useNativeMode && _encodecDecoder is not null)
        {
            return _encodecDecoder.Run(codes);
        }

        // Native mode: use EnCodec-style decoding
        // This is a simplified placeholder - real EnCodec has complex architecture
        int targetSamples = (int)(durationSeconds * _options.SampleRate);
        var waveform = new Tensor<T>([_options.Stereo ? 2 : 1, targetSamples]);

        // In production, this would use the actual EnCodec decoder layers
        // For now, convert discrete codes to continuous through learned embeddings
        int numTokens = codes.Shape[2];
        int samplesPerToken = targetSamples / Math.Max(1, numTokens);

        for (int t = 0; t < numTokens && t * samplesPerToken < targetSamples; t++)
        {
            // Aggregate codebook values for this timestep
            double value = 0;
            for (int cb = 0; cb < _options.NumCodebooks; cb++)
            {
                value += NumOps.ToDouble(codes[0, cb, t]) / _options.CodebookSize;
            }
            value = (value / _options.NumCodebooks - 0.5) * 2; // Normalize to [-1, 1]

            // Apply to sample range
            for (int s = 0; s < samplesPerToken && t * samplesPerToken + s < targetSamples; s++)
            {
                int sampleIdx = t * samplesPerToken + s;
                T sampleValue = NumOps.FromDouble(value);
                waveform[0, sampleIdx] = sampleValue;
                if (_options.Stereo && waveform.Shape[0] > 1)
                {
                    waveform[1, sampleIdx] = sampleValue;
                }
            }
        }

        return waveform;
    }

    private Tensor<T> EncodeAudioToCodes(Tensor<T> audio)
    {
        // In ONNX mode, this would use an EnCodec encoder
        // For native mode, this is a simplified version
        int numSamples = audio.Length;
        int tokensPerSecond = 50;
        double durationSeconds = (double)numSamples / _options.SampleRate;
        int numTokens = (int)(durationSeconds * tokensPerSecond);

        var codes = new Tensor<T>([1, _options.NumCodebooks, numTokens]);

        // Simple quantization (real EnCodec uses RVQ)
        int samplesPerToken = numSamples / Math.Max(1, numTokens);
        for (int t = 0; t < numTokens; t++)
        {
            double sum = 0;
            for (int s = 0; s < samplesPerToken && t * samplesPerToken + s < numSamples; s++)
            {
                sum += NumOps.ToDouble(audio[t * samplesPerToken + s]);
            }
            double avg = sum / samplesPerToken;
            int code = (int)((avg + 1) / 2 * _options.CodebookSize);
            // Use MathHelper.Clamp for net471 compatibility (Math.Clamp not available)
            code = MathHelper.Clamp(code, 0, _options.CodebookSize - 1);

            for (int cb = 0; cb < _options.NumCodebooks; cb++)
            {
                codes[0, cb, t] = NumOps.FromDouble(code);
            }
        }

        return codes;
    }

    private Tensor<T> GenerateContinuation(Tensor<T> inputCodes, Tensor<T>? textEmbeddings, double extensionSeconds, Random random)
    {
        int tokensPerSecond = 50;
        int numNewTokens = (int)(extensionSeconds * tokensPerSecond);

        var codes = new Tensor<T>([1, _options.NumCodebooks, numNewTokens]);

        // Use input codes as context
        var currentTokens = inputCodes;

        for (int t = 0; t < numNewTokens; t++)
        {
            Tensor<T> logits;

            if (!_useNativeMode && _languageModel is not null)
            {
                var inputs = new Dictionary<string, Tensor<T>>
                {
                    ["audio_codes"] = currentTokens
                };
                if (textEmbeddings is not null)
                {
                    inputs["text_embeddings"] = textEmbeddings;
                }
                logits = _languageModel.Run(inputs).Values.First();
            }
            else
            {
                logits = ForwardLanguageModel(textEmbeddings ?? new Tensor<T>([1, _lmHiddenDim]), currentTokens);
            }

            for (int cb = 0; cb < _options.NumCodebooks; cb++)
            {
                int nextToken = SampleFromLogits(logits, cb, random);
                codes[0, cb, t] = NumOps.FromDouble(nextToken);
            }

            // Extend context
            if (t < numNewTokens - 1)
            {
                var newTokens = new Tensor<T>([1, _options.NumCodebooks, currentTokens.Shape[2] + 1]);
                for (int cb = 0; cb < _options.NumCodebooks; cb++)
                {
                    for (int i = 0; i < currentTokens.Shape[2]; i++)
                    {
                        newTokens[0, cb, i] = currentTokens[0, cb, i];
                    }
                    newTokens[0, cb, currentTokens.Shape[2]] = codes[0, cb, t];
                }
                currentTokens = newTokens;
            }
        }

        return codes;
    }

    private Tensor<T> ConcatenateCodes(Tensor<T> first, Tensor<T> second)
    {
        int totalTokens = first.Shape[2] + second.Shape[2];
        var result = new Tensor<T>([1, _options.NumCodebooks, totalTokens]);

        for (int cb = 0; cb < _options.NumCodebooks; cb++)
        {
            for (int t = 0; t < first.Shape[2]; t++)
            {
                result[0, cb, t] = first[0, cb, t];
            }
            for (int t = 0; t < second.Shape[2]; t++)
            {
                result[0, cb, first.Shape[2] + t] = second[0, cb, t];
            }
        }

        return result;
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <summary>
    /// Preprocesses raw audio for model input.
    /// </summary>
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio)
    {
        // For MusicGen, we primarily work with text input
        return rawAudio;
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
        if (_useNativeMode)
        {
            return Forward(input);
        }

        if (_textEncoder is null)
            throw new InvalidOperationException("Text encoder not loaded.");

        return _textEncoder.Run(input);
    }

    /// <summary>
    /// Updates model parameters.
    /// </summary>
    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode)
        {
            throw new NotSupportedException("Cannot update parameters in ONNX mode.");
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
            throw new NotSupportedException("Cannot train in ONNX mode.");
        }

        SetTrainingMode(true);

        var prediction = Forward(input);
        var flatPrediction = prediction.ToVector();
        var flatExpected = expectedOutput.ToVector();

        LastLoss = _lossFunction.CalculateLoss(flatPrediction, flatExpected);
        var lossGradient = _lossFunction.CalculateDerivative(flatPrediction, flatExpected);

        Backpropagate(Tensor<T>.FromVector(lossGradient));

        _optimizer?.UpdateParameters(Layers);

        SetTrainingMode(false);
    }

    /// <summary>
    /// Gets model metadata.
    /// </summary>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = $"MusicGen-{_options.ModelSize}",
            Description = $"Meta MusicGen text-to-music model ({_options.ModelSize})",
            ModelType = ModelType.NeuralNetwork,
            FeatureCount = _options.MaxTextLength,
            Complexity = (int)_options.ModelSize,
            AdditionalInfo = new Dictionary<string, object>
            {
                ["SampleRate"] = _options.SampleRate,
                ["MaxDuration"] = _options.MaxDurationSeconds,
                ["NumCodebooks"] = _options.NumCodebooks,
                ["Stereo"] = _options.Stereo,
                ["Mode"] = _useNativeMode ? "Native" : "ONNX"
            }
        };
    }

    /// <summary>
    /// Serializes network-specific data.
    /// </summary>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_useNativeMode);
        writer.Write((int)_options.ModelSize);
        writer.Write(_options.SampleRate);
        writer.Write(_options.DurationSeconds);
        writer.Write(_options.MaxDurationSeconds);
        writer.Write(_options.Temperature);
        writer.Write(_options.TopK);
        writer.Write(_options.TopP);
        writer.Write(_options.GuidanceScale);
        writer.Write(_options.Stereo);
        writer.Write(_options.NumCodebooks);
        writer.Write(_options.CodebookSize);
        writer.Write(_options.MaxTextLength);
    }

    /// <summary>
    /// Deserializes network-specific data.
    /// </summary>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _ = reader.ReadBoolean();
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadDouble();
        _ = reader.ReadDouble();
        _ = reader.ReadDouble();
        _ = reader.ReadInt32();
        _ = reader.ReadDouble();
        _ = reader.ReadDouble();
        _ = reader.ReadBoolean();
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
    }

    /// <summary>
    /// Creates a new instance for cloning.
    /// </summary>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new MusicGenModel<T>(
            Architecture,
            _options,
            _tokenizer,
            null,
            _lossFunction);
    }

    #endregion

    #region IDisposable

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName);
    }

    /// <summary>
    /// Disposes of resources.
    /// </summary>
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;

        if (disposing)
        {
            _textEncoder?.Dispose();
            _languageModel?.Dispose();
            // _encodecDecoder is disposed by base class (OnnxModel)
        }

        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
