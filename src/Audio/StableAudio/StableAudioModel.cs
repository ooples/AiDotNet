using AiDotNet.Enums;
using AiDotNet.Extensions;
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

namespace AiDotNet.Audio.StableAudio;

/// <summary>
/// Stable Audio model for generating high-quality audio from text descriptions.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Stable Audio is Stability AI's state-of-the-art audio generation model that uses
/// latent diffusion with a Diffusion Transformer (DiT) architecture for high-quality
/// music and sound effects generation.
/// </para>
/// <para>
/// Architecture components:
/// <list type="number">
/// <item><description><b>T5 Text Encoder:</b> Encodes text prompts into conditioning embeddings</description></item>
/// <item><description><b>VAE:</b> Compresses audio to/from latent space (44.1kHz to 21.5Hz latent)</description></item>
/// <item><description><b>DiT (Diffusion Transformer):</b> Predicts noise using transformer blocks</description></item>
/// <item><description><b>Timing Conditioning:</b> Encodes duration and timing information</description></item>
/// </list>
/// </para>
/// <para><b>For Beginners:</b> Stable Audio creates professional-quality audio:
///
/// How it works:
/// 1. You describe the audio you want ("upbeat electronic track")
/// 2. T5 encodes your text into embeddings
/// 3. Duration and timing are encoded as conditioning
/// 4. DiT diffusion generates latent audio representations
/// 5. VAE decoder converts latents to 44.1kHz stereo audio
///
/// Key features:
/// - CD-quality 44.1kHz stereo output
/// - Variable-length generation (up to 3 minutes)
/// - Music and sound effects generation
/// - Timing-aware conditioning
///
/// Usage:
/// <code>
/// var model = new StableAudioModel&lt;float&gt;(options);
/// var audio = model.GenerateAudio("Energetic rock music with electric guitar");
/// </code>
/// </para>
/// <para>
/// Reference: "Stable Audio: Fast Timing-Conditioned Latent Audio Diffusion" by Evans et al., 2024
/// </para>
/// </remarks>
public class StableAudioModel<T> : AudioNeuralNetworkBase<T>, IAudioGenerator<T>
{
    #region Fields

    private readonly StableAudioOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private readonly ITokenizer _tokenizer;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly Random _random;

    // Model dimensions based on size
    private readonly int _textHiddenDim;
    private readonly int _ditHiddenDim;
    private readonly int _numDitBlocks;
    private readonly int _numAttentionHeads;

    // ONNX models for inference mode
    private readonly OnnxModel<T>? _textEncoder;
    private readonly OnnxModel<T>? _vaeModel;
    private readonly OnnxModel<T>? _ditDenoiser;
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
    public bool SupportsTextToAudio => true;

    /// <summary>
    /// Gets whether this model supports text-to-music generation.
    /// </summary>
    public bool SupportsTextToMusic => true;

    /// <summary>
    /// Gets whether this model supports audio continuation.
    /// </summary>
    public bool SupportsAudioContinuation => true;

    /// <summary>
    /// Gets whether this model supports audio inpainting.
    /// </summary>
    public bool SupportsAudioInpainting => true;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a Stable Audio model using pretrained ONNX models for inference.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="textEncoderPath">Path to the T5 text encoder ONNX model.</param>
    /// <param name="vaePath">Path to the VAE ONNX model.</param>
    /// <param name="ditPath">Path to the DiT denoiser ONNX model.</param>
    /// <param name="tokenizer">T5 tokenizer for text processing (REQUIRED).</param>
    /// <param name="options">Stable Audio configuration options.</param>
    /// <param name="optimizer">Optional optimizer for fine-tuning.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <exception cref="ArgumentException">Thrown when required paths are empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown when model files don't exist.</exception>
    /// <exception cref="ArgumentNullException">Thrown when tokenizer is null.</exception>
    public StableAudioModel(
        NeuralNetworkArchitecture<T> architecture,
        string textEncoderPath,
        string vaePath,
        string ditPath,
        ITokenizer tokenizer,
        StableAudioOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        // Validate paths
        if (string.IsNullOrWhiteSpace(textEncoderPath))
            throw new ArgumentException("Text encoder path is required.", nameof(textEncoderPath));
        if (string.IsNullOrWhiteSpace(vaePath))
            throw new ArgumentException("VAE path is required.", nameof(vaePath));
        if (string.IsNullOrWhiteSpace(ditPath))
            throw new ArgumentException("DiT path is required.", nameof(ditPath));
        if (!File.Exists(textEncoderPath))
            throw new FileNotFoundException($"Text encoder not found: {textEncoderPath}");
        if (!File.Exists(vaePath))
            throw new FileNotFoundException($"VAE not found: {vaePath}");
        if (!File.Exists(ditPath))
            throw new FileNotFoundException($"DiT not found: {ditPath}");

        _options = options ?? new StableAudioOptions();
        Options = _options;
        _tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer),
            "Tokenizer is required. Use T5 tokenizer or compatible tokenizer.");
        _useNativeMode = false;

        // Set dimensions based on model size
        (_textHiddenDim, _ditHiddenDim, _numDitBlocks, _numAttentionHeads) = GetModelDimensions(_options.ModelSize);

        // Load ONNX models
        OnnxModel<T>? textEncoder = null;
        OnnxModel<T>? vaeModel = null;
        OnnxModel<T>? ditDenoiser = null;

        try
        {
            textEncoder = new OnnxModel<T>(textEncoderPath);
            vaeModel = new OnnxModel<T>(vaePath);
            ditDenoiser = new OnnxModel<T>(ditPath);

            _textEncoder = textEncoder;
            _vaeModel = vaeModel;
            _ditDenoiser = ditDenoiser;

            // Set base class ONNX model
            OnnxModel = ditDenoiser;
        }
        catch
        {
            textEncoder?.Dispose();
            vaeModel?.Dispose();
            ditDenoiser?.Dispose();
            throw;
        }

        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _random = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSecureRandom();

        base.SampleRate = _options.SampleRate;

        InitializeLayers();
    }

    /// <summary>
    /// Creates a Stable Audio model using native layers for training from scratch.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="options">Stable Audio configuration options.</param>
    /// <param name="tokenizer">Optional tokenizer. If null, creates T5-compatible tokenizer.</param>
    /// <param name="optimizer">Optional optimizer. Defaults to AdamW.</param>
    /// <param name="lossFunction">Optional loss function. Defaults to MSE.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor when:
    /// - Training Stable Audio from scratch (requires significant data and compute)
    /// - Fine-tuning on custom audio types
    /// - Research and experimentation
    ///
    /// For most use cases, load pretrained ONNX models instead.
    /// </para>
    /// </remarks>
    public StableAudioModel(
        NeuralNetworkArchitecture<T> architecture,
        StableAudioOptions? options = null,
        ITokenizer? tokenizer = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        _options = options ?? new StableAudioOptions();
        Options = _options;
        _useNativeMode = true;

        // Set dimensions based on model size
        (_textHiddenDim, _ditHiddenDim, _numDitBlocks, _numAttentionHeads) = GetModelDimensions(_options.ModelSize);

        // Use T5-compatible tokenizer as default
        _tokenizer = tokenizer ?? Tokenization.LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.FlanT5);
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
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
        // 3. If no, use LayerHelper.CreateDefaultStableAudioLayers()
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            ValidateLayerConfiguration(Layers);
        }
        else
        {
            // Use default Stable Audio architecture
            Layers.AddRange(LayerHelper<T>.CreateDefaultStableAudioLayers(
                textHiddenDim: _textHiddenDim,
                latentDim: _options.LatentDimension,
                ditHiddenDim: _ditHiddenDim,
                numDitBlocks: _numDitBlocks,
                numHeads: _numAttentionHeads,
                maxTextLength: _options.MaxTextLength,
                maxAudioLength: _options.MaxAudioLength,
                dropoutRate: _options.DropoutRate));
        }
    }

    /// <summary>
    /// Validates that custom layers meet Stable Audio requirements.
    /// </summary>
    private void ValidateLayerConfiguration(List<ILayer<T>> layers)
    {
        if (layers.Count < 3)
        {
            throw new ArgumentException(
                "Stable Audio requires at least 3 layers: T5 encoder, DiT denoiser, and decoder. " +
                "Use LayerHelper.CreateDefaultStableAudioLayers() as a reference.",
                nameof(layers));
        }
    }

    #endregion

    #region Model Dimensions

    /// <summary>
    /// Gets model dimensions based on the selected size.
    /// </summary>
    private static (int textHiddenDim, int ditHiddenDim, int numDitBlocks, int numAttentionHeads) GetModelDimensions(StableAudioModelSize size)
    {
        return size switch
        {
            StableAudioModelSize.Small => (256, 512, 12, 8),
            StableAudioModelSize.Base => (768, 1024, 24, 16),
            StableAudioModelSize.Large => (1024, 1536, 32, 24),
            StableAudioModelSize.Open => (768, 1024, 24, 16),
            StableAudioModelSize.V2 => (1024, 1536, 28, 16),
            _ => (768, 1024, 24, 16)
        };
    }

    #endregion

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

        if (string.IsNullOrWhiteSpace(prompt))
            throw new ArgumentException("Text prompt is required.", nameof(prompt));

        durationSeconds = Math.Min(durationSeconds, _options.MaxDurationSeconds);

        int seedUsed = seed ?? _random.Next();
        var random = RandomHelper.CreateSeededRandom(seedUsed);

        // Tokenize and encode text
        var tokenTensor = EncodeTextToTensor(prompt);

        Tensor<T>? negativeEmbedding = null;
        if (negativePrompt is string negPrompt && !string.IsNullOrEmpty(negPrompt) && guidanceScale > 1.0)
        {
            var negTokenTensor = EncodeTextToTensor(negPrompt);
            negativeEmbedding = EncodeTextEmbedding(negTokenTensor);
        }

        var textEmbedding = EncodeTextEmbedding(tokenTensor);

        // Create timing conditioning
        var timingConditioning = CreateTimingConditioning(durationSeconds);

        // Calculate latent dimensions (44.1kHz -> ~21.5Hz latent)
        int numLatentFrames = (int)(durationSeconds * 21.5);

        // Initialize random latents
        var latents = InitializeLatents(numLatentFrames, random);

        // Run diffusion denoising
        var denoisedLatents = RunDiffusionLoop(latents, textEmbedding, timingConditioning, negativeEmbedding, numInferenceSteps, guidanceScale, random);

        // Decode to audio
        return DecodeLatentsToAudio(denoisedLatents);
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
    public Tensor<T> GenerateMusic(
        string prompt,
        string? negativePrompt = null,
        double durationSeconds = 10.0,
        int numInferenceSteps = 100,
        double guidanceScale = 3.0,
        int? seed = null)
    {
        // Stable Audio is designed for music generation
        return GenerateAudio(prompt, negativePrompt, durationSeconds, numInferenceSteps, guidanceScale, seed);
    }

    /// <summary>
    /// Continues existing audio by extending it.
    /// </summary>
    public Tensor<T> ContinueAudio(
        Tensor<T> inputAudio,
        string? prompt = null,
        double extensionSeconds = 5.0,
        int numInferenceSteps = 100,
        int? seed = null)
    {
        ThrowIfDisposed();

        if (inputAudio.Length == 0)
            throw new ArgumentException("Input audio cannot be empty.", nameof(inputAudio));

        int seedUsed = seed ?? _random.Next();
        var random = RandomHelper.CreateSeededRandom(seedUsed);

        // Encode input audio to latent space
        var inputLatents = EncodeAudioToLatents(inputAudio);

        // Encode text prompt if provided
        Tensor<T>? textEmbedding = null;
        if (prompt is string textPrompt && !string.IsNullOrEmpty(textPrompt))
        {
            var tokenTensor = EncodeTextToTensor(textPrompt);
            textEmbedding = EncodeTextEmbedding(tokenTensor);
        }

        // Calculate new latent dimensions
        int numLatentFrames = (int)(extensionSeconds * 21.5);

        // Initialize extension latents with noise
        var extensionLatents = InitializeLatents(numLatentFrames, random);

        // Create timing conditioning for extended duration
        double totalDuration = (inputAudio.Length / (double)_options.SampleRate) + extensionSeconds;
        var timingConditioning = CreateTimingConditioning(totalDuration);

        // Run diffusion with audio conditioning
        var conditioning = textEmbedding ?? CreateUnconditionalEmbedding();
        var combinedConditioning = CombineConditionings(inputLatents, conditioning);
        var denoisedExtension = RunDiffusionLoop(extensionLatents, combinedConditioning, timingConditioning, null, numInferenceSteps, _options.GuidanceScale, random);

        // Decode and concatenate
        var extensionAudio = DecodeLatentsToAudio(denoisedExtension);

        // Concatenate original and extension
        int totalLength = inputAudio.Length + extensionAudio.Length;
        var result = new Tensor<T>(new int[] { totalLength });
        for (int i = 0; i < inputAudio.Length; i++)
        {
            result.SetFlat(i, inputAudio.GetFlat(i));
        }
        for (int i = 0; i < extensionAudio.Length; i++)
        {
            result.SetFlat(inputAudio.Length + i, extensionAudio.GetFlat(i));
        }

        return result;
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
        ThrowIfDisposed();

        if (audio.Length != mask.Length)
            throw new ArgumentException("Mask must have the same length as input audio.");

        int seedUsed = seed ?? _random.Next();
        var random = RandomHelper.CreateSeededRandom(seedUsed);

        // Encode input audio to latent space
        var inputLatents = EncodeAudioToLatents(audio);

        // Encode text prompt
        Tensor<T> conditioning;
        if (prompt is string textPrompt && !string.IsNullOrEmpty(textPrompt))
        {
            var tokenTensor = EncodeTextToTensor(textPrompt);
            conditioning = EncodeTextEmbedding(tokenTensor);
        }
        else
        {
            conditioning = CreateUnconditionalEmbedding();
        }

        // Create timing conditioning
        double duration = audio.Length / (double)_options.SampleRate;
        var timingConditioning = CreateTimingConditioning(duration);

        // Initialize with noise where mask is 1
        var noisyLatents = inputLatents.Clone();
        for (int i = 0; i < noisyLatents.Length && i < mask.Length; i++)
        {
            if (NumOps.ToDouble(mask.GetFlat(i)) > 0.5)
            {
                noisyLatents.SetFlat(i, NumOps.FromDouble(random.NextGaussian()));
            }
        }

        // Run diffusion
        var denoised = RunDiffusionLoop(noisyLatents, conditioning, timingConditioning, null, numInferenceSteps, _options.GuidanceScale, random);

        // Blend based on mask
        var blendedLatents = inputLatents.Clone();
        for (int i = 0; i < blendedLatents.Length && i < mask.Length; i++)
        {
            double m = NumOps.ToDouble(mask.GetFlat(i));
            blendedLatents.SetFlat(i, NumOps.Add(
                NumOps.Multiply(inputLatents.GetFlat(i), NumOps.FromDouble(1.0 - m)),
                NumOps.Multiply(denoised.GetFlat(i), NumOps.FromDouble(m))));
        }

        return DecodeLatentsToAudio(blendedLatents);
    }

    /// <summary>
    /// Gets default generation options.
    /// </summary>
    public AudioGenerationOptions<T> GetDefaultOptions()
    {
        return new AudioGenerationOptions<T>
        {
            DurationSeconds = _options.DurationSeconds,
            NumInferenceSteps = _options.NumInferenceSteps,
            GuidanceScale = _options.GuidanceScale,
            Seed = _options.Seed,
            Stereo = _options.Stereo,
            SchedulerType = "euler"
        };
    }

    #endregion

    #region Internal Generation Methods

    /// <summary>
    /// Runs the diffusion denoising loop with DiT.
    /// </summary>
    private Tensor<T> RunDiffusionLoop(
        Tensor<T> latents,
        Tensor<T> textConditioning,
        Tensor<T> timingConditioning,
        Tensor<T>? negativeConditioning,
        int numSteps,
        double guidanceScale,
        Random random)
    {
        var currentLatents = latents;
        var timesteps = GenerateTimesteps(numSteps);

        foreach (var t in timesteps)
        {
            var timestepTensor = CreateTimestepTensor(t);
            var ditInput = PrepareDiTInput(currentLatents, textConditioning, timingConditioning, timestepTensor);

            Tensor<T> noisePred;
            if (IsOnnxMode && _ditDenoiser is not null)
            {
                noisePred = _ditDenoiser.Run(ditInput);
            }
            else
            {
                noisePred = ForwardThroughDiT(ditInput);
            }

            // Classifier-free guidance
            if (guidanceScale > 1.0 && negativeConditioning is not null)
            {
                var uncondInput = PrepareDiTInput(currentLatents, negativeConditioning, timingConditioning, timestepTensor);
                Tensor<T> uncondNoisePred;
                if (IsOnnxMode && _ditDenoiser is not null)
                {
                    uncondNoisePred = _ditDenoiser.Run(uncondInput);
                }
                else
                {
                    uncondNoisePred = ForwardThroughDiT(uncondInput);
                }
                noisePred = ApplyGuidance(uncondNoisePred, noisePred, guidanceScale);
            }

            // Euler step (Stable Audio uses Euler sampler)
            currentLatents = EulerStep(currentLatents, noisePred, t, numSteps);
        }

        return currentLatents;
    }

    /// <summary>
    /// Forwards input through DiT layers.
    /// </summary>
    private Tensor<T> ForwardThroughDiT(Tensor<T> input)
    {
        var current = input;

        // DiT is the middle portion of layers (after T5 encoder, before VAE decoder)
        int startIdx = Layers.Count / 3;
        int endIdx = 2 * Layers.Count / 3;

        for (int i = startIdx; i < endIdx && i < Layers.Count; i++)
        {
            current = Layers[i].Forward(current);
        }

        return current;
    }

    /// <summary>
    /// Encodes text embedding from tokens.
    /// </summary>
    private Tensor<T> EncodeTextEmbedding(Tensor<T> tokenTensor)
    {
        if (_textEncoder is not null)
        {
            return _textEncoder.Run(tokenTensor);
        }

        // Use native layers (first portion - T5 encoder)
        var current = tokenTensor;
        int t5LayerCount = Math.Min(12, Layers.Count / 3);
        for (int i = 0; i < t5LayerCount && i < Layers.Count; i++)
        {
            current = Layers[i].Forward(current);
        }
        return current;
    }

    /// <summary>
    /// Encodes audio to latent space.
    /// </summary>
    private Tensor<T> EncodeAudioToLatents(Tensor<T> audio)
    {
        if (_vaeModel is not null)
        {
            return _vaeModel.Run(audio);
        }

        // Native encoding through layers
        var current = audio;
        int encoderLayers = Layers.Count / 4;
        for (int i = 0; i < encoderLayers && i < Layers.Count; i++)
        {
            current = Layers[i].Forward(current);
        }
        return current;
    }

    /// <summary>
    /// Decodes latents to audio waveform.
    /// </summary>
    private Tensor<T> DecodeLatentsToAudio(Tensor<T> latents)
    {
        if (_vaeModel is not null)
        {
            return _vaeModel.Run(latents);
        }

        // Native decoding through layers
        var current = latents;
        int startIdx = 2 * Layers.Count / 3;
        for (int i = startIdx; i < Layers.Count; i++)
        {
            current = Layers[i].Forward(current);
        }
        return current;
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Encodes text prompt to token tensor.
    /// </summary>
    private Tensor<T> EncodeTextToTensor(string prompt)
    {
        var encoding = _tokenizer.Encode(prompt);
        var tokens = new int[_options.MaxTextLength];
        int copyCount = Math.Min(encoding.TokenIds.Count, _options.MaxTextLength);
        for (int i = 0; i < copyCount; i++)
        {
            tokens[i] = encoding.TokenIds[i];
        }

        var tensor = new Tensor<T>(new int[] { 1, _options.MaxTextLength });
        for (int i = 0; i < _options.MaxTextLength; i++)
        {
            tensor[0, i] = NumOps.FromDouble(tokens[i]);
        }
        return tensor;
    }

    /// <summary>
    /// Creates timing conditioning for the given duration.
    /// </summary>
    private Tensor<T> CreateTimingConditioning(double durationSeconds)
    {
        // Stable Audio uses sinusoidal embeddings for timing
        int embeddingDim = _ditHiddenDim;
        var embedding = new Tensor<T>(new int[] { 1, embeddingDim });

        // Encode duration and start/end times
        double startSeconds = 0.0;
        double endSeconds = durationSeconds;

        for (int i = 0; i < embeddingDim / 4; i++)
        {
            double freq = Math.Pow(10000, -2.0 * i / (embeddingDim / 2));

            // Duration embedding
            embedding[0, i] = NumOps.FromDouble(Math.Sin(durationSeconds * freq));
            embedding[0, i + embeddingDim / 4] = NumOps.FromDouble(Math.Cos(durationSeconds * freq));

            // Start/end time embedding
            embedding[0, i + embeddingDim / 2] = NumOps.FromDouble(Math.Sin(startSeconds * freq + endSeconds * freq));
            embedding[0, i + 3 * embeddingDim / 4] = NumOps.FromDouble(Math.Cos(startSeconds * freq + endSeconds * freq));
        }

        return embedding;
    }

    /// <summary>
    /// Initializes latent noise for diffusion.
    /// </summary>
    private Tensor<T> InitializeLatents(int numFrames, Random random)
    {
        int latentChannels = _options.LatentDimension;
        int numChannels = _options.Stereo ? 2 : 1;

        var shape = new int[] { 1, latentChannels, numChannels, numFrames };
        var latents = new Tensor<T>(shape);

        for (int i = 0; i < latents.Length; i++)
        {
            latents.SetFlat(i, NumOps.FromDouble(random.NextGaussian()));
        }

        return latents;
    }

    /// <summary>
    /// Generates diffusion timesteps using Euler schedule.
    /// </summary>
    private static double[] GenerateTimesteps(int numSteps)
    {
        // Stable Audio uses continuous timesteps (0 to 1)
        var timesteps = new double[numSteps];

        for (int i = 0; i < numSteps; i++)
        {
            timesteps[i] = 1.0 - (double)i / numSteps;
        }

        return timesteps;
    }

    /// <summary>
    /// Creates a timestep tensor with sinusoidal embedding.
    /// </summary>
    private Tensor<T> CreateTimestepTensor(double timestep)
    {
        int embeddingDim = _ditHiddenDim;
        var embedding = new Tensor<T>(new int[] { 1, embeddingDim });

        for (int i = 0; i < embeddingDim / 2; i++)
        {
            double freq = Math.Pow(10000, -2.0 * i / embeddingDim);
            embedding[0, i] = NumOps.FromDouble(Math.Sin(timestep * 1000 * freq));
            embedding[0, i + embeddingDim / 2] = NumOps.FromDouble(Math.Cos(timestep * 1000 * freq));
        }

        return embedding;
    }

    /// <summary>
    /// Prepares DiT input by combining latents, text conditioning, timing, and timestep.
    /// </summary>
    private Tensor<T> PrepareDiTInput(Tensor<T> latents, Tensor<T> textConditioning, Tensor<T> timingConditioning, Tensor<T> timestep)
    {
        var conditionedLatents = ApplyCrossAttentionConditioning(latents, textConditioning);
        var withTiming = AddTimingConditioning(conditionedLatents, timingConditioning);
        return AddTimestepEmbedding(withTiming, timestep);
    }

    /// <summary>
    /// Applies cross-attention conditioning to latents.
    /// </summary>
    private Tensor<T> ApplyCrossAttentionConditioning(Tensor<T> latents, Tensor<T> conditioning)
    {
        var result = latents.Clone();
        var condScale = NumOps.FromDouble(0.1);

        if (conditioning.Length > 0)
        {
            for (int i = 0; i < result.Length; i++)
            {
                int condIdx = i % conditioning.Length;
                result.SetFlat(i, NumOps.Add(result.GetFlat(i), NumOps.Multiply(conditioning.GetFlat(condIdx), condScale)));
            }
        }

        return result;
    }

    /// <summary>
    /// Adds timing conditioning to latents.
    /// </summary>
    private Tensor<T> AddTimingConditioning(Tensor<T> latents, Tensor<T> timingConditioning)
    {
        var result = latents.Clone();
        var scale = NumOps.FromDouble(_options.TimingConditioningScale * 0.1);

        for (int i = 0; i < result.Length; i++)
        {
            int tIdx = i % timingConditioning.Length;
            result.SetFlat(i, NumOps.Add(result.GetFlat(i), NumOps.Multiply(timingConditioning.GetFlat(tIdx), scale)));
        }

        return result;
    }

    /// <summary>
    /// Adds timestep embedding to latents.
    /// </summary>
    private Tensor<T> AddTimestepEmbedding(Tensor<T> latents, Tensor<T> timestep)
    {
        var result = latents.Clone();

        for (int i = 0; i < result.Length; i++)
        {
            int tIdx = i % timestep.Length;
            result.SetFlat(i, NumOps.Add(result.GetFlat(i), NumOps.Multiply(timestep.GetFlat(tIdx), NumOps.FromDouble(0.01))));
        }

        return result;
    }

    /// <summary>
    /// Creates unconditional embedding for classifier-free guidance.
    /// </summary>
    private Tensor<T> CreateUnconditionalEmbedding()
    {
        return new Tensor<T>(new int[] { 1, _options.TextEmbeddingDim });
    }

    /// <summary>
    /// Combines audio and text conditionings.
    /// </summary>
    private Tensor<T> CombineConditionings(Tensor<T> audioLatents, Tensor<T> textConditioning)
    {
        int combinedLength = audioLatents.Length + textConditioning.Length;
        var combined = new Tensor<T>(new int[] { 1, combinedLength });

        for (int i = 0; i < audioLatents.Length; i++)
        {
            combined.SetFlat(i, audioLatents.GetFlat(i));
        }
        for (int i = 0; i < textConditioning.Length; i++)
        {
            combined.SetFlat(audioLatents.Length + i, textConditioning.GetFlat(i));
        }

        return combined;
    }

    /// <summary>
    /// Applies classifier-free guidance.
    /// </summary>
    private Tensor<T> ApplyGuidance(Tensor<T> uncondPred, Tensor<T> condPred, double scale)
    {
        var result = new Tensor<T>(condPred.Shape);
        var scaleT = NumOps.FromDouble(scale);

        for (int i = 0; i < result.Length; i++)
        {
            var diff = NumOps.Subtract(condPred.GetFlat(i), uncondPred.GetFlat(i));
            result.SetFlat(i, NumOps.Add(uncondPred.GetFlat(i), NumOps.Multiply(diff, scaleT)));
        }

        return result;
    }

    /// <summary>
    /// Performs an Euler denoising step.
    /// </summary>
    private Tensor<T> EulerStep(Tensor<T> latents, Tensor<T> noisePred, double timestep, int numSteps)
    {
        // Euler integration: x_{t-1} = x_t + sigma_t * noise_pred * dt
        double sigma = timestep; // sigma schedule proportional to timestep
        double dt = 1.0 / numSteps;

        var result = new Tensor<T>(latents.Shape);

        for (int i = 0; i < result.Length; i++)
        {
            double x_t = NumOps.ToDouble(latents.GetFlat(i));
            double noise = NumOps.ToDouble(noisePred.GetFlat(i));

            // Euler step: x_{t-dt} = x_t - sigma * noise * dt
            double x_prev = x_t - sigma * noise * dt;

            result.SetFlat(i, NumOps.FromDouble(x_prev));
        }

        return result;
    }

    #endregion

    #region NeuralNetworkBase Implementation

    /// <summary>
    /// Preprocesses raw audio for model input.
    /// </summary>
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio)
    {
        // Stable Audio VAE operates directly on waveforms
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

        _optimizer.UpdateParameters(Layers);

        SetTrainingMode(false);
    }

    /// <summary>
    /// Gets model metadata.
    /// </summary>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = $"StableAudio-{_options.ModelSize}",
            Description = $"Stable Audio text-to-audio model ({_options.ModelSize})",
            ModelType = ModelType.NeuralNetwork,
            FeatureCount = _options.MaxTextLength,
            Complexity = (int)_options.ModelSize,
            AdditionalInfo = new Dictionary<string, object>
            {
                ["SampleRate"] = _options.SampleRate,
                ["MaxDuration"] = _options.MaxDurationSeconds,
                ["NumInferenceSteps"] = _options.NumInferenceSteps,
                ["GuidanceScale"] = _options.GuidanceScale,
                ["Mode"] = _useNativeMode ? "Native" : "ONNX",
                ["Stereo"] = _options.Stereo
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
        writer.Write(_options.NumInferenceSteps);
        writer.Write(_options.GuidanceScale);
        writer.Write(_options.Stereo);
    }

    /// <summary>
    /// Deserializes network-specific data.
    /// </summary>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _ = reader.ReadBoolean();
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadDouble();
        _ = reader.ReadBoolean();
    }

    /// <summary>
    /// Creates a new instance for cloning.
    /// </summary>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new StableAudioModel<T>(
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
        {
            throw new ObjectDisposedException(nameof(StableAudioModel<T>));
        }
    }

    /// <summary>
    /// Disposes of model resources.
    /// </summary>
    protected override void Dispose(bool disposing)
    {
        if (!_disposed)
        {
            if (disposing)
            {
                _textEncoder?.Dispose();
                _vaeModel?.Dispose();
                _ditDenoiser?.Dispose();
            }
            _disposed = true;
        }
        base.Dispose(disposing);
    }

    #endregion
}
