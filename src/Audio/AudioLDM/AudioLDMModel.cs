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

namespace AiDotNet.Audio.AudioLDM;

/// <summary>
/// AudioLDM (Audio Latent Diffusion Model) for generating audio from text descriptions.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// AudioLDM is a latent diffusion model that generates audio by learning to reverse
/// a diffusion process in a compressed latent space. It uses CLAP (Contrastive Language-Audio
/// Pretraining) for text conditioning and a VAE for efficient latent space learning.
/// </para>
/// <para>
/// Architecture components:
/// <list type="number">
/// <item><description><b>CLAP Encoder:</b> Contrastive text encoder that aligns text with audio features</description></item>
/// <item><description><b>VAE:</b> Variational autoencoder that compresses mel spectrograms to latent space</description></item>
/// <item><description><b>U-Net Denoiser:</b> Predicts noise to be removed at each diffusion step</description></item>
/// <item><description><b>HiFi-GAN Vocoder:</b> Converts mel spectrograms to audio waveforms</description></item>
/// </list>
/// </para>
/// <para><b>For Beginners:</b> AudioLDM creates realistic audio from your descriptions:
///
/// How it works:
/// 1. You describe the sound you want ("a cat meowing")
/// 2. CLAP encodes your text into an audio-aligned representation
/// 3. The diffusion process generates a latent audio representation
/// 4. The VAE decoder converts latents to mel spectrogram
/// 5. HiFi-GAN vocoder converts the spectrogram to audio
///
/// Key features:
/// - General audio and music generation
/// - Environmental sounds, speech, music
/// - Controllable through text prompts
/// - High-quality 16kHz or 48kHz output
///
/// Usage:
/// <code>
/// var model = new AudioLDMModel&lt;float&gt;(options);
/// var audio = model.GenerateAudio("A dog barking in a park");
/// </code>
/// </para>
/// <para>
/// Reference: "AudioLDM: Text-to-Audio Generation with Latent Diffusion Models" by Liu et al., 2023
/// </para>
/// </remarks>
public class AudioLDMModel<T> : AudioNeuralNetworkBase<T>, IAudioGenerator<T>
{
    #region Fields

    private readonly AudioLDMOptions _options;
    private readonly ITokenizer _tokenizer;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly Random _random;

    // Model dimensions based on size
    private readonly int _clapHiddenDim;
    private readonly int _unetBaseChannels;
    private readonly int _numAttentionHeads;

    // ONNX models for inference mode
    private readonly OnnxModel<T>? _clapEncoder;
    private readonly OnnxModel<T>? _vaeModel;
    private readonly OnnxModel<T>? _unetDenoiser;
    private readonly OnnxModel<T>? _vocoder;
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
    public bool SupportsTextToMusic => _options.ModelSize == AudioLDMModelSize.Music;

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
    /// Creates an AudioLDM model using pretrained ONNX models for inference.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="clapEncoderPath">Path to the CLAP text encoder ONNX model.</param>
    /// <param name="vaePath">Path to the VAE ONNX model.</param>
    /// <param name="unetPath">Path to the U-Net denoiser ONNX model.</param>
    /// <param name="vocoderPath">Path to the HiFi-GAN vocoder ONNX model.</param>
    /// <param name="tokenizer">CLAP tokenizer for text processing (REQUIRED).</param>
    /// <param name="options">AudioLDM configuration options.</param>
    /// <param name="optimizer">Optional optimizer for fine-tuning.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <exception cref="ArgumentException">Thrown when required paths are empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown when model files don't exist.</exception>
    /// <exception cref="ArgumentNullException">Thrown when tokenizer is null.</exception>
    public AudioLDMModel(
        NeuralNetworkArchitecture<T> architecture,
        string clapEncoderPath,
        string vaePath,
        string unetPath,
        string vocoderPath,
        ITokenizer tokenizer,
        AudioLDMOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        // Validate paths
        if (string.IsNullOrWhiteSpace(clapEncoderPath))
            throw new ArgumentException("CLAP encoder path is required.", nameof(clapEncoderPath));
        if (string.IsNullOrWhiteSpace(vaePath))
            throw new ArgumentException("VAE path is required.", nameof(vaePath));
        if (string.IsNullOrWhiteSpace(unetPath))
            throw new ArgumentException("U-Net path is required.", nameof(unetPath));
        if (string.IsNullOrWhiteSpace(vocoderPath))
            throw new ArgumentException("Vocoder path is required.", nameof(vocoderPath));
        if (!File.Exists(clapEncoderPath))
            throw new FileNotFoundException($"CLAP encoder not found: {clapEncoderPath}");
        if (!File.Exists(vaePath))
            throw new FileNotFoundException($"VAE not found: {vaePath}");
        if (!File.Exists(unetPath))
            throw new FileNotFoundException($"U-Net not found: {unetPath}");
        if (!File.Exists(vocoderPath))
            throw new FileNotFoundException($"Vocoder not found: {vocoderPath}");

        _options = options ?? new AudioLDMOptions();
        _tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer),
            "Tokenizer is required. Use CLAP tokenizer or compatible tokenizer.");
        _useNativeMode = false;

        // Set dimensions based on model size
        (_clapHiddenDim, _unetBaseChannels, _numAttentionHeads) = GetModelDimensions(_options.ModelSize);

        // Load ONNX models
        OnnxModel<T>? clapEncoder = null;
        OnnxModel<T>? vaeModel = null;
        OnnxModel<T>? unetDenoiser = null;
        OnnxModel<T>? vocoder = null;

        try
        {
            clapEncoder = new OnnxModel<T>(clapEncoderPath);
            vaeModel = new OnnxModel<T>(vaePath);
            unetDenoiser = new OnnxModel<T>(unetPath);
            vocoder = new OnnxModel<T>(vocoderPath);

            _clapEncoder = clapEncoder;
            _vaeModel = vaeModel;
            _unetDenoiser = unetDenoiser;
            _vocoder = vocoder;

            // Set base class ONNX model
            OnnxModel = vocoder;
        }
        catch
        {
            clapEncoder?.Dispose();
            vaeModel?.Dispose();
            unetDenoiser?.Dispose();
            vocoder?.Dispose();
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
    /// Creates an AudioLDM model using native layers for training from scratch.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="options">AudioLDM configuration options.</param>
    /// <param name="tokenizer">Optional tokenizer. If null, creates CLAP-compatible tokenizer.</param>
    /// <param name="optimizer">Optional optimizer. Defaults to AdamW.</param>
    /// <param name="lossFunction">Optional loss function. Defaults to MSE.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor when:
    /// - Training AudioLDM from scratch (requires significant data)
    /// - Fine-tuning on custom audio types
    /// - Research and experimentation
    ///
    /// For most use cases, load pretrained ONNX models instead.
    /// </para>
    /// </remarks>
    public AudioLDMModel(
        NeuralNetworkArchitecture<T> architecture,
        AudioLDMOptions? options = null,
        ITokenizer? tokenizer = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        _options = options ?? new AudioLDMOptions();
        _useNativeMode = true;

        // Set dimensions based on model size
        (_clapHiddenDim, _unetBaseChannels, _numAttentionHeads) = GetModelDimensions(_options.ModelSize);

        // Use T5-compatible tokenizer as default (CLAP shares similar vocabulary)
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
        // 3. If no, use LayerHelper.CreateDefaultAudioLDMLayers()
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            ValidateLayerConfiguration(Layers);
        }
        else
        {
            // Use default AudioLDM architecture
            Layers.AddRange(LayerHelper<T>.CreateDefaultAudioLDMLayers(
                textHiddenDim: _clapHiddenDim,
                unetChannels: _unetBaseChannels,
                numHeads: _numAttentionHeads,
                latentDim: _options.LatentDimension,
                numMels: _options.NumMelBins,
                maxTextLength: _options.MaxTextLength,
                dropoutRate: _options.DropoutRate));
        }
    }

    /// <summary>
    /// Validates that custom layers meet AudioLDM requirements.
    /// </summary>
    private void ValidateLayerConfiguration(List<ILayer<T>> layers)
    {
        if (layers.Count < 3)
        {
            throw new ArgumentException(
                "AudioLDM requires at least 3 layers: CLAP encoder, U-Net denoiser, and decoder. " +
                "Use LayerHelper.CreateDefaultAudioLDMLayers() as a reference.",
                nameof(layers));
        }
    }

    #endregion

    #region Model Dimensions

    /// <summary>
    /// Gets model dimensions based on the selected size.
    /// </summary>
    private static (int clapHiddenDim, int unetBaseChannels, int numAttentionHeads) GetModelDimensions(AudioLDMModelSize size)
    {
        return size switch
        {
            AudioLDMModelSize.Small => (256, 320, 4),
            AudioLDMModelSize.Base => (512, 512, 8),
            AudioLDMModelSize.Large => (768, 768, 8),
            AudioLDMModelSize.V2 => (768, 512, 8),
            AudioLDMModelSize.Music => (512, 512, 8),
            _ => (512, 512, 8)
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
            negativeEmbedding = EncodeClapEmbedding(negTokenTensor);
        }

        var textEmbedding = EncodeClapEmbedding(tokenTensor);

        // Calculate latent dimensions
        int numLatentFrames = (int)(durationSeconds * _options.SampleRate / _options.HopLength / _options.LatentDownsampleFactor);

        // Initialize random latents
        var latents = InitializeLatents(numLatentFrames, random);

        // Run diffusion denoising
        var denoisedLatents = RunDiffusionLoop(latents, textEmbedding, negativeEmbedding, numInferenceSteps, guidanceScale, random);

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
        // AudioLDM can generate music-like content
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
            textEmbedding = EncodeClapEmbedding(tokenTensor);
        }

        // Calculate new latent dimensions
        int numLatentFrames = (int)(extensionSeconds * _options.SampleRate / _options.HopLength / _options.LatentDownsampleFactor);

        // Initialize extension latents with noise
        var extensionLatents = InitializeLatents(numLatentFrames, random);

        // Run diffusion with audio conditioning
        var conditioning = textEmbedding ?? CreateUnconditionalEmbedding();
        var combinedConditioning = CombineConditionings(inputLatents, conditioning);
        var denoisedExtension = RunDiffusionLoop(extensionLatents, combinedConditioning, null, numInferenceSteps, _options.GuidanceScale, random);

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
            conditioning = EncodeClapEmbedding(tokenTensor);
        }
        else
        {
            conditioning = CreateUnconditionalEmbedding();
        }

        // Initialize with noise where mask is 1
        var noisyLatents = inputLatents.Clone();
        for (int i = 0; i < noisyLatents.Length && i < mask.Length; i++)
        {
            if (NumOps.ToDouble(mask.GetFlat(i)) > 0.5)
            {
                double u1 = 1.0 - random.NextDouble();
                double u2 = random.NextDouble();
                double gaussian = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
                noisyLatents.SetFlat(i, NumOps.FromDouble(gaussian));
            }
        }

        // Run diffusion
        var denoised = RunDiffusionLoop(noisyLatents, conditioning, null, numInferenceSteps, _options.GuidanceScale, random);

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
            SchedulerType = "ddpm"
        };
    }

    #endregion

    #region Internal Generation Methods

    /// <summary>
    /// Runs the diffusion denoising loop.
    /// </summary>
    private Tensor<T> RunDiffusionLoop(
        Tensor<T> latents,
        Tensor<T> conditioning,
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
            var unetInput = PrepareUNetInput(currentLatents, conditioning, timestepTensor);

            Tensor<T> noisePred;
            if (IsOnnxMode && _unetDenoiser is not null)
            {
                noisePred = _unetDenoiser.Run(unetInput);
            }
            else
            {
                noisePred = ForwardThroughUNet(unetInput);
            }

            // Classifier-free guidance
            if (guidanceScale > 1.0 && negativeConditioning is not null)
            {
                var uncondInput = PrepareUNetInput(currentLatents, negativeConditioning, timestepTensor);
                Tensor<T> uncondNoisePred;
                if (IsOnnxMode && _unetDenoiser is not null)
                {
                    uncondNoisePred = _unetDenoiser.Run(uncondInput);
                }
                else
                {
                    uncondNoisePred = ForwardThroughUNet(uncondInput);
                }
                noisePred = ApplyGuidance(uncondNoisePred, noisePred, guidanceScale);
            }

            // DDPM step
            currentLatents = DdpmStep(currentLatents, noisePred, t, random);
        }

        return currentLatents;
    }

    /// <summary>
    /// Forwards input through U-Net layers.
    /// </summary>
    private Tensor<T> ForwardThroughUNet(Tensor<T> input)
    {
        var current = input;

        // U-Net is the middle portion of layers
        int startIdx = Layers.Count / 3;
        int endIdx = 2 * Layers.Count / 3;

        for (int i = startIdx; i < endIdx && i < Layers.Count; i++)
        {
            current = Layers[i].Forward(current);
        }

        return current;
    }

    /// <summary>
    /// Encodes CLAP embedding from tokens.
    /// </summary>
    private Tensor<T> EncodeClapEmbedding(Tensor<T> tokenTensor)
    {
        if (_clapEncoder is not null)
        {
            return _clapEncoder.Run(tokenTensor);
        }

        // Use native layers (first portion)
        var current = tokenTensor;
        int clapLayerCount = Math.Min(6, Layers.Count / 3);
        for (int i = 0; i < clapLayerCount && i < Layers.Count; i++)
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
        var melSpec = ComputeMelSpectrogram(audio);

        if (_vaeModel is not null)
        {
            return _vaeModel.Run(melSpec);
        }

        // Native encoding through layers
        var current = melSpec;
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
        Tensor<T> melSpec;

        if (_vaeModel is not null)
        {
            melSpec = _vaeModel.Run(latents);
        }
        else
        {
            // Native decoding through layers
            var current = latents;
            int startIdx = 2 * Layers.Count / 3;
            for (int i = startIdx; i < Layers.Count; i++)
            {
                current = Layers[i].Forward(current);
            }
            melSpec = current;
        }

        // Vocoder
        if (_vocoder is not null)
        {
            return _vocoder.Run(melSpec);
        }

        return ApplyGriffinLim(melSpec);
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
    /// Initializes latent noise for diffusion.
    /// </summary>
    private Tensor<T> InitializeLatents(int numFrames, Random random)
    {
        int latentChannels = _options.LatentDimension;
        int latentHeight = _options.NumMelBins / _options.LatentDownsampleFactor;

        var shape = new int[] { 1, latentChannels, latentHeight, numFrames };
        var latents = new Tensor<T>(shape);

        for (int i = 0; i < latents.Length; i++)
        {
            double u1 = 1.0 - random.NextDouble();
            double u2 = random.NextDouble();
            double gaussian = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            latents.SetFlat(i, NumOps.FromDouble(gaussian));
        }

        return latents;
    }

    /// <summary>
    /// Generates diffusion timesteps.
    /// </summary>
    private static int[] GenerateTimesteps(int numSteps)
    {
        int maxTimestep = 1000;
        var timesteps = new int[numSteps];

        for (int i = 0; i < numSteps; i++)
        {
            timesteps[i] = maxTimestep - (int)((double)i / numSteps * maxTimestep);
        }

        return timesteps;
    }

    /// <summary>
    /// Creates a timestep tensor with sinusoidal embedding.
    /// </summary>
    private Tensor<T> CreateTimestepTensor(int timestep)
    {
        int embeddingDim = _unetBaseChannels;
        var embedding = new Tensor<T>(new int[] { 1, embeddingDim });

        for (int i = 0; i < embeddingDim / 2; i++)
        {
            double freq = Math.Pow(10000, -2.0 * i / embeddingDim);
            embedding[0, i] = NumOps.FromDouble(Math.Sin(timestep * freq));
            embedding[0, i + embeddingDim / 2] = NumOps.FromDouble(Math.Cos(timestep * freq));
        }

        return embedding;
    }

    /// <summary>
    /// Prepares U-Net input by combining latents, conditioning, and timestep.
    /// </summary>
    private Tensor<T> PrepareUNetInput(Tensor<T> latents, Tensor<T> conditioning, Tensor<T> timestep)
    {
        var conditionedLatents = ApplyCrossAttentionConditioning(latents, conditioning);
        return AddTimestepEmbedding(conditionedLatents, timestep);
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
        return new Tensor<T>(new int[] { 1, _options.ClapEmbeddingDim });
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
    /// Performs a DDPM denoising step.
    /// </summary>
    private Tensor<T> DdpmStep(Tensor<T> latents, Tensor<T> noisePred, int timestep, Random random)
    {
        double beta = GetBetaSchedule(timestep);
        double alphaCumprod = GetAlphaCumprod(timestep);
        double sqrtAlphaCumprod = Math.Sqrt(alphaCumprod);
        double sqrtOneMinusAlphaCumprod = Math.Sqrt(1.0 - alphaCumprod);

        var result = new Tensor<T>(latents.Shape);

        for (int i = 0; i < result.Length; i++)
        {
            double x_t = NumOps.ToDouble(latents.GetFlat(i));
            double noise = NumOps.ToDouble(noisePred.GetFlat(i));

            double x_0 = (x_t - sqrtOneMinusAlphaCumprod * noise) / sqrtAlphaCumprod;
            x_0 = Math.Max(-1.0, Math.Min(1.0, x_0));

            double prevAlphaCumprod = timestep > 1 ? GetAlphaCumprod(timestep - 1) : 1.0;
            double sqrtPrevAlphaCumprod = Math.Sqrt(prevAlphaCumprod);
            double sqrtOneMinusPrevAlphaCumprod = Math.Sqrt(1.0 - prevAlphaCumprod);

            double mean = sqrtPrevAlphaCumprod * x_0 + sqrtOneMinusPrevAlphaCumprod * noise;

            if (timestep > 1)
            {
                double variance = beta * (1.0 - prevAlphaCumprod) / (1.0 - alphaCumprod);
                double std = Math.Sqrt(variance);
                // Use Box-Muller transform for proper Gaussian noise (DDPM requires N(0,1))
                double u1 = 1.0 - random.NextDouble(); // Avoid log(0)
                double u2 = random.NextDouble();
                double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
                mean += std * z;
            }

            result.SetFlat(i, NumOps.FromDouble(mean));
        }

        return result;
    }

    /// <summary>
    /// Gets beta value for a timestep.
    /// </summary>
    private static double GetBetaSchedule(int timestep)
    {
        double betaStart = 0.0001;
        double betaEnd = 0.02;
        int numTimesteps = 1000;

        return betaStart + (betaEnd - betaStart) * timestep / numTimesteps;
    }

    /// <summary>
    /// Gets cumulative product of alphas up to timestep.
    /// </summary>
    private static double GetAlphaCumprod(int timestep)
    {
        double alphaCumprod = 1.0;
        for (int t = 1; t <= timestep; t++)
        {
            alphaCumprod *= (1.0 - GetBetaSchedule(t));
        }
        return alphaCumprod;
    }

    /// <summary>
    /// Computes mel spectrogram from audio.
    /// </summary>
    private Tensor<T> ComputeMelSpectrogram(Tensor<T> audio)
    {
        int nFft = _options.WindowSize;
        int hopLength = _options.HopLength;
        int numMels = _options.NumMelBins;

        int numFrames = (audio.Length - nFft) / hopLength + 1;
        var melSpec = new Tensor<T>(new int[] { numMels, numFrames });

        for (int frame = 0; frame < numFrames; frame++)
        {
            int startIdx = frame * hopLength;

            var frameData = new double[nFft];
            for (int i = 0; i < nFft && startIdx + i < audio.Length; i++)
            {
                frameData[i] = NumOps.ToDouble(audio.GetFlat(startIdx + i));
            }

            // Apply Hann window
            for (int i = 0; i < nFft; i++)
            {
                double window = 0.5 * (1.0 - Math.Cos(2.0 * Math.PI * i / (nFft - 1)));
                frameData[i] *= window;
            }

            // Compute FFT using FftSharp
            var spectrum = FftSharp.FFT.Forward(frameData);

            // Compute power spectrum
            var powerSpectrum = new double[nFft / 2 + 1];
            for (int i = 0; i < powerSpectrum.Length; i++)
            {
                powerSpectrum[i] = spectrum[i].Magnitude * spectrum[i].Magnitude;
            }

            // Apply mel filterbank
            var melEnergies = ApplyMelFilterbank(powerSpectrum, numMels, _options.SampleRate, nFft);

            for (int m = 0; m < numMels; m++)
            {
                melSpec[m, frame] = NumOps.FromDouble(melEnergies[m]);
            }
        }

        return melSpec;
    }

    /// <summary>
    /// Applies mel filterbank to power spectrum.
    /// </summary>
    private static double[] ApplyMelFilterbank(double[] powerSpectrum, int numMels, int sampleRate, int nFft)
    {
        var melEnergies = new double[numMels];

        double melMin = HzToMel(0);
        double melMax = HzToMel(sampleRate / 2.0);

        var melPoints = new double[numMels + 2];
        for (int i = 0; i < melPoints.Length; i++)
        {
            melPoints[i] = melMin + (melMax - melMin) * i / (numMels + 1);
        }

        var binPoints = new int[numMels + 2];
        for (int i = 0; i < binPoints.Length; i++)
        {
            double hz = MelToHz(melPoints[i]);
            binPoints[i] = (int)Math.Floor((nFft + 1) * hz / sampleRate);
        }

        for (int m = 0; m < numMels; m++)
        {
            double energy = 0;

            for (int k = binPoints[m]; k < binPoints[m + 1] && k < powerSpectrum.Length; k++)
            {
                double weight = (double)(k - binPoints[m]) / (binPoints[m + 1] - binPoints[m]);
                energy += powerSpectrum[k] * weight;
            }

            for (int k = binPoints[m + 1]; k < binPoints[m + 2] && k < powerSpectrum.Length; k++)
            {
                double weight = (double)(binPoints[m + 2] - k) / (binPoints[m + 2] - binPoints[m + 1]);
                energy += powerSpectrum[k] * weight;
            }

            melEnergies[m] = Math.Log(Math.Max(energy, 1e-10));
        }

        return melEnergies;
    }

    private static double HzToMel(double hz) => 2595.0 * Math.Log10(1.0 + hz / 700.0);
    private static double MelToHz(double mel) => 700.0 * (Math.Pow(10.0, mel / 2595.0) - 1.0);

    /// <summary>
    /// Applies Griffin-Lim algorithm for phase reconstruction.
    /// </summary>
    private Tensor<T> ApplyGriffinLim(Tensor<T> melSpectrogram)
    {
        int numIterations = 32;
        int nFft = _options.WindowSize;
        int hopLength = _options.HopLength;

        var linearSpec = MelToLinear(melSpectrogram);

        int numFrames = linearSpec.Shape.Length > 1 ? linearSpec.Shape[1] : 1;
        int numBins = nFft / 2 + 1;
        var phase = InitializeRandomPhase(numFrames, numBins);

        for (int iter = 0; iter < numIterations; iter++)
        {
            var complexSpec = CombineMagnitudeAndPhase(linearSpec, phase);
            var waveform = InverseStft(complexSpec, nFft, hopLength);
            var newComplexSpec = ComputeStft(waveform, nFft, hopLength);
            phase = ExtractPhase(newComplexSpec);
        }

        var finalSpec = CombineMagnitudeAndPhase(linearSpec, phase);
        return InverseStft(finalSpec, nFft, hopLength);
    }

    private Tensor<T> MelToLinear(Tensor<T> melSpec)
    {
        int numBins = _options.WindowSize / 2 + 1;
        int numFrames = melSpec.Shape.Length > 1 ? melSpec.Shape[1] : 1;
        int numMels = melSpec.Shape[0];

        var linearSpec = new Tensor<T>(new int[] { numBins, numFrames });

        for (int frame = 0; frame < numFrames; frame++)
        {
            for (int bin = 0; bin < numBins; bin++)
            {
                double hz = (double)bin * _options.SampleRate / _options.WindowSize;
                double mel = HzToMel(hz);
                double melMax = HzToMel(_options.SampleRate / 2.0);
                int melIdx = (int)(mel / melMax * (numMels - 1));
                melIdx = Math.Max(0, Math.Min(numMels - 1, melIdx));

                linearSpec[bin, frame] = melSpec[melIdx, frame];
            }
        }

        return linearSpec;
    }

    private Tensor<T> InitializeRandomPhase(int numFrames, int numBins)
    {
        var phase = new Tensor<T>(new int[] { numBins, numFrames });
        for (int i = 0; i < phase.Length; i++)
        {
            phase.SetFlat(i, NumOps.FromDouble(_random.NextDouble() * 2 * Math.PI - Math.PI));
        }
        return phase;
    }

    private Tensor<T> CombineMagnitudeAndPhase(Tensor<T> magnitude, Tensor<T> phase)
    {
        var complex = new Tensor<T>(new int[] { magnitude.Shape[0], magnitude.Shape[1], 2 });

        for (int bin = 0; bin < magnitude.Shape[0]; bin++)
        {
            for (int frame = 0; frame < magnitude.Shape[1]; frame++)
            {
                double mag = NumOps.ToDouble(magnitude[bin, frame]);
                double ph = NumOps.ToDouble(phase[bin, frame]);
                complex[bin, frame, 0] = NumOps.FromDouble(mag * Math.Cos(ph));
                complex[bin, frame, 1] = NumOps.FromDouble(mag * Math.Sin(ph));
            }
        }

        return complex;
    }

    private Tensor<T> ComputeStft(Tensor<T> waveform, int nFft, int hopLength)
    {
        int numFrames = (waveform.Length - nFft) / hopLength + 1;
        int numBins = nFft / 2 + 1;
        var stft = new Tensor<T>(new int[] { numBins, numFrames, 2 });

        for (int frame = 0; frame < numFrames; frame++)
        {
            int startIdx = frame * hopLength;
            var frameData = new double[nFft];
            for (int i = 0; i < nFft && startIdx + i < waveform.Length; i++)
            {
                double window = 0.5 * (1.0 - Math.Cos(2.0 * Math.PI * i / (nFft - 1)));
                frameData[i] = NumOps.ToDouble(waveform.GetFlat(startIdx + i)) * window;
            }

            var spectrum = FftSharp.FFT.Forward(frameData);
            for (int bin = 0; bin < numBins; bin++)
            {
                stft[bin, frame, 0] = NumOps.FromDouble(spectrum[bin].Real);
                stft[bin, frame, 1] = NumOps.FromDouble(spectrum[bin].Imaginary);
            }
        }

        return stft;
    }

    private Tensor<T> ExtractPhase(Tensor<T> complexSpec)
    {
        var phase = new Tensor<T>(new int[] { complexSpec.Shape[0], complexSpec.Shape[1] });

        for (int bin = 0; bin < complexSpec.Shape[0]; bin++)
        {
            for (int frame = 0; frame < complexSpec.Shape[1]; frame++)
            {
                double real = NumOps.ToDouble(complexSpec[bin, frame, 0]);
                double imag = NumOps.ToDouble(complexSpec[bin, frame, 1]);
                phase[bin, frame] = NumOps.FromDouble(Math.Atan2(imag, real));
            }
        }

        return phase;
    }

    private Tensor<T> InverseStft(Tensor<T> complexSpec, int nFft, int hopLength)
    {
        int numFrames = complexSpec.Shape[1];
        int outputLength = (numFrames - 1) * hopLength + nFft;
        var waveform = new Tensor<T>(new int[] { outputLength });
        var windowSum = new double[outputLength];

        for (int frame = 0; frame < numFrames; frame++)
        {
            var fullSpectrum = new System.Numerics.Complex[nFft];
            int numBins = complexSpec.Shape[0];

            for (int bin = 0; bin < numBins; bin++)
            {
                double real = NumOps.ToDouble(complexSpec[bin, frame, 0]);
                double imag = NumOps.ToDouble(complexSpec[bin, frame, 1]);
                fullSpectrum[bin] = new System.Numerics.Complex(real, imag);

                if (bin > 0 && bin < numBins - 1)
                {
                    fullSpectrum[nFft - bin] = new System.Numerics.Complex(real, -imag);
                }
            }

            FftSharp.FFT.Inverse(fullSpectrum);

            int startIdx = frame * hopLength;
            for (int i = 0; i < nFft && startIdx + i < outputLength; i++)
            {
                double window = 0.5 * (1.0 - Math.Cos(2.0 * Math.PI * i / (nFft - 1)));
                double current = NumOps.ToDouble(waveform.GetFlat(startIdx + i));
                waveform.SetFlat(startIdx + i, NumOps.FromDouble(current + fullSpectrum[i].Real * window));
                windowSum[startIdx + i] += window * window;
            }
        }

        for (int i = 0; i < outputLength; i++)
        {
            if (windowSum[i] > 1e-8)
            {
                double current = NumOps.ToDouble(waveform.GetFlat(i));
                waveform.SetFlat(i, NumOps.FromDouble(current / windowSum[i]));
            }
        }

        return waveform;
    }

    #endregion

    #region NeuralNetworkBase Implementation

    /// <summary>
    /// Preprocesses raw audio for model input.
    /// </summary>
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio)
    {
        return ComputeMelSpectrogram(rawAudio);
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

        if (_clapEncoder is null)
            throw new InvalidOperationException("CLAP encoder not loaded.");

        return _clapEncoder.Run(input);
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
            Name = $"AudioLDM-{_options.ModelSize}",
            Description = $"AudioLDM text-to-audio model ({_options.ModelSize})",
            ModelType = ModelType.NeuralNetwork,
            FeatureCount = _options.MaxTextLength,
            Complexity = (int)_options.ModelSize,
            AdditionalInfo = new Dictionary<string, object>
            {
                ["SampleRate"] = _options.SampleRate,
                ["MaxDuration"] = _options.MaxDurationSeconds,
                ["NumInferenceSteps"] = _options.NumInferenceSteps,
                ["GuidanceScale"] = _options.GuidanceScale,
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
        writer.Write(_options.NumInferenceSteps);
        writer.Write(_options.GuidanceScale);
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
    }

    /// <summary>
    /// Creates a new instance for cloning.
    /// </summary>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new AudioLDMModel<T>(
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
            throw new ObjectDisposedException(nameof(AudioLDMModel<T>));
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
                _clapEncoder?.Dispose();
                _vaeModel?.Dispose();
                _unetDenoiser?.Dispose();
                _vocoder?.Dispose();
            }
            _disposed = true;
        }
        base.Dispose(disposing);
    }

    #endregion
}
