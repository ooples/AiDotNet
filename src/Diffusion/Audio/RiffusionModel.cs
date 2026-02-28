using System.Diagnostics.CodeAnalysis;
using AiDotNet.ActivationFunctions;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Diffusion.Schedulers;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Diffusion.Audio;

/// <summary>
/// Riffusion model for music generation via spectrogram diffusion.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Riffusion generates music by treating audio spectrograms as images and using
/// Stable Diffusion to generate them. The resulting spectrograms are then converted
/// back to audio using the Griffin-Lim algorithm or neural vocoders.
/// </para>
/// <para>
/// <b>For Beginners:</b> Riffusion creates music by first generating a "picture" of
/// the sound (spectrogram), then converting that picture back into actual audio.
///
/// How it works:
/// 1. You describe the music you want: "jazz piano solo"
/// 2. Riffusion generates a spectrogram (visual representation of sound)
/// 3. The spectrogram is converted to playable audio
///
/// Key features:
/// - Text-to-music generation
/// - Style interpolation (blend two music styles)
/// - Real-time streaming generation
/// - Works with any Stable Diffusion checkpoint
///
/// What makes it unique:
/// - Treats audio generation as an image generation problem
/// - Can leverage all SD techniques: ControlNet, img2img, etc.
/// - Fast inference compared to autoregressive music models
/// </para>
/// <para>
/// Technical details:
/// - Uses mel-spectrograms with specific parameters
/// - Typically 512x512 spectrogram images
/// - Griffin-Lim or neural vocoder for audio reconstruction
/// - Supports seed-based interpolation for smooth transitions
/// - Compatible with LoRA adapters for style transfer
///
/// Reference: Based on Riffusion project (riffusion.com)
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a Riffusion model
/// var riffusion = new RiffusionModel&lt;float&gt;();
///
/// // Generate music from text
/// var spectrogram = riffusion.GenerateSpectrogram(
///     prompt: "jazz piano solo, smooth and relaxing",
///     durationSeconds: 5.0);
///
/// // Convert to audio
/// var audio = riffusion.SpectrogramToAudio(spectrogram);
///
/// // Interpolate between two styles
/// var interpolated = riffusion.InterpolateStyles(
///     promptA: "upbeat electronic dance music",
///     promptB: "calm ambient soundscape",
///     alpha: 0.5);
/// </code>
/// </example>
public class RiffusionModel<T> : LatentDiffusionModelBase<T>
{
    #region Constants

    /// <summary>
    /// Number of latent channels for Riffusion (same as SD 1.5).
    /// </summary>
    /// <remarks>
    /// Riffusion reuses Stable Diffusion's 4-channel latent space since it
    /// treats spectrograms as images.
    /// </remarks>
    private const int RIFF_LATENT_CHANNELS = 4;

    /// <summary>
    /// Spatial downsampling factor of the VAE.
    /// </summary>
    /// <remarks>
    /// Standard 8x downsampling inherited from Stable Diffusion's VAE architecture.
    /// </remarks>
    private const int RIFF_VAE_SCALE_FACTOR = 8;

    /// <summary>
    /// Default spectrogram size in pixels.
    /// </summary>
    /// <remarks>
    /// 512x512 spectrograms provide a good balance between audio quality and
    /// generation speed, matching SD 1.5's native resolution.
    /// </remarks>
    private const int DEFAULT_SPEC_SIZE = 512;

    #endregion

    #region Fields

    /// <summary>
    /// The U-Net noise predictor.
    /// </summary>
    private UNetNoisePredictor<T> _unet;

    /// <summary>
    /// The VAE for encoding/decoding spectrograms.
    /// </summary>
    private StandardVAE<T> _vae;

    /// <summary>
    /// The text conditioning module.
    /// </summary>
    private readonly IConditioningModule<T>? _conditioner;

    /// <summary>
    /// Spectrogram configuration.
    /// </summary>
    private readonly SpectrogramConfig _spectrogramConfig;

    /// <summary>
    /// GPU-accelerated Griffin-Lim processor for spectrogram inversion.
    /// </summary>
    private GriffinLim<T> _griffinLim;

    /// <summary>
    /// GPU-accelerated mel spectrogram processor.
    /// </summary>
    private MelSpectrogram<T> _melSpectrogram;

    #endregion

    #region Properties

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _unet;

    /// <inheritdoc />
    public override IVAEModel<T> VAE => _vae;

    /// <inheritdoc />
    public override IConditioningModule<T>? Conditioner => _conditioner;

    /// <inheritdoc />
    public override int LatentChannels => RIFF_LATENT_CHANNELS;

    /// <inheritdoc />
    public override int ParameterCount => _unet.ParameterCount + _vae.ParameterCount;

    /// <summary>
    /// Gets the spectrogram configuration.
    /// </summary>
    public SpectrogramConfig SpectrogramConfiguration => _spectrogramConfig;

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes a new instance of RiffusionModel with full customization support.
    /// </summary>
    /// <param name="architecture">Optional neural network architecture for custom layer configuration.</param>
    /// <param name="options">Configuration options.</param>
    /// <param name="scheduler">Optional custom scheduler.</param>
    /// <param name="unet">Optional custom U-Net.</param>
    /// <param name="vae">Optional custom VAE.</param>
    /// <param name="conditioner">Optional text conditioning module.</param>
    /// <param name="spectrogramConfig">Spectrogram configuration.</param>
    /// <param name="seed">Optional random seed.</param>
    public RiffusionModel(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        UNetNoisePredictor<T>? unet = null,
        StandardVAE<T>? vae = null,
        IConditioningModule<T>? conditioner = null,
        SpectrogramConfig? spectrogramConfig = null,
        int? seed = null)
        : base(
            options ?? new DiffusionModelOptions<T>
            {
                TrainTimesteps = 1000,
                BetaStart = 0.00085,
                BetaEnd = 0.012,
                BetaSchedule = BetaSchedule.ScaledLinear
            },
            scheduler ?? new DDIMScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion()),
            architecture)
    {
        _conditioner = conditioner;
        _spectrogramConfig = spectrogramConfig ?? new SpectrogramConfig();

        InitializeLayers(unet, vae, seed);
    }

    #endregion

    #region Layer Initialization

    /// <summary>
    /// Initializes the U-Net, VAE, and audio processing components.
    /// </summary>
    [MemberNotNull(nameof(_unet), nameof(_vae), nameof(_griffinLim), nameof(_melSpectrogram))]
    private void InitializeLayers(
        UNetNoisePredictor<T>? unet,
        StandardVAE<T>? vae,
        int? seed)
    {
        // Standard SD 1.5 U-Net for spectrogram generation
        _unet = unet ?? new UNetNoisePredictor<T>(
            inputChannels: RIFF_LATENT_CHANNELS,
            outputChannels: RIFF_LATENT_CHANNELS,
            baseChannels: 320,
            channelMultipliers: new[] { 1, 2, 4, 4 },
            numResBlocks: 2,
            attentionResolutions: new[] { 4, 2, 1 },
            contextDim: 768,
            architecture: Architecture,
            seed: seed);

        // Standard SD 1.5 VAE
        _vae = vae ?? new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: RIFF_LATENT_CHANNELS,
            baseChannels: 128,
            channelMultipliers: new[] { 1, 2, 4, 4 },
            numResBlocksPerLevel: 2,
            seed: seed);

        // GPU-accelerated Griffin-Lim for spectrogram inversion
        _griffinLim = new GriffinLim<T>(
            nFft: _spectrogramConfig.FFTSize,
            hopLength: _spectrogramConfig.HopLength,
            iterations: 32,
            momentum: 0.99,
            seed: seed);

        // GPU-accelerated mel spectrogram processor
        _melSpectrogram = new MelSpectrogram<T>(
            sampleRate: _spectrogramConfig.SampleRate,
            nFft: _spectrogramConfig.FFTSize,
            hopLength: _spectrogramConfig.HopLength,
            nMels: _spectrogramConfig.NumMelBins,
            fMin: _spectrogramConfig.MinFrequency,
            fMax: _spectrogramConfig.MaxFrequency,
            logMel: _spectrogramConfig.UseLogScale);
    }

    #endregion

    #region Generation Methods

    /// <summary>
    /// Generates a spectrogram from a text prompt.
    /// </summary>
    /// <param name="prompt">Text description of the desired music.</param>
    /// <param name="negativePrompt">Optional negative prompt.</param>
    /// <param name="durationSeconds">Desired audio duration in seconds.</param>
    /// <param name="numInferenceSteps">Number of denoising steps.</param>
    /// <param name="guidanceScale">Classifier-free guidance scale.</param>
    /// <param name="seed">Optional random seed.</param>
    /// <returns>Generated spectrogram tensor.</returns>
    public virtual Tensor<T> GenerateSpectrogram(
        string prompt,
        string? negativePrompt = null,
        double durationSeconds = 5.0,
        int numInferenceSteps = 50,
        double? guidanceScale = null,
        int? seed = null)
    {
        // Calculate spectrogram dimensions based on duration
        var width = CalculateSpectrogramWidth(durationSeconds);
        var height = _spectrogramConfig.NumMelBins;

        return GenerateSpectrogramInternal(
            prompt,
            negativePrompt,
            width,
            height,
            numInferenceSteps,
            guidanceScale,
            seed);
    }

    /// <summary>
    /// Generates a spectrogram with specific dimensions.
    /// </summary>
    private Tensor<T> GenerateSpectrogramInternal(
        string prompt,
        string? negativePrompt,
        int width,
        int height,
        int numInferenceSteps,
        double? guidanceScale,
        int? seed)
    {
        // Get text conditioning
        Tensor<T>? promptEmbedding = null;
        Tensor<T>? negativeEmbedding = null;
        var effectiveGuidanceScale = guidanceScale ?? GuidanceScale;

        if (_conditioner != null)
        {
            var promptTokens = _conditioner.Tokenize(prompt);
            promptEmbedding = _conditioner.EncodeText(promptTokens);

            if (effectiveGuidanceScale > 1.0)
            {
                if (!string.IsNullOrEmpty(negativePrompt))
                {
                    var negTokens = _conditioner.Tokenize(negativePrompt ?? string.Empty);
                    negativeEmbedding = _conditioner.EncodeText(negTokens);
                }
                else
                {
                    negativeEmbedding = _conditioner.GetUnconditionalEmbedding(1);
                }
            }
        }

        // Calculate latent dimensions
        var latentHeight = height / RIFF_VAE_SCALE_FACTOR;
        var latentWidth = width / RIFF_VAE_SCALE_FACTOR;
        var latentShape = new[] { 1, RIFF_LATENT_CHANNELS, latentHeight, latentWidth };

        // Initialize noise
        var rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomGenerator;
        var latents = SampleNoiseTensor(latentShape, rng);

        // Set up scheduler
        Scheduler.SetTimesteps(numInferenceSteps);

        // Denoising loop
        foreach (var timestep in Scheduler.Timesteps)
        {
            Tensor<T> noisePrediction;

            if (effectiveGuidanceScale > 1.0 && negativeEmbedding != null && promptEmbedding != null)
            {
                var condPred = _unet.PredictNoise(latents, timestep, promptEmbedding);
                var uncondPred = _unet.PredictNoise(latents, timestep, negativeEmbedding);
                noisePrediction = ApplyGuidance(uncondPred, condPred, effectiveGuidanceScale);
            }
            else
            {
                noisePrediction = _unet.PredictNoise(latents, timestep, promptEmbedding);
            }

            var latentVector = latents.ToVector();
            var noiseVector = noisePrediction.ToVector();
            latentVector = Scheduler.Step(noiseVector, timestep, latentVector, NumOps.Zero);
            latents = new Tensor<T>(latentShape, latentVector);
        }

        // Decode to spectrogram image
        return DecodeFromLatent(latents);
    }

    /// <summary>
    /// Converts a spectrogram to audio waveform.
    /// </summary>
    /// <param name="spectrogram">Input spectrogram tensor.</param>
    /// <returns>Audio waveform tensor.</returns>
    public virtual Tensor<T> SpectrogramToAudio(Tensor<T> spectrogram)
    {
        // Extract spectrogram dimensions
        var width = spectrogram.Shape[^1];

        // Calculate audio length
        var audioLength = CalculateAudioLength(width);

        // First invert mel spectrogram to linear magnitude spectrogram
        var magnitude = _melSpectrogram.InvertMelToMagnitude(spectrogram);

        // Use GPU-accelerated Griffin-Lim for phase reconstruction
        return _griffinLim.Reconstruct(magnitude, audioLength);
    }

    /// <summary>
    /// Interpolates between two music styles.
    /// </summary>
    /// <param name="promptA">First style description.</param>
    /// <param name="promptB">Second style description.</param>
    /// <param name="alpha">Interpolation factor (0 = promptA, 1 = promptB).</param>
    /// <param name="durationSeconds">Audio duration.</param>
    /// <param name="numInferenceSteps">Denoising steps.</param>
    /// <param name="guidanceScale">Guidance scale.</param>
    /// <param name="seed">Random seed.</param>
    /// <returns>Interpolated spectrogram.</returns>
    public virtual Tensor<T> InterpolateStyles(
        string promptA,
        string promptB,
        double alpha = 0.5,
        double durationSeconds = 5.0,
        int numInferenceSteps = 50,
        double? guidanceScale = null,
        int? seed = null)
    {
        alpha = MathPolyfill.Clamp(alpha, 0.0, 1.0);

        if (_conditioner == null)
        {
            // Without conditioning, just generate with promptA
            return GenerateSpectrogram(promptA, null, durationSeconds, numInferenceSteps, guidanceScale, seed);
        }

        // Get embeddings for both prompts
        var tokensA = _conditioner.Tokenize(promptA);
        var tokensB = _conditioner.Tokenize(promptB);
        var embedA = _conditioner.EncodeText(tokensA);
        var embedB = _conditioner.EncodeText(tokensB);

        // Interpolate embeddings
        var interpolatedEmbed = InterpolateTensors(embedA, embedB, alpha);

        // Calculate dimensions
        var width = CalculateSpectrogramWidth(durationSeconds);
        var height = _spectrogramConfig.NumMelBins;
        var latentHeight = height / RIFF_VAE_SCALE_FACTOR;
        var latentWidth = width / RIFF_VAE_SCALE_FACTOR;
        var latentShape = new[] { 1, RIFF_LATENT_CHANNELS, latentHeight, latentWidth };

        // Initialize noise
        var rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomGenerator;
        var latents = SampleNoiseTensor(latentShape, rng);

        var effectiveGuidanceScale = guidanceScale ?? GuidanceScale;
        var uncondEmbed = _conditioner.GetUnconditionalEmbedding(1);

        // Set up scheduler
        Scheduler.SetTimesteps(numInferenceSteps);

        // Denoising loop with interpolated embedding
        foreach (var timestep in Scheduler.Timesteps)
        {
            Tensor<T> noisePrediction;

            if (effectiveGuidanceScale > 1.0)
            {
                var condPred = _unet.PredictNoise(latents, timestep, interpolatedEmbed);
                var uncondPred = _unet.PredictNoise(latents, timestep, uncondEmbed);
                noisePrediction = ApplyGuidance(uncondPred, condPred, effectiveGuidanceScale);
            }
            else
            {
                noisePrediction = _unet.PredictNoise(latents, timestep, interpolatedEmbed);
            }

            var latentVector = latents.ToVector();
            var noiseVector = noisePrediction.ToVector();
            latentVector = Scheduler.Step(noiseVector, timestep, latentVector, NumOps.Zero);
            latents = new Tensor<T>(latentShape, latentVector);
        }

        return DecodeFromLatent(latents);
    }

    /// <summary>
    /// Generates audio directly from a text prompt.
    /// </summary>
    /// <param name="prompt">Text description of the desired music.</param>
    /// <param name="negativePrompt">Optional negative prompt.</param>
    /// <param name="durationSeconds">Desired audio duration.</param>
    /// <param name="numInferenceSteps">Denoising steps.</param>
    /// <param name="guidanceScale">Guidance scale.</param>
    /// <param name="seed">Random seed.</param>
    /// <returns>Audio waveform tensor.</returns>
    public virtual Tensor<T> GenerateAudio(
        string prompt,
        string? negativePrompt = null,
        double durationSeconds = 5.0,
        int numInferenceSteps = 50,
        double? guidanceScale = null,
        int? seed = null)
    {
        var spectrogram = GenerateSpectrogram(
            prompt,
            negativePrompt,
            durationSeconds,
            numInferenceSteps,
            guidanceScale,
            seed);

        return SpectrogramToAudio(spectrogram);
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Calculates spectrogram width from duration.
    /// </summary>
    private int CalculateSpectrogramWidth(double durationSeconds)
    {
        var samplesNeeded = (int)(durationSeconds * _spectrogramConfig.SampleRate);
        var framesNeeded = samplesNeeded / _spectrogramConfig.HopLength;
        // Round up to nearest power of 2 for FFT efficiency
        return Math.Max(64, (int)Math.Pow(2, Math.Ceiling(MathPolyfill.Log2(framesNeeded))));
    }

    /// <summary>
    /// Calculates audio length from spectrogram width.
    /// </summary>
    private int CalculateAudioLength(int spectrogramWidth)
    {
        return spectrogramWidth * _spectrogramConfig.HopLength;
    }

    /// <summary>
    /// Interpolates between two tensors.
    /// </summary>
    private Tensor<T> InterpolateTensors(Tensor<T> a, Tensor<T> b, double alpha)
    {
        // result = (1 - alpha) * a + alpha * b
        var oneMinusAlphaT = NumOps.FromDouble(1.0 - alpha);
        var alphaT = NumOps.FromDouble(alpha);
        var scaledA = Engine.TensorMultiplyScalar<T>(a, oneMinusAlphaT);
        var scaledB = Engine.TensorMultiplyScalar<T>(b, alphaT);
        return Engine.TensorAdd<T>(scaledA, scaledB);
    }

    #endregion

    #region IParameterizable Implementation

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var unetParams = _unet.GetParameters();
        var vaeParams = _vae.GetParameters();

        var totalLength = unetParams.Length + vaeParams.Length;
        var combined = new T[totalLength];

        for (int i = 0; i < unetParams.Length; i++)
        {
            combined[i] = unetParams[i];
        }
        for (int i = 0; i < vaeParams.Length; i++)
        {
            combined[unetParams.Length + i] = vaeParams[i];
        }

        return new Vector<T>(combined);
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        var unetCount = _unet.ParameterCount;
        var vaeCount = _vae.ParameterCount;

        if (parameters.Length != unetCount + vaeCount)
        {
            throw new ArgumentException(
                $"Expected {unetCount + vaeCount} parameters, got {parameters.Length}.",
                nameof(parameters));
        }

        var unetParams = new T[unetCount];
        var vaeParams = new T[vaeCount];

        for (int i = 0; i < unetCount; i++)
        {
            unetParams[i] = parameters[i];
        }
        for (int i = 0; i < vaeCount; i++)
        {
            vaeParams[i] = parameters[unetCount + i];
        }

        _unet.SetParameters(new Vector<T>(unetParams));
        _vae.SetParameters(new Vector<T>(vaeParams));
    }

    #endregion

    #region ICloneable Implementation

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy()
    {
        return Clone();
    }

    /// <inheritdoc />
    public override IDiffusionModel<T> Clone()
    {
        var clone = new RiffusionModel<T>(
            conditioner: _conditioner,
            spectrogramConfig: _spectrogramConfig,
            seed: RandomGenerator.Next());

        clone.SetParameters(GetParameters());
        return clone;
    }

    #endregion
}

/// <summary>
/// Configuration for spectrogram generation.
/// </summary>
public class SpectrogramConfig
{
    /// <summary>
    /// Audio sample rate in Hz.
    /// </summary>
    public int SampleRate { get; set; } = 22050;

    /// <summary>
    /// Number of mel frequency bins.
    /// </summary>
    public int NumMelBins { get; set; } = 512;

    /// <summary>
    /// FFT window size.
    /// </summary>
    public int FFTSize { get; set; } = 2048;

    /// <summary>
    /// Hop length between frames.
    /// </summary>
    public int HopLength { get; set; } = 512;

    /// <summary>
    /// Minimum frequency in Hz.
    /// </summary>
    public double MinFrequency { get; set; } = 20.0;

    /// <summary>
    /// Maximum frequency in Hz.
    /// </summary>
    public double MaxFrequency { get; set; } = 8000.0;

    /// <summary>
    /// Whether to use log-scale magnitude.
    /// </summary>
    public bool UseLogScale { get; set; } = true;
}
