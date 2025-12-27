using AiDotNet.ActivationFunctions;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Diffusion.Schedulers;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Diffusion.Models;

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
    /// <summary>
    /// Standard latent channels for Riffusion (same as SD).
    /// </summary>
    private const int RIFF_LATENT_CHANNELS = 4;

    /// <summary>
    /// Standard VAE scale factor.
    /// </summary>
    private const int RIFF_VAE_SCALE_FACTOR = 8;

    /// <summary>
    /// Default spectrogram size.
    /// </summary>
    private const int DEFAULT_SPEC_SIZE = 512;

    /// <summary>
    /// The U-Net noise predictor.
    /// </summary>
    private readonly UNetNoisePredictor<T> _unet;

    /// <summary>
    /// The VAE for encoding/decoding spectrograms.
    /// </summary>
    private readonly StandardVAE<T> _vae;

    /// <summary>
    /// The text conditioning module.
    /// </summary>
    private readonly IConditioningModule<T>? _conditioner;

    /// <summary>
    /// Spectrogram configuration.
    /// </summary>
    private readonly SpectrogramConfig _spectrogramConfig;

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

    /// <summary>
    /// Initializes a new instance of RiffusionModel with default parameters.
    /// </summary>
    public RiffusionModel()
        : this(
            options: null,
            scheduler: null,
            conditioner: null,
            spectrogramConfig: null,
            seed: null)
    {
    }

    /// <summary>
    /// Initializes a new instance of RiffusionModel with custom parameters.
    /// </summary>
    /// <param name="options">Configuration options.</param>
    /// <param name="scheduler">Optional custom scheduler.</param>
    /// <param name="unet">Optional custom U-Net.</param>
    /// <param name="vae">Optional custom VAE.</param>
    /// <param name="conditioner">Optional text conditioning module.</param>
    /// <param name="spectrogramConfig">Spectrogram configuration.</param>
    /// <param name="seed">Optional random seed.</param>
    public RiffusionModel(
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        UNetNoisePredictor<T>? unet = null,
        StandardVAE<T>? vae = null,
        IConditioningModule<T>? conditioner = null,
        SpectrogramConfig? spectrogramConfig = null,
        int? seed = null)
        : base(options ?? CreateDefaultOptions(), scheduler ?? CreateDefaultScheduler())
    {
        _conditioner = conditioner;
        _spectrogramConfig = spectrogramConfig ?? new SpectrogramConfig();

        // Create U-Net
        _unet = unet ?? CreateDefaultUNet(seed);

        // Create VAE
        _vae = vae ?? CreateDefaultVAE(seed);
    }

    /// <summary>
    /// Creates the default options.
    /// </summary>
    private static DiffusionModelOptions<T> CreateDefaultOptions()
    {
        return new DiffusionModelOptions<T>
        {
            TrainTimesteps = 1000,
            BetaStart = 0.00085,
            BetaEnd = 0.012,
            BetaSchedule = BetaSchedule.ScaledLinear
        };
    }

    /// <summary>
    /// Creates the default scheduler.
    /// </summary>
    private static INoiseScheduler<T> CreateDefaultScheduler()
    {
        var config = SchedulerConfig<T>.CreateStableDiffusion();
        return new DDIMScheduler<T>(config);
    }

    /// <summary>
    /// Creates the default U-Net.
    /// </summary>
    private UNetNoisePredictor<T> CreateDefaultUNet(int? seed)
    {
        return new UNetNoisePredictor<T>(
            inputChannels: RIFF_LATENT_CHANNELS,
            outputChannels: RIFF_LATENT_CHANNELS,
            baseChannels: 320,
            channelMultipliers: new[] { 1, 2, 4, 4 },
            numResBlocks: 2,
            attentionResolutions: new[] { 4, 2, 1 },
            contextDim: 768,
            seed: seed);
    }

    /// <summary>
    /// Creates the default VAE.
    /// </summary>
    private StandardVAE<T> CreateDefaultVAE(int? seed)
    {
        return new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: RIFF_LATENT_CHANNELS,
            baseChannels: 128,
            channelMultipliers: new[] { 1, 2, 4, 4 },
            numResBlocksPerLevel: 2,
            seed: seed);
    }

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
        var height = spectrogram.Shape.Length > 2 ? spectrogram.Shape[^2] : spectrogram.Shape[0];
        var width = spectrogram.Shape[^1];

        // Calculate audio length
        var audioLength = CalculateAudioLength(width);

        // Apply Griffin-Lim algorithm (simplified)
        return GriffinLim(spectrogram, audioLength);
    }

    /// <summary>
    /// Griffin-Lim algorithm for spectrogram inversion.
    /// Iteratively estimates phase from magnitude spectrogram.
    /// </summary>
    private Tensor<T> GriffinLim(Tensor<T> spectrogram, int audioLength)
    {
        var numIterations = 32;
        var hopLength = _spectrogramConfig.HopLength;
        var fftSize = _spectrogramConfig.FFTSize;
        var numBins = fftSize / 2 + 1;

        // Extract spectrogram dimensions
        var specShape = spectrogram.Shape;
        var numMelBins = specShape.Length > 2 ? specShape[^2] : specShape[0];
        var numFrames = specShape[^1];

        var specSpan = spectrogram.AsSpan();

        // Convert mel spectrogram to linear spectrogram (approximation)
        var magnitudes = new double[numFrames, numBins];
        for (int frame = 0; frame < numFrames; frame++)
        {
            for (int bin = 0; bin < numBins; bin++)
            {
                // Map linear bin to mel bin
                var melBin = (int)(bin * (double)numMelBins / numBins);
                melBin = Math.Min(melBin, numMelBins - 1);

                var specIdx = frame + melBin * numFrames;
                if (specIdx < specSpan.Length)
                {
                    var val = NumOps.ToDouble(specSpan[specIdx]);
                    // Convert from log scale if needed
                    magnitudes[frame, bin] = _spectrogramConfig.UseLogScale
                        ? Math.Exp(val) - 1e-5
                        : val;
                }
            }
        }

        // Initialize audio signal
        var audioData = new double[audioLength];

        // Initialize phases randomly
        var rng = RandomGenerator;
        var phases = new double[numFrames, numBins];
        for (int frame = 0; frame < numFrames; frame++)
        {
            for (int bin = 0; bin < numBins; bin++)
            {
                phases[frame, bin] = rng.NextDouble() * 2 * Math.PI;
            }
        }

        // Create analysis window (Hann window)
        var window = new double[fftSize];
        for (int i = 0; i < fftSize; i++)
        {
            window[i] = 0.5 * (1 - Math.Cos(2 * Math.PI * i / (fftSize - 1)));
        }

        // Griffin-Lim iterations
        for (int iter = 0; iter < numIterations; iter++)
        {
            // Inverse STFT: Reconstruct audio from magnitudes and phases
            Array.Clear(audioData, 0, audioData.Length);
            var windowSum = new double[audioLength];

            for (int frame = 0; frame < numFrames; frame++)
            {
                var frameStart = frame * hopLength;

                // Inverse DFT for this frame
                var frameData = new double[fftSize];
                for (int n = 0; n < fftSize; n++)
                {
                    double sum = 0;
                    for (int k = 0; k < numBins; k++)
                    {
                        var mag = magnitudes[frame, k];
                        var phase = phases[frame, k];
                        var angle = 2 * Math.PI * k * n / fftSize;
                        sum += mag * Math.Cos(angle + phase);
                    }
                    frameData[n] = sum / fftSize;
                }

                // Apply window and overlap-add
                for (int n = 0; n < fftSize; n++)
                {
                    var idx = frameStart + n;
                    if (idx < audioLength)
                    {
                        audioData[idx] += frameData[n] * window[n];
                        windowSum[idx] += window[n] * window[n];
                    }
                }
            }

            // Normalize by window sum
            for (int i = 0; i < audioLength; i++)
            {
                if (windowSum[i] > 1e-8)
                {
                    audioData[i] /= windowSum[i];
                }
            }

            // Forward STFT: Extract new phases from reconstructed audio
            if (iter < numIterations - 1)
            {
                for (int frame = 0; frame < numFrames; frame++)
                {
                    var frameStart = frame * hopLength;

                    // Extract windowed frame
                    var frameData = new double[fftSize];
                    for (int n = 0; n < fftSize; n++)
                    {
                        var idx = frameStart + n;
                        if (idx < audioLength)
                        {
                            frameData[n] = audioData[idx] * window[n];
                        }
                    }

                    // DFT to get new phases
                    for (int k = 0; k < numBins; k++)
                    {
                        double real = 0, imag = 0;
                        for (int n = 0; n < fftSize; n++)
                        {
                            var angle = -2 * Math.PI * k * n / fftSize;
                            real += frameData[n] * Math.Cos(angle);
                            imag += frameData[n] * Math.Sin(angle);
                        }
                        phases[frame, k] = Math.Atan2(imag, real);
                    }
                }
            }
        }

        // Convert to output format
        var result = new T[audioLength];
        for (int i = 0; i < audioLength; i++)
        {
            result[i] = NumOps.FromDouble(audioData[i]);
        }

        return new Tensor<T>(new[] { 1, audioLength }, new Vector<T>(result));
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
        var result = new Tensor<T>(a.Shape);
        var aSpan = a.AsSpan();
        var bSpan = b.AsSpan();
        var resultSpan = result.AsWritableSpan();

        var minLen = Math.Min(aSpan.Length, bSpan.Length);
        for (int i = 0; i < minLen; i++)
        {
            var valA = NumOps.ToDouble(aSpan[i]);
            var valB = NumOps.ToDouble(bSpan[i]);
            resultSpan[i] = NumOps.FromDouble(valA * (1 - alpha) + valB * alpha);
        }

        return result;
    }

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
