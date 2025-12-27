using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Diffusion;
using AiDotNet.NeuralNetworks.Diffusion.Schedulers;

namespace AiDotNet.Diffusion.Models;

/// <summary>
/// Audio Latent Diffusion Model (AudioLDM) for text-to-audio generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// AudioLDM is a latent diffusion model specifically designed for audio generation.
/// It works by generating mel spectrograms in latent space and then converting
/// them to audio using a vocoder (like HiFi-GAN).
/// </para>
/// <para>
/// <b>For Beginners:</b> AudioLDM lets you create sounds and music from text descriptions:
///
/// Example prompts:
/// - "A dog barking in a park" -> generates dog barking sounds
/// - "Rain falling on a window" -> generates rain sounds
/// - "Jazz piano playing softly" -> generates jazz piano music
///
/// How it works:
/// 1. Text -> CLAP encoder -> text embedding (understands audio concepts)
/// 2. Text embedding guides diffusion in latent space
/// 3. Latent -> AudioVAE decoder -> mel spectrogram
/// 4. Mel spectrogram -> Vocoder -> audio waveform
///
/// Key features:
/// - Text-to-audio: Generate sounds from descriptions
/// - Audio-to-audio: Transform sounds while preserving some characteristics
/// - Variable duration: Generate audio of different lengths
/// - Classifier-free guidance: Control how closely to follow the prompt
/// </para>
/// <para>
/// Technical specifications:
/// - Sample rate: 16 kHz (standard for speech/effects) or 48 kHz (music)
/// - Latent channels: 8
/// - Mel channels: 64 (AudioLDM) or 128 (AudioLDM 2)
/// - Duration: Typically 10 seconds, but configurable
/// - Guidance scale: 2.5-5.0 typical
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create an AudioLDM model
/// var audioLDM = new AudioLDMModel&lt;float&gt;();
///
/// // Generate sound effects
/// var dogBark = audioLDM.GenerateFromText(
///     prompt: "A dog barking excitedly",
///     durationSeconds: 5.0,
///     numInferenceSteps: 100,
///     guidanceScale: 3.5);
///
/// // Generate music
/// var music = audioLDM.GenerateMusic(
///     prompt: "Soft jazz piano with light drums",
///     durationSeconds: 10.0,
///     numInferenceSteps: 200,
///     guidanceScale: 4.0);
///
/// // Save as audio file
/// SaveWav(dogBark, "dog_bark.wav", sampleRate: 16000);
/// </code>
/// </example>
public class AudioLDMModel<T> : AudioDiffusionModelBase<T>
{
    /// <summary>
    /// Standard AudioLDM sample rate.
    /// </summary>
    public const int AUDIOLDM_SAMPLE_RATE = 16000;

    /// <summary>
    /// Standard AudioLDM mel channels.
    /// </summary>
    public const int AUDIOLDM_MEL_CHANNELS = 64;

    /// <summary>
    /// Standard AudioLDM latent channels.
    /// </summary>
    public const int AUDIOLDM_LATENT_CHANNELS = 8;

    /// <summary>
    /// The U-Net noise predictor.
    /// </summary>
    private readonly UNetNoisePredictor<T> _unet;

    /// <summary>
    /// The AudioVAE for mel spectrogram encoding/decoding.
    /// </summary>
    private readonly AudioVAE<T> _audioVAE;

    /// <summary>
    /// The conditioning module (CLAP encoder).
    /// </summary>
    private readonly IConditioningModule<T>? _conditioner;

    /// <summary>
    /// Whether this is AudioLDM 2 (larger model).
    /// </summary>
    private readonly bool _isVersion2;

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _unet;

    /// <inheritdoc />
    public override IVAEModel<T> VAE => _audioVAE;

    /// <inheritdoc />
    public override IConditioningModule<T>? Conditioner => _conditioner;

    /// <inheritdoc />
    public override int LatentChannels => AUDIOLDM_LATENT_CHANNELS;

    /// <inheritdoc />
    public override bool SupportsTextToAudio => _conditioner != null;

    /// <inheritdoc />
    public override bool SupportsTextToMusic => _conditioner != null;

    /// <inheritdoc />
    public override bool SupportsTextToSpeech => false; // TTS requires different architecture

    /// <inheritdoc />
    public override bool SupportsAudioToAudio => true;

    /// <summary>
    /// Gets whether this is AudioLDM version 2.
    /// </summary>
    public bool IsVersion2 => _isVersion2;

    /// <summary>
    /// Gets the AudioVAE used for encoding/decoding.
    /// </summary>
    public AudioVAE<T> AudioVAE => _audioVAE;

    /// <summary>
    /// Initializes a new AudioLDM model with default parameters.
    /// </summary>
    public AudioLDMModel()
        : this(
            options: null,
            scheduler: null,
            unet: null,
            audioVAE: null,
            conditioner: null,
            sampleRate: AUDIOLDM_SAMPLE_RATE,
            defaultDurationSeconds: 10.0,
            melChannels: AUDIOLDM_MEL_CHANNELS,
            isVersion2: false)
    {
    }

    /// <summary>
    /// Initializes a new AudioLDM model with custom parameters.
    /// </summary>
    /// <param name="options">Configuration options for the diffusion model.</param>
    /// <param name="scheduler">Optional custom scheduler.</param>
    /// <param name="unet">Optional custom U-Net noise predictor.</param>
    /// <param name="audioVAE">Optional custom AudioVAE.</param>
    /// <param name="conditioner">Optional CLAP conditioning module.</param>
    /// <param name="sampleRate">Audio sample rate in Hz.</param>
    /// <param name="defaultDurationSeconds">Default audio duration.</param>
    /// <param name="melChannels">Number of mel spectrogram channels.</param>
    /// <param name="isVersion2">Whether to use AudioLDM 2 configuration.</param>
    /// <param name="seed">Optional random seed.</param>
    public AudioLDMModel(
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        UNetNoisePredictor<T>? unet = null,
        AudioVAE<T>? audioVAE = null,
        IConditioningModule<T>? conditioner = null,
        int sampleRate = AUDIOLDM_SAMPLE_RATE,
        double defaultDurationSeconds = 10.0,
        int melChannels = AUDIOLDM_MEL_CHANNELS,
        bool isVersion2 = false,
        int? seed = null)
        : base(
            options ?? CreateDefaultOptions(),
            scheduler ?? CreateDefaultScheduler(),
            sampleRate,
            defaultDurationSeconds,
            melChannels)
    {
        _isVersion2 = isVersion2;
        _conditioner = conditioner;

        // Initialize AudioVAE
        _audioVAE = audioVAE ?? CreateDefaultAudioVAE(melChannels, seed);

        // Initialize U-Net with audio-specific parameters
        _unet = unet ?? CreateDefaultUNet(isVersion2, seed);
    }

    /// <summary>
    /// Creates default options for AudioLDM.
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
    /// Creates the default DDIM scheduler.
    /// </summary>
    private static DDIMScheduler<T> CreateDefaultScheduler()
    {
        var config = SchedulerConfig<T>.CreateStableDiffusion();
        return new DDIMScheduler<T>(config);
    }

    /// <summary>
    /// Creates the default AudioVAE.
    /// </summary>
    private AudioVAE<T> CreateDefaultAudioVAE(int melChannels, int? seed)
    {
        return new AudioVAE<T>(
            melChannels: melChannels,
            latentChannels: AUDIOLDM_LATENT_CHANNELS,
            baseChannels: 64,
            channelMultipliers: new[] { 1, 2, 4, 4 },
            numResBlocks: 2,
            seed: seed);
    }

    /// <summary>
    /// Creates the default U-Net for AudioLDM.
    /// </summary>
    private UNetNoisePredictor<T> CreateDefaultUNet(bool isVersion2, int? seed)
    {
        // AudioLDM uses a U-Net similar to image diffusion but adapted for mel spectrograms
        var baseChannels = isVersion2 ? 384 : 256;
        var contextDim = isVersion2 ? 1024 : 768; // CLAP embedding dimension

        return new UNetNoisePredictor<T>(
            inputChannels: AUDIOLDM_LATENT_CHANNELS,
            outputChannels: AUDIOLDM_LATENT_CHANNELS,
            baseChannels: baseChannels,
            channelMultipliers: new[] { 1, 2, 4, 4 },
            numResBlocks: 2,
            attentionResolutions: new[] { 4, 2, 1 },
            contextDim: contextDim,
            seed: seed);
    }

    /// <summary>
    /// Generates audio from a text prompt.
    /// </summary>
    /// <param name="prompt">Text description of the desired audio.</param>
    /// <param name="negativePrompt">Optional negative prompt.</param>
    /// <param name="durationSeconds">Duration of audio to generate.</param>
    /// <param name="numInferenceSteps">Number of denoising steps.</param>
    /// <param name="guidanceScale">Classifier-free guidance scale.</param>
    /// <param name="seed">Optional random seed.</param>
    /// <returns>Audio waveform tensor [1, samples].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This generates audio matching your text description:
    ///
    /// Prompt tips:
    /// - Be descriptive: "A loud thunderstorm with heavy rain" vs "thunder"
    /// - Include context: "A dog barking in a quiet park"
    /// - Specify style for music: "Upbeat electronic dance music with synth bass"
    ///
    /// Guidance scale effects:
    /// - Lower (2.0-3.0): More variety, may not match prompt exactly
    /// - Medium (3.0-4.0): Good balance of quality and prompt following
    /// - Higher (4.0-6.0): Closely follows prompt, may reduce quality
    /// </para>
    /// </remarks>
    public virtual Tensor<T> GenerateAudio(
        string prompt,
        string? negativePrompt = null,
        double? durationSeconds = null,
        int numInferenceSteps = 100,
        double guidanceScale = 3.5,
        int? seed = null)
    {
        // Generate mel spectrogram in latent space
        var melLatent = GenerateFromText(
            prompt,
            negativePrompt,
            durationSeconds,
            numInferenceSteps,
            guidanceScale,
            seed);

        // Decode latent to mel spectrogram
        var melSpectrogram = _audioVAE.Decode(melLatent);

        // Convert mel spectrogram to audio waveform
        return _audioVAE.MelSpectrogramToAudio(melSpectrogram, SampleRate, HopLength);
    }

    /// <summary>
    /// Generates music from a text prompt.
    /// </summary>
    /// <param name="prompt">Text description of the desired music.</param>
    /// <param name="negativePrompt">Optional negative prompt.</param>
    /// <param name="durationSeconds">Duration of music to generate.</param>
    /// <param name="numInferenceSteps">Number of denoising steps.</param>
    /// <param name="guidanceScale">Classifier-free guidance scale.</param>
    /// <param name="seed">Optional random seed.</param>
    /// <returns>Audio waveform tensor [1, samples].</returns>
    /// <remarks>
    /// <para>
    /// Music generation uses the same underlying model but with prompts
    /// focused on musical content. For best results:
    ///
    /// - Specify genre: "jazz", "electronic", "classical"
    /// - Mention instruments: "piano", "guitar", "synthesizer"
    /// - Describe mood: "upbeat", "melancholic", "energetic"
    /// - Include tempo hints: "slow ballad", "fast dance beat"
    /// </para>
    /// </remarks>
    public override Tensor<T> GenerateMusic(
        string prompt,
        string? negativePrompt = null,
        double? durationSeconds = null,
        int numInferenceSteps = 200,
        double guidanceScale = 4.0,
        int? seed = null)
    {
        // Music typically benefits from more inference steps
        return GenerateAudio(
            prompt,
            negativePrompt ?? "low quality, noise, distortion",
            durationSeconds ?? 10.0,
            numInferenceSteps,
            guidanceScale,
            seed);
    }

    /// <summary>
    /// Transforms audio based on a text prompt (audio-to-audio).
    /// </summary>
    /// <param name="inputAudio">Input audio waveform [batch, samples].</param>
    /// <param name="prompt">Text description for transformation.</param>
    /// <param name="negativePrompt">Optional negative prompt.</param>
    /// <param name="strength">Transformation strength (0.0-1.0).</param>
    /// <param name="numInferenceSteps">Number of denoising steps.</param>
    /// <param name="guidanceScale">Classifier-free guidance scale.</param>
    /// <param name="seed">Optional random seed.</param>
    /// <returns>Transformed audio waveform tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This transforms existing audio based on your description:
    ///
    /// Examples:
    /// - Input: speech, Prompt: "whispered voice" -> quieter, intimate version
    /// - Input: guitar, Prompt: "electric guitar with distortion" -> adds effects
    /// - Input: ambient, Prompt: "add rain sounds" -> mixes in rain
    ///
    /// Strength controls how much to change:
    /// - Low (0.2-0.4): Subtle changes, preserves original character
    /// - Medium (0.4-0.6): Noticeable changes while keeping structure
    /// - High (0.6-0.8): Major changes, may alter original significantly
    /// </para>
    /// </remarks>
    public virtual Tensor<T> TransformAudio(
        Tensor<T> inputAudio,
        string prompt,
        string? negativePrompt = null,
        double strength = 0.5,
        int numInferenceSteps = 100,
        double guidanceScale = 3.5,
        int? seed = null)
    {
        if (_conditioner == null)
            throw new InvalidOperationException("Audio-to-audio transformation requires a conditioning module.");

        // Convert input audio to mel spectrogram
        var inputMel = _audioVAE.AudioToMelSpectrogram(inputAudio, SampleRate, HopLength);

        // Encode to latent
        var latent = _audioVAE.Encode(inputMel, sampleMode: false);
        latent = _audioVAE.ScaleLatent(latent);

        // Encode text prompts
        var promptTokens = _conditioner.Tokenize(prompt);
        var promptEmbedding = _conditioner.EncodeText(promptTokens);

        Tensor<T>? negativeEmbedding = null;
        if (guidanceScale > 1.0)
        {
            negativeEmbedding = !string.IsNullOrEmpty(negativePrompt)
                ? _conditioner.EncodeText(_conditioner.Tokenize(negativePrompt ?? string.Empty))
                : _conditioner.GetUnconditionalEmbedding(1);
        }

        // Calculate starting timestep
        Scheduler.SetTimesteps(numInferenceSteps);
        var startStep = (int)(numInferenceSteps * (1.0 - strength));
        var startTimestep = Scheduler.Timesteps.Skip(startStep).FirstOrDefault();

        // Add noise at starting timestep
        var rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomGenerator;
        var noise = SampleNoiseTensor(latent.Shape, rng);
        latent = AddNoiseAtTimestep(latent, noise, startTimestep);

        // Denoising loop
        foreach (var timestep in Scheduler.Timesteps.Skip(startStep))
        {
            Tensor<T> noisePrediction;

            if (negativeEmbedding != null)
            {
                var condPred = _unet.PredictNoise(latent, timestep, promptEmbedding);
                var uncondPred = _unet.PredictNoise(latent, timestep, negativeEmbedding);
                noisePrediction = ApplyGuidance(uncondPred, condPred, guidanceScale);
            }
            else
            {
                noisePrediction = _unet.PredictNoise(latent, timestep, promptEmbedding);
            }

            var latentVector = latent.ToVector();
            var noiseVector = noisePrediction.ToVector();
            latentVector = Scheduler.Step(noiseVector, timestep, latentVector, NumOps.Zero);
            latent = new Tensor<T>(latent.Shape, latentVector);
        }

        // Decode to audio
        var unscaledLatent = _audioVAE.UnscaleLatent(latent);
        var outputMel = _audioVAE.Decode(unscaledLatent);
        return _audioVAE.MelSpectrogramToAudio(outputMel, SampleRate, HopLength);
    }

    /// <summary>
    /// Adds noise at a specific timestep for audio-to-audio.
    /// </summary>
    private Tensor<T> AddNoiseAtTimestep(Tensor<T> latent, Tensor<T> noise, int timestep)
    {
        var alpha = 1.0 - (timestep / 1000.0);
        var sigma = Math.Sqrt(1.0 - alpha * alpha);

        var result = new Tensor<T>(latent.Shape);
        var resultSpan = result.AsWritableSpan();
        var latentSpan = latent.AsSpan();
        var noiseSpan = noise.AsSpan();

        for (int i = 0; i < resultSpan.Length; i++)
        {
            var latentVal = NumOps.ToDouble(latentSpan[i]);
            var noiseVal = NumOps.ToDouble(noiseSpan[i]);
            resultSpan[i] = NumOps.FromDouble(alpha * latentVal + sigma * noiseVal);
        }

        return result;
    }

    /// <summary>
    /// Generates audio variations from an input audio.
    /// </summary>
    /// <param name="inputAudio">Input audio waveform [batch, samples].</param>
    /// <param name="numVariations">Number of variations to generate.</param>
    /// <param name="variationStrength">How much to vary (0.0-1.0).</param>
    /// <param name="seed">Optional random seed.</param>
    /// <returns>List of audio variation tensors.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This creates multiple variations of your audio:
    ///
    /// Use cases:
    /// - Sound design: Generate similar but unique sounds
    /// - Music production: Create instrument variations
    /// - Audio augmentation: Expand training data
    ///
    /// Each variation will be similar to the input but with random differences.
    /// </para>
    /// </remarks>
    public virtual List<Tensor<T>> GenerateVariations(
        Tensor<T> inputAudio,
        int numVariations = 4,
        double variationStrength = 0.3,
        int? seed = null)
    {
        var variations = new List<Tensor<T>>();
        var baseSeed = seed ?? RandomGenerator.Next();

        // Convert to latent once
        var inputMel = _audioVAE.AudioToMelSpectrogram(inputAudio, SampleRate, HopLength);
        var baseLatent = _audioVAE.Encode(inputMel, sampleMode: false);
        baseLatent = _audioVAE.ScaleLatent(baseLatent);

        for (int i = 0; i < numVariations; i++)
        {
            var variationSeed = baseSeed + i;
            var rng = RandomHelper.CreateSeededRandom(variationSeed);

            // Add controlled noise
            var noise = SampleNoiseTensor(baseLatent.Shape, rng);
            var noisyLatent = new Tensor<T>(baseLatent.Shape);
            var noisySpan = noisyLatent.AsWritableSpan();
            var baseSpan = baseLatent.AsSpan();
            var noiseSpan = noise.AsSpan();

            for (int j = 0; j < noisySpan.Length; j++)
            {
                var baseVal = NumOps.ToDouble(baseSpan[j]);
                var noiseVal = NumOps.ToDouble(noiseSpan[j]);
                noisySpan[j] = NumOps.FromDouble(baseVal + variationStrength * noiseVal);
            }

            // Decode variation
            var unscaled = _audioVAE.UnscaleLatent(noisyLatent);
            var melVar = _audioVAE.Decode(unscaled);
            var audioVar = _audioVAE.MelSpectrogramToAudio(melVar, SampleRate, HopLength);
            variations.Add(audioVar);
        }

        return variations;
    }

    #region IParameterizable Implementation

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var unetParams = _unet.GetParameters();
        var vaeParams = _audioVAE.GetParameters();

        var totalLength = unetParams.Length + vaeParams.Length;
        var combined = new Vector<T>(totalLength);

        for (int i = 0; i < unetParams.Length; i++)
        {
            combined[i] = unetParams[i];
        }

        for (int i = 0; i < vaeParams.Length; i++)
        {
            combined[unetParams.Length + i] = vaeParams[i];
        }

        return combined;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        var unetCount = _unet.ParameterCount;
        var vaeCount = _audioVAE.ParameterCount;

        if (parameters.Length != unetCount + vaeCount)
            throw new ArgumentException($"Expected {unetCount + vaeCount} parameters, got {parameters.Length}.");

        var unetParams = new Vector<T>(unetCount);
        var vaeParams = new Vector<T>(vaeCount);

        for (int i = 0; i < unetCount; i++)
        {
            unetParams[i] = parameters[i];
        }

        for (int i = 0; i < vaeCount; i++)
        {
            vaeParams[i] = parameters[unetCount + i];
        }

        _unet.SetParameters(unetParams);
        _audioVAE.SetParameters(vaeParams);
    }

    /// <inheritdoc />
    public override int ParameterCount => _unet.ParameterCount + _audioVAE.ParameterCount;

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
        // Clone U-Net with trained weights
        var baseChannels = _isVersion2 ? 384 : 256;
        var contextDim = _isVersion2 ? 1024 : 768;
        var clonedUnet = new UNetNoisePredictor<T>(
            inputChannels: AUDIOLDM_LATENT_CHANNELS,
            outputChannels: AUDIOLDM_LATENT_CHANNELS,
            baseChannels: baseChannels,
            channelMultipliers: new[] { 1, 2, 4, 4 },
            numResBlocks: 2,
            attentionResolutions: new[] { 4, 2, 1 },
            contextDim: contextDim);
        clonedUnet.SetParameters(_unet.GetParameters());

        // Clone AudioVAE with trained weights
        var clonedVae = new AudioVAE<T>(
            melChannels: MelChannels,
            latentChannels: AUDIOLDM_LATENT_CHANNELS,
            baseChannels: 64,
            channelMultipliers: new[] { 1, 2, 4, 4 },
            numResBlocks: 2);
        clonedVae.SetParameters(_audioVAE.GetParameters());

        return new AudioLDMModel<T>(
            options: null,
            scheduler: null,
            unet: clonedUnet,
            audioVAE: clonedVae,
            conditioner: _conditioner,
            sampleRate: SampleRate,
            defaultDurationSeconds: DefaultDurationSeconds,
            melChannels: MelChannels,
            isVersion2: _isVersion2);
    }

    #endregion
}
