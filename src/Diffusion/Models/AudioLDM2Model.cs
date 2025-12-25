using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.NeuralNetworks.Diffusion.Schedulers;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Diffusion;
using AiDotNet.Models.Options;

namespace AiDotNet.Diffusion.Models;

/// <summary>
/// AudioLDM 2 - Enhanced Audio Latent Diffusion Model with dual text encoders.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// AudioLDM 2 is an improved version of AudioLDM with significant architectural enhancements
/// for better text-to-audio and text-to-music generation. Key improvements include:
///
/// 1. Dual Text Encoders: Combines CLAP (audio-text) and T5/GPT-2 (language) embeddings
/// 2. Larger Architecture: 384 base channels vs 256 in AudioLDM 1
/// 3. Higher Resolution: 128 mel channels vs 64 for better audio quality
/// 4. Improved Music Generation: Better temporal coherence and musical structure
/// 5. Longer Duration Support: Up to 30 seconds of audio generation
/// </para>
/// <para>
/// <b>For Beginners:</b> AudioLDM 2 generates higher-quality audio than AudioLDM 1:
///
/// Example prompts:
/// - "A symphony orchestra playing a dramatic crescendo" -> orchestral music
/// - "Footsteps on gravel with birds chirping" -> detailed soundscape
/// - "Electric guitar riff with heavy distortion" -> rock music
///
/// The dual encoder architecture means:
/// - CLAP encoder understands audio concepts (instrument sounds, effects)
/// - T5/GPT-2 encoder understands language (descriptions, context)
/// - Combined, they produce audio that matches both sound and meaning
/// </para>
/// <para>
/// Technical specifications:
/// - Sample rate: 16 kHz (speech/effects) or 48 kHz (high-quality music)
/// - Latent channels: 8
/// - Mel channels: 128 (double AudioLDM 1)
/// - Base channels: 384 (1.5x AudioLDM 1)
/// - Context dimension: 1024 (combined encoder output)
/// - Duration: Up to 30 seconds
/// - Guidance scale: 3.0-6.0 typical
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create an AudioLDM 2 model
/// var audioLDM2 = new AudioLDM2Model&lt;float&gt;();
///
/// // Generate high-quality music
/// var music = audioLDM2.GenerateMusic(
///     prompt: "Cinematic orchestral music with dramatic strings and timpani",
///     durationSeconds: 20.0,
///     numInferenceSteps: 200,
///     guidanceScale: 4.5);
///
/// // Generate complex sound effects
/// var soundscape = audioLDM2.GenerateAudio(
///     prompt: "A busy city street with traffic, horns, and people talking",
///     durationSeconds: 15.0,
///     numInferenceSteps: 150,
///     guidanceScale: 4.0);
/// </code>
/// </example>
public class AudioLDM2Model<T> : AudioDiffusionModelBase<T>
{
    /// <summary>
    /// AudioLDM 2 default sample rate for high-quality audio.
    /// </summary>
    public const int AUDIOLDM2_SAMPLE_RATE = 16000;

    /// <summary>
    /// AudioLDM 2 mel spectrogram channels (increased from 64 to 128).
    /// </summary>
    public const int AUDIOLDM2_MEL_CHANNELS = 128;

    /// <summary>
    /// AudioLDM 2 latent space channels.
    /// </summary>
    public const int AUDIOLDM2_LATENT_CHANNELS = 8;

    /// <summary>
    /// AudioLDM 2 U-Net base channels (larger than AudioLDM 1).
    /// </summary>
    public const int AUDIOLDM2_BASE_CHANNELS = 384;

    /// <summary>
    /// Combined context dimension from dual encoders.
    /// </summary>
    public const int AUDIOLDM2_CONTEXT_DIM = 1024;

    /// <summary>
    /// Maximum supported duration in seconds.
    /// </summary>
    public const double AUDIOLDM2_MAX_DURATION = 30.0;

    /// <summary>
    /// The U-Net noise predictor optimized for AudioLDM 2.
    /// </summary>
    private readonly UNetNoisePredictor<T> _unet;

    /// <summary>
    /// The AudioVAE for high-resolution mel spectrogram encoding/decoding.
    /// </summary>
    private readonly AudioVAE<T> _audioVAE;

    /// <summary>
    /// Primary conditioning module (CLAP encoder for audio-text alignment).
    /// </summary>
    private readonly IConditioningModule<T>? _clapConditioner;

    /// <summary>
    /// Secondary conditioning module (T5/GPT-2 for language understanding).
    /// </summary>
    private readonly IConditioningModule<T>? _languageConditioner;

    /// <summary>
    /// Projection layer to combine dual encoder outputs.
    /// </summary>
    private readonly ProjectionLayer<T> _projectionLayer;

    /// <summary>
    /// Model variant configuration.
    /// </summary>
    private readonly AudioLDM2Variant _variant;

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _unet;

    /// <inheritdoc />
    public override IVAEModel<T> VAE => _audioVAE;

    /// <inheritdoc />
    public override IConditioningModule<T>? Conditioner => _clapConditioner;

    /// <inheritdoc />
    public override int LatentChannels => AUDIOLDM2_LATENT_CHANNELS;

    /// <inheritdoc />
    public override bool SupportsTextToAudio => _clapConditioner != null;

    /// <inheritdoc />
    public override bool SupportsTextToMusic => _clapConditioner != null && _languageConditioner != null;

    /// <inheritdoc />
    public override bool SupportsTextToSpeech => false; // TTS requires specialized architecture

    /// <inheritdoc />
    public override bool SupportsAudioToAudio => true;

    /// <summary>
    /// Gets the secondary language conditioning module.
    /// </summary>
    public IConditioningModule<T>? LanguageConditioner => _languageConditioner;

    /// <summary>
    /// Gets the model variant.
    /// </summary>
    public AudioLDM2Variant Variant => _variant;

    /// <summary>
    /// Gets the AudioVAE for direct access.
    /// </summary>
    public AudioVAE<T> AudioVAE => _audioVAE;

    /// <summary>
    /// Initializes a new AudioLDM 2 model with default parameters.
    /// </summary>
    public AudioLDM2Model()
        : this(
            options: null,
            scheduler: null,
            unet: null,
            audioVAE: null,
            clapConditioner: null,
            languageConditioner: null,
            variant: AudioLDM2Variant.Large,
            sampleRate: AUDIOLDM2_SAMPLE_RATE,
            defaultDurationSeconds: 10.0,
            seed: null)
    {
    }

    /// <summary>
    /// Initializes a new AudioLDM 2 model with custom parameters.
    /// </summary>
    /// <param name="options">Configuration options for the diffusion model.</param>
    /// <param name="scheduler">Optional custom scheduler.</param>
    /// <param name="unet">Optional custom U-Net noise predictor.</param>
    /// <param name="audioVAE">Optional custom AudioVAE.</param>
    /// <param name="clapConditioner">Optional CLAP conditioning module.</param>
    /// <param name="languageConditioner">Optional T5/GPT-2 conditioning module.</param>
    /// <param name="variant">Model variant (Base, Large, or Music).</param>
    /// <param name="sampleRate">Audio sample rate in Hz.</param>
    /// <param name="defaultDurationSeconds">Default audio duration.</param>
    /// <param name="seed">Optional random seed.</param>
    public AudioLDM2Model(
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        UNetNoisePredictor<T>? unet = null,
        AudioVAE<T>? audioVAE = null,
        IConditioningModule<T>? clapConditioner = null,
        IConditioningModule<T>? languageConditioner = null,
        AudioLDM2Variant variant = AudioLDM2Variant.Large,
        int sampleRate = AUDIOLDM2_SAMPLE_RATE,
        double defaultDurationSeconds = 10.0,
        int? seed = null)
        : base(
            options ?? CreateDefaultOptions(),
            scheduler ?? CreateDefaultScheduler(),
            sampleRate,
            defaultDurationSeconds,
            AUDIOLDM2_MEL_CHANNELS)
    {
        _variant = variant;
        _clapConditioner = clapConditioner;
        _languageConditioner = languageConditioner;

        // Initialize AudioVAE with higher resolution
        _audioVAE = audioVAE ?? CreateDefaultAudioVAE(seed);

        // Initialize U-Net with variant-specific parameters
        _unet = unet ?? CreateDefaultUNet(variant, seed);

        // Initialize projection layer for combining encoder outputs
        _projectionLayer = CreateProjectionLayer(variant, seed);
    }

    /// <summary>
    /// Creates default options for AudioLDM 2.
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
    /// Creates the default AudioVAE with 128 mel channels.
    /// </summary>
    private AudioVAE<T> CreateDefaultAudioVAE(int? seed)
    {
        return new AudioVAE<T>(
            melChannels: AUDIOLDM2_MEL_CHANNELS,
            latentChannels: AUDIOLDM2_LATENT_CHANNELS,
            baseChannels: 128, // Higher resolution VAE
            channelMultipliers: new[] { 1, 2, 4, 4, 8 }, // More levels for better quality
            numResBlocks: 3, // More residual blocks
            seed: seed);
    }

    /// <summary>
    /// Creates the default U-Net for AudioLDM 2.
    /// </summary>
    private UNetNoisePredictor<T> CreateDefaultUNet(AudioLDM2Variant variant, int? seed)
    {
        var (baseChannels, numResBlocks) = variant switch
        {
            AudioLDM2Variant.Base => (256, 2),
            AudioLDM2Variant.Large => (384, 2),
            AudioLDM2Variant.Music => (384, 3), // More blocks for music
            _ => (384, 2)
        };

        return new UNetNoisePredictor<T>(
            inputChannels: AUDIOLDM2_LATENT_CHANNELS,
            outputChannels: AUDIOLDM2_LATENT_CHANNELS,
            baseChannels: baseChannels,
            channelMultipliers: new[] { 1, 2, 3, 4, 4 }, // AudioLDM 2 architecture
            numResBlocks: numResBlocks,
            attentionResolutions: new[] { 4, 2, 1 },
            contextDim: AUDIOLDM2_CONTEXT_DIM,
            seed: seed);
    }

    /// <summary>
    /// Creates the projection layer for combining dual encoder outputs.
    /// </summary>
    private ProjectionLayer<T> CreateProjectionLayer(AudioLDM2Variant variant, int? seed)
    {
        // CLAP: 512 dimensions, T5/GPT-2: 768-1024 dimensions
        // Combined and projected to context dimension
        var clapDim = 512;
        var languageDim = variant == AudioLDM2Variant.Music ? 1024 : 768;

        return new ProjectionLayer<T>(
            inputDim: clapDim + languageDim,
            outputDim: AUDIOLDM2_CONTEXT_DIM,
            seed: seed);
    }

    /// <summary>
    /// Combines embeddings from both encoders.
    /// </summary>
    /// <param name="clapEmbedding">CLAP encoder output.</param>
    /// <param name="languageEmbedding">T5/GPT-2 encoder output.</param>
    /// <returns>Combined and projected embedding.</returns>
    private Tensor<T> CombineEmbeddings(Tensor<T> clapEmbedding, Tensor<T>? languageEmbedding)
    {
        if (languageEmbedding == null)
        {
            // Only CLAP embedding, project to context dimension
            var batchSize = clapEmbedding.Shape[0];
            var seqLen = clapEmbedding.Shape.Length > 2 ? clapEmbedding.Shape[1] : 1;
            var clapDim = clapEmbedding.Shape.Length > 2 ? clapEmbedding.Shape[2] : clapEmbedding.Shape[1];

            // Create zero padding for language embedding
            var padSize = AUDIOLDM2_CONTEXT_DIM - clapDim;
            var paddedShape = clapEmbedding.Shape.Length > 2
                ? new[] { batchSize, seqLen, AUDIOLDM2_CONTEXT_DIM }
                : new[] { batchSize, AUDIOLDM2_CONTEXT_DIM };

            var padded = new Tensor<T>(paddedShape);
            var paddedSpan = padded.AsWritableSpan();
            var clapSpan = clapEmbedding.AsSpan();

            // Copy CLAP embedding
            for (int b = 0; b < batchSize; b++)
            {
                for (int s = 0; s < seqLen; s++)
                {
                    for (int d = 0; d < clapDim && d < padded.Length; d++)
                    {
                        var srcIdx = b * seqLen * clapDim + s * clapDim + d;
                        var dstIdx = b * seqLen * AUDIOLDM2_CONTEXT_DIM + s * AUDIOLDM2_CONTEXT_DIM + d;
                        if (srcIdx < clapSpan.Length && dstIdx < paddedSpan.Length)
                        {
                            paddedSpan[dstIdx] = clapSpan[srcIdx];
                        }
                    }
                }
            }

            return padded;
        }

        // Concatenate and project both embeddings
        var combined = ConcatenateEmbeddings(clapEmbedding, languageEmbedding);
        return _projectionLayer.Forward(combined);
    }

    /// <summary>
    /// Concatenates CLAP and language embeddings along the feature dimension.
    /// </summary>
    private Tensor<T> ConcatenateEmbeddings(Tensor<T> clap, Tensor<T> language)
    {
        var batchSize = clap.Shape[0];
        var seqLen = clap.Shape.Length > 2 ? Math.Max(clap.Shape[1], language.Shape[1]) : 1;
        var clapDim = clap.Shape.Length > 2 ? clap.Shape[2] : clap.Shape[1];
        var langDim = language.Shape.Length > 2 ? language.Shape[2] : language.Shape[1];
        var totalDim = clapDim + langDim;

        var resultShape = clap.Shape.Length > 2
            ? new[] { batchSize, seqLen, totalDim }
            : new[] { batchSize, totalDim };

        var result = new Tensor<T>(resultShape);
        var resultSpan = result.AsWritableSpan();
        var clapSpan = clap.AsSpan();
        var langSpan = language.AsSpan();

        for (int b = 0; b < batchSize; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                // Copy CLAP features
                for (int d = 0; d < clapDim; d++)
                {
                    var srcIdx = b * seqLen * clapDim + s * clapDim + d;
                    var dstIdx = b * seqLen * totalDim + s * totalDim + d;
                    if (srcIdx < clapSpan.Length && dstIdx < resultSpan.Length)
                    {
                        resultSpan[dstIdx] = clapSpan[srcIdx];
                    }
                }

                // Copy language features
                for (int d = 0; d < langDim; d++)
                {
                    var srcIdx = b * seqLen * langDim + s * langDim + d;
                    var dstIdx = b * seqLen * totalDim + s * totalDim + clapDim + d;
                    if (srcIdx < langSpan.Length && dstIdx < resultSpan.Length)
                    {
                        resultSpan[dstIdx] = langSpan[srcIdx];
                    }
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Generates audio from a text prompt using dual encoders.
    /// </summary>
    /// <param name="prompt">Text description of the desired audio.</param>
    /// <param name="negativePrompt">Optional negative prompt.</param>
    /// <param name="durationSeconds">Duration of audio to generate (max 30s).</param>
    /// <param name="numInferenceSteps">Number of denoising steps.</param>
    /// <param name="guidanceScale">Classifier-free guidance scale.</param>
    /// <param name="seed">Optional random seed.</param>
    /// <returns>Audio waveform tensor [1, samples].</returns>
    /// <remarks>
    /// <para>
    /// AudioLDM 2's dual encoder architecture provides better prompt understanding:
    ///
    /// - CLAP encoder: Understands audio-specific concepts (instruments, sounds, textures)
    /// - T5/GPT-2 encoder: Understands language semantics (descriptions, contexts, styles)
    ///
    /// This combination allows for more nuanced control over generated audio.
    /// </para>
    /// </remarks>
    public virtual Tensor<T> GenerateAudio(
        string prompt,
        string? negativePrompt = null,
        double? durationSeconds = null,
        int numInferenceSteps = 150,
        double guidanceScale = 4.0,
        int? seed = null)
    {
        var duration = Math.Min(durationSeconds ?? DefaultDurationSeconds, AUDIOLDM2_MAX_DURATION);

        // Generate mel spectrogram in latent space with combined embeddings
        var melLatent = GenerateWithDualEncoders(
            prompt,
            negativePrompt,
            duration,
            numInferenceSteps,
            guidanceScale,
            seed);

        // Decode latent to mel spectrogram
        var melSpectrogram = _audioVAE.Decode(melLatent);

        // Convert mel spectrogram to audio waveform
        return _audioVAE.MelSpectrogramToAudio(melSpectrogram, SampleRate, HopLength);
    }

    /// <summary>
    /// Generates latent using both encoders.
    /// </summary>
    private Tensor<T> GenerateWithDualEncoders(
        string prompt,
        string? negativePrompt,
        double durationSeconds,
        int numInferenceSteps,
        double guidanceScale,
        int? seed)
    {
        // Calculate latent dimensions for the duration
        var numSamples = (int)(durationSeconds * SampleRate);
        var numFrames = numSamples / HopLength + 1;
        var latentTimeFrames = numFrames / 4; // VAE downsampling

        var latentShape = new[] { 1, LatentChannels, latentTimeFrames, MelChannels / 8 };

        // Encode prompts with both encoders
        Tensor<T>? combinedEmbedding = null;
        Tensor<T>? negativeEmbedding = null;

        if (_clapConditioner != null)
        {
            var clapTokens = _clapConditioner.Tokenize(prompt);
            var clapEmbed = _clapConditioner.EncodeText(clapTokens);

            Tensor<T>? langEmbed = null;
            if (_languageConditioner != null)
            {
                var langTokens = _languageConditioner.Tokenize(prompt);
                langEmbed = _languageConditioner.EncodeText(langTokens);
            }

            combinedEmbedding = CombineEmbeddings(clapEmbed, langEmbed);

            // Negative prompt embedding
            if (guidanceScale > 1.0)
            {
                if (!string.IsNullOrEmpty(negativePrompt))
                {
                    var negClapTokens = _clapConditioner.Tokenize(negativePrompt);
                    var negClapEmbed = _clapConditioner.EncodeText(negClapTokens);

                    Tensor<T>? negLangEmbed = null;
                    if (_languageConditioner != null)
                    {
                        var negLangTokens = _languageConditioner.Tokenize(negativePrompt);
                        negLangEmbed = _languageConditioner.EncodeText(negLangTokens);
                    }

                    negativeEmbedding = CombineEmbeddings(negClapEmbed, negLangEmbed);
                }
                else
                {
                    negativeEmbedding = _clapConditioner.GetUnconditionalEmbedding(1);
                    if (_languageConditioner != null)
                    {
                        var uncondLang = _languageConditioner.GetUnconditionalEmbedding(1);
                        negativeEmbedding = CombineEmbeddings(negativeEmbedding, uncondLang);
                    }
                }
            }
        }

        // Initialize random latent
        var rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomGenerator;
        var latent = SampleNoiseTensor(latentShape, rng);

        // Set up scheduler
        Scheduler.SetTimesteps(numInferenceSteps);

        // Denoising loop
        foreach (var timestep in Scheduler.Timesteps)
        {
            Tensor<T> noisePrediction;

            if (combinedEmbedding != null && negativeEmbedding != null && guidanceScale > 1.0)
            {
                // Classifier-free guidance
                var condPred = _unet.PredictNoise(latent, timestep, combinedEmbedding);
                var uncondPred = _unet.PredictNoise(latent, timestep, negativeEmbedding);
                noisePrediction = ApplyGuidance(uncondPred, condPred, guidanceScale);
            }
            else if (combinedEmbedding != null)
            {
                noisePrediction = _unet.PredictNoise(latent, timestep, combinedEmbedding);
            }
            else
            {
                noisePrediction = _unet.PredictNoise(latent, timestep, null);
            }

            // Scheduler step
            var latentVector = latent.ToVector();
            var noiseVector = noisePrediction.ToVector();
            latentVector = Scheduler.Step(noiseVector, timestep, latentVector, NumOps.Zero);
            latent = new Tensor<T>(latent.Shape, latentVector);
        }

        // Unscale latent for VAE decoding
        return _audioVAE.UnscaleLatent(latent);
    }

    /// <summary>
    /// Generates music from a text prompt with enhanced musical understanding.
    /// </summary>
    /// <param name="prompt">Text description of the desired music.</param>
    /// <param name="negativePrompt">Optional negative prompt.</param>
    /// <param name="durationSeconds">Duration of music to generate (max 30s).</param>
    /// <param name="numInferenceSteps">Number of denoising steps.</param>
    /// <param name="guidanceScale">Classifier-free guidance scale.</param>
    /// <param name="seed">Optional random seed.</param>
    /// <returns>Audio waveform tensor [1, samples].</returns>
    /// <remarks>
    /// <para>
    /// AudioLDM 2 excels at music generation due to its dual encoder architecture.
    /// The T5/GPT-2 encoder provides better understanding of musical concepts like:
    ///
    /// - Genre descriptions ("jazz fusion", "baroque classical")
    /// - Mood and emotion ("melancholic", "uplifting")
    /// - Instrumentation ("string quartet", "electronic synths")
    /// - Tempo and rhythm ("slow waltz", "fast breakbeat")
    ///
    /// The CLAP encoder ensures the generated audio sounds authentic.
    /// </para>
    /// </remarks>
    public override Tensor<T> GenerateMusic(
        string prompt,
        string? negativePrompt = null,
        double? durationSeconds = null,
        int numInferenceSteps = 200,
        double guidanceScale = 4.5,
        int? seed = null)
    {
        // Music benefits from more inference steps and slightly higher guidance
        var defaultNegative = negativePrompt ?? "low quality, distorted, noise, static, mono, flat";

        return GenerateAudio(
            prompt,
            defaultNegative,
            durationSeconds ?? 20.0, // Longer default for music
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
    public virtual Tensor<T> TransformAudio(
        Tensor<T> inputAudio,
        string prompt,
        string? negativePrompt = null,
        double strength = 0.5,
        int numInferenceSteps = 150,
        double guidanceScale = 4.0,
        int? seed = null)
    {
        if (_clapConditioner == null)
            throw new InvalidOperationException("Audio transformation requires a conditioning module.");

        // Convert input audio to mel spectrogram
        var inputMel = _audioVAE.AudioToMelSpectrogram(inputAudio, SampleRate, HopLength);

        // Encode to latent
        var latent = _audioVAE.Encode(inputMel, sampleMode: false);
        latent = _audioVAE.ScaleLatent(latent);

        // Encode prompts with dual encoders
        var clapTokens = _clapConditioner.Tokenize(prompt);
        var clapEmbed = _clapConditioner.EncodeText(clapTokens);

        Tensor<T>? langEmbed = null;
        if (_languageConditioner != null)
        {
            var langTokens = _languageConditioner.Tokenize(prompt);
            langEmbed = _languageConditioner.EncodeText(langTokens);
        }

        var combinedEmbedding = CombineEmbeddings(clapEmbed, langEmbed);

        Tensor<T>? negativeEmbedding = null;
        if (guidanceScale > 1.0)
        {
            if (!string.IsNullOrEmpty(negativePrompt))
            {
                var negClapTokens = _clapConditioner.Tokenize(negativePrompt);
                var negClapEmbed = _clapConditioner.EncodeText(negClapTokens);

                Tensor<T>? negLangEmbed = null;
                if (_languageConditioner != null)
                {
                    var negLangTokens = _languageConditioner.Tokenize(negativePrompt);
                    negLangEmbed = _languageConditioner.EncodeText(negLangTokens);
                }

                negativeEmbedding = CombineEmbeddings(negClapEmbed, negLangEmbed);
            }
            else
            {
                negativeEmbedding = _clapConditioner.GetUnconditionalEmbedding(1);
            }
        }

        // Calculate starting timestep based on strength
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
                var condPred = _unet.PredictNoise(latent, timestep, combinedEmbedding);
                var uncondPred = _unet.PredictNoise(latent, timestep, negativeEmbedding);
                noisePrediction = ApplyGuidance(uncondPred, condPred, guidanceScale);
            }
            else
            {
                noisePrediction = _unet.PredictNoise(latent, timestep, combinedEmbedding);
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
    /// Adds noise at a specific timestep for audio transformation.
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
    /// Generates audio variations with enhanced diversity.
    /// </summary>
    /// <param name="inputAudio">Input audio waveform.</param>
    /// <param name="numVariations">Number of variations to generate.</param>
    /// <param name="variationStrength">How much to vary (0.0-1.0).</param>
    /// <param name="seed">Optional random seed.</param>
    /// <returns>List of audio variation tensors.</returns>
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

    /// <summary>
    /// Interpolates between two audio samples in latent space.
    /// </summary>
    /// <param name="audio1">First audio sample.</param>
    /// <param name="audio2">Second audio sample.</param>
    /// <param name="numSteps">Number of interpolation steps.</param>
    /// <returns>List of interpolated audio tensors.</returns>
    public virtual List<Tensor<T>> InterpolateAudio(
        Tensor<T> audio1,
        Tensor<T> audio2,
        int numSteps = 5)
    {
        var results = new List<Tensor<T>>();

        // Encode both to latent
        var mel1 = _audioVAE.AudioToMelSpectrogram(audio1, SampleRate, HopLength);
        var mel2 = _audioVAE.AudioToMelSpectrogram(audio2, SampleRate, HopLength);

        var latent1 = _audioVAE.Encode(mel1, sampleMode: false);
        var latent2 = _audioVAE.Encode(mel2, sampleMode: false);

        latent1 = _audioVAE.ScaleLatent(latent1);
        latent2 = _audioVAE.ScaleLatent(latent2);

        // Interpolate
        for (int step = 0; step <= numSteps; step++)
        {
            var t = (double)step / numSteps;
            var interpolated = new Tensor<T>(latent1.Shape);
            var interpSpan = interpolated.AsWritableSpan();
            var span1 = latent1.AsSpan();
            var span2 = latent2.AsSpan();

            for (int i = 0; i < interpSpan.Length; i++)
            {
                var v1 = NumOps.ToDouble(span1[i]);
                var v2 = NumOps.ToDouble(span2[i]);
                interpSpan[i] = NumOps.FromDouble(v1 * (1 - t) + v2 * t);
            }

            // Decode
            var unscaled = _audioVAE.UnscaleLatent(interpolated);
            var mel = _audioVAE.Decode(unscaled);
            var audio = _audioVAE.MelSpectrogramToAudio(mel, SampleRate, HopLength);
            results.Add(audio);
        }

        return results;
    }

    #region IParameterizable Implementation

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var unetParams = _unet.GetParameters();
        var vaeParams = _audioVAE.GetParameters();
        var projParams = _projectionLayer.GetParameters();

        var totalLength = unetParams.Length + vaeParams.Length + projParams.Length;
        var combined = new Vector<T>(totalLength);

        int offset = 0;

        // Copy U-Net parameters
        for (int i = 0; i < unetParams.Length; i++)
        {
            combined[offset + i] = unetParams[i];
        }
        offset += unetParams.Length;

        // Copy VAE parameters
        for (int i = 0; i < vaeParams.Length; i++)
        {
            combined[offset + i] = vaeParams[i];
        }
        offset += vaeParams.Length;

        // Copy projection parameters
        for (int i = 0; i < projParams.Length; i++)
        {
            combined[offset + i] = projParams[i];
        }

        return combined;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        var unetCount = _unet.ParameterCount;
        var vaeCount = _audioVAE.ParameterCount;
        var projCount = _projectionLayer.ParameterCount;

        if (parameters.Length != unetCount + vaeCount + projCount)
            throw new ArgumentException($"Expected {unetCount + vaeCount + projCount} parameters, got {parameters.Length}.");

        var unetParams = new Vector<T>(unetCount);
        var vaeParams = new Vector<T>(vaeCount);
        var projParams = new Vector<T>(projCount);

        int offset = 0;

        // Extract U-Net parameters
        for (int i = 0; i < unetCount; i++)
        {
            unetParams[i] = parameters[offset + i];
        }
        offset += unetCount;

        // Extract VAE parameters
        for (int i = 0; i < vaeCount; i++)
        {
            vaeParams[i] = parameters[offset + i];
        }
        offset += vaeCount;

        // Extract projection parameters
        for (int i = 0; i < projCount; i++)
        {
            projParams[i] = parameters[offset + i];
        }

        _unet.SetParameters(unetParams);
        _audioVAE.SetParameters(vaeParams);
        _projectionLayer.SetParameters(projParams);
    }

    /// <inheritdoc />
    public override int ParameterCount =>
        _unet.ParameterCount + _audioVAE.ParameterCount + _projectionLayer.ParameterCount;

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
        return new AudioLDM2Model<T>(
            options: null,
            scheduler: null,
            unet: null,
            audioVAE: null,
            clapConditioner: _clapConditioner,
            languageConditioner: _languageConditioner,
            variant: _variant,
            sampleRate: SampleRate,
            defaultDurationSeconds: DefaultDurationSeconds);
    }

    #endregion
}

/// <summary>
/// AudioLDM 2 model variant.
/// </summary>
public enum AudioLDM2Variant
{
    /// <summary>
    /// Base model (256 base channels, faster inference).
    /// </summary>
    Base,

    /// <summary>
    /// Large model (384 base channels, higher quality).
    /// </summary>
    Large,

    /// <summary>
    /// Music-optimized model (384 base channels, 3 res blocks).
    /// </summary>
    Music
}

/// <summary>
/// Projection layer for combining dual encoder outputs.
/// </summary>
/// <typeparam name="T">Numeric type.</typeparam>
internal class ProjectionLayer<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly Matrix<T> _weights;
    private readonly Vector<T> _bias;
    private readonly int _inputDim;
    private readonly int _outputDim;

    /// <summary>
    /// Creates a new projection layer.
    /// </summary>
    public ProjectionLayer(int inputDim, int outputDim, int? seed = null)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _inputDim = inputDim;
        _outputDim = outputDim;

        var rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : new Random();
        var scale = Math.Sqrt(2.0 / inputDim);

        _weights = new Matrix<T>(outputDim, inputDim);
        _bias = new Vector<T>(outputDim);

        // Xavier initialization
        for (int i = 0; i < outputDim; i++)
        {
            for (int j = 0; j < inputDim; j++)
            {
                var val = (rng.NextDouble() * 2 - 1) * scale;
                _weights[i, j] = _numOps.FromDouble(val);
            }
            _bias[i] = _numOps.Zero;
        }
    }

    /// <summary>
    /// Forward pass through the projection layer.
    /// </summary>
    public Tensor<T> Forward(Tensor<T> input)
    {
        var batchSize = input.Shape[0];
        var seqLen = input.Shape.Length > 2 ? input.Shape[1] : 1;
        var inputDim = input.Shape.Length > 2 ? input.Shape[2] : input.Shape[1];

        var outputShape = input.Shape.Length > 2
            ? new[] { batchSize, seqLen, _outputDim }
            : new[] { batchSize, _outputDim };

        var output = new Tensor<T>(outputShape);
        var outSpan = output.AsWritableSpan();
        var inSpan = input.AsSpan();

        for (int b = 0; b < batchSize; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                for (int o = 0; o < _outputDim; o++)
                {
                    var sum = _numOps.ToDouble(_bias[o]);

                    for (int i = 0; i < inputDim && i < _inputDim; i++)
                    {
                        var inIdx = b * seqLen * inputDim + s * inputDim + i;
                        if (inIdx < inSpan.Length)
                        {
                            var inVal = _numOps.ToDouble(inSpan[inIdx]);
                            var wVal = _numOps.ToDouble(_weights[o, i]);
                            sum += inVal * wVal;
                        }
                    }

                    var outIdx = b * seqLen * _outputDim + s * _outputDim + o;
                    if (outIdx < outSpan.Length)
                    {
                        outSpan[outIdx] = _numOps.FromDouble(sum);
                    }
                }
            }
        }

        return output;
    }

    /// <summary>
    /// Gets all parameters.
    /// </summary>
    public Vector<T> GetParameters()
    {
        var total = _inputDim * _outputDim + _outputDim;
        var result = new Vector<T>(total);

        int idx = 0;
        for (int i = 0; i < _outputDim; i++)
        {
            for (int j = 0; j < _inputDim; j++)
            {
                result[idx++] = _weights[i, j];
            }
        }

        for (int i = 0; i < _outputDim; i++)
        {
            result[idx++] = _bias[i];
        }

        return result;
    }

    /// <summary>
    /// Sets all parameters.
    /// </summary>
    public void SetParameters(Vector<T> parameters)
    {
        var expected = _inputDim * _outputDim + _outputDim;
        if (parameters.Length != expected)
            throw new ArgumentException($"Expected {expected} parameters, got {parameters.Length}.");

        int idx = 0;
        for (int i = 0; i < _outputDim; i++)
        {
            for (int j = 0; j < _inputDim; j++)
            {
                _weights[i, j] = parameters[idx++];
            }
        }

        for (int i = 0; i < _outputDim; i++)
        {
            _bias[i] = parameters[idx++];
        }
    }

    /// <summary>
    /// Gets the parameter count.
    /// </summary>
    public int ParameterCount => _inputDim * _outputDim + _outputDim;
}
