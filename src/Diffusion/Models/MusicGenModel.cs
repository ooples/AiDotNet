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
/// MusicGen - Diffusion-based music generation model with advanced musical controls.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// MusicGenModel is a specialized diffusion model for music generation that provides
/// fine-grained control over musical characteristics including:
///
/// 1. Text-to-Music: Generate music from natural language descriptions
/// 2. Melody Conditioning: Guide generation with a reference melody
/// 3. Rhythm/Beat Conditioning: Generate music following a specific rhythm pattern
/// 4. Tempo Control: Generate at specific BPM (beats per minute)
/// 5. Key/Scale Guidance: Influence the musical key of generated content
/// 6. Style Transfer: Transform existing music to different styles
/// </para>
/// <para>
/// <b>For Beginners:</b> This model generates music with precise control:
///
/// Example prompts:
/// - "Upbeat electronic dance music at 128 BPM" -> EDM track
/// - "Sad piano ballad in A minor" -> emotional piano piece
/// - "Funky bass groove with drums" -> funk rhythm section
/// - "Orchestral film score, epic and dramatic" -> cinematic music
///
/// Advanced controls:
/// - BPM: Set exact tempo (60-200 BPM typical)
/// - Key: Major/minor keys (C major, A minor, etc.)
/// - Instruments: Specify or exclude instruments
/// - Style: Jazz, rock, classical, electronic, etc.
/// </para>
/// <para>
/// Technical specifications:
/// - Sample rate: 32 kHz (high-quality music)
/// - Latent channels: 16 (more capacity for musical structure)
/// - Mel channels: 128
/// - Duration: Up to 60 seconds
/// - Guidance scale: 3.0-7.0 typical
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a MusicGen model
/// var musicGen = new MusicGenModel&lt;float&gt;();
///
/// // Generate electronic music at specific BPM
/// var edm = musicGen.GenerateMusicWithTempo(
///     prompt: "Energetic electronic dance music with synthesizers",
///     bpm: 128,
///     durationSeconds: 30.0,
///     numInferenceSteps: 200);
///
/// // Generate melody-conditioned music
/// var variation = musicGen.GenerateFromMelody(
///     melodyAudio: originalMelody,
///     prompt: "Jazz version with saxophone",
///     preservationStrength: 0.7);
/// </code>
/// </example>
public class MusicGenModel<T> : AudioDiffusionModelBase<T>
{
    /// <summary>
    /// MusicGen default sample rate for high-quality music.
    /// </summary>
    public const int MUSICGEN_SAMPLE_RATE = 32000;

    /// <summary>
    /// MusicGen mel spectrogram channels.
    /// </summary>
    public const int MUSICGEN_MEL_CHANNELS = 128;

    /// <summary>
    /// MusicGen latent space channels (larger for musical structure).
    /// </summary>
    public const int MUSICGEN_LATENT_CHANNELS = 16;

    /// <summary>
    /// MusicGen U-Net base channels.
    /// </summary>
    public const int MUSICGEN_BASE_CHANNELS = 512;

    /// <summary>
    /// Context dimension for conditioning.
    /// </summary>
    public const int MUSICGEN_CONTEXT_DIM = 1536;

    /// <summary>
    /// Maximum supported duration in seconds.
    /// </summary>
    public const double MUSICGEN_MAX_DURATION = 60.0;

    /// <summary>
    /// Default BPM for music generation.
    /// </summary>
    public const int DEFAULT_BPM = 120;

    /// <summary>
    /// The U-Net noise predictor optimized for music.
    /// </summary>
    private readonly UNetNoisePredictor<T> _unet;

    /// <summary>
    /// The AudioVAE for high-resolution music encoding/decoding.
    /// </summary>
    private readonly AudioVAE<T> _musicVAE;

    /// <summary>
    /// Primary text conditioning module.
    /// </summary>
    private readonly IConditioningModule<T>? _textConditioner;

    /// <summary>
    /// Melody encoder for melody conditioning.
    /// </summary>
    private readonly MelodyEncoder<T> _melodyEncoder;

    /// <summary>
    /// Rhythm encoder for beat conditioning.
    /// </summary>
    private readonly RhythmEncoder<T> _rhythmEncoder;

    /// <summary>
    /// Model size variant.
    /// </summary>
    private readonly MusicGenSize _modelSize;

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _unet;

    /// <inheritdoc />
    public override IVAEModel<T> VAE => _musicVAE;

    /// <inheritdoc />
    public override IConditioningModule<T>? Conditioner => _textConditioner;

    /// <inheritdoc />
    public override int LatentChannels => MUSICGEN_LATENT_CHANNELS;

    /// <inheritdoc />
    public override bool SupportsTextToAudio => _textConditioner != null;

    /// <inheritdoc />
    public override bool SupportsTextToMusic => _textConditioner != null;

    /// <inheritdoc />
    public override bool SupportsTextToSpeech => false; // Music model, not TTS

    /// <inheritdoc />
    public override bool SupportsAudioToAudio => true;

    /// <summary>
    /// Gets the melody encoder for melody conditioning.
    /// </summary>
    public MelodyEncoder<T> MelodyEncoder => _melodyEncoder;

    /// <summary>
    /// Gets the rhythm encoder for beat conditioning.
    /// </summary>
    public RhythmEncoder<T> RhythmEncoder => _rhythmEncoder;

    /// <summary>
    /// Gets the model size variant.
    /// </summary>
    public MusicGenSize ModelSize => _modelSize;

    /// <summary>
    /// Gets the music VAE for direct access.
    /// </summary>
    public AudioVAE<T> MusicVAE => _musicVAE;

    /// <summary>
    /// Initializes a new MusicGen model with default parameters.
    /// </summary>
    public MusicGenModel()
        : this(
            options: null,
            scheduler: null,
            unet: null,
            musicVAE: null,
            textConditioner: null,
            modelSize: MusicGenSize.Medium,
            sampleRate: MUSICGEN_SAMPLE_RATE,
            defaultDurationSeconds: 30.0,
            seed: null)
    {
    }

    /// <summary>
    /// Initializes a new MusicGen model with custom parameters.
    /// </summary>
    /// <param name="options">Configuration options for the diffusion model.</param>
    /// <param name="scheduler">Optional custom scheduler.</param>
    /// <param name="unet">Optional custom U-Net noise predictor.</param>
    /// <param name="musicVAE">Optional custom music VAE.</param>
    /// <param name="textConditioner">Optional text conditioning module.</param>
    /// <param name="modelSize">Model size variant.</param>
    /// <param name="sampleRate">Audio sample rate in Hz.</param>
    /// <param name="defaultDurationSeconds">Default music duration.</param>
    /// <param name="seed">Optional random seed.</param>
    public MusicGenModel(
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        UNetNoisePredictor<T>? unet = null,
        AudioVAE<T>? musicVAE = null,
        IConditioningModule<T>? textConditioner = null,
        MusicGenSize modelSize = MusicGenSize.Medium,
        int sampleRate = MUSICGEN_SAMPLE_RATE,
        double defaultDurationSeconds = 30.0,
        int? seed = null)
        : base(
            options ?? CreateDefaultOptions(),
            scheduler ?? CreateDefaultScheduler(),
            sampleRate,
            defaultDurationSeconds,
            MUSICGEN_MEL_CHANNELS)
    {
        _modelSize = modelSize;
        _textConditioner = textConditioner;

        // Initialize music VAE with high capacity
        _musicVAE = musicVAE ?? CreateDefaultMusicVAE(seed);

        // Initialize U-Net with music-optimized architecture
        _unet = unet ?? CreateDefaultUNet(modelSize, seed);

        // Initialize melody encoder
        _melodyEncoder = new MelodyEncoder<T>(
            inputChannels: 1,
            outputDim: MUSICGEN_CONTEXT_DIM / 2,
            seed: seed);

        // Initialize rhythm encoder
        _rhythmEncoder = new RhythmEncoder<T>(
            outputDim: MUSICGEN_CONTEXT_DIM / 4,
            seed: seed);
    }

    /// <summary>
    /// Creates default options for MusicGen.
    /// </summary>
    private static DiffusionModelOptions<T> CreateDefaultOptions()
    {
        return new DiffusionModelOptions<T>
        {
            TrainTimesteps = 1000,
            BetaStart = 0.0001,
            BetaEnd = 0.02,
            BetaSchedule = BetaSchedule.ScaledLinear
        };
    }

    /// <summary>
    /// Creates the default scheduler optimized for music.
    /// </summary>
    private static DDIMScheduler<T> CreateDefaultScheduler()
    {
        var config = SchedulerConfig<T>.CreateStableDiffusion();
        return new DDIMScheduler<T>(config);
    }

    /// <summary>
    /// Creates the default music VAE.
    /// </summary>
    private AudioVAE<T> CreateDefaultMusicVAE(int? seed)
    {
        return new AudioVAE<T>(
            melChannels: MUSICGEN_MEL_CHANNELS,
            latentChannels: MUSICGEN_LATENT_CHANNELS,
            baseChannels: 128,
            channelMultipliers: new[] { 1, 2, 4, 8, 8 }, // Deep VAE for music
            numResBlocks: 3,
            seed: seed);
    }

    /// <summary>
    /// Creates the default U-Net for music generation.
    /// </summary>
    private UNetNoisePredictor<T> CreateDefaultUNet(MusicGenSize size, int? seed)
    {
        var (baseChannels, numResBlocks) = size switch
        {
            MusicGenSize.Small => (256, 2),
            MusicGenSize.Medium => (384, 2),
            MusicGenSize.Large => (512, 3),
            MusicGenSize.Melody => (512, 3), // Same as Large but different conditioning
            _ => (384, 2)
        };

        return new UNetNoisePredictor<T>(
            inputChannels: MUSICGEN_LATENT_CHANNELS,
            outputChannels: MUSICGEN_LATENT_CHANNELS,
            baseChannels: baseChannels,
            channelMultipliers: new[] { 1, 2, 4, 4, 8 }, // Music-optimized architecture
            numResBlocks: numResBlocks,
            attentionResolutions: new[] { 8, 4, 2, 1 }, // More attention levels for music
            contextDim: MUSICGEN_CONTEXT_DIM,
            seed: seed);
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
    public override Tensor<T> GenerateMusic(
        string prompt,
        string? negativePrompt = null,
        double? durationSeconds = null,
        int numInferenceSteps = 200,
        double guidanceScale = 5.0,
        int? seed = null)
    {
        var duration = Math.Min(durationSeconds ?? DefaultDurationSeconds, MUSICGEN_MAX_DURATION);

        // Generate in latent space
        var latent = GenerateMusicLatent(
            prompt,
            negativePrompt ?? "noise, distortion, low quality, mono, flat",
            duration,
            numInferenceSteps,
            guidanceScale,
            melodyCondition: null,
            rhythmCondition: null,
            bpm: null,
            seed: seed);

        // Decode to audio
        var melSpectrogram = _musicVAE.Decode(latent);
        return _musicVAE.MelSpectrogramToAudio(melSpectrogram, SampleRate, HopLength);
    }

    /// <summary>
    /// Generates music with specific tempo (BPM) control.
    /// </summary>
    /// <param name="prompt">Text description of the desired music.</param>
    /// <param name="bpm">Target beats per minute (60-200 typical).</param>
    /// <param name="negativePrompt">Optional negative prompt.</param>
    /// <param name="durationSeconds">Duration of music to generate.</param>
    /// <param name="numInferenceSteps">Number of denoising steps.</param>
    /// <param name="guidanceScale">Classifier-free guidance scale.</param>
    /// <param name="seed">Optional random seed.</param>
    /// <returns>Audio waveform tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> BPM (Beats Per Minute) controls the tempo:
    ///
    /// Common BPM ranges:
    /// - 60-80: Slow ballads, ambient
    /// - 80-100: Hip-hop, R&amp;B
    /// - 100-120: Pop, house
    /// - 120-140: Techno, trance
    /// - 140-180: Drum and bass, dubstep
    /// </para>
    /// </remarks>
    public virtual Tensor<T> GenerateMusicWithTempo(
        string prompt,
        int bpm,
        string? negativePrompt = null,
        double? durationSeconds = null,
        int numInferenceSteps = 200,
        double guidanceScale = 5.0,
        int? seed = null)
    {
        // Clamp BPM to reasonable range
        bpm = MathPolyfill.Clamp(bpm, 40, 240);

        var duration = Math.Min(durationSeconds ?? DefaultDurationSeconds, MUSICGEN_MAX_DURATION);

        // Augment prompt with BPM information
        var tempoPrompt = $"{prompt}, {bpm} BPM";

        // Generate with tempo conditioning
        var latent = GenerateMusicLatent(
            tempoPrompt,
            negativePrompt,
            duration,
            numInferenceSteps,
            guidanceScale,
            melodyCondition: null,
            rhythmCondition: null,
            bpm: bpm,
            seed: seed);

        var melSpectrogram = _musicVAE.Decode(latent);
        return _musicVAE.MelSpectrogramToAudio(melSpectrogram, SampleRate, HopLength);
    }

    /// <summary>
    /// Generates music conditioned on a reference melody.
    /// </summary>
    /// <param name="melodyAudio">Reference melody audio.</param>
    /// <param name="prompt">Text description for the style/arrangement.</param>
    /// <param name="preservationStrength">How closely to follow the melody (0.0-1.0).</param>
    /// <param name="negativePrompt">Optional negative prompt.</param>
    /// <param name="numInferenceSteps">Number of denoising steps.</param>
    /// <param name="guidanceScale">Classifier-free guidance scale.</param>
    /// <param name="seed">Optional random seed.</param>
    /// <returns>Audio waveform tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Melody conditioning lets you:
    ///
    /// - Create covers: Keep melody, change style
    /// - Add accompaniment: Keep melody, generate instruments
    /// - Style transfer: Transform melody to different genre
    ///
    /// Preservation strength:
    /// - 0.3-0.5: Use melody as loose guide
    /// - 0.5-0.7: Balance melody with new elements
    /// - 0.7-0.9: Closely follow original melody
    /// </para>
    /// </remarks>
    public virtual Tensor<T> GenerateFromMelody(
        Tensor<T> melodyAudio,
        string prompt,
        double preservationStrength = 0.6,
        string? negativePrompt = null,
        int numInferenceSteps = 200,
        double guidanceScale = 5.0,
        int? seed = null)
    {
        if (_textConditioner == null)
            throw new InvalidOperationException("Melody conditioning requires a text conditioning module.");

        // Encode melody to conditioning space
        var melodyCondition = _melodyEncoder.EncodeMelody(melodyAudio, SampleRate);

        // Calculate duration from input melody
        var durationSeconds = melodyAudio.Shape[^1] / (double)SampleRate;
        durationSeconds = Math.Min(durationSeconds, MUSICGEN_MAX_DURATION);

        // Generate with melody conditioning
        var latent = GenerateMusicLatent(
            prompt,
            negativePrompt,
            durationSeconds,
            numInferenceSteps,
            guidanceScale,
            melodyCondition: melodyCondition,
            rhythmCondition: null,
            bpm: null,
            seed: seed,
            conditioningStrength: preservationStrength);

        var melSpectrogram = _musicVAE.Decode(latent);
        return _musicVAE.MelSpectrogramToAudio(melSpectrogram, SampleRate, HopLength);
    }

    /// <summary>
    /// Generates music conditioned on a rhythm/beat pattern.
    /// </summary>
    /// <param name="rhythmAudio">Reference rhythm/percussion audio.</param>
    /// <param name="prompt">Text description for the melody/harmony.</param>
    /// <param name="rhythmStrength">How closely to follow the rhythm (0.0-1.0).</param>
    /// <param name="negativePrompt">Optional negative prompt.</param>
    /// <param name="numInferenceSteps">Number of denoising steps.</param>
    /// <param name="guidanceScale">Classifier-free guidance scale.</param>
    /// <param name="seed">Optional random seed.</param>
    /// <returns>Audio waveform tensor.</returns>
    public virtual Tensor<T> GenerateFromRhythm(
        Tensor<T> rhythmAudio,
        string prompt,
        double rhythmStrength = 0.5,
        string? negativePrompt = null,
        int numInferenceSteps = 200,
        double guidanceScale = 5.0,
        int? seed = null)
    {
        if (_textConditioner == null)
            throw new InvalidOperationException("Rhythm conditioning requires a text conditioning module.");

        // Encode rhythm to conditioning space
        var rhythmCondition = _rhythmEncoder.EncodeRhythm(rhythmAudio, SampleRate);

        // Calculate duration from input rhythm
        var durationSeconds = rhythmAudio.Shape[^1] / (double)SampleRate;
        durationSeconds = Math.Min(durationSeconds, MUSICGEN_MAX_DURATION);

        // Generate with rhythm conditioning
        var latent = GenerateMusicLatent(
            prompt,
            negativePrompt,
            durationSeconds,
            numInferenceSteps,
            guidanceScale,
            melodyCondition: null,
            rhythmCondition: rhythmCondition,
            bpm: null,
            seed: seed,
            conditioningStrength: rhythmStrength);

        var melSpectrogram = _musicVAE.Decode(latent);
        return _musicVAE.MelSpectrogramToAudio(melSpectrogram, SampleRate, HopLength);
    }

    /// <summary>
    /// Generates music continuation from an audio prompt.
    /// </summary>
    /// <param name="audioPrompt">Audio to continue from.</param>
    /// <param name="textPrompt">Optional text guidance for continuation.</param>
    /// <param name="continuationDurationSeconds">Duration of continuation.</param>
    /// <param name="overlapSeconds">Overlap with original for smooth transition.</param>
    /// <param name="numInferenceSteps">Number of denoising steps.</param>
    /// <param name="guidanceScale">Classifier-free guidance scale.</param>
    /// <param name="seed">Optional random seed.</param>
    /// <returns>Continued audio waveform.</returns>
    public virtual Tensor<T> ContinueMusic(
        Tensor<T> audioPrompt,
        string? textPrompt = null,
        double continuationDurationSeconds = 15.0,
        double overlapSeconds = 2.0,
        int numInferenceSteps = 150,
        double guidanceScale = 4.0,
        int? seed = null)
    {
        continuationDurationSeconds = Math.Min(continuationDurationSeconds, MUSICGEN_MAX_DURATION);
        var overlapSamples = (int)(overlapSeconds * SampleRate);

        // Encode prompt audio to latent
        var promptMel = _musicVAE.AudioToMelSpectrogram(audioPrompt, SampleRate, HopLength);
        var promptLatent = _musicVAE.Encode(promptMel, sampleMode: false);
        promptLatent = _musicVAE.ScaleLatent(promptLatent);

        // Extract melody and rhythm features for conditioning
        var melodyCondition = _melodyEncoder.EncodeMelody(audioPrompt, SampleRate);
        var rhythmCondition = _rhythmEncoder.EncodeRhythm(audioPrompt, SampleRate);

        // Combine conditioning
        var combinedCondition = CombineMusicConditions(melodyCondition, rhythmCondition, null);

        // Generate continuation
        var continuationLatent = GenerateContinuationLatent(
            promptLatent,
            combinedCondition,
            textPrompt,
            continuationDurationSeconds,
            numInferenceSteps,
            guidanceScale,
            seed);

        // Decode and blend
        var continuationMel = _musicVAE.Decode(continuationLatent);
        var continuationAudio = _musicVAE.MelSpectrogramToAudio(continuationMel, SampleRate, HopLength);

        // Blend overlapping region
        return BlendAudio(audioPrompt, continuationAudio, overlapSamples);
    }

    /// <summary>
    /// Core generation method with all conditioning options.
    /// </summary>
    private Tensor<T> GenerateMusicLatent(
        string prompt,
        string? negativePrompt,
        double durationSeconds,
        int numInferenceSteps,
        double guidanceScale,
        Tensor<T>? melodyCondition,
        Tensor<T>? rhythmCondition,
        int? bpm,
        int? seed,
        double conditioningStrength = 1.0)
    {
        // Calculate latent dimensions
        var numSamples = (int)(durationSeconds * SampleRate);
        var numFrames = numSamples / HopLength + 1;
        var latentTimeFrames = numFrames / 4;

        var latentShape = new[] { 1, LatentChannels, latentTimeFrames, MelChannels / 8 };

        // Build conditioning
        Tensor<T>? textEmbedding = null;
        Tensor<T>? negativeEmbedding = null;

        if (_textConditioner != null)
        {
            var tokens = _textConditioner.Tokenize(prompt);
            textEmbedding = _textConditioner.EncodeText(tokens);

            if (guidanceScale > 1.0)
            {
                if (!string.IsNullOrEmpty(negativePrompt))
                {
                    var negTokens = _textConditioner.Tokenize(negativePrompt ?? string.Empty);
                    negativeEmbedding = _textConditioner.EncodeText(negTokens);
                }
                else
                {
                    negativeEmbedding = _textConditioner.GetUnconditionalEmbedding(1);
                }
            }
        }

        // Combine all conditioning types
        var combinedCondition = CombineMusicConditions(
            melodyCondition,
            rhythmCondition,
            CreateTempoEmbedding(bpm, latentShape));

        if (textEmbedding != null && combinedCondition != null)
        {
            textEmbedding = BlendConditions(textEmbedding, combinedCondition, conditioningStrength);
        }
        else if (combinedCondition != null)
        {
            textEmbedding = combinedCondition;
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

            if (textEmbedding != null && negativeEmbedding != null && guidanceScale > 1.0)
            {
                var condPred = _unet.PredictNoise(latent, timestep, textEmbedding);
                var uncondPred = _unet.PredictNoise(latent, timestep, negativeEmbedding);
                noisePrediction = ApplyGuidance(uncondPred, condPred, guidanceScale);
            }
            else if (textEmbedding != null)
            {
                noisePrediction = _unet.PredictNoise(latent, timestep, textEmbedding);
            }
            else
            {
                noisePrediction = _unet.PredictNoise(latent, timestep, null);
            }

            var latentVector = latent.ToVector();
            var noiseVector = noisePrediction.ToVector();
            latentVector = Scheduler.Step(noiseVector, timestep, latentVector, NumOps.Zero);
            latent = new Tensor<T>(latent.Shape, latentVector);
        }

        return _musicVAE.UnscaleLatent(latent);
    }

    /// <summary>
    /// Generates continuation latent conditioned on prompt.
    /// </summary>
    private Tensor<T> GenerateContinuationLatent(
        Tensor<T> promptLatent,
        Tensor<T>? condition,
        string? textPrompt,
        double durationSeconds,
        int numInferenceSteps,
        double guidanceScale,
        int? seed)
    {
        var numSamples = (int)(durationSeconds * SampleRate);
        var numFrames = numSamples / HopLength + 1;
        var latentTimeFrames = numFrames / 4;

        var latentShape = new[] { 1, LatentChannels, latentTimeFrames, MelChannels / 8 };

        // Encode text if provided
        Tensor<T>? textEmbedding = null;
        if (_textConditioner != null && !string.IsNullOrEmpty(textPrompt))
        {
            var tokens = _textConditioner.Tokenize(textPrompt ?? string.Empty);
            textEmbedding = _textConditioner.EncodeText(tokens);
        }

        var combinedCondition = condition;
        if (textEmbedding != null && combinedCondition != null)
        {
            combinedCondition = BlendConditions(textEmbedding, combinedCondition, 0.5);
        }
        else if (textEmbedding != null)
        {
            combinedCondition = textEmbedding;
        }

        // Initialize with partial noise (more structure from prompt)
        var rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomGenerator;
        var noise = SampleNoiseTensor(latentShape, rng);

        // Blend prompt latent into initial state
        var latent = new Tensor<T>(latentShape);
        var latentSpan = latent.AsWritableSpan();
        var promptSpan = promptLatent.AsSpan();
        var noiseSpan = noise.AsSpan();

        // Use end of prompt latent to seed continuation
        var promptTimeLen = promptLatent.Shape[2];
        var transitionLen = Math.Min(promptTimeLen / 4, latentTimeFrames / 4);

        for (int i = 0; i < latentSpan.Length; i++)
        {
            // Calculate position
            var timePos = (i / (LatentChannels * (MelChannels / 8))) % latentTimeFrames;
            var blendFactor = Math.Min(1.0, timePos / (double)transitionLen);

            var noiseVal = NumOps.ToDouble(noiseSpan[i]);

            // Find corresponding prompt position
            var promptPos = Math.Max(0, promptTimeLen - transitionLen + Math.Min(timePos, transitionLen));
            var promptIdx = (i % (LatentChannels * (MelChannels / 8))) +
                           promptPos * LatentChannels * (MelChannels / 8);

            if (promptIdx < promptSpan.Length)
            {
                var promptVal = NumOps.ToDouble(promptSpan[promptIdx]);
                latentSpan[i] = NumOps.FromDouble(promptVal * (1 - blendFactor) + noiseVal * blendFactor);
            }
            else
            {
                latentSpan[i] = NumOps.FromDouble(noiseVal);
            }
        }

        // Denoising with fewer steps (already partially structured)
        var effectiveSteps = (int)(numInferenceSteps * 0.7);
        Scheduler.SetTimesteps(effectiveSteps);

        foreach (var timestep in Scheduler.Timesteps)
        {
            var noisePrediction = combinedCondition != null
                ? _unet.PredictNoise(latent, timestep, combinedCondition)
                : _unet.PredictNoise(latent, timestep, null);

            var latentVector = latent.ToVector();
            var noiseVector = noisePrediction.ToVector();
            latentVector = Scheduler.Step(noiseVector, timestep, latentVector, NumOps.Zero);
            latent = new Tensor<T>(latent.Shape, latentVector);
        }

        return _musicVAE.UnscaleLatent(latent);
    }

    /// <summary>
    /// Creates tempo embedding for BPM conditioning.
    /// </summary>
    private Tensor<T>? CreateTempoEmbedding(int? bpm, int[] latentShape)
    {
        if (!bpm.HasValue) return null;

        // Sinusoidal embedding of tempo
        var embedding = new Tensor<T>(new[] { 1, 1, MUSICGEN_CONTEXT_DIM / 4 });
        var span = embedding.AsWritableSpan();

        var normalizedBpm = (bpm.Value - 60) / 180.0; // Normalize to [0, 1]

        for (int i = 0; i < span.Length; i++)
        {
            var freq = Math.Pow(10000, -2.0 * (i / 2) / span.Length);
            var value = i % 2 == 0
                ? Math.Sin(normalizedBpm * freq)
                : Math.Cos(normalizedBpm * freq);
            span[i] = NumOps.FromDouble(value);
        }

        return embedding;
    }

    /// <summary>
    /// Combines multiple music conditioning sources.
    /// </summary>
    private Tensor<T>? CombineMusicConditions(
        Tensor<T>? melody,
        Tensor<T>? rhythm,
        Tensor<T>? tempo)
    {
        var conditions = new List<Tensor<T>>();
        if (melody != null) conditions.Add(melody);
        if (rhythm != null) conditions.Add(rhythm);
        if (tempo != null) conditions.Add(tempo);

        if (conditions.Count == 0) return null;
        if (conditions.Count == 1) return conditions[0];

        // Concatenate along feature dimension
        var totalFeatures = conditions.Sum(c => c.Shape[^1]);
        var batchSize = conditions[0].Shape[0];

        // Use maximum sequence length across all conditions
        var outputSeqLen = conditions.Max(c => c.Shape.Length > 2 ? c.Shape[1] : 1);

        var resultShape = conditions[0].Shape.Length > 2
            ? new[] { batchSize, outputSeqLen, totalFeatures }
            : new[] { batchSize, totalFeatures };

        var result = new Tensor<T>(resultShape);
        var resultSpan = result.AsWritableSpan();

        int featureOffset = 0;
        foreach (var cond in conditions)
        {
            var condSpan = cond.AsSpan();
            var condFeatures = cond.Shape[^1];
            var condSeqLen = cond.Shape.Length > 2 ? cond.Shape[1] : 1;

            for (int b = 0; b < batchSize; b++)
            {
                for (int s = 0; s < outputSeqLen; s++)
                {
                    // Only copy if within this condition's actual sequence length
                    if (s < condSeqLen)
                    {
                        for (int f = 0; f < condFeatures; f++)
                        {
                            var srcIdx = b * condSeqLen * condFeatures + s * condFeatures + f;
                            var dstIdx = b * outputSeqLen * totalFeatures + s * totalFeatures + featureOffset + f;
                            if (srcIdx < condSpan.Length && dstIdx < resultSpan.Length)
                            {
                                resultSpan[dstIdx] = condSpan[srcIdx];
                            }
                        }
                    }
                    // Positions beyond condSeqLen remain zero-initialized
                }
            }
            featureOffset += condFeatures;
        }

        return result;
    }

    /// <summary>
    /// Blends two conditioning tensors.
    /// </summary>
    private Tensor<T> BlendConditions(Tensor<T> primary, Tensor<T> secondary, double secondaryStrength)
    {
        // Resize secondary to match primary if needed, then blend
        var result = new Tensor<T>(primary.Shape);
        var resultSpan = result.AsWritableSpan();
        var primarySpan = primary.AsSpan();
        var secondarySpan = secondary.AsSpan();

        for (int i = 0; i < resultSpan.Length; i++)
        {
            var pVal = NumOps.ToDouble(primarySpan[i]);
            var sVal = i < secondarySpan.Length ? NumOps.ToDouble(secondarySpan[i]) : 0.0;
            resultSpan[i] = NumOps.FromDouble(pVal * (1 - secondaryStrength) + sVal * secondaryStrength);
        }

        return result;
    }

    /// <summary>
    /// Blends two audio tensors with crossfade.
    /// </summary>
    private Tensor<T> BlendAudio(Tensor<T> first, Tensor<T> second, int overlapSamples)
    {
        var firstLen = first.Shape[^1];
        var secondLen = second.Shape[^1];

        // Clamp overlap to valid range to prevent negative indices
        overlapSamples = Math.Max(0, Math.Min(overlapSamples, Math.Min(firstLen, secondLen)));

        var totalLen = firstLen + secondLen - overlapSamples;

        var result = new Tensor<T>(new[] { 1, totalLen });
        var resultSpan = result.AsWritableSpan();
        var firstSpan = first.AsSpan();
        var secondSpan = second.AsSpan();

        // Copy first part (before overlap)
        for (int i = 0; i < firstLen - overlapSamples; i++)
        {
            resultSpan[i] = firstSpan[i];
        }

        // Crossfade overlap
        for (int i = 0; i < overlapSamples; i++)
        {
            var fadeOut = 1.0 - (double)i / overlapSamples;
            var fadeIn = (double)i / overlapSamples;

            var firstIdx = firstLen - overlapSamples + i;
            var secondIdx = i;
            var resultIdx = firstLen - overlapSamples + i;

            var fVal = firstIdx < firstSpan.Length ? NumOps.ToDouble(firstSpan[firstIdx]) : 0.0;
            var sVal = secondIdx < secondSpan.Length ? NumOps.ToDouble(secondSpan[secondIdx]) : 0.0;

            resultSpan[resultIdx] = NumOps.FromDouble(fVal * fadeOut + sVal * fadeIn);
        }

        // Copy second part (after overlap)
        for (int i = overlapSamples; i < secondLen; i++)
        {
            var resultIdx = firstLen - overlapSamples + i;
            if (resultIdx < resultSpan.Length && i < secondSpan.Length)
            {
                resultSpan[resultIdx] = secondSpan[i];
            }
        }

        return result;
    }

    #region IParameterizable Implementation

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var unetParams = _unet.GetParameters();
        var vaeParams = _musicVAE.GetParameters();
        var melodyParams = _melodyEncoder.GetParameters();
        var rhythmParams = _rhythmEncoder.GetParameters();

        var totalLength = unetParams.Length + vaeParams.Length +
                         melodyParams.Length + rhythmParams.Length;
        var combined = new Vector<T>(totalLength);

        int offset = 0;

        for (int i = 0; i < unetParams.Length; i++)
            combined[offset + i] = unetParams[i];
        offset += unetParams.Length;

        for (int i = 0; i < vaeParams.Length; i++)
            combined[offset + i] = vaeParams[i];
        offset += vaeParams.Length;

        for (int i = 0; i < melodyParams.Length; i++)
            combined[offset + i] = melodyParams[i];
        offset += melodyParams.Length;

        for (int i = 0; i < rhythmParams.Length; i++)
            combined[offset + i] = rhythmParams[i];

        return combined;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        var unetCount = _unet.ParameterCount;
        var vaeCount = _musicVAE.ParameterCount;
        var melodyCount = _melodyEncoder.ParameterCount;
        var rhythmCount = _rhythmEncoder.ParameterCount;

        var expected = unetCount + vaeCount + melodyCount + rhythmCount;
        if (parameters.Length != expected)
            throw new ArgumentException($"Expected {expected} parameters, got {parameters.Length}.");

        int offset = 0;

        var unetParams = new Vector<T>(unetCount);
        for (int i = 0; i < unetCount; i++)
            unetParams[i] = parameters[offset + i];
        offset += unetCount;

        var vaeParams = new Vector<T>(vaeCount);
        for (int i = 0; i < vaeCount; i++)
            vaeParams[i] = parameters[offset + i];
        offset += vaeCount;

        var melodyParams = new Vector<T>(melodyCount);
        for (int i = 0; i < melodyCount; i++)
            melodyParams[i] = parameters[offset + i];
        offset += melodyCount;

        var rhythmParams = new Vector<T>(rhythmCount);
        for (int i = 0; i < rhythmCount; i++)
            rhythmParams[i] = parameters[offset + i];

        _unet.SetParameters(unetParams);
        _musicVAE.SetParameters(vaeParams);
        _melodyEncoder.SetParameters(melodyParams);
        _rhythmEncoder.SetParameters(rhythmParams);
    }

    /// <inheritdoc />
    public override int ParameterCount =>
        _unet.ParameterCount + _musicVAE.ParameterCount +
        _melodyEncoder.ParameterCount + _rhythmEncoder.ParameterCount;

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
        return new MusicGenModel<T>(
            options: null,
            scheduler: null,
            unet: null,
            musicVAE: null,
            textConditioner: _textConditioner,
            modelSize: _modelSize,
            sampleRate: SampleRate,
            defaultDurationSeconds: DefaultDurationSeconds);
    }

    #endregion
}

/// <summary>
/// MusicGen model size variants.
/// </summary>
public enum MusicGenSize
{
    /// <summary>
    /// Small model (256 base channels) - faster, less detailed.
    /// </summary>
    Small,

    /// <summary>
    /// Medium model (384 base channels) - balanced.
    /// </summary>
    Medium,

    /// <summary>
    /// Large model (512 base channels) - highest quality.
    /// </summary>
    Large,

    /// <summary>
    /// Melody model (512 channels) - optimized for melody conditioning.
    /// </summary>
    Melody
}

/// <summary>
/// Melody encoder for extracting melodic features from audio.
/// </summary>
/// <typeparam name="T">Numeric type.</typeparam>
public class MelodyEncoder<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly Matrix<T> _convWeights;
    private readonly Vector<T> _convBias;
    private readonly Matrix<T> _projWeights;
    private readonly Vector<T> _projBias;
    private readonly int _inputChannels;
    private readonly int _outputDim;
    private readonly int _intermediateChannels;

    /// <summary>
    /// Creates a new melody encoder.
    /// </summary>
    public MelodyEncoder(int inputChannels, int outputDim, int? seed = null)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _inputChannels = inputChannels;
        _outputDim = outputDim;
        _intermediateChannels = 64;

        var rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();
        var convScale = Math.Sqrt(2.0 / inputChannels);
        var projScale = Math.Sqrt(2.0 / _intermediateChannels);

        _convWeights = new Matrix<T>(_intermediateChannels, inputChannels * 7); // 7-wide kernel
        _convBias = new Vector<T>(_intermediateChannels);
        _projWeights = new Matrix<T>(outputDim, _intermediateChannels);
        _projBias = new Vector<T>(outputDim);

        // Initialize weights
        for (int i = 0; i < _intermediateChannels; i++)
        {
            for (int j = 0; j < inputChannels * 7; j++)
            {
                _convWeights[i, j] = _numOps.FromDouble((rng.NextDouble() * 2 - 1) * convScale);
            }
            _convBias[i] = _numOps.Zero;
        }

        for (int i = 0; i < outputDim; i++)
        {
            for (int j = 0; j < _intermediateChannels; j++)
            {
                _projWeights[i, j] = _numOps.FromDouble((rng.NextDouble() * 2 - 1) * projScale);
            }
            _projBias[i] = _numOps.Zero;
        }
    }

    /// <summary>
    /// Encodes melody from audio waveform.
    /// </summary>
    public Tensor<T> EncodeMelody(Tensor<T> audio, int sampleRate)
    {
        // Simplified pitch/melody extraction
        // In production, this would use pitch detection algorithms
        var numSamples = audio.Shape[^1];
        var frameHop = sampleRate / 50; // 50 Hz frame rate
        var numFrames = numSamples / frameHop;

        var features = new Tensor<T>(new[] { 1, numFrames, _outputDim });
        var featSpan = features.AsWritableSpan();
        var audioSpan = audio.AsSpan();

        for (int f = 0; f < numFrames; f++)
        {
            var frameStart = f * frameHop;

            // Extract frame energy and zero-crossing rate as basic features
            var energy = 0.0;
            var zcr = 0;

            for (int i = frameStart; i < Math.Min(frameStart + frameHop, numSamples); i++)
            {
                var sample = _numOps.ToDouble(audioSpan[i % audioSpan.Length]);
                energy += sample * sample;

                if (i > frameStart)
                {
                    var prev = _numOps.ToDouble(audioSpan[(i - 1) % audioSpan.Length]);
                    if ((sample >= 0 && prev < 0) || (sample < 0 && prev >= 0))
                        zcr++;
                }
            }

            energy = Math.Sqrt(energy / frameHop);
            var zcrNorm = zcr / (double)frameHop;

            // Project features through network
            for (int d = 0; d < _outputDim; d++)
            {
                var val = energy * _numOps.ToDouble(_projWeights[d, 0]) +
                         zcrNorm * _numOps.ToDouble(_projWeights[d, 1 % _intermediateChannels]) +
                         _numOps.ToDouble(_projBias[d]);

                // ReLU activation
                val = Math.Max(0, val);

                var idx = f * _outputDim + d;
                if (idx < featSpan.Length)
                {
                    featSpan[idx] = _numOps.FromDouble(val);
                }
            }
        }

        return features;
    }

    /// <summary>
    /// Gets parameters.
    /// </summary>
    public Vector<T> GetParameters()
    {
        var convParams = _intermediateChannels * _inputChannels * 7 + _intermediateChannels;
        var projParams = _outputDim * _intermediateChannels + _outputDim;
        var total = convParams + projParams;

        var result = new Vector<T>(total);
        int idx = 0;

        for (int i = 0; i < _intermediateChannels; i++)
        {
            for (int j = 0; j < _inputChannels * 7; j++)
                result[idx++] = _convWeights[i, j];
        }
        for (int i = 0; i < _intermediateChannels; i++)
            result[idx++] = _convBias[i];

        for (int i = 0; i < _outputDim; i++)
        {
            for (int j = 0; j < _intermediateChannels; j++)
                result[idx++] = _projWeights[i, j];
        }
        for (int i = 0; i < _outputDim; i++)
            result[idx++] = _projBias[i];

        return result;
    }

    /// <summary>
    /// Sets parameters.
    /// </summary>
    public void SetParameters(Vector<T> parameters)
    {
        int idx = 0;

        for (int i = 0; i < _intermediateChannels; i++)
        {
            for (int j = 0; j < _inputChannels * 7; j++)
                _convWeights[i, j] = parameters[idx++];
        }
        for (int i = 0; i < _intermediateChannels; i++)
            _convBias[i] = parameters[idx++];

        for (int i = 0; i < _outputDim; i++)
        {
            for (int j = 0; j < _intermediateChannels; j++)
                _projWeights[i, j] = parameters[idx++];
        }
        for (int i = 0; i < _outputDim; i++)
            _projBias[i] = parameters[idx++];
    }

    /// <summary>
    /// Gets parameter count.
    /// </summary>
    public int ParameterCount =>
        _intermediateChannels * _inputChannels * 7 + _intermediateChannels +
        _outputDim * _intermediateChannels + _outputDim;
}

/// <summary>
/// Rhythm encoder for extracting beat/rhythm features from audio.
/// </summary>
/// <typeparam name="T">Numeric type.</typeparam>
public class RhythmEncoder<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly Matrix<T> _weights;
    private readonly Vector<T> _bias;
    private readonly int _outputDim;
    private readonly int _inputDim;

    /// <summary>
    /// Creates a new rhythm encoder.
    /// </summary>
    public RhythmEncoder(int outputDim, int? seed = null)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _outputDim = outputDim;
        _inputDim = 32; // Onset strength over 32 frequency bands

        var rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();
        var scale = Math.Sqrt(2.0 / _inputDim);

        _weights = new Matrix<T>(outputDim, _inputDim);
        _bias = new Vector<T>(outputDim);

        for (int i = 0; i < outputDim; i++)
        {
            for (int j = 0; j < _inputDim; j++)
            {
                _weights[i, j] = _numOps.FromDouble((rng.NextDouble() * 2 - 1) * scale);
            }
            _bias[i] = _numOps.Zero;
        }
    }

    /// <summary>
    /// Encodes rhythm from audio waveform.
    /// </summary>
    public Tensor<T> EncodeRhythm(Tensor<T> audio, int sampleRate)
    {
        var numSamples = audio.Shape[^1];
        var frameHop = sampleRate / 100; // 100 Hz frame rate for rhythm
        var numFrames = numSamples / frameHop;

        var features = new Tensor<T>(new[] { 1, numFrames, _outputDim });
        var featSpan = features.AsWritableSpan();
        var audioSpan = audio.AsSpan();

        var prevEnergy = 0.0;

        for (int f = 0; f < numFrames; f++)
        {
            var frameStart = f * frameHop;

            // Compute onset strength (simplified spectral flux)
            var energy = 0.0;
            for (int i = frameStart; i < Math.Min(frameStart + frameHop, numSamples); i++)
            {
                var sample = _numOps.ToDouble(audioSpan[i % audioSpan.Length]);
                energy += sample * sample;
            }
            energy = Math.Sqrt(energy / frameHop);

            // Onset is positive energy change
            var onsetStrength = Math.Max(0, energy - prevEnergy);
            prevEnergy = energy;

            // Project through network
            for (int d = 0; d < _outputDim; d++)
            {
                var val = onsetStrength * _numOps.ToDouble(_weights[d, 0]) +
                         energy * _numOps.ToDouble(_weights[d, 1 % _inputDim]) +
                         _numOps.ToDouble(_bias[d]);

                // ReLU
                val = Math.Max(0, val);

                var idx = f * _outputDim + d;
                if (idx < featSpan.Length)
                {
                    featSpan[idx] = _numOps.FromDouble(val);
                }
            }
        }

        return features;
    }

    /// <summary>
    /// Gets parameters.
    /// </summary>
    public Vector<T> GetParameters()
    {
        var total = _outputDim * _inputDim + _outputDim;
        var result = new Vector<T>(total);

        int idx = 0;
        for (int i = 0; i < _outputDim; i++)
        {
            for (int j = 0; j < _inputDim; j++)
                result[idx++] = _weights[i, j];
        }
        for (int i = 0; i < _outputDim; i++)
            result[idx++] = _bias[i];

        return result;
    }

    /// <summary>
    /// Sets parameters.
    /// </summary>
    public void SetParameters(Vector<T> parameters)
    {
        int idx = 0;
        for (int i = 0; i < _outputDim; i++)
        {
            for (int j = 0; j < _inputDim; j++)
                _weights[i, j] = parameters[idx++];
        }
        for (int i = 0; i < _outputDim; i++)
            _bias[i] = parameters[idx++];
    }

    /// <summary>
    /// Gets parameter count.
    /// </summary>
    public int ParameterCount => _outputDim * _inputDim + _outputDim;
}
