using System.Diagnostics.CodeAnalysis;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Diffusion.Schedulers;

namespace AiDotNet.Diffusion.Audio;

/// <summary>
/// Stable Audio Open model — DiT-based latent diffusion for long-form audio generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Stable Audio uses a DiT operating on audio latents with timing conditioning
/// for variable-length, high-quality audio generation at 44.1 kHz.
/// </para>
/// <para>
/// <b>For Beginners:</b> Stable Audio creates music and sound effects from text:
///
/// How Stable Audio works:
/// 1. Text is encoded using a CLAP text encoder for audio-aware embeddings
/// 2. A DiT (Diffusion Transformer) denoises latents conditioned on text and timing
/// 3. Timing embeddings (start_s, total_s) control where and how long to generate
/// 4. An autoencoder decodes the latents back to 44.1 kHz stereo audio
///
/// Key characteristics:
/// - DiT backbone with timing conditioning (start_s, total_s)
/// - 44.1 kHz sample rate for high-fidelity audio
/// - Autoencoder-based latent space
/// - Up to 47 seconds of stereo audio
/// - Supports both music and sound effect generation
///
/// Advantages:
/// - High audio quality at 44.1 kHz
/// - Variable-length generation via timing conditioning
/// - Strong prompt adherence through CLAP embeddings
/// - Open-source weights and architecture
///
/// Limitations:
/// - Maximum duration of 47 seconds
/// - No text-to-speech capability
/// - Requires significant compute for real-time generation
///
/// Reference: Evans et al., "Stable Audio Open", Stability AI, 2024
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a Stable Audio model with default settings
/// var stableAudio = new StableAudioModel&lt;float&gt;();
///
/// // Generate ambient music from a text prompt
/// var music = stableAudio.GenerateFromText(
///     prompt: "Ambient electronic music with soft pads and gentle arpeggios",
///     durationSeconds: 30.0,
///     numInferenceSteps: 100,
///     guidanceScale: 3.5,
///     seed: 42);
///
/// // Transform existing audio with a new style
/// var transformed = stableAudio.AudioToAudio(
///     inputAudio: originalAudio,
///     prompt: "Lo-fi version with vinyl crackle",
///     strength: 0.6);
/// </code>
/// </example>
public class StableAudioModel<T> : AudioDiffusionModelBase<T>
{
    #region Constants

    private const int LATENT_CHANNELS = 64;
    private const int HIDDEN_DIM = 1536;
    private const int NUM_LAYERS = 24;
    private const int NUM_HEADS = 24;
    private const int CONTEXT_DIM = 768;
    private const int STABLE_AUDIO_SAMPLE_RATE = 44100;
    private const double STABLE_AUDIO_MAX_DURATION = 47.0;
    private const int STABLE_AUDIO_MEL_CHANNELS = 128;

    #endregion

    #region Fields

    private DiTNoisePredictor<T> _dit;
    private AudioVAE<T> _audioVAE;
    private readonly IConditioningModule<T>? _conditioner;

    #endregion

    #region Properties

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _dit;

    /// <inheritdoc />
    public override IVAEModel<T> VAE => _audioVAE;

    /// <inheritdoc />
    public override IConditioningModule<T>? Conditioner => _conditioner;

    /// <inheritdoc />
    public override int LatentChannels => LATENT_CHANNELS;

    /// <inheritdoc />
    public override bool SupportsTextToAudio => true;

    /// <inheritdoc />
    public override bool SupportsTextToMusic => true;

    /// <inheritdoc />
    public override bool SupportsTextToSpeech => false;

    /// <inheritdoc />
    public override bool SupportsAudioToAudio => true;

    /// <inheritdoc />
    public override int ParameterCount => _dit.ParameterCount + _audioVAE.ParameterCount;

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes a new instance of StableAudioModel with full customization support.
    /// </summary>
    /// <param name="architecture">Optional neural network architecture specification.</param>
    /// <param name="options">Configuration options for the diffusion model. If null, uses Stable Audio defaults.</param>
    /// <param name="scheduler">Custom noise scheduler. If null, uses flow matching.</param>
    /// <param name="dit">Custom DiT noise predictor.</param>
    /// <param name="audioVAE">Custom audio VAE encoder/decoder.</param>
    /// <param name="conditioner">Optional CLAP conditioning module for text guidance.</param>
    public StableAudioModel(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        DiTNoisePredictor<T>? dit = null,
        AudioVAE<T>? audioVAE = null,
        IConditioningModule<T>? conditioner = null)
        : base(
            options ?? new DiffusionModelOptions<T>
            {
                TrainTimesteps = 1000,
                BetaStart = 0.0001,
                BetaEnd = 0.02,
                BetaSchedule = BetaSchedule.Linear
            },
            scheduler ?? new FlowMatchingScheduler<T>(SchedulerConfig<T>.CreateDefault()),
            sampleRate: STABLE_AUDIO_SAMPLE_RATE,
            defaultDurationSeconds: STABLE_AUDIO_MAX_DURATION,
            melChannels: STABLE_AUDIO_MEL_CHANNELS,
            architecture: architecture)
    {
        _conditioner = conditioner;

        InitializeLayers(dit, audioVAE);
    }

    #endregion

    #region Layer Initialization

    [MemberNotNull(nameof(_dit), nameof(_audioVAE))]
    private void InitializeLayers(
        DiTNoisePredictor<T>? dit,
        AudioVAE<T>? audioVAE)
    {
        _dit = dit ?? new DiTNoisePredictor<T>(
            inputChannels: LATENT_CHANNELS,
            hiddenSize: HIDDEN_DIM,
            numLayers: NUM_LAYERS,
            numHeads: NUM_HEADS,
            patchSize: 1,
            contextDim: CONTEXT_DIM);

        _audioVAE = audioVAE ?? new AudioVAE<T>(
            melChannels: STABLE_AUDIO_MEL_CHANNELS,
            latentChannels: LATENT_CHANNELS,
            baseChannels: 128,
            numResBlocks: 2);
    }

    #endregion

    #region Generation Methods

    // Generation methods are inherited from AudioDiffusionModelBase<T>.

    #endregion

    #region IParameterizable

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var ditParams = _dit.GetParameters();
        var vaeParams = _audioVAE.GetParameters();

        var combined = new Vector<T>(ditParams.Length + vaeParams.Length);

        for (int i = 0; i < ditParams.Length; i++)
        {
            combined[i] = ditParams[i];
        }

        for (int i = 0; i < vaeParams.Length; i++)
        {
            combined[ditParams.Length + i] = vaeParams[i];
        }

        return combined;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        int ditCount = _dit.ParameterCount;
        int vaeCount = _audioVAE.ParameterCount;

        if (parameters.Length != ditCount + vaeCount)
        {
            throw new ArgumentException(
                $"Expected {ditCount + vaeCount} parameters, got {parameters.Length}.",
                nameof(parameters));
        }

        var ditParams = new Vector<T>(ditCount);
        var vaeParams = new Vector<T>(vaeCount);

        for (int i = 0; i < ditCount; i++)
        {
            ditParams[i] = parameters[i];
        }

        for (int i = 0; i < vaeCount; i++)
        {
            vaeParams[i] = parameters[ditCount + i];
        }

        _dit.SetParameters(ditParams);
        _audioVAE.SetParameters(vaeParams);
    }

    #endregion

    #region ICloneable

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy()
    {
        return Clone();
    }

    /// <inheritdoc />
    public override IDiffusionModel<T> Clone()
    {
        var clonedDit = new DiTNoisePredictor<T>(
            inputChannels: LATENT_CHANNELS,
            hiddenSize: HIDDEN_DIM,
            numLayers: NUM_LAYERS,
            numHeads: NUM_HEADS,
            patchSize: 1,
            contextDim: CONTEXT_DIM);
        clonedDit.SetParameters(_dit.GetParameters());

        return new StableAudioModel<T>(
            dit: clonedDit,
            audioVAE: (AudioVAE<T>)_audioVAE.Clone(),
            conditioner: _conditioner);
    }

    #endregion

    #region Metadata

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "Stable-Audio-Open",
            Version = "1.0",
            ModelType = ModelType.NeuralNetwork,
            Description = "Stable Audio Open DiT-based audio generation with timing conditioning at 44.1 kHz",
            FeatureCount = ParameterCount,
            Complexity = ParameterCount
        };

        metadata.SetProperty("architecture", "dit-timing-conditioned");
        metadata.SetProperty("sample_rate", STABLE_AUDIO_SAMPLE_RATE);
        metadata.SetProperty("max_duration_seconds", STABLE_AUDIO_MAX_DURATION);
        metadata.SetProperty("latent_channels", LATENT_CHANNELS);
        metadata.SetProperty("hidden_dim", HIDDEN_DIM);
        metadata.SetProperty("num_transformer_layers", NUM_LAYERS);
        metadata.SetProperty("num_attention_heads", NUM_HEADS);
        metadata.SetProperty("context_dim", CONTEXT_DIM);
        metadata.SetProperty("mel_channels", STABLE_AUDIO_MEL_CHANNELS);
        metadata.SetProperty("supports_stereo", true);

        return metadata;
    }

    #endregion
}
