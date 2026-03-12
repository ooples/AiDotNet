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
/// JEN-1 model for high-fidelity text-to-music generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// JEN-1 is a universal high-fidelity music generation model that combines autoregressive
/// and non-autoregressive training in a multi-task framework. It generates music at 48kHz
/// in both text-to-music and music continuation modes.
/// </para>
/// <para>
/// <b>For Beginners:</b> JEN-1 creates music from text descriptions:
///
/// How JEN-1 works:
/// 1. Text is encoded by a FLAN-T5 text encoder
/// 2. Audio is encoded by a 1D VAE into a compressed latent representation
/// 3. A diffusion model denoises latents conditioned on text
/// 4. The 1D VAE decodes latents back to 48kHz audio
///
/// Key characteristics:
/// - High-fidelity 48kHz audio output
/// - Text-to-music: Generate music from text descriptions
/// - Music continuation: Extend an existing music clip
/// - Music inpainting: Fill in missing parts of music
/// - Multi-task training: Autoregressive + non-autoregressive
///
/// Advantages:
/// - High audio quality (48kHz)
/// - Versatile (text-to-music, continuation, inpainting)
/// - Good musicality and coherence
/// - Efficient 1D latent representation
///
/// Limitations:
/// - Limited to ~10 second clips
/// - Generation takes several seconds
/// - Music quality varies with prompt complexity
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: 1D VAE + Latent Diffusion
/// - Audio encoder: 1D convolutional VAE, 128 latent channels
/// - Diffusion backbone: 1D U-Net with cross-attention
/// - Text encoder: FLAN-T5 Large (1024-dim embeddings)
/// - Sample rate: 48,000 Hz
/// - Duration: Up to 10 seconds
/// - Noise schedule: Linear beta, 1000 timesteps
///
/// Reference: Li et al., "JEN-1: Text-Guided Universal Music Generation with Omnidirectional Diffusion Models", 2023
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var jen1 = new JEN1Model&lt;float&gt;();
/// var audio = jen1.GenerateFromText(
///     prompt: "Upbeat electronic dance music with a driving bass line",
///     durationSeconds: 10.0,
///     numInferenceSteps: 100,
///     guidanceScale: 3.0,
///     seed: 42);
/// </code>
/// </example>
public class JEN1Model<T> : AudioDiffusionModelBase<T>
{
    #region Constants

    private const int JEN1_LATENT_CHANNELS = 128;
    private const int JEN1_CROSS_ATTENTION_DIM = 1024;
    private const double JEN1_DEFAULT_GUIDANCE_SCALE = 3.0;
    private const int JEN1_SAMPLE_RATE = 48000;
    private const double JEN1_DEFAULT_DURATION = 10.0;

    #endregion

    #region Fields

    private UNetNoisePredictor<T> _unet;
    private AudioVAE<T> _audioVae;
    private readonly IConditioningModule<T>? _conditioner;

    #endregion

    #region Properties

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _unet;

    /// <inheritdoc />
    public override IVAEModel<T> VAE => _audioVae;

    /// <inheritdoc />
    public override IConditioningModule<T>? Conditioner => _conditioner;

    /// <inheritdoc />
    public override int LatentChannels => JEN1_LATENT_CHANNELS;

    /// <inheritdoc />
    public override bool SupportsTextToAudio => true;

    /// <inheritdoc />
    public override bool SupportsTextToMusic => true;

    /// <inheritdoc />
    public override bool SupportsTextToSpeech => false;

    /// <inheritdoc />
    public override bool SupportsAudioToAudio => true;

    /// <inheritdoc />
    public override int ParameterCount => _unet.ParameterCount + _audioVae.ParameterCount;

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes a new instance of JEN1Model with full customization support.
    /// </summary>
    /// <param name="options">Configuration options. If null, uses JEN-1 defaults.</param>
    /// <param name="scheduler">Custom noise scheduler.</param>
    /// <param name="unet">Custom 1D U-Net for latent diffusion.</param>
    /// <param name="audioVae">Custom 1D audio VAE encoder/decoder.</param>
    /// <param name="conditioner">Text encoder conditioning module (typically FLAN-T5).</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public JEN1Model(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        UNetNoisePredictor<T>? unet = null,
        AudioVAE<T>? audioVae = null,
        IConditioningModule<T>? conditioner = null,
        int? seed = null)
        : base(
            options ?? new DiffusionModelOptions<T>
            {
                TrainTimesteps = 1000,
                BetaStart = 0.0001,
                BetaEnd = 0.02,
                BetaSchedule = BetaSchedule.Linear
            },
            scheduler ?? new DDIMScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion()),
            JEN1_SAMPLE_RATE,
            JEN1_DEFAULT_DURATION,
            melChannels: 128,
            architecture: architecture)
    {
        _conditioner = conditioner;

        InitializeLayers(unet, audioVae, seed);

        SetGuidanceScale(JEN1_DEFAULT_GUIDANCE_SCALE);
    }

    #endregion

    #region Layer Initialization

    [MemberNotNull(nameof(_unet), nameof(_audioVae))]
    private void InitializeLayers(
        UNetNoisePredictor<T>? unet,
        AudioVAE<T>? audioVae,
        int? seed)
    {
        // 1D U-Net for latent audio diffusion
        _unet = unet ?? new UNetNoisePredictor<T>(
            inputChannels: JEN1_LATENT_CHANNELS,
            outputChannels: JEN1_LATENT_CHANNELS,
            baseChannels: 256,
            channelMultipliers: [1, 2, 4, 4],
            numResBlocks: 2,
            attentionResolutions: [4, 2, 1],
            contextDim: JEN1_CROSS_ATTENTION_DIM,
            architecture: Architecture,
            seed: seed);

        // 1D Audio VAE for waveform encoding/decoding
        _audioVae = audioVae ?? new AudioVAE<T>(
            melChannels: 128,
            latentChannels: JEN1_LATENT_CHANNELS,
            baseChannels: 64,
            numResBlocks: 2,
            seed: seed);
    }

    #endregion

    #region IParameterizable Implementation

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var unetParams = _unet.GetParameters();
        var vaeParams = _audioVae.GetParameters();

        var totalLength = unetParams.Length + vaeParams.Length;
        var combined = new Vector<T>(totalLength);

        for (int i = 0; i < unetParams.Length; i++)
            combined[i] = unetParams[i];
        for (int i = 0; i < vaeParams.Length; i++)
            combined[unetParams.Length + i] = vaeParams[i];

        return combined;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        var unetCount = _unet.ParameterCount;
        var vaeCount = _audioVae.ParameterCount;

        if (parameters.Length != unetCount + vaeCount)
        {
            throw new ArgumentException(
                $"Expected {unetCount + vaeCount} parameters, got {parameters.Length}.",
                nameof(parameters));
        }

        var unetParams = new Vector<T>(unetCount);
        var vaeParams = new Vector<T>(vaeCount);

        for (int i = 0; i < unetCount; i++)
            unetParams[i] = parameters[i];
        for (int i = 0; i < vaeCount; i++)
            vaeParams[i] = parameters[unetCount + i];

        _unet.SetParameters(unetParams);
        _audioVae.SetParameters(vaeParams);
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
        var clonedUnet = new UNetNoisePredictor<T>(
            inputChannels: JEN1_LATENT_CHANNELS, outputChannels: JEN1_LATENT_CHANNELS,
            baseChannels: 256, channelMultipliers: [1, 2, 4, 4],
            numResBlocks: 2, attentionResolutions: [4, 2, 1],
            contextDim: JEN1_CROSS_ATTENTION_DIM);
        clonedUnet.SetParameters(_unet.GetParameters());

        var clonedVae = new AudioVAE<T>(
            melChannels: 128, latentChannels: JEN1_LATENT_CHANNELS,
            baseChannels: 64, numResBlocks: 2);
        clonedVae.SetParameters(_audioVae.GetParameters());

        return new JEN1Model<T>(
            unet: clonedUnet,
            audioVae: clonedVae,
            conditioner: _conditioner);
    }

    #endregion

    #region Metadata

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "JEN-1",
            Version = "1.0",
            ModelType = ModelType.NeuralNetwork,
            Description = "JEN-1 high-fidelity text-to-music generation with 1D latent diffusion at 48kHz",
            FeatureCount = ParameterCount,
            Complexity = ParameterCount
        };

        metadata.SetProperty("architecture", "1d-latent-diffusion-audio");
        metadata.SetProperty("text_encoder", "FLAN-T5 Large");
        metadata.SetProperty("cross_attention_dim", JEN1_CROSS_ATTENTION_DIM);
        metadata.SetProperty("latent_channels", JEN1_LATENT_CHANNELS);
        metadata.SetProperty("sample_rate", JEN1_SAMPLE_RATE);
        metadata.SetProperty("default_duration_seconds", JEN1_DEFAULT_DURATION);

        return metadata;
    }

    #endregion
}
