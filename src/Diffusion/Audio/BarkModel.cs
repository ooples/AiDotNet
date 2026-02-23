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
/// Bark model for transformer-based text-to-audio generation with multi-lingual speech, music, and sound effects.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Bark uses a GPT-like auto-regressive architecture with three transformer stages to generate
/// diverse audio content from text prompts, including speech in 10+ languages, music, laughter,
/// sighing, and other non-verbal sounds. Audio tokens are produced via an EnCodec codec.
/// </para>
/// <para>
/// Architecture components:
/// <list type="bullet">
/// <item><description>Semantic transformer (GPT-like, 1024 hidden, 24 layers, 16 heads) for text-to-semantic tokens</description></item>
/// <item><description>Coarse acoustic transformer for semantic-to-coarse audio tokens</description></item>
/// <item><description>Fine acoustic transformer for coarse-to-fine audio token refinement</description></item>
/// <item><description>CLIP text encoder for 768-dim conditioning</description></item>
/// <item><description>EnCodec-based audio codec for token-to-waveform synthesis</description></item>
/// <item><description>Speaker voice presets for zero-shot cloning</description></item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> Bark generates realistic speech and sounds from text prompts.
///
/// How Bark works:
/// 1. Text is tokenized and encoded via CLIP into 768-dim conditioning features
/// 2. Semantic transformer converts text tokens to high-level semantic audio tokens
/// 3. Coarse acoustic transformer maps semantic tokens to coarse EnCodec tokens
/// 4. Fine acoustic transformer refines coarse tokens to full-resolution EnCodec tokens
/// 5. EnCodec decoder converts audio tokens to a 24 kHz waveform
/// 6. Speaker presets enable voice cloning from short reference audio
///
/// Key characteristics:
/// - Three-stage GPT-like auto-regressive generation
/// - Multi-lingual speech in 10+ languages
/// - Non-speech audio: laughter, music, sound effects, sighing
/// - Speaker cloning with voice presets
/// - EnCodec-based audio codec at 24 kHz
/// - Open-source (Suno AI, MIT license)
///
/// When to use Bark:
/// - Multi-lingual text-to-speech generation
/// - Expressive speech with emotions and non-verbal sounds
/// - Quick audio prototyping from text descriptions
/// - When diverse audio output types are needed
///
/// Limitations:
/// - Auto-regressive generation is slower than parallel methods
/// - Maximum duration limited by context window
/// - Speaker cloning quality depends on reference audio
/// - Less control over fine-grained prosody
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: Three-stage GPT-like transformer
/// - Hidden dimension: 1024
/// - Transformer layers: 24
/// - Attention heads: 16
/// - Text encoder: CLIP (768-dim)
/// - Audio codec: EnCodec
/// - Sample rate: 24,000 Hz
/// - Default duration: 15 seconds
/// - Mel channels: 100
/// - Open-source: Yes (MIT license)
///
/// Reference: Suno AI, "Bark: Text-Prompted Generative Audio Model", 2023
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var bark = new BarkModel&lt;float&gt;();
/// var speech = bark.GenerateFromText(
///     prompt: "Hello, how are you today? [laughs]",
///     durationSeconds: 10.0,
///     numInferenceSteps: 100,
///     guidanceScale: 3.0);
/// </code>
/// </example>
public class BarkModel<T> : AudioDiffusionModelBase<T>
{
    #region Constants

    private const int LATENT_CHANNELS = 8;
    private const int HIDDEN_DIM = 1024;
    private const int NUM_LAYERS = 24;
    private const int NUM_HEADS = 16;
    private const int CONTEXT_DIM = 768;
    private const int SAMPLE_RATE = 24000;
    private const int MEL_CHANNELS = 100;
    private const double DEFAULT_DURATION = 15.0;

    #endregion

    #region Fields

    private DiTNoisePredictor<T> _transformer;
    private AudioVAE<T> _audioVAE;
    private readonly IConditioningModule<T>? _conditioner;

    #endregion

    #region Properties

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _transformer;
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
    public override bool SupportsTextToSpeech => true;
    /// <inheritdoc />
    public override bool SupportsAudioToAudio => false;
    /// <inheritdoc />
    public override int ParameterCount => _transformer.ParameterCount + _audioVAE.ParameterCount;

    #endregion

    #region Constructor

    public BarkModel(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        DiTNoisePredictor<T>? transformer = null,
        AudioVAE<T>? audioVAE = null,
        IConditioningModule<T>? conditioner = null,
        int? seed = null)
        : base(
            options ?? new DiffusionModelOptions<T>
            {
                TrainTimesteps = 1000, BetaStart = 0.0001,
                BetaEnd = 0.02, BetaSchedule = BetaSchedule.Linear
            },
            scheduler ?? new DDPMScheduler<T>(SchedulerConfig<T>.CreateDefault()),
            sampleRate: SAMPLE_RATE, defaultDurationSeconds: DEFAULT_DURATION,
            melChannels: MEL_CHANNELS, architecture: architecture)
    {
        _conditioner = conditioner;
        InitializeLayers(transformer, audioVAE, seed);
    }

    #endregion

    #region Layer Initialization

    [MemberNotNull(nameof(_transformer), nameof(_audioVAE))]
    private void InitializeLayers(DiTNoisePredictor<T>? transformer, AudioVAE<T>? audioVAE, int? seed)
    {
        _transformer = transformer ?? new DiTNoisePredictor<T>(
            inputChannels: LATENT_CHANNELS, hiddenSize: HIDDEN_DIM,
            numLayers: NUM_LAYERS, numHeads: NUM_HEADS,
            patchSize: 1, contextDim: CONTEXT_DIM);

        _audioVAE = audioVAE ?? new AudioVAE<T>(
            melChannels: MEL_CHANNELS, latentChannels: LATENT_CHANNELS,
            baseChannels: 64, numResBlocks: 2);
    }

    #endregion

    #region IParameterizable Implementation

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var tParams = _transformer.GetParameters();
        var vaeParams = _audioVAE.GetParameters();
        var combined = new Vector<T>(tParams.Length + vaeParams.Length);
        for (int i = 0; i < tParams.Length; i++) combined[i] = tParams[i];
        for (int i = 0; i < vaeParams.Length; i++) combined[tParams.Length + i] = vaeParams[i];
        return combined;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        var tCount = _transformer.ParameterCount;
        var vaeCount = _audioVAE.ParameterCount;
        if (parameters.Length != tCount + vaeCount)
            throw new ArgumentException($"Expected {tCount + vaeCount} parameters, got {parameters.Length}.", nameof(parameters));
        var tParams = new Vector<T>(tCount);
        var vaeParams = new Vector<T>(vaeCount);
        for (int i = 0; i < tCount; i++) tParams[i] = parameters[i];
        for (int i = 0; i < vaeCount; i++) vaeParams[i] = parameters[tCount + i];
        _transformer.SetParameters(tParams);
        _audioVAE.SetParameters(vaeParams);
    }

    #endregion

    #region ICloneable Implementation

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();

    /// <inheritdoc />
    public override IDiffusionModel<T> Clone()
    {
        var clonedTransformer = new DiTNoisePredictor<T>(
            inputChannels: LATENT_CHANNELS, hiddenSize: HIDDEN_DIM,
            numLayers: NUM_LAYERS, numHeads: NUM_HEADS,
            patchSize: 1, contextDim: CONTEXT_DIM);
        clonedTransformer.SetParameters(_transformer.GetParameters());
        return new BarkModel<T>(transformer: clonedTransformer,
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
            Name = "Bark", Version = "1.0", ModelType = ModelType.NeuralNetwork,
            Description = "Bark GPT-based text-to-audio generation with multi-lingual speech and sound effects",
            FeatureCount = ParameterCount, Complexity = ParameterCount
        };
        metadata.SetProperty("architecture", "gpt-encodec-three-stage");
        metadata.SetProperty("hidden_dim", HIDDEN_DIM);
        metadata.SetProperty("num_layers", NUM_LAYERS);
        metadata.SetProperty("num_heads", NUM_HEADS);
        metadata.SetProperty("audio_codec", "EnCodec");
        metadata.SetProperty("multilingual", true);
        metadata.SetProperty("non_speech_audio", true);
        metadata.SetProperty("speaker_cloning", true);
        metadata.SetProperty("sample_rate", SAMPLE_RATE);
        metadata.SetProperty("open_source", true);
        return metadata;
    }

    #endregion
}
