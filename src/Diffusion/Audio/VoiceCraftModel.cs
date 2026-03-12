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
/// VoiceCraft model for zero-shot speech editing and text-to-speech with neural codec language modeling.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// VoiceCraft uses a token-infilling approach with a neural codec language model to achieve
/// high-quality zero-shot speech editing and text-to-speech synthesis. It can modify specific
/// words in existing audio or generate new speech matching a reference voice, all without
/// fine-tuning on the target speaker.
/// </para>
/// <para>
/// Architecture components:
/// <list type="bullet">
/// <item><description>Causal masked transformer (2048 hidden, 16 layers, 16 heads) for token infilling</description></item>
/// <item><description>EnCodec neural audio codec for tokenization and reconstruction</description></item>
/// <item><description>Whisper-based text encoder for 768-dim conditioning</description></item>
/// <item><description>Delay pattern for multi-codebook parallel prediction</description></item>
/// <item><description>Token masking strategy for speech editing</description></item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> VoiceCraft edits and generates speech realistically from any voice.
///
/// How VoiceCraft works:
/// 1. Reference audio is encoded into EnCodec tokens (multi-codebook representation)
/// 2. Target text is encoded by Whisper into 768-dim conditioning features
/// 3. For editing: tokens at the edit region are masked; for TTS: continuation tokens are masked
/// 4. Causal masked transformer predicts tokens at masked positions via infilling
/// 5. Delay pattern allows parallel prediction across codebook levels
/// 6. EnCodec decoder reconstructs 16 kHz waveform from predicted tokens
///
/// Key characteristics:
/// - Zero-shot: works with any voice from a 3-second reference sample
/// - Speech editing: modify specific words while preserving voice and prosody
/// - TTS: generate speech from text matching the reference voice
/// - EnCodec token-based neural codec language model
/// - Delay pattern for efficient multi-codebook generation
/// - Open-source (MIT license)
///
/// When to use VoiceCraft:
/// - Zero-shot speech editing (correcting words in recordings)
/// - Zero-shot text-to-speech with voice cloning
/// - Audio post-production and correction
/// - Personalized speech synthesis from short samples
///
/// Limitations:
/// - 16 kHz output (lower than 24/44.1 kHz models)
/// - Requires aligned transcript for editing mode
/// - Quality depends on reference audio quality
/// - Longer edits may drift from reference prosody
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: Causal masked transformer with token infilling
/// - Hidden dimension: 2048
/// - Transformer layers: 16
/// - Attention heads: 16
/// - Text encoder: Whisper-based (768-dim)
/// - Audio codec: EnCodec (multi-codebook RVQ)
/// - Sample rate: 16,000 Hz
/// - Default duration: 20 seconds
/// - Mel channels: 80
/// - Open-source: Yes (MIT license)
///
/// Reference: Peng et al., "VoiceCraft: Zero-Shot Speech Editing and Text-to-Speech in the Wild", ACL 2024
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var voiceCraft = new VoiceCraftModel&lt;float&gt;();
/// var speech = voiceCraft.TextToSpeech(
///     text: "Hello, this is a test of VoiceCraft speech synthesis.",
///     speakerEmbedding: referenceEmbedding,
///     speakingRate: 1.0,
///     numInferenceSteps: 50);
/// </code>
/// </example>
public class VoiceCraftModel<T> : AudioDiffusionModelBase<T>
{
    #region Constants

    private const int LATENT_CHANNELS = 8;
    private const int HIDDEN_DIM = 2048;
    private const int NUM_LAYERS = 16;
    private const int NUM_HEADS = 16;
    private const int CONTEXT_DIM = 768;
    private const int SAMPLE_RATE = 16000;
    private const int MEL_CHANNELS = 80;
    private const double DEFAULT_DURATION = 20.0;

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
    public override bool SupportsTextToAudio => false;
    /// <inheritdoc />
    public override bool SupportsTextToMusic => false;
    /// <inheritdoc />
    public override bool SupportsTextToSpeech => true;
    /// <inheritdoc />
    public override bool SupportsAudioToAudio => true;
    /// <inheritdoc />
    public override int ParameterCount => _transformer.ParameterCount + _audioVAE.ParameterCount;

    #endregion

    #region Constructor

    public VoiceCraftModel(
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
        return new VoiceCraftModel<T>(transformer: clonedTransformer,
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
            Name = "VoiceCraft", Version = "1.0", ModelType = ModelType.NeuralNetwork,
            Description = "VoiceCraft zero-shot speech editing and TTS with neural codec language modeling",
            FeatureCount = ParameterCount, Complexity = ParameterCount
        };
        metadata.SetProperty("architecture", "neural-codec-lm-infilling");
        metadata.SetProperty("hidden_dim", HIDDEN_DIM);
        metadata.SetProperty("num_layers", NUM_LAYERS);
        metadata.SetProperty("num_heads", NUM_HEADS);
        metadata.SetProperty("zero_shot", true);
        metadata.SetProperty("speech_editing", true);
        metadata.SetProperty("audio_codec", "EnCodec");
        metadata.SetProperty("text_encoder", "Whisper-based");
        metadata.SetProperty("sample_rate", SAMPLE_RATE);
        metadata.SetProperty("open_source", true);
        return metadata;
    }

    #endregion
}
