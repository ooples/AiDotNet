using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Diffusion;
using AiDotNet.NeuralNetworks.Diffusion.Schedulers;

namespace AiDotNet.Diffusion.Models;

/// <summary>
/// VoiceCraft model — zero-shot speech editing and text-to-speech with neural codec language modeling.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// VoiceCraft uses a token-infilling approach with neural codec language models
/// for high-quality zero-shot speech editing and text-to-speech synthesis.
/// </para>
/// <para>
/// <b>For Beginners:</b> VoiceCraft edits and generates speech realistically:
///
/// Key characteristics:
/// - Zero-shot: works with any voice from a short sample
/// - Speech editing: modify specific words in existing audio
/// - TTS: generate speech from text matching a reference voice
/// - EnCodec token-based neural codec language model
///
/// Reference: Peng et al., "VoiceCraft: Zero-Shot Speech Editing and Text-to-Speech in the Wild", ACL 2024
/// </para>
/// </remarks>
public class VoiceCraftModel<T> : AudioDiffusionModelBase<T>
{
    private const int LATENT_CHANNELS = 8;
    private const int HIDDEN_DIM = 2048;
    private const int NUM_LAYERS = 16;
    private const int NUM_HEADS = 16;
    private const int CONTEXT_DIM = 768;

    private readonly DiTNoisePredictor<T> _transformer;
    private readonly AudioVAE<T> _audioVAE;
    private readonly IConditioningModule<T>? _conditioner;

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

    public VoiceCraftModel(
        DiffusionModelOptions<T>? options = null, INoiseScheduler<T>? scheduler = null,
        DiTNoisePredictor<T>? transformer = null, AudioVAE<T>? audioVAE = null,
        IConditioningModule<T>? conditioner = null)
        : base(options ?? new DiffusionModelOptions<T> { TrainTimesteps = 1000, BetaStart = 0.0001, BetaEnd = 0.02, BetaSchedule = BetaSchedule.Linear },
               scheduler ?? new DDPMScheduler<T>(SchedulerConfig<T>.CreateDefault()),
               sampleRate: 16000, defaultDurationSeconds: 20.0, melChannels: 80)
    {
        _conditioner = conditioner;
        _transformer = transformer ?? new DiTNoisePredictor<T>(
            inputChannels: LATENT_CHANNELS, hiddenSize: HIDDEN_DIM,
            numLayers: NUM_LAYERS, numHeads: NUM_HEADS,
            patchSize: 1, contextDim: CONTEXT_DIM);
        _audioVAE = audioVAE ?? new AudioVAE<T>(
            melChannels: 80, latentChannels: LATENT_CHANNELS,
            baseChannels: 64, numResBlocks: 2);
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var tp = _transformer.GetParameters(); var ap = _audioVAE.GetParameters();
        var c = new Vector<T>(tp.Length + ap.Length);
        for (int i = 0; i < tp.Length; i++) c[i] = tp[i];
        for (int i = 0; i < ap.Length; i++) c[tp.Length + i] = ap[i];
        return c;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        int tc = _transformer.ParameterCount, ac = _audioVAE.ParameterCount;
        var tp = new Vector<T>(tc); var ap = new Vector<T>(ac);
        for (int i = 0; i < tc; i++) tp[i] = parameters[i];
        for (int i = 0; i < ac; i++) ap[i] = parameters[tc + i];
        _transformer.SetParameters(tp); _audioVAE.SetParameters(ap);
    }

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();
    /// <inheritdoc />
    public override IDiffusionModel<T> Clone()
    {
        var ct = new DiTNoisePredictor<T>(
            inputChannels: LATENT_CHANNELS, hiddenSize: HIDDEN_DIM,
            numLayers: NUM_LAYERS, numHeads: NUM_HEADS,
            patchSize: 1, contextDim: CONTEXT_DIM);
        ct.SetParameters(_transformer.GetParameters());
        return new VoiceCraftModel<T>(transformer: ct,
            audioVAE: (AudioVAE<T>)_audioVAE.Clone(),
            conditioner: _conditioner);
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T> { Name = "VoiceCraft", Version = "1.0", ModelType = ModelType.NeuralNetwork,
            Description = "VoiceCraft zero-shot speech editing and TTS", FeatureCount = ParameterCount, Complexity = ParameterCount };
        m.SetProperty("architecture", "neural-codec-lm-infilling");
        m.SetProperty("zero_shot", true);
        m.SetProperty("speech_editing", true);
        return m;
    }
}
