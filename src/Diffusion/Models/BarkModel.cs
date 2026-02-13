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
/// Bark model — transformer-based text-to-speech with multi-lingual and non-speech support.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Bark uses a GPT-like architecture for text-to-audio generation, supporting
/// speech, music, sound effects, laughter, and other non-verbal audio.
/// </para>
/// <para>
/// <b>For Beginners:</b> Bark generates realistic speech and sounds:
///
/// Key characteristics:
/// - GPT-like auto-regressive generation of audio tokens
/// - Multi-lingual text-to-speech (10+ languages)
/// - Non-speech audio: laughter, music, sound effects
/// - Speaker cloning with voice presets
/// - EnCodec-based audio codec
///
/// Reference: Suno AI, "Bark: Text-Prompted Generative Audio Model", 2023
/// </para>
/// </remarks>
public class BarkModel<T> : AudioDiffusionModelBase<T>
{
    private const int LATENT_CHANNELS = 8;
    private const int HIDDEN_DIM = 1024;
    private const int NUM_LAYERS = 24;
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
    public override bool SupportsTextToAudio => true;
    /// <inheritdoc />
    public override bool SupportsTextToMusic => true;
    /// <inheritdoc />
    public override bool SupportsTextToSpeech => true;
    /// <inheritdoc />
    public override bool SupportsAudioToAudio => false;
    /// <inheritdoc />
    public override int ParameterCount => _transformer.ParameterCount + _audioVAE.ParameterCount;

    public BarkModel(
        DiffusionModelOptions<T>? options = null, INoiseScheduler<T>? scheduler = null,
        DiTNoisePredictor<T>? transformer = null, AudioVAE<T>? audioVAE = null,
        IConditioningModule<T>? conditioner = null)
        : base(options ?? new DiffusionModelOptions<T> { TrainTimesteps = 1000, BetaStart = 0.0001, BetaEnd = 0.02, BetaSchedule = BetaSchedule.Linear },
               scheduler ?? new DDPMScheduler<T>(SchedulerConfig<T>.CreateDefault()),
               sampleRate: 24000, defaultDurationSeconds: 15.0, melChannels: 100)
    {
        _conditioner = conditioner;
        _transformer = transformer ?? new DiTNoisePredictor<T>(
            inputChannels: LATENT_CHANNELS, hiddenSize: HIDDEN_DIM,
            numLayers: NUM_LAYERS, numHeads: NUM_HEADS,
            patchSize: 1, contextDim: CONTEXT_DIM);
        _audioVAE = audioVAE ?? new AudioVAE<T>(
            melChannels: 100, latentChannels: LATENT_CHANNELS,
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
        return new BarkModel<T>(transformer: ct,
            audioVAE: (AudioVAE<T>)_audioVAE.Clone(),
            conditioner: _conditioner);
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T> { Name = "Bark", Version = "1.0", ModelType = ModelType.NeuralNetwork,
            Description = "Bark GPT-based text-to-audio generation", FeatureCount = ParameterCount, Complexity = ParameterCount };
        m.SetProperty("architecture", "gpt-encodec");
        m.SetProperty("multilingual", true);
        m.SetProperty("non_speech_audio", true);
        return m;
    }
}
