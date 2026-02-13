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
/// Key characteristics:
/// - DiT backbone with timing conditioning (start_s, total_s)
/// - 44.1 kHz sample rate for high-fidelity audio
/// - Autoencoder-based latent space
/// - Up to 47 seconds of stereo audio
///
/// Reference: Evans et al., "Stable Audio Open", Stability AI, 2024
/// </para>
/// </remarks>
public class StableAudioModel<T> : AudioDiffusionModelBase<T>
{
    private const int LATENT_CHANNELS = 64;
    private const int HIDDEN_DIM = 1536;
    private const int NUM_LAYERS = 24;
    private const int NUM_HEADS = 24;
    private const int CONTEXT_DIM = 768;

    private readonly DiTNoisePredictor<T> _dit;
    private readonly AudioVAE<T> _audioVAE;
    private readonly IConditioningModule<T>? _conditioner;

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

    public StableAudioModel(
        DiffusionModelOptions<T>? options = null, INoiseScheduler<T>? scheduler = null,
        DiTNoisePredictor<T>? dit = null, AudioVAE<T>? audioVAE = null,
        IConditioningModule<T>? conditioner = null)
        : base(options ?? new DiffusionModelOptions<T> { TrainTimesteps = 1000, BetaStart = 0.0001, BetaEnd = 0.02, BetaSchedule = BetaSchedule.Linear },
               scheduler ?? new FlowMatchingScheduler<T>(SchedulerConfig<T>.CreateDefault()),
               sampleRate: 44100, defaultDurationSeconds: 47.0, melChannels: 128)
    {
        _conditioner = conditioner;
        _dit = dit ?? new DiTNoisePredictor<T>(
            inputChannels: LATENT_CHANNELS, hiddenSize: HIDDEN_DIM,
            numLayers: NUM_LAYERS, numHeads: NUM_HEADS,
            patchSize: 1, contextDim: CONTEXT_DIM);
        _audioVAE = audioVAE ?? new AudioVAE<T>(
            melChannels: 128, latentChannels: LATENT_CHANNELS,
            baseChannels: 128, numResBlocks: 2);
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var dp = _dit.GetParameters(); var ap = _audioVAE.GetParameters();
        var c = new Vector<T>(dp.Length + ap.Length);
        for (int i = 0; i < dp.Length; i++) c[i] = dp[i];
        for (int i = 0; i < ap.Length; i++) c[dp.Length + i] = ap[i];
        return c;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        int dc = _dit.ParameterCount, ac = _audioVAE.ParameterCount;
        var dp = new Vector<T>(dc); var ap = new Vector<T>(ac);
        for (int i = 0; i < dc; i++) dp[i] = parameters[i];
        for (int i = 0; i < ac; i++) ap[i] = parameters[dc + i];
        _dit.SetParameters(dp); _audioVAE.SetParameters(ap);
    }

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();
    /// <inheritdoc />
    public override IDiffusionModel<T> Clone()
    {
        var cd = new DiTNoisePredictor<T>(
            inputChannels: LATENT_CHANNELS, hiddenSize: HIDDEN_DIM,
            numLayers: NUM_LAYERS, numHeads: NUM_HEADS,
            patchSize: 1, contextDim: CONTEXT_DIM);
        cd.SetParameters(_dit.GetParameters());
        return new StableAudioModel<T>(dit: cd,
            audioVAE: (AudioVAE<T>)_audioVAE.Clone(),
            conditioner: _conditioner);
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T> { Name = "Stable-Audio-Open", Version = "1.0", ModelType = ModelType.NeuralNetwork,
            Description = "Stable Audio Open DiT-based audio generation", FeatureCount = ParameterCount, Complexity = ParameterCount };
        m.SetProperty("architecture", "dit-timing-conditioned");
        m.SetProperty("sample_rate", 44100);
        m.SetProperty("max_duration_seconds", 47);
        return m;
    }
}
