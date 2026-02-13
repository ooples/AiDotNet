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
/// SoundStorm model — parallel masked audio token generation with conformer architecture.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// SoundStorm uses MaskGIT-style parallel decoding of SoundStream tokens,
/// generating all residual levels simultaneously for efficient audio synthesis.
/// </para>
/// <para>
/// <b>For Beginners:</b> SoundStorm generates speech very quickly:
///
/// Key characteristics:
/// - Parallel generation (not auto-regressive) for speed
/// - MaskGIT-style iterative refinement
/// - SoundStream codec tokens
/// - Conformer backbone for audio understanding
/// - 100x faster than auto-regressive approaches
///
/// Reference: Borsos et al., "SoundStorm: Efficient Parallel Audio Generation", 2023
/// </para>
/// </remarks>
public class SoundStormModel<T> : AudioDiffusionModelBase<T>
{
    private const int LATENT_CHANNELS = 8;
    private const int HIDDEN_DIM = 1024;
    private const int NUM_LAYERS = 12;
    private const int NUM_HEADS = 16;
    private const int CONTEXT_DIM = 1024;

    private readonly DiTNoisePredictor<T> _conformer;
    private readonly AudioVAE<T> _audioVAE;
    private readonly IConditioningModule<T>? _conditioner;

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _conformer;
    /// <inheritdoc />
    public override IVAEModel<T> VAE => _audioVAE;
    /// <inheritdoc />
    public override IConditioningModule<T>? Conditioner => _conditioner;
    /// <inheritdoc />
    public override int LatentChannels => LATENT_CHANNELS;
    /// <inheritdoc />
    public override bool SupportsTextToAudio => true;
    /// <inheritdoc />
    public override bool SupportsTextToMusic => false;
    /// <inheritdoc />
    public override bool SupportsTextToSpeech => true;
    /// <inheritdoc />
    public override bool SupportsAudioToAudio => false;
    /// <inheritdoc />
    public override int ParameterCount => _conformer.ParameterCount + _audioVAE.ParameterCount;

    public SoundStormModel(
        DiffusionModelOptions<T>? options = null, INoiseScheduler<T>? scheduler = null,
        DiTNoisePredictor<T>? conformer = null, AudioVAE<T>? audioVAE = null,
        IConditioningModule<T>? conditioner = null)
        : base(options ?? new DiffusionModelOptions<T> { TrainTimesteps = 1000, BetaStart = 0.0001, BetaEnd = 0.02, BetaSchedule = BetaSchedule.Linear },
               scheduler ?? new DDPMScheduler<T>(SchedulerConfig<T>.CreateDefault()),
               sampleRate: 24000, defaultDurationSeconds: 30.0, melChannels: 80)
    {
        _conditioner = conditioner;
        _conformer = conformer ?? new DiTNoisePredictor<T>(
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
        var cp = _conformer.GetParameters(); var ap = _audioVAE.GetParameters();
        var c = new Vector<T>(cp.Length + ap.Length);
        for (int i = 0; i < cp.Length; i++) c[i] = cp[i];
        for (int i = 0; i < ap.Length; i++) c[cp.Length + i] = ap[i];
        return c;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        int cc = _conformer.ParameterCount, ac = _audioVAE.ParameterCount;
        var cp = new Vector<T>(cc); var ap = new Vector<T>(ac);
        for (int i = 0; i < cc; i++) cp[i] = parameters[i];
        for (int i = 0; i < ac; i++) ap[i] = parameters[cc + i];
        _conformer.SetParameters(cp); _audioVAE.SetParameters(ap);
    }

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();
    /// <inheritdoc />
    public override IDiffusionModel<T> Clone()
    {
        var cc = new DiTNoisePredictor<T>(
            inputChannels: LATENT_CHANNELS, hiddenSize: HIDDEN_DIM,
            numLayers: NUM_LAYERS, numHeads: NUM_HEADS,
            patchSize: 1, contextDim: CONTEXT_DIM);
        cc.SetParameters(_conformer.GetParameters());
        return new SoundStormModel<T>(conformer: cc,
            audioVAE: (AudioVAE<T>)_audioVAE.Clone(),
            conditioner: _conditioner);
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T> { Name = "SoundStorm", Version = "1.0", ModelType = ModelType.NeuralNetwork,
            Description = "SoundStorm parallel masked audio generation", FeatureCount = ParameterCount, Complexity = ParameterCount };
        m.SetProperty("architecture", "conformer-maskgit");
        m.SetProperty("parallel_generation", true);
        m.SetProperty("speedup_vs_autoregressive", "100x");
        return m;
    }
}
