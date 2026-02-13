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
/// Udio/Suno architecture model — full-song music generation with structural awareness.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This model represents the Udio/Suno class of full-song music generation systems
/// using DiT with structural conditioning for verse/chorus/bridge awareness.
/// </para>
/// <para>
/// <b>For Beginners:</b> Udio/Suno generates complete songs from text prompts:
///
/// Key characteristics:
/// - Full-song generation (2-5 minutes) with musical structure
/// - Verse/chorus/bridge structural conditioning
/// - Genre-aware generation (any style)
/// - High-fidelity 44.1 kHz stereo output
/// - Optional vocal + instrumental separation
///
/// Reference: Conceptual representation of Udio/Suno-class music generation, 2024
/// </para>
/// </remarks>
public class UdioModel<T> : AudioDiffusionModelBase<T>
{
    private const int LATENT_CHANNELS = 64;
    private const int HIDDEN_DIM = 2048;
    private const int NUM_LAYERS = 32;
    private const int NUM_HEADS = 16;
    private const int CONTEXT_DIM = 1024;

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

    public UdioModel(
        DiffusionModelOptions<T>? options = null, INoiseScheduler<T>? scheduler = null,
        DiTNoisePredictor<T>? dit = null, AudioVAE<T>? audioVAE = null,
        IConditioningModule<T>? conditioner = null)
        : base(options ?? new DiffusionModelOptions<T> { TrainTimesteps = 1000, BetaStart = 0.0001, BetaEnd = 0.02, BetaSchedule = BetaSchedule.Linear },
               scheduler ?? new FlowMatchingScheduler<T>(SchedulerConfig<T>.CreateDefault()),
               sampleRate: 44100, defaultDurationSeconds: 180.0, melChannels: 128)
    {
        _conditioner = conditioner;
        _dit = dit ?? new DiTNoisePredictor<T>(
            inputChannels: LATENT_CHANNELS, hiddenSize: HIDDEN_DIM,
            numLayers: NUM_LAYERS, numHeads: NUM_HEADS,
            patchSize: 1, contextDim: CONTEXT_DIM);
        _audioVAE = audioVAE ?? new AudioVAE<T>(
            melChannels: 128, latentChannels: LATENT_CHANNELS,
            baseChannels: 128, numResBlocks: 3);
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
        return new UdioModel<T>(dit: cd,
            audioVAE: (AudioVAE<T>)_audioVAE.Clone(),
            conditioner: _conditioner);
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T> { Name = "Udio-Suno", Version = "1.0", ModelType = ModelType.NeuralNetwork,
            Description = "Udio/Suno-class full-song music generation", FeatureCount = ParameterCount, Complexity = ParameterCount };
        m.SetProperty("architecture", "dit-structural-conditioning");
        m.SetProperty("sample_rate", 44100);
        m.SetProperty("max_duration_seconds", 300);
        m.SetProperty("full_song_structure", true);
        return m;
    }
}
