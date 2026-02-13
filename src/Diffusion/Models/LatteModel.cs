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
/// Latte model — Latent Diffusion Transformer for video generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Latte explores DiT variants for video, using factorized spatial-temporal
/// attention within transformer blocks for efficient video generation.
/// </para>
/// <para>
/// <b>For Beginners:</b> Latte applies DiT (image transformer) concepts to video:
///
/// Key characteristics:
/// - DiT backbone with spatial-temporal decomposition
/// - 4 attention variants: joint, spatial-first, temporal-first, decomposed
/// - Efficient O(n) attention via factorization
/// - T5-XXL text encoder
///
/// Reference: Ma et al., "Latte: Latent Diffusion Transformer for Video Generation", 2024
/// </para>
/// </remarks>
public class LatteModel<T> : VideoDiffusionModelBase<T>
{
    private const int LATENT_CHANNELS = 4;
    private const int HIDDEN_DIM = 1152;
    private const int NUM_LAYERS = 28;
    private const int NUM_HEADS = 16;
    private const int CONTEXT_DIM = 4096;

    private readonly DiTNoisePredictor<T> _dit;
    private readonly StandardVAE<T> _vae;
    private readonly IConditioningModule<T>? _conditioner;

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _dit;
    /// <inheritdoc />
    public override IVAEModel<T> VAE => _vae;
    /// <inheritdoc />
    public override IConditioningModule<T>? Conditioner => _conditioner;
    /// <inheritdoc />
    public override int LatentChannels => LATENT_CHANNELS;
    /// <inheritdoc />
    public override bool SupportsImageToVideo => false;
    /// <inheritdoc />
    public override bool SupportsTextToVideo => true;
    /// <inheritdoc />
    public override bool SupportsVideoToVideo => false;
    /// <inheritdoc />
    public override int ParameterCount => _dit.ParameterCount + _vae.ParameterCount;

    public LatteModel(
        DiffusionModelOptions<T>? options = null, INoiseScheduler<T>? scheduler = null,
        DiTNoisePredictor<T>? dit = null, StandardVAE<T>? vae = null,
        IConditioningModule<T>? conditioner = null,
        int defaultNumFrames = 16, int defaultFPS = 8)
        : base(options ?? new DiffusionModelOptions<T> { TrainTimesteps = 1000, BetaStart = 0.0001, BetaEnd = 0.02, BetaSchedule = BetaSchedule.Linear },
               scheduler ?? new DDIMScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion()),
               defaultNumFrames, defaultFPS)
    {
        _conditioner = conditioner;
        _dit = dit ?? new DiTNoisePredictor<T>(
            inputChannels: LATENT_CHANNELS, hiddenSize: HIDDEN_DIM,
            numLayers: NUM_LAYERS, numHeads: NUM_HEADS,
            patchSize: 2, contextDim: CONTEXT_DIM);
        _vae = vae ?? new StandardVAE<T>(
            inputChannels: 3, latentChannels: LATENT_CHANNELS,
            baseChannels: 128, channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2, latentScaleFactor: 0.18215);
    }

    /// <inheritdoc />
    protected override Tensor<T> PredictVideoNoise(Tensor<T> latents, int timestep,
        Tensor<T> imageEmbedding, Tensor<T> motionEmbedding)
        => _dit.PredictNoise(latents, timestep, imageEmbedding);

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var dp = _dit.GetParameters(); var vp = _vae.GetParameters();
        var c = new Vector<T>(dp.Length + vp.Length);
        for (int i = 0; i < dp.Length; i++) c[i] = dp[i];
        for (int i = 0; i < vp.Length; i++) c[dp.Length + i] = vp[i];
        return c;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        int dc = _dit.ParameterCount, vc = _vae.ParameterCount;
        var dp = new Vector<T>(dc); var vp = new Vector<T>(vc);
        for (int i = 0; i < dc; i++) dp[i] = parameters[i];
        for (int i = 0; i < vc; i++) vp[i] = parameters[dc + i];
        _dit.SetParameters(dp); _vae.SetParameters(vp);
    }

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();
    /// <inheritdoc />
    public override IDiffusionModel<T> Clone()
    {
        var cd = new DiTNoisePredictor<T>(
            inputChannels: LATENT_CHANNELS, hiddenSize: HIDDEN_DIM,
            numLayers: NUM_LAYERS, numHeads: NUM_HEADS,
            patchSize: 2, contextDim: CONTEXT_DIM);
        cd.SetParameters(_dit.GetParameters());
        return new LatteModel<T>(dit: cd,
            vae: new StandardVAE<T>(inputChannels: 3, latentChannels: LATENT_CHANNELS,
                baseChannels: 128, channelMultipliers: [1, 2, 4, 4],
                numResBlocksPerLevel: 2, latentScaleFactor: 0.18215),
            conditioner: _conditioner, defaultNumFrames: DefaultNumFrames, defaultFPS: DefaultFPS);
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T> { Name = "Latte", Version = "1.0", ModelType = ModelType.NeuralNetwork,
            Description = "Latte latent diffusion transformer for video generation", FeatureCount = ParameterCount, Complexity = ParameterCount };
        m.SetProperty("architecture", "dit-factorized-st-attention");
        m.SetProperty("hidden_dim", HIDDEN_DIM);
        m.SetProperty("num_layers", NUM_LAYERS);
        return m;
    }
}
