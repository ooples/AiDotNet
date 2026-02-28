using System.Diagnostics.CodeAnalysis;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Diffusion.Schedulers;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.Diffusion.ImageEditing;

/// <summary>
/// FLUX Fill model for mask-guided inpainting using rectified flow transformers.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Adapts the FLUX rectified flow architecture for inpainting by conditioning on both
/// the masked image latent and the binary mask. Uses the FLUX hybrid MMDiT with
/// 19 double-stream + 38 single-stream transformer blocks for high-quality 16-channel
/// latent inpainting with dual text encoder conditioning (CLIP + T5).
/// </para>
/// <para>
/// <b>For Beginners:</b> This is inpainting built on the FLUX architecture, which is
/// known for excellent image quality. It fills in masked regions using FLUX's advanced
/// transformer-based generation, producing results with 16-channel latent precision
/// and superior text understanding from dual encoders.
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: FLUX hybrid MMDiT (19 joint + 38 single blocks)
/// - Hidden size: 3072, 24 attention heads
/// - Text encoders: CLIP ViT-L/14 (768-dim) + T5-XXL (4096-dim)
/// - Latent space: 16 channels, patch size 2
/// - Training: Rectified flow matching
/// - Resolution: Up to 2048x2048 (aspect-ratio aware)
///
/// Reference: Black Forest Labs, "FLUX.1 Fill", 2024
/// </para>
/// </remarks>
public class FluxInpaintingModel<T> : LatentDiffusionModelBase<T>
{
    private const int LATENT_CHANNELS = 16;
    private const int FLUX_HIDDEN_SIZE = 3072;
    private const int FLUX_JOINT_LAYERS = 19;
    private const int FLUX_SINGLE_LAYERS = 38;
    private const int FLUX_NUM_HEADS = 24;
    private const int FLUX_CONTEXT_DIM = 4096;
    private const double DEFAULT_GUIDANCE = 3.5;
    private const int DEFAULT_STEPS = 50;

    private FluxDoubleStreamPredictor<T> _predictor;
    private StandardVAE<T> _vae;
    private readonly IConditioningModule<T>? _conditioner;

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _predictor;
    /// <inheritdoc />
    public override IVAEModel<T> VAE => _vae;
    /// <inheritdoc />
    public override IConditioningModule<T>? Conditioner => _conditioner;
    /// <inheritdoc />
    public override int LatentChannels => LATENT_CHANNELS;
    /// <inheritdoc />
    public override int ParameterCount => _predictor.ParameterCount + _vae.ParameterCount;

    public FluxInpaintingModel(
        NeuralNetworkArchitecture<T>? architecture = null, DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null, FluxDoubleStreamPredictor<T>? predictor = null,
        StandardVAE<T>? vae = null, IConditioningModule<T>? conditioner = null, int? seed = null)
        : base(options ?? new DiffusionModelOptions<T> { TrainTimesteps = 1000, BetaStart = 0.0001, BetaEnd = 1.0, BetaSchedule = BetaSchedule.Linear },
            scheduler ?? new FlowMatchingScheduler<T>(SchedulerConfig<T>.CreateRectifiedFlow()), architecture)
    {
        _conditioner = conditioner;
        InitializeLayers(predictor, vae, seed);
        SetGuidanceScale(DEFAULT_GUIDANCE);
    }

    [MemberNotNull(nameof(_predictor), nameof(_vae))]
    private void InitializeLayers(FluxDoubleStreamPredictor<T>? predictor, StandardVAE<T>? vae, int? seed)
    {
        _predictor = predictor ?? new FluxDoubleStreamPredictor<T>(seed: seed);
        _vae = vae ?? new StandardVAE<T>(inputChannels: 3, latentChannels: LATENT_CHANNELS,
            baseChannels: 128, channelMultipliers: [1, 2, 4, 4], numResBlocksPerLevel: 2,
            latentScaleFactor: 1.5305, seed: seed);
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var pp = _predictor.GetParameters();
        var vp = _vae.GetParameters();
        var combined = new Vector<T>(pp.Length + vp.Length);
        for (int i = 0; i < pp.Length; i++) combined[i] = pp[i];
        for (int i = 0; i < vp.Length; i++) combined[pp.Length + i] = vp[i];
        return combined;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        var pc = _predictor.ParameterCount;
        var vc = _vae.ParameterCount;
        if (parameters.Length != pc + vc)
            throw new ArgumentException($"Expected {pc + vc} parameters, got {parameters.Length}.", nameof(parameters));
        var pp = new Vector<T>(pc);
        var vp = new Vector<T>(vc);
        for (int i = 0; i < pc; i++) pp[i] = parameters[i];
        for (int i = 0; i < vc; i++) vp[i] = parameters[pc + i];
        _predictor.SetParameters(pp);
        _vae.SetParameters(vp);
    }

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();

    /// <inheritdoc />
    public override IDiffusionModel<T> Clone()
    {
        var clone = new FluxInpaintingModel<T>(conditioner: _conditioner, seed: RandomGenerator.Next());
        clone.SetParameters(GetParameters());
        return clone;
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = "FLUX Fill", Version = "1.0", ModelType = ModelType.NeuralNetwork,
            Description = "FLUX rectified flow transformer for high-quality mask-guided inpainting",
            FeatureCount = ParameterCount, Complexity = ParameterCount
        };
        m.SetProperty("architecture", "flux-double-stream-inpainting");
        m.SetProperty("hidden_size", FLUX_HIDDEN_SIZE);
        m.SetProperty("joint_layers", FLUX_JOINT_LAYERS);
        m.SetProperty("single_layers", FLUX_SINGLE_LAYERS);
        m.SetProperty("num_heads", FLUX_NUM_HEADS);
        m.SetProperty("context_dim", FLUX_CONTEXT_DIM);
        m.SetProperty("text_encoder_1", "CLIP ViT-L/14");
        m.SetProperty("text_encoder_2", "T5-XXL");
        m.SetProperty("latent_channels", LATENT_CHANNELS);
        m.SetProperty("default_guidance_scale", DEFAULT_GUIDANCE);
        m.SetProperty("default_steps", DEFAULT_STEPS);
        m.SetProperty("training", "rectified-flow");
        return m;
    }
}
