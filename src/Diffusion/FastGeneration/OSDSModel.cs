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

namespace AiDotNet.Diffusion.FastGeneration;

/// <summary>
/// One-Step Diffusion via Shortcut (OSDS) model for single-step high-quality generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// OSDS learns a shortcut function that directly maps noise to clean data in a single
/// forward pass. Unlike distillation approaches, it trains a dedicated shortcut path
/// that bypasses the iterative denoising process entirely. The shortcut is learned
/// alongside the standard diffusion trajectory, enabling both one-step and multi-step
/// generation from the same model.
/// </para>
/// <para>
/// <b>For Beginners:</b> Most diffusion models need many steps to turn noise into an image.
/// OSDS learns a "shortcut" — a direct path from noise to image in just one step. Think of
/// it like learning both the scenic route (multi-step) and a highway (one-step) to the same
/// destination. You can choose speed or quality depending on your needs.
/// </para>
/// <para>
/// Reference: Frans et al., "One Step Diffusion via Shortcut Models", 2024
/// </para>
/// </remarks>
public class OSDSModel<T> : LatentDiffusionModelBase<T>
{
    private const int LATENT_CHANNELS = 4;
    private const double DEFAULT_GUIDANCE = 0.0;

    private UNetNoisePredictor<T> _predictor;
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

    /// <summary>
    /// Initializes a new OSDS model with optional configuration.
    /// </summary>
    /// <param name="architecture">Neural network architecture configuration.</param>
    /// <param name="options">Diffusion model options.</param>
    /// <param name="scheduler">Noise scheduler.</param>
    /// <param name="predictor">UNet noise predictor with shortcut path.</param>
    /// <param name="vae">VAE for latent encoding/decoding.</param>
    /// <param name="conditioner">Text conditioning module.</param>
    /// <param name="seed">Random seed for reproducibility.</param>
    public OSDSModel(
        NeuralNetworkArchitecture<T>? architecture = null, DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null, UNetNoisePredictor<T>? predictor = null,
        StandardVAE<T>? vae = null, IConditioningModule<T>? conditioner = null, int? seed = null)
        : base(options ?? new DiffusionModelOptions<T> { TrainTimesteps = 1000, BetaStart = 0.00085, BetaEnd = 0.012, BetaSchedule = BetaSchedule.ScaledLinear },
            scheduler ?? new DDIMScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion()), architecture)
    {
        _conditioner = conditioner;
        InitializeLayers(predictor, vae, seed);
    }

    [MemberNotNull(nameof(_predictor), nameof(_vae))]
    private void InitializeLayers(UNetNoisePredictor<T>? predictor, StandardVAE<T>? vae, int? seed)
    {
        _predictor = predictor ?? new UNetNoisePredictor<T>(
            architecture: Architecture, inputChannels: 4, outputChannels: LATENT_CHANNELS,
            baseChannels: 320, channelMultipliers: new[] { 1, 2, 4, 4 },
            numResBlocks: 2, attentionResolutions: new[] { 4, 2, 1 }, contextDim: 768, seed: seed);
        _vae = vae ?? new StandardVAE<T>(inputChannels: 3, latentChannels: LATENT_CHANNELS,
            baseChannels: 128, channelMultipliers: new[] { 1, 2, 4, 4 }, numResBlocksPerLevel: 2, seed: seed);
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
        var clone = new OSDSModel<T>(conditioner: _conditioner, seed: RandomGenerator.Next());
        clone.SetParameters(GetParameters());
        return clone;
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T> { Name = "OSDS", Version = "1.0", ModelType = ModelType.NeuralNetwork,
            Description = "One-step diffusion via shortcut models for instant generation",
            FeatureCount = ParameterCount, Complexity = ParameterCount };
        m.SetProperty("architecture", "shortcut-unet");
        m.SetProperty("latent_channels", LATENT_CHANNELS);
        m.SetProperty("default_guidance_scale", DEFAULT_GUIDANCE);
        m.SetProperty("supports_one_step", true);
        m.SetProperty("supports_multi_step", true);
        return m;
    }
}
