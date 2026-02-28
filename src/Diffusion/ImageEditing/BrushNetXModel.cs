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
/// BrushNet-X model extending BrushNet to SDXL architecture for high-resolution inpainting.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// BrushNet-X adapts the dual-branch BrushNet inpainting approach to the SDXL architecture,
/// enabling high-resolution (1024x1024) inpainting with improved coherence and detail.
/// Uses SDXL's dual text encoders (CLIP-L + CLIP-G) for enhanced prompt understanding.
/// </para>
/// <para>
/// <b>For Beginners:</b> This is a higher-resolution version of BrushNet built on SDXL.
/// It generates larger, more detailed inpainting results at 1024x1024 resolution compared
/// to BrushNet's 512x512, with better text understanding from dual encoders.
/// </para>
/// <para>
/// Technical specifications:
/// - Architecture: Dual-branch U-Net (SDXL backbone)
/// - Text encoders: CLIP ViT-L/14 (768-dim) + OpenCLIP ViT-G/14 (1280-dim) = 2048 context
/// - Input channels: 9 (4 latent + 4 masked image latent + 1 mask)
/// - Base channels: 320, multipliers [1, 2, 4]
/// - Resolution: 1024x1024 native
/// </para>
/// </remarks>
public class BrushNetXModel<T> : LatentDiffusionModelBase<T>
{
    private const int LATENT_CHANNELS = 4;
    private const int SDXL_CONTEXT_DIM = 2048;
    private const double DEFAULT_GUIDANCE = 7.5;

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

    public BrushNetXModel(
        NeuralNetworkArchitecture<T>? architecture = null, DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null, UNetNoisePredictor<T>? predictor = null,
        StandardVAE<T>? vae = null, IConditioningModule<T>? conditioner = null, int? seed = null)
        : base(options ?? new DiffusionModelOptions<T> { TrainTimesteps = 1000, BetaStart = 0.00085, BetaEnd = 0.012, BetaSchedule = BetaSchedule.ScaledLinear },
            scheduler ?? new EulerDiscreteScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion()), architecture)
    {
        _conditioner = conditioner;
        InitializeLayers(predictor, vae, seed);
        SetGuidanceScale(DEFAULT_GUIDANCE);
    }

    [MemberNotNull(nameof(_predictor), nameof(_vae))]
    private void InitializeLayers(UNetNoisePredictor<T>? predictor, StandardVAE<T>? vae, int? seed)
    {
        _predictor = predictor ?? new UNetNoisePredictor<T>(
            architecture: Architecture, inputChannels: 9, outputChannels: LATENT_CHANNELS,
            baseChannels: 320, channelMultipliers: [1, 2, 4],
            numResBlocks: 2, attentionResolutions: [4, 2], contextDim: SDXL_CONTEXT_DIM, seed: seed);
        _vae = vae ?? new StandardVAE<T>(inputChannels: 3, latentChannels: LATENT_CHANNELS,
            baseChannels: 128, channelMultipliers: [1, 2, 4, 4], numResBlocksPerLevel: 2, seed: seed);
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
        var clone = new BrushNetXModel<T>(conditioner: _conditioner, seed: RandomGenerator.Next());
        clone.SetParameters(GetParameters());
        return clone;
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = "BrushNet-X", Version = "1.0", ModelType = ModelType.NeuralNetwork,
            Description = "SDXL dual-branch inpainting for high-resolution 1024x1024 generation",
            FeatureCount = ParameterCount, Complexity = ParameterCount
        };
        m.SetProperty("architecture", "dual-branch-sdxl-inpainting");
        m.SetProperty("base_model", "Stable Diffusion XL");
        m.SetProperty("text_encoder_1", "CLIP ViT-L/14");
        m.SetProperty("text_encoder_2", "OpenCLIP ViT-G/14");
        m.SetProperty("context_dim", SDXL_CONTEXT_DIM);
        m.SetProperty("latent_channels", LATENT_CHANNELS);
        m.SetProperty("default_guidance_scale", DEFAULT_GUIDANCE);
        m.SetProperty("default_resolution", 1024);
        m.SetProperty("conditioning", "dual-branch-per-layer-injection");
        return m;
    }
}
