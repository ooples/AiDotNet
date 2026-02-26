using System.Diagnostics.CodeAnalysis;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Diffusion.Schedulers;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.Diffusion.ImageEditing;

/// <summary>
/// CycleGAN-Turbo model combining CycleGAN unpaired translation with diffusion backbone.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Combines the unpaired image-to-image translation paradigm of CycleGAN with a
/// pre-trained diffusion model backbone. Uses cycle consistency loss on a diffusion
/// U-Net to enable single-step unpaired domain translation with high fidelity.
/// </para>
/// <para>
/// <b>For Beginners:</b> CycleGAN-Turbo translates images between two styles (like
/// photos to paintings, summer to winter) using a single fast step. Unlike traditional
/// CycleGAN, it leverages a pre-trained diffusion model for much higher quality results.
/// </para>
/// <para>
/// Reference: Parmar et al., "One-Step Image Translation with Text-to-Image Models", 2024
/// </para>
/// </remarks>
public class CycleGANTurboModel<T> : LatentDiffusionModelBase<T>
{
    private const int LATENT_CHANNELS = 4;
    private const double DEFAULT_GUIDANCE = 1.0;

    private UNetNoisePredictor<T> _predictor;
    private StandardVAE<T> _vae;
    private readonly IConditioningModule<T>? _conditioner;

    public override INoisePredictor<T> NoisePredictor => _predictor;
    public override IVAEModel<T> VAE => _vae;
    public override IConditioningModule<T>? Conditioner => _conditioner;
    public override int LatentChannels => LATENT_CHANNELS;
    public override int ParameterCount => _predictor.ParameterCount;

    public CycleGANTurboModel(
        NeuralNetworkArchitecture<T>? architecture = null, DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null, UNetNoisePredictor<T>? predictor = null,
        StandardVAE<T>? vae = null, IConditioningModule<T>? conditioner = null, int? seed = null)
        : base(options ?? new DiffusionModelOptions<T> { TrainTimesteps = 1000, BetaStart = 0.00085, BetaEnd = 0.012, BetaSchedule = BetaSchedule.ScaledLinear },
            scheduler ?? new EulerDiscreteScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion()), architecture)
    {
        _conditioner = conditioner;
        InitializeLayers(predictor, vae, seed);
    }

    [MemberNotNull(nameof(_predictor), nameof(_vae))]
    private void InitializeLayers(UNetNoisePredictor<T>? predictor, StandardVAE<T>? vae, int? seed)
    {
        _predictor = predictor ?? new UNetNoisePredictor<T>(
            architecture: Architecture, inputChannels: 8, outputChannels: LATENT_CHANNELS,
            baseChannels: 320, channelMultipliers: new[] { 1, 2, 4 },
            numResBlocks: 2, attentionResolutions: new[] { 4, 2 }, contextDim: 2048, seed: seed);
        _vae = vae ?? new StandardVAE<T>(inputChannels: 3, latentChannels: LATENT_CHANNELS,
            baseChannels: 128, channelMultipliers: new[] { 1, 2, 4, 4 }, numResBlocksPerLevel: 2, seed: seed);
    }

    public override Vector<T> GetParameters() => _predictor.GetParameters();
    public override void SetParameters(Vector<T> parameters) => _predictor.SetParameters(parameters);
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();

    public override IDiffusionModel<T> Clone()
    {
        var clone = new CycleGANTurboModel<T>(conditioner: _conditioner, seed: RandomGenerator.Next());
        clone.SetParameters(GetParameters());
        return clone;
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T> { Name = "CycleGAN-Turbo", Version = "1.0", ModelType = ModelType.NeuralNetwork,
            Description = "Single-step unpaired image translation with diffusion backbone",
            FeatureCount = ParameterCount, Complexity = ParameterCount };
        m.SetProperty("architecture", "cyclegan-diffusion-turbo");
        m.SetProperty("latent_channels", LATENT_CHANNELS);
        m.SetProperty("default_guidance_scale", DEFAULT_GUIDANCE);
        return m;
    }
}
