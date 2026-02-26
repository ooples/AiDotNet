using System.Diagnostics.CodeAnalysis;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Diffusion.Schedulers;

namespace AiDotNet.Diffusion.FastGeneration;

/// <summary>
/// Masked Autoregressive (MAR) model for image generation via masked token prediction.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// MAR generates images by progressively predicting masked tokens in a discrete latent
/// space. Unlike standard diffusion which adds/removes continuous noise, MAR works with
/// discrete tokens and uses a masking strategy that enables flexible generation ordering
/// and variable-speed generation.
/// </para>
/// <para>
/// <b>For Beginners:</b> MAR treats image generation like filling in a puzzle. It starts
/// with all pieces hidden (masked) and progressively reveals them, predicting what each
/// piece looks like based on the already-revealed pieces. This approach bridges
/// autoregressive models (like GPT) and diffusion models.
/// </para>
/// <para>
/// Reference: Li et al., "Autoregressive Image Generation without Vector Quantization", NeurIPS 2024
/// </para>
/// </remarks>
public class MARModel<T> : LatentDiffusionModelBase<T>
{
    private const int LATENT_CHANNELS = 16;
    private const double DEFAULT_GUIDANCE = 4.0;

    private SiTPredictor<T> _predictor;
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
    public override int ParameterCount => _predictor.ParameterCount;

    public MARModel(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        SiTPredictor<T>? predictor = null,
        StandardVAE<T>? vae = null,
        IConditioningModule<T>? conditioner = null,
        int? seed = null)
        : base(
            options ?? new DiffusionModelOptions<T>
            {
                TrainTimesteps = 256, BetaStart = 0.0001,
                BetaEnd = 0.02, BetaSchedule = BetaSchedule.Linear
            },
            scheduler ?? new DDIMScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion()),
            architecture)
    {
        _conditioner = conditioner;
        InitializeLayers(predictor, vae, seed);
        SetGuidanceScale(DEFAULT_GUIDANCE);
    }

    [MemberNotNull(nameof(_predictor), nameof(_vae))]
    private void InitializeLayers(SiTPredictor<T>? predictor, StandardVAE<T>? vae, int? seed)
    {
        _predictor = predictor ?? new SiTPredictor<T>(
            inputChannels: LATENT_CHANNELS,
            hiddenSize: 1024,
            numLayers: 24,
            numHeads: 16,
            seed: seed);

        _vae = vae ?? new StandardVAE<T>(
            inputChannels: 3, latentChannels: LATENT_CHANNELS,
            baseChannels: 128, channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2, latentScaleFactor: 0.2325, seed: seed);
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters() => _predictor.GetParameters();
    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters) => _predictor.SetParameters(parameters);
    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();

    /// <inheritdoc />
    public override IDiffusionModel<T> Clone()
    {
        var clone = new MARModel<T>(conditioner: _conditioner, seed: RandomGenerator.Next());
        clone.SetParameters(GetParameters());
        return clone;
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = "Masked Autoregressive (MAR)", Version = "1.0", ModelType = ModelType.NeuralNetwork,
            Description = "Image generation via continuous-valued masked token prediction without VQ",
            FeatureCount = ParameterCount, Complexity = ParameterCount
        };
        m.SetProperty("architecture", "masked-autoregressive-dit");
        m.SetProperty("generation_method", "progressive-unmasking");
        m.SetProperty("guidance_scale", DEFAULT_GUIDANCE);
        m.SetProperty("latent_channels", LATENT_CHANNELS);
        return m;
    }
}
