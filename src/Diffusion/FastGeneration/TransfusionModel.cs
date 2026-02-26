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
/// Transfusion model combining autoregressive language modeling with diffusion generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Transfusion trains a single transformer to handle both text (autoregressive) and images
/// (diffusion) within the same sequence. Text tokens are modeled autoregressively while
/// image patches use a diffusion loss, enabling native multimodal generation without
/// separate text and image models.
/// </para>
/// <para>
/// <b>For Beginners:</b> Most image generators have a separate text model and image model.
/// Transfusion combines both into one unified transformer — it reads and writes text
/// autoregressively (word by word) and generates images through diffusion (noise removal),
/// all in one model. This enables natural interleaving of text and images.
/// </para>
/// <para>
/// Reference: Zhou et al., "Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model", 2024
/// </para>
/// </remarks>
public class TransfusionModel<T> : LatentDiffusionModelBase<T>
{
    private const int LATENT_CHANNELS = 16;
    private const double DEFAULT_GUIDANCE = 5.0;

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

    public TransfusionModel(
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
                TrainTimesteps = 1000, BetaStart = 0.0001,
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
            hiddenSize: 4096,
            numLayers: 32,
            numHeads: 32,
            seed: seed);

        _vae = vae ?? new StandardVAE<T>(
            inputChannels: 3, latentChannels: LATENT_CHANNELS,
            baseChannels: 128, channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2, latentScaleFactor: 0.18215, seed: seed);
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
        var clone = new TransfusionModel<T>(conditioner: _conditioner, seed: RandomGenerator.Next());
        clone.SetParameters(GetParameters());
        return clone;
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = "Transfusion", Version = "1.0", ModelType = ModelType.NeuralNetwork,
            Description = "Unified autoregressive + diffusion transformer for native multimodal generation",
            FeatureCount = ParameterCount, Complexity = ParameterCount
        };
        m.SetProperty("architecture", "transfusion-unified-transformer");
        m.SetProperty("modality", "text+image");
        m.SetProperty("guidance_scale", DEFAULT_GUIDANCE);
        m.SetProperty("latent_channels", LATENT_CHANNELS);
        return m;
    }
}
