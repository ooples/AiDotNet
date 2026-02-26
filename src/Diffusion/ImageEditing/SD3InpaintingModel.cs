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
/// Stable Diffusion 3 inpainting model using MMDiT architecture for mask-guided generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Adapts the SD3 MMDiT (Multimodal Diffusion Transformer) architecture for inpainting.
/// Uses 16-channel latent space with the MMDiT-X noise predictor for high-quality
/// mask-conditioned generation with improved text understanding via triple text encoders.
/// </para>
/// <para>
/// <b>For Beginners:</b> This is Stable Diffusion 3's inpainting mode. SD3 uses a modern
/// transformer architecture (instead of U-Net) and three text encoders for better prompt
/// understanding. It produces high-quality inpainting results with excellent text following.
/// </para>
/// <para>
/// Technical specifications (SD3 Medium):
/// - Architecture: MMDiT-X (Multimodal Diffusion Transformer with cross-attention)
/// - Hidden size: 1536, 24 joint attention layers, 24 heads
/// - Text encoders: CLIP-L (768-dim) + CLIP-G (1280-dim) + T5-XXL (4096-dim)
/// - Latent space: 16 channels with rectified flow training
/// - Prediction type: v-prediction (velocity)
/// - Resolution: Up to 1024x1024 (aspect-ratio buckets)
///
/// Reference: Esser et al., "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis", ICML 2024
/// </para>
/// </remarks>
public class SD3InpaintingModel<T> : LatentDiffusionModelBase<T>
{
    private const int LATENT_CHANNELS = 16;
    private const int SD3_HIDDEN_SIZE = 1536;
    private const int SD3_NUM_LAYERS = 24;
    private const int SD3_NUM_HEADS = 24;
    private const int SD3_CONTEXT_DIM = 4096;
    private const double DEFAULT_GUIDANCE = 7.0;
    private const int DEFAULT_STEPS = 28;

    private MMDiTXNoisePredictor<T> _predictor;
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

    public SD3InpaintingModel(
        NeuralNetworkArchitecture<T>? architecture = null, DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null, MMDiTXNoisePredictor<T>? predictor = null,
        StandardVAE<T>? vae = null, IConditioningModule<T>? conditioner = null, int? seed = null)
        : base(options ?? new DiffusionModelOptions<T> { TrainTimesteps = 1000, BetaStart = 0.0001, BetaEnd = 0.02, BetaSchedule = BetaSchedule.Linear },
            scheduler ?? new FlowMatchingScheduler<T>(SchedulerConfig<T>.CreateRectifiedFlow()), architecture)
    {
        _conditioner = conditioner;
        InitializeLayers(predictor, vae, seed);
        SetGuidanceScale(DEFAULT_GUIDANCE);
    }

    [MemberNotNull(nameof(_predictor), nameof(_vae))]
    private void InitializeLayers(MMDiTXNoisePredictor<T>? predictor, StandardVAE<T>? vae, int? seed)
    {
        _predictor = predictor ?? new MMDiTXNoisePredictor<T>(seed: seed);
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
        var clone = new SD3InpaintingModel<T>(conditioner: _conditioner, seed: RandomGenerator.Next());
        clone.SetParameters(GetParameters());
        return clone;
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = "SD3 Inpainting", Version = "3.0", ModelType = ModelType.NeuralNetwork,
            Description = "SD3 MMDiT-based inpainting with triple text encoders and rectified flow",
            FeatureCount = ParameterCount, Complexity = ParameterCount
        };
        m.SetProperty("architecture", "mmdit-x-inpainting");
        m.SetProperty("hidden_size", SD3_HIDDEN_SIZE);
        m.SetProperty("num_layers", SD3_NUM_LAYERS);
        m.SetProperty("num_heads", SD3_NUM_HEADS);
        m.SetProperty("context_dim", SD3_CONTEXT_DIM);
        m.SetProperty("text_encoder_1", "CLIP ViT-L/14");
        m.SetProperty("text_encoder_2", "CLIP ViT-G/14");
        m.SetProperty("text_encoder_3", "T5-XXL");
        m.SetProperty("latent_channels", LATENT_CHANNELS);
        m.SetProperty("default_guidance_scale", DEFAULT_GUIDANCE);
        m.SetProperty("default_steps", DEFAULT_STEPS);
        m.SetProperty("training", "rectified-flow-v-prediction");
        return m;
    }
}
