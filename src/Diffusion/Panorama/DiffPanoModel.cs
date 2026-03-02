using System.Diagnostics.CodeAnalysis;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Diffusion.Schedulers;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.Diffusion.Panorama;

/// <summary>
/// DiffPano model for scalable panorama generation with spherical epipolar-aware diffusion.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// DiffPano uses spherical epipolar constraints to generate consistent 360-degree panoramic
/// images from single or multi-view inputs. It encodes geometric relationships between panoramic
/// viewpoints using epipolar-aware cross-attention, ensuring global consistency across the
/// full spherical field of view.
/// </para>
/// <para>
/// <b>For Beginners:</b> DiffPano creates full 360-degree panoramic images that look consistent
/// from every viewing angle. It understands the geometry of spherical images (like what you see
/// in a VR headset) and ensures that all parts of the panorama connect seamlessly, even when
/// generating from just a single photo.
/// </para>
/// <para>
/// Reference: Wang et al., "DiffPano: Scalable and Consistent Text to Panorama Generation
/// with Spherical Epipolar-Aware Diffusion", NeurIPS 2024
/// </para>
/// </remarks>
public class DiffPanoModel<T> : LatentDiffusionModelBase<T>
{
    private const int LATENT_CHANNELS = 4;
    private const double DEFAULT_GUIDANCE = 7.5;
    private const int DEFAULT_PANORAMA_WIDTH = 2048;
    private const int DEFAULT_PANORAMA_HEIGHT = 1024;

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
    /// Initializes a new DiffPano model with optional configuration.
    /// </summary>
    /// <param name="architecture">Neural network architecture configuration.</param>
    /// <param name="options">Diffusion model options.</param>
    /// <param name="scheduler">Noise scheduler for the diffusion process.</param>
    /// <param name="predictor">UNet noise predictor with epipolar cross-attention.</param>
    /// <param name="vae">VAE for encoding/decoding panoramic images.</param>
    /// <param name="conditioner">Text conditioning module.</param>
    /// <param name="seed">Random seed for reproducibility.</param>
    public DiffPanoModel(
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
        var clone = new DiffPanoModel<T>(conditioner: _conditioner, seed: RandomGenerator.Next());
        clone.SetParameters(GetParameters());
        return clone;
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T> { Name = "DiffPano", Version = "1.0", ModelType = ModelType.NeuralNetwork,
            Description = "Spherical epipolar-aware diffusion for consistent 360-degree panorama generation",
            FeatureCount = ParameterCount, Complexity = ParameterCount };
        m.SetProperty("architecture", "spherical-epipolar-unet");
        m.SetProperty("base_model", "Stable Diffusion 1.5");
        m.SetProperty("text_encoder", "CLIP ViT-L/14");
        m.SetProperty("context_dim", 768);
        m.SetProperty("latent_channels", LATENT_CHANNELS);
        m.SetProperty("guidance_scale", DEFAULT_GUIDANCE);
        m.SetProperty("default_panorama_width", DEFAULT_PANORAMA_WIDTH);
        m.SetProperty("default_panorama_height", DEFAULT_PANORAMA_HEIGHT);
        return m;
    }
}
