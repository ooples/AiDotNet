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
/// FlowEdit model for image editing via rectified flow inversion and re-generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// FlowEdit performs image editing by first inverting the input image to its noise
/// representation using rectified flow ODE inversion, then re-generating with modified
/// text conditioning. This approach preserves image structure while enabling text-guided edits.
/// </para>
/// <para>
/// <b>For Beginners:</b> FlowEdit works by "un-generating" your image back to noise,
/// then re-generating it with a different text prompt. Because it follows a precise
/// mathematical path (rectified flow), it preserves the original image structure while
/// making the changes you describe.
/// </para>
/// <para>
/// Technical specifications:
/// - Base model: FLUX.1 dev (hybrid MMDiT, 3072 hidden, 19+38 layers)
/// - Editing method: Flow ODE inversion + re-generation
/// - No explicit inversion network needed — uses ODE solver directly
/// - Supports partial inversion for structure preservation control
/// - Text encoders: CLIP ViT-L/14 + T5-XXL (inherited from FLUX)
///
/// Reference: Kulikov et al., "FlowEdit: Inversion-Free Text-Based Editing Using Pre-Trained Flow Models", 2024
/// </para>
/// </remarks>
public class FlowEditModel<T> : LatentDiffusionModelBase<T>
{
    private const int LATENT_CHANNELS = 16;
    private const int FLUX_HIDDEN_SIZE = 3072;
    private const int FLUX_JOINT_LAYERS = 19;
    private const int FLUX_SINGLE_LAYERS = 38;
    private const int FLUX_NUM_HEADS = 24;
    private const int FLUX_CONTEXT_DIM = 4096;
    private const double DEFAULT_GUIDANCE = 3.5;

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

    public FlowEditModel(
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
        var clone = new FlowEditModel<T>(conditioner: _conditioner, seed: RandomGenerator.Next());
        clone.SetParameters(GetParameters());
        return clone;
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = "FlowEdit", Version = "1.0", ModelType = ModelType.NeuralNetwork,
            Description = "Inversion-free text-based editing via rectified flow models",
            FeatureCount = ParameterCount, Complexity = ParameterCount
        };
        m.SetProperty("architecture", "flux-flow-inversion-editing");
        m.SetProperty("hidden_size", FLUX_HIDDEN_SIZE);
        m.SetProperty("joint_layers", FLUX_JOINT_LAYERS);
        m.SetProperty("single_layers", FLUX_SINGLE_LAYERS);
        m.SetProperty("num_heads", FLUX_NUM_HEADS);
        m.SetProperty("context_dim", FLUX_CONTEXT_DIM);
        m.SetProperty("text_encoder_1", "CLIP ViT-L/14");
        m.SetProperty("text_encoder_2", "T5-XXL");
        m.SetProperty("latent_channels", LATENT_CHANNELS);
        m.SetProperty("default_guidance_scale", DEFAULT_GUIDANCE);
        m.SetProperty("editing_method", "flow-ode-inversion");
        m.SetProperty("training", "rectified-flow");
        return m;
    }
}
