using System.Diagnostics.CodeAnalysis;
using AiDotNet.Attributes;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Diffusion.Schedulers;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.Diffusion.Control;

/// <summary>
/// ControlNet adapted for the FLUX.1 architecture with flow matching.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Adapts ControlNet conditioning for FLUX.1's flow-matching diffusion framework
/// with double-stream transformer blocks. Uses 16-channel latent space and
/// rectified flow scheduling native to FLUX.
/// </para>
/// <para>
/// <b>For Beginners:</b> This brings ControlNet's "follow my reference image" capability
/// to FLUX models. FLUX uses a different internal architecture than Stable Diffusion,
/// so this specialized version ensures control signals work correctly with FLUX.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a ControlNet for FLUX.1 architecture
/// var options = new LatentDiffusionOptions&lt;float&gt;
/// {
///     LatentChannels = 16,
///     Height = 1024,
///     Width = 1024,
///     NumInferenceSteps = 28
/// };
/// var model = new ControlNetFluxModel&lt;float&gt;(options, ControlType.Canny);
///
/// // Generate with edge-guided control
/// var edgeCondition = Tensor&lt;float&gt;.Random(new[] { 1, 1, 1024, 1024 });
/// var result = model.Predict(edgeCondition);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
    [ResearchPaper("Adding Conditional Control to Text-to-Image Diffusion Models", "https://arxiv.org/abs/2302.05543")]
public class ControlNetFluxModel<T> : LatentDiffusionModelBase<T>
{
    private const int FLUX_LATENT_CHANNELS = 16;
    private const int FLUX_CONTEXT_DIM = 4096;
    private const double DEFAULT_GUIDANCE = 3.5;

    private FluxDoubleStreamPredictor<T> _predictor;
    private StandardVAE<T> _vae;
    private ControlNetEncoder<T> _controlEncoder;
    private readonly IConditioningModule<T>? _conditioner;
    private readonly ControlType _controlType;

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _predictor;
    /// <inheritdoc />
    public override IVAEModel<T> VAE => _vae;
    /// <inheritdoc />
    public override IConditioningModule<T>? Conditioner => _conditioner;
    /// <inheritdoc />
    public override int LatentChannels => FLUX_LATENT_CHANNELS;
    /// <inheritdoc />
    public override long ParameterCount =>
        _predictor.ParameterCount + _vae.ParameterCount + _controlEncoder.ParameterCount;

    /// <summary>
    /// Initializes a new ControlNet-FLUX model.
    /// </summary>
    public ControlNetFluxModel(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        FluxDoubleStreamPredictor<T>? predictor = null,
        StandardVAE<T>? vae = null,
        IConditioningModule<T>? conditioner = null,
        ControlType controlType = ControlType.Canny,
        int? seed = null)
        : base(
            options ?? new DiffusionModelOptions<T>
            {
                TrainTimesteps = 1000,
                BetaStart = 0.0001,
                BetaEnd = 0.02,
                BetaSchedule = BetaSchedule.Linear
            },
            scheduler ?? new FlowMatchingScheduler<T>(SchedulerConfig<T>.CreateRectifiedFlow()),
            architecture)
    {
        _controlType = controlType;
        _conditioner = conditioner;
        InitializeLayers(predictor, vae, seed);
        SetGuidanceScale(DEFAULT_GUIDANCE);
    }

    [MemberNotNull(nameof(_predictor), nameof(_vae), nameof(_controlEncoder))]
    private void InitializeLayers(FluxDoubleStreamPredictor<T>? predictor, StandardVAE<T>? vae, int? seed)
    {
        _predictor = predictor ?? new FluxDoubleStreamPredictor<T>(seed: seed);

        _vae = vae ?? new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: FLUX_LATENT_CHANNELS,
            baseChannels: 128,
            channelMultipliers: new[] { 1, 2, 4, 4 },
            numResBlocksPerLevel: 2,
            seed: seed);

        _controlEncoder = new ControlNetEncoder<T>(
            inputChannels: 3,
            baseChannels: 320,
            channelMultipliers: new[] { 1, 2, 4, 4 },
            seed: seed);
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        // Pre-allocate to avoid the List<T> doubling + ToArray triple-copy
        // that OOMs CI on real-scale FLUX (~12B params doubles ⇒ ~96 GB).
        int predCount = checked((int)_predictor.ParameterCount);
        int vaeCount = checked((int)_vae.ParameterCount);
        int ctrlCount = checked((int)_controlEncoder.ParameterCount);
        long expectedTotal = (long)predCount + vaeCount + ctrlCount;
        var result = new Vector<T>(checked((int)expectedTotal));
        var predParams = _predictor.GetParameters();
        for (int i = 0; i < predParams.Length; i++) result[i] = predParams[i];
        var vaeParams = _vae.GetParameters();
        for (int i = 0; i < vaeParams.Length; i++) result[predCount + i] = vaeParams[i];
        var ctrlParams = _controlEncoder.GetParameters();
        for (int i = 0; i < ctrlParams.Length; i++) result[predCount + vaeCount + i] = ctrlParams[i];
        return result;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        int predCount = checked((int)_predictor.ParameterCount);
        int vaeCount = checked((int)_vae.ParameterCount);
        int ctrlCount = checked((int)_controlEncoder.ParameterCount);
        long expectedTotal = (long)predCount + vaeCount + ctrlCount;
        if (parameters.Length != expectedTotal)
        {
            throw new ArgumentException(
                $"Expected {expectedTotal} parameters, got {parameters.Length}.",
                nameof(parameters));
        }

        int offset = 0;
        var predParams = new T[predCount];
        for (int i = 0; i < predCount; i++) predParams[i] = parameters[offset + i];
        _predictor.SetParameters(new Vector<T>(predParams));
        offset += predCount;

        var vaeParams = new T[vaeCount];
        for (int i = 0; i < vaeCount; i++) vaeParams[i] = parameters[offset + i];
        _vae.SetParameters(new Vector<T>(vaeParams));
        offset += vaeCount;

        var ctrlParams = new T[ctrlCount];
        for (int i = 0; i < ctrlCount; i++) ctrlParams[i] = parameters[offset + i];
        _controlEncoder.SetParameters(new Vector<T>(ctrlParams));
    }

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();

    /// <inheritdoc />
    public override IDiffusionModel<T> Clone()
    {
        // Clone the ACTUAL predictor and VAE (see InstaFlowModel/MultiDiffusionModel): passing only
        // controlType/conditioner/seed rebuilt InitializeLayers' DEFAULT-sized, lazily-unresolved Flux
        // predictor/VAE, so the field-by-field SetParameters below copied a resolved source predictor
        // into the clone's still-lazy one and threw / mis-shaped. Cloning the resolved predictor/VAE
        // makes those two structurally identical up front; the control encoder isn't a ctor param, so its
        // (config-driven, matching-shape) weights are copied field-by-field afterward.
        var clone = new ControlNetFluxModel<T>(
            architecture: Architecture,
            options: Options as DiffusionModelOptions<T>,
            scheduler: Scheduler,
            predictor: (FluxDoubleStreamPredictor<T>)_predictor.Clone(),
            vae: (StandardVAE<T>)_vae.Clone(),
            controlType: _controlType,
            conditioner: _conditioner,
            seed: RandomGenerator.Next());
        clone._controlEncoder.SetParameters(_controlEncoder.GetParameters());
        return clone;
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "ControlNet-FLUX",
            Version = "1.0",
            Description = "ControlNet adapted for FLUX.1 flow-matching architecture",
            FeatureCount = (int)System.Math.Min((long)int.MaxValue, ParameterCount),
            Complexity = ParameterCount
        };

        metadata.SetProperty("architecture", "flux-double-stream-controlnet");
        metadata.SetProperty("base_model", "FLUX.1");
        metadata.SetProperty("text_encoder", "CLIP-L + T5-XXL");
        metadata.SetProperty("context_dim", FLUX_CONTEXT_DIM);
        metadata.SetProperty("control_type", _controlType.ToString());
        metadata.SetProperty("latent_channels", FLUX_LATENT_CHANNELS);
        metadata.SetProperty("guidance_scale", DEFAULT_GUIDANCE);

        return metadata;
    }
}
