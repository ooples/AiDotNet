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
/// PeRFlow (Piecewise Rectified Flow) model for few-step generation via flow straightening.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// PeRFlow divides the time interval [0,1] into K segments and straightens the flow within
/// each segment independently. This piecewise approach is more effective than global
/// straightening (Reflow) because each segment needs less straightening, achieving better
/// quality at 4-8 steps.
/// </para>
/// <para>
/// <b>For Beginners:</b> Instead of trying to straighten the entire winding road from
/// noise to image at once, PeRFlow divides it into segments and straightens each piece.
/// This is easier and more effective — like fixing a winding highway by straightening
/// each curve individually rather than rebuilding the whole road.
/// </para>
/// <para>
/// Reference: Yan et al., "PeRFlow: Piecewise Rectified Flow as Universal Plug-and-Play Accelerator", 2024
/// </para>
/// </remarks>
public class PeRFlowModel<T> : LatentDiffusionModelBase<T>
{
    private const int LATENT_CHANNELS = 4;
    private const double DEFAULT_GUIDANCE = 3.0;
    private const int DEFAULT_NUM_SEGMENTS = 4;

    private UNetNoisePredictor<T> _predictor;
    private StandardVAE<T> _vae;
    private readonly IConditioningModule<T>? _conditioner;
    private readonly int _numSegments;

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
    /// Gets the number of piecewise segments.
    /// </summary>
    public int NumSegments => _numSegments;

    /// <summary>
    /// Initializes a new PeRFlow model.
    /// </summary>
    /// <param name="numSegments">Number of piecewise segments (default: 4).</param>
    public PeRFlowModel(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        UNetNoisePredictor<T>? predictor = null,
        StandardVAE<T>? vae = null,
        IConditioningModule<T>? conditioner = null,
        int numSegments = DEFAULT_NUM_SEGMENTS,
        int? seed = null)
        : base(
            options ?? new DiffusionModelOptions<T>
            {
                TrainTimesteps = 1000, BetaStart = 0.0001,
                BetaEnd = 0.02, BetaSchedule = BetaSchedule.Linear
            },
            scheduler ?? new FlowMatchingScheduler<T>(SchedulerConfig<T>.CreateRectifiedFlow()),
            architecture)
    {
        _conditioner = conditioner;
        _numSegments = numSegments;
        InitializeLayers(predictor, vae, seed);
        SetGuidanceScale(DEFAULT_GUIDANCE);
    }

    [MemberNotNull(nameof(_predictor), nameof(_vae))]
    private void InitializeLayers(UNetNoisePredictor<T>? predictor, StandardVAE<T>? vae, int? seed)
    {
        _predictor = predictor ?? new UNetNoisePredictor<T>(
            inputChannels: LATENT_CHANNELS, outputChannels: LATENT_CHANNELS,
            baseChannels: 320, channelMultipliers: [1, 2, 4, 4],
            numResBlocks: 2, attentionResolutions: [4, 2, 1],
            contextDim: 768, architecture: Architecture, seed: seed);

        _vae = vae ?? new StandardVAE<T>(
            inputChannels: 3, latentChannels: LATENT_CHANNELS,
            baseChannels: 128, channelMultipliers: [1, 2, 4, 4],
            numResBlocksPerLevel: 2, latentScaleFactor: 0.18215, seed: seed);
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
        var clone = new PeRFlowModel<T>(
            conditioner: _conditioner, numSegments: _numSegments, seed: RandomGenerator.Next());
        clone.SetParameters(GetParameters());
        return clone;
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = "PeRFlow", Version = "1.0", ModelType = ModelType.NeuralNetwork,
            Description = "Piecewise rectified flow with segment-wise straightening for efficient few-step generation",
            FeatureCount = ParameterCount, Complexity = ParameterCount
        };
        m.SetProperty("architecture", "piecewise-rectified-flow-unet");
        m.SetProperty("num_segments", _numSegments);
        m.SetProperty("optimal_steps", _numSegments);
        m.SetProperty("max_recommended_steps", _numSegments * 2);
        m.SetProperty("guidance_scale", DEFAULT_GUIDANCE);
        m.SetProperty("latent_channels", LATENT_CHANNELS);
        return m;
    }
}
