using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Diffusion;
using AiDotNet.NeuralNetworks.Diffusion.Schedulers;

namespace AiDotNet.Diffusion.Models;

/// <summary>
/// Wan video model — Alibaba's scalable DiT for high-quality video generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Wan (Alibaba) uses a scalable DiT with 3D full attention and WanVAE for
/// temporally compressed video generation at multiple scales (1.3B to 14B).
/// </para>
/// <para>
/// <b>For Beginners:</b> Wan generates high-quality videos with multiple size variants:
///
/// Key characteristics:
/// - Scalable DiT: 1.3B, 5B, 14B parameter variants
/// - WanVAE: specialized causal 3D VAE
/// - Full 3D attention (no factorization) for quality
/// - Text-to-video, image-to-video support
///
/// Reference: Alibaba, "Wan: Open and Advanced Large-Scale Video Generative Models", 2025
/// </para>
/// </remarks>
public class WanVideoModel<T> : VideoDiffusionModelBase<T>
{
    private const int LATENT_CHANNELS = 16;
    private const int CONTEXT_DIM = 4096;

    private readonly DiTNoisePredictor<T> _dit;
    private readonly TemporalVAE<T> _temporalVAE;
    private readonly IConditioningModule<T>? _conditioner;
    private readonly string _variant;
    private readonly int _numHeads;

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _dit;
    /// <inheritdoc />
    public override IVAEModel<T> VAE => _temporalVAE;
    /// <inheritdoc />
    public override IVAEModel<T>? TemporalVAE => _temporalVAE;
    /// <inheritdoc />
    public override IConditioningModule<T>? Conditioner => _conditioner;
    /// <inheritdoc />
    public override int LatentChannels => LATENT_CHANNELS;
    /// <inheritdoc />
    public override bool SupportsImageToVideo => true;
    /// <inheritdoc />
    public override bool SupportsTextToVideo => true;
    /// <inheritdoc />
    public override bool SupportsVideoToVideo => false;
    /// <inheritdoc />
    public override int ParameterCount => _dit.ParameterCount + _temporalVAE.GetParameters().Length;

    /// <summary>Gets the model variant (1.3B, 5B, or 14B).</summary>
    public string Variant => _variant;

    public WanVideoModel(
        DiffusionModelOptions<T>? options = null, INoiseScheduler<T>? scheduler = null,
        DiTNoisePredictor<T>? dit = null, TemporalVAE<T>? temporalVAE = null,
        IConditioningModule<T>? conditioner = null,
        string variant = "14B",
        int defaultNumFrames = 81, int defaultFPS = 16)
        : base(options ?? new DiffusionModelOptions<T> { TrainTimesteps = 1000, BetaStart = 0.0001, BetaEnd = 0.02, BetaSchedule = BetaSchedule.Linear },
               scheduler ?? new FlowMatchingScheduler<T>(SchedulerConfig<T>.CreateDefault()),
               defaultNumFrames, defaultFPS)
    {
        _variant = variant;
        _conditioner = conditioner;

        var (hiddenDim, numLayers, numHeads) = variant switch
        {
            "1.3B" => (1536, 30, 12),
            "5B" => (2560, 36, 20),
            _ => (3072, 40, 24),
        };
        _numHeads = numHeads;

        _dit = dit ?? new DiTNoisePredictor<T>(
            inputChannels: LATENT_CHANNELS, hiddenSize: hiddenDim,
            numLayers: numLayers, numHeads: numHeads,
            patchSize: 2, contextDim: CONTEXT_DIM);
        _temporalVAE = temporalVAE ?? new TemporalVAE<T>(
            inputChannels: 3, latentChannels: LATENT_CHANNELS,
            baseChannels: 128, channelMultipliers: new[] { 1, 2, 4, 4 },
            numTemporalLayers: 3, temporalKernelSize: 3,
            causalMode: true, latentScaleFactor: 0.13025);
    }

    /// <summary>Creates a 1.3B lightweight variant.</summary>
    public static WanVideoModel<T> Create1_3B(IConditioningModule<T>? conditioner = null)
        => new(variant: "1.3B", conditioner: conditioner);

    /// <summary>Creates a 5B medium variant.</summary>
    public static WanVideoModel<T> Create5B(IConditioningModule<T>? conditioner = null)
        => new(variant: "5B", conditioner: conditioner);

    /// <inheritdoc />
    protected override Tensor<T> PredictVideoNoise(Tensor<T> latents, int timestep,
        Tensor<T> imageEmbedding, Tensor<T> motionEmbedding)
        => _dit.PredictNoise(latents, timestep, imageEmbedding);

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var dp = _dit.GetParameters(); var vp = _temporalVAE.GetParameters();
        var c = new Vector<T>(dp.Length + vp.Length);
        for (int i = 0; i < dp.Length; i++) c[i] = dp[i];
        for (int i = 0; i < vp.Length; i++) c[dp.Length + i] = vp[i];
        return c;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        int dc = _dit.ParameterCount, vc = _temporalVAE.GetParameters().Length;
        var dp = new Vector<T>(dc); var vp = new Vector<T>(vc);
        for (int i = 0; i < dc; i++) dp[i] = parameters[i];
        for (int i = 0; i < vc; i++) vp[i] = parameters[dc + i];
        _dit.SetParameters(dp); _temporalVAE.SetParameters(vp);
    }

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();
    /// <inheritdoc />
    public override IDiffusionModel<T> Clone()
    {
        return new WanVideoModel<T>(
            dit: new DiTNoisePredictor<T>(
                inputChannels: LATENT_CHANNELS, hiddenSize: _dit.HiddenSize,
                numLayers: _dit.NumLayers, numHeads: _numHeads,
                patchSize: 2, contextDim: CONTEXT_DIM),
            temporalVAE: (TemporalVAE<T>)_temporalVAE.Clone(),
            conditioner: _conditioner, variant: _variant,
            defaultNumFrames: DefaultNumFrames, defaultFPS: DefaultFPS);
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T> { Name = $"Wan-{_variant}", Version = "2.1", ModelType = ModelType.NeuralNetwork,
            Description = $"Wan {_variant} video generation model", FeatureCount = ParameterCount, Complexity = ParameterCount };
        m.SetProperty("architecture", "dit-full-3d-attention");
        m.SetProperty("variant", _variant);
        m.SetProperty("open_source", true);
        return m;
    }
}
