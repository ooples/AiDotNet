using AiDotNet.Diffusion;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Video;

/// <summary>
/// Base class for DIFFUSION-based video super-resolution models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> some video upscalers work by repeatedly "denoising" a noisy picture until a
/// sharp frame emerges (diffusion), rather than transforming pixels in a single pass. Those models need
/// two things at once: the diffusion machinery (a latent space and a denoising loop) and the
/// super-resolution contract (an upscale factor). This base provides both.
/// </para>
/// <para>
/// <b>Why this class exists.</b> Six models in the library are diffusion-based video super-resolution —
/// Stream-DiffVSR, StableVideoSR, MGLD-VSR, DOVE, SeedVR and FlashVSR — yet all of them derived from
/// <see cref="VideoSuperResolutionBase{T}"/>, which has no latent space, no noise scheduler and no
/// denoiser. Each therefore either reimplemented diffusion privately or, more commonly, silently did
/// not implement it at all: Stream-DiffVSR was a plain conv/resblock/upsample CNN trained with one
/// feed-forward MSE pass, with none of its paper's 4-step distilled denoiser, rollout distillation,
/// ARTG or TPM decoder.
/// </para>
/// <para>
/// C# allows a single base class, so the super-resolution capability is expressed as
/// <see cref="IVideoSuperResolution{T}"/> — implemented by both this class and
/// <see cref="VideoSuperResolutionBase{T}"/> — while the diffusion machinery is inherited from
/// <see cref="VideoDiffusionModelBase{T}"/>. Consumers and the test-scaffold generator can then detect
/// the capability by interface rather than by matching a base-class NAME, which silently excludes any
/// model whose base is named differently.
/// </para>
/// <para>
/// Derived models must supply <see cref="DiffusionModelBase{T}.PredictNoise"/> — the denoiser — because
/// that is what makes a diffusion model a diffusion model. A model that cannot provide one does not
/// belong on this base.
/// </para>
/// </remarks>
public abstract class DiffusionVideoSuperResolutionBase<T> : VideoDiffusionModelBase<T>, IVideoSuperResolution<T>
{
    private RAFTFlowCache? _flowCache;

    /// <summary>
    /// Initializes the base.
    /// </summary>
    /// <param name="options">Diffusion options (timesteps, guidance, and related settings).</param>
    /// <param name="scheduler">Noise scheduler; the diffusion base supplies a default when null.</param>
    /// <param name="scaleFactor">Spatial upscaling factor. Default 4, the x4 setting the VSR literature
    /// reports against.</param>
    /// <param name="defaultNumFrames">Default clip length for generation paths.</param>
    /// <param name="defaultFPS">Default frame rate for generation paths.</param>
    /// <param name="architecture">Optional network architecture.</param>
    protected DiffusionVideoSuperResolutionBase(
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        int scaleFactor = 4,
        int defaultNumFrames = 25,
        int defaultFPS = 7,
        NeuralNetworkArchitecture<T>? architecture = null)
        : base(options, scheduler, defaultNumFrames, defaultFPS, architecture)
    {
        if (scaleFactor <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(scaleFactor), scaleFactor, "Scale factor must be positive.");
        }

        ScaleFactor = scaleFactor;
    }

    /// <inheritdoc/>
    public int ScaleFactor { get; protected set; }

    /// <inheritdoc/>
    public abstract Tensor<T> Upscale(Tensor<T> lowResFrames);

    /// <inheritdoc/>
    /// <remarks>
    /// Delegates to the library's RAFT implementation (Teed &amp; Deng 2020), matching the default now
    /// used by <see cref="VideoSuperResolutionBase{T}"/> so both hierarchies estimate real motion
    /// rather than returning a zero field. Cached, since constructing a flow network per frame pair
    /// would dominate cost.
    /// </remarks>
    public virtual Tensor<T> EstimateFlow(Tensor<T> frame1, Tensor<T> frame2)
    {
        _flowCache ??= new RAFTFlowCache();
        return _flowCache.Estimate(frame1, frame2);
    }

    /// <summary>Holds the lazily created RAFT estimator so it is built once per model.</summary>
    private sealed class RAFTFlowCache
    {
        private AiDotNet.Video.Motion.RAFT<T>? _raft;

        internal Tensor<T> Estimate(Tensor<T> frame1, Tensor<T> frame2)
        {
            _raft ??= new AiDotNet.Video.Motion.RAFT<T>();
            return _raft.EstimateFlow(frame1, frame2);
        }
    }
}
