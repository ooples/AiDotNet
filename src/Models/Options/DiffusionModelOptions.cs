using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for diffusion-based generative models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This options class provides configuration for all diffusion model parameters including
/// training hyperparameters, scheduler configuration, and generation settings.
/// </para>
/// <para><b>For Beginners:</b> Diffusion models work by learning to reverse a gradual noising process.
/// These options control how the model trains and generates samples:
/// <list type="bullet">
/// <item><description>LearningRate: How big of a step to take during training</description></item>
/// <item><description>TrainTimesteps: How many noise levels to use (more = finer control)</description></item>
/// <item><description>BetaStart/BetaEnd: How much noise at each step</description></item>
/// </list>
/// </para>
/// </remarks>
public class DiffusionModelOptions<T> : ModelOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public DiffusionModelOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public DiffusionModelOptions(DiffusionModelOptions<T> other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        LearningRate = other.LearningRate;
        OptimizerFactory = other.OptimizerFactory;
        TrainTimesteps = other.TrainTimesteps;
        BetaStart = other.BetaStart;
        BetaEnd = other.BetaEnd;
        BetaSchedule = other.BetaSchedule;
        PredictionType = other.PredictionType;
        ClipSample = other.ClipSample;
        DefaultInferenceSteps = other.DefaultInferenceSteps;
        LossFunction = other.LossFunction;
        UseGpuExecutionGraph = other.UseGpuExecutionGraph;
    }

    /// <summary>
    /// Gets or sets the learning rate for training parameter updates.
    /// </summary>
    /// <value>The learning rate, defaulting to 0.001.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The learning rate controls how big each step is during training.
    /// A value of 0.001 means taking small, careful steps. If this value is too large, the model might
    /// overshoot the optimal solution. If it's too small, training will take a very long time.
    /// The default of 0.001 is a good starting point for most diffusion models.</para>
    /// </remarks>
    public double LearningRate { get; set; } = 0.001;

    /// <summary>
    /// Gets or sets the factory used to create the gradient-based optimizer for training updates.
    /// </summary>
    /// <value>
    /// A factory that creates a fresh <see cref="IGradientBasedOptimizer{T, TInput, TOutput}"/>,
    /// or <c>null</c> (the default) to use the model's industry-standard default optimizer.
    /// </value>
    /// <remarks>
    /// <para>
    /// When this is <c>null</c>, the diffusion model trains with the paper-faithful default: a plain
    /// <b>Adam</b> optimizer (Ho et al. 2020, DDPM Algorithm 1 — Adam, <i>not</i> AdamW) configured with
    /// the standard hyperparameters β₁ = 0.9, β₂ = 0.999, ε = 1e-8, <b>no weight decay</b>, and the
    /// learning rate from <see cref="LearningRate"/>. The non-paper extras of the Adam implementation
    /// (adaptive betas, adaptive learning rate, AMSGrad) are left off so the default reproduces the
    /// paper exactly.
    /// </para>
    /// <para>
    /// Set this to customize the optimizer at every level: a differently-tuned Adam (custom betas /
    /// epsilon / AMSGrad), AdamW with decoupled weight decay, SGD with momentum, or any custom
    /// <see cref="IGradientBasedOptimizer{T, TInput, TOutput}"/>. The factory must create a new optimizer
    /// instance for each model so moment buffers, step counters, and checkpointed optimizer state are
    /// never shared between copied option objects or cloned models.
    /// </para>
    /// <para><b>For Beginners:</b> The optimizer is the rule for turning each step's error signal into a
    /// weight update. Leave this unset to use the proven DDPM default (Adam); set a factory only to take
    /// full control of the optimizer yourself.</para>
    /// </remarks>
    public Func<IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>>? OptimizerFactory { get; set; }

    /// <summary>
    /// Gets or sets the number of timesteps used during training.
    /// </summary>
    /// <value>The number of training timesteps, defaulting to 1000.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This controls how many discrete noise levels the model learns.
    /// More timesteps (like 1000) give finer control over the denoising process but require more
    /// computation. The default of 1000 is standard for DDPM/DDIM models.</para>
    /// </remarks>
    public int TrainTimesteps { get; set; } = 1000;

    /// <summary>
    /// Gets or sets the starting beta value (noise variance at t=0).
    /// </summary>
    /// <value>The starting beta value, defaulting to 0.0001.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Beta start is the amount of noise added at the first timestep.
    /// It should be small since early timesteps should preserve most of the signal.
    /// The default of 0.0001 is from the original DDPM paper.</para>
    /// </remarks>
    public double BetaStart { get; set; } = 0.0001;

    /// <summary>
    /// Gets or sets the ending beta value (noise variance at t=T).
    /// </summary>
    /// <value>The ending beta value, defaulting to 0.02.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Beta end is the amount of noise added at the final timestep.
    /// It should be larger than beta start to ensure samples become nearly pure noise.
    /// The default of 0.02 is from the original DDPM paper.</para>
    /// </remarks>
    public double BetaEnd { get; set; } = 0.02;

    /// <summary>
    /// Gets or sets the type of beta schedule to use.
    /// </summary>
    /// <value>The beta schedule type, defaulting to Linear.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The beta schedule determines how noise variance changes across timesteps.
    /// Linear is simplest and widely compatible. ScaledLinear is used by Stable Diffusion.
    /// SquaredCosine often provides better quality results.</para>
    /// </remarks>
    public BetaSchedule BetaSchedule { get; set; } = BetaSchedule.Linear;

    /// <summary>
    /// Gets or sets what the model is trained to predict.
    /// </summary>
    /// <value>The prediction type, defaulting to Epsilon (noise).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This must match what the diffusion model was trained to predict.
    /// Most models use Epsilon (noise) prediction. Some newer models use velocity prediction
    /// or direct sample prediction.</para>
    /// </remarks>
    public DiffusionPredictionType PredictionType { get; set; } = DiffusionPredictionType.Epsilon;

    /// <summary>
    /// Gets or sets whether to clip predicted samples to [-1, 1].
    /// </summary>
    /// <value>True to clip samples, false otherwise. Defaults to false.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Clipping can prevent extreme values that might cause numerical
    /// instability, but may reduce output quality if the model is well-trained.
    /// Generally leave this as false unless you experience numerical issues.</para>
    /// </remarks>
    public bool ClipSample { get; set; } = false;

    /// <summary>
    /// Gets or sets the default number of inference steps for generation.
    /// </summary>
    /// <value>The number of inference steps, defaulting to 10.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> During generation, you don't need all 1000 training timesteps.
    /// Using fewer steps is much faster while still producing good quality.
    /// DDIM (Song et al. 2020, "Denoising Diffusion Implicit Models") shows 20 inference
    /// steps produce near-identical quality to 1000 on ImageNet-scale benchmarks;
    /// DPM-Solver (Lu et al. 2022) further demonstrates 10 steps suffice with a
    /// higher-order solver. 10 sits inside the paper-validated range for the default
    /// DDIM/PNDM schedulers and fits the 120-second xUnit smoke test budget on
    /// channel-heavy UNets (e.g. SD-Inpainting's 9-channel input where each UNet
    /// forward takes ~5s on CPU). Callers needing full-quality 50+ step output
    /// (DDPM full-sampling per Ho et al. 2020) should pass the step count to
    /// <see cref="Diffusion.DiffusionModelBase{T}.Generate"/> directly.</para>
    /// </remarks>
    public int DefaultInferenceSteps { get; set; } = 10;

    /// <summary>
    /// Gets or sets the loss function for training.
    /// </summary>
    /// <value>The loss function, or null to use Mean Squared Error.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The loss function measures how wrong the model's predictions are.
    /// Mean Squared Error (MSE) is standard for diffusion models since they predict continuous values.
    /// Leave as null to use the default MSE loss.</para>
    /// </remarks>
    public ILossFunction<T>? LossFunction { get; set; } = null;

    /// <summary>
    /// Gets or sets whether each synchronous denoising-step noise prediction runs inside a GPU
    /// deferred execution graph (device-resident, fused, multi-stream) instead of eager per-op
    /// dispatch. Default <c>false</c> (opt-in).
    /// </summary>
    /// <value><c>true</c> to record each synchronous denoising step into one fused GPU deferred
    /// execution graph (CUDA <c>DirectGpuTensorEngine</c> only); <c>false</c> (the default) to use
    /// eager per-op dispatch.</value>
    /// <remarks>
    /// <para>
    /// When enabled AND the active engine is a CUDA <c>DirectGpuTensorEngine</c>, each SYNCHRONOUS
    /// <see cref="Diffusion.DiffusionModelBase{T}.PredictNoise"/> call (the <c>Generate</c> path, via
    /// <c>PredictNoiseStep</c>) is recorded into one fused GPU execution graph (AiDotNet.Tensors #642)
    /// — keeping intermediates on-device across the forward, applying kernel fusion / multi-stream
    /// overlap / buffer reuse, and eliminating per-op host round-trips. This is the substrate a
    /// CUDA-graph capture replays.
    /// </para>
    /// <para><b>Applies to synchronous generation only.</b> The asynchronous loop (<c>GenerateAsync</c>
    /// → <c>PredictNoiseAsync</c>) routes through the compile host's async execution path and does NOT
    /// consult this flag, so enabling it has no effect on <c>GenerateAsync</c>.</para>
    /// <para><b>Correctness caveat — this is NOT an unconditional "never worse than eager" guarantee.</b>
    /// A deferred forward that throws a recoverable exception transparently falls back to the eager
    /// <see cref="Diffusion.DiffusionModelBase{T}.PredictNoise"/>, so <i>detected</i> failures are safe;
    /// but an op that is not yet deferred-correct can silently produce wrong-but-finite output, which
    /// the fallback CANNOT detect. <b>Requires AiDotNet.Tensors with the #642 deferred-graph correctness
    /// fixes</b> (GroupNorm arg-order, GroupNorm/InstanceNorm recording, FusedConv2D lazy download,
    /// GPU-resident in-place activations). Until every op a given model's forward uses is verified
    /// deferred-correct, leave this off. Validated end-to-end for the GroupNorm→Swish→Conv ResBlock;
    /// attention / up- &amp; down-sampling / concat / projection paths are pending the Tensors
    /// op-coverage audit.</para>
    /// <para><b>For Beginners:</b> a speed switch for GPU image/audio generation. Leave it off
    /// unless you've confirmed your model + Tensors version support it; when on, it makes the GPU
    /// do a whole denoising step as one batched job instead of hundreds of tiny ones.</para>
    /// </remarks>
    public bool UseGpuExecutionGraph { get; set; } = false;
}
