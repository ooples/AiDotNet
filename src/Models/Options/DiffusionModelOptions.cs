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
        TrainTimesteps = other.TrainTimesteps;
        BetaStart = other.BetaStart;
        BetaEnd = other.BetaEnd;
        BetaSchedule = other.BetaSchedule;
        PredictionType = other.PredictionType;
        ClipSample = other.ClipSample;
        DefaultInferenceSteps = other.DefaultInferenceSteps;
        LossFunction = other.LossFunction;
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
    /// <value>The number of inference steps, defaulting to 50.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> During generation, you don't need all 1000 training timesteps.
    /// Using fewer steps (like 50) is much faster while still producing good quality.
    /// DDIM and PNDM schedulers are designed to work well with fewer steps.</para>
    /// </remarks>
    public int DefaultInferenceSteps { get; set; } = 50;

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
}
