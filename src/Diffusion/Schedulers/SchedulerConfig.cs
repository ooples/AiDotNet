using AiDotNet.Enums;

namespace AiDotNet.Diffusion.Schedulers;

/// <summary>
/// Configuration options for diffusion model step schedulers.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This configuration class defines all the parameters needed to initialize a step scheduler.
/// These parameters control the noise schedule and behavior of the diffusion process.
/// </para>
/// <para>
/// <b>For Beginners:</b> This is like a settings panel for the scheduler. You can control:
///
/// - How many steps to use (more = higher quality, slower)
/// - How much noise to start and end with (the beta values)
/// - What pattern of noise to use (linear, scaled, cosine)
/// - Whether to clip values to prevent extreme outputs
/// - What the model is predicting (noise, sample, or velocity)
///
/// The default values are research-backed and work well for most cases.
/// </para>
/// </remarks>
public sealed class SchedulerConfig<T>
{
    /// <summary>
    /// Gets the number of timesteps used during training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is typically 1000 for most diffusion models. More timesteps allow for
    /// finer-grained noise schedules but increase training time.
    /// </para>
    /// <para>
    /// <b>Default:</b> 1000 (standard for DDPM/DDIM)
    /// </para>
    /// </remarks>
    public int TrainTimesteps { get; }

    /// <summary>
    /// Gets the starting beta value (noise variance at t=0).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Beta start is the amount of noise added at the first timestep.
    /// It should be small since early timesteps should preserve most of the signal.
    /// </para>
    /// <para>
    /// <b>Default:</b> 0.0001 for Linear schedule (from DDPM paper)
    /// </para>
    /// </remarks>
    public T BetaStart { get; }

    /// <summary>
    /// Gets the ending beta value (noise variance at t=T).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Beta end is the amount of noise added at the final timestep.
    /// It should be larger than beta start to ensure samples become nearly pure noise.
    /// </para>
    /// <para>
    /// <b>Default:</b> 0.02 for Linear schedule (from DDPM paper)
    /// </para>
    /// </remarks>
    public T BetaEnd { get; }

    /// <summary>
    /// Gets the type of beta schedule to use.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Different schedules provide different noise progression characteristics.
    /// Linear is simplest, ScaledLinear is used by Stable Diffusion, SquaredCosine
    /// often provides better quality.
    /// </para>
    /// <para>
    /// <b>Default:</b> Linear (most widely compatible)
    /// </para>
    /// </remarks>
    public BetaSchedule BetaSchedule { get; }

    /// <summary>
    /// Gets whether to clip predicted samples to [-1, 1].
    /// </summary>
    /// <remarks>
    /// <para>
    /// Clipping can prevent extreme values that might cause numerical instability,
    /// but may reduce output quality if the model is well-trained.
    /// </para>
    /// <para>
    /// <b>Default:</b> false (let the model produce natural outputs)
    /// </para>
    /// </remarks>
    public bool ClipSample { get; }

    /// <summary>
    /// Gets the type of prediction the model makes.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This must match what the diffusion model was trained to predict.
    /// Most models use Epsilon (noise) prediction.
    /// </para>
    /// <para>
    /// <b>Default:</b> Epsilon (most common)
    /// </para>
    /// </remarks>
    public DiffusionPredictionType PredictionType { get; }

    /// <summary>
    /// Initializes a new scheduler configuration.
    /// </summary>
    /// <param name="trainTimesteps">Number of timesteps for training. Must be greater than 1. Default: 1000.</param>
    /// <param name="betaStart">Starting beta value. Must be positive. Default: 0.0001.</param>
    /// <param name="betaEnd">Ending beta value. Must be greater than betaStart. Default: 0.02.</param>
    /// <param name="betaSchedule">The beta schedule type. Default: Linear.</param>
    /// <param name="clipSample">Whether to clip samples to [-1, 1]. Default: false.</param>
    /// <param name="predictionType">What the model predicts. Default: Epsilon.</param>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when trainTimesteps is less than 2, or when beta values are invalid.
    /// </exception>
    public SchedulerConfig(
        int trainTimesteps,
        T betaStart,
        T betaEnd,
        BetaSchedule betaSchedule = BetaSchedule.Linear,
        bool clipSample = false,
        DiffusionPredictionType predictionType = DiffusionPredictionType.Epsilon)
    {
        if (trainTimesteps <= 1)
            throw new ArgumentOutOfRangeException(nameof(trainTimesteps),
                "Training timesteps must be greater than 1. Typical value is 1000.");

        TrainTimesteps = trainTimesteps;
        BetaStart = betaStart;
        BetaEnd = betaEnd;
        BetaSchedule = betaSchedule;
        ClipSample = clipSample;
        PredictionType = predictionType;
    }

    /// <summary>
    /// Creates a default configuration for DDPM-style models.
    /// </summary>
    /// <returns>A scheduler configuration with standard DDPM defaults.</returns>
    /// <remarks>
    /// <para>
    /// Uses values from the original DDPM paper:
    /// - 1000 training timesteps
    /// - Beta start: 0.0001
    /// - Beta end: 0.02
    /// - Linear beta schedule
    /// - Epsilon prediction
    /// </para>
    /// </remarks>
    public static SchedulerConfig<T> CreateDefault()
    {
        var ops = MathHelper.GetNumericOperations<T>();
        return new SchedulerConfig<T>(
            trainTimesteps: 1000,
            betaStart: ops.FromDouble(0.0001),
            betaEnd: ops.FromDouble(0.02),
            betaSchedule: BetaSchedule.Linear,
            clipSample: false,
            predictionType: DiffusionPredictionType.Epsilon);
    }

    /// <summary>
    /// Creates a configuration optimized for Stable Diffusion-style models.
    /// </summary>
    /// <returns>A scheduler configuration with Stable Diffusion defaults.</returns>
    /// <remarks>
    /// <para>
    /// Uses values from Stable Diffusion:
    /// - 1000 training timesteps
    /// - Beta start: 0.00085
    /// - Beta end: 0.012
    /// - Scaled linear beta schedule
    /// - Epsilon prediction
    /// </para>
    /// </remarks>
    public static SchedulerConfig<T> CreateStableDiffusion()
    {
        var ops = MathHelper.GetNumericOperations<T>();
        return new SchedulerConfig<T>(
            trainTimesteps: 1000,
            betaStart: ops.FromDouble(0.00085),
            betaEnd: ops.FromDouble(0.012),
            betaSchedule: BetaSchedule.ScaledLinear,
            clipSample: false,
            predictionType: DiffusionPredictionType.Epsilon);
    }

    /// <summary>
    /// Creates a configuration for rectified flow models (SD3, FLUX.1).
    /// </summary>
    /// <returns>A scheduler configuration with rectified flow defaults.</returns>
    /// <remarks>
    /// <para>
    /// Uses values for rectified flow matching:
    /// - 1000 training timesteps
    /// - Beta start: 0.0001
    /// - Beta end: 1.0
    /// - Linear beta schedule
    /// - V-prediction (velocity prediction)
    /// </para>
    /// </remarks>
    public static SchedulerConfig<T> CreateRectifiedFlow()
    {
        var ops = MathHelper.GetNumericOperations<T>();
        return new SchedulerConfig<T>(
            trainTimesteps: 1000,
            betaStart: ops.FromDouble(0.0001),
            betaEnd: ops.FromDouble(1.0),
            betaSchedule: BetaSchedule.Linear,
            clipSample: false,
            predictionType: DiffusionPredictionType.VPrediction);
    }

    /// <summary>
    /// Creates a configuration for LCM (Latent Consistency Model) sampling.
    /// </summary>
    /// <returns>A scheduler configuration with LCM defaults.</returns>
    /// <remarks>
    /// <para>
    /// Uses values optimized for LCM:
    /// - 1000 training timesteps
    /// - Scaled linear beta schedule (same as Stable Diffusion base)
    /// - Epsilon prediction
    /// </para>
    /// </remarks>
    public static SchedulerConfig<T> CreateLCM()
    {
        var ops = MathHelper.GetNumericOperations<T>();
        return new SchedulerConfig<T>(
            trainTimesteps: 1000,
            betaStart: ops.FromDouble(0.00085),
            betaEnd: ops.FromDouble(0.012),
            betaSchedule: BetaSchedule.ScaledLinear,
            clipSample: false,
            predictionType: DiffusionPredictionType.Epsilon);
    }
}
