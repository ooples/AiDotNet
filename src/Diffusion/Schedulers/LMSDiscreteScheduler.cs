using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Diffusion.Schedulers;

/// <summary>
/// Linear Multi-Step (LMS) discrete scheduler for diffusion model sampling.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The LMS scheduler uses a linear multi-step method to solve the diffusion ODE.
/// It maintains a history of previous derivatives and uses polynomial interpolation
/// to predict the next step more accurately.
/// </para>
/// <para>
/// <b>For Beginners:</b> LMS uses memory of past steps to predict the future better:
///
/// Imagine navigating a winding road:
/// - Euler: Looks only at the current direction
/// - LMS: Remembers the last 4 turns and uses that pattern to predict the road ahead
///
/// Key characteristics:
/// - Multi-step method using derivative history (order 1-4)
/// - Better accuracy than single-step methods at the same cost
/// - One model evaluation per step
/// - Deterministic: same seed always produces the same result
///
/// The method computes Adams-Bashforth-style coefficients from the sigma schedule
/// and applies them to the stored derivative history.
/// </para>
/// <para>
/// <b>Reference:</b> Based on Adams-Bashforth multi-step ODE methods applied to diffusion.
/// </para>
/// </remarks>
[ComponentType(ComponentType.Scheduler)]
[PipelineStage(PipelineStage.Training)]
public sealed class LMSDiscreteScheduler<T> : NoiseSchedulerBase<T>
{
    [AiDotNet.Attributes.Scratch]
    private Vector<T>? _sigmas;
    private readonly List<Vector<T>> _derivativeHistory = [];
    private readonly int _order;

    /// <summary>
    /// Initializes a new instance of the LMS discrete scheduler.
    /// </summary>
    /// <param name="config">Configuration for the scheduler.</param>
    /// <param name="order">Order of the multi-step method (1-4). Higher = more accurate but needs warmup steps.</param>
    public LMSDiscreteScheduler(SchedulerConfig<T> config, int order = 4) : base(config)
    {
        _order = Math.Max(1, Math.Min(4, order));
    }

    /// <inheritdoc />
    public override void SetTimesteps(int inferenceSteps)
    {
        base.SetTimesteps(inferenceSteps);
        ComputeSigmas();
        _derivativeHistory.Clear();
    }

    private void ComputeSigmas()
    {
        _sigmas = SchedulerTensorOps.KarrasSigmas(AlphasCumulativeProduct, Timesteps);
    }

    /// <summary>
    /// Performs one LMS denoising step using stored derivative history.
    /// </summary>
    public override Vector<T> Step(Vector<T> modelOutput, int timestep, Vector<T> sample, T eta, Vector<T>? noise = null)
    {
        ValidateStepParameters(modelOutput, sample, timestep);

        if (_sigmas == null)
            throw new InvalidOperationException("Sigmas not initialized. Call SetTimesteps() before Step().");

        int stepIndex = FindTimestepIndex(timestep);
        T sigma = _sigmas[stepIndex];

        // Convert model output to predicted original sample
        Vector<T> predOriginal = ConvertToPredOriginal(modelOutput, sample, sigma, timestep);
        predOriginal = ClipSampleIfNeeded(predOriginal);

        // Compute derivative d = (x_t - pred_x_0) / sigma_t
        var derivative = Engine.Subtract(sample, predOriginal);
        derivative = Engine.Divide(derivative, sigma);

        // Store derivative in history, trimming to keep only the last _order entries
        // to prevent unbounded memory growth over many steps
        _derivativeHistory.Add(derivative);
        if (_derivativeHistory.Count > _order)
        {
            _derivativeHistory.RemoveAt(0);
        }

        // Determine the effective order (limited by available history)
        // (and by the sigmas available before this step, as in diffusers' min(step_index + 1, order))
        int effectiveOrder = Math.Min(Math.Min(_order, _derivativeHistory.Count), stepIndex + 1);

        // Compute LMS coefficients for the current step (index 0 = newest derivative)
        var coefficients = ComputeLMSCoefficients(stepIndex, effectiveOrder);

        // Apply multi-step update: x_{t-1} = x_t + sum(coeff_k * d_{t-k}), newest derivative first
        var newestFirst = new List<Vector<T>>(effectiveOrder);
        for (int k = 0; k < effectiveOrder; k++)
        {
            newestFirst.Add(_derivativeHistory[_derivativeHistory.Count - 1 - k]);
        }

        return SchedulerTensorOps.AddWeightedSum(sample, coefficients, newestFirst);
    }

    /// <summary>
    /// Computes the linear multistep coefficients for the current step.
    /// </summary>
    /// <remarks>
    /// As in k-diffusion and diffusers (<c>get_lms_coefficient</c>), coefficient k is the integral over
    /// [sigma_t, sigma_{t+1}] of the Lagrange basis polynomial of node sigma_{t-k} built on the nodes
    /// sigma_t, sigma_{t-1}, ..., sigma_{t-order+1}. Index 0 belongs to the newest derivative.
    /// </remarks>
    private Vector<T> ComputeLMSCoefficients(int stepIndex, int order)
    {
        if (_sigmas is null)
            throw new InvalidOperationException("Sigmas not initialized. Call SetTimesteps() before Step().");

        T sigmaNext = _sigmas[stepIndex + 1];
        T sigmaCurrent = _sigmas[stepIndex];

        // Nodes newest first: sigma_t, sigma_{t-1}, ...
        var oldestFirst = _sigmas.GetSubVector(stepIndex - order + 1, order);
        var nodes = Engine.TensorFlip(oldestFirst.AsTensor(), new[] { 0 }).ToVector();
        return SchedulerTensorOps.IntegrateLagrangeBasis(nodes, sigmaCurrent, sigmaNext);
    }

    private Vector<T> ConvertToPredOriginal(Vector<T> modelOutput, Vector<T> sample, T sigma, int timestep)
    {
        switch (Config.PredictionType)
        {
            case DiffusionPredictionType.Sample:
                return modelOutput;

            case DiffusionPredictionType.VPrediction:
                // The sample lives in sigma space (x_t / sqrt(alpha_cumprod)), so as in diffusers
                // x0 = sample / (sigma^2 + 1) - v * sigma / sqrt(sigma^2 + 1).
                T sigmaSqPlusOne = NumOps.Add(NumOps.Multiply(sigma, sigma), NumOps.One);
                var vX0 = Engine.Divide(sample, sigmaSqPlusOne);
                var vEps = Engine.Multiply(modelOutput, NumOps.Divide(sigma, NumOps.Sqrt(sigmaSqPlusOne)));
                return Engine.Subtract(vX0, vEps);

            default: // Epsilon
                var scaledNoise = Engine.Multiply(modelOutput, sigma);
                return Engine.Subtract(sample, scaledNoise);
        }
    }

    private int FindTimestepIndex(int timestep)
    {
        var timesteps = Timesteps;
        for (int i = 0; i < timesteps.Length; i++)
        {
            if (timesteps[i] == timestep)
                return i;
        }

        int closestIdx = 0;
        int closestDist = Math.Abs(timesteps[0] - timestep);
        for (int i = 1; i < timesteps.Length; i++)
        {
            int dist = Math.Abs(timesteps[i] - timestep);
            if (dist < closestDist)
            {
                closestDist = dist;
                closestIdx = i;
            }
        }

        return closestIdx;
    }

    /// <summary>
    /// Creates an LMS scheduler with default Stable Diffusion settings.
    /// </summary>
    public static LMSDiscreteScheduler<T> CreateDefault()
    {
        return new LMSDiscreteScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion());
    }
}
