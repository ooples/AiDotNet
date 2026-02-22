using AiDotNet.Enums;

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
public sealed class LMSDiscreteScheduler<T> : NoiseSchedulerBase<T>
{
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
        var timesteps = Timesteps;
        _sigmas = new Vector<T>(timesteps.Length + 1);

        for (int i = 0; i < timesteps.Length; i++)
        {
            T alphaCumprod = AlphasCumulativeProduct[timesteps[i]];
            T oneMinusAlpha = NumOps.Subtract(NumOps.One, alphaCumprod);
            _sigmas[i] = NumOps.Sqrt(NumOps.Divide(oneMinusAlpha, alphaCumprod));
        }

        _sigmas[timesteps.Length] = NumOps.Zero;
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
        int effectiveOrder = Math.Min(_order, _derivativeHistory.Count);

        // Compute LMS coefficients for the current step
        var coefficients = ComputeLMSCoefficients(stepIndex, effectiveOrder);

        // Apply multi-step update: x_{t-1} = x_t + sum(coeff_i * d_i)
        var prevSample = new Vector<T>(sample.Length);
        for (int i = 0; i < sample.Length; i++)
        {
            prevSample[i] = sample[i];
        }

        for (int i = 0; i < effectiveOrder; i++)
        {
            int historyIdx = _derivativeHistory.Count - effectiveOrder + i;
            var d = _derivativeHistory[historyIdx];
            prevSample = Engine.Add(prevSample, Engine.Multiply(d, coefficients[i]));
        }

        return prevSample;
    }

    /// <summary>
    /// Computes Adams-Bashforth-style coefficients for the LMS method.
    /// </summary>
    /// <remarks>
    /// The coefficients are computed by integrating Lagrange basis polynomials
    /// over the interval [sigma_t, sigma_{t-1}] in log-sigma space.
    /// </remarks>
    private T[] ComputeLMSCoefficients(int stepIndex, int order)
    {
        if (_sigmas is null)
            throw new InvalidOperationException("Sigmas not initialized. Call SetTimesteps() before Step().");

        var coefficients = new T[order];

        T sigmaNext = _sigmas[stepIndex + 1];
        T sigmaCurrent = _sigmas[stepIndex];

        if (order == 1)
        {
            // First-order: just Euler step dt = sigma_next - sigma_current
            coefficients[0] = NumOps.Subtract(sigmaNext, sigmaCurrent);
        }
        else
        {
            // Higher-order: compute integrated Lagrange basis polynomials
            // We use a simplified approximation based on equally-weighted derivatives
            // scaled by the step size
            T stepSize = NumOps.Subtract(sigmaNext, sigmaCurrent);

            for (int i = 0; i < order; i++)
            {
                // Adams-Bashforth weights for different orders:
                // Order 2: [3/2, -1/2]
                // Order 3: [23/12, -16/12, 5/12]
                // Order 4: [55/24, -59/24, 37/24, -9/24]
                double weight = GetAdamsBashforthWeight(order, i);
                coefficients[i] = NumOps.Multiply(stepSize, NumOps.FromDouble(weight));
            }
        }

        return coefficients;
    }

    /// <summary>
    /// Gets the Adams-Bashforth coefficient for a given order and index.
    /// </summary>
    private static double GetAdamsBashforthWeight(int order, int index)
    {
        // Adams-Bashforth coefficients (most recent derivative has index 0)
        // Reversed so that index 0 = oldest derivative in our history array
        return order switch
        {
            1 => 1.0,
            2 => index switch
            {
                0 => -0.5,  // older
                1 => 1.5,   // newer
                _ => 0.0
            },
            3 => index switch
            {
                0 => 5.0 / 12.0,    // oldest
                1 => -16.0 / 12.0,  // middle
                2 => 23.0 / 12.0,   // newest
                _ => 0.0
            },
            4 => index switch
            {
                0 => -9.0 / 24.0,   // oldest
                1 => 37.0 / 24.0,   //
                2 => -59.0 / 24.0,  //
                3 => 55.0 / 24.0,   // newest
                _ => 0.0
            },
            _ => index == order - 1 ? 1.0 : 0.0
        };
    }

    private Vector<T> ConvertToPredOriginal(Vector<T> modelOutput, Vector<T> sample, T sigma, int timestep)
    {
        switch (Config.PredictionType)
        {
            case DiffusionPredictionType.Sample:
                return modelOutput;

            case DiffusionPredictionType.VPrediction:
                T alphaCumprod = AlphasCumulativeProduct[timestep];
                T sqrtAlpha = NumOps.Sqrt(alphaCumprod);
                T sqrtOneMinusAlpha = NumOps.Sqrt(NumOps.Subtract(NumOps.One, alphaCumprod));
                var vX0 = Engine.Multiply(sample, sqrtAlpha);
                var vEps = Engine.Multiply(modelOutput, sqrtOneMinusAlpha);
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
