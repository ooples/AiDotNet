using AiDotNet.Enums;

namespace AiDotNet.NeuralNetworks.Diffusion.Schedulers;

/// <summary>
/// Diffusion Exponential Integrator Sampler (DEIS) for fast diffusion model sampling.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// DEIS uses exponential integrators with polynomial extrapolation to solve
/// the diffusion ODE. It achieves high-quality samples with very few steps
/// by leveraging the exponential structure of the diffusion process.
/// </para>
/// <para>
/// <b>For Beginners:</b> DEIS is a smart math-based sampler:
///
/// The key insight is that the diffusion ODE has an exponential structure.
/// By using exponential integrators (which are exact for exponential functions),
/// DEIS captures more of the solution's behavior in fewer steps.
///
/// Key characteristics:
/// - Multi-step method (orders 1-3)
/// - Uses exponential integrator formulas
/// - Excellent quality at low step counts (10-20 steps)
/// - Deterministic: same seed always produces the same result
/// - Stores derivative history like LMS, but uses exponential interpolation
///
/// Think of it like: instead of approximating a curve with straight lines (Euler)
/// or polynomials (LMS), DEIS uses exponentials which better match the
/// diffusion process's natural shape.
/// </para>
/// <para>
/// <b>Reference:</b> Zhang and Chen, "Fast Sampling of Diffusion Models with Exponential Integrator", ICLR 2023
/// </para>
/// </remarks>
public sealed class DEISMultistepScheduler<T> : NoiseSchedulerBase<T>
{
    private Vector<T>? _sigmas;
    private readonly List<Vector<T>> _modelOutputHistory = [];
    private readonly int _order;

    /// <summary>
    /// Initializes a new instance of the DEIS multistep scheduler.
    /// </summary>
    /// <param name="config">Configuration for the scheduler.</param>
    /// <param name="order">Order of the DEIS method (1-3). Higher orders are more accurate.</param>
    public DEISMultistepScheduler(SchedulerConfig<T> config, int order = 3) : base(config)
    {
        _order = Math.Max(1, Math.Min(3, order));
    }

    /// <inheritdoc />
    public override void SetTimesteps(int inferenceSteps)
    {
        base.SetTimesteps(inferenceSteps);
        ComputeSigmas();
        _modelOutputHistory.Clear();
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
    /// Performs one DEIS denoising step using exponential integrator formulas.
    /// </summary>
    public override Vector<T> Step(Vector<T> modelOutput, int timestep, Vector<T> sample, T eta, Vector<T>? noise = null)
    {
        ValidateStepParameters(modelOutput, sample, timestep);

        if (_sigmas == null)
            throw new InvalidOperationException("Sigmas not initialized. Call SetTimesteps() before Step().");

        int stepIndex = FindTimestepIndex(timestep);
        T sigma = _sigmas[stepIndex];
        T sigmaNext = _sigmas[stepIndex + 1];

        // Convert model output to data prediction (x_0 prediction)
        Vector<T> dataPrediction = ConvertToDataPrediction(modelOutput, sample, sigma, timestep);
        dataPrediction = ClipSampleIfNeeded(dataPrediction);

        // Store model output history for multi-step
        _modelOutputHistory.Add(dataPrediction);

        int effectiveOrder = Math.Min(_order, _modelOutputHistory.Count);

        // Compute lambda values (log-SNR)
        double lambdaCurrent = ComputeLambda(sigma);
        double lambdaNext = ComputeLambda(sigmaNext);
        double h = lambdaNext - lambdaCurrent;

        // Exponential integrator step
        double sigmaDbl = NumOps.ToDouble(sigma);
        double sigmaNextDbl = NumOps.ToDouble(sigmaNext);

        // alpha_t / alpha_s ratio (using sigma parameterization)
        double alphaRatio = Math.Sqrt((1.0 + sigmaNextDbl * sigmaNextDbl) / (1.0 + sigmaDbl * sigmaDbl));

        // Exponential factor
        double expH = Math.Exp(h);

        Vector<T> prevSample;

        if (effectiveOrder == 1 || NumOps.ToDouble(sigmaNext) < 1e-10)
        {
            // First-order: exponential Euler
            // x_{t-1} = (sigma_next/sigma) * x_t + (alpha_next - sigma_next/sigma * alpha_t) * D_0
            var scaledSample = Engine.Multiply(sample, NumOps.FromDouble(sigmaNextDbl / Math.Max(1e-10, sigmaDbl)));
            var coeff = NumOps.FromDouble(alphaRatio - sigmaNextDbl / Math.Max(1e-10, sigmaDbl));
            prevSample = Engine.Add(scaledSample, Engine.Multiply(dataPrediction, coeff));
        }
        else if (effectiveOrder == 2)
        {
            // Second-order DEIS
            var d0 = _modelOutputHistory[^1];
            var d1 = _modelOutputHistory[^2];

            var scaledSample = Engine.Multiply(sample, NumOps.FromDouble(sigmaNextDbl / Math.Max(1e-10, sigmaDbl)));
            var coeff0 = NumOps.FromDouble(alphaRatio * (1.0 - Math.Exp(-h)));
            var coeff1 = NumOps.FromDouble(alphaRatio * (h - 1.0 + Math.Exp(-h)) / Math.Max(1e-10, h));

            var diff = Engine.Subtract(d0, d1);
            prevSample = Engine.Add(scaledSample, Engine.Multiply(d0, coeff0));
            prevSample = Engine.Add(prevSample, Engine.Multiply(diff, coeff1));
        }
        else
        {
            // Third-order DEIS
            var d0 = _modelOutputHistory[^1];
            var d1 = _modelOutputHistory[^2];
            var d2 = _modelOutputHistory[^3];

            var scaledSample = Engine.Multiply(sample, NumOps.FromDouble(sigmaNextDbl / Math.Max(1e-10, sigmaDbl)));
            var coeff0 = NumOps.FromDouble(alphaRatio * (1.0 - Math.Exp(-h)));
            var coeff1 = NumOps.FromDouble(alphaRatio * (h - 1.0 + Math.Exp(-h)) / Math.Max(1e-10, h));
            var coeff2 = NumOps.FromDouble(alphaRatio * (0.5 * h * h - h + 1.0 - Math.Exp(-h)) / Math.Max(1e-10, h * h));

            var diff1 = Engine.Subtract(d0, d1);
            var diff2 = Engine.Add(Engine.Subtract(d0, Engine.Multiply(d1, NumOps.FromDouble(2.0))), d2);

            prevSample = Engine.Add(scaledSample, Engine.Multiply(d0, coeff0));
            prevSample = Engine.Add(prevSample, Engine.Multiply(diff1, coeff1));
            prevSample = Engine.Add(prevSample, Engine.Multiply(diff2, coeff2));
        }

        return prevSample;
    }

    /// <summary>
    /// Computes lambda = log(alpha/sigma) = -log(sigma) for the sigma parameterization.
    /// </summary>
    private static double ComputeLambda(T sigma)
    {
        double s = NumOps.ToDouble(sigma);
        return -Math.Log(Math.Max(1e-10, s));
    }

    private Vector<T> ConvertToDataPrediction(Vector<T> modelOutput, Vector<T> sample, T sigma, int timestep)
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
    /// Creates a DEIS scheduler with default Stable Diffusion settings.
    /// </summary>
    public static DEISMultistepScheduler<T> CreateDefault()
    {
        return new DEISMultistepScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion());
    }
}
