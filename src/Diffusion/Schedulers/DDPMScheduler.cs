using AiDotNet.Enums;

namespace AiDotNet.Diffusion;

/// <summary>
/// DDPM (Denoising Diffusion Probabilistic Models) scheduler implementation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// DDPM is the original diffusion model scheduler that uses a Markov chain of Gaussian
/// transitions to gradually denoise samples. It requires many steps (typically 1000) but
/// produces high-quality results with well-understood theoretical properties.
/// </para>
/// <para>
/// <b>For Beginners:</b> DDPM is the foundational method for diffusion image generation.
///
/// Think of it like restoring a photograph that has been progressively damaged:
/// - Training: Learn how each level of damage looks
/// - Generation: Start with pure static and remove damage one tiny step at a time
///
/// Key characteristics:
/// - Stochastic: Each step adds a small amount of random noise
/// - Many steps needed: Typically 1000 steps for good quality
/// - Well-studied: Strong theoretical guarantees on output quality
/// - Variance can be learned or fixed
///
/// The step formula:
/// x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1-alpha_cumprod_t)) * eps) + sigma_t * z
///
/// Where z is random noise and sigma_t controls the stochasticity.
/// </para>
/// <para>
/// <b>Reference:</b> Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020
/// </para>
/// </remarks>
public sealed class DDPMScheduler<T> : NoiseSchedulerBase<T>
{
    /// <summary>
    /// Initializes a new instance of the DDPM scheduler.
    /// </summary>
    /// <param name="config">Configuration for the scheduler including beta schedule parameters.</param>
    /// <exception cref="ArgumentNullException">Thrown when config is null.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Create a DDPM scheduler with custom settings, or use
    /// <see cref="SchedulerConfig{T}.CreateDefault"/> for standard DDPM defaults (1000 steps).
    /// </para>
    /// <example>
    /// <code>
    /// var config = SchedulerConfig&lt;double&gt;.CreateDefault();
    /// var scheduler = new DDPMScheduler&lt;double&gt;(config);
    /// scheduler.SetTimesteps(1000);
    /// </code>
    /// </example>
    /// </remarks>
    public DDPMScheduler(SchedulerConfig<T> config) : base(config)
    {
    }

    /// <summary>
    /// Performs one DDPM denoising step.
    /// </summary>
    /// <param name="modelOutput">The model's noise prediction (epsilon) or sample prediction.</param>
    /// <param name="timestep">The current timestep in the diffusion process.</param>
    /// <param name="sample">The current noisy sample.</param>
    /// <param name="eta">Not directly used in DDPM (always stochastic). Included for interface compatibility.</param>
    /// <param name="noise">
    /// Random noise for stochastic sampling. Required for all timesteps except t=0.
    /// If null at non-zero timesteps, falls back to deterministic (no noise added).
    /// </param>
    /// <returns>The denoised sample for the previous timestep.</returns>
    /// <exception cref="ArgumentNullException">Thrown when modelOutput or sample is null.</exception>
    /// <exception cref="ArgumentException">Thrown when modelOutput and sample have different lengths.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when timestep is out of range.</exception>
    /// <remarks>
    /// <para>
    /// The DDPM reverse step:
    /// 1. Predict x_0 from model output (depends on prediction type)
    /// 2. Compute mean: mu = (1/sqrt(alpha_t)) * (x_t - (beta_t/sqrt(1-alpha_bar_t)) * eps)
    /// 3. Add noise: x_{t-1} = mu + sigma_t * z (where z ~ N(0,I))
    /// </para>
    /// </remarks>
    public override Vector<T> Step(Vector<T> modelOutput, int timestep, Vector<T> sample, T eta, Vector<T>? noise = null)
    {
        ValidateStepParameters(modelOutput, sample, timestep);

        int t = timestep;

        // Get alpha and beta values for current timestep
        T alphaCumprod = AlphasCumulativeProduct[t];
        T beta = Betas[t];
        T alpha = Alphas[t];

        // Get alpha_cumprod for previous timestep
        int stepSize = Timesteps.Length > 0 ? Config.TrainTimesteps / Timesteps.Length : 1;
        int prevT = Math.Max(t - stepSize, 0);
        T alphaCumprodPrev = t > 0 ? AlphasCumulativeProduct[prevT] : NumOps.One;

        // Compute useful quantities
        T sqrtAlphaCumprod = NumOps.Sqrt(alphaCumprod);
        T sqrtOneMinusAlphaCumprod = NumOps.Sqrt(NumOps.Subtract(NumOps.One, alphaCumprod));
        T sqrtRecipAlpha = NumOps.Sqrt(NumOps.Divide(NumOps.One, alpha));

        // Step 1: Predict x_0 based on prediction type
        Vector<T> predictedOriginal;
        Vector<T> epsilon;

        switch (Config.PredictionType)
        {
            case DiffusionPredictionType.Sample:
                // Model directly predicts x_0
                predictedOriginal = modelOutput;
                // Back-compute epsilon: eps = (x_t - sqrt(alpha_cumprod) * x_0) / sqrt(1-alpha_cumprod)
                var scaledPred = Engine.Multiply(modelOutput, sqrtAlphaCumprod);
                var diff = Engine.Subtract(sample, scaledPred);
                epsilon = Engine.Divide(diff, sqrtOneMinusAlphaCumprod);
                break;

            case DiffusionPredictionType.VPrediction:
                // v = sqrt(alpha_cumprod) * eps - sqrt(1-alpha_cumprod) * x_0
                // x_0 = sqrt(alpha_cumprod) * x_t - sqrt(1-alpha_cumprod) * v
                var x0Term = Engine.Multiply(sample, sqrtAlphaCumprod);
                var vTerm = Engine.Multiply(modelOutput, sqrtOneMinusAlphaCumprod);
                predictedOriginal = Engine.Subtract(x0Term, vTerm);
                // eps = sqrt(alpha_cumprod) * v + sqrt(1-alpha_cumprod) * x_t
                var epsTermA = Engine.Multiply(modelOutput, sqrtAlphaCumprod);
                var epsTermB = Engine.Multiply(sample, sqrtOneMinusAlphaCumprod);
                epsilon = Engine.Add(epsTermA, epsTermB);
                break;

            default: // Epsilon prediction
                epsilon = modelOutput;
                // x_0 = (x_t - sqrt(1-alpha_cumprod) * eps) / sqrt(alpha_cumprod)
                var noiseTerm = Engine.Multiply(modelOutput, sqrtOneMinusAlphaCumprod);
                var numerator = Engine.Subtract(sample, noiseTerm);
                predictedOriginal = Engine.Divide(numerator, sqrtAlphaCumprod);
                break;
        }

        // Optionally clip predicted original
        predictedOriginal = ClipSampleIfNeeded(predictedOriginal);

        // Step 2: Compute the mean of p(x_{t-1} | x_t)
        // mu = (sqrt(alpha_cumprod_prev) * beta_t / (1-alpha_cumprod_t)) * x_0
        //    + (sqrt(alpha_t) * (1-alpha_cumprod_prev) / (1-alpha_cumprod_t)) * x_t
        T oneMinusAlphaCumprod = NumOps.Subtract(NumOps.One, alphaCumprod);
        T oneMinusAlphaCumprodPrev = NumOps.Subtract(NumOps.One, alphaCumprodPrev);

        T coeff1 = NumOps.Divide(
            NumOps.Multiply(NumOps.Sqrt(alphaCumprodPrev), beta),
            oneMinusAlphaCumprod);
        T coeff2 = NumOps.Divide(
            NumOps.Multiply(NumOps.Sqrt(alpha), oneMinusAlphaCumprodPrev),
            oneMinusAlphaCumprod);

        var meanTerm1 = Engine.Multiply(predictedOriginal, coeff1);
        var meanTerm2 = Engine.Multiply(sample, coeff2);
        var prevMean = Engine.Add(meanTerm1, meanTerm2);

        // Step 3: Compute variance and add noise (except at t=0)
        if (t > 0 && noise != null)
        {
            // Posterior variance: beta_tilde = beta_t * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t)
            T variance = NumOps.Divide(
                NumOps.Multiply(beta, oneMinusAlphaCumprodPrev),
                oneMinusAlphaCumprod);

            // Clamp for numerical stability
            T minVariance = NumOps.FromDouble(1e-20);
            if (NumOps.LessThan(variance, minVariance))
                variance = minVariance;

            T sigma = NumOps.Sqrt(variance);
            var noisyTerm = Engine.Multiply(noise, sigma);
            prevMean = Engine.Add(prevMean, noisyTerm);
        }

        return prevMean;
    }
}
