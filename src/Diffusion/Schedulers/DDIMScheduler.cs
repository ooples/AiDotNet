namespace AiDotNet.Diffusion.Schedulers;

/// <summary>
/// DDIM (Denoising Diffusion Implicit Models) scheduler implementation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// DDIM is a faster variant of DDPM that can achieve similar quality with far fewer
/// denoising steps. While DDPM requires many steps (often 1000), DDIM can achieve
/// similar quality with 50 or fewer steps by using a different mathematical formulation.
/// </para>
/// <para>
/// <b>For Beginners:</b> DDIM is like a shortcut for removing noise from images.
///
/// Imagine you have a very blurry photo and need to make it clear:
/// - DDPM (original method): Take 1000 tiny steps to slowly reveal the image
/// - DDIM (this method): Take 50 larger steps to reveal the image faster
///
/// The magic is the "eta" parameter:
/// - eta=0: Deterministic - same input always produces same output (faster, consistent)
/// - eta=1: Stochastic - adds randomness like DDPM (slower, more variety)
/// - eta between 0-1: Mix of both behaviors
///
/// Key advantages of DDIM:
/// - Much faster generation (10-50x fewer steps needed)
/// - Deterministic option allows reproducible results
/// - Can interpolate smoothly between images (useful for animations)
/// </para>
/// <para>
/// <b>Reference:</b> "Denoising Diffusion Implicit Models" by Song et al., 2020
/// </para>
/// </remarks>
public sealed class DDIMScheduler<T> : NoiseSchedulerBase<T>
{
    /// <summary>
    /// Initializes a new instance of the DDIM scheduler.
    /// </summary>
    /// <param name="config">Configuration for the scheduler including beta schedule parameters.</param>
    /// <exception cref="ArgumentNullException">Thrown when config is null.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Create a DDIM scheduler with custom settings, or use
    /// <see cref="SchedulerConfig{T}.CreateDefault"/> for standard DDPM defaults.
    /// </para>
    /// <example>
    /// <code>
    /// // Create with default settings
    /// var config = SchedulerConfig&lt;double&gt;.CreateDefault();
    /// var scheduler = new DDIMScheduler&lt;double&gt;(config);
    ///
    /// // Set up for 50 inference steps
    /// scheduler.SetTimesteps(50);
    /// </code>
    /// </example>
    /// </remarks>
    public DDIMScheduler(SchedulerConfig<T> config) : base(config)
    {
    }

    /// <summary>
    /// Performs one DDIM denoising step.
    /// </summary>
    /// <param name="modelOutput">The model's noise prediction (epsilon).</param>
    /// <param name="timestep">The current timestep in the diffusion process.</param>
    /// <param name="sample">The current noisy sample.</param>
    /// <param name="eta">
    /// Stochasticity parameter:
    /// - 0 = fully deterministic (pure DDIM)
    /// - 1 = fully stochastic (equivalent to DDPM)
    /// - Values in between interpolate the behavior
    /// </param>
    /// <param name="noise">
    /// Optional noise for stochastic sampling. Required when eta > 0 for true stochastic behavior.
    /// If null and eta > 0, falls back to deterministic (zero noise).
    /// </param>
    /// <returns>The denoised sample for the previous timestep.</returns>
    /// <exception cref="ArgumentNullException">Thrown when modelOutput or sample is null.</exception>
    /// <exception cref="ArgumentException">Thrown when modelOutput and sample have different lengths.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when timestep is out of range.</exception>
    /// <remarks>
    /// <para>
    /// The DDIM step formula (simplified):
    /// 1. Predict original sample: x_0 = (x_t - sqrt(1-alpha_cumprod) * eps) / sqrt(alpha_cumprod)
    /// 2. Compute "direction pointing to x_t": d = sqrt(1-alpha_prev - sigma^2) * eps
    /// 3. Compute previous sample: x_{t-1} = sqrt(alpha_prev) * x_0 + d + sigma * noise
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This function takes your current noisy image and the model's
    /// guess of what noise is in it, then removes some of that noise to get a cleaner image.
    /// The eta parameter controls whether this removal is exact (eta=0) or has some randomness (eta>0).
    /// </para>
    /// </remarks>
    public override Vector<T> Step(Vector<T> modelOutput, int timestep, Vector<T> sample, T eta, Vector<T>? noise = null)
    {
        ValidateStepParameters(modelOutput, sample, timestep);

        int t = timestep;
        // Calculate step size for non-uniform inference schedules (e.g., 50 steps on 1000 training steps)
        int stepSize = Timesteps.Length > 0 ? Config.TrainTimesteps / Timesteps.Length : 1;
        int prevT = Math.Max(t - stepSize, 0);

        // Get alpha cumulative products for current and previous timesteps
        T alphaCumprod = AlphasCumulativeProduct[t];
        T alphaCumprodPrev = AlphasCumulativeProduct[prevT];

        // Compute useful quantities
        T sqrtAlphaCumprod = NumOps.Sqrt(alphaCumprod);
        T oneMinusAlphaCumprod = NumOps.Subtract(NumOps.One, alphaCumprod);
        T sqrtOneMinusAlphaCumprod = NumOps.Sqrt(oneMinusAlphaCumprod);

        // Step 1: Predict the original sample (x_0) from noise prediction (VECTORIZED)
        // x_0 = (x_t - sqrt(1 - alpha_cumprod) * epsilon) / sqrt(alpha_cumprod)
        var noiseTerm = Engine.Multiply(modelOutput, sqrtOneMinusAlphaCumprod);
        var numerator = Engine.Subtract(sample, noiseTerm);
        var predictedOriginal = Engine.Divide(numerator, sqrtAlphaCumprod);

        // Optionally clip the predicted original sample
        predictedOriginal = ClipSampleIfNeeded(predictedOriginal);

        // Step 2: Compute DDIM variance (sigma)
        // sigma = eta * sqrt((1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev))
        T sigma = NumOps.Zero;
        if (NumOps.GreaterThan(eta, NumOps.Zero))
        {
            var oneMinusAlphaCumprodPrev = NumOps.Subtract(NumOps.One, alphaCumprodPrev);
            var ratio = NumOps.Divide(oneMinusAlphaCumprodPrev, oneMinusAlphaCumprod);
            var alphaRatio = NumOps.Divide(alphaCumprod, alphaCumprodPrev);
            var frac = NumOps.Subtract(NumOps.One, alphaRatio);
            var varianceInside = NumOps.Multiply(ratio, frac);

            // Ensure non-negative before sqrt (numerical stability)
            if (NumOps.GreaterThan(varianceInside, NumOps.Zero))
            {
                sigma = NumOps.Multiply(eta, NumOps.Sqrt(varianceInside));
            }
        }

        // Step 3: Compute coefficients for the DDIM formula
        T sqrtAlphaCumprodPrev = NumOps.Sqrt(alphaCumprodPrev);
        T sigmaSq = NumOps.Multiply(sigma, sigma);

        // coeffEps = sqrt(1 - alpha_prev - sigma^2)
        var coeffEpsInside = NumOps.Subtract(NumOps.Subtract(NumOps.One, alphaCumprodPrev), sigmaSq);

        // Numerical stability: ensure non-negative
        T coeffEps;
        if (NumOps.GreaterThan(coeffEpsInside, NumOps.Zero))
        {
            coeffEps = NumOps.Sqrt(coeffEpsInside);
        }
        else
        {
            // If sigma is too large, fall back to deterministic
            coeffEps = NumOps.Sqrt(NumOps.Subtract(NumOps.One, alphaCumprodPrev));
            sigma = NumOps.Zero;
        }

        // Step 4: Compute the previous sample (VECTORIZED)
        // x_{t-1} = sqrt(alpha_prev) * x_0 + coeffEps * epsilon + sigma * noise
        var originalTerm = Engine.Multiply(predictedOriginal, sqrtAlphaCumprodPrev);
        var epsTerm = Engine.Multiply(modelOutput, coeffEps);
        var prevSample = Engine.Add(originalTerm, epsTerm);

        // Add noise term for stochastic sampling
        if (NumOps.GreaterThan(sigma, NumOps.Zero) && noise != null)
        {
            var noisyTerm = Engine.Multiply(noise, sigma);
            prevSample = Engine.Add(prevSample, noisyTerm);
        }

        return prevSample;
    }
}

