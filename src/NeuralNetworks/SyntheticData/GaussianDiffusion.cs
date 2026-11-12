using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;

namespace AiDotNet.NeuralNetworks.SyntheticData;

/// <summary>
/// Implements the Gaussian diffusion process for continuous/numerical features in TabDDPM.
/// </summary>
/// <remarks>
/// <para>
/// The Gaussian diffusion process operates on continuous features:
/// - <b>Forward process</b> (add noise): q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
/// - <b>Training</b>: The model learns to predict the noise epsilon that was added at timestep t
/// - <b>Loss</b>: MSE between predicted and actual noise
/// - <b>Reverse process</b> (denoise): x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1-alpha_bar_t)) * predicted_noise) + sigma_t * z
///   with sigma_t^2 = beta_t by default, as in the TabDDPM reference code (see <see cref="ReverseDiffusionVariance"/>).
/// </para>
/// <para>
/// <b>For Beginners:</b> This handles the "numbers" part of the diffusion process.
///
/// Forward (adding noise to numbers):
/// - Start with a real number (e.g., salary = $50,000)
/// - At each step, mix in a little bit of random Gaussian noise
/// - After 1000 steps, the number is pure noise (could be anything)
///
/// Reverse (removing noise from numbers):
/// - Start with a random number
/// - The model predicts what noise was added
/// - Remove that noise to get slightly cleaner number
/// - After 1000 removals, you have a realistic salary value
///
/// The math ensures that each step is a small, reversible change.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ComponentType(ComponentType.Optimizer)]
[PipelineStage(PipelineStage.Training)]
public class GaussianDiffusion<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly int _numTimesteps;
    private readonly Random _random;
    private readonly ReverseDiffusionVariance _varianceType;

    // Precomputed noise schedule values, one entry per timestep.
    private readonly Vector<T> _betas;
    private readonly Vector<T> _sqrtAlphasCumprod;
    private readonly Vector<T> _sqrtOneMinusAlphasCumprod;
    private readonly Vector<T> _sqrtRecipAlphas;
    private readonly Vector<T> _posteriorVariance;

    /// <summary>
    /// Gets the number of diffusion timesteps.
    /// </summary>
    public int NumTimesteps => _numTimesteps;

    /// <summary>
    /// Gets the reverse-step variance in use.
    /// </summary>
    public ReverseDiffusionVariance VarianceType => _varianceType;

    /// <summary>
    /// Initializes a new Gaussian diffusion process.
    /// </summary>
    /// <param name="numTimesteps">Number of diffusion steps.</param>
    /// <param name="betaStart">Starting beta value.</param>
    /// <param name="betaEnd">Ending beta value.</param>
    /// <param name="schedule">Beta schedule type: "linear" or "cosine".</param>
    /// <param name="random">Random number generator.</param>
    public GaussianDiffusion(int numTimesteps, double betaStart, double betaEnd,
        string schedule, Random random)
        : this(numTimesteps, betaStart, betaEnd,
            string.Equals(schedule, "cosine", StringComparison.OrdinalIgnoreCase) ? BetaSchedule.SquaredCosine : BetaSchedule.Linear,
            ReverseDiffusionVariance.FixedLarge, random)
    {
    }

    /// <summary>
    /// Initializes a new Gaussian diffusion process.
    /// </summary>
    /// <param name="numTimesteps">Number of diffusion steps.</param>
    /// <param name="betaStart">Starting beta value (linear schedule).</param>
    /// <param name="betaEnd">Ending beta value (linear schedule).</param>
    /// <param name="schedule">Beta schedule: <see cref="BetaSchedule.SquaredCosine"/> (Nichol and Dhariwal 2021);
    /// any other value uses the linear schedule.</param>
    /// <param name="varianceType">Reverse-step variance; TabDDPM uses <see cref="ReverseDiffusionVariance.FixedLarge"/>.</param>
    /// <param name="random">Random number generator.</param>
    public GaussianDiffusion(int numTimesteps, double betaStart, double betaEnd,
        BetaSchedule schedule, ReverseDiffusionVariance varianceType, Random random)
    {
        _numTimesteps = numTimesteps;
        _random = random;
        _varianceType = varianceType;

        var engine = AiDotNetEngine.Current;
        T one = NumOps.One;

        var betas = ComputeBetaSchedule(numTimesteps, betaStart, betaEnd, schedule);
        var alphas = engine.ScalarMinusTensor(one, betas);
        var alphasCumprod = engine.TensorCumProd(alphas, 0);
        var oneMinusCumprod = engine.ScalarMinusTensor(one, alphasCumprod);

        // alpha_bar_{t-1}, with alpha_bar_{-1} = 1. The posterior variance at t = 0 is never used:
        // the last reverse step adds no noise.
        var alphasCumprodPrev = numTimesteps > 1
            ? engine.TensorConcatenate(new[]
            {
                Tensor<T>.CreateDefault(new[] { 1 }, one),
                engine.TensorSlice(alphasCumprod, new[] { 0 }, new[] { numTimesteps - 1 })
            }, 0)
            : Tensor<T>.CreateDefault(new[] { numTimesteps }, one);

        _betas = betas.ToVector();
        _sqrtAlphasCumprod = engine.TensorSqrt(alphasCumprod).ToVector();
        _sqrtOneMinusAlphasCumprod = engine.TensorSqrt(oneMinusCumprod).ToVector();
        _sqrtRecipAlphas = engine.TensorRsqrt(alphas).ToVector();
        _posteriorVariance = engine.TensorDivide(
            engine.TensorMultiply(betas, engine.ScalarMinusTensor(one, alphasCumprodPrev)),
            oneMinusCumprod).ToVector();
    }

    /// <summary>
    /// Adds noise to clean data at a given timestep (forward process).
    /// q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
    /// </summary>
    /// <param name="x0">Clean data vector (numerical features only).</param>
    /// <param name="t">Timestep (0 to numTimesteps-1).</param>
    /// <param name="noise">Optional pre-sampled noise. If null, samples fresh noise.</param>
    /// <returns>Noisy data vector x_t and the noise that was added.</returns>
    public (Vector<T> NoisyData, Vector<T> Noise) AddNoise(Vector<T> x0, int t, Vector<T>? noise = null)
    {
        int len = x0.Length;

        if (noise is null)
        {
            noise = new Vector<T>(len);
            for (int i = 0; i < len; i++)
            {
                noise[i] = SampleStandardNormal();
            }
        }

        var engine = AiDotNetEngine.Current;
        var xt = engine.TensorAdd(
            engine.TensorMultiplyScalar(Tensor<T>.FromVector(x0), _sqrtAlphasCumprod[t]),
            engine.TensorMultiplyScalar(FitLength(noise, len), _sqrtOneMinusAlphasCumprod[t]));

        return (xt.ToVector(), noise);
    }

    /// <summary>
    /// Computes the MSE loss between predicted and actual noise.
    /// </summary>
    /// <param name="predictedNoise">Noise predicted by the denoiser model.</param>
    /// <param name="actualNoise">The actual noise that was added in the forward process.</param>
    /// <returns>The MSE loss value.</returns>
    public T ComputeLoss(Vector<T> predictedNoise, Vector<T> actualNoise)
    {
        int len = Math.Min(predictedNoise.Length, actualNoise.Length);
        if (len == 0)
        {
            return NumOps.Zero;
        }

        var engine = AiDotNetEngine.Current;
        var diff = engine.TensorSubtract(FitLength(predictedNoise, len), FitLength(actualNoise, len));
        return engine.TensorMean(engine.TensorSquare(diff));
    }

    /// <summary>
    /// Computes the gradient of the MSE loss with respect to predicted noise.
    /// </summary>
    /// <param name="predictedNoise">Noise predicted by the denoiser model.</param>
    /// <param name="actualNoise">The actual noise that was added.</param>
    /// <returns>The gradient vector.</returns>
    public Vector<T> ComputeLossGradient(Vector<T> predictedNoise, Vector<T> actualNoise)
    {
        int len = Math.Min(predictedNoise.Length, actualNoise.Length);
        if (len == 0)
        {
            return new Vector<T>(predictedNoise.Length);
        }

        var engine = AiDotNetEngine.Current;
        T scale = NumOps.FromDouble(2.0 / len);
        var diff = engine.TensorSubtract(FitLength(predictedNoise, len), FitLength(actualNoise, len));

        // Entries past the shorter vector have no target and get a zero gradient.
        return FitLength(engine.TensorMultiplyScalar(diff, scale).ToVector(), predictedNoise.Length).ToVector();
    }

    /// <summary>
    /// Performs one reverse diffusion step (denoising): x_{t-1} from x_t.
    /// </summary>
    /// <param name="xt">Noisy data at timestep t.</param>
    /// <param name="predictedNoise">Noise predicted by the model.</param>
    /// <param name="t">Current timestep.</param>
    /// <returns>Denoised data at timestep t-1.</returns>
    /// <remarks>
    /// mean = (1/sqrt(alpha_t)) (x_t - beta_t / sqrt(1 - alpha_bar_t) eps). For t &gt; 0 the result is
    /// mean + sigma_t z, with sigma_t^2 chosen by <see cref="VarianceType"/> (beta_t by default, as in
    /// TabDDPM's <c>gaussian_p_mean_variance</c>). Missing entries of <paramref name="predictedNoise"/> count as 0.
    /// </remarks>
    public Vector<T> DenoisingStep(Vector<T> xt, Vector<T> predictedNoise, int t)
    {
        int len = xt.Length;
        var engine = AiDotNetEngine.Current;

        T betaOverSqrt = NumOps.Divide(_betas[t], _sqrtOneMinusAlphasCumprod[t]);
        var eps = FitLength(predictedNoise, len);
        var meanPred = engine.TensorMultiplyScalar(
            engine.TensorSubtract(Tensor<T>.FromVector(xt), engine.TensorMultiplyScalar(eps, betaOverSqrt)),
            _sqrtRecipAlphas[t]);

        if (t <= 0)
        {
            return meanPred.ToVector();
        }

        // One standard normal per element, drawn in master's order.
        var z = new Vector<T>(len);
        for (int i = 0; i < len; i++)
        {
            z[i] = SampleStandardNormal();
        }

        T variance = _varianceType == ReverseDiffusionVariance.FixedSmall ? _posteriorVariance[t] : _betas[t];
        T sigma = NumOps.Sqrt(variance);
        return engine.TensorAdd(meanPred, engine.TensorMultiplyScalar(Tensor<T>.FromVector(z), sigma)).ToVector();
    }

    /// <summary>
    /// Samples a random timestep uniformly from [0, numTimesteps).
    /// </summary>
    /// <returns>A random timestep index.</returns>
    public int SampleTimestep()
    {
        return _random.Next(_numTimesteps);
    }

    #region Beta Schedule

    private static Tensor<T> ComputeBetaSchedule(int numTimesteps, double betaStart, double betaEnd, BetaSchedule schedule)
    {
        var engine = AiDotNetEngine.Current;
        var steps = engine.TensorArange<T>(0.0, numTimesteps, 1.0);

        if (schedule == BetaSchedule.SquaredCosine)
        {
            // Cosine schedule from Improved DDPM (Nichol and Dhariwal 2021):
            // beta_t = min(1 - alpha_bar((t+1)/T) / alpha_bar(t/T), 0.999),
            // alpha_bar(u) = cos((u + s) / (1 + s) * pi / 2)^2 with s = 0.008.
            double s = 0.008;
            T invSteps = NumOps.FromDouble(1.0 / numTimesteps);
            var t1 = engine.TensorMultiplyScalar(steps, invSteps);
            var t2 = engine.TensorMultiplyScalar(engine.TensorAddScalar(steps, NumOps.One), invSteps);
            var alphaBarT1 = CosineAlphaBar(engine, t1, s);
            var alphaBarT2 = CosineAlphaBar(engine, t2, s);
            var betas = engine.ScalarMinusTensor(NumOps.One, engine.TensorDivide(alphaBarT2, alphaBarT1));
            return engine.TensorClampMax(betas, NumOps.FromDouble(0.999));
        }

        // Linear schedule: beta_t = betaStart + t * (betaEnd - betaStart) / (T - 1).
        T step = NumOps.FromDouble((betaEnd - betaStart) / Math.Max(numTimesteps - 1, 1));
        return engine.TensorAddScalar(engine.TensorMultiplyScalar(steps, step), NumOps.FromDouble(betaStart));
    }

    private static Tensor<T> CosineAlphaBar(IEngine engine, Tensor<T> u, double s)
    {
        T scale = NumOps.FromDouble(Math.PI / 2.0 / (1.0 + s));
        var angle = engine.TensorMultiplyScalar(engine.TensorAddScalar(u, NumOps.FromDouble(s)), scale);
        return engine.TensorSquare(engine.TensorCos(angle));
    }

    #endregion

    #region Helpers

    /// <summary>
    /// <paramref name="v"/> as a rank-1 tensor of exactly <paramref name="length"/> entries:
    /// truncated when longer, zero-padded when shorter.
    /// </summary>
    private static Tensor<T> FitLength(Vector<T> v, int length)
    {
        var t = Tensor<T>.FromVector(v);
        if (v.Length == length)
        {
            return t;
        }

        var engine = AiDotNetEngine.Current;
        if (v.Length > length)
        {
            return engine.TensorSlice(t, new[] { 0 }, new[] { length });
        }

        return engine.TensorConcatenate(new[] { t, new Tensor<T>(new[] { length - v.Length }) }, 0);
    }

    private T SampleStandardNormal()
    {
        double u1 = 1.0 - _random.NextDouble();
        double u2 = _random.NextDouble();
        double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        return NumOps.FromDouble(z);
    }

    #endregion
}
