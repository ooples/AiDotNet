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
public class GaussianDiffusion<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly int _numTimesteps;
    private readonly Random _random;

    // Precomputed noise schedule values
    private readonly double[] _betas;
    private readonly double[] _alphas;
    private readonly double[] _alphasCumprod;
    private readonly double[] _sqrtAlphasCumprod;
    private readonly double[] _sqrtOneMinusAlphasCumprod;
    private readonly double[] _sqrtRecipAlphas;
    private readonly double[] _posteriorVariance;

    /// <summary>
    /// Gets the number of diffusion timesteps.
    /// </summary>
    public int NumTimesteps => _numTimesteps;

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
    {
        _numTimesteps = numTimesteps;
        _random = random;

        // Compute beta schedule
        _betas = ComputeBetaSchedule(numTimesteps, betaStart, betaEnd, schedule);

        // Compute derived values
        _alphas = new double[numTimesteps];
        _alphasCumprod = new double[numTimesteps];
        _sqrtAlphasCumprod = new double[numTimesteps];
        _sqrtOneMinusAlphasCumprod = new double[numTimesteps];
        _sqrtRecipAlphas = new double[numTimesteps];
        _posteriorVariance = new double[numTimesteps];

        double alphaCumprod = 1.0;
        for (int t = 0; t < numTimesteps; t++)
        {
            _alphas[t] = 1.0 - _betas[t];
            alphaCumprod *= _alphas[t];
            _alphasCumprod[t] = alphaCumprod;
            _sqrtAlphasCumprod[t] = Math.Sqrt(alphaCumprod);
            _sqrtOneMinusAlphasCumprod[t] = Math.Sqrt(1.0 - alphaCumprod);
            _sqrtRecipAlphas[t] = 1.0 / Math.Sqrt(_alphas[t]);

            if (t > 0)
            {
                double alphaCumprodPrev = _alphasCumprod[t - 1];
                _posteriorVariance[t] = _betas[t] * (1.0 - alphaCumprodPrev) / (1.0 - alphaCumprod);
            }
            else
            {
                _posteriorVariance[t] = _betas[t];
            }
        }
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

        double sqrtAlphaBar = _sqrtAlphasCumprod[t];
        double sqrtOneMinusAlphaBar = _sqrtOneMinusAlphasCumprod[t];

        var xt = new Vector<T>(len);
        for (int i = 0; i < len; i++)
        {
            double x = NumOps.ToDouble(x0[i]);
            double n = NumOps.ToDouble(noise[i]);
            xt[i] = NumOps.FromDouble(sqrtAlphaBar * x + sqrtOneMinusAlphaBar * n);
        }

        return (xt, noise);
    }

    /// <summary>
    /// Computes the MSE loss between predicted and actual noise.
    /// </summary>
    /// <param name="predictedNoise">Noise predicted by the denoiser model.</param>
    /// <param name="actualNoise">The actual noise that was added in the forward process.</param>
    /// <returns>The MSE loss value.</returns>
    public T ComputeLoss(Vector<T> predictedNoise, Vector<T> actualNoise)
    {
        double loss = 0;
        int len = Math.Min(predictedNoise.Length, actualNoise.Length);

        for (int i = 0; i < len; i++)
        {
            double diff = NumOps.ToDouble(predictedNoise[i]) - NumOps.ToDouble(actualNoise[i]);
            loss += diff * diff;
        }

        return NumOps.FromDouble(loss / Math.Max(len, 1));
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
        var grad = new Vector<T>(predictedNoise.Length);

        double scale = 2.0 / Math.Max(len, 1);
        for (int i = 0; i < len; i++)
        {
            double diff = NumOps.ToDouble(predictedNoise[i]) - NumOps.ToDouble(actualNoise[i]);
            grad[i] = NumOps.FromDouble(scale * diff);
        }

        return grad;
    }

    /// <summary>
    /// Performs one reverse diffusion step (denoising): x_{t-1} from x_t.
    /// </summary>
    /// <param name="xt">Noisy data at timestep t.</param>
    /// <param name="predictedNoise">Noise predicted by the model.</param>
    /// <param name="t">Current timestep.</param>
    /// <returns>Denoised data at timestep t-1.</returns>
    public Vector<T> DenoisingStep(Vector<T> xt, Vector<T> predictedNoise, int t)
    {
        int len = xt.Length;
        var result = new Vector<T>(len);

        double sqrtRecipAlpha = _sqrtRecipAlphas[t];
        double betaOverSqrt = _betas[t] / _sqrtOneMinusAlphasCumprod[t];

        for (int i = 0; i < len; i++)
        {
            double x = NumOps.ToDouble(xt[i]);
            double eps = i < predictedNoise.Length ? NumOps.ToDouble(predictedNoise[i]) : 0;
            double meanPred = sqrtRecipAlpha * (x - betaOverSqrt * eps);

            if (t > 0)
            {
                // Add noise scaled by posterior variance
                double sigma = Math.Sqrt(_posteriorVariance[t]);
                double z = NumOps.ToDouble(SampleStandardNormal());
                result[i] = NumOps.FromDouble(meanPred + sigma * z);
            }
            else
            {
                result[i] = NumOps.FromDouble(meanPred);
            }
        }

        return result;
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

    private static double[] ComputeBetaSchedule(int numTimesteps, double betaStart, double betaEnd, string schedule)
    {
        var betas = new double[numTimesteps];

        if (string.Equals(schedule, "cosine", StringComparison.OrdinalIgnoreCase))
        {
            // Cosine schedule from Improved DDPM paper
            double s = 0.008;
            for (int t = 0; t < numTimesteps; t++)
            {
                double t1 = (double)t / numTimesteps;
                double t2 = (double)(t + 1) / numTimesteps;
                double alphaBarT1 = Math.Pow(Math.Cos((t1 + s) / (1.0 + s) * Math.PI / 2.0), 2);
                double alphaBarT2 = Math.Pow(Math.Cos((t2 + s) / (1.0 + s) * Math.PI / 2.0), 2);
                betas[t] = Math.Min(1.0 - alphaBarT2 / alphaBarT1, 0.999);
            }
        }
        else
        {
            // Linear schedule
            double step = (betaEnd - betaStart) / Math.Max(numTimesteps - 1, 1);
            for (int t = 0; t < numTimesteps; t++)
            {
                betas[t] = betaStart + step * t;
            }
        }

        return betas;
    }

    #endregion

    #region Helpers

    private T SampleStandardNormal()
    {
        double u1 = 1.0 - _random.NextDouble();
        double u2 = _random.NextDouble();
        double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        return NumOps.FromDouble(z);
    }

    #endregion
}
