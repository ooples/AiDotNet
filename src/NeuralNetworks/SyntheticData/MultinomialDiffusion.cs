using AiDotNet.Helpers;

namespace AiDotNet.NeuralNetworks.SyntheticData;

/// <summary>
/// Implements the multinomial diffusion process for categorical features in TabDDPM.
/// </summary>
/// <remarks>
/// <para>
/// Multinomial diffusion operates on categorical (one-hot) features:
/// - <b>Forward process</b>: Gradually mixes the true category distribution toward uniform
///   q(x_t | x_0) = (1 - beta_t) * x_{t-1} + beta_t * (1/K) where K is the number of categories
/// - <b>Training</b>: The model predicts log-probabilities of the original categories
/// - <b>Loss</b>: KL divergence between predicted posterior and true posterior
/// - <b>Reverse process</b>: Sample categories from the predicted probability distribution
/// </para>
/// <para>
/// <b>For Beginners:</b> This handles the "category" part of the diffusion process.
///
/// Forward (corrupting categories):
/// - Start with a real category (e.g., color = "red")
/// - At each step, there's a small chance the category "flips" to a random one
/// - After many steps, the category is essentially random (equally likely to be any color)
///
/// Reverse (recovering categories):
/// - Start with a random category
/// - The model predicts the probability of each possible category
/// - Sample from those probabilities to get a cleaner category
/// - After many steps, you get a realistic category assignment
///
/// The noise schedule controls how quickly categories become randomized.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class MultinomialDiffusion<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly int _numTimesteps;
    private readonly Random _random;

    // Precomputed schedule values
    private readonly double[] _betas;
    private readonly double[] _alphasCumprod; // Product of (1 - beta) up to t

    /// <summary>
    /// Gets the number of diffusion timesteps.
    /// </summary>
    public int NumTimesteps => _numTimesteps;

    /// <summary>
    /// Initializes a new multinomial diffusion process.
    /// </summary>
    /// <param name="numTimesteps">Number of diffusion steps for categorical features.</param>
    /// <param name="betaStart">Starting noise rate.</param>
    /// <param name="betaEnd">Ending noise rate.</param>
    /// <param name="random">Random number generator.</param>
    public MultinomialDiffusion(int numTimesteps, double betaStart, double betaEnd, Random random)
    {
        _numTimesteps = numTimesteps;
        _random = random;

        // Linear beta schedule for multinomial diffusion
        _betas = new double[numTimesteps];
        double step = (betaEnd - betaStart) / Math.Max(numTimesteps - 1, 1);
        for (int t = 0; t < numTimesteps; t++)
        {
            _betas[t] = betaStart + step * t;
        }

        // Cumulative product of (1 - beta)
        _alphasCumprod = new double[numTimesteps];
        double cumprod = 1.0;
        for (int t = 0; t < numTimesteps; t++)
        {
            cumprod *= (1.0 - _betas[t]);
            _alphasCumprod[t] = cumprod;
        }
    }

    /// <summary>
    /// Adds categorical noise to a one-hot vector at timestep t (forward process).
    /// q(x_t | x_0) = alphaBar_t * x_0 + (1 - alphaBar_t) * (1/K)
    /// </summary>
    /// <param name="x0">Clean one-hot category vector (length = number of categories).</param>
    /// <param name="t">Timestep (0 to numTimesteps-1).</param>
    /// <returns>Noisy category probability vector at timestep t.</returns>
    public Vector<T> AddNoise(Vector<T> x0, int t)
    {
        int k = x0.Length;
        double alphaBar = _alphasCumprod[t];
        double uniform = 1.0 / k;

        var xt = new Vector<T>(k);
        for (int i = 0; i < k; i++)
        {
            double x = NumOps.ToDouble(x0[i]);
            xt[i] = NumOps.FromDouble(alphaBar * x + (1.0 - alphaBar) * uniform);
        }

        return xt;
    }

    /// <summary>
    /// Computes the KL divergence loss between predicted and true posterior.
    /// </summary>
    /// <param name="predictedLogProbs">Log-probabilities predicted by the model (length = K).</param>
    /// <param name="trueOneHot">True one-hot category vector.</param>
    /// <param name="noisyProbs">Noisy category probabilities at timestep t.</param>
    /// <param name="t">The current timestep.</param>
    /// <returns>KL divergence loss value.</returns>
    public T ComputeLoss(Vector<T> predictedLogProbs, Vector<T> trueOneHot, Vector<T> noisyProbs, int t)
    {
        int k = predictedLogProbs.Length;

        // Cross-entropy between true distribution and predicted log-probabilities
        double loss = 0;
        for (int i = 0; i < k; i++)
        {
            double target = NumOps.ToDouble(trueOneHot[i]);
            double logPred = NumOps.ToDouble(predictedLogProbs[i]);
            // Clamp for numerical stability
            logPred = Math.Max(logPred, -20.0);
            loss -= target * logPred;
        }

        return NumOps.FromDouble(loss);
    }

    /// <summary>
    /// Computes the gradient of the loss with respect to predicted log-probabilities.
    /// </summary>
    /// <param name="predictedLogProbs">Log-probabilities predicted by the model.</param>
    /// <param name="trueOneHot">True one-hot category vector.</param>
    /// <returns>Gradient vector.</returns>
    public Vector<T> ComputeLossGradient(Vector<T> predictedLogProbs, Vector<T> trueOneHot)
    {
        int k = predictedLogProbs.Length;
        var grad = new Vector<T>(k);

        // Gradient of cross-entropy: softmax(logProbs) - target
        // First compute softmax of log-probs
        double maxLogP = double.MinValue;
        for (int i = 0; i < k; i++)
        {
            double v = NumOps.ToDouble(predictedLogProbs[i]);
            if (v > maxLogP) maxLogP = v;
        }

        double sumExp = 0;
        for (int i = 0; i < k; i++)
        {
            sumExp += Math.Exp(NumOps.ToDouble(predictedLogProbs[i]) - maxLogP);
        }

        for (int i = 0; i < k; i++)
        {
            double softmax = Math.Exp(NumOps.ToDouble(predictedLogProbs[i]) - maxLogP) / Math.Max(sumExp, 1e-10);
            double target = NumOps.ToDouble(trueOneHot[i]);
            grad[i] = NumOps.FromDouble(softmax - target);
        }

        return grad;
    }

    /// <summary>
    /// Performs one reverse diffusion step for categorical features.
    /// Samples a category from the predicted probability distribution.
    /// </summary>
    /// <param name="noisyProbs">Current noisy probability vector at timestep t.</param>
    /// <param name="predictedLogProbs">Log-probabilities predicted by the model for x_0.</param>
    /// <param name="t">Current timestep.</param>
    /// <returns>Denoised probability vector at timestep t-1.</returns>
    public Vector<T> DenoisingStep(Vector<T> noisyProbs, Vector<T> predictedLogProbs, int t)
    {
        int k = noisyProbs.Length;

        // Convert log-probs to probabilities via softmax
        var probs = SoftmaxFromLogProbs(predictedLogProbs);

        if (t > 0)
        {
            double alphaBar = _alphasCumprod[t];
            double alphaBarPrev = _alphasCumprod[t - 1];

            // Compute posterior: p(x_{t-1} | x_t, x_0_pred)
            // Using Bayes' rule to combine forward process with prediction
            var posterior = new Vector<T>(k);
            double posteriorSum = 0;

            for (int i = 0; i < k; i++)
            {
                double p0 = NumOps.ToDouble(probs[i]); // predicted x_0 probability
                double uniform = 1.0 / k;

                // Forward transition probability from x_{t-1}=i to observed x_t distribution
                double forwardLikelihood = 0;
                for (int j = 0; j < k; j++)
                {
                    double xtj = NumOps.ToDouble(noisyProbs[j]);
                    double transProb = (i == j) ? (1.0 - _betas[t]) + _betas[t] / k : _betas[t] / k;
                    forwardLikelihood += xtj * Math.Log(Math.Max(transProb, 1e-10));
                }

                double posteriorVal = p0 * Math.Exp(forwardLikelihood);
                posterior[i] = NumOps.FromDouble(posteriorVal);
                posteriorSum += posteriorVal;
            }

            // Normalize posterior
            if (posteriorSum > 1e-10)
            {
                for (int i = 0; i < k; i++)
                {
                    posterior[i] = NumOps.FromDouble(NumOps.ToDouble(posterior[i]) / posteriorSum);
                }
            }
            else
            {
                // Fallback to predicted probabilities
                posterior = probs;
            }

            // Sample from posterior
            return SampleCategorical(posterior);
        }
        else
        {
            // t = 0: directly use predicted probabilities
            return SampleCategorical(probs);
        }
    }

    /// <summary>
    /// Samples a random timestep uniformly from [0, numTimesteps).
    /// </summary>
    /// <returns>A random timestep index.</returns>
    public int SampleTimestep()
    {
        return _random.Next(_numTimesteps);
    }

    #region Helpers

    private static Vector<T> SoftmaxFromLogProbs(Vector<T> logProbs)
    {
        int k = logProbs.Length;
        var result = new Vector<T>(k);

        double maxVal = double.MinValue;
        for (int i = 0; i < k; i++)
        {
            double v = NumOps.ToDouble(logProbs[i]);
            if (v > maxVal) maxVal = v;
        }

        double sumExp = 0;
        for (int i = 0; i < k; i++)
        {
            sumExp += Math.Exp(NumOps.ToDouble(logProbs[i]) - maxVal);
        }

        for (int i = 0; i < k; i++)
        {
            double expVal = Math.Exp(NumOps.ToDouble(logProbs[i]) - maxVal);
            result[i] = NumOps.FromDouble(expVal / Math.Max(sumExp, 1e-10));
        }

        return result;
    }

    private Vector<T> SampleCategorical(Vector<T> probs)
    {
        int k = probs.Length;
        var result = new Vector<T>(k);

        // Sample from categorical distribution using CDF
        double u = _random.NextDouble();
        double cumsum = 0;
        int sampled = k - 1; // Default to last category

        for (int i = 0; i < k; i++)
        {
            cumsum += NumOps.ToDouble(probs[i]);
            if (u <= cumsum)
            {
                sampled = i;
                break;
            }
        }

        // Return one-hot
        result[sampled] = NumOps.One;
        return result;
    }

    #endregion
}
