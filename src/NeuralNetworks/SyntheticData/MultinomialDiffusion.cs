using AiDotNet.Attributes;
using AiDotNet.Enums;
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
[ComponentType(ComponentType.Optimizer)]
[PipelineStage(PipelineStage.Training)]
public class MultinomialDiffusion<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    // Floor applied before taking a logarithm of a probability.
    private const double LogFloor = 1e-30;

    private readonly int _numTimesteps;
    private readonly Random _random;

    // Precomputed schedule values, one entry per timestep.
    private readonly Vector<T> _betas;
    private readonly Vector<T> _alphasCumprod; // Product of (1 - beta) up to t

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

        var engine = AiDotNetEngine.Current;

        // Linear beta schedule: beta_t = betaStart + t * (betaEnd - betaStart) / (T - 1).
        T step = NumOps.FromDouble((betaEnd - betaStart) / Math.Max(numTimesteps - 1, 1));
        var betas = engine.TensorAddScalar(
            engine.TensorMultiplyScalar(engine.TensorArange<T>(0.0, numTimesteps, 1.0), step),
            NumOps.FromDouble(betaStart));

        _betas = betas.ToVector();
        _alphasCumprod = engine.TensorCumProd(engine.ScalarMinusTensor(NumOps.One, betas), 0).ToVector();
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
        return MixWithUniform(Tensor<T>.FromVector(x0), _alphasCumprod[t]).ToVector();
    }

    /// <summary>
    /// Computes the KL divergence loss between predicted and true posterior.
    /// </summary>
    /// <param name="predictedLogProbs">Log-probabilities predicted by the model (length = K).</param>
    /// <param name="trueOneHot">True one-hot category vector.</param>
    /// <param name="noisyProbs">Noisy category probabilities at timestep t.</param>
    /// <param name="t">The current timestep.</param>
    /// <returns>KL divergence loss value.</returns>
    /// <remarks>
    /// Hoogeboom et al. (2021) and the TabDDPM reference (<c>_multinomial_loss</c>): for t &gt; 0 the loss is
    /// KL(q(x_{t-1} | x_t, x_0) || q(x_{t-1} | x_t, x0_hat)), where x0_hat = softmax(<paramref name="predictedLogProbs"/>);
    /// at t = 0 it is the decoder negative log-likelihood -sum x_0 log q(x_{t-1} | x_t, x0_hat).
    /// </remarks>
    public T ComputeLoss(Vector<T> predictedLogProbs, Vector<T> trueOneHot, Vector<T> noisyProbs, int t)
    {
        var engine = AiDotNetEngine.Current;
        var target = Tensor<T>.FromVector(trueOneHot);
        var predPosterior = Posterior(Tensor<T>.FromVector(noisyProbs), Softmax(predictedLogProbs), t);
        var logPred = engine.TensorLog(engine.TensorMax(predPosterior, NumOps.FromDouble(LogFloor)));

        if (t <= 0)
        {
            return NumOps.Negate(engine.TensorSum(engine.TensorMultiply(target, logPred)));
        }

        var truePosterior = Posterior(Tensor<T>.FromVector(noisyProbs), target, t);
        var logTrue = engine.TensorLog(engine.TensorMax(truePosterior, NumOps.FromDouble(LogFloor)));
        return engine.TensorSum(engine.TensorMultiply(truePosterior, engine.TensorSubtract(logTrue, logPred)));
    }

    /// <summary>
    /// Computes the gradient of the auxiliary x_0 cross-entropy with respect to predicted log-probabilities.
    /// </summary>
    /// <param name="predictedLogProbs">Log-probabilities predicted by the model.</param>
    /// <param name="trueOneHot">True one-hot category vector.</param>
    /// <returns>Gradient vector: softmax(predictedLogProbs) - trueOneHot.</returns>
    /// <remarks>
    /// This is the gradient of -sum x_0 log softmax(z). For the gradient of <see cref="ComputeLoss"/>
    /// use <see cref="ComputeLossGradient(Vector{T}, Vector{T}, Vector{T}, int)"/>.
    /// </remarks>
    public Vector<T> ComputeLossGradient(Vector<T> predictedLogProbs, Vector<T> trueOneHot)
    {
        var engine = AiDotNetEngine.Current;
        return engine.TensorSubtract(Softmax(predictedLogProbs), Tensor<T>.FromVector(trueOneHot)).ToVector();
    }

    /// <summary>
    /// Computes the gradient of <see cref="ComputeLoss"/> with respect to predicted log-probabilities.
    /// </summary>
    /// <param name="predictedLogProbs">Log-probabilities (logits) predicted by the model.</param>
    /// <param name="trueOneHot">True one-hot category vector.</param>
    /// <param name="noisyProbs">Noisy category probabilities at timestep t.</param>
    /// <param name="t">The current timestep.</param>
    /// <returns>The gradient vector.</returns>
    public Vector<T> ComputeLossGradient(Vector<T> predictedLogProbs, Vector<T> trueOneHot, Vector<T> noisyProbs, int t)
    {
        var engine = AiDotNetEngine.Current;
        var target = Tensor<T>.FromVector(trueOneHot);
        var xt = Tensor<T>.FromVector(noisyProbs);
        var probs = Softmax(predictedLogProbs);

        // q = true posterior (t > 0) or x_0 (t = 0); p ~ f * g with f = q(x_t | x_{t-1}), g = alphaBar_{t-1} x0_hat + c.
        // dL/dg = -q / g + f / sum(f * g); dL/dx0_hat = alphaBar_{t-1} dL/dg; dL/dz = x0_hat * (dL/dx0_hat - <dL/dx0_hat, x0_hat>).
        var q = t <= 0 ? target : Posterior(xt, target, t);
        var f = StepLikelihood(xt, t);
        T alphaBarPrev = AlphaBarPrev(t);
        var g = MixWithUniform(probs, alphaBarPrev);
        T total = engine.TensorSum(engine.TensorMultiply(f, g));

        var dg = engine.TensorSubtract(
            engine.TensorDivideScalar(f, total),
            engine.TensorDivide(q, engine.TensorMax(g, NumOps.FromDouble(LogFloor))));
        var dProbs = engine.TensorMultiplyScalar(dg, alphaBarPrev);
        T inner = engine.TensorSum(engine.TensorMultiply(dProbs, probs));
        return engine.TensorMultiply(probs, engine.TensorSubtractScalar(dProbs, inner)).ToVector();
    }

    /// <summary>
    /// Performs one reverse diffusion step for categorical features.
    /// Samples a category from the predicted posterior q(x_{t-1} | x_t, x0_hat).
    /// </summary>
    /// <param name="noisyProbs">Current noisy probability vector at timestep t.</param>
    /// <param name="predictedLogProbs">Log-probabilities predicted by the model for x_0.</param>
    /// <param name="t">Current timestep.</param>
    /// <returns>Denoised probability vector at timestep t-1.</returns>
    /// <remarks>
    /// The posterior is theta ~ [alpha_t x_t + (1 - alpha_t)/K] * [alphaBar_{t-1} x0_hat + (1 - alphaBar_{t-1})/K]
    /// (Hoogeboom et al. 2021, Eq. 9 and TabDDPM's <c>q_posterior</c>), with alphaBar_{-1} = 1 at t = 0.
    /// If the posterior has no mass the predicted x0_hat is sampled instead.
    /// </remarks>
    public Vector<T> DenoisingStep(Vector<T> noisyProbs, Vector<T> predictedLogProbs, int t)
    {
        var engine = AiDotNetEngine.Current;
        var probs = Softmax(predictedLogProbs);
        var unnormalized = engine.TensorMultiply(
            StepLikelihood(Tensor<T>.FromVector(noisyProbs), t),
            MixWithUniform(probs, AlphaBarPrev(t)));

        T posteriorSum = engine.TensorSum(unnormalized);
        var posterior = NumOps.GreaterThan(posteriorSum, NumOps.FromDouble(1e-10))
            ? engine.TensorDivideScalar(unnormalized, posteriorSum)
            : probs;

        return SampleCategorical(posterior);
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

    /// <summary>alphaBar_{t-1}, with alphaBar_{-1} = 1.</summary>
    private T AlphaBarPrev(int t) => t > 0 ? _alphasCumprod[t - 1] : NumOps.One;

    /// <summary>a * x + (1 - a) / K.</summary>
    private static Tensor<T> MixWithUniform(Tensor<T> x, T a)
    {
        var engine = AiDotNetEngine.Current;
        T uniformMass = NumOps.Divide(NumOps.Subtract(NumOps.One, a), NumOps.FromDouble(x.Length));
        return engine.TensorAddScalar(engine.TensorMultiplyScalar(x, a), uniformMass);
    }

    /// <summary>q(x_t | x_{t-1} = i) for every i: alpha_t x_t + beta_t / K.</summary>
    private Tensor<T> StepLikelihood(Tensor<T> xt, int t)
    {
        return MixWithUniform(xt, NumOps.Subtract(NumOps.One, _betas[t]));
    }

    /// <summary>Normalized posterior q(x_{t-1} | x_t, x_0) for a (possibly soft) x_0.</summary>
    private Tensor<T> Posterior(Tensor<T> xt, Tensor<T> x0, int t)
    {
        var engine = AiDotNetEngine.Current;
        var unnormalized = engine.TensorMultiply(StepLikelihood(xt, t), MixWithUniform(x0, AlphaBarPrev(t)));
        T sum = engine.TensorSum(unnormalized);
        return NumOps.GreaterThan(sum, NumOps.Zero) ? engine.TensorDivideScalar(unnormalized, sum) : unnormalized;
    }

    /// <summary>Softmax of the log-probabilities, sum floored at 1e-10 as in master.</summary>
    private static Tensor<T> Softmax(Vector<T> logProbs)
    {
        var engine = AiDotNetEngine.Current;
        var x = Tensor<T>.FromVector(logProbs);
        if (x.Length == 0)
        {
            return x;
        }

        var exp = engine.TensorExp(engine.TensorSubtractScalar(x, engine.TensorMaxValue(x)));
        T sumExp = engine.TensorSum(exp);
        T floor = NumOps.FromDouble(1e-10);
        return engine.TensorDivideScalar(exp, NumOps.LessThan(sumExp, floor) ? floor : sumExp);
    }

    private Vector<T> SampleCategorical(Tensor<T> probs)
    {
        int k = probs.Length;
        var result = new Vector<T>(k);

        // Inverse-CDF draw: the first category whose cumulative probability reaches u;
        // the last category when rounding leaves the total below u.
        double u = _random.NextDouble();
        var engine = AiDotNetEngine.Current;
        var cdf = engine.TensorCumSum(probs, 0);
        var position = engine.TensorSearchSorted(cdf, Tensor<T>.CreateDefault(new[] { 1 }, NumOps.FromDouble(u)));
        int sampled = Math.Min(position[0], k - 1);

        result[sampled] = NumOps.One;
        return result;
    }

    #endregion
}
