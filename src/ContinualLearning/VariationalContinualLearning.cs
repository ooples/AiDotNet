using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.Extensions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.ContinualLearning;

/// <summary>
/// Implements Variational Continual Learning (VCL) for Bayesian continual learning.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> VCL uses Bayesian neural networks where each weight has a
/// probability distribution (mean and variance) rather than a single value. This allows
/// the network to represent uncertainty and naturally prevents forgetting by using the
/// posterior from previous tasks as the prior for new tasks.</para>
///
/// <para><b>How it works:</b></para>
/// <list type="number">
/// <item><description>Represent weights as Gaussian distributions: w ~ N(μ, σ²).</description></item>
/// <item><description>Train using variational inference, optimizing the ELBO (Evidence Lower Bound).</description></item>
/// <item><description>After each task, the posterior becomes the prior for the next task.</description></item>
/// <item><description>The KL divergence between current and previous posterior prevents forgetting.</description></item>
/// </list>
///
/// <para><b>Key Formula:</b></para>
/// <para>Loss = E_q[log p(D|w)] - KL(q(w|D_new) || p(w|D_old))</para>
/// <para>where q(w|D_new) is the current posterior and p(w|D_old) is the previous posterior (now prior).</para>
///
/// <para><b>Advantages:</b></para>
/// <list type="bullet">
/// <item><description>Principled Bayesian approach to continual learning.</description></item>
/// <item><description>Natural uncertainty quantification.</description></item>
/// <item><description>Sequential posterior updates match online learning.</description></item>
/// </list>
///
/// <para><b>Reference:</b> Nguyen, C.V., Li, Y., Bui, T.D., and Turner, R.E.
/// "Variational Continual Learning" (2018). ICLR.</para>
/// </remarks>
public class VariationalContinualLearning<T> : IContinualLearningStrategy<T>
{
    private readonly INumericOperations<T> _numOps;
    private Vector<T> _priorMean;           // Prior mean (posterior from previous task)
    private Vector<T> _priorLogVar;         // Prior log-variance
    private Vector<T> _posteriorMean;       // Current posterior mean
    private Vector<T> _posteriorLogVar;     // Current posterior log-variance
    private double _lambda;
    private readonly double _initialLogVar;
    private int _taskCount;
    private readonly Random _random;

    /// <summary>
    /// Initializes a new instance of the VariationalContinualLearning class.
    /// </summary>
    /// <param name="lambda">KL divergence weight (default: 1.0).</param>
    /// <param name="initialLogVar">Initial log-variance for weight distributions (default: -3.0).</param>
    /// <param name="seed">Random seed for sampling (default: null).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b></para>
    /// <list type="bullet">
    /// <item><description>Lambda controls how much to penalize deviation from the prior (previous posterior).</description></item>
    /// <item><description>Initial log-variance controls initial uncertainty. -3 means σ ≈ 0.22.</description></item>
    /// </list>
    /// </remarks>
    public VariationalContinualLearning(double lambda = 1.0, double initialLogVar = -3.0, int? seed = null)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _priorMean = new Vector<T>(0);
        _priorLogVar = new Vector<T>(0);
        _posteriorMean = new Vector<T>(0);
        _posteriorLogVar = new Vector<T>(0);
        _lambda = lambda;
        _initialLogVar = initialLogVar;
        _taskCount = 0;
        _random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();
    }

    /// <inheritdoc />
    public double Lambda
    {
        get => _lambda;
        set => _lambda = value;
    }

    /// <summary>
    /// Gets the number of tasks processed.
    /// </summary>
    public int TaskCount => _taskCount;

    /// <summary>
    /// Gets the initial log-variance value.
    /// </summary>
    public double InitialLogVar => _initialLogVar;

    /// <inheritdoc />
    public void BeforeTask(INeuralNetwork<T> network, int taskId)
    {
        Guard.NotNull(network);

        var paramCount = network.ParameterCount;

        if (_priorMean.Length == 0)
        {
            // First task: initialize prior with standard normal and given variance
            _priorMean = new Vector<T>(paramCount);
            _priorLogVar = new Vector<T>(paramCount);

            var initialLogVarT = _numOps.FromDouble(_initialLogVar);
            for (int i = 0; i < paramCount; i++)
            {
                _priorMean[i] = _numOps.Zero;
                _priorLogVar[i] = initialLogVarT;
            }
        }
        else
        {
            // Subsequent tasks: use posterior from previous task as prior
            _priorMean = _posteriorMean.Clone();
            _priorLogVar = _posteriorLogVar.Clone();
        }

        // Initialize posterior from prior
        _posteriorMean = _priorMean.Clone();
        _posteriorLogVar = _priorLogVar.Clone();
    }

    /// <inheritdoc />
    public void AfterTask(INeuralNetwork<T> network, (Tensor<T> inputs, Tensor<T> targets) taskData, int taskId)
    {
        Guard.NotNull(network);

        // Store the current posterior (network parameters represent the mean)
        _posteriorMean = network.GetParameters().Clone();

        // Log-variance could be stored separately or estimated from gradients
        // For simplicity, we update based on gradient variance
        if (_posteriorLogVar.Length == 0)
        {
            _posteriorLogVar = new Vector<T>(network.ParameterCount);
            var initialLogVarT = _numOps.FromDouble(_initialLogVar);
            for (int i = 0; i < _posteriorLogVar.Length; i++)
            {
                _posteriorLogVar[i] = initialLogVarT;
            }
        }

        _taskCount++;
    }

    /// <inheritdoc />
    public T ComputeLoss(INeuralNetwork<T> network)
    {
        Guard.NotNull(network);

        if (_taskCount == 0 || _priorMean.Length == 0)
        {
            return _numOps.Zero;
        }

        var currentMean = network.GetParameters();

        // Compute KL divergence: KL(q||p) for Gaussians
        // KL(N(μ_q, σ_q²) || N(μ_p, σ_p²)) =
        //   log(σ_p/σ_q) + (σ_q² + (μ_q - μ_p)²)/(2σ_p²) - 0.5
        var klDiv = _numOps.Zero;

        for (int i = 0; i < currentMean.Length; i++)
        {
            var meanDiff = _numOps.Subtract(currentMean[i], _priorMean[i]);
            var meanDiffSq = _numOps.Multiply(meanDiff, meanDiff);

            // For simplicity, use the stored log-variances
            var qLogVar = _posteriorLogVar[i];
            var pLogVar = _priorLogVar[i];

            // σ² = exp(logVar)
            var qVar = Exp(qLogVar);
            var pVar = Exp(pLogVar);

            // log(σ_p) - log(σ_q) = 0.5 * (logVar_p - logVar_q)
            var logRatio = _numOps.Multiply(
                _numOps.FromDouble(0.5),
                _numOps.Subtract(pLogVar, qLogVar));

            // (σ_q² + (μ_q - μ_p)²) / (2σ_p²)
            var numerator = _numOps.Add(qVar, meanDiffSq);
            var denominator = _numOps.Multiply(_numOps.FromDouble(2.0), pVar);
            var fraction = _numOps.Divide(numerator, denominator);

            // KL for this weight
            var kl_i = _numOps.Subtract(
                _numOps.Add(logRatio, fraction),
                _numOps.FromDouble(0.5));

            klDiv = _numOps.Add(klDiv, kl_i);
        }

        // Scale by lambda
        var lambdaT = _numOps.FromDouble(_lambda);
        return _numOps.Multiply(lambdaT, klDiv);
    }

    /// <inheritdoc />
    public Vector<T> ModifyGradients(INeuralNetwork<T> network, Vector<T> gradients)
    {
        Guard.NotNull(network);
        Guard.NotNull(gradients);

        if (_taskCount == 0 || _priorMean.Length == 0)
        {
            return gradients;
        }

        var currentMean = network.GetParameters();
        var lambdaT = _numOps.FromDouble(_lambda);

        // Add gradient of KL divergence w.r.t. mean
        // ∂KL/∂μ_q = (μ_q - μ_p) / σ_p²
        for (int i = 0; i < gradients.Length; i++)
        {
            var meanDiff = _numOps.Subtract(currentMean[i], _priorMean[i]);
            var pVar = Exp(_priorLogVar[i]);

            // Add small epsilon to prevent division by zero
            var epsilon = _numOps.FromDouble(1e-8);
            pVar = _numOps.Add(pVar, epsilon);

            var klGrad = _numOps.Divide(meanDiff, pVar);
            klGrad = _numOps.Multiply(lambdaT, klGrad);

            gradients[i] = _numOps.Add(gradients[i], klGrad);
        }

        return gradients;
    }

    /// <inheritdoc />
    public void Reset()
    {
        _priorMean = new Vector<T>(0);
        _priorLogVar = new Vector<T>(0);
        _posteriorMean = new Vector<T>(0);
        _posteriorLogVar = new Vector<T>(0);
        _taskCount = 0;
    }

    /// <summary>
    /// Samples weights from the posterior distribution for prediction.
    /// </summary>
    /// <param name="network">The neural network.</param>
    /// <returns>Sampled weights.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> In Bayesian neural networks, we sample weights from
    /// their distribution during inference. Multiple samples can be used for
    /// Monte Carlo estimation of predictions and uncertainty.</para>
    /// </remarks>
    public Vector<T> SampleWeights(INeuralNetwork<T> network)
    {
        Guard.NotNull(network);

        if (_posteriorMean.Length == 0)
        {
            return network.GetParameters();
        }

        var samples = new Vector<T>(_posteriorMean.Length);

        for (int i = 0; i < _posteriorMean.Length; i++)
        {
            // Sample from N(μ, σ²) using reparameterization trick
            // w = μ + σ * ε, where ε ~ N(0, 1)
            var epsilon = SampleStandardNormal();
            var sigma = Sqrt(Exp(_posteriorLogVar[i]));
            var sample = _numOps.Add(
                _posteriorMean[i],
                _numOps.Multiply(sigma, _numOps.FromDouble(epsilon)));
            samples[i] = sample;
        }

        return samples;
    }

    /// <summary>
    /// Updates the posterior log-variance based on gradient information.
    /// </summary>
    /// <param name="gradients">Parameter gradients.</param>
    /// <param name="learningRate">Learning rate for variance update.</param>
    public void UpdateVariance(Vector<T> gradients, double learningRate)
    {
        Guard.NotNull(gradients);

        if (_posteriorLogVar.Length == 0 || _posteriorLogVar.Length != gradients.Length)
        {
            return;
        }

        // Simple variance update based on squared gradients
        // Higher gradients indicate more uncertainty
        var lr = _numOps.FromDouble(learningRate);

        for (int i = 0; i < _posteriorLogVar.Length; i++)
        {
            var gradSq = _numOps.Multiply(gradients[i], gradients[i]);
            var logGradSq = Log(_numOps.Add(gradSq, _numOps.FromDouble(1e-8)));

            // Exponential moving average
            var alpha = _numOps.FromDouble(0.1);
            var oneMinusAlpha = _numOps.FromDouble(0.9);

            _posteriorLogVar[i] = _numOps.Add(
                _numOps.Multiply(oneMinusAlpha, _posteriorLogVar[i]),
                _numOps.Multiply(alpha, logGradSq));
        }
    }

    /// <summary>
    /// Gets the current prior distribution parameters.
    /// </summary>
    /// <returns>Tuple of prior mean and log-variance vectors.</returns>
    public (Vector<T> mean, Vector<T> logVar) GetPrior()
    {
        return (_priorMean.Clone(), _priorLogVar.Clone());
    }

    /// <summary>
    /// Gets the current posterior distribution parameters.
    /// </summary>
    /// <returns>Tuple of posterior mean and log-variance vectors.</returns>
    public (Vector<T> mean, Vector<T> logVar) GetPosterior()
    {
        return (_posteriorMean.Clone(), _posteriorLogVar.Clone());
    }

    /// <summary>
    /// Computes exp(x) for a value.
    /// </summary>
    private T Exp(T x)
    {
        var xDouble = _numOps.ToDouble(x);
        return _numOps.FromDouble(Math.Exp(xDouble));
    }

    /// <summary>
    /// Computes log(x) for a value.
    /// </summary>
    private T Log(T x)
    {
        var xDouble = _numOps.ToDouble(x);
        return _numOps.FromDouble(Math.Log(Math.Max(xDouble, 1e-10)));
    }

    /// <summary>
    /// Computes sqrt(x) for a value.
    /// </summary>
    private T Sqrt(T x)
    {
        var xDouble = _numOps.ToDouble(x);
        return _numOps.FromDouble(Math.Sqrt(Math.Max(xDouble, 0)));
    }

    /// <summary>
    /// Samples from standard normal distribution.
    /// </summary>
    private double SampleStandardNormal()
    {
        return _random.NextGaussian();
    }
}
