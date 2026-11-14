using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.Attributes;
using AiDotNet.Enums;
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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Bayesian)]
[ModelCategory(ModelCategory.Regularization)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Variational Continual Learning", "https://arxiv.org/abs/1710.10628", Year = 2018, Authors = "Cuong V. Nguyen, Yingzhen Li, Thang D. Bui, Richard E. Turner")]
[ComponentType(ComponentType.ContinualLearner)]
[PipelineStage(PipelineStage.Training)]
public class VariationalContinualLearning<T> : IContinualLearningStrategy<T>
{
    private readonly INumericOperations<T> _numOps;
    private static IEngine Engine => AiDotNetEngine.Current;
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
    /// <inheritdoc />
    /// <remarks>VCL shifts prior to last posterior without precision accumulation per the reference implementation.</remarks>
    public bool AccumulatesAcrossTasks => false;

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

        int paramCount = checked((int)network.ParameterCount);

        if (_priorMean.Length == 0)
        {
            // First task: initialize prior with standard normal and given variance
            var initialLogVarT = _numOps.FromDouble(_initialLogVar);
            _priorMean = Engine.Fill(paramCount, _numOps.Zero);
            _priorLogVar = Engine.Fill(paramCount, initialLogVarT);
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
            var initialLogVarT = _numOps.FromDouble(_initialLogVar);
            _posteriorLogVar = Engine.Fill(checked((int)network.ParameterCount), initialLogVarT);
        }

        // Per the VCL reference implementation (Nguyen et al., ICLR 2018):
        // the posterior from this task becomes the prior for the next task.
        // Variance is NOT explicitly tightened — it's learned via gradient descent
        // during training. There is no precision accumulation like in EWC.
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
        int n = currentMean.Length;
        var half = _numOps.FromDouble(0.5);
        var two = _numOps.FromDouble(2.0);

        var meanDiff = Engine.TensorSubtract(currentMean.AsTensor(), Head(_priorMean, n));
        var meanDiffSq = Engine.TensorMultiply(meanDiff, meanDiff);

        // For simplicity, use the stored log-variances
        var qLogVar = Head(_posteriorLogVar, n);
        var pLogVar = Head(_priorLogVar, n);

        // σ² = exp(logVar)
        var qVar = Engine.TensorExp(qLogVar);
        var pVar = Engine.TensorExp(pLogVar);

        // log(σ_p) - log(σ_q) = 0.5 * (logVar_p - logVar_q)
        var logRatio = Engine.TensorMultiplyScalar(Engine.TensorSubtract(pLogVar, qLogVar), half);

        // (σ_q² + (μ_q - μ_p)²) / (2σ_p²)
        var numerator = Engine.TensorAdd(qVar, meanDiffSq);
        var denominator = Engine.TensorMultiplyScalar(pVar, two);
        var fraction = Engine.TensorDivide(numerator, denominator);

        // KL for each weight, summed
        var klPerWeight = Engine.TensorSubtractScalar(Engine.TensorAdd(logRatio, fraction), half);
        var klDiv = Engine.TensorSum(klPerWeight);

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
        int n = gradients.Length;
        var meanDiff = Engine.TensorSubtract(Head(currentMean, n), Head(_priorMean, n));

        // Add small epsilon to prevent division by zero
        var epsilon = _numOps.FromDouble(1e-8);
        var pVar = Engine.TensorAddScalar(Engine.TensorExp(Head(_priorLogVar, n)), epsilon);

        var klGrad = Engine.TensorMultiplyScalar(Engine.TensorDivide(meanDiff, pVar), lambdaT);
        Engine.TensorAdd(gradients.AsTensor(), klGrad).AsSpan().CopyTo(gradients.AsWritableSpan());

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

        // log(max(g^2 + 1e-8, 1e-10)), master's floored log. The floor still matters: 1e-8 rounds to zero
        // in Half, so a zero gradient reaches the log as 0. It is applied in the log domain because
        // 1e-10 is not representable in Half while log 1e-10 is.
        var g = gradients.AsTensor();
        var gradSq = Engine.TensorMultiply(g, g);
        var argument = Engine.TensorAddScalar(gradSq, _numOps.FromDouble(1e-8));
        var logGradSq = Engine.TensorWhere(
            Engine.TensorGreaterThan(argument, _numOps.FromDouble(1e-10)),
            Engine.TensorLog(argument),
            Engine.TensorFullLike(argument, _numOps.FromDouble(Math.Log(1e-10))));

        // Exponential moving average, written back in place
        var alpha = _numOps.FromDouble(0.1);
        var oneMinusAlpha = _numOps.FromDouble(0.9);
        var logVar = _posteriorLogVar.AsTensor();
        var updated = Engine.TensorAdd(
            Engine.TensorMultiplyScalar(logVar, oneMinusAlpha),
            Engine.TensorMultiplyScalar(logGradSq, alpha));
        updated.AsSpan().CopyTo(_posteriorLogVar.AsWritableSpan());
    }

    /// <summary>
    /// Zero-copy rank-1 tensor view over the first <paramref name="count"/> elements of a vector.
    /// </summary>
    private static Tensor<T> Head(Vector<T> vector, int count)
    {
        return count == vector.Length
            ? vector.AsTensor()
            : Tensor<T>.FromMemory(vector.AsWritableMemory().Slice(0, count), new[] { count });
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
