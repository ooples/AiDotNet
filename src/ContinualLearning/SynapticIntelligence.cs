using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.ContinualLearning;

/// <summary>
/// Implements Synaptic Intelligence (SI) for continual learning.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Synaptic Intelligence is similar to EWC but estimates weight
/// importance online during training rather than computing Fisher information after training.
/// It tracks how much each weight contributes to the loss reduction during learning.</para>
///
/// <para><b>How it works:</b></para>
/// <list type="number">
/// <item><description>During training, SI tracks the "path integral" of gradients for each weight,
/// which measures how much each weight contributed to learning the current task.</description></item>
/// <item><description>After each task, these contributions are used to compute importance scores.</description></item>
/// <item><description>When learning new tasks, changes to important weights are penalized.</description></item>
/// </list>
///
/// <para><b>Formula:</b> Ω_i = Σ_tasks [ω_i^task / (Δθ_i^task)² + ξ]</para>
/// <para>where ω_i is the path integral of gradients for weight i.</para>
///
/// <para><b>Advantages over EWC:</b></para>
/// <list type="bullet">
/// <item><description>Online computation - no need to store training data.</description></item>
/// <item><description>Lower memory overhead than Fisher Information computation.</description></item>
/// <item><description>Naturally handles streaming data scenarios.</description></item>
/// </list>
///
/// <para><b>Reference:</b> Zenke, F., Poole, B., and Ganguli, S. "Continual Learning Through
/// Synaptic Intelligence" (2017). ICML.</para>
/// </remarks>
public class SynapticIntelligence<T> : IContinualLearningStrategy<T>
{
    private readonly INumericOperations<T> _numOps;
    private Vector<T> _omega;                    // Consolidated importance
    private Vector<T> _previousParameters;       // Parameters at start of task
    private Vector<T> _pathIntegral;            // Running sum of gradient * parameter change
    private double _lambda;
    private readonly double _damping;            // Small constant to prevent division by zero
    private bool _isTrackingTask;
    private Vector<T>? _lastGradients;

    /// <summary>
    /// Initializes a new instance of the SynapticIntelligence class.
    /// </summary>
    /// <param name="lambda">The regularization strength (default: 1.0).</param>
    /// <param name="damping">Small constant for numerical stability (default: 0.1).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b></para>
    /// <list type="bullet">
    /// <item><description>Lambda controls how strongly to protect previous knowledge.</description></item>
    /// <item><description>Damping prevents numerical issues when weights don't change much.</description></item>
    /// </list>
    /// </remarks>
    public SynapticIntelligence(double lambda = 1.0, double damping = 0.1)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _omega = new Vector<T>(0);
        _previousParameters = new Vector<T>(0);
        _pathIntegral = new Vector<T>(0);
        _lambda = lambda;
        _damping = damping;
        _isTrackingTask = false;
        _lastGradients = null;
    }

    /// <inheritdoc />
    public double Lambda
    {
        get => _lambda;
        set => _lambda = value;
    }

    /// <inheritdoc />
    public void BeforeTask(INeuralNetwork<T> network, int taskId)
    {
        _ = network ?? throw new ArgumentNullException(nameof(network));

        var paramCount = network.ParameterCount;

        // Initialize omega if this is the first task
        if (_omega.Length == 0)
        {
            _omega = new Vector<T>(paramCount);
        }

        // Store parameters at the start of this task
        _previousParameters = network.GetParameters().Clone();

        // Reset path integral for new task
        _pathIntegral = new Vector<T>(paramCount);

        _isTrackingTask = true;
        _lastGradients = null;
    }

    /// <inheritdoc />
    public void AfterTask(INeuralNetwork<T> network, (Tensor<T> inputs, Tensor<T> targets) taskData, int taskId)
    {
        _ = network ?? throw new ArgumentNullException(nameof(network));

        if (!_isTrackingTask)
        {
            return;
        }

        var currentParams = network.GetParameters();
        var dampingT = _numOps.FromDouble(_damping);

        // Compute importance and consolidate into omega
        // Ω_i += ω_i / (Δθ_i² + ξ)
        for (int i = 0; i < _omega.Length; i++)
        {
            var delta = _numOps.Subtract(currentParams[i], _previousParameters[i]);
            var deltaSquared = _numOps.Multiply(delta, delta);
            var denominator = _numOps.Add(deltaSquared, dampingT);
            var importance = _numOps.Divide(_pathIntegral[i], denominator);

            // Only add positive importance (contributed to loss reduction)
            if (_numOps.GreaterThan(importance, _numOps.Zero))
            {
                _omega[i] = _numOps.Add(_omega[i], importance);
            }
        }

        // Store current parameters as reference for next task
        _previousParameters = currentParams.Clone();
        _isTrackingTask = false;
    }

    /// <inheritdoc />
    public T ComputeLoss(INeuralNetwork<T> network)
    {
        _ = network ?? throw new ArgumentNullException(nameof(network));

        if (_omega.Length == 0 || !HasPreviousTasks())
        {
            return _numOps.Zero;
        }

        var currentParams = network.GetParameters();
        var loss = _numOps.Zero;

        // SI loss: λ/2 * Σ Ω_i * (θ_i - θ*_i)²
        for (int i = 0; i < currentParams.Length; i++)
        {
            var diff = _numOps.Subtract(currentParams[i], _previousParameters[i]);
            var squaredDiff = _numOps.Multiply(diff, diff);
            var penalty = _numOps.Multiply(_omega[i], squaredDiff);
            loss = _numOps.Add(loss, penalty);
        }

        var halfLambda = _numOps.FromDouble(_lambda / 2.0);
        return _numOps.Multiply(halfLambda, loss);
    }

    /// <inheritdoc />
    public Vector<T> ModifyGradients(INeuralNetwork<T> network, Vector<T> gradients)
    {
        _ = network ?? throw new ArgumentNullException(nameof(network));
        _ = gradients ?? throw new ArgumentNullException(nameof(gradients));

        // Track path integral during training
        if (_isTrackingTask && _lastGradients != null)
        {
            var currentParams = network.GetParameters();
            UpdatePathIntegral(currentParams, gradients);
        }

        _lastGradients = gradients.Clone();

        // Add SI regularization gradient
        if (_omega.Length == 0 || !HasPreviousTasks())
        {
            return gradients;
        }

        var currentParams2 = network.GetParameters();
        var lambdaT = _numOps.FromDouble(_lambda);

        for (int i = 0; i < gradients.Length; i++)
        {
            var diff = _numOps.Subtract(currentParams2[i], _previousParameters[i]);
            var siGrad = _numOps.Multiply(_omega[i], diff);
            siGrad = _numOps.Multiply(lambdaT, siGrad);
            gradients[i] = _numOps.Add(gradients[i], siGrad);
        }

        return gradients;
    }

    /// <inheritdoc />
    public void Reset()
    {
        _omega = new Vector<T>(0);
        _previousParameters = new Vector<T>(0);
        _pathIntegral = new Vector<T>(0);
        _isTrackingTask = false;
        _lastGradients = null;
    }

    /// <summary>
    /// Updates the path integral with the current gradient and parameter change.
    /// </summary>
    private void UpdatePathIntegral(Vector<T> currentParams, Vector<T> gradients)
    {
        // ω_i += -g_i * Δθ_i
        // The negative gradient times parameter change approximates contribution to loss reduction
        for (int i = 0; i < _pathIntegral.Length; i++)
        {
            var paramChange = _numOps.Subtract(currentParams[i], _previousParameters[i]);
            var negGrad = _numOps.Negate(gradients[i]);
            var contribution = _numOps.Multiply(negGrad, paramChange);
            _pathIntegral[i] = _numOps.Add(_pathIntegral[i], contribution);
        }
    }

    /// <summary>
    /// Checks if there are any previous tasks stored.
    /// </summary>
    private bool HasPreviousTasks()
    {
        for (int i = 0; i < _omega.Length; i++)
        {
            if (_numOps.GreaterThan(_omega[i], _numOps.Zero))
            {
                return true;
            }
        }
        return false;
    }
}
