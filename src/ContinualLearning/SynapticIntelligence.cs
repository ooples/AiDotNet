using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Validation;

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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Regularization)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Continual Learning Through Synaptic Intelligence", "https://arxiv.org/abs/1703.04200", Year = 2017, Authors = "Friedemann Zenke, Ben Poole, Surya Ganguli")]
[ComponentType(ComponentType.ContinualLearner)]
[PipelineStage(PipelineStage.Training)]
public class SynapticIntelligence<T> : IContinualLearningStrategy<T>
{
    private readonly INumericOperations<T> _numOps;
    private static IEngine Engine => AiDotNetEngine.Current;
    private Vector<T> _omega;                    // Consolidated importance
    private Vector<T> _previousParameters;       // Parameters at start of task
    private Vector<T> _pathIntegral;            // Running sum of gradient * parameter change
    private double _lambda;
    private readonly double _damping;            // Small constant to prevent division by zero
    private bool _isTrackingTask;
    [Scratch]
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
    public bool AccumulatesAcrossTasks => true;

    /// <inheritdoc />
    public double Lambda
    {
        get => _lambda;
        set => _lambda = value;
    }

    /// <inheritdoc />
    public void BeforeTask(INeuralNetwork<T> network, int taskId)
    {
        Guard.NotNull(network);

        int paramCount = checked((int)network.ParameterCount);

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
        Guard.NotNull(network);

        if (!_isTrackingTask)
        {
            return;
        }

        var currentParams = network.GetParameters();
        var dampingT = _numOps.FromDouble(_damping);

        // Compute importance and consolidate into omega
        // Ω_i += ω_i / (Δθ_i² + ξ)
        int n = _omega.Length;
        var omega = _omega.AsTensor();
        var delta = Engine.TensorSubtract(Head(currentParams, n), Head(_previousParameters, n));
        var denominator = Engine.TensorAddScalar(Engine.TensorMultiply(delta, delta), dampingT);
        var importance = Engine.TensorDivide(Head(_pathIntegral, n), denominator);

        // Only add positive importance (contributed to loss reduction); written back in place
        var consolidated = Engine.TensorWhere(
            Engine.TensorGreaterThan(importance, _numOps.Zero),
            Engine.TensorAdd(omega, importance),
            omega);
        consolidated.AsSpan().CopyTo(_omega.AsWritableSpan());

        // Store current parameters as reference for next task
        _previousParameters = currentParams.Clone();
        _isTrackingTask = false;
    }

    /// <inheritdoc />
    public T ComputeLoss(INeuralNetwork<T> network)
    {
        Guard.NotNull(network);

        if (_omega.Length == 0 || !HasPreviousTasks())
        {
            return _numOps.Zero;
        }

        var currentParams = network.GetParameters();
        int n = currentParams.Length;

        // SI loss: λ/2 * Σ Ω_i * (θ_i - θ*_i)²
        var diff = Engine.TensorSubtract(currentParams.AsTensor(), Head(_previousParameters, n));
        var penalty = Engine.TensorMultiply(Head(_omega, n), Engine.TensorMultiply(diff, diff));
        var loss = Engine.TensorSum(penalty);

        var halfLambda = _numOps.FromDouble(_lambda / 2.0);
        return _numOps.Multiply(halfLambda, loss);
    }

    /// <inheritdoc />
    public Vector<T> ModifyGradients(INeuralNetwork<T> network, Vector<T> gradients)
    {
        Guard.NotNull(network);
        Guard.NotNull(gradients);

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

        int n = gradients.Length;
        var diff = Engine.TensorSubtract(Head(currentParams2, n), Head(_previousParameters, n));
        var siGrad = Engine.TensorMultiplyScalar(Engine.TensorMultiply(Head(_omega, n), diff), lambdaT);
        Engine.TensorAdd(gradients.AsTensor(), siGrad).AsSpan().CopyTo(gradients.AsWritableSpan());

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
        int n = _pathIntegral.Length;
        var paramChange = Engine.TensorSubtract(Head(currentParams, n), Head(_previousParameters, n));
        var contribution = Engine.TensorMultiply(Engine.TensorNegate(Head(gradients, n)), paramChange);
        Engine.TensorAdd(_pathIntegral.AsTensor(), contribution).AsSpan().CopyTo(_pathIntegral.AsWritableSpan());
    }

    /// <summary>
    /// Checks if there are any previous tasks stored.
    /// </summary>
    private bool HasPreviousTasks()
    {
        return _omega.Length > 0
            && Engine.TensorCountNonzero(Engine.TensorGreaterThan(_omega.AsTensor(), _numOps.Zero)) > 0;
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
}
