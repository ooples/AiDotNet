using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.ContinualLearning.Interfaces;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.ContinualLearning.Strategies;

/// <summary>
/// Configuration options for Expected Gradient Length strategy.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class EGLOptions<T>
{
    /// <summary>
    /// Gets or sets the regularization strength (lambda).
    /// Higher values = more protection of previous task knowledge.
    /// </summary>
    public T? Lambda { get; set; }

    /// <summary>
    /// Gets or sets the number of samples for gradient length estimation.
    /// More samples = better estimate but slower computation.
    /// </summary>
    public int? NumSamples { get; set; }

    /// <summary>
    /// Gets or sets whether to use squared gradient lengths.
    /// Squared lengths give more weight to larger gradients.
    /// </summary>
    public bool? UseSquaredLength { get; set; }

    /// <summary>
    /// Gets or sets the decay factor for online accumulation.
    /// Values closer to 1 give more weight to recent tasks.
    /// </summary>
    public double? DecayFactor { get; set; }

    /// <summary>
    /// Gets or sets whether to normalize importance values.
    /// Helps when different parameters have very different scales.
    /// </summary>
    public bool? NormalizeImportance { get; set; }

    /// <summary>
    /// Gets or sets the minimum importance value (to prevent division by zero).
    /// </summary>
    public double? MinImportanceValue { get; set; }
}

/// <summary>
/// Expected Gradient Length (EGL) strategy for continual learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> EGL protects important parameters by measuring how much
/// each parameter affects the output when training on a task. Parameters with larger
/// expected gradient lengths are considered more important.</para>
///
/// <para><b>How it works:</b></para>
/// <list type="number">
/// <item><description>During training, accumulate gradient lengths for each parameter</description></item>
/// <item><description>After each task, store the expected gradient length as importance</description></item>
/// <item><description>When learning new tasks, penalize changes to important parameters</description></item>
/// </list>
///
/// <para><b>The Math:</b></para>
/// <para>EGL loss: L_total = L_task + (λ/2) * Σᵢ Ωᵢ * (θᵢ - θ*ᵢ)²</para>
/// <para>Where Ωᵢ = E[|∂L/∂θᵢ|] is the expected gradient length for parameter i</para>
///
/// <para><b>Comparison to Other Methods:</b></para>
/// <list type="bullet">
/// <item><description><b>EWC:</b> Uses Fisher Information (squared gradients of log-likelihood)</description></item>
/// <item><description><b>MAS:</b> Uses output sensitivity (gradient of output magnitude)</description></item>
/// <item><description><b>SI:</b> Uses path integral (gradient * parameter change)</description></item>
/// <item><description><b>EGL:</b> Uses expected gradient length directly</description></item>
/// </list>
///
/// <para>EGL is simpler than EWC and provides a direct measure of how much each
/// parameter contributes to the loss during training.</para>
/// </remarks>
public class ExpectedGradientLength<T, TInput, TOutput> : ContinualLearningStrategyBase<T, TInput, TOutput>
{
    private readonly T _lambda;
    private readonly int _numSamples;
    private readonly bool _useSquaredLength;
    private readonly T _decayFactor;
    private readonly bool _normalizeImportance;
    private readonly T _minImportanceValue;

    // Accumulated importance weights
    private Vector<T>? _importance;

    // Parameters after each task
    private Vector<T>? _previousParameters;

    // Running gradient length accumulator
    private Vector<T>? _gradientLengthSum;
    private int _gradientCount;

    /// <summary>
    /// Initializes a new EGL strategy with default options.
    /// </summary>
    /// <param name="lossFunction">The loss function to use.</param>
    /// <param name="lambda">Regularization strength (higher = more protection).</param>
    /// <param name="numSamples">Number of samples for gradient length estimation.</param>
    public ExpectedGradientLength(
        ILossFunction<T> lossFunction,
        T lambda,
        int numSamples = 200)
        : this(lossFunction, new EGLOptions<T>
        {
            Lambda = lambda,
            NumSamples = numSamples
        })
    {
    }

    /// <summary>
    /// Initializes a new EGL strategy with custom options.
    /// </summary>
    /// <param name="lossFunction">The loss function to use.</param>
    /// <param name="options">Configuration options.</param>
    public ExpectedGradientLength(ILossFunction<T> lossFunction, EGLOptions<T>? options = null)
        : base(lossFunction)
    {
        var opts = options ?? new EGLOptions<T>();

        _lambda = opts.Lambda ?? NumOps.FromDouble(1.0);
        _numSamples = opts.NumSamples ?? 200;
        _useSquaredLength = opts.UseSquaredLength ?? false;
        _decayFactor = NumOps.FromDouble(opts.DecayFactor ?? 0.9);
        _normalizeImportance = opts.NormalizeImportance ?? true;
        _minImportanceValue = NumOps.FromDouble(opts.MinImportanceValue ?? 1e-8);
    }

    /// <inheritdoc/>
    public override string Name => "EGL";

    /// <inheritdoc/>
    public override bool RequiresMemoryBuffer => false;

    /// <inheritdoc/>
    public override bool ModifiesArchitecture => false;

    /// <inheritdoc/>
    public override long MemoryUsageBytes
    {
        get
        {
            long bytes = 0;
            bytes += EstimateVectorMemory(_importance);
            bytes += EstimateVectorMemory(_previousParameters);
            bytes += EstimateVectorMemory(_gradientLengthSum);
            return bytes;
        }
    }

    /// <inheritdoc/>
    public override void PrepareForTask(IFullModel<T, TInput, TOutput> model, IDataset<T, TInput, TOutput> taskData)
    {
        // Initialize gradient length accumulator for the new task
        int numParams = model.ParameterCount;
        _gradientLengthSum = new Vector<T>(numParams);
        for (int i = 0; i < numParams; i++)
        {
            _gradientLengthSum[i] = NumOps.Zero;
        }
        _gradientCount = 0;

        RecordMetric($"Task{TaskCount}_PrepareTime", DateTime.UtcNow);
    }

    /// <inheritdoc/>
    public override T ComputeRegularizationLoss(IFullModel<T, TInput, TOutput> model)
    {
        if (_importance == null || _previousParameters == null)
            return NumOps.Zero;

        var currentParams = model.GetParameters();

        if (currentParams.Length != _previousParameters.Length)
            throw new InvalidOperationException(
                $"Parameter dimension mismatch: current={currentParams.Length}, stored={_previousParameters.Length}");

        T loss = NumOps.Zero;

        // EGL loss: (lambda/2) * sum_i Omega_i * (theta_i - theta*_i)^2
        for (int i = 0; i < currentParams.Length; i++)
        {
            var diff = NumOps.Subtract(currentParams[i], _previousParameters[i]);
            var squaredDiff = NumOps.Multiply(diff, diff);
            var weightedDiff = NumOps.Multiply(_importance[i], squaredDiff);
            loss = NumOps.Add(loss, weightedDiff);
        }

        // Multiply by lambda/2
        var halfLambda = NumOps.Divide(_lambda, NumOps.FromDouble(2.0));
        return NumOps.Multiply(halfLambda, loss);
    }

    /// <inheritdoc/>
    public override Vector<T> AdjustGradients(Vector<T> gradients)
    {
        // Accumulate gradient lengths
        if (_gradientLengthSum != null && _gradientCount < _numSamples)
        {
            for (int i = 0; i < Math.Min(gradients.Length, _gradientLengthSum.Length); i++)
            {
                T gradLength;
                if (_useSquaredLength)
                {
                    // Use squared gradient: |∂L/∂θᵢ|²
                    gradLength = NumOps.Multiply(gradients[i], gradients[i]);
                }
                else
                {
                    // Use absolute gradient: |∂L/∂θᵢ|
                    double absVal = Math.Abs(Convert.ToDouble(gradients[i]));
                    gradLength = NumOps.FromDouble(absVal);
                }
                _gradientLengthSum[i] = NumOps.Add(_gradientLengthSum[i], gradLength);
            }
            _gradientCount++;
        }

        return gradients;
    }

    /// <inheritdoc/>
    public override void FinalizeTask(IFullModel<T, TInput, TOutput> model)
    {
        var currentParams = model.GetParameters();
        var taskImportance = ComputeTaskImportance();

        if (_normalizeImportance)
        {
            taskImportance = NormalizeImportanceVector(taskImportance);
        }

        if (_importance == null)
        {
            // First task: initialize importance and parameters
            _importance = taskImportance;
            _previousParameters = CloneVector(currentParams);
        }
        else
        {
            // Subsequent tasks: accumulate importance with decay
            UpdateAccumulatedImportance(taskImportance, currentParams);
        }

        TaskCount++;

        // Record metrics
        RecordMetric($"Task{TaskCount}_ImportanceMean", ComputeMean(_importance));
        RecordMetric($"Task{TaskCount}_ImportanceMax", Convert.ToDouble(ComputeMax(_importance)));
        RecordMetric($"Task{TaskCount}_ParamNorm", Convert.ToDouble(ComputeL2Norm(currentParams)));
        RecordMetric($"Task{TaskCount}_GradientSamples", _gradientCount);
    }

    /// <summary>
    /// Computes the importance for the current task from accumulated gradient lengths.
    /// </summary>
    private Vector<T> ComputeTaskImportance()
    {
        if (_gradientLengthSum == null || _gradientCount == 0)
        {
            throw new InvalidOperationException("No gradients were accumulated for importance computation");
        }

        var importance = new Vector<T>(_gradientLengthSum.Length);
        var sampleCount = NumOps.FromDouble(_gradientCount);

        for (int i = 0; i < _gradientLengthSum.Length; i++)
        {
            // Average gradient length = Expected gradient length
            var avgLength = NumOps.Divide(_gradientLengthSum[i], sampleCount);
            importance[i] = NumOps.Add(avgLength, _minImportanceValue);
        }

        return importance;
    }

    /// <summary>
    /// Updates accumulated importance with new task importance using exponential decay.
    /// </summary>
    private void UpdateAccumulatedImportance(Vector<T> taskImportance, Vector<T> currentParams)
    {
        // Online update: Ω = γ * Ω_old + Ω_new
        // Parameters: θ* = (γ * Ω_old * θ*_old + Ω_new * θ_new) / (γ * Ω_old + Ω_new)

        for (int i = 0; i < _importance!.Length; i++)
        {
            var decayedOldImportance = NumOps.Multiply(_decayFactor, _importance[i]);
            var newTotalImportance = NumOps.Add(decayedOldImportance, taskImportance[i]);

            // Weighted average of parameters
            var weightedOld = NumOps.Multiply(decayedOldImportance, _previousParameters![i]);
            var weightedNew = NumOps.Multiply(taskImportance[i], currentParams[i]);
            var numerator = NumOps.Add(weightedOld, weightedNew);

            // Avoid division by zero
            var safeDenom = NumOps.Add(newTotalImportance, _minImportanceValue);
            _previousParameters[i] = NumOps.Divide(numerator, safeDenom);

            // Update accumulated importance
            _importance[i] = newTotalImportance;
        }
    }

    /// <summary>
    /// Normalizes importance values to [0, 1] range.
    /// </summary>
    private Vector<T> NormalizeImportanceVector(Vector<T> importance)
    {
        var maxVal = ComputeMax(importance);
        if (Convert.ToDouble(maxVal) < 1e-10)
            return importance;

        var normalized = new Vector<T>(importance.Length);
        for (int i = 0; i < importance.Length; i++)
        {
            normalized[i] = NumOps.Divide(importance[i], maxVal);
        }
        return normalized;
    }

    /// <inheritdoc/>
    public override void Reset()
    {
        base.Reset();
        _importance = null;
        _previousParameters = null;
        _gradientLengthSum = null;
        _gradientCount = 0;
    }

    /// <inheritdoc/>
    protected override Dictionary<string, object> GetStateForSerialization()
    {
        var state = base.GetStateForSerialization();
        state["Lambda"] = Convert.ToDouble(_lambda);
        state["NumSamples"] = _numSamples;
        state["UseSquaredLength"] = _useSquaredLength;
        return state;
    }

    /// <summary>
    /// Gets the accumulated importance weights.
    /// </summary>
    public Vector<T>? Importance => _importance;

    /// <summary>
    /// Gets the stored parameters from previous tasks.
    /// </summary>
    public Vector<T>? PreviousParameters => _previousParameters;

    /// <summary>
    /// Gets the regularization strength (lambda).
    /// </summary>
    public T Lambda => _lambda;

    /// <summary>
    /// Computes the mean of a vector.
    /// </summary>
    private double ComputeMean(Vector<T> vector)
    {
        double sum = 0;
        for (int i = 0; i < vector.Length; i++)
        {
            sum += Convert.ToDouble(vector[i]);
        }
        return sum / vector.Length;
    }

    /// <summary>
    /// Computes the maximum value in a vector.
    /// </summary>
    private T ComputeMax(Vector<T> vector)
    {
        T max = vector[0];
        for (int i = 1; i < vector.Length; i++)
        {
            if (Convert.ToDouble(vector[i]) > Convert.ToDouble(max))
                max = vector[i];
        }
        return max;
    }
}
