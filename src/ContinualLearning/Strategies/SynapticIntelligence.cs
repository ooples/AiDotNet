using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.ContinualLearning.Interfaces;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using System.Text.Json;

namespace AiDotNet.ContinualLearning.Strategies;

/// <summary>
/// Configuration options for Synaptic Intelligence.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class SIOptions<T>
{
    /// <summary>
    /// Gets or sets the regularization strength (c in the paper).
    /// Higher values = more protection of previous task knowledge.
    /// </summary>
    /// <remarks>
    /// <para><b>Typical Values:</b></para>
    /// <list type="bullet">
    /// <item><description>0.1-1.0: Light protection, allows more plasticity</description></item>
    /// <item><description>1.0-10.0: Moderate protection, good balance</description></item>
    /// <item><description>10.0+: Strong protection, may limit learning new tasks</description></item>
    /// </list>
    /// <para>Default is 1.0 as suggested in the original SI paper.</para>
    /// </remarks>
    public T? Lambda { get; set; }

    /// <summary>
    /// Gets or sets the damping constant (xi in the paper).
    /// Prevents division by zero when parameter changes are small.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When a parameter barely changes during training,
    /// we add this small value to prevent numerical instability.</para>
    /// <para>Default is 0.1 as recommended in the paper.</para>
    /// </remarks>
    public double? Damping { get; set; }

    /// <summary>
    /// Gets or sets whether to normalize importance values.
    /// Helps when different parameters have very different scales.
    /// </summary>
    public bool? NormalizeImportance { get; set; }

    /// <summary>
    /// Gets or sets the importance accumulation mode.
    /// </summary>
    /// <remarks>
    /// <list type="bullet">
    /// <item><description><b>Sum:</b> Simply sum importance across tasks</description></item>
    /// <item><description><b>Max:</b> Keep maximum importance (more aggressive protection)</description></item>
    /// <item><description><b>WeightedSum:</b> Exponentially weight recent tasks more</description></item>
    /// </list>
    /// </remarks>
    public ImportanceAccumulationMode? AccumulationMode { get; set; }

    /// <summary>
    /// Gets or sets the decay factor for weighted accumulation.
    /// Values closer to 1 give more weight to recent tasks.
    /// </summary>
    public double? DecayFactor { get; set; }

    /// <summary>
    /// Gets or sets whether to use running average for path integral.
    /// Can help with numerical stability for long training runs.
    /// </summary>
    public bool? UseRunningAverage { get; set; }

    /// <summary>
    /// Gets or sets the minimum importance value to prevent underflow.
    /// </summary>
    public double? MinImportanceValue { get; set; }

    /// <summary>
    /// Gets or sets whether to track per-layer importance statistics.
    /// </summary>
    public bool? TrackLayerStatistics { get; set; }
}

/// <summary>
/// Mode for accumulating importance across tasks.
/// </summary>
public enum ImportanceAccumulationMode
{
    /// <summary>
    /// Sum importance values across all tasks.
    /// </summary>
    Sum,

    /// <summary>
    /// Keep maximum importance value (most conservative).
    /// </summary>
    Max,

    /// <summary>
    /// Exponentially weighted sum favoring recent tasks.
    /// </summary>
    WeightedSum
}

/// <summary>
/// Synaptic Intelligence (SI) strategy for continual learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Synaptic Intelligence is similar to EWC but estimates weight
/// importance online during training rather than computing Fisher Information afterward.
/// It tracks how much each weight contributes to loss reduction using a "path integral".</para>
///
/// <para><b>Key Insight:</b> SI measures how much each parameter contributed to learning,
/// not just how important it is for the current solution. This is done by integrating
/// the gradient signal along the learning trajectory.</para>
///
/// <para><b>How it works:</b></para>
/// <list type="number">
/// <item><description>During training, track the running sum: ω_i += -gradient_i × Δθ_i</description></item>
/// <item><description>After each task, compute importance: Ω_i = ω_i / (Δθ_i² + ξ)</description></item>
/// <item><description>When learning new tasks, penalize changes: L = L_task + (c/2) × Σ Ω_i × (θ_i - θ*_i)²</description></item>
/// </list>
///
/// <para><b>The Math (Path Integral):</b></para>
/// <para>ω_i = Σ_t (-∂L/∂θ_i(t)) × (θ_i(t) - θ_i(t-1))</para>
/// <para>This approximates the contribution of parameter i to loss reduction.</para>
///
/// <para><b>Advantages over EWC:</b></para>
/// <list type="bullet">
/// <item><description>Online computation - no need to store training data after task completion</description></item>
/// <item><description>Lower memory overhead than Fisher Information computation</description></item>
/// <item><description>Naturally handles streaming data scenarios</description></item>
/// <item><description>Captures dynamics of learning, not just final importance</description></item>
/// </list>
///
/// <para><b>Reference:</b> Zenke, F., Poole, B., and Ganguli, S.
/// "Continual Learning Through Synaptic Intelligence" (2017). ICML.</para>
/// </remarks>
public class SynapticIntelligence<T, TInput, TOutput> : ContinualLearningStrategyBase<T, TInput, TOutput>
{
    private readonly T _lambda;
    private readonly T _damping;
    private readonly bool _normalizeImportance;
    private readonly ImportanceAccumulationMode _accumulationMode;
    private readonly T _decayFactor;
    private readonly bool _useRunningAverage;
    private readonly T _minImportanceValue;
    private readonly bool _trackLayerStatistics;

    // Consolidated importance across tasks (Ω in the paper)
    private Vector<T>? _omega;

    // Parameters at the start of current task (θ*)
    private Vector<T>? _taskStartParameters;

    // Running sum of gradient × parameter change during task (ω in the paper)
    private Vector<T>? _pathIntegral;

    // Previous gradients for computing parameter changes
    private Vector<T>? _lastGradients;

    // Previous parameters for delta computation
    private Vector<T>? _lastParameters;

    // Whether we're currently tracking a task
    private bool _isTrackingTask;

    // Running average normalization factor
    private int _updateCount;

    // Layer-wise statistics for debugging
    private readonly Dictionary<int, double> _layerImportance;

    /// <summary>
    /// Initializes a new SI strategy with a lambda value.
    /// </summary>
    /// <param name="lossFunction">The loss function to use.</param>
    /// <param name="lambda">Regularization strength (higher = more protection).</param>
    public SynapticIntelligence(ILossFunction<T> lossFunction, T lambda)
        : this(lossFunction, new SIOptions<T> { Lambda = lambda })
    {
    }

    /// <summary>
    /// Initializes a new SI strategy with custom options.
    /// </summary>
    /// <param name="lossFunction">The loss function to use.</param>
    /// <param name="options">Configuration options.</param>
    public SynapticIntelligence(ILossFunction<T> lossFunction, SIOptions<T>? options = null)
        : base(lossFunction)
    {
        var opts = options ?? new SIOptions<T>();

        _lambda = opts.Lambda ?? NumOps.FromDouble(1.0);
        _damping = NumOps.FromDouble(opts.Damping ?? 0.1);
        _normalizeImportance = opts.NormalizeImportance ?? true;
        _accumulationMode = opts.AccumulationMode ?? ImportanceAccumulationMode.Sum;
        _decayFactor = NumOps.FromDouble(opts.DecayFactor ?? 0.9);
        _useRunningAverage = opts.UseRunningAverage ?? false;
        _minImportanceValue = NumOps.FromDouble(opts.MinImportanceValue ?? 1e-8);
        _trackLayerStatistics = opts.TrackLayerStatistics ?? false;

        _layerImportance = new Dictionary<int, double>();
        _isTrackingTask = false;
        _updateCount = 0;
    }

    /// <inheritdoc/>
    public override string Name => "Synaptic-Intelligence";

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
            bytes += EstimateVectorMemory(_omega);
            bytes += EstimateVectorMemory(_taskStartParameters);
            bytes += EstimateVectorMemory(_pathIntegral);
            bytes += EstimateVectorMemory(_lastGradients);
            bytes += EstimateVectorMemory(_lastParameters);
            bytes += _layerImportance.Count * 16; // int + double
            return bytes;
        }
    }

    /// <inheritdoc/>
    public override void PrepareForTask(IFullModel<T, TInput, TOutput> model, IDataset<T, TInput, TOutput> taskData)
    {
        var paramCount = model.ParameterCount;

        // Initialize omega if this is the first task
        if (_omega == null || _omega.Length != paramCount)
        {
            _omega = new Vector<T>(paramCount);
            for (int i = 0; i < paramCount; i++)
            {
                _omega[i] = _minImportanceValue;
            }
        }

        // Store parameters at the start of this task
        _taskStartParameters = CloneVector(model.GetParameters());

        // Reset path integral for new task
        _pathIntegral = new Vector<T>(paramCount);
        for (int i = 0; i < paramCount; i++)
        {
            _pathIntegral[i] = NumOps.Zero;
        }

        // Reset tracking state
        _lastGradients = null;
        _lastParameters = CloneVector(_taskStartParameters);
        _isTrackingTask = true;
        _updateCount = 0;

        RecordMetric($"Task{TaskCount}_PrepareTime", DateTime.UtcNow);
    }

    /// <inheritdoc/>
    public override T ComputeRegularizationLoss(IFullModel<T, TInput, TOutput> model)
    {
        if (_omega == null || _taskStartParameters == null || !HasPreviousTasks())
        {
            return NumOps.Zero;
        }

        var currentParams = model.GetParameters();

        if (currentParams.Length != _omega.Length)
        {
            throw new InvalidOperationException(
                $"Parameter dimension mismatch: current={currentParams.Length}, stored={_omega.Length}");
        }

        // SI loss: (λ/2) × Σ Ω_i × (θ_i - θ*_i)²
        T loss = NumOps.Zero;

        for (int i = 0; i < currentParams.Length; i++)
        {
            var diff = NumOps.Subtract(currentParams[i], _taskStartParameters[i]);
            var squaredDiff = NumOps.Multiply(diff, diff);
            var weightedDiff = NumOps.Multiply(_omega[i], squaredDiff);
            loss = NumOps.Add(loss, weightedDiff);
        }

        var halfLambda = NumOps.Divide(_lambda, NumOps.FromDouble(2.0));
        return NumOps.Multiply(halfLambda, loss);
    }

    /// <inheritdoc/>
    public override Vector<T> AdjustGradients(Vector<T> gradients)
    {
        if (!_isTrackingTask || _pathIntegral == null || _lastParameters == null)
        {
            _lastGradients = CloneVector(gradients);
            return gradients;
        }

        // Update path integral with current gradient and parameter change
        // ω_i += -g_i × Δθ_i (negative because we want contribution to loss reduction)
        // Note: We use the previous gradients with the parameter change since last gradient update
        if (_lastGradients != null)
        {
            UpdatePathIntegral(_lastParameters, gradients);
        }

        _lastGradients = CloneVector(gradients);
        _updateCount++;

        // Add SI regularization gradient to the task gradient
        if (_omega != null && _taskStartParameters != null && HasPreviousTasks())
        {
            var result = CloneVector(gradients);

            for (int i = 0; i < gradients.Length; i++)
            {
                // SI gradient: λ × Ω_i × (θ_i - θ*_i)
                var diff = NumOps.Subtract(_lastParameters[i], _taskStartParameters[i]);
                var siGrad = NumOps.Multiply(_omega[i], diff);
                siGrad = NumOps.Multiply(_lambda, siGrad);
                result[i] = NumOps.Add(result[i], siGrad);
            }

            return result;
        }

        return gradients;
    }

    /// <summary>
    /// Updates the path integral with current gradient and parameter change.
    /// </summary>
    private void UpdatePathIntegral(Vector<T> previousParams, Vector<T> currentGradients)
    {
        if (_pathIntegral == null)
            return;

        // Get current parameters (they've changed since previousParams)
        // We approximate current params as: previous + learning_rate * (-gradient)
        // But since we don't know the learning rate, we track the actual delta
        // In practice, this should be called with parameters AFTER optimizer step

        for (int i = 0; i < Math.Min(_pathIntegral.Length, currentGradients.Length); i++)
        {
            // ω_i += -g_i × Δθ_i
            // The parameter change happened between calls, so we use the gradient
            // that caused the change (previous gradient)
            if (_lastGradients != null && i < _lastGradients.Length)
            {
                // Use previous gradient with the parameter delta
                var negGrad = NumOps.Negate(_lastGradients[i]);
                // For now, we approximate parameter change as proportional to gradient
                // A more accurate implementation would get actual parameter change from optimizer
                var contribution = NumOps.Multiply(negGrad, currentGradients[i]);

                if (_useRunningAverage && _updateCount > 0)
                {
                    // Running average for numerical stability
                    var alpha = NumOps.FromDouble(1.0 / (_updateCount + 1));
                    var oneMinusAlpha = NumOps.Subtract(NumOps.FromDouble(1.0), alpha);
                    _pathIntegral[i] = NumOps.Add(
                        NumOps.Multiply(oneMinusAlpha, _pathIntegral[i]),
                        NumOps.Multiply(alpha, contribution));
                }
                else
                {
                    _pathIntegral[i] = NumOps.Add(_pathIntegral[i], contribution);
                }
            }
        }
    }

    /// <summary>
    /// Notifies SI of a parameter update (call this after optimizer.step()).
    /// </summary>
    /// <param name="currentParameters">The current parameter values after update.</param>
    public void NotifyParameterUpdate(Vector<T> currentParameters)
    {
        if (!_isTrackingTask || _pathIntegral == null || _lastParameters == null || _lastGradients == null)
        {
            _lastParameters = CloneVector(currentParameters);
            return;
        }

        // Compute actual parameter delta
        for (int i = 0; i < Math.Min(_pathIntegral.Length, currentParameters.Length); i++)
        {
            var delta = NumOps.Subtract(currentParameters[i], _lastParameters[i]);

            // ω_i += -g_i × Δθ_i
            var negGrad = NumOps.Negate(_lastGradients[i]);
            var contribution = NumOps.Multiply(negGrad, delta);

            if (_useRunningAverage && _updateCount > 0)
            {
                var alpha = NumOps.FromDouble(1.0 / (_updateCount + 1));
                var oneMinusAlpha = NumOps.Subtract(NumOps.FromDouble(1.0), alpha);
                _pathIntegral[i] = NumOps.Add(
                    NumOps.Multiply(oneMinusAlpha, _pathIntegral[i]),
                    NumOps.Multiply(alpha, contribution));
            }
            else
            {
                _pathIntegral[i] = NumOps.Add(_pathIntegral[i], contribution);
            }
        }

        _lastParameters = CloneVector(currentParameters);
        _updateCount++;
    }

    /// <inheritdoc/>
    public override void FinalizeTask(IFullModel<T, TInput, TOutput> model)
    {
        if (_omega == null || _pathIntegral == null || _taskStartParameters == null)
        {
            TaskCount++;
            _isTrackingTask = false;
            return;
        }

        var currentParams = model.GetParameters();

        // Compute task-specific importance and consolidate
        // Ω_i = ω_i / (Δθ_i² + ξ)
        var taskImportance = new Vector<T>(currentParams.Length);

        for (int i = 0; i < currentParams.Length; i++)
        {
            var delta = NumOps.Subtract(currentParams[i], _taskStartParameters[i]);
            var deltaSquared = NumOps.Multiply(delta, delta);
            var denominator = NumOps.Add(deltaSquared, _damping);
            var importance = NumOps.Divide(_pathIntegral[i], denominator);

            // Only keep positive importance (contributed to loss reduction)
            if (NumOps.GreaterThan(importance, NumOps.Zero))
            {
                taskImportance[i] = importance;
            }
            else
            {
                taskImportance[i] = _minImportanceValue;
            }
        }

        // Normalize importance if requested
        if (_normalizeImportance)
        {
            taskImportance = NormalizeImportanceValues(taskImportance);
        }

        // Accumulate into omega based on mode
        AccumulateImportance(taskImportance);

        // Store current parameters as the new optimal for future tasks
        _taskStartParameters = CloneVector(currentParams);

        // Track layer statistics if enabled
        if (_trackLayerStatistics)
        {
            ComputeLayerStatistics(taskImportance);
        }

        // Record metrics
        RecordMetric($"Task{TaskCount}_MeanImportance", ComputeMean(taskImportance));
        RecordMetric($"Task{TaskCount}_MaxImportance", ComputeMax(taskImportance));
        RecordMetric($"Task{TaskCount}_UpdateCount", _updateCount);
        RecordMetric($"Task{TaskCount}_NonZeroImportance", CountNonZero(taskImportance));

        TaskCount++;
        _isTrackingTask = false;
        _pathIntegral = null;
        _lastGradients = null;
        _updateCount = 0;
    }

    /// <summary>
    /// Accumulates task importance into the consolidated omega.
    /// </summary>
    private void AccumulateImportance(Vector<T> taskImportance)
    {
        if (_omega == null)
            return;

        switch (_accumulationMode)
        {
            case ImportanceAccumulationMode.Sum:
                for (int i = 0; i < _omega.Length; i++)
                {
                    _omega[i] = NumOps.Add(_omega[i], taskImportance[i]);
                }
                break;

            case ImportanceAccumulationMode.Max:
                for (int i = 0; i < _omega.Length; i++)
                {
                    if (NumOps.GreaterThan(taskImportance[i], _omega[i]))
                    {
                        _omega[i] = taskImportance[i];
                    }
                }
                break;

            case ImportanceAccumulationMode.WeightedSum:
                for (int i = 0; i < _omega.Length; i++)
                {
                    // Decay old importance and add new
                    var decayed = NumOps.Multiply(_decayFactor, _omega[i]);
                    _omega[i] = NumOps.Add(decayed, taskImportance[i]);
                }
                break;
        }
    }

    /// <summary>
    /// Normalizes importance values to prevent numerical issues.
    /// </summary>
    private Vector<T> NormalizeImportanceValues(Vector<T> importance)
    {
        T maxVal = _minImportanceValue;
        for (int i = 0; i < importance.Length; i++)
        {
            if (NumOps.GreaterThan(importance[i], maxVal))
            {
                maxVal = importance[i];
            }
        }

        double maxDouble = Convert.ToDouble(maxVal);
        if (maxDouble < 1e-10)
            return importance;

        var normalized = new Vector<T>(importance.Length);
        for (int i = 0; i < importance.Length; i++)
        {
            normalized[i] = NumOps.Divide(importance[i], maxVal);
        }
        return normalized;
    }

    /// <summary>
    /// Computes per-layer importance statistics.
    /// </summary>
    private void ComputeLayerStatistics(Vector<T> importance)
    {
        // This is a simplified version - in practice, you'd need layer boundary info
        // For now, compute statistics on chunks
        int chunkSize = Math.Max(1, importance.Length / 10);
        int layerIndex = 0;

        for (int start = 0; start < importance.Length; start += chunkSize)
        {
            int end = Math.Min(start + chunkSize, importance.Length);
            double sum = 0;
            for (int i = start; i < end; i++)
            {
                sum += Convert.ToDouble(importance[i]);
            }
            double mean = sum / (end - start);
            _layerImportance[layerIndex] = mean;
            layerIndex++;
        }
    }

    /// <summary>
    /// Checks if there are any previous tasks with non-zero importance.
    /// </summary>
    private bool HasPreviousTasks()
    {
        if (_omega == null || TaskCount == 0)
            return false;

        for (int i = 0; i < _omega.Length; i++)
        {
            if (NumOps.GreaterThan(_omega[i], _minImportanceValue))
            {
                return true;
            }
        }
        return false;
    }

    /// <inheritdoc/>
    public override void Reset()
    {
        base.Reset();
        _omega = null;
        _taskStartParameters = null;
        _pathIntegral = null;
        _lastGradients = null;
        _lastParameters = null;
        _isTrackingTask = false;
        _updateCount = 0;
        _layerImportance.Clear();
    }

    /// <inheritdoc/>
    protected override Dictionary<string, object> GetStateForSerialization()
    {
        var state = base.GetStateForSerialization();
        state["Lambda"] = Convert.ToDouble(_lambda);
        state["Damping"] = Convert.ToDouble(_damping);
        state["AccumulationMode"] = _accumulationMode.ToString();
        state["NormalizeImportance"] = _normalizeImportance;
        return state;
    }

    /// <summary>
    /// Gets the consolidated importance values.
    /// </summary>
    public Vector<T>? ConsolidatedImportance => _omega;

    /// <summary>
    /// Gets the optimal parameters from the last completed task.
    /// </summary>
    public Vector<T>? OptimalParameters => _taskStartParameters;

    /// <summary>
    /// Gets the regularization strength.
    /// </summary>
    public T Lambda => _lambda;

    /// <summary>
    /// Gets the damping constant.
    /// </summary>
    public T Damping => _damping;

    /// <summary>
    /// Gets per-layer importance statistics (if tracking enabled).
    /// </summary>
    public IReadOnlyDictionary<int, double> LayerImportance => _layerImportance;

    /// <summary>
    /// Gets whether the strategy is currently tracking a task.
    /// </summary>
    public bool IsTrackingTask => _isTrackingTask;

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
    private double ComputeMax(Vector<T> vector)
    {
        double max = Convert.ToDouble(vector[0]);
        for (int i = 1; i < vector.Length; i++)
        {
            double val = Convert.ToDouble(vector[i]);
            if (val > max)
                max = val;
        }
        return max;
    }

    /// <summary>
    /// Counts non-zero elements in a vector.
    /// </summary>
    private int CountNonZero(Vector<T> vector)
    {
        int count = 0;
        double minVal = Convert.ToDouble(_minImportanceValue);
        for (int i = 0; i < vector.Length; i++)
        {
            if (Convert.ToDouble(vector[i]) > minVal)
                count++;
        }
        return count;
    }
}
