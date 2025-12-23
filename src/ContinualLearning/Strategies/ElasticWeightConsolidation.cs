using System.Text.Json;
using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.ContinualLearning.Interfaces;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.ContinualLearning.Strategies;

/// <summary>
/// Configuration options for Elastic Weight Consolidation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class EWCOptions<T>
{
    /// <summary>
    /// Gets or sets the regularization strength (lambda).
    /// Higher values = more protection of previous task knowledge.
    /// </summary>
    /// <remarks>
    /// <para><b>Typical Values:</b></para>
    /// <list type="bullet">
    /// <item><description>1-10: Light protection, allows more plasticity</description></item>
    /// <item><description>100-1000: Strong protection, may limit learning new tasks</description></item>
    /// <item><description>10000+: Very strong protection, mostly preserves old knowledge</description></item>
    /// </list>
    /// <para>Default is 1000 as suggested in the original EWC paper.</para>
    /// </remarks>
    public T? Lambda { get; set; }

    /// <summary>
    /// Gets or sets the number of samples for Fisher Information estimation.
    /// More samples = better estimate but slower computation.
    /// </summary>
    public int? NumFisherSamples { get; set; }

    /// <summary>
    /// Gets or sets whether to use online EWC (accumulate Fisher info across tasks).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b></para>
    /// <list type="bullet">
    /// <item><description>Online EWC (true): Accumulates Fisher Information across tasks, single importance matrix</description></item>
    /// <item><description>Original EWC (false): Stores separate Fisher Information for each task</description></item>
    /// </list>
    /// <para>Online EWC is more memory efficient but may be less precise.</para>
    /// </remarks>
    public bool? UseOnlineEWC { get; set; }

    /// <summary>
    /// Gets or sets the decay factor for online EWC (gamma in the paper).
    /// Values closer to 1 give more weight to recent tasks.
    /// </summary>
    public double? OnlineDecayFactor { get; set; }

    /// <summary>
    /// Gets or sets whether to normalize Fisher Information values.
    /// Helps when different parameters have very different scales.
    /// </summary>
    public bool? NormalizeFisher { get; set; }

    /// <summary>
    /// Gets or sets the minimum Fisher Information value (to prevent division by zero).
    /// </summary>
    public double? MinFisherValue { get; set; }
}

/// <summary>
/// Elastic Weight Consolidation (EWC) strategy for continual learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> EWC protects important parameters from changing when learning new tasks.
/// Think of it like a spring (elastic) that pulls parameters back toward their optimal values
/// for previous tasks. The strength of the spring depends on how important each parameter was.</para>
///
/// <para><b>How it works:</b></para>
/// <list type="number">
/// <item><description>After learning a task, compute which parameters were important (Fisher Information)</description></item>
/// <item><description>Store the optimal parameter values for that task</description></item>
/// <item><description>When learning new tasks, add a penalty for changing important parameters</description></item>
/// </list>
///
/// <para><b>The Math:</b></para>
/// <para>EWC loss: L_total = L_task + (λ/2) * Σᵢ Fᵢ * (θᵢ - θ*ᵢ)²</para>
/// <para>Where:</para>
/// <list type="bullet">
/// <item><description>L_task: The loss on the current task</description></item>
/// <item><description>λ: Regularization strength (how much to protect old tasks)</description></item>
/// <item><description>Fᵢ: Fisher Information for parameter i (importance)</description></item>
/// <item><description>θᵢ: Current parameter value</description></item>
/// <item><description>θ*ᵢ: Optimal parameter value from previous task</description></item>
/// </list>
///
/// <para><b>Variants Supported:</b></para>
/// <list type="bullet">
/// <item><description><b>Original EWC:</b> Stores separate Fisher for each task (more memory, precise)</description></item>
/// <item><description><b>Online EWC:</b> Accumulates Fisher across tasks (less memory, efficient)</description></item>
/// </list>
///
/// <para><b>Reference:</b> Kirkpatrick et al. "Overcoming catastrophic forgetting in neural networks" (2017)</para>
/// <para><b>Online EWC Reference:</b> Schwarz et al. "Progress & Compress" (2018)</para>
/// </remarks>
public class ElasticWeightConsolidation<T, TInput, TOutput> : ContinualLearningStrategyBase<T, TInput, TOutput>
{
    private readonly T _lambda;
    private readonly int _numFisherSamples;
    private readonly bool _useOnlineEWC;
    private readonly T _onlineDecayFactor;
    private readonly bool _normalizeFisher;
    private readonly T _minFisherValue;

    // For original EWC: store parameters and Fisher for each task
    private readonly List<Vector<T>> _taskParameters;
    private readonly List<Vector<T>> _taskFisherInfo;

    // For online EWC: single accumulated importance matrix
    private Vector<T>? _accumulatedFisher;
    private Vector<T>? _consolidatedParameters;

    // Cached gradients for Fisher computation
    private readonly List<Vector<T>> _gradientCache;

    /// <summary>
    /// Initializes a new EWC strategy with default options.
    /// </summary>
    /// <param name="lossFunction">The loss function to use.</param>
    /// <param name="lambda">Regularization strength (higher = more protection).</param>
    /// <param name="numFisherSamples">Number of samples for Fisher estimation.</param>
    public ElasticWeightConsolidation(
        ILossFunction<T> lossFunction,
        T lambda,
        int numFisherSamples = 200)
        : this(lossFunction, new EWCOptions<T>
        {
            Lambda = lambda,
            NumFisherSamples = numFisherSamples
        })
    {
    }

    /// <summary>
    /// Initializes a new EWC strategy with custom options.
    /// </summary>
    /// <param name="lossFunction">The loss function to use.</param>
    /// <param name="options">Configuration options.</param>
    public ElasticWeightConsolidation(ILossFunction<T> lossFunction, EWCOptions<T>? options = null)
        : base(lossFunction)
    {
        var opts = options ?? new EWCOptions<T>();

        _lambda = opts.Lambda ?? NumOps.FromDouble(1000.0);
        _numFisherSamples = opts.NumFisherSamples ?? 200;
        _useOnlineEWC = opts.UseOnlineEWC ?? false;
        _onlineDecayFactor = NumOps.FromDouble(opts.OnlineDecayFactor ?? 0.9);
        _normalizeFisher = opts.NormalizeFisher ?? true;
        _minFisherValue = NumOps.FromDouble(opts.MinFisherValue ?? 1e-8);

        _taskParameters = new List<Vector<T>>();
        _taskFisherInfo = new List<Vector<T>>();
        _gradientCache = new List<Vector<T>>();
    }

    /// <inheritdoc/>
    public override string Name => _useOnlineEWC ? "Online-EWC" : "EWC";

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

            // Task-specific storage
            foreach (var p in _taskParameters)
                bytes += EstimateVectorMemory(p);
            foreach (var f in _taskFisherInfo)
                bytes += EstimateVectorMemory(f);

            // Online EWC storage
            bytes += EstimateVectorMemory(_accumulatedFisher);
            bytes += EstimateVectorMemory(_consolidatedParameters);

            // Gradient cache
            foreach (var g in _gradientCache)
                bytes += EstimateVectorMemory(g);

            return bytes;
        }
    }

    /// <inheritdoc/>
    public override void PrepareForTask(IFullModel<T, TInput, TOutput> model, IDataset<T, TInput, TOutput> taskData)
    {
        // Clear gradient cache for the new task
        _gradientCache.Clear();

        RecordMetric($"Task{TaskCount}_PrepareTime", DateTime.UtcNow);
    }

    /// <inheritdoc/>
    public override T ComputeRegularizationLoss(IFullModel<T, TInput, TOutput> model)
    {
        if (_useOnlineEWC)
        {
            return ComputeOnlineEWCLoss(model);
        }
        else
        {
            return ComputeOriginalEWCLoss(model);
        }
    }

    /// <summary>
    /// Computes EWC loss using the original formulation (separate Fisher per task).
    /// </summary>
    private T ComputeOriginalEWCLoss(IFullModel<T, TInput, TOutput> model)
    {
        if (_taskParameters.Count == 0)
            return NumOps.Zero;

        var currentParams = model.GetParameters();
        T totalLoss = NumOps.Zero;

        // Sum over all previous tasks
        for (int task = 0; task < _taskParameters.Count; task++)
        {
            var optimalParams = _taskParameters[task];
            var fisherInfo = _taskFisherInfo[task];

            if (currentParams.Length != optimalParams.Length)
                throw new InvalidOperationException(
                    $"Parameter dimension mismatch: current={currentParams.Length}, task{task}={optimalParams.Length}");

            // EWC loss for this task: (lambda/2) * sum_i F_i * (theta_i - theta*_i)^2
            for (int i = 0; i < currentParams.Length; i++)
            {
                var diff = NumOps.Subtract(currentParams[i], optimalParams[i]);
                var squaredDiff = NumOps.Multiply(diff, diff);
                var weightedDiff = NumOps.Multiply(fisherInfo[i], squaredDiff);
                totalLoss = NumOps.Add(totalLoss, weightedDiff);
            }
        }

        // Multiply by lambda/2
        var halfLambda = NumOps.Divide(_lambda, NumOps.FromDouble(2.0));
        return NumOps.Multiply(halfLambda, totalLoss);
    }

    /// <summary>
    /// Computes EWC loss using online formulation (accumulated Fisher).
    /// </summary>
    private T ComputeOnlineEWCLoss(IFullModel<T, TInput, TOutput> model)
    {
        if (_accumulatedFisher == null || _consolidatedParameters == null)
            return NumOps.Zero;

        var currentParams = model.GetParameters();

        if (currentParams.Length != _consolidatedParameters.Length)
            throw new InvalidOperationException("Parameter dimension mismatch");

        T loss = NumOps.Zero;

        for (int i = 0; i < currentParams.Length; i++)
        {
            var diff = NumOps.Subtract(currentParams[i], _consolidatedParameters[i]);
            var squaredDiff = NumOps.Multiply(diff, diff);
            var weightedDiff = NumOps.Multiply(_accumulatedFisher[i], squaredDiff);
            loss = NumOps.Add(loss, weightedDiff);
        }

        var halfLambda = NumOps.Divide(_lambda, NumOps.FromDouble(2.0));
        return NumOps.Multiply(halfLambda, loss);
    }

    /// <inheritdoc/>
    public override Vector<T> AdjustGradients(Vector<T> gradients)
    {
        // EWC doesn't adjust gradients directly - regularization loss adds gradient contribution
        // through automatic differentiation/backpropagation

        // Cache gradient for Fisher computation
        _gradientCache.Add(CloneVector(gradients));

        // Limit cache size
        while (_gradientCache.Count > _numFisherSamples)
        {
            _gradientCache.RemoveAt(0);
        }

        return gradients;
    }

    /// <inheritdoc/>
    public override void FinalizeTask(IFullModel<T, TInput, TOutput> model)
    {
        var currentParams = model.GetParameters();
        var fisherInfo = ComputeFisherInformation(model);

        if (_normalizeFisher)
        {
            fisherInfo = NormalizeFisherInformation(fisherInfo);
        }

        if (_useOnlineEWC)
        {
            // Online EWC: accumulate Fisher Information
            UpdateOnlineEWC(currentParams, fisherInfo);
        }
        else
        {
            // Original EWC: store separate Fisher for each task
            _taskParameters.Add(CloneVector(currentParams));
            _taskFisherInfo.Add(fisherInfo);
        }

        TaskCount++;
        _gradientCache.Clear();

        // Record metrics
        RecordMetric($"Task{TaskCount}_FisherMean", ComputeMean(fisherInfo));
        RecordMetric($"Task{TaskCount}_FisherMax", Convert.ToDouble(ComputeMax(fisherInfo)));
        RecordMetric($"Task{TaskCount}_ParamNorm", Convert.ToDouble(ComputeL2Norm(currentParams)));
    }

    /// <summary>
    /// Updates the online EWC state with new task information.
    /// </summary>
    private void UpdateOnlineEWC(Vector<T> currentParams, Vector<T> newFisher)
    {
        if (_accumulatedFisher == null)
        {
            _accumulatedFisher = newFisher;
            _consolidatedParameters = CloneVector(currentParams);
        }
        else
        {
            // Update consolidated parameters: θ* = (γ·F_old·θ*_old + F_new·θ_new) / (γ·F_old + F_new)
            // Update accumulated Fisher: F = γ·F_old + F_new

            for (int i = 0; i < _accumulatedFisher.Length; i++)
            {
                var decayedOldFisher = NumOps.Multiply(_onlineDecayFactor, _accumulatedFisher[i]);
                var newTotalFisher = NumOps.Add(decayedOldFisher, newFisher[i]);

                // Weighted average of parameters
                var weightedOld = NumOps.Multiply(decayedOldFisher, _consolidatedParameters![i]);
                var weightedNew = NumOps.Multiply(newFisher[i], currentParams[i]);
                var numerator = NumOps.Add(weightedOld, weightedNew);

                // Avoid division by zero
                var safeDenom = NumOps.Add(newTotalFisher, _minFisherValue);
                _consolidatedParameters[i] = NumOps.Divide(numerator, safeDenom);

                // Update accumulated Fisher
                _accumulatedFisher[i] = newTotalFisher;
            }
        }
    }

    /// <summary>
    /// Computes the diagonal Fisher Information Matrix from cached gradients.
    /// </summary>
    private Vector<T> ComputeFisherInformation(IFullModel<T, TInput, TOutput> model)
    {
        int numParams = model.ParameterCount;
        var fisher = new Vector<T>(numParams);

        // Initialize with minimum value
        for (int i = 0; i < numParams; i++)
        {
            fisher[i] = _minFisherValue;
        }

        if (_gradientCache.Count == 0)
        {
            // No gradients cached - use uniform importance
            for (int i = 0; i < numParams; i++)
            {
                fisher[i] = NumOps.FromDouble(1.0);
            }
            return fisher;
        }

        // Fisher Information = E[(gradient)^2]
        // Empirical estimate using cached gradients
        foreach (var gradient in _gradientCache)
        {
            for (int i = 0; i < Math.Min(numParams, gradient.Length); i++)
            {
                var squaredGrad = NumOps.Multiply(gradient[i], gradient[i]);
                fisher[i] = NumOps.Add(fisher[i], squaredGrad);
            }
        }

        // Average over samples
        var sampleCount = NumOps.FromDouble(_gradientCache.Count);
        for (int i = 0; i < numParams; i++)
        {
            fisher[i] = NumOps.Divide(fisher[i], sampleCount);
        }

        return fisher;
    }

    /// <summary>
    /// Normalizes Fisher Information to prevent numerical issues.
    /// </summary>
    private Vector<T> NormalizeFisherInformation(Vector<T> fisher)
    {
        var maxVal = ComputeMax(fisher);
        if (Convert.ToDouble(maxVal) < 1e-10)
            return fisher;

        var normalized = new Vector<T>(fisher.Length);
        for (int i = 0; i < fisher.Length; i++)
        {
            normalized[i] = NumOps.Divide(fisher[i], maxVal);
        }
        return normalized;
    }

    /// <inheritdoc/>
    public override void Reset()
    {
        base.Reset();
        _taskParameters.Clear();
        _taskFisherInfo.Clear();
        _accumulatedFisher = null;
        _consolidatedParameters = null;
        _gradientCache.Clear();
    }

    /// <inheritdoc/>
    protected override Dictionary<string, object> GetStateForSerialization()
    {
        var state = base.GetStateForSerialization();
        state["Lambda"] = Convert.ToDouble(_lambda);
        state["NumFisherSamples"] = _numFisherSamples;
        state["UseOnlineEWC"] = _useOnlineEWC;
        state["TaskParametersCount"] = _taskParameters.Count;
        return state;
    }

    /// <summary>
    /// Gets the stored optimal parameters from previous tasks.
    /// </summary>
    public IReadOnlyList<Vector<T>> OptimalParameters => _taskParameters.AsReadOnly();

    /// <summary>
    /// Gets the stored Fisher Information from previous tasks.
    /// </summary>
    public IReadOnlyList<Vector<T>> FisherInformation => _taskFisherInfo.AsReadOnly();

    /// <summary>
    /// Gets the accumulated Fisher Information (for online EWC).
    /// </summary>
    public Vector<T>? AccumulatedFisher => _accumulatedFisher;

    /// <summary>
    /// Gets the consolidated parameters (for online EWC).
    /// </summary>
    public Vector<T>? ConsolidatedParameters => _consolidatedParameters;

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
