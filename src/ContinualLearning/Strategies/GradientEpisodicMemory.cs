using System.Text.Json;
using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.ContinualLearning.Interfaces;
using AiDotNet.ContinualLearning.Memory;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.ContinualLearning.Strategies;

/// <summary>
/// Configuration options for Gradient Episodic Memory strategies.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> These options control how GEM prevents forgetting by
/// constraining gradients to not harm previous task performance.</para>
/// </remarks>
public class GEMOptions<T>
{
    /// <summary>
    /// Number of examples to store per task. Default: 256.
    /// </summary>
    /// <remarks>
    /// <para>More examples = better gradient estimates but higher memory cost.</para>
    /// <para><b>Industry Standard:</b> 256-1000 examples per task is common.</para>
    /// </remarks>
    public int? MemorySizePerTask { get; set; }

    /// <summary>
    /// Margin/epsilon for constraint violations. Default: 0.0.
    /// </summary>
    /// <remarks>
    /// <para>Positive values allow small constraint violations for faster convergence.</para>
    /// <para>Negative values require strict improvement on all tasks.</para>
    /// <para><b>Reference:</b> Lopez-Paz et al. used 0.5 in their experiments.</para>
    /// </remarks>
    public double? Margin { get; set; }

    /// <summary>
    /// Whether to use Averaged GEM (A-GEM) instead of full GEM. Default: false.
    /// </summary>
    /// <remarks>
    /// <para><b>A-GEM:</b> Projects gradient onto the average reference gradient from a
    /// random sample of memory, instead of checking all task constraints individually.</para>
    /// <para><b>Advantage:</b> O(1) complexity vs O(t) for t tasks.</para>
    /// <para><b>Disadvantage:</b> Weaker guarantees but much faster.</para>
    /// <para><b>Reference:</b> Chaudhry et al. "Efficient Lifelong Learning with A-GEM" (2019)</para>
    /// </remarks>
    public bool? UseAGEM { get; set; }

    /// <summary>
    /// Batch size for gradient computation from memory. Default: 64.
    /// </summary>
    /// <remarks>
    /// <para>Larger batches give more accurate gradient estimates but are slower.</para>
    /// </remarks>
    public int? GradientBatchSize { get; set; }

    /// <summary>
    /// Maximum number of iterations for the QP solver. Default: 100.
    /// </summary>
    /// <remarks>
    /// <para>The gradient projection is solved via quadratic programming.
    /// More iterations = more accurate solution but slower.</para>
    /// </remarks>
    public int? MaxQPIterations { get; set; }

    /// <summary>
    /// Tolerance for QP solver convergence. Default: 1e-6.
    /// </summary>
    public double? QPTolerance { get; set; }

    /// <summary>
    /// Whether to normalize gradients before checking constraints. Default: true.
    /// </summary>
    /// <remarks>
    /// <para>Normalization helps when task gradients have very different magnitudes.</para>
    /// </remarks>
    public bool? NormalizeGradients { get; set; }

    /// <summary>
    /// Memory sampling strategy for storing examples. Default: Reservoir.
    /// </summary>
    public MemorySamplingStrategy? MemorySamplingStrategy { get; set; }

    /// <summary>
    /// Replay sampling strategy for gradient computation. Default: TaskBalanced.
    /// </summary>
    public ReplaySamplingStrategy? ReplaySamplingStrategy { get; set; }

    /// <summary>
    /// Random seed for reproducibility. Default: null (use secure random).
    /// </summary>
    public int? RandomSeed { get; set; }
}

/// <summary>
/// Gradient Episodic Memory (GEM) strategy for continual learning with gradient constraints.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> GEM prevents forgetting by ensuring that parameter updates
/// for new tasks don't increase the loss on any previous task. It does this by:
/// 1. Storing representative examples from each task (episodic memory)
/// 2. Computing reference gradients for each previous task
/// 3. Projecting the current gradient to satisfy all task constraints
/// </para>
///
/// <para><b>How Gradient Projection Works:</b>
/// If the current gradient g would increase loss on task k (i.e., g · g_k &lt; 0), we project
/// g to the closest gradient g' such that g' · g_k ≥ 0 for all previous tasks.
/// This is a Quadratic Programming (QP) problem.</para>
///
/// <para><b>GEM vs A-GEM:</b>
/// <list type="bullet">
/// <item><description><b>GEM:</b> Checks all task constraints individually. Strong guarantees but O(t) complexity.</description></item>
/// <item><description><b>A-GEM:</b> Uses average reference gradient. O(1) but weaker guarantees.</description></item>
/// </list>
/// </para>
///
/// <para><b>Advantages:</b>
/// <list type="bullet">
/// <item><description>Strong theoretical guarantees (never increases loss on previous tasks)</description></item>
/// <item><description>Works with limited memory</description></item>
/// <item><description>Applicable to any gradient-based optimizer</description></item>
/// </list>
/// </para>
///
/// <para><b>Disadvantages:</b>
/// <list type="bullet">
/// <item><description>Requires solving QP for gradient projection</description></item>
/// <item><description>Needs access to gradients (not just loss values)</description></item>
/// <item><description>Memory scales linearly with number of tasks</description></item>
/// </list>
/// </para>
///
/// <para><b>References:</b>
/// <list type="bullet">
/// <item><description>Lopez-Paz &amp; Ranzato "Gradient Episodic Memory for Continual Learning" (NeurIPS 2017)</description></item>
/// <item><description>Chaudhry et al. "Efficient Lifelong Learning with A-GEM" (ICLR 2019)</description></item>
/// </list>
/// </para>
/// </remarks>
public class GradientEpisodicMemory<T, TInput, TOutput>
    : ContinualLearningStrategyBase<T, TInput, TOutput>,
      IGradientConstraintStrategy<T, TInput, TOutput>
{
    // Configuration
    private readonly int _memorySizePerTask;
    private readonly T _margin;
    private readonly bool _useAGEM;
    private readonly int _gradientBatchSize;
    private readonly int _maxQPIterations;
    private readonly double _qpTolerance;
    private readonly bool _normalizeGradients;

    // Task gradients for constraint checking
    private readonly List<Vector<T>> _taskGradients;

    // Memory buffer for storing examples
    private readonly ExperienceReplayBuffer<T, TInput, TOutput> _memoryBuffer;

    // Metrics tracking
    private int _totalProjections;
    private int _violationCount;
    private double _averageProjectionMagnitude;

    /// <summary>
    /// Initializes a new Gradient Episodic Memory strategy.
    /// </summary>
    /// <param name="lossFunction">The loss function to use.</param>
    /// <param name="options">Configuration options. If null, uses defaults.</param>
    public GradientEpisodicMemory(
        ILossFunction<T> lossFunction,
        GEMOptions<T>? options = null)
        : base(lossFunction)
    {
        options ??= new GEMOptions<T>();

        _memorySizePerTask = options.MemorySizePerTask ?? 256;
        _margin = NumOps.FromDouble(options.Margin ?? 0.0);
        _useAGEM = options.UseAGEM ?? false;
        _gradientBatchSize = options.GradientBatchSize ?? 64;
        _maxQPIterations = options.MaxQPIterations ?? 100;
        _qpTolerance = options.QPTolerance ?? 1e-6;
        _normalizeGradients = options.NormalizeGradients ?? true;

        var memorySamplingStrategy = options.MemorySamplingStrategy ?? Interfaces.MemorySamplingStrategy.Reservoir;
        var replaySamplingStrategy = options.ReplaySamplingStrategy ?? ReplaySamplingStrategy.TaskBalanced;

        _taskGradients = new List<Vector<T>>();
        _memoryBuffer = new ExperienceReplayBuffer<T, TInput, TOutput>(
            _memorySizePerTask * 100, // Total memory capacity for all tasks
            memorySamplingStrategy,
            replaySamplingStrategy,
            options.RandomSeed);

        _totalProjections = 0;
        _violationCount = 0;
        _averageProjectionMagnitude = 0.0;
    }

    /// <inheritdoc/>
    public override string Name => _useAGEM ? "A-GEM" : "GEM";

    /// <inheritdoc/>
    public override bool RequiresMemoryBuffer => true;

    /// <inheritdoc/>
    public override bool ModifiesArchitecture => false;

    /// <inheritdoc/>
    public override long MemoryUsageBytes
    {
        get
        {
            long gradientMemory = _taskGradients.Sum(g => EstimateVectorMemory(g));
            long bufferMemory = _memoryBuffer.EstimatedMemoryBytes;
            return gradientMemory + bufferMemory + 1024; // +1KB overhead
        }
    }

    #region IGradientConstraintStrategy Implementation

    /// <inheritdoc/>
    public int StoredExampleCount => _memoryBuffer.Count;

    /// <inheritdoc/>
    public int MaxExamples => _memoryBuffer.MaxSize;

    /// <inheritdoc/>
    public int StoredGradientCount => _taskGradients.Count;

    /// <inheritdoc/>
    public void StoreTaskExamples(IDataset<T, TInput, TOutput> taskData)
    {
        _memoryBuffer.AddTaskExamples(taskData, TaskCount, _memorySizePerTask);
    }

    /// <inheritdoc/>
    public IReadOnlyList<(TInput Input, TOutput Output, int TaskId)> SampleExamples(int batchSize)
    {
        var samples = _memoryBuffer.SampleBatch(batchSize);
        return samples.Select(dp => (dp.Input, dp.Output, dp.TaskId)).ToList();
    }

    /// <inheritdoc/>
    public void ClearMemory()
    {
        _memoryBuffer.Clear();
        _taskGradients.Clear();
    }

    /// <inheritdoc/>
    public void StoreTaskGradient(Vector<T> taskGradient)
    {
        if (taskGradient == null)
            throw new ArgumentNullException(nameof(taskGradient));

        var gradientToStore = _normalizeGradients ? NormalizeVector(taskGradient) : CloneVector(taskGradient);

        // Replace the most recent placeholder or add new
        if (_taskGradients.Count > TaskCount)
        {
            _taskGradients[TaskCount] = gradientToStore;
        }
        else
        {
            _taskGradients.Add(gradientToStore);
        }

        RecordMetric($"Task{TaskCount}_GradientNorm", Convert.ToDouble(ComputeL2Norm(gradientToStore)));
    }

    /// <inheritdoc/>
    public Vector<T> ProjectGradient(Vector<T> gradient)
    {
        if (_taskGradients.Count == 0)
            return gradient;

        if (_useAGEM)
        {
            return ProjectGradientAGEM(gradient);
        }
        else
        {
            return ProjectGradientGEM(gradient);
        }
    }

    /// <inheritdoc/>
    public bool ViolatesConstraint(Vector<T> gradient)
    {
        return _taskGradients.Any(taskGrad =>
            Convert.ToDouble(ComputeDotProduct(gradient, taskGrad)) < Convert.ToDouble(_margin));
    }

    #endregion

    #region IContinualLearningStrategy Implementation

    /// <inheritdoc/>
    public override void PrepareForTask(IFullModel<T, TInput, TOutput> model, IDataset<T, TInput, TOutput> taskData)
    {
        // GEM preparation: nothing special needed before training
        // The gradients from memory are computed on-demand during AdjustGradients
        RecordMetric($"Task{TaskCount}_TrainingSamples", taskData.Count);
    }

    /// <inheritdoc/>
    public override T ComputeRegularizationLoss(IFullModel<T, TInput, TOutput> model)
    {
        // GEM doesn't use regularization loss - it modifies gradients directly
        return NumOps.Zero;
    }

    /// <inheritdoc/>
    public override Vector<T> AdjustGradients(Vector<T> gradients)
    {
        if (_taskGradients.Count == 0)
            return gradients;

        var normalizedGradient = _normalizeGradients ? NormalizeVector(gradients) : gradients;
        bool violates = ViolatesConstraint(normalizedGradient);

        if (!violates)
        {
            return gradients; // No violation, use original gradients
        }

        _violationCount++;
        var projected = ProjectGradient(normalizedGradient);
        _totalProjections++;

        // Scale back to original magnitude if we normalized
        if (_normalizeGradients)
        {
            T originalMagnitude = ComputeL2Norm(gradients);
            T projectedMagnitude = ComputeL2Norm(projected);

            if (Convert.ToDouble(projectedMagnitude) > 1e-10)
            {
                T scale = NumOps.Divide(originalMagnitude, projectedMagnitude);
                projected = ScaleVector(projected, scale);
            }
        }

        // Track projection magnitude
        T projMag = ComputeL2Norm(SubtractVectors(projected, gradients));
        _averageProjectionMagnitude = (_averageProjectionMagnitude * (_totalProjections - 1) +
                                        Convert.ToDouble(projMag)) / _totalProjections;

        return projected;
    }

    /// <inheritdoc/>
    public override void FinalizeTask(IFullModel<T, TInput, TOutput> model)
    {
        // Store a placeholder gradient if not already provided
        // In practice, the trainer should call StoreTaskGradient with actual gradients
        if (_taskGradients.Count <= TaskCount)
        {
            var zeroGradient = new Vector<T>(model.ParameterCount);
            for (int i = 0; i < zeroGradient.Length; i++)
            {
                zeroGradient[i] = NumOps.Zero;
            }
            _taskGradients.Add(zeroGradient);
        }

        TaskCount++;
        RecordMetric($"Task{TaskCount - 1}_MemoryExamples", _memoryBuffer.GetTaskExamples(TaskCount - 1).Count);
    }

    /// <inheritdoc/>
    public override void Reset()
    {
        base.Reset();
        _taskGradients.Clear();
        _memoryBuffer.Clear();
        _totalProjections = 0;
        _violationCount = 0;
        _averageProjectionMagnitude = 0.0;
    }

    #endregion

    #region Gradient Projection Methods

    /// <summary>
    /// Projects gradient using full GEM (all task constraints).
    /// </summary>
    /// <remarks>
    /// <para>Solves: min ||g' - g||^2 s.t. g' · g_k ≥ margin ∀k</para>
    /// <para>Uses the dual formulation with iterative projected gradient descent.</para>
    /// </remarks>
    private Vector<T> ProjectGradientGEM(Vector<T> gradient)
    {
        // Find violated constraints
        var violations = new List<int>();
        for (int k = 0; k < _taskGradients.Count; k++)
        {
            T dotProduct = ComputeDotProduct(gradient, _taskGradients[k]);
            if (Convert.ToDouble(dotProduct) < Convert.ToDouble(_margin))
            {
                violations.Add(k);
            }
        }

        if (violations.Count == 0)
            return gradient;

        // Build the gradient matrix G for violated constraints: G[i,j] = g_i · g_j
        int numViolations = violations.Count;
        var G = new double[numViolations, numViolations];
        var a = new double[numViolations];

        for (int i = 0; i < numViolations; i++)
        {
            int ki = violations[i];
            a[i] = Convert.ToDouble(_margin) - Convert.ToDouble(ComputeDotProduct(gradient, _taskGradients[ki]));

            for (int j = 0; j < numViolations; j++)
            {
                int kj = violations[j];
                G[i, j] = Convert.ToDouble(ComputeDotProduct(_taskGradients[ki], _taskGradients[kj]));
            }
        }

        // Solve QP using projected gradient descent on dual
        var lambda = SolveQPDual(G, a);

        // Compute projected gradient: g' = g + sum(lambda_k * g_k)
        var projected = CloneVector(gradient);
        for (int i = 0; i < numViolations; i++)
        {
            if (lambda[i] > 1e-10)
            {
                int ki = violations[i];
                T lambdaT = NumOps.FromDouble(lambda[i]);
                for (int j = 0; j < projected.Length; j++)
                {
                    projected[j] = NumOps.Add(projected[j],
                        NumOps.Multiply(lambdaT, _taskGradients[ki][j]));
                }
            }
        }

        return projected;
    }

    /// <summary>
    /// Projects gradient using A-GEM (average reference gradient).
    /// </summary>
    /// <remarks>
    /// <para>A-GEM simplifies GEM by using a single average gradient constraint instead of
    /// checking all task constraints individually.</para>
    /// <para>Projection: g' = g - (g · g_ref / ||g_ref||^2) * g_ref when g · g_ref &lt; 0</para>
    /// </remarks>
    private Vector<T> ProjectGradientAGEM(Vector<T> gradient)
    {
        // Compute average reference gradient from random memory sample
        var refGradient = ComputeAverageReferenceGradient();
        if (refGradient == null)
            return gradient;

        T dotProduct = ComputeDotProduct(gradient, refGradient);
        T refNormSq = ComputeDotProduct(refGradient, refGradient);

        // If no violation, return original gradient
        if (Convert.ToDouble(dotProduct) >= Convert.ToDouble(_margin))
            return gradient;

        // Project: g' = g - (g · g_ref / ||g_ref||^2) * g_ref
        if (Convert.ToDouble(refNormSq) < 1e-10)
            return gradient;

        T factor = NumOps.Divide(dotProduct, refNormSq);
        var projected = CloneVector(gradient);
        for (int i = 0; i < projected.Length; i++)
        {
            projected[i] = NumOps.Subtract(projected[i],
                NumOps.Multiply(factor, refGradient[i]));
        }

        return projected;
    }

    /// <summary>
    /// Computes average gradient from a random sample of memory examples.
    /// </summary>
    private Vector<T>? ComputeAverageReferenceGradient()
    {
        if (_taskGradients.Count == 0)
            return null;

        // For A-GEM, we average all stored task gradients
        // (In full implementation, this would be computed from sampled memory examples)
        var avgGradient = new Vector<T>(_taskGradients[0].Length);
        for (int i = 0; i < avgGradient.Length; i++)
        {
            avgGradient[i] = NumOps.Zero;
        }

        foreach (var taskGrad in _taskGradients)
        {
            for (int i = 0; i < avgGradient.Length; i++)
            {
                avgGradient[i] = NumOps.Add(avgGradient[i], taskGrad[i]);
            }
        }

        T invCount = NumOps.FromDouble(1.0 / _taskGradients.Count);
        for (int i = 0; i < avgGradient.Length; i++)
        {
            avgGradient[i] = NumOps.Multiply(avgGradient[i], invCount);
        }

        return avgGradient;
    }

    /// <summary>
    /// Solves the dual QP using projected gradient descent.
    /// </summary>
    /// <remarks>
    /// <para>Dual problem: max -0.5 * λᵀGλ + aᵀλ s.t. λ ≥ 0</para>
    /// </remarks>
    private double[] SolveQPDual(double[,] G, double[] a)
    {
        int n = a.Length;
        var lambda = new double[n];
        double stepSize = 0.1;

        for (int iter = 0; iter < _maxQPIterations; iter++)
        {
            var gradDual = new double[n];
            double maxGrad = 0;

            // Compute dual gradient: grad = -Gλ + a
            for (int i = 0; i < n; i++)
            {
                gradDual[i] = a[i];
                for (int j = 0; j < n; j++)
                {
                    gradDual[i] -= G[i, j] * lambda[j];
                }
                maxGrad = Math.Max(maxGrad, Math.Abs(gradDual[i]));
            }

            // Check convergence
            if (maxGrad < _qpTolerance)
                break;

            // Projected gradient step
            for (int i = 0; i < n; i++)
            {
                lambda[i] = Math.Max(0, lambda[i] + stepSize * gradDual[i]);
            }
        }

        return lambda;
    }

    #endregion

    #region Helper Methods

    private Vector<T> NormalizeVector(Vector<T> vector)
    {
        T norm = ComputeL2Norm(vector);
        if (Convert.ToDouble(norm) < 1e-10)
            return CloneVector(vector);

        var normalized = new Vector<T>(vector.Length);
        for (int i = 0; i < vector.Length; i++)
        {
            normalized[i] = NumOps.Divide(vector[i], norm);
        }
        return normalized;
    }

    private Vector<T> ScaleVector(Vector<T> vector, T scale)
    {
        var scaled = new Vector<T>(vector.Length);
        for (int i = 0; i < vector.Length; i++)
        {
            scaled[i] = NumOps.Multiply(vector[i], scale);
        }
        return scaled;
    }

    private Vector<T> SubtractVectors(Vector<T> a, Vector<T> b)
    {
        var result = new Vector<T>(a.Length);
        for (int i = 0; i < a.Length; i++)
        {
            result[i] = NumOps.Subtract(a[i], b[i]);
        }
        return result;
    }

    #endregion

    #region State Serialization

    /// <inheritdoc/>
    protected override Dictionary<string, object> GetStateForSerialization()
    {
        var state = base.GetStateForSerialization();

        state["MemorySizePerTask"] = _memorySizePerTask;
        state["Margin"] = Convert.ToDouble(_margin);
        state["UseAGEM"] = _useAGEM;
        state["NormalizeGradients"] = _normalizeGradients;
        state["TotalProjections"] = _totalProjections;
        state["ViolationCount"] = _violationCount;
        state["AverageProjectionMagnitude"] = _averageProjectionMagnitude;

        // Save task gradients
        var gradients = new List<double[]>();
        foreach (var g in _taskGradients)
        {
            var arr = new double[g.Length];
            for (int i = 0; i < g.Length; i++)
            {
                arr[i] = Convert.ToDouble(g[i]);
            }
            gradients.Add(arr);
        }
        state["TaskGradients"] = gradients;

        return state;
    }

    /// <inheritdoc/>
    protected override void LoadStateFromSerialization(Dictionary<string, JsonElement> state)
    {
        base.LoadStateFromSerialization(state);

        if (state.TryGetValue("TotalProjections", out var projElement))
        {
            _totalProjections = projElement.GetInt32();
        }

        if (state.TryGetValue("ViolationCount", out var violElement))
        {
            _violationCount = violElement.GetInt32();
        }

        if (state.TryGetValue("AverageProjectionMagnitude", out var avgProjElement))
        {
            _averageProjectionMagnitude = avgProjElement.GetDouble();
        }

        if (state.TryGetValue("TaskGradients", out var gradientsElement))
        {
            _taskGradients.Clear();
            foreach (var gradElement in gradientsElement.EnumerateArray())
            {
                var arr = gradElement.EnumerateArray().Select(e => e.GetDouble()).ToArray();
                var grad = new Vector<T>(arr.Length);
                for (int i = 0; i < arr.Length; i++)
                {
                    grad[i] = NumOps.FromDouble(arr[i]);
                }
                _taskGradients.Add(grad);
            }
        }
    }

    #endregion

    /// <inheritdoc/>
    public override IReadOnlyDictionary<string, object> GetMetrics()
    {
        var baseMetrics = base.GetMetrics();
        var metrics = new Dictionary<string, object>(baseMetrics.Count + 8);
        foreach (var kvp in baseMetrics)
        {
            metrics[kvp.Key] = kvp.Value;
        }
        metrics["StoredExamples"] = StoredExampleCount;
        metrics["StoredGradients"] = StoredGradientCount;
        metrics["TotalProjections"] = _totalProjections;
        metrics["ViolationCount"] = _violationCount;
        metrics["ViolationRate"] = _totalProjections > 0 ? (double)_violationCount / _totalProjections : 0;
        metrics["AverageProjectionMagnitude"] = _averageProjectionMagnitude;
        metrics["UseAGEM"] = _useAGEM;
        metrics["MemorySizePerTask"] = _memorySizePerTask;

        // Add buffer statistics
        var bufferStats = _memoryBuffer.GetStatistics();
        metrics["BufferFillRatio"] = bufferStats.FillRatio;
        metrics["BufferTaskCount"] = bufferStats.TaskCount;

        return metrics;
    }

    /// <summary>
    /// Gets the episodic memory buffer.
    /// </summary>
    public ExperienceReplayBuffer<T, TInput, TOutput> MemoryBuffer => _memoryBuffer;
}
