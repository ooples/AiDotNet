using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.Attributes;
using AiDotNet.ContinualLearning.Interfaces;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.ContinualLearning.Strategies;

/// <summary>
/// Configuration options for Memory Aware Synapses.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class MASOptions<T>
{
    /// <summary>
    /// Gets or sets the regularization strength (lambda).
    /// Higher values = more protection of previous task knowledge.
    /// </summary>
    /// <remarks>
    /// <para><b>Typical Values:</b></para>
    /// <list type="bullet">
    /// <item><description>0.1-1.0: Light protection</description></item>
    /// <item><description>1.0-10.0: Moderate protection, recommended starting point</description></item>
    /// <item><description>10.0+: Strong protection</description></item>
    /// </list>
    /// <para>Default is 1.0 as suggested in the original MAS paper.</para>
    /// </remarks>
    public T? Lambda { get; set; }

    /// <summary>
    /// Gets or sets the number of samples to use for importance estimation.
    /// More samples = better estimate but slower computation.
    /// </summary>
    /// <remarks>
    /// <para>Default is 256. Use more samples for larger/more complex models.</para>
    /// </remarks>
    public int? NumSamples { get; set; }

    /// <summary>
    /// Gets or sets whether to normalize importance values.
    /// Helps when different parameters have very different scales.
    /// </summary>
    public bool? NormalizeImportance { get; set; }

    /// <summary>
    /// Gets or sets the importance computation mode.
    /// </summary>
    public MASImportanceMode? ImportanceMode { get; set; }

    /// <summary>
    /// Gets or sets the importance accumulation mode across tasks.
    /// </summary>
    public ImportanceAccumulationMode? AccumulationMode { get; set; }

    /// <summary>
    /// Gets or sets the decay factor for weighted accumulation.
    /// </summary>
    public double? DecayFactor { get; set; }

    /// <summary>
    /// Gets or sets whether to use mini-batching for importance computation.
    /// Can be more memory efficient for large datasets.
    /// </summary>
    public bool? UseBatching { get; set; }

    /// <summary>
    /// Gets or sets the batch size for importance computation.
    /// </summary>
    public int? BatchSize { get; set; }

    /// <summary>
    /// Gets or sets the minimum importance value to prevent underflow.
    /// </summary>
    public double? MinImportanceValue { get; set; }

    /// <summary>
    /// Gets or sets whether to use L1 norm instead of L2 for output sensitivity.
    /// L1 can be more robust to outliers.
    /// </summary>
    public bool? UseL1Norm { get; set; }
}

/// <summary>
/// Mode for computing parameter importance in MAS.
/// </summary>
public enum MASImportanceMode
{
    /// <summary>
    /// Original MAS: use gradient of output L2 norm.
    /// </summary>
    OutputSensitivity,

    /// <summary>
    /// Use gradient of output with respect to randomly sampled directions.
    /// </summary>
    RandomProjection,

    /// <summary>
    /// Use diagonal of Fisher Information (hybrid with EWC).
    /// </summary>
    FisherDiagonal,

    /// <summary>
    /// Hebbian-style importance based on activation magnitudes.
    /// </summary>
    Hebbian
}

/// <summary>
/// Memory Aware Synapses (MAS) strategy for continual learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> MAS estimates weight importance in an unsupervised way
/// by measuring how sensitive the network output is to each parameter. This means
/// it doesn't need task labels to compute importance - just unlabeled data!</para>
///
/// <para><b>Key Insight:</b> If changing a parameter causes a large change in the
/// network output, that parameter is important and should be protected. This is
/// measured by the gradient of the output norm with respect to each parameter.</para>
///
/// <para><b>How it works:</b></para>
/// <list type="number">
/// <item><description>After each task, compute output sensitivity for each parameter</description></item>
/// <item><description>Ω_i = (1/N) × Σ_n |∂||F(x_n)||²/∂θ_i|</description></item>
/// <item><description>When learning new tasks, penalize: L = L_task + (λ/2) × Σ Ω_i × (θ_i - θ*_i)²</description></item>
/// </list>
///
/// <para><b>The Math:</b></para>
/// <para>For each sample x_n:</para>
/// <para>1. Forward pass: y = F(x_n)</para>
/// <para>2. Compute output norm gradient: ∂||y||²/∂θ = 2y × (∂y/∂θ)</para>
/// <para>3. Importance: Ω_i = (1/N) × Σ_n |∂||y||²/∂θ_i|</para>
///
/// <para><b>Advantages over EWC:</b></para>
/// <list type="bullet">
/// <item><description>Unsupervised - doesn't need task labels, just input data</description></item>
/// <item><description>Can be computed on any unlabeled data distribution</description></item>
/// <item><description>Simpler than Fisher Information - no loss function needed</description></item>
/// <item><description>Works well for transfer learning scenarios</description></item>
/// </list>
///
/// <para><b>Reference:</b> Aljundi, R., Babiloni, F., Elhoseiny, M., Rohrbach, M., and Tuytelaars, T.
/// "Memory Aware Synapses: Learning what (not) to forget" (2018). ECCV.</para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Regularization)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Memory Aware Synapses: Learning What (Not) to Forget", "https://arxiv.org/abs/1711.09601", Year = 2018, Authors = "Rahaf Aljundi, Francesca Babiloni, Mohamed Elhoseiny, Marcus Rohrbach, Tinne Tuytelaars")]
[ComponentType(ComponentType.ContinualLearner)]
[PipelineStage(PipelineStage.Training)]
public class MemoryAwareSynapses<T, TInput, TOutput> : ContinualLearningStrategyBase<T, TInput, TOutput>
{
    private readonly T _lambda;
    private readonly int _numSamples;
    private readonly bool _normalizeImportance;
    private readonly MASImportanceMode _importanceMode;
    private readonly ImportanceAccumulationMode _accumulationMode;
    private readonly T _decayFactor;
    private readonly bool _useBatching;
    private readonly int _batchSize;
    private readonly T _minImportanceValue;
    private readonly bool _useL1Norm;

    // Accumulated importance across tasks (Ω)
    [AiDotNet.Attributes.Buffer]
    private Vector<T>? _omega;

    // Optimal parameters from the last completed task (θ*)
    private Vector<T>? _optimalParameters;

    // Cached inputs for importance computation
    private readonly List<TInput> _importanceInputs;

    // Per-task importance history for analysis
    private readonly List<Vector<T>> _taskImportanceHistory;

    /// <summary>
    /// Initializes a new MAS strategy with a lambda value.
    /// </summary>
    /// <param name="lossFunction">The loss function to use.</param>
    /// <param name="lambda">Regularization strength (higher = more protection).</param>
    public MemoryAwareSynapses(ILossFunction<T> lossFunction, T lambda)
        : this(lossFunction, new MASOptions<T> { Lambda = lambda })
    {
    }

    /// <summary>
    /// Initializes a new MAS strategy with custom options.
    /// </summary>
    /// <param name="lossFunction">The loss function to use.</param>
    /// <param name="options">Configuration options.</param>
    public MemoryAwareSynapses(ILossFunction<T> lossFunction, MASOptions<T>? options = null)
        : base(lossFunction)
    {
        var opts = options ?? new MASOptions<T>();

        _lambda = opts.Lambda ?? NumOps.FromDouble(1.0);
        _numSamples = opts.NumSamples ?? 256;
        _normalizeImportance = opts.NormalizeImportance ?? true;
        _importanceMode = opts.ImportanceMode ?? MASImportanceMode.OutputSensitivity;
        _accumulationMode = opts.AccumulationMode ?? ImportanceAccumulationMode.Sum;
        _decayFactor = NumOps.FromDouble(opts.DecayFactor ?? 0.9);
        _useBatching = opts.UseBatching ?? true;
        _batchSize = opts.BatchSize ?? 32;
        _minImportanceValue = NumOps.FromDouble(opts.MinImportanceValue ?? 1e-8);
        _useL1Norm = opts.UseL1Norm ?? false;

        _importanceInputs = new List<TInput>();
        _taskImportanceHistory = new List<Vector<T>>();
    }

    /// <inheritdoc/>
    public override string Name => "Memory-Aware-Synapses";

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
            bytes += EstimateVectorMemory(_optimalParameters);

            // Estimate input cache size (rough)
            bytes += _importanceInputs.Count * 1024; // Assume 1KB per input average

            // Task history
            foreach (var h in _taskImportanceHistory)
            {
                bytes += EstimateVectorMemory(h);
            }

            return bytes;
        }
    }

    /// <inheritdoc/>
    public override void PrepareForTask(IFullModel<T, TInput, TOutput> model, IDataset<T, TInput, TOutput> taskData)
    {
        int paramCount = (int)InterfaceGuard.Parameterizable(model).ParameterCount;

        // Initialize omega if first task
        if (_omega == null || _omega.Length != paramCount)
        {
            _omega = Engine.Fill(paramCount, _minImportanceValue);
        }

        // Cache inputs for importance computation (sample from dataset)
        _importanceInputs.Clear();
        int samplesToCache = Math.Min(_numSamples, taskData.Count);

        var indices = GetRandomIndices(taskData.Count, samplesToCache);
        foreach (var idx in indices)
        {
            var input = taskData.GetInput(idx);
            _importanceInputs.Add(input);
        }

        RecordMetric($"Task{TaskCount}_PrepareTime", DateTime.UtcNow);
        RecordMetric($"Task{TaskCount}_SamplesForImportance", samplesToCache);
    }

    /// <inheritdoc/>
    public override T ComputeRegularizationLoss(IFullModel<T, TInput, TOutput> model)
    {
        if (_omega == null || _optimalParameters == null || TaskCount == 0)
        {
            return NumOps.Zero;
        }

        var currentParams = InterfaceGuard.Parameterizable(model).GetParameters();

        if (currentParams.Length != _omega.Length)
        {
            throw new InvalidOperationException(
                $"Parameter dimension mismatch: current={currentParams.Length}, stored={_omega.Length}");
        }

        // MAS loss: (λ/2) × Σ Ω_i × (θ_i - θ*_i)²
        int n = currentParams.Length;
        var diff = Engine.TensorSubtract(currentParams.AsTensor(), Head(_optimalParameters, n));
        var weightedDiff = Engine.TensorMultiply(_omega.AsTensor(), Engine.TensorMultiply(diff, diff));
        T loss = Engine.TensorSum(weightedDiff);

        var halfLambda = NumOps.Divide(_lambda, NumOps.FromDouble(2.0));
        return NumOps.Multiply(halfLambda, loss);
    }

    /// <inheritdoc/>
    public override Vector<T> AdjustGradients(Vector<T> gradients)
    {
        // MAS doesn't adjust gradients directly - regularization is through loss
        if (_omega == null || _optimalParameters == null || TaskCount == 0)
        {
            return gradients;
        }

        // Add MAS gradient contribution: λ × Ω_i × (θ_i - θ*_i)
        // Note: We need current parameters, but we can't get them here
        // So this is done in ComputeRegularizationLoss and the trainer handles it
        return gradients;
    }

    /// <inheritdoc/>
    public override void FinalizeTask(IFullModel<T, TInput, TOutput> model)
    {
        if (_omega == null)
        {
            TaskCount++;
            return;
        }

        // Compute parameter importance using output sensitivity
        var taskImportance = ComputeImportance(model);

        // Normalize if requested
        if (_normalizeImportance)
        {
            taskImportance = NormalizeImportanceValues(taskImportance);
        }

        // Store for analysis
        _taskImportanceHistory.Add(CloneVector(taskImportance));

        // Accumulate into omega
        AccumulateImportance(taskImportance);

        // Store current parameters as optimal
        _optimalParameters = CloneVector(InterfaceGuard.Parameterizable(model).GetParameters());

        // Record metrics
        RecordMetric($"Task{TaskCount}_MeanImportance", ComputeMean(taskImportance));
        RecordMetric($"Task{TaskCount}_MaxImportance", ComputeMax(taskImportance));
        RecordMetric($"Task{TaskCount}_NonZeroImportance", CountNonZero(taskImportance));

        TaskCount++;
        _importanceInputs.Clear();
    }

    /// <summary>
    /// Computes parameter importance based on output sensitivity.
    /// </summary>
    private Vector<T> ComputeImportance(IFullModel<T, TInput, TOutput> model)
    {
        return _importanceMode switch
        {
            MASImportanceMode.OutputSensitivity => ComputeOutputSensitivity(model),
            MASImportanceMode.RandomProjection => ComputeRandomProjectionImportance(model),
            MASImportanceMode.FisherDiagonal => ComputeFisherDiagonalImportance(model),
            MASImportanceMode.Hebbian => ComputeHebbianImportance(model),
            _ => ComputeOutputSensitivity(model)
        };
    }

    /// <summary>
    /// Computes importance using output sensitivity (original MAS method).
    /// </summary>
    private Vector<T> ComputeOutputSensitivity(IFullModel<T, TInput, TOutput> model)
    {
        int paramCount = (int)InterfaceGuard.Parameterizable(model).ParameterCount;
        if (_importanceInputs.Count == 0)
        {
            // No inputs cached, return uniform importance
            return Engine.Fill(paramCount, NumOps.FromDouble(1.0));
        }

        var omega = Engine.Fill(paramCount, NumOps.Zero);

        int processed = 0;

        if (_useBatching)
        {
            // Process in batches
            for (int start = 0; start < _importanceInputs.Count; start += _batchSize)
            {
                int end = Math.Min(start + _batchSize, _importanceInputs.Count);
                ProcessImportanceBatch(model, _importanceInputs.Skip(start).Take(end - start).ToList(), omega);
                processed += (end - start);
            }
        }
        else
        {
            // Process one at a time
            foreach (var input in _importanceInputs)
            {
                ProcessSingleSampleImportance(model, input, omega);
                processed++;
            }
        }

        // Average over samples
        if (processed > 0)
        {
            var sampleCount = NumOps.FromDouble(processed);
            omega = (Vector<T>)Engine.Divide(omega, sampleCount);
        }

        return omega;
    }

    /// <summary>
    /// Processes a batch of samples for importance computation.
    /// </summary>
    private void ProcessImportanceBatch(IFullModel<T, TInput, TOutput> model, List<TInput> inputs, Vector<T> omega)
    {
        foreach (var input in inputs)
        {
            ProcessSingleSampleImportance(model, input, omega);
        }
    }

    /// <summary>
    /// Processes a single sample for importance computation.
    /// </summary>
    private void ProcessSingleSampleImportance(IFullModel<T, TInput, TOutput> model, TInput input, Vector<T> omega)
    {
        try
        {
            // Forward pass
            var output = model.Predict(input);

            // We need gradients of output norm with respect to parameters
            // This requires backpropagation through the network

            // For models that support gradient computation:
            if (model is IGradientCapable<T, TInput, TOutput> gradModel)
            {
                // Compute gradient of output L2 norm: d||y||²/dθ = 2y · (dy/dθ)
                var outputVector = ConvertToVector(output);
                var gradOutput = ComputeOutputNormGradient(outputVector);

                // Get parameter gradients
                var paramGrads = gradModel.ComputeParameterGradients(input, gradOutput);

                // Accumulate absolute gradients in-place (mutation required — omega is caller's vector)
                var absGrads = (Vector<T>)Engine.Abs(paramGrads);
                int count = Math.Min(omega.Length, absGrads.Length);
                if (count > 0)
                {
                    Engine.TensorAdd(Head(omega, count), Head(absGrads, count)).AsSpan().CopyTo(Head(omega, count).AsWritableSpan());
                }
            }
            else
            {
                // Fallback: use finite difference approximation
                ComputeFiniteDifferenceImportance(model, input, omega);
            }
        }
        catch (Exception ex)
        {
            System.Diagnostics.Trace.TraceWarning($"MAS importance: skipping sample due to error: {ex.Message}");
        }
    }

    /// <summary>
    /// Computes importance using finite difference when gradients aren't available.
    /// </summary>
    private void ComputeFiniteDifferenceImportance(IFullModel<T, TInput, TOutput> model, TInput input, Vector<T> omega)
    {
        var parameters = InterfaceGuard.Parameterizable(model).GetParameters();
        var epsilon = NumOps.FromDouble(1e-5);

        // Get baseline output norm
        var baseOutput = model.Predict(input);
        var baseNorm = ComputeOutputNorm(ConvertToVector(baseOutput));

        // Finite difference for each parameter (expensive!)
        // In practice, only do this for a subset of parameters
        int maxParams = Math.Min(parameters.Length, 1000);
        int step = Math.Max(1, parameters.Length / maxParams);

        var perturbedNorms = new List<T>();
        try
        {
            ProbePerturbedOutputNorms(model, input, parameters, epsilon, step, perturbedNorms);
        }
        finally
        {
            // Applied even when a prediction throws part-way, as the per-parameter loop had already done
            if (perturbedNorms.Count > 0)
            {
                // Finite difference gradient magnitude |(norm(θ + ε e_i) - norm(θ)) / ε|,
                // applied to this and neighboring parameters
                var norms = new Vector<T>(perturbedNorms).AsTensor();
                var absGrad = Engine.TensorAbs(
                    Engine.TensorDivideScalar(Engine.TensorSubtractScalar(norms, baseNorm), epsilon));
                AddToParameterBlocks(omega, absGrad, step, parameters.Length);
            }
        }
    }

    /// <summary>
    /// Perturbs every <paramref name="step"/>-th parameter by <paramref name="epsilon"/> in turn, recording the output
    /// norm of each perturbed model and restoring the parameter afterwards.
    /// </summary>
    /// <remarks>
    /// Norms are appended to <paramref name="norms"/> as they are measured, so a caller still sees the
    /// ones recorded before a prediction throws.
    /// </remarks>
    private void ProbePerturbedOutputNorms(
        IFullModel<T, TInput, TOutput> model, TInput input, Vector<T> parameters, T epsilon, int step, List<T> norms)
    {
        var perturbed = Engine.TensorAddScalar(parameters.AsTensor(), epsilon);
        var originals = parameters.Clone();
        var parameterizable = InterfaceGuard.Parameterizable(model);

        for (int i = 0; i < parameters.Length; i += step)
        {
            // Perturb parameter
            perturbed.AsSpan().Slice(i, 1).CopyTo(parameters.AsWritableSpan().Slice(i, 1));
            parameterizable.SetParameters(parameters);

            var perturbedOutput = model.Predict(input);
            norms.Add(ComputeOutputNorm(ConvertToVector(perturbedOutput)));

            // Restore parameter
            originals.AsSpan().Slice(i, 1).CopyTo(parameters.AsWritableSpan().Slice(i, 1));
            parameterizable.SetParameters(parameters);
        }
    }

    /// <summary>
    /// Adds value k to omega[k * step .. min((k + 1) * step, length)) for every block k.
    /// </summary>
    private static void AddToParameterBlocks(Vector<T> omega, Tensor<T> blockValues, int step, int length)
    {
        int blocks = blockValues.Length;
        var expanded = Engine.TensorTile(blockValues.Reshape(blocks, 1), new[] { 1, step }).Reshape(blocks * step);
        int count = Math.Min(length, blocks * step);
        var increments = count == expanded.Length
            ? expanded
            : Engine.TensorSlice(expanded, new[] { 0 }, new[] { count });
        Engine.TensorAdd(Head(omega, count), increments).AsSpan().CopyTo(Head(omega, count).AsWritableSpan());
    }

    /// <summary>
    /// Computes importance using random projection method.
    /// Projects output onto random directions and measures sensitivity.
    /// </summary>
    /// <remarks>
    /// <para>This method is computationally cheaper than full output norm gradient.
    /// It uses random projections to approximate the sensitivity of each parameter
    /// to changes in the output space.</para>
    /// </remarks>
    private Vector<T> ComputeRandomProjectionImportance(IFullModel<T, TInput, TOutput> model)
    {
        int paramCount = (int)InterfaceGuard.Parameterizable(model).ParameterCount;
        if (_importanceInputs.Count == 0)
        {
            return Engine.Fill(paramCount, NumOps.FromDouble(1.0));
        }

        var omega = Engine.Fill(paramCount, NumOps.Zero);

        // Number of random projection directions
        const int numProjections = 10;

        foreach (var input in _importanceInputs)
        {
            try
            {
                var output = model.Predict(input);
                var outputVec = ConvertToVector(output);

                if (model is IGradientCapable<T, TInput, TOutput> gradModel)
                {
                    // Project output onto multiple random directions
                    for (int p = 0; p < numProjections; p++)
                    {
                        // Generate random unit direction vector
                        var randomDir = new Vector<T>(outputVec.Length);
                        double normSquared = 0;
                        for (int j = 0; j < outputVec.Length; j++)
                        {
                            double randVal = RandomHelper.ThreadSafeRandom.NextDouble() * 2.0 - 1.0;
                            randomDir[j] = NumOps.FromDouble(randVal);
                            normSquared += randVal * randVal;
                        }

                        // Normalize to unit vector
                        double norm = Math.Sqrt(normSquared);
                        if (norm > 1e-8)
                        {
                            randomDir = Engine.Divide(randomDir, NumOps.FromDouble(norm));
                        }

                        // Compute gradient with respect to random projection
                        var paramGrads = gradModel.ComputeParameterGradients(input, randomDir);

                        // Accumulate absolute gradients (vectorized)
                        var absGrads = (Vector<T>)Engine.Abs(paramGrads);
                        omega = (Vector<T>)Engine.Add(omega, absGrads);
                    }
                }
                else
                {
                    // Fall back to output sensitivity for non-gradient-capable models
                    ProcessSingleSampleImportance(model, input, omega);
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Trace.TraceWarning($"MAS importance: skipping sample due to error: {ex.Message}");
            }
        }

        // Average over samples and projections
        int totalCount = _importanceInputs.Count * numProjections;
        if (totalCount > 0)
        {
            var divisor = NumOps.FromDouble(totalCount);
            omega = (Vector<T>)Engine.Divide(omega, divisor);
        }

        return omega;
    }

    /// <summary>
    /// Computes importance using Fisher diagonal (hybrid with EWC).
    /// Uses squared loss gradients as diagonal approximation of Fisher Information Matrix.
    /// </summary>
    /// <remarks>
    /// <para>This method is similar to EWC's Fisher computation but uses the loss function
    /// gradients. The Fisher diagonal F_ii ≈ E[(∂L/∂θ_i)²] measures how sensitive
    /// the loss is to each parameter.</para>
    /// </remarks>
    private Vector<T> ComputeFisherDiagonalImportance(IFullModel<T, TInput, TOutput> model)
    {
        int paramCount = (int)InterfaceGuard.Parameterizable(model).ParameterCount;
        if (_importanceInputs.Count == 0)
        {
            return Engine.Fill(paramCount, NumOps.FromDouble(1.0));
        }

        var omega = Engine.Fill(paramCount, NumOps.Zero);

        // We need loss gradients, which requires labels
        // Since MAS is unsupervised, we approximate using output-based loss
        // Use cross-entropy-like loss: -log(softmax(output)) or MSE reconstruction

        foreach (var input in _importanceInputs)
        {
            try
            {
                var output = model.Predict(input);
                var outputVec = ConvertToVector(output);

                if (model is IGradientCapable<T, TInput, TOutput> gradModel)
                {
                    // For unsupervised importance, use self-supervised signal
                    // We create a pseudo-target from the model's own confident predictions
                    // Gradient of squared error: 2 * (output - target) = 2 * (output - output) when target=output
                    // But we want to measure sensitivity, so we use output directly

                    // Fisher diagonal approximation: E[g²] where g = ∂L/∂θ
                    // Using L = ||output||² gives us consistent behavior with MAS output sensitivity
                    // but squared for Fisher-like weighting

                    var gradOutput = ComputeOutputNormGradient(outputVec);
                    var paramGrads = gradModel.ComputeParameterGradients(input, gradOutput);

                    // Square the gradients for Fisher diagonal approximation (vectorized)
                    var gradSquared = (Vector<T>)Engine.Multiply(paramGrads, paramGrads);
                    omega = (Vector<T>)Engine.Add(omega, gradSquared);
                }
                else
                {
                    // For non-gradient models, use finite difference with squared sensitivity
                    ComputeFiniteDifferenceFisher(model, input, omega);
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Trace.TraceWarning($"MAS importance: skipping sample due to error: {ex.Message}");
            }
        }

        // Average over samples
        if (_importanceInputs.Count > 0)
        {
            var divisor = NumOps.FromDouble(_importanceInputs.Count);
            omega = (Vector<T>)Engine.Divide(omega, divisor);
        }

        return omega;
    }

    /// <summary>
    /// Computes Fisher diagonal using finite differences for non-gradient models.
    /// </summary>
    private void ComputeFiniteDifferenceFisher(IFullModel<T, TInput, TOutput> model, TInput input, Vector<T> omega)
    {
        var parameters = InterfaceGuard.Parameterizable(model).GetParameters();
        var epsilon = NumOps.FromDouble(1e-5);

        var baseOutput = model.Predict(input);
        var baseNorm = ComputeOutputNorm(ConvertToVector(baseOutput));

        int maxParams = Math.Min(parameters.Length, 1000);
        int step = Math.Max(1, parameters.Length / maxParams);

        var perturbedNorms = new List<T>();
        try
        {
            ProbePerturbedOutputNorms(model, input, parameters, epsilon, step, perturbedNorms);
        }
        finally
        {
            // Applied even when a prediction throws part-way, as the per-parameter loop had already done
            if (perturbedNorms.Count > 0)
            {
                // Finite difference gradient magnitude, squared for Fisher
                var norms = new Vector<T>(perturbedNorms).AsTensor();
                var grad = Engine.TensorDivideScalar(Engine.TensorSubtractScalar(norms, baseNorm), epsilon);
                AddToParameterBlocks(omega, Engine.TensorMultiply(grad, grad), step, parameters.Length);
            }
        }
    }

    /// <summary>
    /// Computes importance using Hebbian-style activation magnitudes.
    /// Parameters connected to highly active neurons are considered more important.
    /// </summary>
    /// <remarks>
    /// <para>Hebbian learning principle: "neurons that fire together, wire together".
    /// Parameters that are involved in producing high activations are likely important
    /// for the learned representations.</para>
    /// <para>Importance is estimated as: Ω_i ∝ |activation_pre| × |activation_post|</para>
    /// </remarks>
    private Vector<T> ComputeHebbianImportance(IFullModel<T, TInput, TOutput> model)
    {
        int paramCount = (int)InterfaceGuard.Parameterizable(model).ParameterCount;
        if (_importanceInputs.Count == 0)
        {
            return Engine.Fill(paramCount, NumOps.FromDouble(1.0));
        }

        var omega = Engine.Fill(paramCount, NumOps.Zero);

        // For Hebbian importance, we measure how active each part of the network is
        // Since we don't have direct access to layer activations, we use a proxy:
        // - High output magnitude suggests active neurons contributed
        // - We weight parameters by how much they contribute to high-magnitude outputs

        // Collect output statistics across samples
        var outputMagnitudes = new List<T>();
        var paramContributions = Engine.Fill(paramCount, NumOps.Zero);

        foreach (var input in _importanceInputs)
        {
            try
            {
                var output = model.Predict(input);
                var outputVec = ConvertToVector(output);

                // Compute output magnitude as activation proxy
                T outputMag = Engine.Norm(outputVec);
                outputMagnitudes.Add(outputMag);

                // Get parameter values and weight by output magnitude
                // Hebbian: importance ~ input_activation * output_activation
                // We approximate this as: parameter_magnitude * output_magnitude
                var parameters = InterfaceGuard.Parameterizable(model).GetParameters();

                // Weight parameter importance by output magnitude
                int count = Math.Min(paramCount, parameters.Length);
                if (count > 0)
                {
                    var weightedContrib = Engine.TensorMultiplyScalar(Engine.TensorAbs(Head(parameters, count)), outputMag);
                    Engine.TensorAdd(Head(paramContributions, count), weightedContrib).AsSpan().CopyTo(Head(paramContributions, count).AsWritableSpan());
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Trace.TraceWarning($"MAS importance: skipping sample due to error: {ex.Message}");
            }
        }

        // Compute final importance with Hebbian weighting
        if (_importanceInputs.Count > 0 && outputMagnitudes.Count > 0)
        {
            // Normalize by sample count
            var divisor = NumOps.FromDouble(_importanceInputs.Count);
            omega = Engine.Divide(paramContributions, divisor);

            // Apply exponential weighting to emphasize high-activation parameters
            // This follows the Hebbian principle more closely
            T meanMag = Engine.Mean(new Vector<T>(outputMagnitudes));
            if (NumOps.GreaterThan(meanMag, NumOps.FromDouble(1e-8)))
            {
                var scaleFactor = NumOps.Divide(NumOps.One, meanMag);
                omega = Engine.Multiply(omega, scaleFactor);
            }
        }
        else
        {
            // Fallback to uniform importance
            omega = Engine.Fill(paramCount, NumOps.FromDouble(1.0));
        }

        return omega;
    }

    /// <summary>
    /// Computes the gradient of output L2 norm.
    /// </summary>
    private Vector<T> ComputeOutputNormGradient(Vector<T> output)
    {
        // d||y||²/dy = 2y
        return Engine.Multiply(output, NumOps.FromDouble(2.0));
    }

    /// <summary>
    /// Computes the norm of an output vector.
    /// </summary>
    private T ComputeOutputNorm(Vector<T> output)
    {
        T sum = NumOps.Zero;

        if (_useL1Norm)
        {
            sum = Engine.Sum(Engine.Abs(output));
        }
        else
        {
            sum = NumOps.Add(sum, Engine.DotProduct(output, output));
            sum = NumOps.FromDouble(Math.Sqrt(Convert.ToDouble(sum)));
        }

        return sum;
    }

    /// <summary>
    /// Converts output to a vector for gradient computation.
    /// </summary>
    private Vector<T> ConvertToVector(TOutput output)
    {
        if (output is Vector<T> vec)
            return vec;

        if (output is T scalar)
            return new Vector<T>(new[] { scalar });

        if (output is T[] arr)
            return new Vector<T>(arr);

        if (output is IEnumerable<T> enumerable)
            return new Vector<T>(enumerable.ToArray());

        throw new InvalidOperationException(
            $"Cannot convert output of type {typeof(TOutput).Name} to Vector<{typeof(T).Name}>. " +
            "Supported types: Vector<T>, T, T[], IEnumerable<T>.");
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
                _omega = Engine.Add(_omega, taskImportance);
                break;

            case ImportanceAccumulationMode.Max:
            {
                var omega = _omega.AsTensor();
                var incoming = Head(taskImportance, _omega.Length);
                var merged = Engine.TensorWhere(Engine.TensorGreaterThan(incoming, omega), incoming, omega);
                merged.AsSpan().CopyTo(_omega.AsWritableSpan());
                break;
            }

            case ImportanceAccumulationMode.WeightedSum:
            {
                var decayed = Engine.TensorMultiplyScalar(_omega.AsTensor(), _decayFactor);
                var merged = Engine.TensorAdd(decayed, Head(taskImportance, _omega.Length));
                merged.AsSpan().CopyTo(_omega.AsWritableSpan());
                break;
            }
        }
    }

    /// <summary>
    /// Normalizes importance values.
    /// </summary>
    private Vector<T> NormalizeImportanceValues(Vector<T> importance)
    {
        T maxVal = _minImportanceValue;
        if (importance.Length > 0)
        {
            T largest = Engine.TensorMaxValue(importance.AsTensor());
            if (NumOps.GreaterThan(largest, maxVal))
            {
                maxVal = largest;
            }
        }

        double maxDouble = Convert.ToDouble(maxVal);
        if (maxDouble < 1e-10)
            return importance;

        var normalized = (Vector<T>)Engine.Divide(importance, maxVal);
        return normalized;
    }

    /// <summary>
    /// Gets random indices for sampling.
    /// </summary>
    private int[] GetRandomIndices(int total, int count)
    {
        if (count >= total)
        {
            return Enumerable.Range(0, total).ToArray();
        }

        var indices = new HashSet<int>();
        while (indices.Count < count)
        {
            indices.Add(RandomHelper.GetSecureRandomInt(0, total));
        }
        return indices.ToArray();
    }

    /// <inheritdoc/>
    public override void Reset()
    {
        base.Reset();
        _omega = null;
        _optimalParameters = null;
        _importanceInputs.Clear();
        _taskImportanceHistory.Clear();
    }

    /// <inheritdoc/>
    protected override Dictionary<string, object> GetStateForSerialization()
    {
        var state = base.GetStateForSerialization();
        state["Lambda"] = Convert.ToDouble(_lambda);
        state["NumSamples"] = _numSamples;
        state["ImportanceMode"] = _importanceMode.ToString();
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
    public Vector<T>? OptimalParameters => _optimalParameters;

    /// <summary>
    /// Gets the regularization strength.
    /// </summary>
    public T Lambda => _lambda;

    /// <summary>
    /// Gets the importance history for all tasks.
    /// </summary>
    public IReadOnlyList<Vector<T>> TaskImportanceHistory => _taskImportanceHistory.AsReadOnly();

    /// <summary>
    /// Computes the mean of a vector.
    /// </summary>
    private double ComputeMean(Vector<T> vector)
    {
        // Reported as a double metric, like master
        return Convert.ToDouble(Engine.Mean(vector));
    }

    /// <summary>
    /// Computes the maximum value in a vector.
    /// </summary>
    private double ComputeMax(Vector<T> vector)
    {
        // Reported as a double metric, like master
        return Convert.ToDouble(Engine.TensorMaxValue(vector.AsTensor()));
    }

    /// <summary>
    /// Counts non-zero elements in a vector.
    /// </summary>
    private int CountNonZero(Vector<T> vector)
    {
        if (vector.Length == 0)
            return 0;

        return Engine.TensorCountNonzero(Engine.TensorGreaterThan(vector.AsTensor(), _minImportanceValue));
    }
}

/// <summary>
/// Interface for models that support gradient computation.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <typeparam name="TInput">The input type.</typeparam>
/// <typeparam name="TOutput">The output type.</typeparam>
[AiDotNet.Configuration.YamlConfigurable("GradientCapable")]
public interface IGradientCapable<T, TInput, TOutput>
{
    /// <summary>
    /// Computes gradients of parameters with respect to an output gradient.
    /// </summary>
    /// <param name="input">The input that produced the output.</param>
    /// <param name="outputGradient">Gradient with respect to output.</param>
    /// <returns>Gradient with respect to parameters.</returns>
    Vector<T> ComputeParameterGradients(TInput input, Vector<T> outputGradient);
}
