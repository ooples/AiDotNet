using AiDotNet.Tensors.Engines.DirectGpu;
using Newtonsoft.Json;

namespace AiDotNet.Optimizers;

/// <summary>
/// Implements the LARS (Layer-wise Adaptive Rate Scaling) optimization algorithm.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// LARS is specifically designed for training with very large batch sizes (4096-32768).
/// It automatically adapts the learning rate for each layer based on the ratio of
/// parameter norm to gradient norm, which helps maintain stable training at scale.
/// </para>
/// <para><b>Key Formula:</b></para>
/// <code>
/// local_lr = trust_coeff * ||w|| / (||g|| + weight_decay * ||w|| + epsilon)
/// update = local_lr * (g + weight_decay * w)
/// w = w - lr * update (with momentum)
/// </code>
/// <para><b>For Beginners:</b> When training with very large batches (common in self-supervised
/// learning like SimCLR), regular optimizers can become unstable because gradients get averaged
/// over more samples, making them smaller. LARS solves this by looking at each layer and asking
/// "how big are the weights compared to the gradients?" and scaling the learning rate accordingly.
/// This allows stable training with batch sizes of 4096 or even larger.</para>
/// <para>
/// Based on the paper "Large Batch Training of Convolutional Networks" by You et al. (2017).
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // For SimCLR training with large batches
/// var options = new LARSOptimizerOptions&lt;float, Matrix&lt;float&gt;, Vector&lt;float&gt;&gt;
/// {
///     InitialLearningRate = 0.3 * batchSize / 256.0,  // Linear scaling rule
///     Momentum = 0.9,
///     WeightDecay = 1e-4,
///     TrustCoefficient = 0.001,
///     WarmupEpochs = 10
/// };
/// var optimizer = new LARSOptimizer&lt;float, Matrix&lt;float&gt;, Vector&lt;float&gt;&gt;(model, options);
/// </code>
/// </example>
public class LARSOptimizer<T, TInput, TOutput> : GradientBasedOptimizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// The options specific to the LARS optimizer.
    /// </summary>
    private LARSOptimizerOptions<T, TInput, TOutput> _options;

    /// <summary>
    /// The velocity/momentum buffer for each parameter.
    /// </summary>
    private Vector<T> _velocity;

    /// <summary>
    /// The current time step (iteration count).
    /// </summary>
    private int _t;

    /// <summary>
    /// The total number of warmup steps.
    /// </summary>
    private int _warmupSteps;

    /// <summary>
    /// Previous velocity for reverse updates.
    /// </summary>
    private Vector<T>? _previousVelocity;

    /// <summary>
    /// Previous step count for reverse updates.
    /// </summary>
    private int _previousT;

    /// <summary>
    /// Initializes a new instance of the LARSOptimizer class.
    /// </summary>
    /// <param name="model">The model to optimize.</param>
    /// <param name="options">The options for configuring the LARS optimizer.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This sets up the LARS optimizer with its initial configuration.
    /// The most important parameters for SSL are:
    /// - Learning rate: Use linear scaling (base_lr * batch_size / 256)
    /// - Trust coefficient: Controls layer-wise scaling (0.001 is typical)
    /// - Warmup epochs: Gradually ramp up learning rate (10 epochs typical)
    /// </para>
    /// </remarks>
    public LARSOptimizer(
        IFullModel<T, TInput, TOutput>? model,
        LARSOptimizerOptions<T, TInput, TOutput>? options = null)
        : base(model, options ?? new())
    {
        _velocity = Vector<T>.Empty();
        _t = 0;
        _warmupSteps = 0;
        _options = options ?? new();

        InitializeAdaptiveParameters();
    }

    /// <summary>
    /// Initializes the adaptive parameters used by the LARS optimizer.
    /// </summary>
    protected override void InitializeAdaptiveParameters()
    {
        // Base learning rate is handled by base class
    }

    /// <summary>
    /// Gets the current weight decay coefficient.
    /// </summary>
    public double WeightDecay => _options.WeightDecay;

    /// <summary>
    /// Gets the current momentum coefficient.
    /// </summary>
    public double Momentum => _options.Momentum;

    /// <summary>
    /// Gets the LARS trust coefficient.
    /// </summary>
    public double TrustCoefficient => _options.TrustCoefficient;

    /// <summary>
    /// Performs the optimization process using the LARS algorithm.
    /// </summary>
    /// <param name="inputData">The input data for optimization.</param>
    /// <returns>The result of the optimization process.</returns>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        // Initialize with random solution
        var currentSolution = InitializeRandomSolution(inputData.XTrain);
        var bestStepData = new OptimizationStepData<T, TInput, TOutput>();
        var parameters = currentSolution.GetParameters();
        _velocity = new Vector<T>(parameters.Length);
        _t = 0;

        // Calculate warmup steps based on data size and batch size
        int stepsPerEpoch = (int)Math.Ceiling((double)GetDataSize(inputData) / _options.BatchSize);
        _warmupSteps = _options.WarmupEpochs * stepsPerEpoch;

        // Initialize parameters
        InitializeAdaptiveParameters();

        var previousStepData = PrepareAndEvaluateSolution(currentSolution, inputData);

        for (int epoch = 0; epoch < _options.MaxIterations; epoch++)
        {
            // Notify sampler of new epoch
            NotifyEpochStart(epoch);

            // Create batcher for the current epoch
            var batcher = CreateBatcher(inputData, _options.BatchSize);

            foreach (var (xBatch, yBatch, batchIndices) in batcher.GetBatches())
            {
                _t++;

                // Calculate gradient on the batch
                var gradient = CalculateGradient(currentSolution, xBatch, yBatch);

                // Apply warmup learning rate
                var effectiveLr = GetWarmupLearningRate();

                // Update solution using LARS algorithm
                var newSolution = UpdateSolutionWithLARS(currentSolution, gradient, effectiveLr);

                currentSolution = newSolution;
            }

            // Evaluate after processing all batches in the epoch
            var currentStepData = EvaluateSolution(currentSolution, inputData);
            UpdateBestSolution(currentStepData, ref bestStepData);
            UpdateAdaptiveParameters(currentStepData, previousStepData);

            // Check early stopping criteria
            if (UpdateIterationHistoryAndCheckEarlyStopping(epoch, bestStepData))
            {
                return CreateOptimizationResult(bestStepData, inputData);
            }

            // Check convergence
            if (NumOps.LessThan(
                NumOps.Abs(NumOps.Subtract(bestStepData.FitnessScore, currentStepData.FitnessScore)),
                NumOps.FromDouble(_options.Tolerance)))
            {
                return CreateOptimizationResult(bestStepData, inputData);
            }

            previousStepData = currentStepData;
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    /// <summary>
    /// Gets the learning rate with warmup applied.
    /// </summary>
    private double GetWarmupLearningRate()
    {
        if (_t < _warmupSteps && _warmupSteps > 0)
        {
            // Linear warmup
            return _options.InitialLearningRate * (double)_t / _warmupSteps;
        }
        return _options.InitialLearningRate;
    }

    /// <summary>
    /// Updates the solution using the LARS algorithm with layer-wise adaptive rate scaling.
    /// </summary>
    private IFullModel<T, TInput, TOutput> UpdateSolutionWithLARS(
        IFullModel<T, TInput, TOutput> currentSolution,
        Vector<T> gradient,
        double effectiveLr)
    {
        var parameters = currentSolution.GetParameters();

        // Get layer boundaries or treat as single layer
        var layerBoundaries = _options.LayerBoundaries ?? [parameters.Length];
        var skipLayers = new HashSet<int>(_options.SkipLARSLayers ?? []);

        int layerStart = 0;
        var layerIndex = 0;

        // Pre-compute constants
        T momentum = NumOps.FromDouble(_options.Momentum);
        T weightDecay = NumOps.FromDouble(_options.WeightDecay);
        T trustCoeff = NumOps.FromDouble(_options.TrustCoefficient);
        T epsilon = NumOps.FromDouble(_options.Epsilon);
        T baseLr = NumOps.FromDouble(effectiveLr);
        T one = NumOps.One;
        T zero = NumOps.Zero;

        // Process each layer
        foreach (var layerEnd in layerBoundaries)
        {
            int layerSize = layerEnd - layerStart;
            if (layerSize <= 0)
            {
                layerIndex++;
                continue;
            }

            // Extract layer parameters and gradients
            var layerParams = ExtractLayerVector(parameters, layerStart, layerSize);
            var layerGrad = ExtractLayerVector(gradient, layerStart, layerSize);

            // Compute layer-wise LARS scaling
            T localLr;
            if (skipLayers.Contains(layerIndex))
            {
                // Skip LARS for this layer, use base learning rate only
                localLr = baseLr;
            }
            else
            {
                // Compute parameter norm ||w||
                T paramNorm = ComputeL2Norm(layerParams);

                // Compute gradient norm ||g||
                T gradNorm = ComputeL2Norm(layerGrad);

                // LARS scaling: trust_coeff * ||w|| / (||g|| + weight_decay * ||w|| + epsilon)
                T denominator = NumOps.Add(gradNorm,
                    NumOps.Add(NumOps.Multiply(weightDecay, paramNorm), epsilon));

                T larsRatio = NumOps.Divide(NumOps.Multiply(trustCoeff, paramNorm), denominator);

                // Only apply LARS if both norms are non-zero
                bool paramNormZero = NumOps.LessThan(paramNorm, epsilon);
                bool gradNormZero = NumOps.LessThan(gradNorm, epsilon);

                if (paramNormZero || gradNormZero)
                {
                    localLr = baseLr;
                }
                else
                {
                    localLr = NumOps.Multiply(baseLr, larsRatio);
                }
            }

            // Compute update with weight decay: update = g + weight_decay * w
            var weightDecayTerm = (Vector<T>)Engine.Multiply(layerParams, weightDecay);
            var updateWithDecay = (Vector<T>)Engine.Add(layerGrad, weightDecayTerm);

            // Scale by local learning rate
            var scaledUpdate = (Vector<T>)Engine.Multiply(updateWithDecay, localLr);

            // Apply momentum: v = momentum * v + scaled_update
            var layerVelocity = ExtractLayerVector(_velocity, layerStart, layerSize);
            var velocityScaled = (Vector<T>)Engine.Multiply(layerVelocity, momentum);

            Vector<T> newVelocity;
            if (_options.UseNesterov)
            {
                // Nesterov momentum: v = momentum * v + scaled_update
                // update = momentum * v + scaled_update
                newVelocity = (Vector<T>)Engine.Add(velocityScaled, scaledUpdate);
                var nesterovUpdate = (Vector<T>)Engine.Add(
                    (Vector<T>)Engine.Multiply(newVelocity, momentum),
                    scaledUpdate);
                scaledUpdate = nesterovUpdate;
            }
            else
            {
                // Standard momentum
                newVelocity = (Vector<T>)Engine.Add(velocityScaled, scaledUpdate);
                scaledUpdate = newVelocity;
            }

            // Update velocity buffer
            UpdateLayerVector(_velocity, newVelocity, layerStart, layerSize);

            // Update parameters: w = w - v
            var newLayerParams = (Vector<T>)Engine.Subtract(layerParams, scaledUpdate);

            // Write back to parameters
            UpdateLayerVector(parameters, newLayerParams, layerStart, layerSize);

            layerStart = layerEnd;
            layerIndex++;
        }

        return currentSolution.WithParameters(parameters);
    }

    /// <summary>
    /// Updates the current solution using the LARS update rule.
    /// </summary>
    protected override IFullModel<T, TInput, TOutput> UpdateSolution(
        IFullModel<T, TInput, TOutput> currentSolution,
        Vector<T> gradient)
    {
        return UpdateSolutionWithLARS(currentSolution, gradient, GetWarmupLearningRate());
    }

    /// <summary>
    /// Updates a vector of parameters using the LARS optimization algorithm.
    /// </summary>
    public override Vector<T> UpdateParameters(Vector<T> parameters, Vector<T> gradient)
    {
        if (_velocity == null || _velocity.Length != parameters.Length)
        {
            _velocity = new Vector<T>(parameters.Length);
            _previousVelocity = new Vector<T>(parameters.Length);
            _t = 0;
        }

        // Save pre-update state
        if (_previousVelocity == null || _previousVelocity.Length != parameters.Length)
        {
            _previousVelocity = new Vector<T>(parameters.Length);
        }
        _previousVelocity = new Vector<T>(_velocity);
        _previousT = _t;

        _t++;

        var effectiveLr = GetWarmupLearningRate();
        return UpdateParametersLARS(parameters, gradient, effectiveLr);
    }

    /// <summary>
    /// Internal LARS parameter update.
    /// </summary>
    private Vector<T> UpdateParametersLARS(Vector<T> parameters, Vector<T> gradient, double effectiveLr)
    {
        // Treat as single layer if no boundaries specified
        T momentum = NumOps.FromDouble(_options.Momentum);
        T weightDecay = NumOps.FromDouble(_options.WeightDecay);
        T trustCoeff = NumOps.FromDouble(_options.TrustCoefficient);
        T epsilon = NumOps.FromDouble(_options.Epsilon);
        T baseLr = NumOps.FromDouble(effectiveLr);

        // Compute norms
        T paramNorm = ComputeL2Norm(parameters);
        T gradNorm = ComputeL2Norm(gradient);

        // LARS scaling
        T denominator = NumOps.Add(gradNorm,
            NumOps.Add(NumOps.Multiply(weightDecay, paramNorm), epsilon));
        T larsRatio = NumOps.Divide(NumOps.Multiply(trustCoeff, paramNorm), denominator);

        // Apply LARS only if norms are non-trivial
        T localLr;
        bool paramNormZero = NumOps.LessThan(paramNorm, epsilon);
        bool gradNormZero = NumOps.LessThan(gradNorm, epsilon);

        if (paramNormZero || gradNormZero)
        {
            localLr = baseLr;
        }
        else
        {
            localLr = NumOps.Multiply(baseLr, larsRatio);
        }

        // Update with weight decay
        var weightDecayTerm = (Vector<T>)Engine.Multiply(parameters, weightDecay);
        var updateWithDecay = (Vector<T>)Engine.Add(gradient, weightDecayTerm);
        var scaledUpdate = (Vector<T>)Engine.Multiply(updateWithDecay, localLr);

        // Momentum update
        var velocityScaled = (Vector<T>)Engine.Multiply(_velocity, momentum);
        _velocity = (Vector<T>)Engine.Add(velocityScaled, scaledUpdate);

        // Apply update
        return (Vector<T>)Engine.Subtract(parameters, _velocity);
    }

    /// <summary>
    /// Updates a matrix of parameters using the LARS optimization algorithm.
    /// </summary>
    public override Matrix<T> UpdateParameters(Matrix<T> parameters, Matrix<T> gradient)
    {
        int totalSize = parameters.Rows * parameters.Columns;

        if (_velocity == null || _velocity.Length != totalSize)
        {
            _velocity = new Vector<T>(totalSize);
            _t = 0;
        }

        _t++;

        // Flatten matrix to vector
        var paramVec = new Vector<T>(totalSize);
        var gradVec = new Vector<T>(totalSize);
        int idx = 0;
        for (int i = 0; i < parameters.Rows; i++)
        {
            for (int j = 0; j < parameters.Columns; j++)
            {
                paramVec[idx] = parameters[i, j];
                gradVec[idx] = gradient[i, j];
                idx++;
            }
        }

        // Apply LARS update
        var updatedVec = UpdateParametersLARS(paramVec, gradVec, GetWarmupLearningRate());

        // Unflatten vector back to matrix
        var updatedMatrix = new Matrix<T>(parameters.Rows, parameters.Columns);
        idx = 0;
        for (int i = 0; i < parameters.Rows; i++)
        {
            for (int j = 0; j < parameters.Columns; j++)
            {
                updatedMatrix[i, j] = updatedVec[idx];
                idx++;
            }
        }

        return updatedMatrix;
    }

    /// <summary>
    /// Computes the L2 norm of a vector.
    /// </summary>
    private T ComputeL2Norm(Vector<T> vector)
    {
        T sumSquared = NumOps.Zero;
        for (int i = 0; i < vector.Length; i++)
        {
            sumSquared = NumOps.Add(sumSquared, NumOps.Multiply(vector[i], vector[i]));
        }
        return NumOps.Sqrt(sumSquared);
    }

    /// <summary>
    /// Extracts a layer's parameters from the full parameter vector.
    /// </summary>
    private Vector<T> ExtractLayerVector(Vector<T> fullVector, int start, int length)
    {
        var layer = new Vector<T>(length);
        for (int i = 0; i < length; i++)
        {
            layer[i] = fullVector[start + i];
        }
        return layer;
    }

    /// <summary>
    /// Updates a layer's values in the full parameter vector.
    /// </summary>
    private void UpdateLayerVector(Vector<T> fullVector, Vector<T> layerVector, int start, int length)
    {
        for (int i = 0; i < length; i++)
        {
            fullVector[start + i] = layerVector[i];
        }
    }

    /// <summary>
    /// Gets the size of the training data.
    /// </summary>
    private int GetDataSize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        // Try to get the batch dimension from XTrain
        if (inputData.XTrain is Matrix<T> matrix)
        {
            return matrix.Rows;
        }
        if (inputData.XTrain is Tensor<T> tensor && tensor.Shape.Length > 0)
        {
            return tensor.Shape[0];
        }
        // Fallback to a reasonable default
        return 1000;
    }

    /// <summary>
    /// Reverses a LARS gradient update to recover original parameters.
    /// </summary>
    public override Vector<T> ReverseUpdate(Vector<T> updatedParameters, Vector<T> appliedGradients)
    {
        if (updatedParameters == null)
            throw new ArgumentNullException(nameof(updatedParameters));
        if (appliedGradients == null)
            throw new ArgumentNullException(nameof(appliedGradients));

        if (_previousVelocity == null || _previousT == 0)
        {
            return base.ReverseUpdate(updatedParameters, appliedGradients);
        }

        // LARS reverse is approximate due to layer-wise scaling complexity
        // Use the stored velocity for reversal
        return (Vector<T>)Engine.Add(updatedParameters, _previousVelocity);
    }

    /// <summary>
    /// Resets the optimizer's internal state.
    /// </summary>
    public override void Reset()
    {
        base.Reset();
        _velocity = Vector<T>.Empty();
        _previousVelocity = null;
        _t = 0;
    }

    /// <summary>
    /// Updates the optimizer's options.
    /// </summary>
    protected override void UpdateOptions(OptimizationAlgorithmOptions<T, TInput, TOutput> options)
    {
        if (options is LARSOptimizerOptions<T, TInput, TOutput> larsOptions)
        {
            _options = larsOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected LARSOptimizerOptions.");
        }
    }

    /// <summary>
    /// Gets the current optimizer options.
    /// </summary>
    public override OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions()
    {
        return _options;
    }

    /// <summary>
    /// Serializes the optimizer's state into a byte array.
    /// </summary>
    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        byte[] baseData = base.Serialize();
        writer.Write(baseData.Length);
        writer.Write(baseData);

        string optionsJson = JsonConvert.SerializeObject(_options);
        writer.Write(optionsJson);

        writer.Write(_t);
        writer.Write(_warmupSteps);
        writer.Write(_velocity.Length);
        foreach (var value in _velocity)
        {
            writer.Write(Convert.ToDouble(value));
        }

        return ms.ToArray();
    }

    /// <summary>
    /// Deserializes the optimizer's state from a byte array.
    /// </summary>
    public override void Deserialize(byte[] data)
    {
        using var ms = new MemoryStream(data);
        using var reader = new BinaryReader(ms);

        int baseDataLength = reader.ReadInt32();
        byte[] baseData = reader.ReadBytes(baseDataLength);

        // Read base class data manually
        using (var baseMs = new MemoryStream(baseData))
        using (var baseReader = new BinaryReader(baseMs))
        {
            string typeName = baseReader.ReadString();
            if (typeName != this.GetType().AssemblyQualifiedName)
            {
                throw new InvalidOperationException("Mismatched optimizer type during deserialization.");
            }
            baseReader.ReadString(); // Skip base options
        }

        string optionsJson = reader.ReadString();
        _options = JsonConvert.DeserializeObject<LARSOptimizerOptions<T, TInput, TOutput>>(optionsJson)
            ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");

        _t = reader.ReadInt32();
        _warmupSteps = reader.ReadInt32();
        int vLength = reader.ReadInt32();
        _velocity = new Vector<T>(vLength);
        for (int i = 0; i < vLength; i++)
        {
            _velocity[i] = NumOps.FromDouble(reader.ReadDouble());
        }

        InitializeAdaptiveParameters();
    }

    /// <summary>
    /// Generates a unique key for caching gradients.
    /// </summary>
    protected override string GenerateGradientCacheKey(IFullModel<T, TInput, TOutput> model, TInput X, TOutput y)
    {
        var baseKey = base.GenerateGradientCacheKey(model, X, y);
        return $"{baseKey}_LARS_{_options.InitialLearningRate}_{_options.TrustCoefficient}_{_options.WeightDecay}";
    }

    #region GPU Optimizer Support

    /// <summary>
    /// GPU buffer for velocity state.
    /// </summary>
    private IGpuBuffer? _gpuVelocity;

    /// <summary>
    /// Gets whether this optimizer supports GPU-accelerated parameter updates.
    /// </summary>
    public override bool SupportsGpuUpdate => true;

    /// <summary>
    /// Initializes LARS optimizer state on the GPU.
    /// </summary>
    public override void InitializeGpuState(int parameterCount, IDirectGpuBackend backend)
    {
        if (_gpuStateInitialized && _gpuVelocity != null)
            return;

        var zeros = new float[parameterCount];
        _gpuVelocity = backend.AllocateBuffer(zeros);

        _gpuStateInitialized = true;
    }

    /// <summary>
    /// Updates parameters on the GPU using the LARS kernel.
    /// </summary>
    public override void UpdateParametersGpu(IGpuBuffer parameters, IGpuBuffer gradients, int parameterCount, IDirectGpuBackend backend)
    {
        if (!_gpuStateInitialized || _gpuVelocity == null)
        {
            InitializeGpuState(parameterCount, backend);
        }

        backend.LarsUpdate(
            parameters,
            gradients,
            _gpuVelocity!,
            (float)NumOps.ToDouble(CurrentLearningRate),
            (float)_options.InitialMomentum,
            (float)_options.WeightDecay,
            (float)_options.TrustCoefficient,
            parameterCount
        );
    }

    /// <summary>
    /// Disposes GPU-allocated optimizer state.
    /// </summary>
    public override void DisposeGpuState()
    {
        _gpuVelocity?.Dispose();
        _gpuVelocity = null;
        _gpuStateInitialized = false;
    }

    #endregion
}
