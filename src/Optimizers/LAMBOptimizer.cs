using AiDotNet.Tensors.Engines.DirectGpu;
using Newtonsoft.Json;

namespace AiDotNet.Optimizers;

/// <summary>
/// Implements the LAMB (Layer-wise Adaptive Moments for Batch training) optimization algorithm.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// LAMB combines Adam's adaptive learning rates with LARS's layer-wise scaling, enabling
/// training with extremely large batch sizes (up to 32K) while maintaining accuracy.
/// </para>
/// <para><b>Key Formula:</b></para>
/// <code>
/// m = beta1 * m + (1 - beta1) * g
/// v = beta2 * v + (1 - beta2) * g^2
/// m_hat = m / (1 - beta1^t)
/// v_hat = v / (1 - beta2^t)
/// r = m_hat / (sqrt(v_hat) + epsilon) + weight_decay * w
/// trust_ratio = ||w|| / ||r||
/// w = w - lr * trust_ratio * r
/// </code>
/// <para><b>For Beginners:</b> LAMB is the optimizer of choice for training large language models
/// like BERT with massive batch sizes. It works by:
/// <list type="number">
/// <item>Computing Adam-style updates (momentum + adaptive learning rates)</item>
/// <item>Adding weight decay to prevent overfitting</item>
/// <item>Scaling the update per-layer based on weight/update magnitude ratios</item>
/// </list>
/// This combination allows training to scale linearly with batch size while maintaining
/// the same final accuracy as small-batch training.
/// </para>
/// <para>
/// Based on the paper "Large Batch Optimization for Deep Learning: Training BERT in 76 minutes"
/// by You et al. (2019).
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // For BERT-style transformer training
/// var options = new LAMBOptimizerOptions&lt;float, Matrix&lt;float&gt;, Vector&lt;float&gt;&gt;
/// {
///     InitialLearningRate = 0.00176 * Math.Sqrt(batchSize / 256.0),
///     Beta1 = 0.9,
///     Beta2 = 0.999,
///     WeightDecay = 0.01,
///     WarmupEpochs = 1
/// };
/// var optimizer = new LAMBOptimizer&lt;float, Matrix&lt;float&gt;, Vector&lt;float&gt;&gt;(model, options);
/// </code>
/// </example>
public class LAMBOptimizer<T, TInput, TOutput> : GradientBasedOptimizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// The options specific to the LAMB optimizer.
    /// </summary>
    private LAMBOptimizerOptions<T, TInput, TOutput> _options;

    /// <summary>
    /// The first moment vector (moving average of gradients).
    /// </summary>
    private Vector<T> _m;

    /// <summary>
    /// The second moment vector (moving average of squared gradients).
    /// </summary>
    private Vector<T> _v;

    /// <summary>
    /// The current time step (iteration count).
    /// </summary>
    private int _t;

    /// <summary>
    /// The total number of warmup steps.
    /// </summary>
    private int _warmupSteps;

    /// <summary>
    /// Previous first moment for reverse updates.
    /// </summary>
    private Vector<T>? _previousM;

    /// <summary>
    /// Previous second moment for reverse updates.
    /// </summary>
    private Vector<T>? _previousV;

    /// <summary>
    /// Previous step count for reverse updates.
    /// </summary>
    private int _previousT;

    /// <summary>
    /// Initializes a new instance of the LAMBOptimizer class.
    /// </summary>
    /// <param name="model">The model to optimize.</param>
    /// <param name="options">The options for configuring the LAMB optimizer.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This sets up the LAMB optimizer. Key parameters:
    /// <list type="bullet">
    /// <item>Learning rate: Use sqrt scaling (base_lr * sqrt(batch_size / 256))</item>
    /// <item>Beta1/Beta2: Keep at 0.9/0.999 for most cases</item>
    /// <item>Weight decay: 0.01 is typical for transformers</item>
    /// </list>
    /// </para>
    /// </remarks>
    public LAMBOptimizer(
        IFullModel<T, TInput, TOutput>? model,
        LAMBOptimizerOptions<T, TInput, TOutput>? options = null)
        : base(model, options ?? new())
    {
        _m = Vector<T>.Empty();
        _v = Vector<T>.Empty();
        _t = 0;
        _warmupSteps = 0;
        _options = options ?? new();

        InitializeAdaptiveParameters();
    }

    /// <summary>
    /// Initializes the adaptive parameters used by the LAMB optimizer.
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
    /// Gets the current beta1 value.
    /// </summary>
    public double Beta1 => _options.Beta1;

    /// <summary>
    /// Gets the current beta2 value.
    /// </summary>
    public double Beta2 => _options.Beta2;

    /// <summary>
    /// Performs the optimization process using the LAMB algorithm.
    /// </summary>
    /// <param name="inputData">The input data for optimization.</param>
    /// <returns>The result of the optimization process.</returns>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        // Initialize with random solution
        var currentSolution = InitializeRandomSolution(inputData.XTrain);
        var bestStepData = new OptimizationStepData<T, TInput, TOutput>();
        var parameters = currentSolution.GetParameters();
        _m = new Vector<T>(parameters.Length);
        _v = new Vector<T>(parameters.Length);
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

                // Update solution using LAMB algorithm
                var newSolution = UpdateSolutionWithLAMB(currentSolution, gradient, effectiveLr);

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
    /// Updates the solution using the LAMB algorithm.
    /// </summary>
    private IFullModel<T, TInput, TOutput> UpdateSolutionWithLAMB(
        IFullModel<T, TInput, TOutput> currentSolution,
        Vector<T> gradient,
        double effectiveLr)
    {
        var parameters = currentSolution.GetParameters();

        // Get layer boundaries or treat as single layer
        var layerBoundaries = _options.LayerBoundaries ?? [parameters.Length];
        var skipLayers = new HashSet<int>(_options.SkipTrustRatioLayers ?? []);

        int layerStart = 0;
        var layerIndex = 0;

        // Pre-compute constants
        T beta1 = NumOps.FromDouble(_options.Beta1);
        T beta2 = NumOps.FromDouble(_options.Beta2);
        T oneMinusBeta1 = NumOps.FromDouble(1.0 - _options.Beta1);
        T oneMinusBeta2 = NumOps.FromDouble(1.0 - _options.Beta2);
        T epsilon = NumOps.FromDouble(_options.Epsilon);
        T weightDecay = NumOps.FromDouble(_options.WeightDecay);
        T baseLr = NumOps.FromDouble(effectiveLr);
        T maxTrustRatio = NumOps.FromDouble(_options.MaxTrustRatio);

        // Bias correction factors
        T biasCorrection1 = _options.UseBiasCorrection
            ? NumOps.FromDouble(1.0 - Math.Pow(_options.Beta1, _t))
            : NumOps.One;
        T biasCorrection2 = _options.UseBiasCorrection
            ? NumOps.FromDouble(1.0 - Math.Pow(_options.Beta2, _t))
            : NumOps.One;

        // Process each layer
        foreach (var layerEnd in layerBoundaries)
        {
            int layerSize = layerEnd - layerStart;
            if (layerSize <= 0)
            {
                layerIndex++;
                continue;
            }

            // Extract layer parameters, gradients, and moment vectors
            var layerParams = ExtractLayerVector(parameters, layerStart, layerSize);
            var layerGrad = ExtractLayerVector(gradient, layerStart, layerSize);
            var layerM = ExtractLayerVector(_m, layerStart, layerSize);
            var layerV = ExtractLayerVector(_v, layerStart, layerSize);

            // Update first moment: m = beta1 * m + (1 - beta1) * g
            var mScaled = (Vector<T>)Engine.Multiply(layerM, beta1);
            var gradScaled = (Vector<T>)Engine.Multiply(layerGrad, oneMinusBeta1);
            var newM = (Vector<T>)Engine.Add(mScaled, gradScaled);

            // Update second moment: v = beta2 * v + (1 - beta2) * g^2
            var gradSquared = (Vector<T>)Engine.Multiply(layerGrad, layerGrad);
            var vScaled = (Vector<T>)Engine.Multiply(layerV, beta2);
            var gradSquaredScaled = (Vector<T>)Engine.Multiply(gradSquared, oneMinusBeta2);
            var newV = (Vector<T>)Engine.Add(vScaled, gradSquaredScaled);

            // Store updated moments
            UpdateLayerVector(_m, newM, layerStart, layerSize);
            UpdateLayerVector(_v, newV, layerStart, layerSize);

            // Bias correction
            var mHat = (Vector<T>)Engine.Divide(newM, biasCorrection1);
            var vHat = (Vector<T>)Engine.Divide(newV, biasCorrection2);

            // Compute Adam update: r = m_hat / (sqrt(v_hat) + epsilon)
            var vHatSqrt = (Vector<T>)Engine.Sqrt(vHat);
            var epsilonVec = Vector<T>.CreateDefault(vHatSqrt.Length, epsilon);
            var denominator = (Vector<T>)Engine.Add(vHatSqrt, epsilonVec);
            var adamUpdate = (Vector<T>)Engine.Divide(mHat, denominator);

            // Add weight decay: r = r + weight_decay * w
            var weightDecayTerm = (Vector<T>)Engine.Multiply(layerParams, weightDecay);
            var fullUpdate = (Vector<T>)Engine.Add(adamUpdate, weightDecayTerm);

            // Compute trust ratio for LAMB
            T trustRatio;
            if (skipLayers.Contains(layerIndex))
            {
                // Skip trust ratio for this layer, use 1.0
                trustRatio = NumOps.One;
            }
            else
            {
                // Compute parameter norm ||w||
                T paramNorm = ComputeL2Norm(layerParams);

                // Compute update norm ||r||
                T updateNorm = ComputeL2Norm(fullUpdate);

                // Trust ratio: ||w|| / ||r||
                T zero = NumOps.Zero;
                bool paramNormZero = NumOps.LessThan(paramNorm, epsilon);
                bool updateNormZero = NumOps.LessThan(updateNorm, epsilon);

                if (paramNormZero || updateNormZero)
                {
                    trustRatio = NumOps.One;
                }
                else
                {
                    trustRatio = NumOps.Divide(paramNorm, updateNorm);

                    // Optionally clip trust ratio
                    if (_options.ClipTrustRatio)
                    {
                        if (NumOps.GreaterThan(trustRatio, maxTrustRatio))
                        {
                            trustRatio = maxTrustRatio;
                        }
                        if (NumOps.LessThan(trustRatio, zero))
                        {
                            trustRatio = zero;
                        }
                    }
                }
            }

            // Final update: w = w - lr * trust_ratio * r
            var scaledUpdate = (Vector<T>)Engine.Multiply(fullUpdate, NumOps.Multiply(baseLr, trustRatio));
            var newLayerParams = (Vector<T>)Engine.Subtract(layerParams, scaledUpdate);

            // Write back to parameters
            UpdateLayerVector(parameters, newLayerParams, layerStart, layerSize);

            layerStart = layerEnd;
            layerIndex++;
        }

        return currentSolution.WithParameters(parameters);
    }

    /// <summary>
    /// Updates the current solution using the LAMB update rule.
    /// </summary>
    protected override IFullModel<T, TInput, TOutput> UpdateSolution(
        IFullModel<T, TInput, TOutput> currentSolution,
        Vector<T> gradient)
    {
        return UpdateSolutionWithLAMB(currentSolution, gradient, GetWarmupLearningRate());
    }

    /// <summary>
    /// Updates a vector of parameters using the LAMB optimization algorithm.
    /// </summary>
    public override Vector<T> UpdateParameters(Vector<T> parameters, Vector<T> gradient)
    {
        if (_m == null || _v == null || _m.Length != parameters.Length)
        {
            _m = new Vector<T>(parameters.Length);
            _v = new Vector<T>(parameters.Length);
            _previousM = new Vector<T>(parameters.Length);
            _previousV = new Vector<T>(parameters.Length);
            _t = 0;
        }

        // Save pre-update state
        if (_previousM == null || _previousV == null || _previousM.Length != parameters.Length)
        {
            _previousM = new Vector<T>(parameters.Length);
            _previousV = new Vector<T>(parameters.Length);
        }
        _previousM = new Vector<T>(_m);
        _previousV = new Vector<T>(_v);
        _previousT = _t;

        _t++;

        var effectiveLr = GetWarmupLearningRate();
        return UpdateParametersLAMB(parameters, gradient, effectiveLr);
    }

    /// <summary>
    /// Internal LAMB parameter update.
    /// </summary>
    private Vector<T> UpdateParametersLAMB(Vector<T> parameters, Vector<T> gradient, double effectiveLr)
    {
        T beta1 = NumOps.FromDouble(_options.Beta1);
        T beta2 = NumOps.FromDouble(_options.Beta2);
        T oneMinusBeta1 = NumOps.FromDouble(1.0 - _options.Beta1);
        T oneMinusBeta2 = NumOps.FromDouble(1.0 - _options.Beta2);
        T epsilon = NumOps.FromDouble(_options.Epsilon);
        T weightDecay = NumOps.FromDouble(_options.WeightDecay);
        T baseLr = NumOps.FromDouble(effectiveLr);
        T maxTrustRatio = NumOps.FromDouble(_options.MaxTrustRatio);

        // Bias correction
        T biasCorrection1 = _options.UseBiasCorrection
            ? NumOps.FromDouble(1.0 - Math.Pow(_options.Beta1, _t))
            : NumOps.One;
        T biasCorrection2 = _options.UseBiasCorrection
            ? NumOps.FromDouble(1.0 - Math.Pow(_options.Beta2, _t))
            : NumOps.One;

        // Update first moment
        var mScaled = (Vector<T>)Engine.Multiply(_m, beta1);
        var gradScaled = (Vector<T>)Engine.Multiply(gradient, oneMinusBeta1);
        _m = (Vector<T>)Engine.Add(mScaled, gradScaled);

        // Update second moment
        var gradSquared = (Vector<T>)Engine.Multiply(gradient, gradient);
        var vScaled = (Vector<T>)Engine.Multiply(_v, beta2);
        var gradSquaredScaled = (Vector<T>)Engine.Multiply(gradSquared, oneMinusBeta2);
        _v = (Vector<T>)Engine.Add(vScaled, gradSquaredScaled);

        // Bias correction
        var mHat = (Vector<T>)Engine.Divide(_m, biasCorrection1);
        var vHat = (Vector<T>)Engine.Divide(_v, biasCorrection2);

        // Adam update
        var vHatSqrt = (Vector<T>)Engine.Sqrt(vHat);
        var epsilonVec = Vector<T>.CreateDefault(vHatSqrt.Length, epsilon);
        var denominator = (Vector<T>)Engine.Add(vHatSqrt, epsilonVec);
        var adamUpdate = (Vector<T>)Engine.Divide(mHat, denominator);

        // Add weight decay
        var weightDecayTerm = (Vector<T>)Engine.Multiply(parameters, weightDecay);
        var fullUpdate = (Vector<T>)Engine.Add(adamUpdate, weightDecayTerm);

        // Compute trust ratio
        T paramNorm = ComputeL2Norm(parameters);
        T updateNorm = ComputeL2Norm(fullUpdate);

        T trustRatio;
        bool paramNormZero = NumOps.LessThan(paramNorm, epsilon);
        bool updateNormZero = NumOps.LessThan(updateNorm, epsilon);

        if (paramNormZero || updateNormZero)
        {
            trustRatio = NumOps.One;
        }
        else
        {
            trustRatio = NumOps.Divide(paramNorm, updateNorm);
            if (_options.ClipTrustRatio && NumOps.GreaterThan(trustRatio, maxTrustRatio))
            {
                trustRatio = maxTrustRatio;
            }
        }

        // Apply update
        var scaledUpdate = (Vector<T>)Engine.Multiply(fullUpdate, NumOps.Multiply(baseLr, trustRatio));
        return (Vector<T>)Engine.Subtract(parameters, scaledUpdate);
    }

    /// <summary>
    /// Updates a matrix of parameters using the LAMB optimization algorithm.
    /// </summary>
    public override Matrix<T> UpdateParameters(Matrix<T> parameters, Matrix<T> gradient)
    {
        int totalSize = parameters.Rows * parameters.Columns;

        if (_m == null || _v == null || _m.Length != totalSize)
        {
            _m = new Vector<T>(totalSize);
            _v = new Vector<T>(totalSize);
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

        // Apply LAMB update
        var updatedVec = UpdateParametersLAMB(paramVec, gradVec, GetWarmupLearningRate());

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
        if (inputData.XTrain is Matrix<T> matrix)
        {
            return matrix.Rows;
        }
        if (inputData.XTrain is Tensor<T> tensor && tensor.Shape.Length > 0)
        {
            return tensor.Shape[0];
        }
        return 1000;
    }

    /// <summary>
    /// Reverses a LAMB gradient update to recover original parameters.
    /// </summary>
    public override Vector<T> ReverseUpdate(Vector<T> updatedParameters, Vector<T> appliedGradients)
    {
        if (updatedParameters == null)
            throw new ArgumentNullException(nameof(updatedParameters));
        if (appliedGradients == null)
            throw new ArgumentNullException(nameof(appliedGradients));

        if (_previousM == null || _previousV == null || _previousT == 0)
        {
            return base.ReverseUpdate(updatedParameters, appliedGradients);
        }

        // LAMB reverse is approximate due to trust ratio complexity
        // Recompute the update direction and reverse it
        T beta1 = NumOps.FromDouble(_options.Beta1);
        T beta2 = NumOps.FromDouble(_options.Beta2);
        T oneMinusBeta1 = NumOps.FromDouble(1.0 - _options.Beta1);
        T oneMinusBeta2 = NumOps.FromDouble(1.0 - _options.Beta2);
        T epsilon = NumOps.FromDouble(_options.Epsilon);

        // Recompute moments at update time
        var mAtUpdate = (Vector<T>)Engine.Add(
            (Vector<T>)Engine.Multiply(_previousM, beta1),
            (Vector<T>)Engine.Multiply(appliedGradients, oneMinusBeta1));

        var gradSquared = (Vector<T>)Engine.Multiply(appliedGradients, appliedGradients);
        var vAtUpdate = (Vector<T>)Engine.Add(
            (Vector<T>)Engine.Multiply(_previousV, beta2),
            (Vector<T>)Engine.Multiply(gradSquared, oneMinusBeta2));

        // Bias correction
        T biasCorrection1 = NumOps.FromDouble(1.0 - Math.Pow(_options.Beta1, _previousT + 1));
        T biasCorrection2 = NumOps.FromDouble(1.0 - Math.Pow(_options.Beta2, _previousT + 1));

        var mHat = (Vector<T>)Engine.Divide(mAtUpdate, biasCorrection1);
        var vHat = (Vector<T>)Engine.Divide(vAtUpdate, biasCorrection2);

        // Compute the update that was applied
        var vHatSqrt = (Vector<T>)Engine.Sqrt(vHat);
        var epsilonVec = Vector<T>.CreateDefault(vHatSqrt.Length, epsilon);
        var denominator = (Vector<T>)Engine.Add(vHatSqrt, epsilonVec);
        var adamUpdate = (Vector<T>)Engine.Divide(mHat, denominator);

        // Weight decay term (using updated params as approximation)
        var weightDecay = NumOps.FromDouble(_options.WeightDecay);
        var weightDecayTerm = (Vector<T>)Engine.Multiply(updatedParameters, weightDecay);
        var fullUpdate = (Vector<T>)Engine.Add(adamUpdate, weightDecayTerm);

        // Approximate reversal
        var baseLr = NumOps.FromDouble(_options.InitialLearningRate);
        var scaledUpdate = (Vector<T>)Engine.Multiply(fullUpdate, baseLr);

        return (Vector<T>)Engine.Add(updatedParameters, scaledUpdate);
    }

    /// <summary>
    /// Resets the optimizer's internal state.
    /// </summary>
    public override void Reset()
    {
        base.Reset();
        _m = Vector<T>.Empty();
        _v = Vector<T>.Empty();
        _previousM = null;
        _previousV = null;
        _t = 0;
    }

    /// <summary>
    /// Updates the optimizer's options.
    /// </summary>
    protected override void UpdateOptions(OptimizationAlgorithmOptions<T, TInput, TOutput> options)
    {
        if (options is LAMBOptimizerOptions<T, TInput, TOutput> lambOptions)
        {
            _options = lambOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected LAMBOptimizerOptions.");
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

        writer.Write(_m.Length);
        foreach (var value in _m)
        {
            writer.Write(Convert.ToDouble(value));
        }

        writer.Write(_v.Length);
        foreach (var value in _v)
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

        using (var baseMs = new MemoryStream(baseData))
        using (var baseReader = new BinaryReader(baseMs))
        {
            string typeName = baseReader.ReadString();
            if (typeName != this.GetType().AssemblyQualifiedName)
            {
                throw new InvalidOperationException("Mismatched optimizer type during deserialization.");
            }
            baseReader.ReadString();
        }

        string optionsJson = reader.ReadString();
        _options = JsonConvert.DeserializeObject<LAMBOptimizerOptions<T, TInput, TOutput>>(optionsJson)
            ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");

        _t = reader.ReadInt32();
        _warmupSteps = reader.ReadInt32();

        int mLength = reader.ReadInt32();
        _m = new Vector<T>(mLength);
        for (int i = 0; i < mLength; i++)
        {
            _m[i] = NumOps.FromDouble(reader.ReadDouble());
        }

        int vLength = reader.ReadInt32();
        _v = new Vector<T>(vLength);
        for (int i = 0; i < vLength; i++)
        {
            _v[i] = NumOps.FromDouble(reader.ReadDouble());
        }

        InitializeAdaptiveParameters();
    }

    /// <summary>
    /// Generates a unique key for caching gradients.
    /// </summary>
    protected override string GenerateGradientCacheKey(IFullModel<T, TInput, TOutput> model, TInput X, TOutput y)
    {
        var baseKey = base.GenerateGradientCacheKey(model, X, y);
        return $"{baseKey}_LAMB_{_options.InitialLearningRate}_{_options.Beta1}_{_options.Beta2}_{_options.WeightDecay}";
    }

    #region GPU Optimizer Support

    /// <summary>
    /// GPU buffer for first moment estimates (m).
    /// </summary>
    private IGpuBuffer? _gpuM;

    /// <summary>
    /// GPU buffer for second moment estimates (v).
    /// </summary>
    private IGpuBuffer? _gpuV;

    /// <summary>
    /// Gets whether this optimizer supports GPU-accelerated parameter updates.
    /// </summary>
    public override bool SupportsGpuUpdate => true;

    /// <summary>
    /// Initializes LAMB optimizer state on the GPU.
    /// </summary>
    public override void InitializeGpuState(int parameterCount, IDirectGpuBackend backend)
    {
        if (_gpuStateInitialized && _gpuM != null && _gpuV != null)
            return;

        var zeros = new float[parameterCount];
        _gpuM = backend.AllocateBuffer(zeros);
        _gpuV = backend.AllocateBuffer(zeros);

        _t = 0;
        _gpuStateInitialized = true;
    }

    /// <summary>
    /// Updates parameters on the GPU using the LAMB kernel.
    /// </summary>
    public override void UpdateParametersGpu(IGpuBuffer parameters, IGpuBuffer gradients, int parameterCount, IDirectGpuBackend backend)
    {
        if (!_gpuStateInitialized || _gpuM == null || _gpuV == null)
        {
            InitializeGpuState(parameterCount, backend);
        }

        _t++;

        backend.LambUpdate(
            parameters,
            gradients,
            _gpuM!,
            _gpuV!,
            (float)NumOps.ToDouble(CurrentLearningRate),
            (float)_options.Beta1,
            (float)_options.Beta2,
            (float)_options.Epsilon,
            (float)_options.WeightDecay,
            _t,
            parameterCount
        );
    }

    /// <summary>
    /// Disposes GPU-allocated optimizer state.
    /// </summary>
    public override void DisposeGpuState()
    {
        _gpuM?.Dispose();
        _gpuM = null;
        _gpuV?.Dispose();
        _gpuV = null;
        _gpuStateInitialized = false;
    }

    #endregion
}
