using Newtonsoft.Json;

namespace AiDotNet.Optimizers;

/// <summary>
/// Implements the AdamW (Adam with decoupled Weight decay) optimization algorithm.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// AdamW is a variant of Adam that fixes the weight decay implementation. In standard Adam with L2 regularization,
/// weight decay is coupled with the adaptive learning rate, which can lead to suboptimal regularization effects.
/// AdamW decouples weight decay from the gradient-based update, applying it directly to the weights.
/// </para>
/// <para>
/// The key difference:
/// - Adam with L2: gradient = gradient + lambda * weights (then apply Adam update)
/// - AdamW: weights = weights - lr * adam_update - lr * lambda * weights (decoupled)
/// </para>
/// <para><b>For Beginners:</b> AdamW is like Adam but handles regularization (preventing overfitting) in a smarter way.
/// The difference might seem technical, but AdamW consistently achieves better results on tasks like training transformers
/// and large neural networks. If you're choosing between Adam and AdamW, AdamW is generally the better choice.
/// </para>
/// <para>
/// Based on the paper "Decoupled Weight Decay Regularization" by Ilya Loshchilov and Frank Hutter.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var options = new AdamWOptimizerOptions&lt;float, Matrix&lt;float&gt;, Vector&lt;float&gt;&gt;
/// {
///     LearningRate = 0.001,
///     WeightDecay = 0.01,
///     Beta1 = 0.9,
///     Beta2 = 0.999
/// };
/// var optimizer = new AdamWOptimizer&lt;float, Matrix&lt;float&gt;, Vector&lt;float&gt;&gt;(model, options);
/// </code>
/// </example>
public class AdamWOptimizer<T, TInput, TOutput> : GradientBasedOptimizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// The options specific to the AdamW optimizer.
    /// </summary>
    private AdamWOptimizerOptions<T, TInput, TOutput> _options;

    /// <summary>
    /// The first moment vector (moving average of gradients).
    /// </summary>
    private Vector<T> _m;

    /// <summary>
    /// The second moment vector (moving average of squared gradients).
    /// </summary>
    private Vector<T> _v;

    /// <summary>
    /// Maximum of past squared gradients (used when AMSGrad is enabled).
    /// </summary>
    private Vector<T>? _vMax;

    /// <summary>
    /// The current time step (iteration count).
    /// </summary>
    private int _t;

    /// <summary>
    /// The current value of beta1 (exponential decay rate for first moment estimates).
    /// </summary>
    private T _currentBeta1;

    /// <summary>
    /// The current value of beta2 (exponential decay rate for second moment estimates).
    /// </summary>
    private T _currentBeta2;

    /// <summary>
    /// Stores the pre-update snapshot of first moment vector for accurate reverse updates.
    /// </summary>
    private Vector<T>? _previousM;

    /// <summary>
    /// Stores the pre-update snapshot of second moment vector for accurate reverse updates.
    /// </summary>
    private Vector<T>? _previousV;

    /// <summary>
    /// Stores the pre-update timestep for accurate reverse updates.
    /// </summary>
    private int _previousT;

    /// <summary>
    /// Initializes a new instance of the AdamWOptimizer class.
    /// </summary>
    /// <param name="model">The model to optimize.</param>
    /// <param name="options">The options for configuring the AdamW optimizer.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This sets up the AdamW optimizer with its initial configuration.
    /// The most important parameters are learning rate (how fast to learn) and weight decay (how much to regularize).
    /// </para>
    /// </remarks>
    public AdamWOptimizer(
        IFullModel<T, TInput, TOutput>? model,
        AdamWOptimizerOptions<T, TInput, TOutput>? options = null)
        : base(model, options ?? new())
    {
        _m = Vector<T>.Empty();
        _v = Vector<T>.Empty();
        _t = 0;
        _options = options ?? new();
        _currentBeta1 = NumOps.Zero;
        _currentBeta2 = NumOps.Zero;

        InitializeAdaptiveParameters();
    }

    /// <summary>
    /// Initializes the adaptive parameters used by the AdamW optimizer.
    /// </summary>
    protected override void InitializeAdaptiveParameters()
    {
        // Learning rate is handled by base class (synced with scheduler)
        _currentBeta1 = NumOps.FromDouble(_options.Beta1);
        _currentBeta2 = NumOps.FromDouble(_options.Beta2);
    }

    /// <summary>
    /// Gets the current weight decay coefficient.
    /// </summary>
    public double WeightDecay => _options.WeightDecay;

    /// <summary>
    /// Gets whether AMSGrad variant is enabled.
    /// </summary>
    public bool UseAMSGrad => _options.UseAMSGrad;

    /// <summary>
    /// Performs the optimization process using the AdamW algorithm.
    /// </summary>
    /// <param name="inputData">The input data for optimization, including training data and targets.</param>
    /// <returns>The result of the optimization process, including the best solution found.</returns>
    /// <remarks>
    /// <para><b>DataLoader Integration:</b>
    /// This optimizer now uses the DataLoader batching infrastructure which supports:
    /// - Custom samplers (weighted, stratified, curriculum, importance, active learning)
    /// - Reproducible shuffling via RandomSeed
    /// - Option to drop incomplete final batches
    /// Set these options via GradientBasedOptimizerOptions.DataSampler, ShuffleData, DropLastBatch, and RandomSeed.
    /// </para>
    /// </remarks>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        // Initialize with random solution
        var currentSolution = InitializeRandomSolution(inputData.XTrain);
        var bestStepData = new OptimizationStepData<T, TInput, TOutput>();
        var parameters = currentSolution.GetParameters();
        _m = new Vector<T>(parameters.Length);
        _v = new Vector<T>(parameters.Length);
        if (_options.UseAMSGrad)
        {
            _vMax = new Vector<T>(parameters.Length);
        }
        _t = 0;

        // Initialize parameters
        InitializeAdaptiveParameters();

        var previousStepData = PrepareAndEvaluateSolution(currentSolution, inputData);

        for (int epoch = 0; epoch < _options.MaxIterations; epoch++)
        {
            // Notify sampler of new epoch (for curriculum/self-paced learning)
            NotifyEpochStart(epoch);

            // Create batcher for the current epoch using DataLoader infrastructure
            var batcher = CreateBatcher(inputData, _options.BatchSize);

            foreach (var (xBatch, yBatch, batchIndices) in batcher.GetBatches())
            {
                _t++;
                // Calculate gradient on the batch
                var gradient = CalculateGradient(currentSolution, xBatch, yBatch);

                // Update solution using AdamW algorithm
                var newSolution = UpdateSolution(currentSolution, gradient);

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
    /// Updates the adaptive parameters of the optimizer based on the current and previous optimization steps.
    /// </summary>
    protected override void UpdateAdaptiveParameters(OptimizationStepData<T, TInput, TOutput> currentStepData, OptimizationStepData<T, TInput, TOutput> previousStepData)
    {
        base.UpdateAdaptiveParameters(currentStepData, previousStepData);

        if (_options.UseAdaptiveBetas)
        {
            _currentBeta1 = MathHelper.Max(NumOps.FromDouble(_options.MinBeta1),
                MathHelper.Min(NumOps.FromDouble(_options.MaxBeta1), _currentBeta1));
            _currentBeta2 = MathHelper.Max(NumOps.FromDouble(_options.MinBeta2),
                MathHelper.Min(NumOps.FromDouble(_options.MaxBeta2), _currentBeta2));
        }
    }

    /// <summary>
    /// Updates the current solution using the AdamW update rule with decoupled weight decay.
    /// </summary>
    /// <param name="currentSolution">The current solution being optimized.</param>
    /// <param name="gradient">The calculated gradient for the current solution.</param>
    /// <returns>A new solution with updated parameters.</returns>
    protected override IFullModel<T, TInput, TOutput> UpdateSolution(IFullModel<T, TInput, TOutput> currentSolution, Vector<T> gradient)
    {
        var parameters = currentSolution.GetParameters();

        T oneMinusBeta1 = NumOps.Subtract(NumOps.One, _currentBeta1);
        T oneMinusBeta2 = NumOps.Subtract(NumOps.One, _currentBeta2);
        T biasCorrection1 = NumOps.Subtract(NumOps.One, NumOps.Power(_currentBeta1, NumOps.FromDouble(_t)));
        T biasCorrection2 = NumOps.Subtract(NumOps.One, NumOps.Power(_currentBeta2, NumOps.FromDouble(_t)));
        T epsilon = NumOps.FromDouble(_options.Epsilon);
        T weightDecay = NumOps.FromDouble(_options.WeightDecay);

        // Update biased first moment: m = beta1 * m + (1 - beta1) * gradient
        var mScaled = (Vector<T>)Engine.Multiply(_m, _currentBeta1);
        var gradScaled = (Vector<T>)Engine.Multiply(gradient, oneMinusBeta1);
        _m = (Vector<T>)Engine.Add(mScaled, gradScaled);

        // Update biased second moment: v = beta2 * v + (1 - beta2) * gradient^2
        var gradSquared = (Vector<T>)Engine.Multiply(gradient, gradient);
        var vScaled = (Vector<T>)Engine.Multiply(_v, _currentBeta2);
        var gradSquaredScaled = (Vector<T>)Engine.Multiply(gradSquared, oneMinusBeta2);
        _v = (Vector<T>)Engine.Add(vScaled, gradSquaredScaled);

        // Compute bias-corrected first moment: mHat = m / (1 - beta1^t)
        var mHat = (Vector<T>)Engine.Divide(_m, biasCorrection1);

        // Compute bias-corrected second moment: vHat = v / (1 - beta2^t)
        var vHat = (Vector<T>)Engine.Divide(_v, biasCorrection2);

        // Handle AMSGrad variant using vectorized max operation
        Vector<T> vHatEffective;
        if (_options.UseAMSGrad && _vMax != null)
        {
            // Update vMax = max(vMax, vHat) using vectorized operation
            _vMax = (Vector<T>)Engine.Max(_vMax, vHat);
            vHatEffective = _vMax;
        }
        else
        {
            vHatEffective = vHat;
        }

        // Compute Adam update: update = mHat / (sqrt(vHat) + epsilon)
        var vHatSqrt = (Vector<T>)Engine.Sqrt(vHatEffective);
        var epsilonVec = Vector<T>.CreateDefault(vHatSqrt.Length, epsilon);
        var denominator = (Vector<T>)Engine.Add(vHatSqrt, epsilonVec);
        var adamUpdate = (Vector<T>)Engine.Divide(mHat, denominator);

        // Scale Adam update by learning rate
        var scaledAdamUpdate = (Vector<T>)Engine.Multiply(adamUpdate, CurrentLearningRate);

        // DECOUPLED WEIGHT DECAY: Apply weight decay directly to parameters
        // AdamW: parameters = parameters - lr * adam_update - lr * weight_decay * parameters
        var weightDecayTerm = (Vector<T>)Engine.Multiply(parameters, weightDecay);
        var scaledWeightDecay = (Vector<T>)Engine.Multiply(weightDecayTerm, CurrentLearningRate);

        // Combine: parameters = parameters - scaledAdamUpdate - scaledWeightDecay
        var afterAdamUpdate = (Vector<T>)Engine.Subtract(parameters, scaledAdamUpdate);
        var updatedParams = (Vector<T>)Engine.Subtract(afterAdamUpdate, scaledWeightDecay);

        return currentSolution.WithParameters(updatedParams);
    }

    /// <summary>
    /// Updates a vector of parameters using the AdamW optimization algorithm with decoupled weight decay.
    /// </summary>
    /// <param name="parameters">The current parameter vector to be updated.</param>
    /// <param name="gradient">The gradient vector corresponding to the parameters.</param>
    /// <returns>The updated parameter vector.</returns>
    public override Vector<T> UpdateParameters(Vector<T> parameters, Vector<T> gradient)
    {
        if (_m == null || _v == null || _m.Length != parameters.Length)
        {
            _m = new Vector<T>(parameters.Length);
            _v = new Vector<T>(parameters.Length);
            if (_options.UseAMSGrad)
            {
                _vMax = new Vector<T>(parameters.Length);
            }
            _previousM = new Vector<T>(parameters.Length);
            _previousV = new Vector<T>(parameters.Length);
            _t = 0;
        }

        // Save pre-update state for accurate reverse updates
        if (_previousM == null || _previousV == null)
        {
            _previousM = new Vector<T>(parameters.Length);
            _previousV = new Vector<T>(parameters.Length);
        }

        _previousM = new Vector<T>(_m);
        _previousV = new Vector<T>(_v);
        _previousT = _t;

        _t++;

        T beta1 = NumOps.FromDouble(_options.Beta1);
        T beta2 = NumOps.FromDouble(_options.Beta2);
        T oneMinusBeta1 = NumOps.FromDouble(1 - _options.Beta1);
        T oneMinusBeta2 = NumOps.FromDouble(1 - _options.Beta2);
        T epsilon = NumOps.FromDouble(_options.Epsilon);
        T biasCorrection1 = NumOps.FromDouble(1 - Math.Pow(_options.Beta1, _t));
        T biasCorrection2 = NumOps.FromDouble(1 - Math.Pow(_options.Beta2, _t));
        T weightDecay = NumOps.FromDouble(_options.WeightDecay);

        // Update biased first moment: m = beta1 * m + (1 - beta1) * gradient
        var mScaled = (Vector<T>)Engine.Multiply(_m, beta1);
        var gradScaled = (Vector<T>)Engine.Multiply(gradient, oneMinusBeta1);
        _m = (Vector<T>)Engine.Add(mScaled, gradScaled);

        // Update biased second moment: v = beta2 * v + (1 - beta2) * gradient^2
        var gradSquared = (Vector<T>)Engine.Multiply(gradient, gradient);
        var vScaled = (Vector<T>)Engine.Multiply(_v, beta2);
        var gradSquaredScaled = (Vector<T>)Engine.Multiply(gradSquared, oneMinusBeta2);
        _v = (Vector<T>)Engine.Add(vScaled, gradSquaredScaled);

        // Compute bias-corrected first moment: mHat = m / (1 - beta1^t)
        var mHat = (Vector<T>)Engine.Divide(_m, biasCorrection1);

        // Compute bias-corrected second moment: vHat = v / (1 - beta2^t)
        var vHat = (Vector<T>)Engine.Divide(_v, biasCorrection2);

        // Handle AMSGrad variant using vectorized max operation
        Vector<T> vHatEffective;
        if (_options.UseAMSGrad && _vMax != null)
        {
            // Update vMax = max(vMax, vHat) using vectorized operation
            _vMax = (Vector<T>)Engine.Max(_vMax, vHat);
            vHatEffective = _vMax;
        }
        else
        {
            vHatEffective = vHat;
        }

        // Compute Adam update: update = mHat / (sqrt(vHat) + epsilon)
        var vHatSqrt = (Vector<T>)Engine.Sqrt(vHatEffective);
        var epsilonVec = Vector<T>.CreateDefault(vHatSqrt.Length, epsilon);
        var denominator = (Vector<T>)Engine.Add(vHatSqrt, epsilonVec);
        var adamUpdate = (Vector<T>)Engine.Divide(mHat, denominator);

        // Scale Adam update by learning rate
        var scaledAdamUpdate = (Vector<T>)Engine.Multiply(adamUpdate, CurrentLearningRate);

        // DECOUPLED WEIGHT DECAY: Apply weight decay directly to parameters
        var weightDecayTerm = (Vector<T>)Engine.Multiply(parameters, weightDecay);
        var scaledWeightDecay = (Vector<T>)Engine.Multiply(weightDecayTerm, CurrentLearningRate);

        // Combine: parameters = parameters - scaledAdamUpdate - scaledWeightDecay
        var afterAdamUpdate = (Vector<T>)Engine.Subtract(parameters, scaledAdamUpdate);
        var updatedParameters = (Vector<T>)Engine.Subtract(afterAdamUpdate, scaledWeightDecay);

        return updatedParameters;
    }

    /// <summary>
    /// Updates a matrix of parameters using the AdamW optimization algorithm.
    /// </summary>
    public override Matrix<T> UpdateParameters(Matrix<T> parameters, Matrix<T> gradient)
    {
        int totalSize = parameters.Rows * parameters.Columns;

        if (_m == null || _v == null || _m.Length != totalSize)
        {
            _m = new Vector<T>(totalSize);
            _v = new Vector<T>(totalSize);
            if (_options.UseAMSGrad)
            {
                _vMax = new Vector<T>(totalSize);
            }
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

        // Apply AdamW update
        var updatedVec = UpdateParametersInternal(paramVec, gradVec);

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
    /// Internal method to update parameters without reinitializing moment vectors.
    /// </summary>
    private Vector<T> UpdateParametersInternal(Vector<T> parameters, Vector<T> gradient)
    {
        T beta1 = NumOps.FromDouble(_options.Beta1);
        T beta2 = NumOps.FromDouble(_options.Beta2);
        T oneMinusBeta1 = NumOps.FromDouble(1 - _options.Beta1);
        T oneMinusBeta2 = NumOps.FromDouble(1 - _options.Beta2);
        T epsilon = NumOps.FromDouble(_options.Epsilon);
        T biasCorrection1 = NumOps.FromDouble(1 - Math.Pow(_options.Beta1, _t));
        T biasCorrection2 = NumOps.FromDouble(1 - Math.Pow(_options.Beta2, _t));
        T weightDecay = NumOps.FromDouble(_options.WeightDecay);

        // Update moments
        var mScaled = (Vector<T>)Engine.Multiply(_m, beta1);
        var gradScaled = (Vector<T>)Engine.Multiply(gradient, oneMinusBeta1);
        _m = (Vector<T>)Engine.Add(mScaled, gradScaled);

        var gradSquared = (Vector<T>)Engine.Multiply(gradient, gradient);
        var vScaled = (Vector<T>)Engine.Multiply(_v, beta2);
        var gradSquaredScaled = (Vector<T>)Engine.Multiply(gradSquared, oneMinusBeta2);
        _v = (Vector<T>)Engine.Add(vScaled, gradSquaredScaled);

        // Bias correction
        var mHat = (Vector<T>)Engine.Divide(_m, biasCorrection1);
        var vHat = (Vector<T>)Engine.Divide(_v, biasCorrection2);

        // Compute update
        var vHatSqrt = (Vector<T>)Engine.Sqrt(vHat);
        var epsilonVec = Vector<T>.CreateDefault(vHatSqrt.Length, epsilon);
        var denominator = (Vector<T>)Engine.Add(vHatSqrt, epsilonVec);
        var adamUpdate = (Vector<T>)Engine.Divide(mHat, denominator);
        var scaledAdamUpdate = (Vector<T>)Engine.Multiply(adamUpdate, CurrentLearningRate);

        // Decoupled weight decay
        var weightDecayTerm = (Vector<T>)Engine.Multiply(parameters, weightDecay);
        var scaledWeightDecay = (Vector<T>)Engine.Multiply(weightDecayTerm, CurrentLearningRate);

        var afterAdamUpdate = (Vector<T>)Engine.Subtract(parameters, scaledAdamUpdate);
        return (Vector<T>)Engine.Subtract(afterAdamUpdate, scaledWeightDecay);
    }

    /// <summary>
    /// Reverses an AdamW gradient update to recover original parameters.
    /// </summary>
    public override Vector<T> ReverseUpdate(Vector<T> updatedParameters, Vector<T> appliedGradients)
    {
        if (updatedParameters == null)
            throw new ArgumentNullException(nameof(updatedParameters));
        if (appliedGradients == null)
            throw new ArgumentNullException(nameof(appliedGradients));

        if (updatedParameters.Length != appliedGradients.Length)
        {
            throw new ArgumentException(
                $"Updated parameters size ({updatedParameters.Length}) must match applied gradients size ({appliedGradients.Length})",
                nameof(appliedGradients));
        }

        if (_previousM == null || _previousV == null || _previousM.Length != updatedParameters.Length || _previousT == 0)
        {
            return base.ReverseUpdate(updatedParameters, appliedGradients);
        }

        // Recompute the moments that were used during the update
        var beta1Vec = Vector<T>.CreateDefault(_previousM.Length, NumOps.FromDouble(_options.Beta1));
        var oneMinusBeta1Vec = Vector<T>.CreateDefault(_previousM.Length, NumOps.FromDouble(1 - _options.Beta1));
        var beta2Vec = Vector<T>.CreateDefault(_previousV.Length, NumOps.FromDouble(_options.Beta2));
        var oneMinusBeta2Vec = Vector<T>.CreateDefault(_previousV.Length, NumOps.FromDouble(1 - _options.Beta2));

        var mAtUpdateTime = (Vector<T>)Engine.Add(
            (Vector<T>)Engine.Multiply(_previousM, beta1Vec),
            (Vector<T>)Engine.Multiply(appliedGradients, oneMinusBeta1Vec)
        );

        var gradSquared = (Vector<T>)Engine.Multiply(appliedGradients, appliedGradients);
        var vAtUpdateTime = (Vector<T>)Engine.Add(
            (Vector<T>)Engine.Multiply(_previousV, beta2Vec),
            (Vector<T>)Engine.Multiply(gradSquared, oneMinusBeta2Vec)
        );

        // Compute bias-corrected moments
        T biasCorrection1 = NumOps.FromDouble(1 - Math.Pow(_options.Beta1, _previousT + 1));
        T biasCorrection2 = NumOps.FromDouble(1 - Math.Pow(_options.Beta2, _previousT + 1));
        var biasCorrection1Vec = Vector<T>.CreateDefault(mAtUpdateTime.Length, biasCorrection1);
        var biasCorrection2Vec = Vector<T>.CreateDefault(vAtUpdateTime.Length, biasCorrection2);

        var mHat = (Vector<T>)Engine.Divide(mAtUpdateTime, biasCorrection1Vec);
        var vHat = (Vector<T>)Engine.Divide(vAtUpdateTime, biasCorrection2Vec);

        // Compute the Adam update that was applied
        var vHatSqrt = (Vector<T>)Engine.Sqrt(vHat);
        var epsilonVec = Vector<T>.CreateDefault(vHatSqrt.Length, NumOps.FromDouble(_options.Epsilon));
        var denominator = (Vector<T>)Engine.Add(vHatSqrt, epsilonVec);
        var adamUpdate = (Vector<T>)Engine.Divide(mHat, denominator);
        var currentLrVec = Vector<T>.CreateDefault(adamUpdate.Length, CurrentLearningRate);
        var scaledAdamUpdate = (Vector<T>)Engine.Multiply(adamUpdate, currentLrVec);

        // Reverse: params_old = params_new + scaledAdamUpdate + scaledWeightDecay
        // But we need the original params for weight decay, which is what we're trying to find
        // This is an approximation using the updated params
        var weightDecayVec = Vector<T>.CreateDefault(updatedParameters.Length, NumOps.FromDouble(_options.WeightDecay));
        var weightDecayTerm = (Vector<T>)Engine.Multiply(updatedParameters, weightDecayVec);
        var scaledWeightDecay = (Vector<T>)Engine.Multiply(weightDecayTerm, currentLrVec);

        var afterAdamReverse = (Vector<T>)Engine.Add(updatedParameters, scaledAdamUpdate);
        return (Vector<T>)Engine.Add(afterAdamReverse, scaledWeightDecay);
    }

    /// <summary>
    /// Resets the optimizer's internal state.
    /// </summary>
    public override void Reset()
    {
        base.Reset();
        _m = Vector<T>.Empty();
        _v = Vector<T>.Empty();
        _vMax = null;
        _t = 0;
    }

    /// <summary>
    /// Updates the optimizer's options.
    /// </summary>
    protected override void UpdateOptions(OptimizationAlgorithmOptions<T, TInput, TOutput> options)
    {
        if (options is AdamWOptimizerOptions<T, TInput, TOutput> adamWOptions)
        {
            _options = adamWOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected AdamWOptimizerOptions.");
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
        using (MemoryStream ms = new MemoryStream())
        using (BinaryWriter writer = new BinaryWriter(ms))
        {
            byte[] baseData = base.Serialize();
            writer.Write(baseData.Length);
            writer.Write(baseData);

            string optionsJson = JsonConvert.SerializeObject(_options);
            writer.Write(optionsJson);

            writer.Write(_t);
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

            // Serialize vMax if AMSGrad is enabled
            writer.Write(_vMax != null);
            if (_vMax != null)
            {
                writer.Write(_vMax.Length);
                foreach (var value in _vMax)
                {
                    writer.Write(Convert.ToDouble(value));
                }
            }

            return ms.ToArray();
        }
    }

    /// <summary>
    /// Deserializes the optimizer's state from a byte array.
    /// </summary>
    public override void Deserialize(byte[] data)
    {
        using (MemoryStream ms = new MemoryStream(data))
        using (BinaryReader reader = new BinaryReader(ms))
        {
            int baseDataLength = reader.ReadInt32();
            byte[] baseData = reader.ReadBytes(baseDataLength);

            // Read base class data manually to avoid type mismatch in UpdateOptions
            using (MemoryStream baseMs = new MemoryStream(baseData))
            using (BinaryReader baseReader = new BinaryReader(baseMs))
            {
                // Read and verify the type (same as base class)
                string typeName = baseReader.ReadString();
                if (typeName != this.GetType().AssemblyQualifiedName)
                {
                    throw new InvalidOperationException("Mismatched optimizer type during deserialization.");
                }

                // Skip the options JSON from base class - we'll read our own below
                baseReader.ReadString();

                // Read additional base class data if any
                // (The base class DeserializeAdditionalData is empty, so nothing to do)
            }

            string optionsJson = reader.ReadString();
            _options = JsonConvert.DeserializeObject<AdamWOptimizerOptions<T, TInput, TOutput>>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");

            _t = reader.ReadInt32();
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

            // Deserialize vMax if present
            bool hasVMax = reader.ReadBoolean();
            if (hasVMax)
            {
                int vMaxLength = reader.ReadInt32();
                _vMax = new Vector<T>(vMaxLength);
                for (int i = 0; i < vMaxLength; i++)
                {
                    _vMax[i] = NumOps.FromDouble(reader.ReadDouble());
                }
            }

            InitializeAdaptiveParameters();
        }
    }

    /// <summary>
    /// Generates a unique key for caching gradients.
    /// </summary>
    protected override string GenerateGradientCacheKey(IFullModel<T, TInput, TOutput> model, TInput X, TOutput y)
    {
        var baseKey = base.GenerateGradientCacheKey(model, X, y);
        return $"{baseKey}_AdamW_{_options.InitialLearningRate}_{_options.WeightDecay}_{_options.MaxIterations}";
    }
}
