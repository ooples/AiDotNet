using Newtonsoft.Json;

namespace AiDotNet.Optimizers;

/// <summary>
/// Implements the Lion (Evolved Sign Momentum) optimization algorithm for gradient-based optimization.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Lion is a modern optimization algorithm discovered through symbolic program search that offers significant
/// advantages over traditional optimizers like Adam. It achieves 50% memory reduction by maintaining only a
/// single momentum state (compared to Adam's two states) while often achieving superior performance on large
/// transformer models and other deep learning architectures.
/// </para>
/// <para>
/// The algorithm uses sign-based gradient updates, which provides implicit regularization and better
/// generalization. Unlike Adam's magnitude-based updates, Lion focuses purely on the direction of gradients,
/// making it more robust to gradient scale variations and leading to more consistent training dynamics.
/// </para>
/// <para><b>For Beginners:</b> Lion is like a simplified but more powerful version of Adam. Instead of
/// carefully measuring how big each step should be (like Adam does), Lion only looks at which direction
/// to go and takes consistent-sized steps in that direction. This is like following a compass that only
/// shows direction - it's simpler, uses less memory, and often gets you to your destination faster.
/// Lion is particularly good for training large neural networks.</para>
/// </remarks>
public class LionOptimizer<T, TInput, TOutput> : GradientBasedOptimizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// The options specific to the Lion optimizer.
    /// </summary>
    private LionOptimizerOptions<T, TInput, TOutput> _options;

    /// <summary>
    /// The momentum vector (exponentially moving average of gradients).
    /// </summary>
    private Vector<T> _m;

    /// <summary>
    /// The current time step (iteration count).
    /// </summary>
    private int _t;

    /// <summary>
    /// The current value of beta1 (interpolation momentum).
    /// </summary>
    private T _currentBeta1;

    /// <summary>
    /// The current value of beta2 (update momentum).
    /// </summary>
    private T _currentBeta2;

    /// <summary>
    /// Initializes a new instance of the LionOptimizer class.
    /// </summary>
    /// <param name="model">The model to optimize.</param>
    /// <param name="options">The options for configuring the Lion optimizer.</param>
    /// <param name="engine">The computation engine (CPU or GPU) for vectorized operations.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This sets up the Lion optimizer with its initial configuration.
    /// Lion requires minimal tuning compared to other optimizers - the default settings work well
    /// for most deep learning problems. The main parameter you might want to adjust is the learning
    /// rate, which is typically set lower than Adam (around 1e-4 instead of 1e-3).</para>
    /// </remarks>
    public LionOptimizer(
        IFullModel<T, TInput, TOutput>? model,
        LionOptimizerOptions<T, TInput, TOutput>? options = null,
        IEngine? engine = null)
        : base(model, options ?? new())
    {
        _m = Vector<T>.Empty();
        _t = 0;
        _options = options ?? new();
        _currentBeta1 = NumOps.Zero;
        _currentBeta2 = NumOps.Zero;

        InitializeAdaptiveParameters();
    }

    /// <summary>
    /// Initializes the adaptive parameters used by the Lion optimizer.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This sets up the momentum factors.
    /// Lion typically uses fixed values for these parameters, but they can be made adaptive if needed.
    /// Learning rate is handled by the base class and synced with any configured scheduler.</para>
    /// </remarks>
    protected override void InitializeAdaptiveParameters()
    {
        // Learning rate is handled by base class (synced with scheduler)
        _currentBeta1 = NumOps.FromDouble(_options.Beta1);
        _currentBeta2 = NumOps.FromDouble(_options.Beta2);
    }

    /// <summary>
    /// Performs the optimization process using the Lion algorithm.
    /// </summary>
    /// <param name="inputData">The input data for optimization, including training data and targets.</param>
    /// <returns>The result of the optimization process, including the best solution found.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the main learning process. It repeatedly improves the model's
    /// parameters using the Lion algorithm. Lion's sign-based updates make it particularly efficient
    /// for large-scale optimization problems, often converging faster than Adam while using less memory.</para>
    /// </remarks>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        var currentSolution = InitializeRandomSolution(inputData.XTrain);
        var bestStepData = new OptimizationStepData<T, TInput, TOutput>();
        var parameters = currentSolution.GetParameters();
        _m = new Vector<T>(parameters.Length);
        _t = 0;

        InitializeAdaptiveParameters();

        var previousStepData = PrepareAndEvaluateSolution(currentSolution, inputData);

        for (int iteration = 0; iteration < _options.MaxIterations; iteration++)
        {
            _t++;
            var gradient = CalculateGradient(currentSolution, inputData.XTrain, inputData.YTrain);
            var newSolution = UpdateSolution(currentSolution, gradient);

            var currentStepData = EvaluateSolution(newSolution, inputData);
            UpdateBestSolution(currentStepData, ref bestStepData);
            UpdateAdaptiveParameters(currentStepData, previousStepData);

            if (UpdateIterationHistoryAndCheckEarlyStopping(iteration, bestStepData))
            {
                break;
            }

            if (NumOps.LessThan(NumOps.Abs(NumOps.Subtract(bestStepData.FitnessScore, currentStepData.FitnessScore)), NumOps.FromDouble(_options.Tolerance)))
            {
                break;
            }

            currentSolution = newSolution;
            previousStepData = currentStepData;
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    /// <summary>
    /// Updates the adaptive parameters of the optimizer based on the current and previous optimization steps.
    /// </summary>
    /// <param name="currentStepData">Data from the current optimization step.</param>
    /// <param name="previousStepData">Data from the previous optimization step.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method can adjust how the optimizer learns based on its recent performance.
    /// However, Lion typically works well with fixed parameters, so this is mainly useful for advanced scenarios.</para>
    /// </remarks>
    protected override void UpdateAdaptiveParameters(OptimizationStepData<T, TInput, TOutput> currentStepData, OptimizationStepData<T, TInput, TOutput> previousStepData)
    {
        base.UpdateAdaptiveParameters(currentStepData, previousStepData);

        // Adaptive Beta1 updates (if enabled)
        if (_options.UseAdaptiveBeta1)
        {
            // Increase Beta1 (more smoothing) if fitness is improving, decrease (faster adaptation) otherwise
            if (NumOps.GreaterThan(currentStepData.FitnessScore, previousStepData.FitnessScore))
            {
                _currentBeta1 = NumOps.Multiply(_currentBeta1, NumOps.FromDouble(_options.Beta1IncreaseFactor));
            }
            else
            {
                _currentBeta1 = NumOps.Multiply(_currentBeta1, NumOps.FromDouble(_options.Beta1DecreaseFactor));
            }

            // Clamp to configured bounds
            _currentBeta1 = MathHelper.Max(NumOps.FromDouble(_options.MinBeta1),
                MathHelper.Min(NumOps.FromDouble(_options.MaxBeta1), _currentBeta1));
        }

        // Adaptive Beta2 updates (if enabled)
        if (_options.UseAdaptiveBeta2)
        {
            // Increase Beta2 (more stability) if fitness is improving, decrease otherwise
            if (NumOps.GreaterThan(currentStepData.FitnessScore, previousStepData.FitnessScore))
            {
                _currentBeta2 = NumOps.Multiply(_currentBeta2, NumOps.FromDouble(_options.Beta2IncreaseFactor));
            }
            else
            {
                _currentBeta2 = NumOps.Multiply(_currentBeta2, NumOps.FromDouble(_options.Beta2DecreaseFactor));
            }

            // Clamp to configured bounds
            _currentBeta2 = MathHelper.Max(NumOps.FromDouble(_options.MinBeta2),
                MathHelper.Min(NumOps.FromDouble(_options.MaxBeta2), _currentBeta2));
        }
    }

    /// <summary>
    /// Updates the current solution using the Lion update rule.
    /// </summary>
    /// <param name="currentSolution">The current solution being optimized.</param>
    /// <param name="gradient">The calculated gradient for the current solution.</param>
    /// <returns>A new solution with updated parameters.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method applies the Lion algorithm's unique approach to parameter updates.
    /// The algorithm works in three steps:
    /// 1. Interpolate between current gradient and past momentum
    /// 2. Take the sign of this interpolation and use it to update parameters
    /// 3. Update the momentum for the next iteration
    /// This sign-based approach is what makes Lion both simple and powerful.</para>
    /// </remarks>
    protected override IFullModel<T, TInput, TOutput> UpdateSolution(IFullModel<T, TInput, TOutput> currentSolution, Vector<T> gradient)
    {
        // === Vectorized Lion Update using IEngine (Phase B: US-GPU-015) ===
        // c_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        // theta_t = theta_{t-1} - lr * (sign(c_t) + lambda * theta_{t-1})
        // m_t = beta2 * m_{t-1} + (1 - beta2) * g_t

        var parameters = currentSolution.GetParameters();
        var weightDecay = NumOps.FromDouble(_options.WeightDecay);
        var effectiveLearningRate = CurrentLearningRate;

        // Step 1: Interpolate between momentum and gradient
        var oneMinusBeta1 = NumOps.Subtract(NumOps.One, _currentBeta1);
        var beta1TimesM = (Vector<T>)Engine.Multiply(_m, _currentBeta1);
        var oneMinusBeta1TimesGrad = (Vector<T>)Engine.Multiply(gradient, oneMinusBeta1);
        var interpolated = (Vector<T>)Engine.Add(beta1TimesM, oneMinusBeta1TimesGrad);

        // Step 2: Compute sign of interpolated values
        var signVec = (Vector<T>)Engine.Sign(interpolated);

        // Step 3: Apply weight decay if needed
        var update = signVec;
        if (!NumOps.Equals(weightDecay, NumOps.Zero))
        {
            var weightDecayTerm = (Vector<T>)Engine.Multiply(parameters, weightDecay);
            update = (Vector<T>)Engine.Add(update, weightDecayTerm);
        }

        // Step 4: Update parameters
        var lrTimesUpdate = (Vector<T>)Engine.Multiply(update, CurrentLearningRate);
        var newParameters = (Vector<T>)Engine.Subtract(parameters, lrTimesUpdate);

        // Step 5: Update momentum for next iteration
        var oneMinusBeta2 = NumOps.Subtract(NumOps.One, _currentBeta2);
        var beta2TimesM = (Vector<T>)Engine.Multiply(_m, _currentBeta2);
        var oneMinusBeta2TimesGrad = (Vector<T>)Engine.Multiply(gradient, oneMinusBeta2);
        _m = (Vector<T>)Engine.Add(beta2TimesM, oneMinusBeta2TimesGrad);

        return currentSolution.WithParameters(newParameters);
    }

    /// <summary>
    /// Updates a vector of parameters using the Lion optimization algorithm.
    /// </summary>
    /// <param name="parameters">The current parameter vector to be updated.</param>
    /// <param name="gradient">The gradient vector corresponding to the parameters.</param>
    /// <returns>The updated parameter vector.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method applies the Lion algorithm to a vector of parameters.
    /// Unlike Adam which considers both the direction and magnitude of gradients, Lion only cares about
    /// the direction (sign). This makes it simpler and often more robust to different scales of gradients.</para>
    /// </remarks>
    public override Vector<T> UpdateParameters(Vector<T> parameters, Vector<T> gradient)
    {
        if (_m == null || _m.Length != parameters.Length)
        {
            _m = new Vector<T>(parameters.Length);
            _t = 0;
        }

        _t++;

        // === Vectorized Lion Update using IEngine (Phase B: US-GPU-015) ===
        var weightDecay = NumOps.FromDouble(_options.WeightDecay);

        // Interpolate: c_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        var oneMinusBeta1 = NumOps.Subtract(NumOps.One, _currentBeta1);
        var beta1TimesM = (Vector<T>)Engine.Multiply(_m, _currentBeta1);
        var oneMinusBeta1TimesGrad = (Vector<T>)Engine.Multiply(gradient, oneMinusBeta1);
        var interpolated = (Vector<T>)Engine.Add(beta1TimesM, oneMinusBeta1TimesGrad);

        // Compute sign
        var signVec = (Vector<T>)Engine.Sign(interpolated);

        // Update with weight decay
        var update = signVec;
        if (!NumOps.Equals(weightDecay, NumOps.Zero))
        {
            var weightDecayTerm = (Vector<T>)Engine.Multiply(parameters, weightDecay);
            update = (Vector<T>)Engine.Add(update, weightDecayTerm);
        }

        // Update parameters
        var lrTimesUpdate = (Vector<T>)Engine.Multiply(update, CurrentLearningRate);
        var updatedParams = (Vector<T>)Engine.Subtract(parameters, lrTimesUpdate);

        // Update momentum: m_t = beta2 * m_{t-1} + (1 - beta2) * g_t
        var oneMinusBeta2 = NumOps.Subtract(NumOps.One, _currentBeta2);
        var beta2TimesM = (Vector<T>)Engine.Multiply(_m, _currentBeta2);
        var oneMinusBeta2TimesGrad = (Vector<T>)Engine.Multiply(gradient, oneMinusBeta2);
        _m = (Vector<T>)Engine.Add(beta2TimesM, oneMinusBeta2TimesGrad);

        return updatedParams;
    }

    /// <summary>
    /// Reverses a Lion gradient update to recover original parameters.
    /// </summary>
    /// <param name="updatedParameters">Parameters after Lion update</param>
    /// <param name="appliedGradients">The gradients that were applied</param>
    /// <returns>Original parameters before the update</returns>
    /// <remarks>
    /// <para>
    /// Lion's reverse update is complex due to its sign-based updates and optional weight decay.
    /// This method must be called immediately after UpdateParameters while the momentum state (_m) is fresh.
    /// It recalculates the sign of the interpolated momentum-gradient and reverses the weight decay effect.
    /// </para>
    /// <para><b>For Beginners:</b> This calculates where parameters were before a Lion update.
    /// Lion uses only the direction (sign) of updates, not their magnitude. To reverse, we need to
    /// remember what direction was used (calculated from momentum and gradients) and also undo
    /// the weight decay that was applied to prevent parameters from growing too large.
    /// </para>
    /// </remarks>
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

        if (_m == null || _m.Length != updatedParameters.Length)
        {
            throw new InvalidOperationException(
                "Lion optimizer state is not initialized. ReverseUpdate must be called after UpdateParameters.");
        }

        // === Vectorized Reverse Lion Update using IEngine (Phase B: US-GPU-015) ===
        var weightDecay = NumOps.FromDouble(_options.WeightDecay);
        var oneMinusBeta2 = NumOps.Subtract(NumOps.One, _currentBeta2);
        var oneMinusBeta1 = NumOps.Subtract(NumOps.One, _currentBeta1);

        // Work backwards from current _m to get the old _m:
        // _m[i] = beta2 * m_old[i] + (1 - beta2) * gradient[i]
        // m_old[i] = (_m[i] - (1 - beta2) * gradient[i]) / beta2
        var oneMinusBeta2Vec = Vector<T>.CreateDefault(appliedGradients.Length, oneMinusBeta2);
        var beta2Vec = Vector<T>.CreateDefault(appliedGradients.Length, _currentBeta2);
        var oneMinusBeta2TimesGrad = (Vector<T>)Engine.Multiply(appliedGradients, oneMinusBeta2Vec);
        var mMinusOneMinusBeta2TimesGrad = (Vector<T>)Engine.Subtract(_m, oneMinusBeta2TimesGrad);
        var mOld = (Vector<T>)Engine.Divide(mMinusOneMinusBeta2TimesGrad, beta2Vec);

        // Recalculate the interpolation: c_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        var beta1Vec = Vector<T>.CreateDefault(appliedGradients.Length, _currentBeta1);
        var oneMinusBeta1Vec = Vector<T>.CreateDefault(appliedGradients.Length, oneMinusBeta1);
        var beta1TimesMOld = (Vector<T>)Engine.Multiply(mOld, beta1Vec);
        var oneMinusBeta1TimesGrad = (Vector<T>)Engine.Multiply(appliedGradients, oneMinusBeta1Vec);
        var interpolated = (Vector<T>)Engine.Add(beta1TimesMOld, oneMinusBeta1TimesGrad);

        // Recalculate the sign using vectorized sign operation
        var signValue = (Vector<T>)Engine.Sign(interpolated);

        // Reverse the update: params_old = (params_new + lr * sign) / (1 - lr * wd)
        var currentLrVec = Vector<T>.CreateDefault(signValue.Length, CurrentLearningRate);
        var lrTimesSign = (Vector<T>)Engine.Multiply(currentLrVec, signValue);
        var numerator = (Vector<T>)Engine.Add(updatedParameters, lrTimesSign);

        Vector<T> original;
        if (!NumOps.Equals(weightDecay, NumOps.Zero))
        {
            var denominator = NumOps.Subtract(NumOps.One, NumOps.Multiply(CurrentLearningRate, weightDecay));
            var denominatorVec = Vector<T>.CreateDefault(numerator.Length, denominator);
            original = (Vector<T>)Engine.Divide(numerator, denominatorVec);
        }
        else
        {
            original = numerator;
        }

        return original;
    }

    /// <summary>
    /// Updates a matrix of parameters using the Lion optimization algorithm.
    /// </summary>
    /// <param name="parameters">The current parameter matrix to be updated.</param>
    /// <param name="gradient">The gradient matrix corresponding to the parameters.</param>
    /// <returns>The updated parameter matrix.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method is similar to UpdateParameters for vectors, but it works on
    /// a 2D grid of parameters instead of a 1D list. This is commonly used for weight matrices in neural networks.
    /// Lion's sign-based updates make it particularly effective for large parameter matrices.</para>
    /// </remarks>
    public override Matrix<T> UpdateParameters(Matrix<T> parameters, Matrix<T> gradient)
    {
        if (_m == null || _m.Length != parameters.Rows * parameters.Columns)
        {
            _m = new Vector<T>(parameters.Rows * parameters.Columns);
            _t = 0;
        }

        _t++;

        // === Vectorized Lion Update using IEngine (Phase B: US-GPU-015) ===
        // Flatten matrices to vectors for vectorized processing
        var paramVector = parameters.ToVector();
        var gradVector = gradient.ToVector();

        var weightDecay = NumOps.FromDouble(_options.WeightDecay);

        // Interpolate: c_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        var oneMinusBeta1 = NumOps.Subtract(NumOps.One, _currentBeta1);
        var beta1TimesM = (Vector<T>)Engine.Multiply(_m, _currentBeta1);
        var oneMinusBeta1TimesGrad = (Vector<T>)Engine.Multiply(gradVector, oneMinusBeta1);
        var interpolated = (Vector<T>)Engine.Add(beta1TimesM, oneMinusBeta1TimesGrad);

        // Compute sign
        var signVec = (Vector<T>)Engine.Sign(interpolated);

        // Update with weight decay
        var update = signVec;
        if (!NumOps.Equals(weightDecay, NumOps.Zero))
        {
            var weightDecayTerm = (Vector<T>)Engine.Multiply(paramVector, weightDecay);
            update = (Vector<T>)Engine.Add(update, weightDecayTerm);
        }

        // Update parameters
        var lrTimesUpdate = (Vector<T>)Engine.Multiply(update, CurrentLearningRate);
        var updatedParams = (Vector<T>)Engine.Subtract(paramVector, lrTimesUpdate);

        // Update momentum: m_t = beta2 * m_{t-1} + (1 - beta2) * g_t
        var oneMinusBeta2 = NumOps.Subtract(NumOps.One, _currentBeta2);
        var beta2TimesM = (Vector<T>)Engine.Multiply(_m, _currentBeta2);
        var oneMinusBeta2TimesGrad = (Vector<T>)Engine.Multiply(gradVector, oneMinusBeta2);
        _m = (Vector<T>)Engine.Add(beta2TimesM, oneMinusBeta2TimesGrad);

        // Reshape back to matrix
        return updatedParams.Reshape(parameters.Rows, parameters.Columns);
    }


    /// <summary>
    /// Resets the optimizer's internal state.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is like resetting the optimizer's memory.
    /// It forgets all past momentum and starts fresh, which can be useful when you want to
    /// reuse the optimizer for a new problem or restart training from scratch.</para>
    /// </remarks>
    public override void Reset()
    {
        _m = Vector<T>.Empty();
        _t = 0;
    }

    /// <summary>
    /// Updates the optimizer's options.
    /// </summary>
    /// <param name="options">The new options to be set.</param>
    /// <exception cref="ArgumentException">Thrown when the provided options are not of type LionOptimizerOptions.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method allows you to change the optimizer's settings during training.
    /// However, Lion is designed to work well with fixed settings, so you typically won't need to change
    /// these mid-training.</para>
    /// </remarks>
    protected override void UpdateOptions(OptimizationAlgorithmOptions<T, TInput, TOutput> options)
    {
        if (options is LionOptimizerOptions<T, TInput, TOutput> lionOptions)
        {
            _options = lionOptions;
            InitializeAdaptiveParameters();
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected LionOptimizerOptions.");
        }
    }

    /// <summary>
    /// Gets the current optimizer options.
    /// </summary>
    /// <returns>The current LionOptimizerOptions.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method lets you check what settings the optimizer is currently using.
    /// It's useful for debugging or logging your training configuration.</para>
    /// </remarks>
    public override OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions()
    {
        return _options;
    }

    /// <summary>
    /// Serializes the optimizer's state into a byte array.
    /// </summary>
    /// <returns>A byte array representing the serialized state of the optimizer.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method saves the optimizer's current state into a compact form.
    /// You can use this to pause training, save your progress, and resume later from exactly where you left off.
    /// Lion's single momentum state makes serialization more efficient than Adam.</para>
    /// </remarks>
    public override byte[] Serialize()
    {
        using (MemoryStream ms = new MemoryStream())
        using (BinaryWriter writer = new BinaryWriter(ms))
        {
            // Serialize base class data
            byte[] baseData = base.Serialize();
            writer.Write(baseData.Length);
            writer.Write(baseData);

            // Serialize LionOptimizerOptions
            string optionsJson = JsonConvert.SerializeObject(_options);
            writer.Write(optionsJson);

            // Serialize Lion-specific data
            writer.Write(_t);
            writer.Write(_m.Length);
            foreach (var value in _m)
            {
                writer.Write(Convert.ToDouble(value));
            }

            return ms.ToArray();
        }
    }

    /// <summary>
    /// Deserializes the optimizer's state from a byte array.
    /// </summary>
    /// <param name="data">The byte array containing the serialized optimizer state.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method rebuilds the optimizer's state from a saved snapshot.
    /// Use this to resume training from a checkpoint, restoring all momentum and configuration exactly
    /// as it was when you saved it.</para>
    /// </remarks>
    public override void Deserialize(byte[] data)
    {
        using (MemoryStream ms = new MemoryStream(data))
        using (BinaryReader reader = new BinaryReader(ms))
        {
            // Deserialize base class data
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

            // Deserialize LionOptimizerOptions
            string optionsJson = reader.ReadString();
            _options = JsonConvert.DeserializeObject<LionOptimizerOptions<T, TInput, TOutput>>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");

            InitializeAdaptiveParameters();

            // Deserialize Lion-specific data
            _t = reader.ReadInt32();
            int mLength = reader.ReadInt32();
            _m = new Vector<T>(mLength);
            for (int i = 0; i < mLength; i++)
            {
                _m[i] = NumOps.FromDouble(reader.ReadDouble());
            }
        }
    }

    /// <summary>
    /// Generates a unique key for caching gradients.
    /// </summary>
    /// <param name="model">The symbolic model.</param>
    /// <param name="X">The input matrix.</param>
    /// <param name="y">The target vector.</param>
    /// <returns>A string key for gradient caching.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method creates a unique identifier for a specific optimization scenario.
    /// It helps the optimizer efficiently store and retrieve previously calculated gradients, speeding up training.</para>
    /// </remarks>
    protected override string GenerateGradientCacheKey(IFullModel<T, TInput, TOutput> model, TInput X, TOutput y)
    {
        var baseKey = base.GenerateGradientCacheKey(model, X, y);
        return $"{baseKey}_Lion_{_options.InitialLearningRate}_{_options.MaxIterations}";
    }
}
