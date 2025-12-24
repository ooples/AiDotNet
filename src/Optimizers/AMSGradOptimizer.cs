using Newtonsoft.Json;

namespace AiDotNet.Optimizers;

/// <summary>
/// Implements the AMSGrad optimization algorithm, an improved version of Adam optimizer.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// AMSGrad is an adaptive learning rate optimization algorithm that addresses some of the convergence issues in Adam.
/// It maintains the maximum of past squared gradients to ensure non-decreasing step sizes.
/// </para>
/// <para><b>For Beginners:</b> AMSGrad is like a smart assistant that helps adjust the learning process.
/// It remembers past information to make better decisions about how quickly to learn in different parts of the problem.
/// </para>
/// </remarks>
public class AMSGradOptimizer<T, TInput, TOutput> : GradientBasedOptimizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// The options specific to the AMSGrad optimizer.
    /// </summary>
    private AMSGradOptimizerOptions<T, TInput, TOutput> _options;

    /// <summary>
    /// The first moment vector (moving average of gradients).
    /// </summary>
    private Vector<T>? _m;

    /// <summary>
    /// The second moment vector (moving average of squared gradients).
    /// </summary>
    private Vector<T>? _v;

    /// <summary>
    /// The maximum of past second moments.
    /// </summary>
    private Vector<T>? _vHat;

    /// <summary>
    /// The current time step.
    /// </summary>
    private int _t;

    /// <summary>
    /// Initializes a new instance of the AMSGradOptimizer class.
    /// </summary>
    /// <param name="options">The options for configuring the AMSGrad optimizer.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This sets up the AMSGrad optimizer with its initial configuration.
    /// You can customize various aspects of how it learns, or use default settings.
    /// </para>
    /// </remarks>
    public AMSGradOptimizer(
        IFullModel<T, TInput, TOutput> model,
        AMSGradOptimizerOptions<T, TInput, TOutput>? options = null,
        IEngine? engine = null)
        : base(model, options ?? new())
    {
        _options = options ?? new AMSGradOptimizerOptions<T, TInput, TOutput>();

        InitializeAdaptiveParameters();
    }

    /// <summary>
    /// Initializes the adaptive parameters used by the AMSGrad optimizer.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This resets the learning rate and time step to their starting values,
    /// preparing the optimizer for a new optimization run.
    /// </para>
    /// </remarks>
    protected override void InitializeAdaptiveParameters()
    {
        base.InitializeAdaptiveParameters();

        CurrentLearningRate = NumOps.FromDouble(_options.LearningRate);
        _t = 0;
    }

    /// <summary>
    /// Performs the optimization process using the AMSGrad algorithm.
    /// </summary>
    /// <param name="inputData">The input data for optimization, including training data and targets.</param>
    /// <returns>The result of the optimization process, including the best solution found.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the main optimization process. It repeatedly updates the solution
    /// using the AMSGrad steps until it reaches the best possible solution or hits a stopping condition.
    /// </para>
    /// </remarks>
    /// <remarks>
    /// <para><b>DataLoader Integration:</b> This method uses the DataLoader API for efficient batch processing.
    /// It creates a batcher using <see cref="GradientBasedOptimizerBase{T,TInput,TOutput}.CreateBatcher"/>
    /// and notifies the sampler of epoch starts using
    /// <see cref="GradientBasedOptimizerBase{T,TInput,TOutput}.NotifyEpochStart"/>.
    /// </para>
    /// </remarks>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        ValidationHelper<T>.ValidateInputData(inputData);

        var currentSolution = InitializeRandomSolution(inputData.XTrain);
        var bestStepData = new OptimizationStepData<T, TInput, TOutput>();
        var previousStepData = new OptimizationStepData<T, TInput, TOutput>();
        var parameters = currentSolution.GetParameters();
        _m = new Vector<T>(parameters.Length);
        _v = new Vector<T>(parameters.Length);
        _vHat = new Vector<T>(parameters.Length);
        InitializeAdaptiveParameters();

        for (int epoch = 0; epoch < _options.MaxIterations; epoch++)
        {
            NotifyEpochStart(epoch);
            var batcher = CreateBatcher(inputData, _options.BatchSize);

            foreach (var (xBatch, yBatch, batchIndices) in batcher.GetBatches())
            {
                _t++;
                var gradient = CalculateGradient(currentSolution, xBatch, yBatch);
                var newSolution = UpdateSolution(currentSolution, gradient);
                currentSolution = newSolution;
            }

            var currentStepData = EvaluateSolution(currentSolution, inputData);
            UpdateBestSolution(currentStepData, ref bestStepData);

            UpdateAdaptiveParameters(currentStepData, previousStepData);

            if (UpdateIterationHistoryAndCheckEarlyStopping(epoch, bestStepData))
            {
                return CreateOptimizationResult(bestStepData, inputData);
            }

            if (NumOps.LessThan(NumOps.Abs(NumOps.Subtract(bestStepData.FitnessScore, currentStepData.FitnessScore)), NumOps.FromDouble(_options.Tolerance)))
            {
                return CreateOptimizationResult(bestStepData, inputData);
            }

            previousStepData = currentStepData;
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    /// <summary>
    /// Updates the current solution using the AMSGrad update rule.
    /// </summary>
    /// <param name="currentSolution">The current solution being optimized.</param>
    /// <param name="gradient">The gradient of the current solution.</param>
    /// <returns>A new solution with updated coefficients.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method applies the AMSGrad formula to update each parameter of the solution.
    /// It uses the current and past gradients to determine how much to change each parameter.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, TInput, TOutput> UpdateSolution(IFullModel<T, TInput, TOutput> currentSolution, Vector<T> gradient)
    {
        var parameters = currentSolution.GetParameters();

        // Use shared UpdateParameters method to eliminate duplication
        var newCoefficients = UpdateParameters(parameters, gradient);

        return currentSolution.WithParameters(newCoefficients);
    }

    /// <summary>
    /// Updates a vector of parameters using the AMSGrad optimization algorithm.
    /// </summary>
    /// <param name="parameters">The current parameter vector to be updated.</param>
    /// <param name="gradient">The gradient vector corresponding to the parameters.</param>
    /// <returns>The updated parameter vector.</returns>
    /// <remarks>
    /// <para>
    /// AMSGrad is a variant of Adam that uses the maximum of past second moments to ensure convergence.
    /// This prevents the learning rate from becoming too large and helps with non-convex optimization.
    /// </para>
    /// <para><b>For Beginners:</b> AMSGrad is like Adam but keeps track of the largest variance
    /// it has seen so far, preventing the optimizer from taking overly aggressive steps.
    /// </para>
    /// </remarks>
    public override Vector<T> UpdateParameters(Vector<T> parameters, Vector<T> gradient)
    {
        if (_m == null || _v == null || _vHat == null || _m.Length != parameters.Length)
        {
            _m = new Vector<T>(parameters.Length);
            _v = new Vector<T>(parameters.Length);
            _vHat = new Vector<T>(parameters.Length);
            _t = 0;
        }

        _t++;

        // === Vectorized AMSGrad Update using IEngine (Phase B: US-GPU-015) ===
        T beta1 = NumOps.FromDouble(_options.Beta1);
        T beta2 = NumOps.FromDouble(_options.Beta2);
        T oneMinusBeta1 = NumOps.FromDouble(1 - _options.Beta1);
        T oneMinusBeta2 = NumOps.FromDouble(1 - _options.Beta2);
        T epsilon = NumOps.FromDouble(_options.Epsilon);
        T biasCorrectionFactor = NumOps.FromDouble(1 - Math.Pow(_options.Beta1, _t));

        // Update biased first moment estimate: m = beta1 * m + (1 - beta1) * gradient
        var beta1TimesM = (Vector<T>)Engine.Multiply(_m, beta1);
        var oneMinusBeta1TimesGrad = (Vector<T>)Engine.Multiply(gradient, oneMinusBeta1);
        _m = (Vector<T>)Engine.Add(beta1TimesM, oneMinusBeta1TimesGrad);

        // Update biased second raw moment estimate: v = beta2 * v + (1 - beta2) * gradient^2
        var gradSquared = (Vector<T>)Engine.Multiply(gradient, gradient);
        var beta2TimesV = (Vector<T>)Engine.Multiply(_v, beta2);
        var oneMinusBeta2TimesGradSq = (Vector<T>)Engine.Multiply(gradSquared, oneMinusBeta2);
        _v = (Vector<T>)Engine.Add(beta2TimesV, oneMinusBeta2TimesGradSq);

        // Update maximum of second raw moment estimate: vHat = max(vHat, v)
        _vHat = (Vector<T>)Engine.Max(_vHat, _v);

        // Compute bias-corrected first moment estimate: mHat = m / (1 - beta1^t)
        var mHat = (Vector<T>)Engine.Divide(_m, biasCorrectionFactor);

        // Compute update: update = (lr * mHat) / (sqrt(vHat) + epsilon)
        var sqrtVHat = (Vector<T>)Engine.Sqrt(_vHat);
        var epsilonVec = new Vector<T>(Enumerable.Repeat(epsilon, sqrtVHat.Length));
        var denominator = (Vector<T>)Engine.Add(sqrtVHat, epsilonVec);
        var lrTimesMHat = (Vector<T>)Engine.Multiply(mHat, CurrentLearningRate);
        var update = (Vector<T>)Engine.Divide(lrTimesMHat, denominator);

        // Update parameters: params = params - update
        var updatedParams = (Vector<T>)Engine.Subtract(parameters, update);

        return updatedParams;
    }

    /// <summary>
    /// Reverses an AMSGrad gradient update to recover original parameters.
    /// </summary>
    /// <param name="updatedParameters">Parameters after AMSGrad update</param>
    /// <param name="appliedGradients">The gradients that were applied</param>
    /// <returns>Original parameters before the update</returns>
    /// <remarks>
    /// <para>
    /// AMSGrad's reverse update requires the optimizer's internal state (_m, _v, _vHat, _t) from the forward pass.
    /// This method must be called immediately after UpdateParameters while the state is fresh.
    /// It uses the maximum of past second moments to recalculate the update.
    /// </para>
    /// <para><b>For Beginners:</b> This calculates where parameters were before an AMSGrad update.
    /// AMSGrad remembers the largest variance seen for each parameter, which prevents taking too-large steps.
    /// To reverse the update, we need this maximum variance history (_vHat) along with momentum (_m).
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

        if (_m == null || _v == null || _vHat == null || _m.Length != updatedParameters.Length)
        {
            throw new InvalidOperationException(
                "AMSGrad optimizer state is not initialized. ReverseUpdate must be called after UpdateParameters.");
        }

        // === Vectorized Reverse AMSGrad Update using IEngine (Phase B: US-GPU-015) ===
        // Recalculate bias-corrected first moment: mHat = m / (1 - beta1^t)
        T biasCorrection1 = NumOps.FromDouble(1 - Math.Pow(_options.Beta1, _t));
        var biasCorrection1Vec = Vector<T>.CreateDefault(_m.Length, biasCorrection1);
        var mHat = (Vector<T>)Engine.Divide(_m, biasCorrection1Vec);

        // Recalculate the update: update = (lr * mHat) / (sqrt(vHat) + epsilon)
        var currentLrVec = Vector<T>.CreateDefault(_m.Length, CurrentLearningRate);
        var lrTimesMHat = (Vector<T>)Engine.Multiply(currentLrVec, mHat);

        var vHatSqrt = (Vector<T>)Engine.Sqrt(_vHat);
        var epsilonVec = Vector<T>.CreateDefault(_vHat.Length, NumOps.FromDouble(_options.Epsilon));
        var denominator = (Vector<T>)Engine.Add(vHatSqrt, epsilonVec);

        var update = (Vector<T>)Engine.Divide(lrTimesMHat, denominator);

        // Reverse: original = updated + update
        return (Vector<T>)Engine.Add(updatedParameters, update);
    }

    /// <summary>
    /// Updates the adaptive parameters of the optimizer based on the current and previous optimization steps.
    /// </summary>
    /// <param name="currentStepData">Data from the current optimization step.</param>
    /// <param name="previousStepData">Data from the previous optimization step.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method adjusts the learning rate based on how well the optimization is progressing.
    /// If the solution is improving, it might increase the learning rate to learn faster.
    /// If not, it might decrease the rate to be more careful.
    /// </para>
    /// </remarks>
    protected override void UpdateAdaptiveParameters(OptimizationStepData<T, TInput, TOutput> currentStepData, OptimizationStepData<T, TInput, TOutput> previousStepData)
    {
        base.UpdateAdaptiveParameters(currentStepData, previousStepData);

        if (_options.UseAdaptiveLearningRate)
        {
            if (NumOps.GreaterThan(currentStepData.FitnessScore, previousStepData.FitnessScore))
            {
                CurrentLearningRate = NumOps.Multiply(CurrentLearningRate, NumOps.FromDouble(_options.LearningRateIncreaseFactor));
            }
            else
            {
                CurrentLearningRate = NumOps.Multiply(CurrentLearningRate, NumOps.FromDouble(_options.LearningRateDecreaseFactor));
            }

            CurrentLearningRate = MathHelper.Clamp(CurrentLearningRate,
                NumOps.FromDouble(_options.MinLearningRate),
                NumOps.FromDouble(_options.MaxLearningRate));
        }
    }

    /// <summary>
    /// Updates the optimizer's options with new settings.
    /// </summary>
    /// <param name="options">The new options to be applied to the optimizer.</param>
    /// <exception cref="ArgumentException">Thrown when the provided options are not of the correct type.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method allows you to change the settings of the AMSGrad optimizer while it's running.
    /// It's like adjusting the controls on a machine that's already operating. If you provide the wrong type of settings,
    /// it will stop and let you know there's an error.
    /// </para>
    /// </remarks>
    protected override void UpdateOptions(OptimizationAlgorithmOptions<T, TInput, TOutput> options)
    {
        if (options is AMSGradOptimizerOptions<T, TInput, TOutput> amsGradOptions)
        {
            _options = amsGradOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected AMSGradOptimizerOptions.");
        }
    }

    /// <summary>
    /// Retrieves the current options of the optimizer.
    /// </summary>
    /// <returns>The current optimization algorithm options.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method lets you check what settings the AMSGrad optimizer is currently using.
    /// It's like looking at the current settings on a machine.
    /// </para>
    /// </remarks>
    public override OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions()
    {
        return _options;
    }

    /// <summary>
    /// Converts the current state of the optimizer into a byte array for storage or transmission.
    /// </summary>
    /// <returns>A byte array representing the serialized state of the optimizer.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method saves all the important information about the AMSGrad optimizer's current state.
    /// It's like taking a snapshot of the optimizer that can be used to recreate its exact state later.
    /// This is useful for saving progress or sharing the optimizer's state with others.
    /// </para>
    /// </remarks>
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

            return ms.ToArray();
        }
    }

    /// <summary>
    /// Restores the optimizer's state from a byte array previously created by the Serialize method.
    /// </summary>
    /// <param name="data">The byte array containing the serialized optimizer state.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method rebuilds the AMSGrad optimizer's state from a saved snapshot.
    /// It's like restoring a machine to a previous configuration using a backup.
    /// This allows you to continue optimization from where you left off or use a shared optimizer state.
    /// </para>
    /// </remarks>
    public override void Deserialize(byte[] data)
    {
        using (MemoryStream ms = new MemoryStream(data))
        using (BinaryReader reader = new BinaryReader(ms))
        {
            int baseDataLength = reader.ReadInt32();
            byte[] baseData = reader.ReadBytes(baseDataLength);
            base.Deserialize(baseData);

            string optionsJson = reader.ReadString();
            _options = JsonConvert.DeserializeObject<AMSGradOptimizerOptions<T, TInput, TOutput>>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");

            _t = reader.ReadInt32();
        }
    }

    /// <summary>
    /// Generates a unique key for caching gradients based on the current state of the optimizer and input data.
    /// </summary>
    /// <param name="model">The symbolic model being optimized.</param>
    /// <param name="X">The input matrix.</param>
    /// <param name="y">The target vector.</param>
    /// <returns>A string that uniquely identifies the current optimization state for gradient caching.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method creates a unique label for the current state of the AMSGrad optimization.
    /// It's used to efficiently store and retrieve calculated gradients, which helps speed up the optimization process.
    /// The key includes specific AMSGrad parameters to ensure it's unique to this optimizer's current state.
    /// </para>
    /// </remarks>
    protected override string GenerateGradientCacheKey(IFullModel<T, TInput, TOutput> model, TInput X, TOutput y)
    {
        var baseKey = base.GenerateGradientCacheKey(model, X, y);
        return $"{baseKey}_AMSGrad_{_options.Beta1}_{_options.Beta2}_{_t}";
    }
}
