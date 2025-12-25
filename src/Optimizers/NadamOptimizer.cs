using Newtonsoft.Json;

namespace AiDotNet.Optimizers;

/// <summary>
/// Implements the Nesterov-accelerated Adaptive Moment Estimation (Nadam) optimization algorithm.
/// </summary>
/// <remarks>
/// <para>
/// Nadam combines the ideas of Adam (adaptive learning rates) and Nesterov accelerated gradient (NAG).
/// It adapts the learning rates of each parameter and incorporates momentum using Nesterov's method.
/// </para>
/// <para><b>For Beginners:</b>
/// Imagine you're rolling a smart ball down a hill. This ball can adjust its speed for different parts of the hill (adaptive learning rates),
/// and it can look ahead to anticipate slopes (Nesterov's method). This combination helps it find the lowest point more efficiently.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class NadamOptimizer<T, TInput, TOutput> : GradientBasedOptimizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// The options specific to the Nadam optimizer.
    /// </summary>
    private NadamOptimizerOptions<T, TInput, TOutput> _options;

    /// <summary>
    /// The first moment vector (momentum).
    /// </summary>
    private Vector<T>? _m;

    /// <summary>
    /// The second moment vector (adaptive learning rates).
    /// </summary>
    private Vector<T>? _v;

    /// <summary>
    /// The current time step.
    /// </summary>
    private int _t;

    /// <summary>
    /// Stores the pre-update snapshot of first moment vector for accurate reverse updates.
    /// </summary>
    private Vector<T>? _previousM;

    /// <summary>
    /// Stores the pre-update snapshot of second moment vector for accurate reverse updates.
    /// </summary>
    private Vector<T>? _previousV;

    /// <summary>
    /// Stores the pre-update snapshot of the time step for accurate reverse updates.
    /// </summary>
    private int _previousT;

    /// <summary>
    /// Initializes a new instance of the NadamOptimizer class.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This constructor sets up the Nadam optimizer with the provided options and dependencies.
    /// If no options are provided, it uses default settings.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like preparing your smart ball for the hill-rolling experiment. You're setting up its initial properties
    /// and deciding how it will adapt during its journey.
    /// </para>
    /// </remarks>
    /// <param name="model">The model to optimize.</param>
    /// <param name="options">The Nadam-specific optimization options.</param>
    public NadamOptimizer(
        IFullModel<T, TInput, TOutput> model,
        NadamOptimizerOptions<T, TInput, TOutput>? options = null)
        : base(model, options ?? new())
    {
        _options = options ?? new NadamOptimizerOptions<T, TInput, TOutput>();

        InitializeAdaptiveParameters();
    }

    /// <summary>
    /// Initializes the adaptive parameters for the Nadam optimizer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method sets up the initial learning rate and resets the time step counter.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like setting the initial speed of your smart ball and resetting its internal clock before it starts rolling.
    /// </para>
    /// </remarks>
    protected override void InitializeAdaptiveParameters()
    {
        base.InitializeAdaptiveParameters();
        // Learning rate is now set by base class from options.InitialLearningRate
        _t = 0;
    }

    /// <summary>
    /// Performs the optimization process using the Nadam algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method implements the main optimization loop. It iterates through the data, calculating gradients,
    /// updating the momentum and adaptive learning rates, and adjusting the model parameters accordingly.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is the actual process of rolling your smart ball down the hill. In each step, you're calculating which way
    /// the ball should roll (gradient), how fast it's moving (momentum), and how it should adapt its speed (adaptive learning rates).
    /// You keep doing this until the ball finds the lowest point or you've rolled it enough times.
    /// </para>
    /// <para><b>DataLoader Integration:</b> This method uses the DataLoader API for efficient batch processing.
    /// It creates a batcher using <see cref="GradientBasedOptimizerBase{T,TInput,TOutput}.CreateBatcher"/>
    /// and notifies the sampler of epoch starts using
    /// <see cref="GradientBasedOptimizerBase{T,TInput,TOutput}.NotifyEpochStart"/>.
    /// </para>
    /// </remarks>
    /// <param name="inputData">The input data for the optimization process.</param>
    /// <returns>The result of the optimization process.</returns>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        ValidationHelper<T>.ValidateInputData(inputData);

        var currentSolution = InitializeRandomSolution(inputData.XTrain);
        var bestStepData = new OptimizationStepData<T, TInput, TOutput>();
        var previousStepData = new OptimizationStepData<T, TInput, TOutput>();
        var parameters = currentSolution.GetParameters();
        _m = new Vector<T>(parameters.Length);
        _v = new Vector<T>(parameters.Length);

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
    /// Updates the current solution based on the calculated gradient using the Nadam algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method applies the Nadam update rule to adjust the model parameters. It uses both momentum
    /// and adaptive learning rates, incorporating Nesterov's accelerated gradient.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like adjusting the ball's position based on its current speed, the slope it's on, and its ability
    /// to look ahead. It's a complex calculation that helps the ball move more efficiently towards the lowest point.
    /// </para>
    /// </remarks>
    /// <param name="currentSolution">The current model solution.</param>
    /// <param name="gradient">The calculated gradient.</param>
    /// <returns>An updated symbolic model with improved coefficients.</returns>
    protected override IFullModel<T, TInput, TOutput> UpdateSolution(IFullModel<T, TInput, TOutput> currentSolution, Vector<T> gradient)
    {
        var parameters = currentSolution.GetParameters();

        // === Vectorized Nadam Update using IEngine (Phase B: US-GPU-015) ===
        T beta1 = NumOps.FromDouble(_options.Beta1);
        T beta2 = NumOps.FromDouble(_options.Beta2);
        T oneMinusBeta1 = NumOps.FromDouble(1 - _options.Beta1);
        T oneMinusBeta2 = NumOps.FromDouble(1 - _options.Beta2);
        T epsilon = NumOps.FromDouble(_options.Epsilon);
        T biasCorrectionM = NumOps.FromDouble(1 - Math.Pow(_options.Beta1, _t));
        T biasCorrectionV = NumOps.FromDouble(1 - Math.Pow(_options.Beta2, _t));
        T nesterovFactor = NumOps.Divide(oneMinusBeta1, biasCorrectionM);

        // Update biased first moment estimate: m = beta1 * m + (1 - beta1) * gradient
        var beta1TimesM = (Vector<T>)Engine.Multiply(_m!, beta1);
        var oneMinusBeta1TimesGrad = (Vector<T>)Engine.Multiply(gradient, oneMinusBeta1);
        _m = (Vector<T>)Engine.Add(beta1TimesM, oneMinusBeta1TimesGrad);

        // Update biased second raw moment estimate: v = beta2 * v + (1 - beta2) * gradient^2
        var gradSquared = (Vector<T>)Engine.Multiply(gradient, gradient);
        var beta2TimesV = (Vector<T>)Engine.Multiply(_v!, beta2);
        var oneMinusBeta2TimesGradSq = (Vector<T>)Engine.Multiply(gradSquared, oneMinusBeta2);
        _v = (Vector<T>)Engine.Add(beta2TimesV, oneMinusBeta2TimesGradSq);

        // Compute bias-corrected first moment estimate: mHat = m / (1 - beta1^t)
        var mHat = (Vector<T>)Engine.Divide(_m, biasCorrectionM);

        // Compute bias-corrected second raw moment estimate: vHat = v / (1 - beta2^t)
        var vHat = (Vector<T>)Engine.Divide(_v, biasCorrectionV);

        // Compute the Nesterov momentum term: mHatNesterov = beta1 * mHat + nesterovFactor * gradient
        var beta1TimesMHat = (Vector<T>)Engine.Multiply(mHat, beta1);
        var nesterovGrad = (Vector<T>)Engine.Multiply(gradient, nesterovFactor);
        var mHatNesterov = (Vector<T>)Engine.Add(beta1TimesMHat, nesterovGrad);

        // Update parameters: update = (lr * mHatNesterov) / (sqrt(vHat) + epsilon)
        var sqrtVHat = (Vector<T>)Engine.Sqrt(vHat);
        var epsilonVec = new Vector<T>(Enumerable.Repeat(epsilon, sqrtVHat.Length));
        var denominator = (Vector<T>)Engine.Add(sqrtVHat, epsilonVec);
        var lrTimesMHatNesterov = (Vector<T>)Engine.Multiply(mHatNesterov, CurrentLearningRate);
        var update = (Vector<T>)Engine.Divide(lrTimesMHatNesterov, denominator);

        // params = params - update
        var newCoefficients = (Vector<T>)Engine.Subtract(parameters, update);

        return currentSolution.WithParameters(newCoefficients);
    }

    /// <summary>
    /// Updates a vector of parameters using the Nadam optimization algorithm.
    /// </summary>
    /// <param name="parameters">The current parameter vector to be updated.</param>
    /// <param name="gradient">The gradient vector corresponding to the parameters.</param>
    /// <returns>The updated parameter vector.</returns>
    /// <remarks>
    /// <para>
    /// Nadam combines Adam's adaptive learning rates with Nesterov's accelerated gradient, providing
    /// the benefits of both techniques: adaptive per-parameter learning rates and lookahead momentum.
    /// </para>
    /// <para><b>For Beginners:</b> Nadam is like a smart ball that not only adapts its speed for
    /// different parts of the hill (Adam) but also looks ahead to anticipate slopes (Nesterov).
    /// </para>
    /// </remarks>
    public override Vector<T> UpdateParameters(Vector<T> parameters, Vector<T> gradient)
    {
        if (_m == null || _v == null || _m.Length != parameters.Length || _v.Length != parameters.Length)
        {
            _m = new Vector<T>(parameters.Length);
            _v = new Vector<T>(parameters.Length);
            _t = 0;
        }

        // Save previous state BEFORE updating for ReverseUpdate
        _previousM = _m.Clone();
        _previousV = _v.Clone();
        _previousT = _t;

        _t++;

        // === Vectorized Nadam Update using IEngine (Phase B: US-GPU-015) ===
        T beta1 = NumOps.FromDouble(_options.Beta1);
        T beta2 = NumOps.FromDouble(_options.Beta2);
        T oneMinusBeta1 = NumOps.FromDouble(1 - _options.Beta1);
        T oneMinusBeta2 = NumOps.FromDouble(1 - _options.Beta2);
        T epsilon = NumOps.FromDouble(_options.Epsilon);
        T biasCorrectionM = NumOps.FromDouble(1 - Math.Pow(_options.Beta1, _t));
        T biasCorrectionV = NumOps.FromDouble(1 - Math.Pow(_options.Beta2, _t));
        T nesterovFactor = NumOps.Divide(oneMinusBeta1, biasCorrectionM);

        // Update biased first moment estimate: m = beta1 * m + (1 - beta1) * gradient
        var beta1TimesM = (Vector<T>)Engine.Multiply(_m, beta1);
        var oneMinusBeta1TimesGrad = (Vector<T>)Engine.Multiply(gradient, oneMinusBeta1);
        _m = (Vector<T>)Engine.Add(beta1TimesM, oneMinusBeta1TimesGrad);

        // Update biased second raw moment estimate: v = beta2 * v + (1 - beta2) * gradient^2
        var gradSquared = (Vector<T>)Engine.Multiply(gradient, gradient);
        var beta2TimesV = (Vector<T>)Engine.Multiply(_v, beta2);
        var oneMinusBeta2TimesGradSq = (Vector<T>)Engine.Multiply(gradSquared, oneMinusBeta2);
        _v = (Vector<T>)Engine.Add(beta2TimesV, oneMinusBeta2TimesGradSq);

        // Compute bias-corrected first moment estimate: mHat = m / (1 - beta1^t)
        var mHat = (Vector<T>)Engine.Divide(_m, biasCorrectionM);

        // Compute bias-corrected second raw moment estimate: vHat = v / (1 - beta2^t)
        var vHat = (Vector<T>)Engine.Divide(_v, biasCorrectionV);

        // Compute the Nesterov momentum term: mHatNesterov = beta1 * mHat + nesterovFactor * gradient
        var beta1TimesMHat = (Vector<T>)Engine.Multiply(mHat, beta1);
        var nesterovGrad = (Vector<T>)Engine.Multiply(gradient, nesterovFactor);
        var mHatNesterov = (Vector<T>)Engine.Add(beta1TimesMHat, nesterovGrad);

        // Update parameters: update = (lr * mHatNesterov) / (sqrt(vHat) + epsilon)
        var sqrtVHat = (Vector<T>)Engine.Sqrt(vHat);
        var epsilonVec = new Vector<T>(Enumerable.Repeat(epsilon, sqrtVHat.Length));
        var denominator = (Vector<T>)Engine.Add(sqrtVHat, epsilonVec);
        var lrTimesMHatNesterov = (Vector<T>)Engine.Multiply(mHatNesterov, CurrentLearningRate);
        var update = (Vector<T>)Engine.Divide(lrTimesMHatNesterov, denominator);

        // params = params - update
        var updatedParams = (Vector<T>)Engine.Subtract(parameters, update);

        return updatedParams;
    }

    /// <summary>
    /// Reverses a Nadam gradient update to recover original parameters.
    /// </summary>
    /// <param name="updatedParameters">Parameters after Nadam update</param>
    /// <param name="appliedGradients">The gradients that were applied</param>
    /// <returns>Original parameters before the update</returns>
    /// <remarks>
    /// <para>
    /// Nadam's reverse update requires the optimizer's internal state (_m, _v, _t) from the forward pass.
    /// This method must be called immediately after UpdateParameters while the state is fresh.
    /// It recalculates the Nesterov-accelerated adaptive update that was applied.
    /// </para>
    /// <para><b>For Beginners:</b> This calculates where parameters were before a Nadam update.
    /// Nadam combines lookahead (Nesterov) with adaptive learning (Adam), so reversing requires
    /// both the momentum history (_m) and variance history (_v) to reconstruct the lookahead step.
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

        if (_m == null || _v == null || _m.Length != updatedParameters.Length || _v.Length != updatedParameters.Length || _t == 0)
        {
            throw new InvalidOperationException(
                "Nadam optimizer state is not initialized or timestep is zero. ReverseUpdate must be called after UpdateParameters.");
        }

        if (_previousM == null || _previousV == null || _previousM.Length != updatedParameters.Length || _previousV.Length != updatedParameters.Length)
        {
            throw new InvalidOperationException(
                "Nadam optimizer previous state is not available. ReverseUpdate must be called after UpdateParameters.");
        }

        // === Vectorized Reverse Nadam Update (Phase B: US-GPU-015) ===
        T beta1 = NumOps.FromDouble(_options.Beta1);
        T beta2 = NumOps.FromDouble(_options.Beta2);
        T oneMinusBeta1 = NumOps.FromDouble(1 - _options.Beta1);
        T biasCorrection1 = NumOps.FromDouble(1 - Math.Pow(_options.Beta1, _t));
        T biasCorrection2 = NumOps.FromDouble(1 - Math.Pow(_options.Beta2, _t));

        // CRITICAL: Use UPDATED moments (current _m and _v), not previous moments
        // Bias-corrected moments
        var biasCorr1Vec = Vector<T>.CreateDefault(_m.Length, biasCorrection1);
        var biasCorr2Vec = Vector<T>.CreateDefault(_v.Length, biasCorrection2);
        var mHat = (Vector<T>)Engine.Divide(_m, biasCorr1Vec);
        var vHat = (Vector<T>)Engine.Divide(_v, biasCorr2Vec);

        // Recalculate the Nesterov momentum term
        var beta1Vec = Vector<T>.CreateDefault(_m.Length, beta1);
        var beta1_mHat = (Vector<T>)Engine.Multiply(beta1Vec, mHat);
        var gradCoeff = NumOps.Divide(oneMinusBeta1, biasCorrection1);
        var gradCoeffVec = Vector<T>.CreateDefault(appliedGradients.Length, gradCoeff);
        var gradTerm = (Vector<T>)Engine.Multiply(gradCoeffVec, appliedGradients);
        var mHatNesterov = (Vector<T>)Engine.Add(beta1_mHat, gradTerm);

        // Recalculate the update that was applied
        var vHatSqrt = (Vector<T>)Engine.Sqrt(vHat);
        var epsilonVec = Vector<T>.CreateDefault(vHatSqrt.Length, NumOps.FromDouble(_options.Epsilon));
        var denominator = (Vector<T>)Engine.Add(vHatSqrt, epsilonVec);
        var currentLrVec = Vector<T>.CreateDefault(mHatNesterov.Length, CurrentLearningRate);
        var numerator = (Vector<T>)Engine.Multiply(currentLrVec, mHatNesterov);
        var update = (Vector<T>)Engine.Divide(numerator, denominator);

        // Reverse: original = updated + update
        var original = (Vector<T>)Engine.Add(updatedParameters, update);

        // Restore state so the rollback fully reverts the step
        _m = new Vector<T>(_previousM);
        _v = new Vector<T>(_previousV);

        // Restore time step to complete the rollback
        _t = _previousT;

        return original;
    }

    /// <summary>
    /// Updates the adaptive parameters of the optimizer based on the current and previous optimization steps.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method adjusts the learning rate based on the performance of the current step compared to the previous step.
    /// If improvement is seen, the learning rate may be increased, otherwise it may be decreased.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like adjusting how fast your ball rolls based on whether it's getting closer to the bottom of the hill.
    /// If it's improving, you might let it roll a bit faster. If not, you might slow it down to be more careful.
    /// </para>
    /// </remarks>
    /// <param name="currentStepData">Data from the current optimization step.</param>
    /// <param name="previousStepData">Data from the previous optimization step.</param>
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
    /// <remarks>
    /// <para>
    /// This method ensures that only compatible option types are used with this optimizer.
    /// It updates the internal options if the provided options are of the correct type.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like changing the rules of how your smart ball rolls mid-experiment. It makes sure you're only
    /// using rules that work for this specific type of smart ball (Nadam optimization).
    /// </para>
    /// </remarks>
    /// <param name="options">The new options to be applied to the optimizer.</param>
    /// <exception cref="ArgumentException">Thrown when the provided options are not of the correct type.</exception>
    protected override void UpdateOptions(OptimizationAlgorithmOptions<T, TInput, TOutput> options)
    {
        if (options is NadamOptimizerOptions<T, TInput, TOutput> nadamOptions)
        {
            _options = nadamOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected NadamOptimizerOptions.");
        }
    }

    /// <summary>
    /// Gets the current optimization algorithm options.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method returns the current options used by the Nadam optimizer.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like checking your current smart ball rolling rules. It lets you see all the settings and strategies 
    /// you're currently using in your experiment.
    /// </para>
    /// </remarks>
    /// <returns>The current NadamOptimizerOptions object.</returns>
    public override OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions()
    {
        return _options;
    }

    /// <summary>
    /// Serializes the optimizer's state into a byte array.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method converts the current state of the optimizer, including its base class state, options, and time step,
    /// into a byte array. This is useful for saving the optimizer's state or transferring it between systems.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// Think of this as taking a snapshot of your entire smart ball rolling experiment. It captures all the details of your 
    /// current setup, including the ball's position, speed, and all your rules. This snapshot can be used to recreate 
    /// the exact same experiment later or share it with others.
    /// </para>
    /// </remarks>
    /// <returns>A byte array representing the serialized state of the optimizer.</returns>
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
    /// Deserializes a byte array to restore the optimizer's state.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method takes a byte array (previously created by Serialize) and uses it to restore the optimizer's state, 
    /// including its base class state, options, and time step.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like using a detailed blueprint to recreate your smart ball rolling experiment exactly as it was at a certain point. 
    /// It allows you to set up the experiment to match a previous state, with all the same rules and conditions.
    /// </para>
    /// </remarks>
    /// <param name="data">The byte array containing the serialized optimizer state.</param>
    /// <exception cref="InvalidOperationException">Thrown when the optimizer options cannot be deserialized.</exception>
    public override void Deserialize(byte[] data)
    {
        using (MemoryStream ms = new MemoryStream(data))
        using (BinaryReader reader = new BinaryReader(ms))
        {
            int baseDataLength = reader.ReadInt32();
            byte[] baseData = reader.ReadBytes(baseDataLength);
            base.Deserialize(baseData);

            string optionsJson = reader.ReadString();
            _options = JsonConvert.DeserializeObject<NadamOptimizerOptions<T, TInput, TOutput>>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");

            _t = reader.ReadInt32();
        }
    }

    /// <summary>
    /// Generates a unique key for caching gradients.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method creates a unique identifier for caching gradients based on the current model, input data,
    /// and Nadam-specific parameters. This helps in efficiently reusing previously calculated gradients when possible.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like creating a special label for each unique situation your smart ball encounters. It helps the ball
    /// remember and quickly recall how it should move in similar situations, making the whole process more efficient.
    /// </para>
    /// </remarks>
    /// <param name="model">The current symbolic model.</param>
    /// <param name="X">The input feature matrix.</param>
    /// <param name="y">The target vector.</param>
    /// <returns>A string that uniquely identifies the current gradient calculation scenario.</returns>
    protected override string GenerateGradientCacheKey(IFullModel<T, TInput, TOutput> model, TInput X, TOutput y)
    {
        var baseKey = base.GenerateGradientCacheKey(model, X, y);
        return $"{baseKey}_Nadam_{_options.Beta1}_{_options.Beta2}_{_t}";
    }
}
