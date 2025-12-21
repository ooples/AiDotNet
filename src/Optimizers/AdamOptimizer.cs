using Newtonsoft.Json;

namespace AiDotNet.Optimizers;

/// <summary>
/// Implements the Adam (Adaptive Moment Estimation) optimization algorithm for gradient-based optimization.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Adam is an advanced optimization algorithm that combines ideas from RMSprop and Momentum optimization methods.
/// It adapts the learning rates for each parameter individually and is well-suited for problems with noisy or sparse gradients.
/// </para>
/// <para><b>For Beginners:</b> Adam is like a smart personal trainer for your machine learning model.
/// It helps your model learn efficiently by adjusting how it learns based on past experiences.
/// </para>
/// </remarks>
public class AdamOptimizer<T, TInput, TOutput> : GradientBasedOptimizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// The options specific to the Adam optimizer.
    /// </summary>
    private AdamOptimizerOptions<T, TInput, TOutput> _options;

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
    /// The current learning rate.
    /// </summary>
    private T _currentLearningRate;

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
    /// Initializes a new instance of the AdamOptimizer class.
    /// </summary>
    /// <param name="model">The model to optimize.</param>
    /// <param name="options">The options for configuring the Adam optimizer.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This sets up the Adam optimizer with its initial configuration.
    /// You can customize various aspects of how it learns, or use default settings that work well for many problems.
    /// </para>
    /// </remarks>
    public AdamOptimizer(
        IFullModel<T, TInput, TOutput>? model,
        AdamOptimizerOptions<T, TInput, TOutput>? options = null)
        : base(model, options ?? new())
    {
        _m = Vector<T>.Empty();
        _v = Vector<T>.Empty();
        _t = 0;
        _options = options ?? new();
        _currentLearningRate = NumOps.Zero;
        _currentBeta1 = NumOps.Zero;
        _currentBeta2 = NumOps.Zero;

        InitializeAdaptiveParameters();
    }

    /// <summary>
    /// Initializes the adaptive parameters used by the Adam optimizer.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This sets up the initial learning rate and momentum factors.
    /// These values will be adjusted as the optimizer learns more about the problem.
    /// </para>
    /// </remarks>
    protected override void InitializeAdaptiveParameters()
    {
        _currentLearningRate = NumOps.FromDouble(_options.LearningRate);
        _currentBeta1 = NumOps.FromDouble(_options.Beta1);
        _currentBeta2 = NumOps.FromDouble(_options.Beta2);
    }

    /// <summary>
    /// Performs the optimization process using the Adam algorithm.
    /// </summary>
    /// <param name="inputData">The input data for optimization, including training data and targets.</param>
    /// <returns>The result of the optimization process, including the best solution found.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the main learning process. It repeatedly tries to improve
    /// the model's parameters, using the Adam algorithm to decide how to change them.
    /// </para>
    /// </remarks>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        var currentSolution = InitializeRandomSolution(inputData.XTrain);
        var bestStepData = new OptimizationStepData<T, TInput, TOutput>();
        var parameters = currentSolution.GetParameters();
        _m = new Vector<T>(parameters.Length);
        _v = new Vector<T>(parameters.Length);
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
    /// <para><b>For Beginners:</b> This method adjusts how the optimizer learns based on its recent performance.
    /// It can change the learning rate and momentum factors to help the optimizer learn more effectively.
    /// </para>
    /// </remarks>
    protected override void UpdateAdaptiveParameters(OptimizationStepData<T, TInput, TOutput> currentStepData, OptimizationStepData<T, TInput, TOutput> previousStepData)
    {
        base.UpdateAdaptiveParameters(currentStepData, previousStepData);

        // Adam-specific adaptive parameter updates
        if (_options.UseAdaptiveLearningRate)
        {
            _currentLearningRate = MathHelper.Max(NumOps.FromDouble(_options.MinLearningRate),
                MathHelper.Min(NumOps.FromDouble(_options.MaxLearningRate), _currentLearningRate));
        }

        if (_options.UseAdaptiveBetas)
        {
            _currentBeta1 = MathHelper.Max(NumOps.FromDouble(_options.MinBeta1),
                MathHelper.Min(NumOps.FromDouble(_options.MaxBeta1), _currentBeta1));
            _currentBeta2 = MathHelper.Max(NumOps.FromDouble(_options.MinBeta2),
                MathHelper.Min(NumOps.FromDouble(_options.MaxBeta2), _currentBeta2));
        }
    }

    /// <summary>
    /// Updates the current solution using the Adam update rule.
    /// </summary>
    /// <param name="currentSolution">The current solution being optimized.</param>
    /// <param name="gradient">The calculated gradient for the current solution.</param>
    /// <returns>A new solution with updated parameters.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method applies the Adam algorithm to adjust the model's parameters.
    /// It uses the current gradient and past information to decide how to change each parameter.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, TInput, TOutput> UpdateSolution(IFullModel<T, TInput, TOutput> currentSolution, Vector<T> gradient)
    {
        var parameters = currentSolution.GetParameters();

        // === Vectorized Adam Update using IEngine ===
        // Phase B: US-GPU-015 - GPU-accelerated gradient updates

        T oneMinusBeta1 = NumOps.Subtract(NumOps.One, _currentBeta1);
        T oneMinusBeta2 = NumOps.Subtract(NumOps.One, _currentBeta2);
        T biasCorrection1 = NumOps.Subtract(NumOps.One, NumOps.Power(_currentBeta1, NumOps.FromDouble(_t)));
        T biasCorrection2 = NumOps.Subtract(NumOps.One, NumOps.Power(_currentBeta2, NumOps.FromDouble(_t)));
        T epsilon = NumOps.FromDouble(_options.Epsilon);

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

        // Compute update: update = learningRate * mHat / (sqrt(vHat) + epsilon)
        var vHatSqrt = (Vector<T>)Engine.Sqrt(vHat);
        // Create epsilon vector for addition
        var epsilonVec = Vector<T>.CreateDefault(vHatSqrt.Length, epsilon);
        var denominator = (Vector<T>)Engine.Add(vHatSqrt, epsilonVec);
        var updateDiv = (Vector<T>)Engine.Divide(mHat, denominator);
        var update = (Vector<T>)Engine.Multiply(updateDiv, _currentLearningRate);

        // Apply update: parameters = parameters - update
        var updatedParams = (Vector<T>)Engine.Subtract(parameters, update);

        return currentSolution.WithParameters(updatedParams);
    }

    /// <summary>
    /// Updates a vector of parameters using the Adam optimization algorithm.
    /// </summary>
    /// <param name="parameters">The current parameter vector to be updated.</param>
    /// <param name="gradient">The gradient vector corresponding to the parameters.</param>
    /// <returns>The updated parameter vector.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method applies the Adam algorithm to a vector of parameters.
    /// It's like adjusting multiple knobs on a machine all at once, where each knob represents a parameter.
    /// The method decides how much to turn each knob based on past adjustments and the current gradient.
    /// </para>
    /// </remarks>
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

        // Save pre-update state for accurate reverse updates
        if (_previousM == null || _previousV == null)
        {
            _previousM = new Vector<T>(parameters.Length);
            _previousV = new Vector<T>(parameters.Length);
        }

        // Copy _m and _v to _previousM and _previousV (vectorized copy)
        _previousM = new Vector<T>(_m);
        _previousV = new Vector<T>(_v);
        _previousT = _t;

        _t++;

        // === Vectorized Adam Update using IEngine ===
        // Phase B: US-GPU-015 - GPU-accelerated gradient updates

        T beta1 = NumOps.FromDouble(_options.Beta1);
        T beta2 = NumOps.FromDouble(_options.Beta2);
        T oneMinusBeta1 = NumOps.FromDouble(1 - _options.Beta1);
        T oneMinusBeta2 = NumOps.FromDouble(1 - _options.Beta2);
        T epsilon = NumOps.FromDouble(_options.Epsilon);
        T biasCorrection1 = NumOps.FromDouble(1 - Math.Pow(_options.Beta1, _t));
        T biasCorrection2 = NumOps.FromDouble(1 - Math.Pow(_options.Beta2, _t));

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

        // Compute update: update = mHat / (sqrt(vHat) + epsilon)
        var vHatSqrt = (Vector<T>)Engine.Sqrt(vHat);
        // Create epsilon vector for addition
        var epsilonVec = Vector<T>.CreateDefault(vHatSqrt.Length, epsilon);
        var denominator = (Vector<T>)Engine.Add(vHatSqrt, epsilonVec);
        var update = (Vector<T>)Engine.Divide(mHat, denominator);

        // Apply update: parameters = parameters - learningRate * update
        var scaledUpdate = (Vector<T>)Engine.Multiply(update, _currentLearningRate);
        var updatedParameters = (Vector<T>)Engine.Subtract(parameters, scaledUpdate);

        return updatedParameters;
    }


    /// <summary>
    /// Updates a matrix of parameters using the Adam optimization algorithm.
    /// </summary>
    /// <param name="parameters">The current parameter matrix to be updated.</param>
    /// <param name="gradient">The gradient matrix corresponding to the parameters.</param>
    /// <returns>The updated parameter matrix.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method is similar to UpdateVector, but it works on a 2D grid of parameters instead of a 1D list.
    /// It's like adjusting a whole panel of knobs, where each knob is positioned in a grid.
    /// </para>
    /// </remarks>
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

        // === Vectorized Adam Update using IEngine ===
        // Phase B: US-GPU-015 - GPU-accelerated gradient updates
        // Flatten matrices to vectors for vectorized operations

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

        T beta1 = NumOps.FromDouble(_options.Beta1);
        T beta2 = NumOps.FromDouble(_options.Beta2);
        T oneMinusBeta1 = NumOps.FromDouble(1 - _options.Beta1);
        T oneMinusBeta2 = NumOps.FromDouble(1 - _options.Beta2);
        T epsilon = NumOps.FromDouble(_options.Epsilon);
        T biasCorrection1 = NumOps.FromDouble(1 - Math.Pow(_options.Beta1, _t));
        T biasCorrection2 = NumOps.FromDouble(1 - Math.Pow(_options.Beta2, _t));

        // Update biased first moment: m = beta1 * m + (1 - beta1) * gradient
        var mScaled = (Vector<T>)Engine.Multiply(_m, beta1);
        var gradScaled = (Vector<T>)Engine.Multiply(gradVec, oneMinusBeta1);
        _m = (Vector<T>)Engine.Add(mScaled, gradScaled);

        // Update biased second moment: v = beta2 * v + (1 - beta2) * gradient^2
        var gradSquared = (Vector<T>)Engine.Multiply(gradVec, gradVec);
        var vScaled = (Vector<T>)Engine.Multiply(_v, beta2);
        var gradSquaredScaled = (Vector<T>)Engine.Multiply(gradSquared, oneMinusBeta2);
        _v = (Vector<T>)Engine.Add(vScaled, gradSquaredScaled);

        // Compute bias-corrected moments
        var mHat = (Vector<T>)Engine.Divide(_m, biasCorrection1);
        var vHat = (Vector<T>)Engine.Divide(_v, biasCorrection2);

        // Compute update
        var vHatSqrt = (Vector<T>)Engine.Sqrt(vHat);
        var epsilonVec = new Vector<T>(Enumerable.Repeat(epsilon, vHatSqrt.Length));
        var denominator = (Vector<T>)Engine.Add(vHatSqrt, epsilonVec);
        var update = (Vector<T>)Engine.Divide(mHat, denominator);
        var scaledUpdate = (Vector<T>)Engine.Multiply(update, _currentLearningRate);

        // Apply update
        var updatedVec = (Vector<T>)Engine.Subtract(paramVec, scaledUpdate);

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
    /// Reverses an Adam gradient update to recover original parameters.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This override provides accurate reversal for Adam's adaptive update rule:
    /// params_old = params_new + lr * m_hat / (sqrt(v_hat) + epsilon)
    /// </para>
    /// <para>
    /// Uses the current moment estimates (_m, _v, _t) to reconstruct the exact
    /// update that was applied, accounting for bias correction and adaptive learning rates.
    /// </para>
    /// <para><b>For Beginners:</b> This accurately undoes an Adam update by accounting
    /// for all of Adam's special features (momentum, adaptive learning rate, bias correction).
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

        // Ensure previous moment buffers are initialized
        if (_previousM == null || _previousV == null || _previousM.Length != updatedParameters.Length || _previousT == 0)
        {
            // If moments aren't initialized, fall back to SGD-style reversal
            // This shouldn't happen in normal usage but provides a safe fallback
            return base.ReverseUpdate(updatedParameters, appliedGradients);
        }

        // === Vectorized Reverse Adam Update using IEngine (Phase B: US-GPU-015) ===
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

        // Compute the update that was applied
        var vHatSqrt = (Vector<T>)Engine.Sqrt(vHat);
        var epsilonVec = Vector<T>.CreateDefault(vHatSqrt.Length, NumOps.FromDouble(_options.Epsilon));
        var denominator = (Vector<T>)Engine.Add(vHatSqrt, epsilonVec);
        var update = (Vector<T>)Engine.Divide(mHat, denominator);
        var currentLrVec = Vector<T>.CreateDefault(update.Length, _currentLearningRate);
        var scaledUpdate = (Vector<T>)Engine.Multiply(update, currentLrVec);

        // Reverse: params_old = params_new + scaled_update
        return (Vector<T>)Engine.Add(updatedParameters, scaledUpdate);
    }

    /// <summary>
    /// Resets the optimizer's internal state.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is like resetting the optimizer's memory.
    /// It forgets all past adjustments and starts fresh, which can be useful when you want to reuse the optimizer for a new problem.
    /// </para>
    /// </remarks>
    public override void Reset()
    {
        _m = Vector<T>.Empty();
        _v = Vector<T>.Empty();
        _t = 0;
    }

    /// <summary>
    /// Updates the optimizer's options.
    /// </summary>
    /// <param name="options">The new options to be set.</param>
    /// <exception cref="ArgumentException">Thrown when the provided options are not of type AdamOptimizerOptions.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method allows you to change the optimizer's settings mid-way.
    /// It's like adjusting the personal trainer's approach based on new instructions.
    /// </para>
    /// </remarks>
    protected override void UpdateOptions(OptimizationAlgorithmOptions<T, TInput, TOutput> options)
    {
        if (options is AdamOptimizerOptions<T, TInput, TOutput> adamOptions)
        {
            _options = adamOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected AdamOptimizerOptions.");
        }
    }

    /// <summary>
    /// Gets the current optimizer options.
    /// </summary>
    /// <returns>The current AdamOptimizerOptions.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method lets you check what settings the optimizer is currently using.
    /// It's like asking your personal trainer about their current training plan for you.
    /// </para>
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
    /// It's like taking a snapshot of the optimizer's memory and settings, which can be used later to recreate its exact state.
    /// </para>
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

            // Serialize AdamOptimizerOptions
            string optionsJson = JsonConvert.SerializeObject(_options);
            writer.Write(optionsJson);

            // Serialize Adam-specific data
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

            return ms.ToArray();
        }
    }

    /// <summary>
    /// Deserializes the optimizer's state from a byte array.
    /// </summary>
    /// <param name="data">The byte array containing the serialized optimizer state.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method rebuilds the optimizer's state from a saved snapshot.
    /// It's like restoring the optimizer's memory and settings from a backup, allowing you to continue from where you left off.
    /// </para>
    /// </remarks>
    public override void Deserialize(byte[] data)
    {
        using (MemoryStream ms = new MemoryStream(data))
        using (BinaryReader reader = new BinaryReader(ms))
        {
            // Deserialize base class data
            int baseDataLength = reader.ReadInt32();
            byte[] baseData = reader.ReadBytes(baseDataLength);
            base.Deserialize(baseData);

            // Deserialize AdamOptimizerOptions
            string optionsJson = reader.ReadString();
            _options = JsonConvert.DeserializeObject<AdamOptimizerOptions<T, TInput, TOutput>>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");

            // Deserialize Adam-specific data
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

            // Initialize adaptive parameters from deserialized options
            InitializeAdaptiveParameters();
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
    /// It's like creating a label for a particular training session, which helps in efficiently storing and retrieving calculated gradients.
    /// </para>
    /// </remarks>
    protected override string GenerateGradientCacheKey(IFullModel<T, TInput, TOutput> model, TInput X, TOutput y)
    {
        var baseKey = base.GenerateGradientCacheKey(model, X, y);
        return $"{baseKey}_Adam_{_options.LearningRate}_{_options.MaxIterations}";
    }
}
