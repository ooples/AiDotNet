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
    /// Initializes a new instance of the AdamOptimizer class.
    /// </summary>
    /// <param name="options">The options for configuring the Adam optimizer.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This sets up the Adam optimizer with its initial configuration.
    /// You can customize various aspects of how it learns, or use default settings that work well for many problems.
    /// </para>
    /// </remarks>
    public AdamOptimizer(
        AdamOptimizerOptions<T, TInput, TOutput>? options = null)
        : base(options ?? new())
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
    private IFullModel<T, TInput, TOutput> UpdateSolution(IFullModel<T, TInput, TOutput> currentSolution, Vector<T> gradient)
    {
        var parameters = currentSolution.GetParameters();
        for (int i = 0; i < gradient.Length; i++)
        {
            _m[i] = NumOps.Add(NumOps.Multiply(_currentBeta1, _m[i]), NumOps.Multiply(NumOps.Subtract(NumOps.One, _currentBeta1), gradient[i]));
            _v[i] = NumOps.Add(NumOps.Multiply(_currentBeta2, _v[i]), NumOps.Multiply(NumOps.Subtract(NumOps.One, _currentBeta2), NumOps.Multiply(gradient[i], gradient[i])));

            var mHat = NumOps.Divide(_m[i], NumOps.Subtract(NumOps.One, NumOps.Power(_currentBeta1, NumOps.FromDouble(_t))));
            var vHat = NumOps.Divide(_v[i], NumOps.Subtract(NumOps.One, NumOps.Power(_currentBeta2, NumOps.FromDouble(_t))));

            var update = NumOps.Divide(NumOps.Multiply(_currentLearningRate, mHat), NumOps.Add(NumOps.Sqrt(vHat), NumOps.FromDouble(_options.Epsilon)));

            parameters[i] = NumOps.Subtract(parameters[i], update);
        }

        return currentSolution;
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
            _t = 0;
        }

        _t++;

        for (int i = 0; i < parameters.Length; i++)
        {
            _m[i] = NumOps.Add(
                NumOps.Multiply(_m[i], NumOps.FromDouble(_options.Beta1)),
                NumOps.Multiply(gradient[i], NumOps.FromDouble(1 - _options.Beta1))
            );

            _v[i] = NumOps.Add(
                NumOps.Multiply(_v[i], NumOps.FromDouble(_options.Beta2)),
                NumOps.Multiply(NumOps.Multiply(gradient[i], gradient[i]), NumOps.FromDouble(1 - _options.Beta2))
            );

            T mHat = NumOps.Divide(_m[i], NumOps.FromDouble(1 - Math.Pow(_options.Beta1, _t)));
            T vHat = NumOps.Divide(_v[i], NumOps.FromDouble(1 - Math.Pow(_options.Beta2, _t)));

            T update = NumOps.Divide(
                mHat,
                NumOps.Add(NumOps.Sqrt(vHat), NumOps.FromDouble(_options.Epsilon))
            );

            parameters[i] = NumOps.Subtract(
                parameters[i],
                NumOps.Multiply(update, NumOps.FromDouble(_options.LearningRate))
            );
        }

        return parameters;
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
        if (_m == null || _v == null || _m.Length != parameters.Rows * parameters.Columns)
        {
            _m = new Vector<T>(parameters.Rows * parameters.Columns);
            _v = new Vector<T>(parameters.Rows * parameters.Columns);
            _t = 0;
        }

        _t++;

        var updatedMatrix = new Matrix<T>(parameters.Rows, parameters.Columns);
        int index = 0;

        for (int i = 0; i < parameters.Rows; i++)
        {
            for (int j = 0; j < parameters.Columns; j++)
            {
                T g = gradient[i, j];

                _m[index] = NumOps.Add(
                    NumOps.Multiply(_m[index], NumOps.FromDouble(_options.Beta1)),
                    NumOps.Multiply(g, NumOps.FromDouble(1 - _options.Beta1))
                );

                _v[index] = NumOps.Add(
                    NumOps.Multiply(_v[index], NumOps.FromDouble(_options.Beta2)),
                    NumOps.Multiply(NumOps.Multiply(g, g), NumOps.FromDouble(1 - _options.Beta2))
                );

                T mHat = NumOps.Divide(_m[index], NumOps.FromDouble(1 - Math.Pow(_options.Beta1, _t)));
                T vHat = NumOps.Divide(_v[index], NumOps.FromDouble(1 - Math.Pow(_options.Beta2, _t)));

                T update = NumOps.Divide(
                    mHat,
                    NumOps.Add(NumOps.Sqrt(vHat), NumOps.FromDouble(_options.Epsilon))
                );

                updatedMatrix[i, j] = NumOps.Subtract(
                    parameters[i, j],
                    NumOps.Multiply(update, NumOps.FromDouble(_options.LearningRate))
                );

                index++;
            }
        }

        return updatedMatrix;
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