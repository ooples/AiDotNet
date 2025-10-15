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
    private AdamOptimizerOptions<T, TInput, TOutput> _options = default!;

    /// <summary>
    /// The first moment vector (moving average of gradients).
    /// </summary>
    private Vector<T> _m = default!;

    /// <summary>
    /// The second moment vector (moving average of squared gradients).
    /// </summary>
    private Vector<T> _v = default!;

    /// <summary>
    /// The current time step (iteration count).
    /// </summary>
    private int _t;

    /// <summary>
    /// The current learning rate.
    /// </summary>
    private T _currentLearningRate = default!;

    /// <summary>
    /// The current value of beta1 (exponential decay rate for first moment estimates).
    /// </summary>
    private T _currentBeta1 = default!;

    /// <summary>
    /// The current value of beta2 (exponential decay rate for second moment estimates).
    /// </summary>
    private T _currentBeta2 = default!;

    /// <summary>
    /// Initializes a new instance of the AdamOptimizer class.
    /// </summary>
    /// <param name="model">The model to be optimized.</param>
    /// <param name="options">The options for configuring the Adam optimizer.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This sets up the Adam optimizer with its initial configuration.
    /// You provide:
    /// - model: The specific model you want to improve (like a recipe you want to perfect)
    /// - options: How the optimizer should learn (like instructions for improving the recipe)
    /// 
    /// You can customize various aspects of how it learns, or use default settings that work well for many problems.
    /// </para>
    /// </remarks>
    public AdamOptimizer(
        IFullModel<T, TInput, TOutput> model,
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
    /// <para><b>For Beginners:</b> This is the main learning process. It repeatedly creates and evaluates
    /// new versions of the model based on the optimization mode (feature selection, parameter adjustment, or both).
    /// Each iteration, it keeps track of the best version found so far and adjusts its approach based on how well it's doing.
    /// </para>
    /// </remarks>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        ValidationHelper<T>.ValidateInputData(inputData);

        var bestStepData = new OptimizationStepData<T, TInput, TOutput>
        {
            Solution = Model.DeepCopy(),
            FitnessScore = FitnessCalculator.IsHigherScoreBetter ? NumOps.MinValue : NumOps.MaxValue
        };
        var previousStepData = new OptimizationStepData<T, TInput, TOutput>();

        _t = 0;

        InitializeAdaptiveParameters();

        for (int iteration = 0; iteration < _options.MaxIterations; iteration++)
        {
            _t++;
            var currentSolution = CreateSolution(inputData.XTrain);
            var currentStepData = EvaluateSolution(currentSolution, inputData);

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

            previousStepData = currentStepData;
        }

        var parameters = bestStepData.Solution.GetParameters();
        _m = new Vector<T>(parameters.Length);
        _v = new Vector<T>(parameters.Length);

        return CreateOptimizationResult(bestStepData, inputData);
    }

    /// <summary>
    /// Updates the adaptive parameters of the optimizer based on the current and previous optimization steps.
    /// </summary>
    /// <param name="currentStepData">Data from the current optimization step.</param>
    /// <param name="previousStepData">Data from the previous optimization step.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method adjusts how the optimizer learns based on its recent performance.
    /// It can change the learning rate, momentum factors, and other parameters to help the optimizer learn more effectively.
    /// </para>
    /// </remarks>
    protected override void UpdateAdaptiveParameters(OptimizationStepData<T, TInput, TOutput> currentStepData, OptimizationStepData<T, TInput, TOutput> previousStepData)
    {
        // Call the base implementation to update common parameters
        base.UpdateAdaptiveParameters(currentStepData, previousStepData);

        // Skip if previous step data is null (first iteration)
        if (previousStepData.Solution == null)
            return;

        bool isImproving = FitnessCalculator.IsBetterFitness(currentStepData.FitnessScore, previousStepData.FitnessScore);

        // Adaptive feature selection parameters
        if ((_options.OptimizationMode == OptimizationMode.FeatureSelectionOnly ||
             _options.OptimizationMode == OptimizationMode.Both))
        {
            UpdateFeatureSelectionParameters(isImproving);
        }

        // Adaptive parameter adjustment settings
        if ((_options.OptimizationMode == OptimizationMode.ParametersOnly ||
             _options.OptimizationMode == OptimizationMode.Both))
        {
            UpdateParameterAdjustmentSettings(isImproving);
        }

        // Adam-specific adaptive parameter updates
        if (_options.UseAdaptiveLearningRate)
        {
            UpdateLearningRate(isImproving);
        }

        if (_options.UseAdaptiveBetas)
        {
            UpdateBetaParameters(isImproving);
        }
    }

    /// <summary>
    /// Updates the learning rate based on whether the solution is improving.
    /// </summary>
    /// <param name="isImproving">Indicates whether the solution is improving.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method adjusts how big the steps are that the optimizer takes.
    /// If the solution is improving, it might take slightly bigger steps.
    /// If the solution is not improving, it might take smaller, more careful steps.
    /// </para>
    /// </remarks>
    private void UpdateLearningRate(bool isImproving)
    {
        if (isImproving)
        {
            _currentLearningRate = NumOps.Multiply(_currentLearningRate, NumOps.FromDouble(_options.LearningRateIncreaseFactor));
        }
        else
        {
            _currentLearningRate = NumOps.Multiply(_currentLearningRate, NumOps.FromDouble(_options.LearningRateDecreaseFactor));
        }

        _currentLearningRate = MathHelper.Clamp(_currentLearningRate,
            NumOps.FromDouble(_options.MinLearningRate),
            NumOps.FromDouble(_options.MaxLearningRate));
    }

    /// <summary>
    /// Updates the beta parameters based on whether the solution is improving.
    /// </summary>
    /// <param name="isImproving">Indicates whether the solution is improving.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method adjusts how much the optimizer relies on past information.
    /// Beta1 controls how much it remembers about past gradients, while Beta2 controls how much it remembers about past squared gradients.
    /// These values help the optimizer adapt to different types of landscapes in the solution space.
    /// </para>
    /// </remarks>
    private void UpdateBetaParameters(bool isImproving)
    {
        if (isImproving)
        {
            _currentBeta1 = NumOps.Add(_currentBeta1, NumOps.FromDouble(_options.Beta1IncreaseFactor));
            _currentBeta2 = NumOps.Add(_currentBeta2, NumOps.FromDouble(_options.Beta2IncreaseFactor));
        }
        else
        {
            _currentBeta1 = NumOps.Subtract(_currentBeta1, NumOps.FromDouble(_options.Beta1DecreaseFactor));
            _currentBeta2 = NumOps.Subtract(_currentBeta2, NumOps.FromDouble(_options.Beta2DecreaseFactor));
        }

        _currentBeta1 = MathHelper.Clamp(_currentBeta1,
            NumOps.FromDouble(_options.MinBeta1),
            NumOps.FromDouble(_options.MaxBeta1));

        _currentBeta2 = MathHelper.Clamp(_currentBeta2,
            NumOps.FromDouble(_options.MinBeta2),
            NumOps.FromDouble(_options.MaxBeta2));
    }

    /// <summary>
    /// Updates the feature selection parameters based on whether the solution is improving.
    /// </summary>
    /// <param name="isImproving">Indicates whether the solution is improving.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method adjusts how the optimizer selects ingredients for the recipe.
    /// 
    /// If the solution is improving:
    /// - It might expand the range of ingredients it considers
    /// - It might be more likely to try changing ingredients in future steps
    /// 
    /// If the solution is not improving:
    /// - It might narrow its focus to a smaller set of ingredients
    /// - It might be less aggressive about changing ingredients
    /// </para>
    /// </remarks>
    private void UpdateFeatureSelectionParameters(bool isImproving)
    {
        if (isImproving)
        {
            // If improving, gradually expand the range of features to consider
            _options.MinimumFeatures = Math.Max(1, _options.MinimumFeatures - 1);
            _options.MaximumFeatures = Math.Min(_options.MaximumFeatures + 1, _options.AbsoluteMaximumFeatures);

            // Slightly increase the probability of feature selection for future iterations
            _options.FeatureSelectionProbability *= 1.02;
        }
        else
        {
            // If not improving, narrow the range to focus the search
            _options.MinimumFeatures = Math.Min(_options.MinimumFeatures + 1, _options.AbsoluteMaximumFeatures - 1);
            _options.MaximumFeatures = Math.Max(_options.MaximumFeatures - 1, _options.MinimumFeatures + 1);

            // Slightly decrease the probability of feature selection for future iterations
            _options.FeatureSelectionProbability *= 0.98;
        }

        // Ensure probabilities stay within bounds
        _options.FeatureSelectionProbability = MathHelper.Clamp(
            _options.FeatureSelectionProbability,
            _options.MinFeatureSelectionProbability,
            _options.MaxFeatureSelectionProbability);
    }

    /// <summary>
    /// Updates the parameter adjustment settings based on whether the solution is improving.
    /// </summary>
    /// <param name="isImproving">Indicates whether the solution is improving.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method adjusts how the optimizer changes the amounts of ingredients.
    /// 
    /// If the solution is improving:
    /// - It might make smaller, more precise adjustments
    /// - It might be less likely to make dramatic changes (like flipping signs)
    /// - It might be more likely to adjust parameters in future steps
    /// 
    /// If the solution is not improving:
    /// - It might make larger adjustments to explore more options
    /// - It might be more willing to try dramatic changes
    /// - It might adjust its strategy to try something different
    /// </para>
    /// </remarks>
    private void UpdateParameterAdjustmentSettings(bool isImproving)
    {
        if (isImproving)
        {
            // If improving, make smaller adjustments to fine-tune
            _options.ParameterAdjustmentScale *= 0.95;

            // Decrease the probability of sign flips when things are going well
            _options.SignFlipProbability *= 0.9;

            // Increase the probability of parameter adjustments
            _options.ParameterAdjustmentProbability *= 1.02;
        }
        else
        {
            // If not improving, make larger adjustments to explore more
            _options.ParameterAdjustmentScale *= 1.05;

            // Increase the probability of sign flips to try more dramatic changes
            _options.SignFlipProbability *= 1.1;

            // Slightly decrease the probability of parameter adjustments
            _options.ParameterAdjustmentProbability *= 0.98;
        }

        // Ensure values stay within bounds
        _options.ParameterAdjustmentScale = MathHelper.Clamp(
            _options.ParameterAdjustmentScale,
            _options.MinParameterAdjustmentScale,
            _options.MaxParameterAdjustmentScale);

        _options.SignFlipProbability = MathHelper.Clamp(
            _options.SignFlipProbability,
            _options.MinSignFlipProbability,
            _options.MaxSignFlipProbability);

        _options.ParameterAdjustmentProbability = MathHelper.Clamp(
            _options.ParameterAdjustmentProbability,
            _options.MinParameterAdjustmentProbability,
            _options.MaxParameterAdjustmentProbability);
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

            // Serialize optimization mode
            writer.Write((int)_options.OptimizationMode);

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

            // Deserialize optimization mode
            _options.OptimizationMode = (OptimizationMode)reader.ReadInt32();

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
        return $"{baseKey}_Adam_{_options.LearningRate}_{_options.MaxIterations}_{_options.OptimizationMode}";
    }
}