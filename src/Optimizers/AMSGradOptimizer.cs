namespace AiDotNet.Optimizers;

/// <summary>
/// Implements the AMSGrad optimization algorithm, a variant of Adam with improved convergence properties.
/// </summary>
/// <remarks>
/// <para>
/// AMSGrad is an adaptive learning rate optimization algorithm that extends Adam by maintaining
/// the maximum of all past squared gradients, which helps prevent the learning rate from decreasing too quickly.
/// This often leads to better convergence properties on some problems.
/// </para>
/// <para><b>For Beginners:</b>
/// AMSGrad is like a smart driving assistant that adjusts the speed of your car (learning rate)
/// based on the road conditions (data). When the road is straight and smooth (simple patterns),
/// it goes faster. When the road is curvy or bumpy (complex patterns), it slows down.
/// 
/// What makes AMSGrad special is that it remembers the bumpiest parts of the journey so far,
/// ensuring that it never speeds up too quickly after encountering difficult terrain.
/// This "memory" helps it avoid taking wrong turns that might look promising at first.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <typeparam name="TInput">The type of input data for the model.</typeparam>
/// <typeparam name="TOutput">The type of output data for the model.</typeparam>
public class AMSGradOptimizer<T, TInput, TOutput> : GradientBasedOptimizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// The options specific to the AMSGrad optimizer.
    /// </summary>
    private AMSGradOptimizerOptions<T, TInput, TOutput> _amsGradOptions = default!;

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
    /// <param name="model">The machine learning model to optimize.</param>
    /// <param name="options">The options for configuring the AMSGrad optimizer.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This sets up the AMSGrad optimizer with its initial configuration.
    /// You can customize various aspects of how it learns, or use default settings.
    /// </para>
    /// </remarks>
    public AMSGradOptimizer(
        IFullModel<T, TInput, TOutput> model,
        AMSGradOptimizerOptions<T, TInput, TOutput>? options = null)
        : base(model, options ?? new AMSGradOptimizerOptions<T, TInput, TOutput>())
    {
        _amsGradOptions = options ?? new AMSGradOptimizerOptions<T, TInput, TOutput>();
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

        CurrentLearningRate = NumOps.FromDouble(_amsGradOptions.LearningRate);
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
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        ValidationHelper<T>.ValidateInputData(inputData);

        var currentSolution = Model.DeepCopy();
        var bestStepData = new OptimizationStepData<T, TInput, TOutput>
        {
            Solution = Model.DeepCopy(),
            FitnessScore = FitnessCalculator.IsHigherScoreBetter ? NumOps.MinValue : NumOps.MaxValue
        };
        var previousStepData = new OptimizationStepData<T, TInput, TOutput>();
        var parameters = currentSolution.GetParameters();
        _m = new Vector<T>(parameters.Length);
        _v = new Vector<T>(parameters.Length);
        _vHat = new Vector<T>(parameters.Length);
        InitializeAdaptiveParameters();

        for (int iteration = 0; iteration < Options.MaxIterations; iteration++)
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

            if (NumOps.LessThan(NumOps.Abs(NumOps.Subtract(bestStepData.FitnessScore, currentStepData.FitnessScore)), NumOps.FromDouble(_amsGradOptions.Tolerance)))
            {
                break;
            }

            currentSolution = newSolution;
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
        var newCoefficients = new Vector<T>(parameters.Length);
        var beta1 = NumOps.FromDouble(_amsGradOptions.Beta1);
        var beta2 = NumOps.FromDouble(_amsGradOptions.Beta2);
        var oneMinusBeta1 = NumOps.FromDouble(1 - _amsGradOptions.Beta1);
        var oneMinusBeta2 = NumOps.FromDouble(1 - _amsGradOptions.Beta2);

        for (int i = 0; i < parameters.Length; i++)
        {
            // Update biased first moment estimate
            _m![i] = NumOps.Add(NumOps.Multiply(beta1, _m[i]), NumOps.Multiply(oneMinusBeta1, gradient[i]));

            // Update biased second raw moment estimate
            _v![i] = NumOps.Add(NumOps.Multiply(beta2, _v[i]), NumOps.Multiply(oneMinusBeta2, NumOps.Multiply(gradient[i], gradient[i])));

            // Update maximum of second raw moment estimate
            _vHat![i] = MathHelper.Max(_vHat[i], _v[i]);

            // Compute bias-corrected first moment estimate
            var mHat = NumOps.Divide(_m[i], NumOps.FromDouble(1 - Math.Pow(_amsGradOptions.Beta1, _t)));

            // Update parameter
            var update = NumOps.Divide(NumOps.Multiply(CurrentLearningRate, mHat), NumOps.Add(NumOps.Sqrt(_vHat[i]), NumOps.FromDouble(_amsGradOptions.Epsilon)));
            newCoefficients[i] = NumOps.Subtract(parameters[i], update);
        }

        return currentSolution.WithParameters(newCoefficients);
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
        // Skip if previous step data is null (first iteration)
        if (previousStepData.Solution == null)
            return;

        base.UpdateAdaptiveParameters(currentStepData, previousStepData);

        if (_amsGradOptions.UseAdaptiveLearningRate)
        {
            if (NumOps.GreaterThan(currentStepData.FitnessScore, previousStepData.FitnessScore))
            {
                CurrentLearningRate = NumOps.Multiply(CurrentLearningRate, NumOps.FromDouble(_amsGradOptions.LearningRateIncreaseFactor));
            }
            else
            {
                CurrentLearningRate = NumOps.Multiply(CurrentLearningRate, NumOps.FromDouble(_amsGradOptions.LearningRateDecreaseFactor));
            }

            CurrentLearningRate = MathHelper.Clamp(CurrentLearningRate,
                NumOps.FromDouble(_amsGradOptions.MinLearningRate),
                NumOps.FromDouble(_amsGradOptions.MaxLearningRate));
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
            _amsGradOptions = amsGradOptions;
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
        return _amsGradOptions;
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

            string optionsJson = JsonConvert.SerializeObject(_amsGradOptions);
            writer.Write(optionsJson);

            writer.Write(_t);

            // Serialize the moment vectors if they exist
            if (_m != null && _v != null && _vHat != null)
            {
                writer.Write(true); // Indicates moment vectors exist
                writer.Write(_m.Length);
                writer.Write(_m.Serialize());
                writer.Write(_v.Serialize());
                writer.Write(_vHat.Serialize());
            }
            else
            {
                writer.Write(false); // Indicates moment vectors don't exist
            }

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
            _amsGradOptions = JsonConvert.DeserializeObject<AMSGradOptimizerOptions<T, TInput, TOutput>>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");

            _t = reader.ReadInt32();

            // Deserialize the moment vectors if they exist
            bool momentVectorsExist = reader.ReadBoolean();
            if (momentVectorsExist)
            {
                int vectorLength = reader.ReadInt32();
                _m = Vector<T>.Deserialize(reader.ReadBytes(reader.ReadInt32()));
                _v = Vector<T>.Deserialize(reader.ReadBytes(reader.ReadInt32()));
                _vHat = Vector<T>.Deserialize(reader.ReadBytes(reader.ReadInt32()));
            }
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
        return $"{baseKey}_AMSGrad_{_amsGradOptions.Beta1}_{_amsGradOptions.Beta2}_{_t}";
    }
}