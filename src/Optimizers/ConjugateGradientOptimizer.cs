namespace AiDotNet.Optimizers;

/// <summary>
/// Implements the Conjugate Gradient optimization algorithm for numerical optimization problems.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// The Conjugate Gradient method is an algorithm for the numerical solution of particular systems of linear equations, 
/// namely those whose matrix is symmetric and positive-definite. It is often used to solve unconstrained optimization problems 
/// such as energy minimization.
/// </para>
/// <para><b>For Beginners:</b> This optimizer is like a smart hiker trying to find the lowest point in a hilly landscape. 
/// It uses information about the slope (gradient) and its previous steps to decide on the best direction to move next, 
/// allowing it to find the lowest point (optimal solution) more efficiently than simpler methods.
/// </para>
/// </remarks>
public class ConjugateGradientOptimizer<T> : GradientBasedOptimizerBase<T>
{
    /// <summary>
    /// The options specific to the Conjugate Gradient optimization algorithm.
    /// </summary>
    private ConjugateGradientOptimizerOptions _options;

    /// <summary>
    /// The direction vector from the previous iteration.
    /// </summary>
    private Vector<T>? _previousDirection;

    /// <summary>
    /// The gradient vector from the previous iteration.
    /// </summary>
    private new Vector<T> _previousGradient;

    /// <summary>
    /// The current iteration count.
    /// </summary>
    private int _iteration;

    /// <summary>
    /// Initializes a new instance of the ConjugateGradientOptimizer class.
    /// </summary>
    /// <param name="options">The options for configuring the Conjugate Gradient algorithm.</param>
    /// <param name="predictionOptions">Options for prediction statistics.</param>
    /// <param name="modelOptions">Options for model statistics.</param>
    /// <param name="modelEvaluator">The model evaluator to use.</param>
    /// <param name="fitDetector">The fit detector to use.</param>
    /// <param name="fitnessCalculator">The fitness calculator to use.</param>
    /// <param name="modelCache">The model cache to use.</param>
    /// <param name="gradientCache">The gradient cache to use.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This constructor sets up the Conjugate Gradient optimizer with its initial configuration.
    /// You can customize various aspects of how it works, or use default settings.
    /// </para>
    /// </remarks>
    public ConjugateGradientOptimizer(
        ConjugateGradientOptimizerOptions? options = null,
        PredictionStatsOptions? predictionOptions = null,
        ModelStatsOptions? modelOptions = null,
        IModelEvaluator<T>? modelEvaluator = null,
        IFitDetector<T>? fitDetector = null,
        IFitnessCalculator<T>? fitnessCalculator = null,
        IModelCache<T>? modelCache = null,
        IGradientCache<T>? gradientCache = null)
        : base(options, predictionOptions, modelOptions, modelEvaluator, fitDetector, fitnessCalculator, modelCache, gradientCache)
    {
        _options = options ?? new ConjugateGradientOptimizerOptions();
        _previousGradient = Vector<T>.Empty();
        InitializeAdaptiveParameters();
    }

    /// <summary>
    /// Initializes the adaptive parameters used in the Conjugate Gradient algorithm.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method sets up the initial state for the optimizer,
    /// including the learning rate and iteration count.
    /// </para>
    /// </remarks>
    protected override void InitializeAdaptiveParameters()
    {
        base.InitializeAdaptiveParameters();
        CurrentLearningRate = NumOps.FromDouble(_options.InitialLearningRate);
        _iteration = 0;
    }

    /// <summary>
    /// Performs the main optimization process using the Conjugate Gradient algorithm.
    /// </summary>
    /// <param name="inputData">The input data for the optimization process.</param>
    /// <returns>The result of the optimization process.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the heart of the Conjugate Gradient algorithm. It iteratively improves the solution
    /// by calculating gradients, determining search directions, and updating the solution. The process continues until it reaches
    /// the maximum number of iterations or meets the stopping criteria.
    /// </para>
    /// </remarks>
    public override OptimizationResult<T> Optimize(OptimizationInputData<T> inputData)
    {
        ValidationHelper<T>.ValidateInputData(inputData);

        var currentSolution = InitializeRandomSolution(inputData.XTrain.Columns);
        var bestStepData = new OptimizationStepData<T>();
        var previousStepData = new OptimizationStepData<T>();

        _previousDirection = null;
        InitializeAdaptiveParameters();

        for (int iteration = 0; iteration < _options.MaxIterations; iteration++)
        {
            _iteration++;
            var gradient = CalculateGradient(currentSolution, inputData.XTrain, inputData.YTrain);
            var direction = CalculateDirection(gradient);
            var newSolution = UpdateSolution(currentSolution, direction, gradient, inputData);

            var currentStepData = EvaluateSolution(newSolution, inputData);
            UpdateBestSolution(currentStepData, ref bestStepData);

            UpdateAdaptiveParameters(currentStepData, previousStepData);

            if (UpdateIterationHistoryAndCheckEarlyStopping(iteration, bestStepData))
            {
                return CreateOptimizationResult(bestStepData, inputData);
            }

            if (NumOps.LessThan(NumOps.Abs(NumOps.Subtract(bestStepData.FitnessScore, currentStepData.FitnessScore)), NumOps.FromDouble(_options.Tolerance)))
            {
                return CreateOptimizationResult(bestStepData, inputData);
            }

            _previousGradient = gradient;
            _previousDirection = direction;
            currentSolution = newSolution;
            previousStepData = currentStepData;
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    /// <summary>
    /// Calculates the search direction for the current iteration.
    /// </summary>
    /// <param name="gradient">The current gradient vector.</param>
    /// <returns>The calculated search direction.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method determines which direction the optimizer should move in.
    /// It uses the current gradient and information from the previous iteration to make this decision.
    /// </para>
    /// </remarks>
    private Vector<T> CalculateDirection(Vector<T> gradient)
    {
        if (_previousGradient == null || _previousDirection == null)
        {
            return gradient.Transform(x => NumOps.Negate(x));
        }

        var beta = CalculateBeta(gradient);
        return gradient.Transform(x => NumOps.Negate(x)).Add(_previousDirection.Multiply(beta));
    }

    /// <summary>
    /// Calculates the beta factor used in the Conjugate Gradient method.
    /// </summary>
    /// <param name="gradient">The current gradient vector.</param>
    /// <returns>The calculated beta factor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Beta is a special number that helps determine how much of the previous
    /// direction should be mixed with the current gradient to form the new direction.
    /// </para>
    /// </remarks>
    private T CalculateBeta(Vector<T> gradient)
    {
        // Fletcher-Reeves formula
        var numerator = gradient.DotProduct(gradient);
        var denominator = _previousGradient.DotProduct(_previousGradient);
        return NumOps.Divide(numerator, denominator);
    }

    /// <summary>
    /// Updates the current solution based on the calculated direction and gradient.
    /// </summary>
    /// <param name="currentSolution">The current solution.</param>
    /// <param name="direction">The calculated search direction.</param>
    /// <param name="gradient">The current gradient.</param>
    /// <param name="inputData">The input data for the optimization process.</param>
    /// <returns>The updated solution.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method takes a step in the calculated direction to find a better solution.
    /// It uses line search to determine how big of a step to take.
    /// </para>
    /// </remarks>
    private ISymbolicModel<T> UpdateSolution(ISymbolicModel<T> currentSolution, Vector<T> direction, Vector<T> gradient, OptimizationInputData<T> inputData)
    {
        var step = LineSearch(currentSolution, direction, gradient, inputData);
        var scaledDirection = direction.Transform(x => NumOps.Multiply(x, step));
        var newCoefficients = currentSolution.Coefficients.Add(scaledDirection);

        return new VectorModel<T>(newCoefficients);
    }

    /// <summary>
    /// Performs a line search to find the optimal step size in the given direction.
    /// </summary>
    /// <param name="currentSolution">The current solution.</param>
    /// <param name="direction">The search direction.</param>
    /// <param name="gradient">The current gradient.</param>
    /// <param name="inputData">The input data for the optimization process.</param>
    /// <returns>The optimal step size.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method is like looking ahead along the chosen direction to find out
    /// how far to step to get the best improvement in the solution.
    /// </para>
    /// </remarks>
    private T LineSearch(ISymbolicModel<T> currentSolution, Vector<T> direction, Vector<T> gradient, OptimizationInputData<T> inputData)
    {
        var alpha = CurrentLearningRate;
        var c1 = NumOps.FromDouble(1e-4);
        var c2 = NumOps.FromDouble(0.9);
        var xTrain = inputData.XTrain;
        var yTrain = inputData.YTrain;

        var initialValue = CalculateLoss(currentSolution, inputData);
        var initialSlope = gradient.DotProduct(direction);

        while (true)
        {
            var newCoefficients = currentSolution.Coefficients.Add(direction.Multiply(alpha));
            var newSolution = new VectorModel<T>(newCoefficients);
            var newValue = CalculateLoss(newSolution, inputData);

            if (NumOps.LessThanOrEquals(newValue, NumOps.Add(initialValue, NumOps.Multiply(NumOps.Multiply(c1, alpha), initialSlope))))
            {
                var newGradient = CalculateGradient(newSolution, xTrain, yTrain);
                var newSlope = newGradient.DotProduct(direction);

                if (NumOps.GreaterThanOrEquals(NumOps.Abs(newSlope), NumOps.Multiply(c2, NumOps.Abs(initialSlope))))
                {
                    return alpha;
                }
            }

            alpha = NumOps.Multiply(alpha, NumOps.FromDouble(0.5));

            if (NumOps.LessThan(alpha, NumOps.FromDouble(1e-10)))
            {
                return NumOps.FromDouble(1e-10);
            }
        }
    }

    /// <summary>
    /// Updates the adaptive parameters of the Conjugate Gradient optimizer based on the current and previous optimization steps.
    /// </summary>
    /// <param name="currentStepData">Data from the current optimization step.</param>
    /// <param name="previousStepData">Data from the previous optimization step.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method adjusts the learning rate of the optimizer based on how well it's performing.
    /// If the current step improved the solution, it increases the learning rate to potentially make bigger improvements.
    /// If not, it decreases the learning rate to be more cautious.
    /// </para>
    /// </remarks>
    protected override void UpdateAdaptiveParameters(OptimizationStepData<T> currentStepData, OptimizationStepData<T> previousStepData)
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
    /// Updates the options for the Conjugate Gradient optimizer.
    /// </summary>
    /// <param name="options">The new options to be set.</param>
    /// <exception cref="ArgumentException">Thrown when the provided options are not of the correct type.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method allows you to change the settings of the Conjugate Gradient optimizer during runtime.
    /// It checks to make sure you're providing the right kind of options specific to this algorithm.
    /// </para>
    /// </remarks>
    protected override void UpdateOptions(OptimizationAlgorithmOptions options)
    {
        if (options is ConjugateGradientOptimizerOptions cgOptions)
        {
            _options = cgOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected ConjugateGradientOptimizerOptions.");
        }
    }

    /// <summary>
    /// Gets the current options of the Conjugate Gradient optimizer.
    /// </summary>
    /// <returns>The current optimization algorithm options.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method allows you to retrieve the current settings of the Conjugate Gradient optimizer.
    /// You can use this to check or save the current configuration.
    /// </para>
    /// </remarks>
    public override OptimizationAlgorithmOptions GetOptions()
    {
        return _options;
    }

    /// <summary>
    /// Serializes the current state of the Conjugate Gradient optimizer into a byte array.
    /// </summary>
    /// <returns>A byte array representing the serialized state of the optimizer.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method saves the current state of the optimizer into a format
    /// that can be stored or transmitted. This is useful for saving progress or sharing the optimizer's state.
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

            writer.Write(_iteration);

            return ms.ToArray();
        }
    }

    /// <summary>
    /// Deserializes a byte array to restore the state of the Conjugate Gradient optimizer.
    /// </summary>
    /// <param name="data">The byte array containing the serialized state of the optimizer.</param>
    /// <exception cref="InvalidOperationException">Thrown when deserialization of optimizer options fails.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method loads a previously saved state of the optimizer.
    /// It's like restoring a saved game, allowing you to continue from where you left off or use a shared optimizer state.
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
            _options = JsonConvert.DeserializeObject<ConjugateGradientOptimizerOptions>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");

            _iteration = reader.ReadInt32();
        }
    }

    /// <summary>
    /// Generates a unique key for caching gradients in the Conjugate Gradient optimizer.
    /// </summary>
    /// <param name="model">The symbolic model for which the gradient is calculated.</param>
    /// <param name="X">The input data matrix.</param>
    /// <param name="y">The target vector.</param>
    /// <returns>A string representing the unique cache key.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method creates a special identifier for storing and retrieving calculated gradients.
    /// It helps avoid recalculating gradients unnecessarily, which can save a lot of computation time.
    /// </para>
    /// </remarks>
    protected override string GenerateGradientCacheKey(ISymbolicModel<T> model, Matrix<T> X, Vector<T> y)
    {
        var baseKey = base.GenerateGradientCacheKey(model, X, y);
        return $"{baseKey}_CG_{_options.InitialLearningRate}_{_options.Tolerance}_{_iteration}";
    }
}