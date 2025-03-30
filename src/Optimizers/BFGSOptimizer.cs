namespace AiDotNet.Optimizers;

/// <summary>
/// Implements the Broyden–Fletcher–Goldfarb–Shanno (BFGS) optimization algorithm.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// BFGS is a quasi-Newton method for solving unconstrained nonlinear optimization problems.
/// It approximates the Hessian matrix of second derivatives of the function to be minimized.
/// </para>
/// <para><b>For Beginners:</b> BFGS is an advanced optimization algorithm that tries to find the best solution
/// by making smart steps based on the function's behavior. It's particularly good at handling complex problems
/// where the function being optimized is smooth but potentially has many variables.
/// </para>
/// </remarks>
public class BFGSOptimizer<T> : GradientBasedOptimizerBase<T>
{
    /// <summary>
    /// The options specific to the BFGS optimization algorithm.
    /// </summary>
    private BFGSOptimizerOptions _options;

    /// <summary>
    /// The approximation of the inverse Hessian matrix.
    /// </summary>
    private Matrix<T>? _inverseHessian;

    /// <summary>
    /// The gradient from the previous iteration.
    /// </summary>
    private new Vector<T>? _previousGradient;

    /// <summary>
    /// The parameters from the previous iteration.
    /// </summary>
    private Vector<T>? _previousParameters;

    /// <summary>
    /// The current iteration count.
    /// </summary>
    private int _iteration;

    /// <summary>
    /// Initializes a new instance of the BFGSOptimizer class.
    /// </summary>
    /// <param name="options">The options for configuring the BFGS algorithm.</param>
    /// <param name="predictionOptions">Options for prediction statistics.</param>
    /// <param name="modelOptions">Options for model statistics.</param>
    /// <param name="modelEvaluator">The model evaluator to use.</param>
    /// <param name="fitDetector">The fit detector to use.</param>
    /// <param name="fitnessCalculator">The fitness calculator to use.</param>
    /// <param name="modelCache">The model cache to use.</param>
    /// <param name="gradientCache">The gradient cache to use.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This constructor sets up the BFGS optimizer with its initial configuration.
    /// You can customize various aspects of how it works, or use default settings.
    /// </para>
    /// </remarks>
    public BFGSOptimizer(
        BFGSOptimizerOptions? options = null,
        PredictionStatsOptions? predictionOptions = null,
        ModelStatsOptions? modelOptions = null,
        IModelEvaluator<T>? modelEvaluator = null,
        IFitDetector<T>? fitDetector = null,
        IFitnessCalculator<T>? fitnessCalculator = null,
        IModelCache<T>? modelCache = null,
        IGradientCache<T>? gradientCache = null)
        : base(options, predictionOptions, modelOptions, modelEvaluator, fitDetector, fitnessCalculator, modelCache, gradientCache)
    {
        _options = options ?? new BFGSOptimizerOptions();
        InitializeAdaptiveParameters();
    }

    /// <summary>
    /// Initializes the adaptive parameters used in the BFGS algorithm.
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
    /// Performs the main optimization process using the BFGS algorithm.
    /// </summary>
    /// <param name="inputData">The input data for the optimization process.</param>
    /// <returns>The result of the optimization process.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the heart of the BFGS algorithm. It iteratively improves the solution
    /// by updating the parameters based on the gradient and the approximated inverse Hessian matrix.
    /// The process continues until it reaches the maximum number of iterations or meets the convergence criteria.
    /// </para>
    /// </remarks>
    public override OptimizationResult<T> Optimize(OptimizationInputData<T> inputData)
    {
        ValidationHelper<T>.ValidateInputData(inputData);

        var currentSolution = InitializeRandomSolution(inputData.XTrain.Columns);
        var bestStepData = new OptimizationStepData<T>();
        var previousStepData = new OptimizationStepData<T>();

        _inverseHessian = Matrix<T>.CreateIdentity(currentSolution.Coefficients.Length);
        _previousGradient = null;
        _previousParameters = null;
        InitializeAdaptiveParameters();

        for (int iteration = 0; iteration < _options.MaxIterations; iteration++)
        {
            _iteration++;
            var gradient = CalculateGradient(currentSolution, inputData.XTrain, inputData.YTrain);
            var newSolution = UpdateSolution(currentSolution, gradient, inputData);

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
            _previousParameters = currentSolution.Coefficients;
            currentSolution = newSolution;
            previousStepData = currentStepData;
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    /// <summary>
    /// Updates the current solution using the BFGS update formula.
    /// </summary>
    /// <param name="currentSolution">The current solution.</param>
    /// <param name="gradient">The current gradient.</param>
    /// <param name="inputData">The input data for the optimization process.</param>
    /// <returns>The updated solution.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method calculates the next step in the optimization process.
    /// It uses the inverse Hessian approximation to determine the direction and magnitude of the update.
    /// </para>
    /// </remarks>
    private ISymbolicModel<T> UpdateSolution(ISymbolicModel<T> currentSolution, Vector<T> gradient, OptimizationInputData<T> inputData)
    {
        if (_previousGradient != null && _previousParameters != null)
        {
            UpdateInverseHessian(currentSolution.Coefficients, gradient);
        }

        var direction = _inverseHessian!.Multiply(gradient);
        direction = direction.Transform(x => NumOps.Negate(x));

        var step = LineSearch(currentSolution, direction, gradient, inputData);
        var scaledDirection = direction.Transform(x => NumOps.Multiply(x, step));
        var newCoefficients = currentSolution.Coefficients.Add(scaledDirection);

        return new VectorModel<T>(newCoefficients);
    }

    /// <summary>
    /// Updates the approximation of the inverse Hessian matrix.
    /// </summary>
    /// <param name="currentParameters">The current parameter values.</param>
    /// <param name="currentGradient">The current gradient.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method updates the BFGS algorithm's internal representation of the function's curvature.
    /// It helps the algorithm make more informed decisions about how to update the parameters in future iterations.
    /// </para>
    /// </remarks>
    private void UpdateInverseHessian(Vector<T> currentParameters, Vector<T> currentGradient)
    {
        var s = currentParameters.Subtract(_previousParameters!);
        var y = currentGradient.Subtract(_previousGradient!);

        var rho = NumOps.Divide(NumOps.FromDouble(1), y.DotProduct(s));
        var I = Matrix<T>.CreateIdentity(currentParameters.Length);

        var term1 = I.Subtract(s.OuterProduct(y).Multiply(rho));
        var term2 = I.Subtract(y.OuterProduct(s).Multiply(rho));
        var term3 = s.OuterProduct(s).Multiply(rho);

        _inverseHessian = term1.Multiply(_inverseHessian!).Multiply(term2).Add(term3);
    }

    /// <summary>
    /// Performs a line search to find an appropriate step size.
    /// </summary>
    /// <param name="currentSolution">The current solution.</param>
    /// <param name="direction">The search direction.</param>
    /// <param name="gradient">The current gradient.</param>
    /// <param name="inputData">The input data for the optimization process.</param>
    /// <returns>The step size to use.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method determines how big of a step to take in the chosen direction.
    /// It tries to find a step size that sufficiently decreases the function value while not being too small.
    /// </para>
    /// </remarks>
    private T LineSearch(ISymbolicModel<T> currentSolution, Vector<T> direction, Vector<T> gradient, OptimizationInputData<T> inputData)
    {
        var alpha = NumOps.FromDouble(1.0);
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
    /// Updates the adaptive parameters of the optimizer.
    /// </summary>
    /// <param name="currentStepData">The current step data.</param>
    /// <param name="previousStepData">The previous step data.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method adjusts the learning rate based on the performance of the current step
    /// compared to the previous step. If the current step improved the fitness score, the learning rate is increased;
    /// otherwise, it's decreased. This helps the optimizer adapt to the landscape of the problem.
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
    /// Updates the options for the BFGS optimizer.
    /// </summary>
    /// <param name="options">The new options to be set.</param>
    /// <exception cref="ArgumentException">Thrown when the provided options are not of the correct type.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method allows you to change the settings of the BFGS optimizer during runtime.
    /// It checks to make sure you're providing the right kind of options specific to the BFGS algorithm.
    /// </para>
    /// </remarks>
    protected override void UpdateOptions(OptimizationAlgorithmOptions options)
    {
        if (options is BFGSOptimizerOptions bfgsOptions)
        {
            _options = bfgsOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected BFGSOptimizerOptions.");
        }
    }

    /// <summary>
    /// Gets the current options for the BFGS optimizer.
    /// </summary>
    /// <returns>The current optimization options.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method lets you see what settings the BFGS optimizer is currently using.
    /// </para>
    /// </remarks>
    public override OptimizationAlgorithmOptions GetOptions()
    {
        return _options;
    }

    /// <summary>
    /// Converts the current state of the BFGS optimizer into a byte array for storage or transmission.
    /// </summary>
    /// <returns>A byte array representing the serialized state of the optimizer.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method takes all the important information about the current state
    /// of the BFGS Optimizer and turns it into a format that can be easily saved or sent to another computer.
    /// It includes both the base optimizer data and BFGS-specific data.
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
    /// Restores the state of the BFGS optimizer from a byte array.
    /// </summary>
    /// <param name="data">The byte array containing the serialized state of the optimizer.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method takes a saved state of the BFGS Optimizer (in the form of a byte array)
    /// and uses it to restore the optimizer to that state. It's like loading a saved game, bringing back all the
    /// important settings and progress that were saved earlier.
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
            _options = JsonConvert.DeserializeObject<BFGSOptimizerOptions>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");

            _iteration = reader.ReadInt32();
        }
    }

    /// <summary>
    /// Generates a unique key for caching gradients in the BFGS optimization process.
    /// </summary>
    /// <param name="model">The current model.</param>
    /// <param name="X">The input data matrix.</param>
    /// <param name="y">The target values vector.</param>
    /// <returns>A string representing the unique cache key.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method creates a unique identifier for storing and retrieving gradients
    /// during the optimization process. It helps avoid recalculating gradients unnecessarily, which can save time.
    /// The key includes BFGS-specific information to ensure it's unique to this optimizer's current state.
    /// </para>
    /// </remarks>
    protected override string GenerateGradientCacheKey(ISymbolicModel<T> model, Matrix<T> X, Vector<T> y)
    {
        var baseKey = base.GenerateGradientCacheKey(model, X, y);
        return $"{baseKey}_BFGS_{_options.InitialLearningRate}_{_options.Tolerance}_{_iteration}";
    }
}