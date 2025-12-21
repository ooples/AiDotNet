using Newtonsoft.Json;

namespace AiDotNet.Optimizers;

/// <summary>
/// Implements the Broyden-Fletcher-Goldfarb-Shanno (BFGS) optimization algorithm.
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
public class BFGSOptimizer<T, TInput, TOutput> : GradientBasedOptimizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// The options specific to the BFGS optimization algorithm.
    /// </summary>
    private BFGSOptimizerOptions<T, TInput, TOutput> _options;

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
    /// <param name="model">The model to optimize.</param>
    /// <param name="options">The options for configuring the BFGS algorithm.</param>
    /// <param name="engine">The computation engine (CPU or GPU) for vectorized operations.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This constructor sets up the BFGS optimizer with its initial configuration.
    /// You can customize various aspects of how it works, or use default settings.
    /// </para>
    /// </remarks>
    public BFGSOptimizer(
        IFullModel<T, TInput, TOutput> model,
        BFGSOptimizerOptions<T, TInput, TOutput>? options = null,
        IEngine? engine = null)
        : base(model, options ?? new())
    {
        _options = options ?? new BFGSOptimizerOptions<T, TInput, TOutput>();
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
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        ValidationHelper<T>.ValidateInputData(inputData);

        var currentSolution = InitializeRandomSolution(inputData.XTrain);
        var bestStepData = new OptimizationStepData<T, TInput, TOutput>();
        var previousStepData = new OptimizationStepData<T, TInput, TOutput>();
        var parameters = currentSolution.GetParameters();

        _inverseHessian = Matrix<T>.CreateIdentity(parameters.Length);
        _previousGradient = null;
        _previousParameters = null;
        InitializeAdaptiveParameters();

        for (int iteration = 0; iteration < _options.MaxIterations; iteration++)
        {
            _iteration++;
            parameters = currentSolution.GetParameters();
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
            _previousParameters = parameters;
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
    private IFullModel<T, TInput, TOutput> UpdateSolution(IFullModel<T, TInput, TOutput> currentSolution, Vector<T> gradient,
        OptimizationInputData<T, TInput, TOutput> inputData)
    {
        // === Vectorized BFGS Update using IEngine (Phase B: US-GPU-015) ===

        var parameters = currentSolution.GetParameters();
        if (_previousGradient != null && _previousParameters != null)
        {
            UpdateInverseHessian(parameters, gradient);
        }

        var direction = _inverseHessian!.Multiply(gradient);
        // Vectorized negation
        direction = (Vector<T>)Engine.Multiply(direction, NumOps.Negate(NumOps.One));

        var step = LineSearch(currentSolution, direction, gradient, inputData);
        // Vectorized scaling
        var scaledDirection = (Vector<T>)Engine.Multiply(direction, step);
        var newCoefficients = parameters.Add(scaledDirection);

        return currentSolution.WithParameters(newCoefficients);
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
        // === Partially Vectorized Hessian Update using IEngine (Phase B: US-GPU-015) ===
        // s = current_params - previous_params
        // y = current_grad - previous_grad

        var s = (Vector<T>)Engine.Subtract(currentParameters, _previousParameters!);
        var y = (Vector<T>)Engine.Subtract(currentGradient, _previousGradient!);

        var rho = NumOps.Divide(NumOps.FromDouble(1), y.DotProduct(s));
        var I = Matrix<T>.CreateIdentity(currentParameters.Length);

        var term1 = I.Subtract(s.OuterProduct(y).Multiply(rho));
        var term2 = I.Subtract(y.OuterProduct(s).Multiply(rho));
        var term3 = s.OuterProduct(s).Multiply(rho);

        _inverseHessian = term1.Multiply(_inverseHessian!).Multiply(term2).Add(term3);
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
    /// Updates the options for the BFGS optimizer.
    /// </summary>
    /// <param name="options">The new options to be set.</param>
    /// <exception cref="ArgumentException">Thrown when the provided options are not of the correct type.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method allows you to change the settings of the BFGS optimizer during runtime.
    /// It checks to make sure you're providing the right kind of options specific to the BFGS algorithm.
    /// </para>
    /// </remarks>
    protected override void UpdateOptions(OptimizationAlgorithmOptions<T, TInput, TOutput> options)
    {
        if (options is BFGSOptimizerOptions<T, TInput, TOutput> bfgsOptions)
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
    public override OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions()
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
            _options = JsonConvert.DeserializeObject<BFGSOptimizerOptions<T, TInput, TOutput>>(optionsJson)
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
    protected override string GenerateGradientCacheKey(IFullModel<T, TInput, TOutput> model, TInput X, TOutput y)
    {
        var baseKey = base.GenerateGradientCacheKey(model, X, y);
        return $"{baseKey}_BFGS_{_options.InitialLearningRate}_{_options.Tolerance}_{_iteration}";
    }
}
