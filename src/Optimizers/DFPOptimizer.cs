using Newtonsoft.Json;

namespace AiDotNet.Optimizers;

/// <summary>
/// Implements the Davidon-Fletcher-Powell (DFP) optimization algorithm for numerical optimization problems.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// The DFP algorithm is a quasi-Newton method for solving unconstrained nonlinear optimization problems.
/// It approximates the inverse Hessian matrix to determine the search direction, combining the efficiency
/// of Newton's method with the stability of gradient descent.
/// </para>
/// <para><b>For Beginners:</b> This optimizer is like a smart navigator that learns from its past steps
/// to make better decisions about which direction to move in the future. It's particularly good at
/// handling complex optimization problems where the landscape of possible solutions is intricate.
/// </para>
/// </remarks>
public class DFPOptimizer<T, TInput, TOutput> : GradientBasedOptimizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// The options specific to the DFP optimization algorithm.
    /// </summary>
    private DFPOptimizerOptions<T, TInput, TOutput> _options;

    /// <summary>
    /// The inverse Hessian matrix approximation used in the DFP algorithm.
    /// </summary>
    private Matrix<T> _inverseHessian;

    /// <summary>
    /// The gradient from the previous iteration.
    /// </summary>
    private new Vector<T> _previousGradient;

    /// <summary>
    /// The current adaptive learning rate.
    /// </summary>
    private T _adaptiveLearningRate;

    /// <summary>
    /// Initializes a new instance of the DFPOptimizer class.
    /// </summary>
    /// <param name="options">The options for configuring the DFP algorithm.</param>
    /// <param name="predictionOptions">Options for prediction statistics.</param>
    /// <param name="modelOptions">Options for model statistics.</param>
    /// <param name="modelEvaluator">The model evaluator to use.</param>
    /// <param name="fitDetector">The fit detector to use.</param>
    /// <param name="fitnessCalculator">The fitness calculator to use.</param>
    /// <param name="modelCache">The model cache to use.</param>
    /// <param name="gradientCache">The gradient cache to use.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This constructor sets up the DFP optimizer with its initial configuration.
    /// You can customize various aspects of how it works, or use default settings.
    /// </para>
    /// </remarks>
    /// <param name="model">The model to optimize.</param>
    /// <param name="engine">The computation engine (CPU or GPU) for vectorized operations.</param>
    public DFPOptimizer(
        IFullModel<T, TInput, TOutput> model,
        DFPOptimizerOptions<T, TInput, TOutput>? options = null,
        IEngine? engine = null)
        : base(model, options ?? new())
    {
        _options = options ?? new DFPOptimizerOptions<T, TInput, TOutput>();
        _previousGradient = Vector<T>.Empty();
        _inverseHessian = Matrix<T>.Empty();
        _adaptiveLearningRate = NumOps.Zero;

        InitializeAdaptiveParameters();
    }

    /// <summary>
    /// Initializes the adaptive parameters used in the DFP algorithm.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method sets up the initial learning rate for the optimizer.
    /// The learning rate determines how big of steps the optimizer takes when improving the solution.
    /// </para>
    /// </remarks>
    protected override void InitializeAdaptiveParameters()
    {
        _adaptiveLearningRate = NumOps.FromDouble(_options.InitialLearningRate);
    }

    /// <summary>
    /// Performs the main optimization process using the DFP algorithm.
    /// </summary>
    /// <param name="inputData">The input data for the optimization process.</param>
    /// <returns>The result of the optimization process.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the heart of the DFP algorithm. It iteratively improves the solution
    /// by calculating gradients, determining search directions, and updating the solution. The process continues
    /// until it reaches the maximum number of iterations or meets the stopping criteria.
    /// </para>
    /// </remarks>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        ValidationHelper<T>.ValidateInputData(inputData);

        var currentSolution = InitializeRandomSolution(inputData.XTrain);
        var bestStepData = new OptimizationStepData<T, TInput, TOutput>();
        var previousStepData = new OptimizationStepData<T, TInput, TOutput>();

        _inverseHessian = Matrix<T>.CreateIdentity(currentSolution.GetParameters().Length);

        InitializeAdaptiveParameters();

        for (int iteration = 0; iteration < _options.MaxIterations; iteration++)
        {
            var gradient = CalculateGradient(currentSolution, inputData.XTrain, inputData.YTrain);
            var direction = CalculateDirection(gradient);
            var newSolution = UpdateSolution(currentSolution, direction, gradient, inputData);

            var currentStepData = EvaluateSolution(newSolution, inputData);
            UpdateBestSolution(currentStepData, ref bestStepData);

            UpdateAdaptiveParameters(currentStepData, previousStepData);
            UpdateInverseHessian(currentSolution, newSolution, gradient);

            if (UpdateIterationHistoryAndCheckEarlyStopping(iteration, bestStepData))
            {
                return CreateOptimizationResult(bestStepData, inputData);
            }

            if (NumOps.LessThan(NumOps.Abs(NumOps.Subtract(bestStepData.FitnessScore, currentStepData.FitnessScore)), NumOps.FromDouble(_options.Tolerance)))
            {
                return CreateOptimizationResult(bestStepData, inputData);
            }

            _previousGradient = gradient;
            currentSolution = newSolution;
            previousStepData = currentStepData;
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    /// <summary>
    /// Calculates the search direction using the inverse Hessian approximation and the current gradient.
    /// </summary>
    /// <param name="gradient">The current gradient.</param>
    /// <returns>The calculated search direction.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method determines which direction the optimizer should move in
    /// to improve the solution. It uses the inverse Hessian matrix to make this decision more intelligent
    /// than just following the steepest descent.
    /// </para>
    /// </remarks>
    private Vector<T> CalculateDirection(Vector<T> gradient)
    {
        // === Vectorized Direction Calculation using IEngine (Phase B: US-GPU-015) ===
        // direction = -H^{-1} * gradient

        var Hg = _inverseHessian.Multiply(gradient);
        return (Vector<T>)Engine.Multiply(Hg, NumOps.Negate(NumOps.One));
    }

    /// <summary>
    /// Updates the current solution by moving in the calculated direction.
    /// </summary>
    /// <param name="currentSolution">The current solution model.</param>
    /// <param name="direction">The calculated search direction.</param>
    /// <param name="gradient">The current gradient.</param>
    /// <param name="inputData">The input data for the optimization process.</param>
    /// <returns>The updated solution model.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method takes a step in the direction calculated by CalculateDirection.
    /// It uses line search to determine how big of a step to take, then updates the solution accordingly.
    /// </para>
    /// </remarks>
    private IFullModel<T, TInput, TOutput> UpdateSolution(IFullModel<T, TInput, TOutput> currentSolution, Vector<T> direction, Vector<T> gradient,
        OptimizationInputData<T, TInput, TOutput> inputData)
    {
        // === Vectorized Solution Update using IEngine (Phase B: US-GPU-015) ===
        // new_params = current_params + stepSize * direction

        var stepSize = LineSearch(currentSolution, direction, gradient, inputData);
        var scaledDirection = (Vector<T>)Engine.Multiply(direction, stepSize);
        var newCoefficients = currentSolution.GetParameters().Add(scaledDirection);

        return currentSolution.WithParameters(newCoefficients);
    }

    /// <summary>
    /// Updates the inverse Hessian approximation using the DFP update formula.
    /// </summary>
    /// <param name="currentSolution">The current solution model.</param>
    /// <param name="newSolution">The new solution model.</param>
    /// <param name="gradient">The current gradient.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method updates the optimizer's "memory" of how the solution space looks.
    /// It helps the optimizer make better decisions about which direction to move in future iterations.
    /// </para>
    /// </remarks>
    private void UpdateInverseHessian(IFullModel<T, TInput, TOutput> currentSolution, IFullModel<T, TInput, TOutput> newSolution, Vector<T> gradient)
    {
        // === Partially Vectorized DFP Hessian Update using IEngine (Phase B: US-GPU-015) ===

        if (_previousGradient == null)
        {
            _previousGradient = gradient;
            return;
        }

        // Vectorized parameter and gradient differences
        var s = (Vector<T>)Engine.Subtract(newSolution.GetParameters(), currentSolution.GetParameters());
        var y = (Vector<T>)Engine.Subtract(gradient, _previousGradient);

        var sTy = s.DotProduct(y);
        if (NumOps.LessThanOrEquals(sTy, NumOps.Zero))
        {
            return; // Skip update if sTy is not positive
        }

        // DFP update formula: H = H + (s s^T) / (s^T y) - (H y) (H y)^T / (y^T H y)
        var term1 = Matrix<T>.OuterProduct(s, s).Divide(sTy);
        var Hy = _inverseHessian.Multiply(y);
        var yTHy = y.DotProduct(Hy);
        var term2 = Matrix<T>.OuterProduct(Hy, Hy).Divide(yTHy);

        _inverseHessian = _inverseHessian.Add(term1).Subtract(term2);
    }

    /// <summary>
    /// Updates the adaptive parameters based on the optimization progress.
    /// </summary>
    /// <param name="currentStepData">Data from the current optimization step.</param>
    /// <param name="previousStepData">Data from the previous optimization step.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method adjusts how big of steps the optimizer takes.
    /// If the solution is improving, it might increase the step size to progress faster.
    /// If not, it might decrease the step size to be more careful.
    /// </para>
    /// </remarks>
    protected override void UpdateAdaptiveParameters(OptimizationStepData<T, TInput, TOutput> currentStepData, OptimizationStepData<T, TInput, TOutput> previousStepData)
    {
        base.UpdateAdaptiveParameters(currentStepData, previousStepData);

        if (_options.UseAdaptiveLearningRate)
        {
            var improvement = NumOps.Subtract(currentStepData.FitnessScore, previousStepData.FitnessScore);
            var adaptationRate = NumOps.FromDouble(_options.AdaptationRate);

            if (NumOps.GreaterThan(improvement, NumOps.Zero))
            {
                _adaptiveLearningRate = NumOps.Multiply(_adaptiveLearningRate, NumOps.Add(NumOps.One, adaptationRate));
            }
            else
            {
                _adaptiveLearningRate = NumOps.Multiply(_adaptiveLearningRate, NumOps.Subtract(NumOps.One, adaptationRate));
            }

            _adaptiveLearningRate = MathHelper.Clamp(_adaptiveLearningRate, NumOps.FromDouble(_options.MinLearningRate), NumOps.FromDouble(_options.MaxLearningRate));
        }
    }

    /// <summary>
    /// Updates the options for the DFP optimizer.
    /// </summary>
    /// <param name="options">The new options to be set.</param>
    /// <exception cref="ArgumentException">Thrown when the provided options are not of type DFPOptimizerOptions.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method allows you to change the settings of the optimizer during runtime.
    /// It ensures that only the correct type of options (specific to DFP) can be used.
    /// </para>
    /// </remarks>
    protected override void UpdateOptions(OptimizationAlgorithmOptions<T, TInput, TOutput> options)
    {
        if (options is DFPOptimizerOptions<T, TInput, TOutput> dfpOptions)
        {
            _options = dfpOptions;
        }
        else
        {
            throw new ArgumentException("Options must be of type DFPOptimizerOptions", nameof(options));
        }
    }

    /// <summary>
    /// Retrieves the current options of the DFP optimizer.
    /// </summary>
    /// <returns>The current optimization algorithm options.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method allows you to check the current settings of the optimizer.
    /// It's useful if you need to inspect or copy the current configuration.
    /// </para>
    /// </remarks>
    public override OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions()
    {
        return _options;
    }

    /// <summary>
    /// Serializes the DFP optimizer to a byte array.
    /// </summary>
    /// <returns>A byte array representing the serialized state of the optimizer.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method converts the current state of the optimizer into a series of bytes.
    /// This is useful for saving the optimizer's state to a file or sending it over a network.
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

            // Serialize _inverseHessian
            byte[] inverseHessianData = _inverseHessian.Serialize();
            writer.Write(inverseHessianData.Length);
            writer.Write(inverseHessianData);

            // Serialize _previousGradient
            byte[] previousGradientData = _previousGradient.Serialize();
            writer.Write(previousGradientData.Length);
            writer.Write(previousGradientData);

            // Serialize _adaptiveLearningRate
            writer.Write(Convert.ToDouble(_adaptiveLearningRate));

            return ms.ToArray();
        }
    }

    /// <summary>
    /// Deserializes the DFP optimizer from a byte array.
    /// </summary>
    /// <param name="data">The byte array containing the serialized optimizer state.</param>
    /// <exception cref="InvalidOperationException">Thrown when deserialization of optimizer options fails.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method reconstructs the optimizer's state from a series of bytes.
    /// It's used to restore a previously saved state of the optimizer, allowing you to continue from where you left off.
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
            _options = JsonConvert.DeserializeObject<DFPOptimizerOptions<T, TInput, TOutput>>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");

            // Deserialize _inverseHessian
            int inverseHessianLength = reader.ReadInt32();
            byte[] inverseHessianData = reader.ReadBytes(inverseHessianLength);
            _inverseHessian = Matrix<T>.Deserialize(inverseHessianData);

            // Deserialize _previousGradient
            int previousGradientLength = reader.ReadInt32();
            byte[] previousGradientData = reader.ReadBytes(previousGradientLength);
            _previousGradient = Vector<T>.Deserialize(previousGradientData);

            // Deserialize _adaptiveLearningRate
            _adaptiveLearningRate = NumOps.FromDouble(reader.ReadDouble());
        }
    }
}
