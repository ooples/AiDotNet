using Newtonsoft.Json;

namespace AiDotNet.Optimizers;

/// <summary>
/// Implements the Coordinate Descent optimization algorithm for numerical optimization problems.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Coordinate Descent is an optimization algorithm that minimizes a multivariable function by solving a series of 
/// single-variable optimization problems. It cycles through each variable (coordinate) and optimizes it while holding 
/// the others constant.
/// </para>
/// <para><b>For Beginners:</b> This optimizer is like adjusting the knobs on a complex machine one at a time. 
/// It focuses on improving one aspect of the solution at a time, which can be more manageable and sometimes 
/// more effective than trying to adjust everything at once.
/// </para>
/// </remarks>
public class CoordinateDescentOptimizer<T, TInput, TOutput> : GradientBasedOptimizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// The options specific to the Coordinate Descent optimization algorithm.
    /// </summary>
    private CoordinateDescentOptimizerOptions<T, TInput, TOutput> _options;

    /// <summary>
    /// Vector of learning rates for each coordinate (variable) in the optimization problem.
    /// </summary>
    private Vector<T> _learningRates;

    /// <summary>
    /// Vector of momentum values for each coordinate (variable) in the optimization problem.
    /// </summary>
    private Vector<T> _momentums;

    /// <summary>
    /// Vector of previous update values for each coordinate (variable) in the optimization problem.
    /// </summary>
    private Vector<T> _previousUpdate;

    /// <summary>
    /// Initializes a new instance of the CoordinateDescentOptimizer class.
    /// </summary>
    /// <param name="model">The model to optimize.</param>
    /// <param name="options">The options for configuring the Coordinate Descent algorithm.</param>
    /// <param name="engine">The computation engine (CPU or GPU) for vectorized operations.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This constructor sets up the Coordinate Descent optimizer with its initial configuration.
    /// You can customize various aspects of how it works, or use default settings.
    /// </para>
    /// </remarks>
    public CoordinateDescentOptimizer(
        IFullModel<T, TInput, TOutput> model,
        CoordinateDescentOptimizerOptions<T, TInput, TOutput>? options = null,
        IEngine? engine = null)
        : base(model, options ?? new())
    {
        _options = options ?? new CoordinateDescentOptimizerOptions<T, TInput, TOutput>();
        _learningRates = Vector<T>.Empty();
        _momentums = Vector<T>.Empty();
        _previousUpdate = Vector<T>.Empty();
    }

    /// <summary>
    /// Initializes the adaptive parameters used in the Coordinate Descent algorithm.
    /// </summary>
    /// <param name="currentSolution">The current solution model.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method sets up the initial state for the optimizer,
    /// including learning rates, momentums, and previous updates for each coordinate (variable).
    /// </para>
    /// </remarks>
    private void InitializeAdaptiveParameters(IFullModel<T, TInput, TOutput> currentSolution)
    {
        base.InitializeAdaptiveParameters();
        int dimensions = currentSolution.GetParameters().Length;
        _learningRates = Vector<T>.CreateDefault(dimensions, NumOps.FromDouble(_options.InitialLearningRate));
        _momentums = Vector<T>.CreateDefault(dimensions, NumOps.FromDouble(_options.InitialMomentum));
        _previousUpdate = Vector<T>.CreateDefault(dimensions, NumOps.Zero);
    }

    /// <summary>
    /// Performs the main optimization process using the Coordinate Descent algorithm.
    /// </summary>
    /// <param name="inputData">The input data for the optimization process.</param>
    /// <returns>The result of the optimization process.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the heart of the Coordinate Descent algorithm. It iteratively improves the solution
    /// by updating one coordinate (variable) at a time. The process continues until it reaches the maximum number of iterations 
    /// or meets the stopping criteria.
    /// </para>
    /// </remarks>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        ValidationHelper<T>.ValidateInputData(inputData);

        var currentSolution = InitializeRandomSolution(inputData.XTrain);
        var bestStepData = new OptimizationStepData<T, TInput, TOutput>();
        var previousStepData = new OptimizationStepData<T, TInput, TOutput>();

        InitializeAdaptiveParameters(currentSolution);

        for (int iteration = 0; iteration < _options.MaxIterations; iteration++)
        {
            var newSolution = UpdateSolution(currentSolution, inputData);
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

            currentSolution = newSolution;
            previousStepData = currentStepData;
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    /// <summary>
    /// Updates the current solution by optimizing each coordinate (variable) individually.
    /// </summary>
    /// <param name="currentSolution">The current solution model.</param>
    /// <param name="inputData">The input data for the optimization process.</param>
    /// <returns>The updated solution model.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method goes through each variable in the solution and tries to improve it individually.
    /// It's like fine-tuning each knob on a machine one at a time to get the best overall performance.
    /// </para>
    /// </remarks>
    private IFullModel<T, TInput, TOutput> UpdateSolution(IFullModel<T, TInput, TOutput> currentSolution, OptimizationInputData<T, TInput, TOutput> inputData)
    {
        var newCoefficients = currentSolution.GetParameters().Clone();

        for (int i = 0; i < newCoefficients.Length; i++)
        {
            var gradient = CalculatePartialDerivative(currentSolution, inputData, i);
            var update = CalculateUpdate(gradient, i);
            newCoefficients[i] = NumOps.Add(newCoefficients[i], update);
        }

        return currentSolution.WithParameters(newCoefficients);
    }

    /// <summary>
    /// Calculates the partial derivative (gradient) for a specific coordinate (variable).
    /// </summary>
    /// <param name="model">The current solution model.</param>
    /// <param name="inputData">The input data for the optimization process.</param>
    /// <param name="index">The index of the coordinate to calculate the partial derivative for.</param>
    /// <returns>The calculated partial derivative.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method estimates how much the overall performance would change if we slightly adjust
    /// one specific variable. It helps determine which direction to move that variable to improve the solution.
    /// </para>
    /// </remarks>
    private T CalculatePartialDerivative(IFullModel<T, TInput, TOutput> model, OptimizationInputData<T, TInput, TOutput> inputData, int index)
    {
        var epsilon = NumOps.FromDouble(1e-6);
        var parameters = model.GetParameters();
        var originalCoeff = parameters[index];

        var coefficientsPlus = parameters.Clone();
        coefficientsPlus[index] = NumOps.Add(originalCoeff, epsilon);
        var modelPlus = model.WithParameters(coefficientsPlus);

        var coefficientsMinus = parameters.Clone();
        coefficientsMinus[index] = NumOps.Subtract(originalCoeff, epsilon);
        var modelMinus = model.WithParameters(coefficientsMinus);

        var lossPlus = CalculateLoss(modelPlus, inputData);
        var lossMinus = CalculateLoss(modelMinus, inputData);

        return NumOps.Divide(NumOps.Subtract(lossPlus, lossMinus), NumOps.Multiply(NumOps.FromDouble(2.0), epsilon));
    }

    /// <summary>
    /// Calculates the update for a specific coordinate based on its gradient and momentum.
    /// </summary>
    /// <param name="gradient">The calculated gradient for the coordinate.</param>
    /// <param name="index">The index of the coordinate.</param>
    /// <returns>The calculated update value.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method determines how much to change a specific variable. It considers both
    /// the current gradient (which suggests the best direction to move) and momentum (which helps maintain consistent movement).
    /// </para>
    /// </remarks>
    private T CalculateUpdate(T gradient, int index)
    {
        var update = NumOps.Add(
            NumOps.Multiply(_learningRates[index], gradient),
            NumOps.Multiply(_momentums[index], _previousUpdate[index])
        );
        _previousUpdate[index] = update;

        return NumOps.Negate(update);
    }

    /// <summary>
    /// Updates the adaptive parameters (learning rates and momentums) based on the optimization progress.
    /// </summary>
    /// <param name="currentStepData">Data from the current optimization step.</param>
    /// <param name="previousStepData">Data from the previous optimization step.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method adjusts how big of steps the optimizer takes for each variable.
    /// If the solution is improving, it might increase the step sizes to progress faster. If not, it might decrease
    /// them to be more careful.
    /// </para>
    /// </remarks>
    protected override void UpdateAdaptiveParameters(OptimizationStepData<T, TInput, TOutput> currentStepData, OptimizationStepData<T, TInput, TOutput> previousStepData)
    {
        // === Vectorized Adaptive Parameter Update using IEngine (Phase B: US-GPU-015) ===
        // All learning rates and momentums updated in parallel

        base.UpdateAdaptiveParameters(currentStepData, previousStepData);

        var improvement = NumOps.Subtract(currentStepData.FitnessScore, previousStepData.FitnessScore);

        if (NumOps.GreaterThan(improvement, NumOps.Zero))
        {
            // Improvement: increase learning rates and momentums
            var lrIncreaseFactor = NumOps.Add(NumOps.One, NumOps.FromDouble(_options.LearningRateIncreaseRate));
            var momentumIncreaseFactor = NumOps.Add(NumOps.One, NumOps.FromDouble(_options.MomentumIncreaseRate));

            _learningRates = (Vector<T>)Engine.Multiply(_learningRates, lrIncreaseFactor);
            _momentums = (Vector<T>)Engine.Multiply(_momentums, momentumIncreaseFactor);
        }
        else
        {
            // No improvement: decrease learning rates and momentums
            var lrDecreaseFactor = NumOps.Subtract(NumOps.One, NumOps.FromDouble(_options.LearningRateDecreaseRate));
            var momentumDecreaseFactor = NumOps.Subtract(NumOps.One, NumOps.FromDouble(_options.MomentumDecreaseRate));

            _learningRates = (Vector<T>)Engine.Multiply(_learningRates, lrDecreaseFactor);
            _momentums = (Vector<T>)Engine.Multiply(_momentums, momentumDecreaseFactor);
        }

        // Clamp values to configured ranges (per-element still needed for now)
        var minLr = NumOps.FromDouble(_options.MinLearningRate);
        // === Vectorized Parameter Clamping (Phase B: US-GPU-015) ===
        var maxLr = NumOps.FromDouble(_options.MaxLearningRate);
        var minMom = NumOps.FromDouble(_options.MinMomentum);
        var maxMom = NumOps.FromDouble(_options.MaxMomentum);

        // Clamp all learning rates and momentums at once using Transform
        _learningRates = _learningRates.Transform(lr => MathHelper.Clamp(lr, minLr, maxLr));
        _momentums = _momentums.Transform(mom => MathHelper.Clamp(mom, minMom, maxMom));
    }

    /// <summary>
    /// Updates the options for the Coordinate Descent optimizer.
    /// </summary>
    /// <param name="options">The new options to be set.</param>
    /// <exception cref="ArgumentException">Thrown when the provided options are not of type CoordinateDescentOptimizerOptions.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method allows you to change the settings of the optimizer during runtime.
    /// It ensures that only the correct type of options (specific to Coordinate Descent) can be used.
    /// </para>
    /// </remarks>
    protected override void UpdateOptions(OptimizationAlgorithmOptions<T, TInput, TOutput> options)
    {
        if (options is CoordinateDescentOptimizerOptions<T, TInput, TOutput> cdOptions)
        {
            _options = cdOptions;
        }
        else
        {
            throw new ArgumentException("Options must be of type CoordinateDescentOptimizerOptions", nameof(options));
        }
    }

    /// <summary>
    /// Retrieves the current options of the Coordinate Descent optimizer.
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
    /// Serializes the Coordinate Descent optimizer to a byte array.
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

            // Serialize _learningRates
            byte[] learningRatesData = _learningRates.Serialize();
            writer.Write(learningRatesData.Length);
            writer.Write(learningRatesData);

            // Serialize _momentums
            byte[] momentumsData = _momentums.Serialize();
            writer.Write(momentumsData.Length);
            writer.Write(momentumsData);

            // Serialize _previousUpdate
            byte[] previousUpdateData = _previousUpdate.Serialize();
            writer.Write(previousUpdateData.Length);
            writer.Write(previousUpdateData);

            return ms.ToArray();
        }
    }

    /// <summary>
    /// Deserializes the Coordinate Descent optimizer from a byte array.
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
            _options = JsonConvert.DeserializeObject<CoordinateDescentOptimizerOptions<T, TInput, TOutput>>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");

            // Deserialize _learningRates
            int learningRatesLength = reader.ReadInt32();
            byte[] learningRatesData = reader.ReadBytes(learningRatesLength);
            _learningRates = Vector<T>.Deserialize(learningRatesData);

            // Deserialize _momentums
            int momentumsLength = reader.ReadInt32();
            byte[] momentumsData = reader.ReadBytes(momentumsLength);
            _momentums = Vector<T>.Deserialize(momentumsData);

            // Deserialize _previousUpdate
            int previousUpdateLength = reader.ReadInt32();
            byte[] previousUpdateData = reader.ReadBytes(previousUpdateLength);
            _previousUpdate = Vector<T>.Deserialize(previousUpdateData);
        }
    }
}
