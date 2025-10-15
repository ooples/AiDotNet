namespace AiDotNet.Optimizers;

/// <summary>
/// Implements the Conjugate Gradient optimization algorithm for machine learning models.
/// </summary>
/// <remarks>
/// <para>
/// Conjugate Gradient is an advanced optimization algorithm that improves upon basic gradient descent
/// by using conjugate directions instead of the local gradient, making it more efficient for certain
/// types of problems, especially quadratic or nearly quadratic functions.
/// </para>
/// <para><b>For Beginners:</b>
/// Conjugate Gradient is like hiking in the mountains with a smart compass that:
/// 
/// 1. Doesn't just point downhill in the steepest direction (like regular gradient descent)
/// 2. Instead, remembers your previous direction and calculates a new direction that's "conjugate"
///    (a special mathematical relationship) to your previous directions
/// 3. This helps avoid zigzagging back and forth in valleys, leading to faster progress
/// 
/// Imagine you're trying to descend a long, narrow valley:
/// - Regular gradient descent would zigzag from side to side, making slow progress
/// - Conjugate gradient would find a direction that heads diagonally down the valley,
///   reaching the bottom much more efficiently
/// 
/// This algorithm is particularly effective for optimization problems where:
/// - The function being optimized is quadratic or nearly quadratic
/// - The problem has many dimensions but a simple structure
/// - Memory is limited (compared to methods that store approximations of second derivatives)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <typeparam name="TInput">The type of input data for the model.</typeparam>
/// <typeparam name="TOutput">The type of output data for the model.</typeparam>
public class ConjugateGradientOptimizer<T, TInput, TOutput> : GradientBasedOptimizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// The options specific to the Conjugate Gradient optimization algorithm.
    /// </summary>
    private ConjugateGradientOptimizerOptions<T, TInput, TOutput> _cgOptions = default!;

    /// <summary>
    /// The direction vector from the previous iteration.
    /// </summary>
    private Vector<T>? _previousDirection;

    /// <summary>
    /// The gradient vector from the previous iteration.
    /// </summary>
    private new Vector<T>? _previousGradient;

    /// <summary>
    /// The current iteration count.
    /// </summary>
    private int _iteration;

    /// <summary>
    /// Initializes a new instance of the ConjugateGradientOptimizer class.
    /// </summary>
    /// <param name="model">The machine learning model to optimize.</param>
    /// <param name="options">The options for configuring the Conjugate Gradient algorithm.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This constructor sets up the Conjugate Gradient optimizer with its initial configuration.
    /// You provide the model you want to optimize, and you can customize various aspects of how it works,
    /// or use default settings.
    /// </para>
    /// </remarks>
    public ConjugateGradientOptimizer(
        IFullModel<T, TInput, TOutput> model,
        ConjugateGradientOptimizerOptions<T, TInput, TOutput>? options = null)
        : base(model, options ?? new ConjugateGradientOptimizerOptions<T, TInput, TOutput>())
    {
        _cgOptions = options ?? new ConjugateGradientOptimizerOptions<T, TInput, TOutput>();
        _previousGradient = null;
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
        CurrentLearningRate = NumOps.FromDouble(_cgOptions.InitialLearningRate);
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

        _previousDirection = null;
        _previousGradient = null;
        InitializeAdaptiveParameters();

        for (int iteration = 0; iteration < Options.MaxIterations; iteration++)
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
                break;
            }

            if (NumOps.LessThan(NumOps.Abs(NumOps.Subtract(bestStepData.FitnessScore, currentStepData.FitnessScore)), NumOps.FromDouble(_cgOptions.Tolerance)))
            {
                break;
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
    /// <para>
    /// This method implements the Fletcher-Reeves formula for calculating the beta coefficient,
    /// which determines how much of the previous search direction is retained when computing
    /// the new search direction.
    /// </para>
    /// <para><b>For Beginners:</b> Beta is a special number that helps determine how much of the previous
    /// direction should be mixed with the current gradient to form the new direction.
    /// </para>
    /// </remarks>
    private T CalculateBeta(Vector<T> gradient)
    {
        // Fletcher-Reeves formula
        var numerator = gradient.DotProduct(gradient);

        // Ensure _previousGradient is not null before using it
        if (_previousGradient == null)
        {
            // If _previousGradient is null, return zero (no contribution from previous direction)
            return NumOps.Zero;
        }

        var denominator = _previousGradient.DotProduct(_previousGradient);

        // Avoid division by zero
        if (MathHelper.AlmostEqual(denominator, NumOps.Zero))
        {
            return NumOps.Zero;
        }

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
    private IFullModel<T, TInput, TOutput> UpdateSolution(IFullModel<T, TInput, TOutput> currentSolution, Vector<T> direction, Vector<T> gradient,
        OptimizationInputData<T, TInput, TOutput> inputData)
    {
        var step = LineSearch(currentSolution, direction, gradient, inputData);
        var scaledDirection = direction.Transform(x => NumOps.Multiply(x, step));
        var newCoefficients = currentSolution.GetParameters().Add(scaledDirection);

        return currentSolution.WithParameters(newCoefficients);
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
    protected override void UpdateAdaptiveParameters(OptimizationStepData<T, TInput, TOutput> currentStepData, OptimizationStepData<T, TInput, TOutput> previousStepData)
    {
        // Skip if previous step data is null (first iteration)
        if (previousStepData.Solution == null)
            return;

        base.UpdateAdaptiveParameters(currentStepData, previousStepData);

        if (_cgOptions.UseAdaptiveLearningRate)
        {
            if (NumOps.GreaterThan(currentStepData.FitnessScore, previousStepData.FitnessScore))
            {
                CurrentLearningRate = NumOps.Multiply(CurrentLearningRate, NumOps.FromDouble(_cgOptions.LearningRateIncreaseFactor));
            }
            else
            {
                CurrentLearningRate = NumOps.Multiply(CurrentLearningRate, NumOps.FromDouble(_cgOptions.LearningRateDecreaseFactor));
            }

            CurrentLearningRate = MathHelper.Clamp(CurrentLearningRate,
                NumOps.FromDouble(_cgOptions.MinLearningRate),
                NumOps.FromDouble(_cgOptions.MaxLearningRate));
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
    protected override void UpdateOptions(OptimizationAlgorithmOptions<T, TInput, TOutput> options)
    {
        if (options is ConjugateGradientOptimizerOptions<T, TInput, TOutput> cgOptions)
        {
            _cgOptions = cgOptions;
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
    public override OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions()
    {
        return _cgOptions;
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

            string optionsJson = JsonConvert.SerializeObject(_cgOptions);
            writer.Write(optionsJson);

            writer.Write(_iteration);

            // Serialize _previousGradient if it exists
            if (_previousGradient != null)
            {
                writer.Write(true); // Indicates _previousGradient exists
                writer.Write(_previousGradient.Length);
                for (int i = 0; i < _previousGradient.Length; i++)
                {
                    writer.Write(Convert.ToDouble(_previousGradient[i]));
                }
            }
            else
            {
                writer.Write(false); // Indicates _previousGradient doesn't exist
            }

            // Serialize _previousDirection if it exists
            if (_previousDirection != null)
            {
                writer.Write(true); // Indicates _previousDirection exists
                writer.Write(_previousDirection.Length);
                for (int i = 0; i < _previousDirection.Length; i++)
                {
                    writer.Write(Convert.ToDouble(_previousDirection[i]));
                }
            }
            else
            {
                writer.Write(false); // Indicates _previousDirection doesn't exist
            }

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
            _cgOptions = JsonConvert.DeserializeObject<ConjugateGradientOptimizerOptions<T, TInput, TOutput>>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");

            _iteration = reader.ReadInt32();

            // Deserialize _previousGradient if it exists
            bool previousGradientExists = reader.ReadBoolean();
            if (previousGradientExists)
            {
                int length = reader.ReadInt32();
                _previousGradient = new Vector<T>(length);
                for (int i = 0; i < length; i++)
                {
                    _previousGradient[i] = NumOps.FromDouble(reader.ReadDouble());
                }
            }
            else
            {
                _previousGradient = null;
            }

            // Deserialize _previousDirection if it exists
            bool previousDirectionExists = reader.ReadBoolean();
            if (previousDirectionExists)
            {
                int length = reader.ReadInt32();
                _previousDirection = new Vector<T>(length);
                for (int i = 0; i < length; i++)
                {
                    _previousDirection[i] = NumOps.FromDouble(reader.ReadDouble());
                }
            }
            else
            {
                _previousDirection = null;
            }
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
    protected override string GenerateGradientCacheKey(IFullModel<T, TInput, TOutput> model, TInput X, TOutput y)
    {
        var baseKey = base.GenerateGradientCacheKey(model, X, y);
        return $"{baseKey}_CG_{_cgOptions.InitialLearningRate}_{_cgOptions.Tolerance}_{_iteration}";
    }
}