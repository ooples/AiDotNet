global using AiDotNet.GaussianProcesses;
using Newtonsoft.Json;

namespace AiDotNet.Optimizers;

/// <summary>
/// Represents a Bayesian Optimizer for optimization problems.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Bayesian Optimization is a powerful technique for optimizing black-box functions that are expensive to evaluate.
/// It uses a probabilistic model to make predictions about the function's behavior and decides where to sample next.
/// </para>
/// <para><b>For Beginners:</b> Think of this optimizer as a smart guessing game. It tries to find the best solution
/// by making educated guesses based on what it has learned from previous attempts. It's particularly useful when
/// each guess is time-consuming or expensive to evaluate.
/// </para>
/// </remarks>
public class BayesianOptimizer<T, TInput, TOutput> : OptimizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// The options for configuring the Bayesian Optimization algorithm.
    /// </summary>
    private BayesianOptimizerOptions<T, TInput, TOutput> _options;

    /// <summary>
    /// A matrix storing the points that have been sampled during the optimization process.
    /// </summary>
    private Matrix<T> _sampledPoints;

    /// <summary>
    /// A vector storing the corresponding function values for the sampled points.
    /// </summary>
    private Vector<T> _sampledValues;

    /// <summary>
    /// The Gaussian Process model used to approximate the objective function.
    /// </summary>
    private IGaussianProcess<T> _gaussianProcess;

    /// <summary>
    /// Initializes a new instance of the BayesianOptimizer class.
    /// </summary>
    /// <param name="model">The model to optimize.</param>
    /// <param name="options">The options for configuring the Bayesian Optimization algorithm.</param>
    /// <param name="gaussianProcess"> The Gaussian Process model to use for approximating the objective function.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This sets up the Bayesian Optimizer with its initial configuration.
    /// You can customize various aspects of how it works, or use default settings. The Gaussian Process
    /// is a key component that helps the optimizer make predictions about unknown points.
    /// </para>
    /// </remarks>
    public BayesianOptimizer(
        IFullModel<T, TInput, TOutput> model,
        BayesianOptimizerOptions<T, TInput, TOutput>? options = null,
        IGaussianProcess<T>? gaussianProcess = null,
        IEngine? engine = null)
        : base(model, options ?? new())
    {
        _options = options ?? new BayesianOptimizerOptions<T, TInput, TOutput>();
        _sampledPoints = Matrix<T>.Empty();
        _sampledValues = Vector<T>.Empty();
        _gaussianProcess = gaussianProcess ?? new StandardGaussianProcess<T>(_options.KernelFunction);

        InitializeAdaptiveParameters();
    }

    /// <summary>
    /// Initializes the adaptive parameters used in the Bayesian Optimization algorithm.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method sets up the initial state for the optimizer.
    /// It clears any previously sampled points and their corresponding values, preparing for a fresh optimization run.
    /// </para>
    /// </remarks>
    protected override void InitializeAdaptiveParameters()
    {
        base.InitializeAdaptiveParameters();

        _sampledPoints = Matrix<T>.Empty();
        _sampledValues = Vector<T>.Empty();
    }

    /// <summary>
    /// Performs the main optimization process using the Bayesian Optimization algorithm.
    /// </summary>
    /// <param name="inputData">The input data for the optimization process.</param>
    /// <returns>The result of the optimization process.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the heart of the Bayesian Optimization algorithm. It starts by taking some
    /// random samples, then iteratively uses what it has learned to make smart guesses about where the best solution
    /// might be. It keeps doing this until it finds a good solution or runs out of allowed attempts.
    /// </para>
    /// </remarks>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        ValidationHelper<T>.ValidateInputData(inputData);

        var bestStepData = new OptimizationStepData<T, TInput, TOutput>();
        var previousStepData = new OptimizationStepData<T, TInput, TOutput>();

        InitializeAdaptiveParameters();

        // Get the parameter count from a random solution (includes coefficients + intercept)
        // This is different from inputSize which is just the feature count
        var firstSolution = InitializeRandomSolution(inputData.XTrain);
        int paramCount = firstSolution.ParameterCount;

        // Initial random sampling - use paramCount (model parameters), not inputSize (features)
        _sampledPoints = new Matrix<T>(_options.InitialSamples, paramCount);
        _sampledValues = new Vector<T>(_options.InitialSamples);

        // Evaluate the first solution we already created
        var firstStepData = EvaluateSolution(firstSolution, inputData);
        UpdateBestSolution(firstStepData, ref bestStepData);
        var firstParams = firstSolution.GetParameters();
        for (int j = 0; j < firstParams.Length; j++)
        {
            _sampledPoints[0, j] = firstParams[j];
        }
        _sampledValues[0] = firstStepData.FitnessScore;

        // Continue with remaining initial samples
        for (int i = 1; i < _options.InitialSamples; i++)
        {
            var randomSolution = InitializeRandomSolution(inputData.XTrain);
            var stepData = EvaluateSolution(randomSolution, inputData);
            UpdateBestSolution(stepData, ref bestStepData);
            var parameters = randomSolution.GetParameters();

            // Set values in the matrix and vector using indexing
            for (int j = 0; j < parameters.Length; j++)
            {
                _sampledPoints[i, j] = parameters[j];
            }

            _sampledValues[i] = stepData.FitnessScore;
        }

        for (int iteration = _options.InitialSamples; iteration < _options.MaxIterations; iteration++)
        {
            // Fit Gaussian Process to observed data
            _gaussianProcess.Fit(_sampledPoints, _sampledValues);

            // Find next point to sample using acquisition function - use paramCount, not inputSize
            var nextPoint = OptimizeAcquisitionFunction(paramCount);
            var baseSolution = InitializeRandomSolution(inputData.XTrain);
            var currentSolution = baseSolution.WithParameters(nextPoint);

            var currentStepData = EvaluateSolution(currentSolution, inputData);
            UpdateBestSolution(currentStepData, ref bestStepData);

            // Resize _sampledPoints and _sampledValues
            int newSize = _sampledPoints.Rows + 1;
            // Use existing matrix column count (paramCount) for consistency
            var newSampledPoints = new Matrix<T>(newSize, _sampledPoints.Columns);
            var newSampledValues = new Vector<T>(newSize);

            // Copy existing data
            for (int i = 0; i < _sampledPoints.Rows; i++)
            {
                for (int j = 0; j < _sampledPoints.Columns; j++)
                {
                    newSampledPoints[i, j] = _sampledPoints[i, j];
                }
                newSampledValues[i] = _sampledValues[i];
            }

            // Add new point and value
            for (int j = 0; j < nextPoint.Length; j++)
            {
                newSampledPoints[newSize - 1, j] = nextPoint[j];
            }
            newSampledValues[newSize - 1] = currentStepData.FitnessScore;

            // Replace old data with new data
            _sampledPoints = newSampledPoints;
            _sampledValues = newSampledValues;

            UpdateAdaptiveParameters(currentStepData, previousStepData);

            if (UpdateIterationHistoryAndCheckEarlyStopping(iteration, bestStepData))
            {
                break;
            }

            previousStepData = currentStepData;
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    /// <summary>
    /// Optimizes the acquisition function to determine the next point to sample.
    /// </summary>
    /// <param name="dimensions">The number of dimensions in the optimization space.</param>
    /// <returns>The next point to sample.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method decides where to look next for a good solution. It does this by
    /// balancing between exploring new areas and focusing on areas that seem promising based on what we've seen so far.
    /// </para>
    /// </remarks>
    private Vector<T> OptimizeAcquisitionFunction(int dimensions)
    {
        Vector<T> bestPoint = Vector<T>.Empty();
        T bestValue = NumOps.MinValue;

        for (int i = 0; i < _options.AcquisitionOptimizationSamples; i++)
        {
            var candidatePoint = GenerateRandomPoint(dimensions);
            var acquisitionValue = CalculateAcquisitionFunction(candidatePoint);

            if (NumOps.GreaterThan(acquisitionValue, bestValue))
            {
                bestPoint = candidatePoint;
                bestValue = acquisitionValue;
            }
        }

        return bestPoint;
    }

    /// <summary>
    /// Generates a random point within the specified bounds of the optimization space.
    /// </summary>
    /// <param name="dimensions">The number of dimensions in the optimization space.</param>
    /// <returns>A randomly generated point.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method creates a random guess within the allowed range. It's used to explore
    /// the solution space, especially at the beginning when we don't have much information.
    /// </para>
    /// </remarks>
    private Vector<T> GenerateRandomPoint(int dimensions)
    {
        var point = new T[dimensions];
        for (int i = 0; i < dimensions; i++)
        {
            point[i] = NumOps.FromDouble(Random.NextDouble() * (_options.UpperBound - _options.LowerBound) + _options.LowerBound);
        }

        return new Vector<T>(point);
    }

    /// <summary>
    /// Calculates the value of the acquisition function for a given point.
    /// </summary>
    /// <param name="point">The point to evaluate.</param>
    /// <returns>The value of the acquisition function at the given point.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method helps decide how promising a particular point is. It balances between
    /// picking points that the model thinks will be good (exploitation) and points where the model is uncertain
    /// (exploration).
    /// </para>
    /// </remarks>
    private T CalculateAcquisitionFunction(Vector<T> point)
    {
        var (mean, variance) = _gaussianProcess.Predict(point);
        var stdDev = NumOps.Sqrt(variance);

        switch (_options.AcquisitionFunction)
        {
            case AcquisitionFunctionType.UpperConfidenceBound:
                return NumOps.Add(mean, NumOps.Multiply(NumOps.FromDouble(_options.ExplorationFactor), stdDev));
            case AcquisitionFunctionType.ExpectedImprovement:
                var bestObserved = _options.IsMaximization ? _sampledValues.Max() : _sampledValues.Min();
                var improvement = _options.IsMaximization ? NumOps.Subtract(mean, bestObserved) : NumOps.Subtract(bestObserved, mean);
                var z = NumOps.Divide(improvement, stdDev);
                var cdf = StatisticsHelper<T>.CalculateNormalCDF(mean, stdDev, z);
                return NumOps.Multiply(improvement, cdf);
            default:
                throw new InvalidOperationException($"Unsupported acquisition function: {_options.AcquisitionFunction}");
        }
    }

    /// <summary>
    /// Updates the options for the Bayesian Optimization algorithm.
    /// </summary>
    /// <param name="options">The new options to be set.</param>
    /// <exception cref="ArgumentException">Thrown when the provided options are not of the correct type.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method allows you to change how the Bayesian Optimization algorithm behaves
    /// by updating its settings. It checks to make sure you're providing the right kind of settings.
    /// </para>
    /// </remarks>
    protected override void UpdateOptions(OptimizationAlgorithmOptions<T, TInput, TOutput> options)
    {
        if (options is BayesianOptimizerOptions<T, TInput, TOutput> bayesianOptions)
        {
            _options = bayesianOptions;
            _gaussianProcess.UpdateKernel(_options.KernelFunction);
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected BayesianOptimizerOptions.", nameof(options));
        }
    }

    /// <summary>
    /// Gets the current options for the Bayesian Optimization algorithm.
    /// </summary>
    /// <returns>The current optimization options.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method lets you see what settings the Bayesian Optimization algorithm is currently using.</para>
    /// </remarks>
    public override OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions()
    {
        return _options;
    }

    /// <summary>
    /// Converts the current state of the optimizer into a byte array for storage or transmission.
    /// </summary>
    /// <returns>A byte array representing the serialized state of the optimizer.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method takes all the important information about the current state
    /// of the Bayesian Optimizer and turns it into a format that can be easily saved or sent to another computer.
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

            // Serialize _sampledPoints
            writer.Write(_sampledPoints.Rows);
            writer.Write(_sampledPoints.Columns);
            for (int i = 0; i < _sampledPoints.Rows; i++)
            {
                for (int j = 0; j < _sampledPoints.Columns; j++)
                {
                    writer.Write(Convert.ToDouble(_sampledPoints[i, j]));
                }
            }

            // Serialize _sampledValues
            writer.Write(_sampledValues.Length);
            for (int i = 0; i < _sampledValues.Length; i++)
            {
                writer.Write(Convert.ToDouble(_sampledValues[i]));
            }

            return ms.ToArray();
        }
    }

    /// <summary>
    /// Restores the state of the optimizer from a byte array.
    /// </summary>
    /// <param name="data">The byte array containing the serialized state of the optimizer.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method takes a saved state of the Bayesian Optimizer (in the form of a byte array)
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
            _options = JsonConvert.DeserializeObject<BayesianOptimizerOptions<T, TInput, TOutput>>(optionsJson)
                ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");

            // Deserialize _sampledPoints
            int rows = reader.ReadInt32();
            int columns = reader.ReadInt32();
            _sampledPoints = new Matrix<T>(rows, columns);
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < columns; j++)
                {
                    _sampledPoints[i, j] = NumOps.FromDouble(reader.ReadDouble());
                }
            }

            // Deserialize _sampledValues
            int valueCount = reader.ReadInt32();
            _sampledValues = new Vector<T>(valueCount);
            for (int i = 0; i < valueCount; i++)
            {
                _sampledValues[i] = NumOps.FromDouble(reader.ReadDouble());
            }

            _gaussianProcess = new StandardGaussianProcess<T>(_options.KernelFunction);
        }
    }
}

