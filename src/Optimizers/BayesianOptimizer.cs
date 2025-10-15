global using AiDotNet.GaussianProcesses;

namespace AiDotNet.Optimizers;

/// <summary>
/// Implements a Bayesian Optimization algorithm for efficiently optimizing expensive-to-evaluate functions.
/// </summary>
/// <remarks>
/// <para>
/// Bayesian Optimization is a sequential design strategy for global optimization of black-box functions that
/// doesn't require derivatives. It's particularly useful when the objective function is expensive to evaluate,
/// either computationally or financially.
/// </para>
/// <para><b>For Beginners:</b>
/// Bayesian Optimization is like a smart strategy for finding the highest point of a mountain range in fog:
/// 
/// 1. Instead of blindly climbing everywhere, you:
///    - Take a few initial measurements at different locations
///    - Build a map (model) predicting what the entire mountain range looks like
///    - Use that map to decide where to take your next measurement
///    
/// 2. After each new measurement:
///    - Update your map to be more accurate
///    - Choose a new spot that either:
///      * Looks promising (high peak on your map)
///      * Is very uncertain (where your map might be wrong)
/// 
/// 3. By balancing exploration (checking uncertain areas) and exploitation (checking promising areas),
///    you can find the highest peak with surprisingly few measurements.
/// 
/// This makes Bayesian Optimization perfect for scenarios where each evaluation is:
/// - Expensive (takes hours to run)
/// - Time-consuming (requires physical experiments)
/// - Limited (you only get a few tries)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <typeparam name="TInput">The type of input data for the model.</typeparam>
/// <typeparam name="TOutput">The type of output data for the model.</typeparam>
public class BayesianOptimizer<T, TInput, TOutput> : OptimizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// The options for configuring the Bayesian Optimization algorithm.
    /// </summary>
    private BayesianOptimizerOptions<T, TInput, TOutput> _bayesianOptions = default!;

    /// <summary>
    /// A matrix storing the points that have been sampled during the optimization process.
    /// </summary>
    private Matrix<T> _sampledPoints = default!;

    /// <summary>
    /// A vector storing the corresponding function values for the sampled points.
    /// </summary>
    private Vector<T> _sampledValues = default!;

    /// <summary>
    /// The Gaussian Process model used to approximate the objective function.
    /// </summary>
    private IGaussianProcess<T> _gaussianProcess = default!;

    /// <summary>
    /// Initializes a new instance of the BayesianOptimizer class.
    /// </summary>
    /// <param name="model">The machine learning model to optimize.</param>
    /// <param name="options">The options for configuring the Bayesian Optimization algorithm.</param>
    /// <param name="gaussianProcess">The Gaussian Process model to use for approximating the objective function.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This sets up the Bayesian Optimizer with its initial configuration.
    /// You provide the model you want to optimize, and you can customize various aspects of how it works,
    /// or use default settings. The Gaussian Process is a key component that helps the optimizer make
    /// predictions about unknown points.
    /// </para>
    /// </remarks>
    public BayesianOptimizer(
        IFullModel<T, TInput, TOutput> model,
        BayesianOptimizerOptions<T, TInput, TOutput>? options = null,
        IGaussianProcess<T>? gaussianProcess = null)
        : base(model, options ?? new BayesianOptimizerOptions<T, TInput, TOutput>())
    {
        _bayesianOptions = options ?? new BayesianOptimizerOptions<T, TInput, TOutput>();
        _sampledPoints = Matrix<T>.Empty();
        _sampledValues = Vector<T>.Empty();
        _gaussianProcess = gaussianProcess ?? new StandardGaussianProcess<T>(_bayesianOptions.KernelFunction);

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

        var bestStepData = new OptimizationStepData<T, TInput, TOutput>
        {
            Solution = Model.DeepCopy(),
            FitnessScore = FitnessCalculator.IsHigherScoreBetter ? NumOps.MinValue : NumOps.MaxValue
        };
        var previousStepData = new OptimizationStepData<T, TInput, TOutput>();
        var inputSize = InputHelper<T, TInput>.GetInputSize(inputData.XTrain);

        InitializeAdaptiveParameters();

        // Initial random sampling
        _sampledPoints = new Matrix<T>(_bayesianOptions.InitialSamples, inputSize);
        _sampledValues = new Vector<T>(_bayesianOptions.InitialSamples);

        for (int i = 0; i < _bayesianOptions.InitialSamples; i++)
        {
            // Create a deep copy of the model and randomize its parameters
            var randomSolution = Model.DeepCopy();
            var randomParams = GenerateRandomPoint(inputSize);
            randomSolution = randomSolution.WithParameters(randomParams);

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

        for (int iteration = _bayesianOptions.InitialSamples; iteration < Options.MaxIterations; iteration++)
        {
            // Fit Gaussian Process to observed data
            _gaussianProcess.Fit(_sampledPoints, _sampledValues);

            // Find next point to sample using acquisition function
            var nextPoint = OptimizeAcquisitionFunction(inputSize);
            var currentSolution = Model.DeepCopy().WithParameters(nextPoint);

            var currentStepData = EvaluateSolution(currentSolution, inputData);
            UpdateBestSolution(currentStepData, ref bestStepData);

            // Resize _sampledPoints and _sampledValues
            int newSize = _sampledPoints.Rows + 1;
            var newSampledPoints = new Matrix<T>(newSize, inputSize);
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

        for (int i = 0; i < _bayesianOptions.AcquisitionOptimizationSamples; i++)
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
            point[i] = NumOps.FromDouble(Random.NextDouble() * (_bayesianOptions.UpperBound - _bayesianOptions.LowerBound) + _bayesianOptions.LowerBound);
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

        switch (_bayesianOptions.AcquisitionFunction)
        {
            case AcquisitionFunctionType.UpperConfidenceBound:
                return NumOps.Add(mean, NumOps.Multiply(NumOps.FromDouble(_bayesianOptions.ExplorationFactor), stdDev));
            case AcquisitionFunctionType.ExpectedImprovement:
                var maxObservedValue = _sampledValues.Max();
                var improvement = NumOps.Subtract(mean, maxObservedValue);
                var z = NumOps.Divide(improvement, stdDev);
                var cdf = StatisticsHelper<T>.CalculateNormalCDF(mean, stdDev, z);
                return NumOps.Multiply(improvement, cdf);
            default:
                throw new NotImplementedException("Unsupported acquisition function.");
        }
    }

    /// <summary>
    /// Updates the adaptive parameters based on the current and previous optimization steps.
    /// </summary>
    /// <param name="currentStepData">Data from the current optimization step.</param>
    /// <param name="previousStepData">Data from the previous optimization step.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method adjusts how the algorithm behaves based on whether it's improving
    /// or not. It can change various parameters to help find better solutions more efficiently.
    /// </para>
    /// </remarks>
    protected override void UpdateAdaptiveParameters(OptimizationStepData<T, TInput, TOutput> currentStepData, OptimizationStepData<T, TInput, TOutput> previousStepData)
    {
        // Skip if previous step data is null (first iteration)
        if (previousStepData.Solution == null)
            return;

        base.UpdateAdaptiveParameters(currentStepData, previousStepData);

        // Additional Bayesian-specific parameter updates could be added here if needed
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
            _bayesianOptions = bayesianOptions;
            _gaussianProcess.UpdateKernel(_bayesianOptions.KernelFunction);
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected BayesianOptimizerOptions.");
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
        return _bayesianOptions;
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

            string optionsJson = JsonConvert.SerializeObject(_bayesianOptions);
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
    /// <exception cref="InvalidOperationException">Thrown when deserialization of optimizer options fails.</exception>
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
            _bayesianOptions = JsonConvert.DeserializeObject<BayesianOptimizerOptions<T, TInput, TOutput>>(optionsJson)
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

            _gaussianProcess = new StandardGaussianProcess<T>(_bayesianOptions.KernelFunction);
        }
    }
}