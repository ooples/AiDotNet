using System.Linq;
using Newtonsoft.Json;

namespace AiDotNet.Optimizers;

/// <summary>
/// Implements the Covariance Matrix Adaptation Evolution Strategy (CMA-ES) optimization algorithm.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// CMA-ES is a powerful optimization algorithm for non-linear, non-convex optimization problems.
/// It is particularly effective for problems with up to about 100 dimensions and is known for its
/// robustness and ability to handle complex fitness landscapes.
/// </para>
/// <para><b>For Beginners:</b> CMA-ES is like an advanced search algorithm that tries to find the best solution
/// by learning from previous attempts. It's especially good at solving complex problems where the relationship
/// between inputs and outputs isn't straightforward.
/// </para>
/// </remarks>
public class CMAESOptimizer<T, TInput, TOutput> : OptimizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// The options specific to the CMA-ES optimization algorithm.
    /// </summary>
    private CMAESOptimizerOptions<T, TInput, TOutput> _options;

    /// <summary>
    /// The current population of candidate solutions.
    /// </summary>
    private Matrix<T> _population;

    /// <summary>
    /// The mean of the current distribution.
    /// </summary>
    private Vector<T> _mean;

    /// <summary>
    /// The covariance matrix of the distribution.
    /// </summary>
    private Matrix<T> _C;

    /// <summary>
    /// Evolution path for covariance matrix adaptation.
    /// </summary>
    private Vector<T> _pc;

    /// <summary>
    /// Evolution path for step-size adaptation.
    /// </summary>
    private Vector<T> _ps;

    /// <summary>
    /// The current step size.
    /// </summary>
    private T _sigma;

    /// <summary>
    /// Initializes a new instance of the CMAESOptimizer class.
    /// </summary>
    /// <param name="model">The model to optimize.</param>
    /// <param name="options">The options for configuring the CMA-ES algorithm.</param>
    /// <param name="predictionOptions">Options for prediction statistics.</param>
    /// <param name="modelOptions">Options for model statistics.</param>
    /// <param name="modelEvaluator">The model evaluator to use.</param>
    /// <param name="fitDetector">The fit detector to use.</param>
    /// <param name="fitnessCalculator">The fitness calculator to use.</param>
    /// <param name="modelCache">The model cache to use.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This constructor sets up the CMA-ES optimizer with its initial configuration.
    /// You can customize various aspects of how it works, or use default settings.
    /// </para>
    /// </remarks>
    public CMAESOptimizer(
        IFullModel<T, TInput, TOutput> model,
        CMAESOptimizerOptions<T, TInput, TOutput>? options = null,
        IEngine? engine = null)
        : base(model, options ?? new())
    {
        _options = (CMAESOptimizerOptions<T, TInput, TOutput>)Options;
        _population = Matrix<T>.Empty();
        _mean = Vector<T>.Empty();
        _C = Matrix<T>.Empty();
        _pc = Vector<T>.Empty();
        _ps = Vector<T>.Empty();
        _sigma = NumOps.Zero;

        InitializeAdaptiveParameters();
    }

    /// <summary>
    /// Initializes the adaptive parameters used in the CMA-ES algorithm.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method sets up the initial state for the optimizer,
    /// including the population, mean, covariance matrix, and step size.
    /// </para>
    /// </remarks>
    protected override void InitializeAdaptiveParameters()
    {
        base.InitializeAdaptiveParameters();
        _population = Matrix<T>.Empty();
        _mean = Vector<T>.Empty();
        _C = Matrix<T>.Empty();
        _pc = Vector<T>.Empty();
        _ps = Vector<T>.Empty();
        _sigma = NumOps.FromDouble(_options.InitialStepSize);
    }

    /// <summary>
    /// Performs the main optimization process using the CMA-ES algorithm.
    /// </summary>
    /// <param name="inputData">The input data for the optimization process.</param>
    /// <returns>The result of the optimization process.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the heart of the CMA-ES algorithm. It iteratively improves the solution
    /// by generating new populations, evaluating their fitness, and updating the distribution parameters.
    /// The process continues until it reaches the maximum number of generations or meets the stopping criteria.
    /// </para>
    /// </remarks>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        ValidationHelper<T>.ValidateInputData(inputData);

        var bestStepData = new OptimizationStepData<T, TInput, TOutput>();
        var previousStepData = new OptimizationStepData<T, TInput, TOutput>();

        InitializeAdaptiveParameters();
        int dimensions = InputHelper<T, TInput>.GetInputSize(inputData.XTrain);
        // Always use a deep copy of Model to avoid mutating the original during optimization
        var initialSolution = InitializeRandomSolution(inputData.XTrain);
        _mean = initialSolution.GetParameters();
        _C = Matrix<T>.CreateIdentity(dimensions);
        _pc = new Vector<T>(dimensions);
        _ps = new Vector<T>(dimensions);

        // Keep track of our current best model to use as a template
        var currentBestModel = initialSolution;

        for (int generation = 0; generation < _options.MaxGenerations; generation++)
        {
            var population = GeneratePopulation();

            // Use the current best model as a template for evaluating the population
            var populationResults = EvaluatePopulationWithModels(population, inputData, currentBestModel);
            var fitnessValues = populationResults.Item1;

            // Store the best model from this population for the next iteration
            if (populationResults.Item2 != null)
            {
                currentBestModel = populationResults.Item2;
            }

            UpdateDistribution(population, fitnessValues);

            // Create a new solution with the updated mean parameters
            var currentSolution = currentBestModel.WithParameters(_mean);
            var currentStepData = EvaluateSolution(currentSolution, inputData);

            UpdateBestSolution(currentStepData, ref bestStepData);

            // Update our current best model if this solution is better
            if (NumOps.GreaterThan(currentStepData.FitnessScore, bestStepData.FitnessScore))
            {
                currentBestModel = currentSolution;
            }

            UpdateAdaptiveParameters(currentStepData, previousStepData);

            if (UpdateIterationHistoryAndCheckEarlyStopping(generation, bestStepData))
            {
                break;
            }

            if (NumOps.LessThan(_sigma, NumOps.FromDouble(_options.StopTolerance)))
            {
                break;
            }

            previousStepData = currentStepData;
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    /// <summary>
    /// Generates a new population of candidate solutions.
    /// </summary>
    /// <returns>A matrix representing the new population.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method creates a set of new potential solutions by sampling
    /// from a multivariate normal distribution centered around the current mean.
    /// </para>
    /// </remarks>
    private Matrix<T> GeneratePopulation()
    {
        int dimensions = _mean.Length;
        var population = new Matrix<T>(_options.PopulationSize, dimensions);

        for (int i = 0; i < _options.PopulationSize; i++)
        {
            // === Vectorized Population Generation using IEngine (Phase B: US-GPU-015) ===
            // population[i] = mean + sigma * sample
            var sample = GenerateMultivariateNormalSample(dimensions);
            var scaledSample = (Vector<T>)AiDotNetEngine.Current.Multiply(sample, _sigma);
            var individual = (Vector<T>)AiDotNetEngine.Current.Add(_mean, scaledSample);

            for (int j = 0; j < dimensions; j++)
            {
                population[i, j] = individual[j];
            }
        }

        return population;
    }

    /// <summary>
    /// Generates a sample from a multivariate normal distribution.
    /// </summary>
    /// <param name="dimensions">The number of dimensions for the sample.</param>
    /// <returns>A vector representing the sample.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method creates a random sample that follows a specific
    /// statistical distribution, which is key to how CMA-ES explores the solution space.
    /// </para>
    /// </remarks>
    private Vector<T> GenerateMultivariateNormalSample(int dimensions)
    {
        // Generate a vector of standard normal samples
        var standardNormal = new Vector<T>(dimensions);
        for (int i = 0; i < dimensions; i++)
        {
            standardNormal[i] = NumOps.FromDouble(GenerateStandardNormal());
        }

        if (!_C.IsPositiveDefiniteMatrix())
        {
            // If the matrix is not positive definite, add a small value to the diagonal
            var epsilon = NumOps.FromDouble(1e-6);
            for (int i = 0; i < dimensions; i++)
            {
                _C[i, i] = NumOps.Add(_C[i, i], epsilon);
            }
        }

        // Perform Cholesky decomposition of the covariance matrix
        var choleskyDecomposition = new CholeskyDecomposition<T>(_C);

        // Transform the standard normal samples using the Cholesky decomposition
        var lowerTriangular = choleskyDecomposition.L;

        return lowerTriangular.Multiply(standardNormal);
    }

    /// <summary>
    /// Generates a standard normal random number.
    /// </summary>
    /// <returns>A random number from a standard normal distribution.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method creates a single random number that follows
    /// a standard normal distribution (bell curve centered at 0 with a standard deviation of 1).
    /// </para>
    /// </remarks>
    private double GenerateStandardNormal()
    {
        double u1 = 1.0 - Random.NextDouble();
        double u2 = 1.0 - Random.NextDouble();

        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
    }

    /// <summary>
    /// Evaluates the fitness of each individual in the population and returns the best model.
    /// </summary>
    /// <param name="population">The population to evaluate.</param>
    /// <param name="inputData">The input data for evaluation.</param>
    /// <param name="templateModel">A template model to use for creating new models with updated parameters.</param>
    /// <returns>A tuple containing: 1) A vector of fitness scores for the population, 2) The best model from this population.</returns>
    private (Vector<T>, IFullModel<T, TInput, TOutput>?) EvaluatePopulationWithModels(
        Matrix<T> population,
        OptimizationInputData<T, TInput, TOutput> inputData,
        IFullModel<T, TInput, TOutput> templateModel)
    {
        var fitnessValues = new Vector<T>(population.Rows);
        IFullModel<T, TInput, TOutput>? bestModel = null;
        T bestFitness = NumOps.MinValue;

        for (int i = 0; i < population.Rows; i++)
        {
            // Create a new solution with the population member's parameters
            var solution = templateModel.WithParameters(population.GetRow(i));
            var stepData = EvaluateSolution(solution, inputData);
            fitnessValues[i] = stepData.FitnessScore;

            // Keep track of the best model in this population
            if (bestModel == null || NumOps.GreaterThan(stepData.FitnessScore, bestFitness))
            {
                bestModel = solution;
                bestFitness = stepData.FitnessScore;
            }
        }

        return (fitnessValues, bestModel);
    }

    /// <summary>
    /// Updates the distribution parameters of the CMA-ES algorithm based on the current population and their fitness values.
    /// </summary>
    /// <param name="population">The current population of candidate solutions.</param>
    /// <param name="fitnessValues">The fitness values corresponding to each individual in the population.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method is the core of the CMA-ES algorithm. It adjusts the search
    /// distribution based on the performance of the current population. This allows the algorithm to
    /// adapt its search strategy as it progresses, focusing on more promising areas of the solution space.
    /// </para>
    /// </remarks>
    private void UpdateDistribution(Matrix<T> population, Vector<T> fitnessValues)
    {
        int dimensions = _mean.Length;
        int lambda = _options.PopulationSize;
        int mu = lambda / 2;

        // Sort and select the best individuals
        // Create index-fitness pairs and sort by fitness descending
        var indexedFitness = new List<(int index, T fitness)>();
        for (int i = 0; i < lambda; i++)
        {
            indexedFitness.Add((i, fitnessValues[i]));
        }

        // Sort descending by fitness (best first)
        indexedFitness.Sort((a, b) =>
        {
            if (NumOps.GreaterThan(a.fitness, b.fitness)) return -1;
            if (NumOps.LessThan(a.fitness, b.fitness)) return 1;
            return 0;
        });
        var selectedPopulation = new Matrix<T>(mu, dimensions);
        for (int i = 0; i < mu; i++)
        {
            int sourceIndex = indexedFitness[i].index;
            for (int j = 0; j < dimensions; j++)
            {
                selectedPopulation[i, j] = population[sourceIndex, j];
            }
        }

        // Calculate weights - vectorized
        // Create vector of indices [0, 1, 2, ..., mu-1]
        var indices = new Vector<T>(mu);
        for (int i = 0; i < mu; i++)
        {
            indices[i] = NumOps.FromDouble(i);
        }

        // weights[i] = (mu + 0.5) - log(mu + 0.5 + i)
        var muPlusHalf = AiDotNetEngine.Current.Fill<T>(mu, NumOps.FromDouble(mu + 0.5));
        var indexPlusMu = (Vector<T>)AiDotNetEngine.Current.Add(indices, muPlusHalf);
        var logValues = (Vector<T>)AiDotNetEngine.Current.Log(indexPlusMu);
        var weights = (Vector<T>)AiDotNetEngine.Current.Subtract(muPlusHalf, logValues);

        // Normalize weights
        T sumWeights = AiDotNetEngine.Current.Sum(weights);
        weights = weights.Divide(sumWeights);

        // Calculate effective mu - vectorized
        // muEff = 1 / sum(weights^2)
        var weightsSquared = new Vector<T>(weights.Length);
        for (int i = 0; i < weights.Length; i++)
        {
            weightsSquared[i] = NumOps.Square(weights[i]);
        }
        T sumSquaredWeights = AiDotNetEngine.Current.Sum(weightsSquared);
        T muEff = NumOps.Divide(NumOps.One, sumSquaredWeights);

        // Update mean
        var oldMean = _mean;
        _mean = selectedPopulation.Transpose().Multiply(weights);

        // Calculate learning rates
        T c1 = NumOps.Divide(NumOps.FromDouble(2.0), NumOps.Add(NumOps.FromDouble(Math.Pow(dimensions + 1.3, 2)), muEff));
        T cmu = MathHelper.Min(
            NumOps.Subtract(NumOps.One, c1),
            NumOps.Divide(
                NumOps.Multiply(NumOps.FromDouble(2), NumOps.Add(NumOps.Subtract(muEff, NumOps.FromDouble(2)), NumOps.Divide(NumOps.One, muEff))),
                NumOps.Add(NumOps.FromDouble(Math.Pow(dimensions + 2, 2)), muEff)
            )
        );
        T cc = NumOps.Divide(
            NumOps.Add(NumOps.FromDouble(4), NumOps.Divide(muEff, NumOps.FromDouble(dimensions))),
            NumOps.Add(NumOps.FromDouble(dimensions + 4), NumOps.Multiply(NumOps.FromDouble(2), NumOps.Divide(muEff, NumOps.FromDouble(dimensions))))
        );
        T cs = NumOps.Divide(
            NumOps.Add(muEff, NumOps.FromDouble(2)),
            NumOps.Add(NumOps.FromDouble(dimensions), NumOps.Add(muEff, NumOps.FromDouble(5)))
        );

        // Update evolution paths
        var y = _mean.Subtract(oldMean).Divide(_sigma);
        _ps = _ps.Multiply(NumOps.Subtract(NumOps.One, cs)).Add(
            y.Multiply(NumOps.Sqrt(NumOps.Multiply(cs, NumOps.Subtract(NumOps.FromDouble(2), cs)))).Multiply(NumOps.Sqrt(muEff)));

        T hsig = NumOps.LessThan(
            NumOps.Divide(
                _ps.Norm(),
                NumOps.Sqrt(NumOps.Subtract(NumOps.One, NumOps.Power(NumOps.Subtract(NumOps.One, cs), NumOps.FromDouble(2 * _options.MaxGenerations))))
            ),
            NumOps.FromDouble(1.4 + 2 / (dimensions + 1.0))
        ) ? NumOps.One : NumOps.Zero;

        _pc = _pc.Multiply(NumOps.Subtract(NumOps.One, cc)).Add(
            y.Multiply(NumOps.Sqrt(NumOps.Multiply(cc, NumOps.Subtract(NumOps.FromDouble(2), cc)))).Multiply(NumOps.Sqrt(muEff)).Multiply(hsig));

        // Update covariance matrix
        var artmp = selectedPopulation.Subtract(_mean.Repeat(mu).Reshape(mu, dimensions)).Divide(_sigma);
        _C = _C.Multiply(NumOps.Subtract(NumOps.One, NumOps.Add(c1, cmu)))
            .Add(_pc.OuterProduct(_pc).Multiply(c1))
            .Add(artmp.Transpose().Multiply(weights.CreateDiagonal()).Multiply(artmp).Multiply(cmu));

        // Update step size
        T damps = NumOps.Add(NumOps.One, NumOps.Multiply(NumOps.FromDouble(2),
            MathHelper.Max(NumOps.Zero, NumOps.Subtract(NumOps.Sqrt(NumOps.Divide(NumOps.Subtract(muEff, NumOps.One), NumOps.FromDouble(dimensions + 1))), NumOps.One))
        ));
        damps = NumOps.Add(damps, cs);
        _sigma = NumOps.Multiply(_sigma, NumOps.Exp(NumOps.Multiply(
            NumOps.Divide(cs, damps),
            NumOps.Subtract(
                NumOps.Divide(
                    _ps.Norm(),
                    NumOps.Sqrt(NumOps.FromDouble(dimensions))
                ),
                NumOps.One
            )
        )));
    }

    /// <summary>
    /// Updates the options for the CMA-ES optimizer.
    /// </summary>
    /// <param name="options">The new options to be set.</param>
    /// <exception cref="ArgumentException">Thrown when the provided options are not of the correct type.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method allows you to change the settings of the CMA-ES optimizer during runtime.
    /// It checks to make sure you're providing the right kind of options specific to the CMA-ES algorithm.
    /// </para>
    /// </remarks>
    protected override void UpdateOptions(OptimizationAlgorithmOptions<T, TInput, TOutput> options)
    {
        if (options is CMAESOptimizerOptions<T, TInput, TOutput> cmaesOptions)
        {
            _options = cmaesOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected CMAESOptimizerOptions.");
        }
    }

    /// <summary>
    /// Gets the current options of the CMA-ES optimizer.
    /// </summary>
    /// <returns>The current optimization algorithm options.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method allows you to retrieve the current settings of the CMA-ES optimizer.
    /// You can use this to check or save the current configuration.
    /// </para>
    /// </remarks>
    public override OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions()
    {
        return _options;
    }

    /// <summary>
    /// Serializes the current state of the CMA-ES optimizer into a byte array.
    /// </summary>
    /// <returns>A byte array representing the serialized state of the optimizer.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method saves the current state of the optimizer into a format
    /// that can be stored or transmitted. This is useful for saving progress or sharing the optimizer's state.
    /// </para>
    /// </remarks>
    public override byte[] Serialize()
    {
        using MemoryStream ms = new MemoryStream();
        using BinaryWriter writer = new BinaryWriter(ms);

        byte[] baseData = base.Serialize();
        writer.Write(baseData.Length);
        writer.Write(baseData);

        string optionsJson = JsonConvert.SerializeObject(_options);
        writer.Write(optionsJson);

        // Serialize CMA-ES specific data
        SerializationHelper<T>.SerializeMatrix(writer, _population);
        SerializationHelper<T>.SerializeVector(writer, _mean);
        SerializationHelper<T>.SerializeMatrix(writer, _C);
        SerializationHelper<T>.SerializeVector(writer, _pc);
        SerializationHelper<T>.SerializeVector(writer, _ps);
        SerializationHelper<T>.WriteValue(writer, _sigma);

        return ms.ToArray();
    }

    /// <summary>
    /// Deserializes a byte array to restore the state of the CMA-ES optimizer.
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
        using MemoryStream ms = new MemoryStream(data);
        using BinaryReader reader = new BinaryReader(ms);

        int baseDataLength = reader.ReadInt32();
        byte[] baseData = reader.ReadBytes(baseDataLength);
        base.Deserialize(baseData);

        string optionsJson = reader.ReadString();
        _options = JsonConvert.DeserializeObject<CMAESOptimizerOptions<T, TInput, TOutput>>(optionsJson)
            ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");

        // Deserialize CMA-ES specific data
        _population = SerializationHelper<T>.DeserializeMatrix(reader);
        _mean = SerializationHelper<T>.DeserializeVector(reader);
        _C = SerializationHelper<T>.DeserializeMatrix(reader);
        _pc = SerializationHelper<T>.DeserializeVector(reader);
        _ps = SerializationHelper<T>.DeserializeVector(reader);
        _sigma = SerializationHelper<T>.ReadValue(reader);
    }
}
