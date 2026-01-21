using Newtonsoft.Json;

namespace AiDotNet.Optimizers;

/// <summary>
/// Implements the Ant Colony Optimization algorithm for solving optimization problems.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The type of input data structure.</typeparam>
/// <typeparam name="TOutput">The type of output data structure.</typeparam>
/// <remarks>
/// <para>
/// Ant Colony Optimization is inspired by the behavior of ants in finding paths between their colony and food sources.
/// It uses virtual "ants" to explore the solution space and find optimal solutions.
/// </para>
/// <para><b>For Beginners:</b> Think of this algorithm as a group of ants searching for the best path to food.
/// Each ant leaves a trail (pheromone) that other ants can follow. Over time, the best paths get stronger trails,
/// leading to better solutions.
/// </para>
/// </remarks>
public class AntColonyOptimizer<T, TInput, TOutput> : OptimizerBase<T, TInput, TOutput>
{
    /// <summary>
    /// Options specific to the Ant Colony Optimization algorithm.
    /// </summary>
    private AntColonyOptimizationOptions<T, TInput, TOutput> _antColonyOptions;

    /// <summary>
    /// The current rate at which pheromone evaporates from the trails.
    /// </summary>
    private T _currentPheromoneEvaporationRate;

    /// <summary>
    /// The current intensity of pheromone deposited by ants.
    /// </summary>
    private T _currentPheromoneIntensity;

    /// <summary>
    /// Initializes a new instance of the AntColonyOptimizer class.
    /// </summary>
    /// <param name="model">The model to be optimized.</param>
    /// <param name="options">The options for configuring the Ant Colony Optimization algorithm.</param>
    /// <param name="engine">The computation engine (CPU or GPU) for vectorized operations.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This sets up the Ant Colony Optimizer with its initial configuration.
    /// You can customize various aspects of how it works, or use default settings.
    /// </para>
    /// </remarks>
    public AntColonyOptimizer(
        IFullModel<T, TInput, TOutput> model,
        AntColonyOptimizationOptions<T, TInput, TOutput>? options = null,
        IEngine? engine = null)
        : base(model, options ?? new())
    {
        _antColonyOptions = options ?? new AntColonyOptimizationOptions<T, TInput, TOutput>();
        _currentPheromoneEvaporationRate = NumOps.Zero;
        _currentPheromoneIntensity = NumOps.Zero;

        InitializeAdaptiveParameters();
    }

    /// <summary>
    /// Initializes the adaptive parameters used in the Ant Colony Optimization algorithm.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method sets up the initial values for how quickly pheromones evaporate
    /// and how strongly new pheromones are deposited. These values will change as the algorithm runs to help
    /// find better solutions.
    /// </para>
    /// </remarks>
    protected override void InitializeAdaptiveParameters()
    {
        _currentPheromoneEvaporationRate = NumOps.FromDouble(_antColonyOptions.InitialPheromoneEvaporationRate);
        _currentPheromoneIntensity = NumOps.FromDouble(_antColonyOptions.InitialPheromoneIntensity);
    }

    /// <summary>
    /// Updates the adaptive parameters based on the current and previous optimization steps.
    /// </summary>
    /// <param name="currentStepData">Data from the current optimization step.</param>
    /// <param name="previousStepData">Data from the previous optimization step.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method adjusts how the algorithm behaves based on whether it's improving
    /// or not. It changes how quickly pheromones evaporate and how strongly new ones are deposited.
    /// </para>
    /// </remarks>
    protected override void UpdateAdaptiveParameters(OptimizationStepData<T, TInput, TOutput> currentStepData, OptimizationStepData<T, TInput, TOutput> previousStepData)
    {
        base.UpdateAdaptiveParameters(currentStepData, previousStepData);

        UpdatePheromoneEvaporationRate(currentStepData, previousStepData);
        UpdatePheromoneIntensity(currentStepData, previousStepData);
    }

    /// <summary>
    /// Updates the pheromone evaporation rate based on the current and previous optimization steps.
    /// </summary>
    /// <param name="currentStepData">Data from the current optimization step.</param>
    /// <param name="previousStepData">Data from the previous optimization step.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method changes how quickly pheromones evaporate. If the algorithm is
    /// improving, it might increase the evaporation rate to focus more on new solutions. If it's not improving,
    /// it might decrease the rate to explore more widely.
    /// </para>
    /// </remarks>
    private void UpdatePheromoneEvaporationRate(OptimizationStepData<T, TInput, TOutput> currentStepData, OptimizationStepData<T, TInput, TOutput> previousStepData)
    {
        double currentRate = Convert.ToDouble(_currentPheromoneEvaporationRate);

        if (NumOps.GreaterThan(currentStepData.FitnessScore, previousStepData.FitnessScore))
        {
            currentRate *= _antColonyOptions.PheromoneEvaporationRateIncrease;
        }
        else
        {
            currentRate *= _antColonyOptions.PheromoneEvaporationRateDecay;
        }

        currentRate = Math.Max(_antColonyOptions.MinPheromoneEvaporationRate, Math.Min(_antColonyOptions.MaxPheromoneEvaporationRate, currentRate));
        _currentPheromoneEvaporationRate = NumOps.FromDouble(currentRate);
    }

    /// <summary>
    /// Updates the pheromone intensity based on the current and previous optimization steps.
    /// </summary>
    /// <param name="currentStepData">Data from the current optimization step.</param>
    /// <param name="previousStepData">Data from the previous optimization step.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method adjusts how strongly new pheromones are deposited. If the algorithm
    /// is improving, it might increase the intensity to reinforce good paths. If it's not improving, it might
    /// decrease the intensity to allow for more exploration.
    /// </para>
    /// </remarks>
    private void UpdatePheromoneIntensity(OptimizationStepData<T, TInput, TOutput> currentStepData, OptimizationStepData<T, TInput, TOutput> previousStepData)
    {
        double currentIntensity = Convert.ToDouble(_currentPheromoneIntensity);

        if (NumOps.GreaterThan(currentStepData.FitnessScore, previousStepData.FitnessScore))
        {
            currentIntensity *= _antColonyOptions.PheromoneIntensityIncrease;
        }
        else
        {
            currentIntensity *= _antColonyOptions.PheromoneIntensityDecay;
        }

        currentIntensity = Math.Max(_antColonyOptions.MinPheromoneIntensity, Math.Min(_antColonyOptions.MaxPheromoneIntensity, currentIntensity));
        _currentPheromoneIntensity = NumOps.FromDouble(currentIntensity);
    }

    /// <summary>
    /// Performs the main optimization process using the Ant Colony Optimization algorithm.
    /// </summary>
    /// <param name="inputData">The input data for the optimization process.</param>
    /// <returns>The result of the optimization process.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the heart of the Ant Colony Optimization algorithm. It simulates ants
    /// exploring different paths (solutions) over multiple iterations. Each ant builds a solution, then the best
    /// solutions are used to update the pheromone trails. This process repeats until a good solution is found or
    /// the maximum number of iterations is reached.
    /// </para>
    /// </remarks>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        int dimensions = InputHelper<T, TInput>.GetInputSize(inputData.XTrain);
        var pheromones = InitializePheromones(dimensions);
        var bestStepData = new OptimizationStepData<T, TInput, TOutput>();
        var prevStepData = new OptimizationStepData<T, TInput, TOutput>();
        var currentStepData = new OptimizationStepData<T, TInput, TOutput>();

        for (int iteration = 0; iteration < Options.MaxIterations; iteration++)
        {
            var solutions = new List<IFullModel<T, TInput, TOutput>>();

            for (int ant = 0; ant < _antColonyOptions.AntCount; ant++)
            {
                var solution = ConstructSolution(pheromones, inputData.XTrain);
                solutions.Add(solution);

                currentStepData = EvaluateSolution(solution, inputData);
                FitnessList.Add(currentStepData.FitnessScore);
                UpdateBestSolution(currentStepData, ref bestStepData);
            }

            // Update pheromones
            UpdatePheromones(pheromones, solutions);

            // Update adaptive parameters
            UpdateAdaptiveParameters(currentStepData, prevStepData);

            if (UpdateIterationHistoryAndCheckEarlyStopping(iteration, bestStepData))
            {
                break; // Early stopping criteria met, exit the loop
            }

            prevStepData = currentStepData;
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    /// <summary>
    /// Initializes the pheromone matrix for the Ant Colony Optimization algorithm.
    /// </summary>
    /// <param name="dimensions">The number of dimensions (features) in the problem space.</param>
    /// <returns>A matrix representing the initial pheromone levels between features.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method creates a starting point for the ants' trails. It sets up a grid
    /// where each cell represents how attractive it is to go from one feature to another. Initially, all paths
    /// are equally attractive.</para>
    /// </remarks>
    private Matrix<T> InitializePheromones(int dimensions)
    {
        var pheromones = new Matrix<T>(dimensions, dimensions);
        for (int i = 0; i < dimensions; i++)
        {
            for (int j = 0; j < dimensions; j++)
            {
                pheromones[i, j] = NumOps.One;
            }
        }

        return pheromones;
    }

    /// <summary>
    /// Constructs a solution (model) based on the current pheromone levels and input data.
    /// </summary>
    /// <param name="pheromones">The current pheromone matrix.</param>
    /// <param name="xTrain">The input training data.</param>
    /// <returns>A model representing a potential solution.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method simulates an ant's journey to build a solution. It chooses features
    /// to include in the model based on the pheromone levels and the input data.</para>
    /// </remarks>
    private IFullModel<T, TInput, TOutput> ConstructSolution(Matrix<T> pheromones, TInput xTrain)
    {
        var dimensions = InputHelper<T, TInput>.GetInputSize(xTrain);

        // Create a base model with InitializeRandomSolution
        var model = InitializeRandomSolution(xTrain);
        // Parameters should match the model's expected parameter count (features + intercept)
        int paramCount = model.ParameterCount;
        var parameters = new Vector<T>(paramCount);

        var visited = new bool[dimensions];
        int current = Random.Next(dimensions);
        visited[current] = true;

        for (int i = 0; i < dimensions; i++)
        {
            int next = SelectNextFeature(current, pheromones, xTrain, visited);
            if (next == -1) break; // No more unvisited features

            var coefficient = NumOps.Multiply(
                _currentPheromoneIntensity,
                InputHelper<T, TInput>.GetElement(xTrain, current, next)
            );

            parameters[next] = coefficient;

            visited[next] = true;
            current = next;
        }

        // Create a new model with updated parameters
        return model.WithParameters(parameters);
    }

    /// <summary>
    /// Selects the next feature to be included in the solution based on pheromone levels and heuristics.
    /// </summary>
    /// <param name="current">The current feature index.</param>
    /// <param name="pheromones">The pheromone matrix.</param>
    /// <param name="xTrain">The input training data.</param>
    /// <param name="visited">An array indicating which features have already been visited.</param>
    /// <returns>The index of the next selected feature, or -1 if no unvisited features remain.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method decides which feature the "ant" should choose next when building
    /// its solution. It uses a combination of pheromone levels (how good this path was in previous iterations)
    /// and a heuristic (a guess at how good this feature might be) to make a weighted random choice.</para>
    /// </remarks>
    private int SelectNextFeature(int current, Matrix<T> pheromones, TInput xTrain, bool[] visited)
    {
        int dimensions = InputHelper<T, TInput>.GetInputSize(xTrain);
        var probabilities = new Vector<T>(dimensions);
        T total = NumOps.Zero;

        for (int i = 0; i < dimensions; i++)
        {
            if (!visited[i])
            {
                T pheromone = pheromones[current, i];
                T element = InputHelper<T, TInput>.GetElement(xTrain, current, i);
                T heuristic = NumOps.Divide(NumOps.One, NumOps.Add(NumOps.One, NumOps.Abs(element)));
                T probability = NumOps.Multiply(
                    NumOps.Power(pheromone, NumOps.FromDouble(1 - Convert.ToDouble(_currentPheromoneEvaporationRate))),
                    NumOps.Power(heuristic, NumOps.FromDouble(_antColonyOptions.Beta))
                );
                probabilities[i] = probability;
                total = NumOps.Add(total, probability);
            }
        }

        // If all features are visited or total probability is zero, return -1
        if (NumOps.Equals(total, NumOps.Zero))
        {
            return -1;
        }

        T random = NumOps.Multiply(NumOps.FromDouble(Random.NextDouble()), total);
        T sum = NumOps.Zero;
        for (int i = 0; i < dimensions; i++)
        {
            if (!visited[i])
            {
                sum = NumOps.Add(sum, probabilities[i]);
                if (NumOps.GreaterThanOrEquals(sum, random))
                {
                    return i;
                }
            }
        }

        return -1; // This should only happen if there are no unvisited features
    }

    /// <summary>
    /// Updates the pheromone levels based on the quality of solutions found in the current iteration.
    /// </summary>
    /// <param name="pheromones">The pheromone matrix to be updated.</param>
    /// <param name="solutions">The list of solutions (models) found in the current iteration.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method adjusts the pheromone levels after all ants have built their
    /// solutions. It first reduces all pheromone levels (evaporation), then increases pheromone levels on
    /// paths used by good solutions. This helps guide future ants towards promising areas of the solution space.</para>
    /// </remarks>
    private void UpdatePheromones(Matrix<T> pheromones, List<IFullModel<T, TInput, TOutput>> solutions)
    {
        // === Partially Vectorized Pheromone Update using IEngine (Phase B: US-GPU-015) ===

        // Evaporation: Apply (1 - evaporationRate) to each element in-place
        var evaporationFactor = NumOps.Subtract(NumOps.One, _currentPheromoneEvaporationRate);
        for (int i = 0; i < pheromones.Rows; i++)
        {
            for (int j = 0; j < pheromones.Columns; j++)
            {
                pheromones[i, j] = NumOps.Multiply(pheromones[i, j], evaporationFactor);
            }
        }

        // Deposit - partially vectorized
        // Use pheromone matrix dimensions (features), not parameter count (which includes intercept)
        int pheromoneSize = pheromones.Rows;
        for (int k = 0; k < solutions.Count; k++)
        {
            var model = solutions[k];
            var parameters = model.GetParameters();
            var deposit = NumOps.Divide(_currentPheromoneIntensity, NumOps.Add(NumOps.One, FitnessList[k]));

            // Vectorized absolute value of parameters
            var absParams = (Vector<T>)AiDotNetEngine.Current.Abs(parameters);

            // Only iterate up to pheromone matrix size, not parameter length
            // Parameters may include intercept which is not a feature
            int paramLimit = Math.Min(pheromoneSize, parameters.Length);
            for (int i = 0; i < paramLimit; i++)
            {
                var depositScaled = NumOps.Multiply(deposit, absParams[i]);
                for (int j = 0; j < paramLimit; j++)
                {
                    if (i != j)
                    {
                        pheromones[i, j] = NumOps.Add(pheromones[i, j], depositScaled);
                    }
                }
            }
        }
    }

    /// <summary>
    /// Updates the options for the Ant Colony Optimization algorithm.
    /// </summary>
    /// <param name="options">The new options to be set.</param>
    /// <exception cref="ArgumentException">Thrown when the provided options are not of the correct type.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method allows you to change how the ant colony algorithm behaves
    /// by updating its settings. It checks to make sure you're providing the right kind of settings.</para>
    /// </remarks>
    protected override void UpdateOptions(OptimizationAlgorithmOptions<T, TInput, TOutput> options)
    {
        if (options is AntColonyOptimizationOptions<T, TInput, TOutput> antColonyOptions)
        {
            _antColonyOptions = antColonyOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected AntColonyOptimizationOptions.");
        }
    }

    /// <summary>
    /// Gets the current options for the Ant Colony Optimization algorithm.
    /// </summary>
    /// <returns>The current optimization options.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method lets you see what settings the ant colony algorithm is currently using.</para>
    /// </remarks>
    public override OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions()
    {
        return _antColonyOptions;
    }

    /// <summary>
    /// Converts the current state of the optimizer into a byte array for storage or transmission.
    /// </summary>
    /// <returns>A byte array representing the serialized state of the optimizer.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method takes all the important information about the current state
    /// of the ant colony optimizer and turns it into a format that can be easily saved or sent to another computer.</para>
    /// </remarks>
    public override byte[] Serialize()
    {
        using MemoryStream ms = new MemoryStream();
        using BinaryWriter writer = new BinaryWriter(ms);

        // Serialize base class data
        byte[] baseData = base.Serialize();
        writer.Write(baseData.Length);
        writer.Write(baseData);

        // Serialize AntColonyOptimizerOptions
        string optionsJson = JsonConvert.SerializeObject(_antColonyOptions);
        writer.Write(optionsJson);

        // Serialize adaptive parameters
        writer.Write(Convert.ToDouble(_currentPheromoneEvaporationRate));
        writer.Write(Convert.ToDouble(_currentPheromoneIntensity));

        return ms.ToArray();
    }

    /// <summary>
    /// Restores the state of the optimizer from a byte array.
    /// </summary>
    /// <param name="data">The byte array containing the serialized state of the optimizer.</param>
    /// <exception cref="InvalidOperationException">Thrown when deserialization of optimizer options fails.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method takes a saved state of the ant colony optimizer (in the form of a byte array)
    /// and uses it to restore the optimizer to that state. It's like loading a saved game, bringing back all the
    /// important settings and progress that were saved earlier.</para>
    /// </remarks>
    public override void Deserialize(byte[] data)
    {
        using MemoryStream ms = new MemoryStream(data);
        using BinaryReader reader = new BinaryReader(ms);

        // Deserialize base class data
        int baseDataLength = reader.ReadInt32();
        byte[] baseData = reader.ReadBytes(baseDataLength);
        base.Deserialize(baseData);

        // Deserialize AntColonyOptimizerOptions
        string optionsJson = reader.ReadString();
        _antColonyOptions = JsonConvert.DeserializeObject<AntColonyOptimizationOptions<T, TInput, TOutput>>(optionsJson)
            ?? throw new InvalidOperationException("Failed to deserialize optimizer options.");

        // Deserialize adaptive parameters
        _currentPheromoneEvaporationRate = NumOps.FromDouble(reader.ReadDouble());
        _currentPheromoneIntensity = NumOps.FromDouble(reader.ReadDouble());
    }
}
