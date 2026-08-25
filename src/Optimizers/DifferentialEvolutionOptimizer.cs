using AiDotNet.Helpers;
using Newtonsoft.Json;

using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.Optimizers;

/// <summary>
/// Implements the Differential Evolution optimization algorithm for numerical optimization problems.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Differential Evolution is a population-based optimization algorithm that is particularly well-suited
/// for solving non-linear, non-differentiable continuous space functions. It's known for its simplicity,
/// robustness, and effectiveness in various optimization scenarios.
/// </para>
/// <para><b>For Beginners:</b> This optimizer works by evolving a population of candidate solutions over time.
/// It's inspired by biological evolution and is good at finding global optima in complex problem spaces.
/// </para>
/// </remarks>
[ComponentType(ComponentType.Optimizer)]
[PipelineStage(PipelineStage.Training)]
public partial class DifferentialEvolutionOptimizer<T, TInput, TOutput> : OptimizerBase<T, TInput, TOutput>, IDerivativeFreeFunctionOptimizer<T>
{
    /// <summary>
    /// Configuration options specific to the Differential Evolution algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the parameters that control the behavior of the Differential Evolution algorithm,
    /// such as population size, crossover rate, mutation rate, and other algorithm-specific settings.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the rulebook for our evolutionary process.
    /// It defines how many candidate solutions we'll work with (population size), how often we'll
    /// combine solutions (crossover rate), and how much random variation we'll introduce (mutation rate).
    /// These settings control the balance between exploration (trying new areas) and exploitation
    /// (refining good solutions).
    /// </para>
    /// </remarks>
    private DifferentialEvolutionOptions<T, TInput, TOutput> _deOptions;

    /// <summary>
    /// The current crossover rate used in the optimization process.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The crossover rate determines the probability of exchanging components between solutions
    /// during the creation of new trial solutions. This value may adapt during optimization.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the probability of combining features from different
    /// solutions. A higher value means more mixing of solutions, which helps explore new combinations.
    /// This value can change during optimization to balance exploration and refinement.
    /// </para>
    /// </remarks>
    private T _currentCrossoverRate;

    /// <summary>
    /// The current mutation rate used in the optimization process.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The mutation rate controls the magnitude of random changes applied to solutions
    /// during the differential mutation step. This value may adapt during optimization.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the amount of random tweaking we apply to solutions.
    /// A higher value means bigger changes, which helps explore more diverse possibilities.
    /// This value can change during optimization to balance exploration and refinement.
    /// </para>
    /// </remarks>
    private T _currentMutationRate;

    /// <summary>
    /// Initializes a new instance of the DifferentialEvolutionOptimizer class.
    /// </summary>
    /// <param name="model">The model to be optimized.</param>
    /// <param name="options">The options for configuring the Differential Evolution algorithm.</param>
    /// <param name="engine">The computation engine (CPU or GPU) for vectorized operations.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This constructor sets up the Differential Evolution optimizer with its initial configuration.
    /// You can customize various aspects of how it works, or use default settings.
    /// </para>
    /// </remarks>
    public DifferentialEvolutionOptimizer(
        IFullModel<T, TInput, TOutput> model,
        DifferentialEvolutionOptions<T, TInput, TOutput>? options = null,
        IEngine? engine = null)
        : base(model, options ?? new())
    {
        _deOptions = options ?? new DifferentialEvolutionOptions<T, TInput, TOutput>();
        _currentCrossoverRate = NumOps.Zero;
        _currentMutationRate = NumOps.Zero;

        InitializeAdaptiveParameters();
    }

    /// <summary>
    /// Updates the adaptive parameters based on the optimization progress.
    /// </summary>
    /// <param name="currentStepData">Data from the current optimization step.</param>
    /// <param name="previousStepData">Data from the previous optimization step.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method adjusts the crossover and mutation rates during the optimization process.
    /// It helps the algorithm adapt its behavior based on how well it's performing.
    /// </para>
    /// </remarks>
    protected override void UpdateAdaptiveParameters(OptimizationStepData<T, TInput, TOutput> currentStepData, OptimizationStepData<T, TInput, TOutput> previousStepData)
    {
        base.UpdateAdaptiveParameters(currentStepData, previousStepData);
        AdaptiveParametersHelper<T, TInput, TOutput>.UpdateAdaptiveGeneticParameters(ref _currentCrossoverRate, ref _currentMutationRate, currentStepData, previousStepData, _deOptions);
    }

    /// <summary>
    /// Initializes the adaptive parameters used in the Differential Evolution algorithm.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method sets up the initial crossover and mutation rates.
    /// These rates determine how the algorithm combines and changes solutions during optimization.
    /// </para>
    /// </remarks>
    private new void InitializeAdaptiveParameters()
    {
        _currentCrossoverRate = NumOps.FromDouble(_deOptions.CrossoverRate);
        _currentMutationRate = NumOps.FromDouble(_deOptions.MutationRate);
    }

    /// <summary>
    /// Performs the main optimization process using the Differential Evolution algorithm.
    /// </summary>
    /// <param name="inputData">The input data for the optimization process.</param>
    /// <returns>The result of the optimization process.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the heart of the Differential Evolution algorithm. It creates an initial population
    /// of solutions and then evolves them over multiple generations. In each generation, it creates new trial solutions,
    /// evaluates them, and keeps the best ones.
    /// </para>
    /// </remarks>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        var population = InitializePopulation(inputData.XTrain, _deOptions.PopulationSize);
        // Use model's parameter count instead of input dimensions for trial vectors
        int paramCount = (int)population.Count > 0 ? (int)InterfaceGuard.Parameterizable(population[0]).ParameterCount : InputHelper<T, TInput>.GetInputSize(inputData.XTrain);
        var bestStepData = new OptimizationStepData<T, TInput, TOutput>();
        var prevStepData = new OptimizationStepData<T, TInput, TOutput>();
        var currentStepData = new OptimizationStepData<T, TInput, TOutput>();

        for (int generation = 0; generation < Options.MaxIterations; generation++)
        {
            for (int i = 0; i < _deOptions.PopulationSize; i++)
            {
                var trial = GenerateTrialModel(population, i, paramCount);
                currentStepData = EvaluateSolution(trial, inputData);
                UpdateBestSolution(currentStepData, ref bestStepData);
                population[i] = currentStepData.Solution;
            }

            // Update adaptive parameters
            UpdateAdaptiveParameters(currentStepData, prevStepData);

            if (UpdateIterationHistoryAndCheckEarlyStopping(generation, bestStepData))
            {
                break; // Early stopping criteria met, exit the loop
            }

            prevStepData = currentStepData;
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    /// <summary>
    /// Initializes the population for the Differential Evolution algorithm.
    /// </summary>
    /// <param name="dimensions">The number of dimensions in the problem space.</param>
    /// <param name="populationSize">The size of the population to initialize.</param>
    /// <returns>A list of randomly initialized symbolic models.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method creates the initial set of candidate solutions.
    /// Each solution is a random guess at what might be a good answer to the optimization problem.
    /// </para>
    /// </remarks>
    private List<IFullModel<T, TInput, TOutput>> InitializePopulation(TInput input, int populationSize)
    {
        var population = new List<IFullModel<T, TInput, TOutput>>();
        for (int i = 0; i < populationSize; i++)
        {
            population.Add(SpawnIndividual(input));
        }

        return population;
    }

    /// <summary>
    /// Generates a trial model using the Differential Evolution algorithm's mutation and crossover operations.
    /// </summary>
    /// <param name="population">The current population of models.</param>
    /// <param name="currentIndex">The index of the current model in the population.</param>
    /// <param name="dimensions">The number of dimensions in the problem space.</param>
    /// <returns>A new trial model.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method creates a new candidate solution by combining and mutating
    /// existing solutions. It's how the algorithm explores new possibilities and improves over time.
    /// </para>
    /// </remarks>
    private IFullModel<T, TInput, TOutput> GenerateTrialModel(List<IFullModel<T, TInput, TOutput>> population, int currentIndex, int dimensions)
    {
        int a, b, c;
        do
        {
            a = Random.Next(population.Count);
            b = Random.Next(population.Count);
            c = Random.Next(population.Count);
        } while (a == currentIndex || b == currentIndex || c == currentIndex || a == b || a == c || b == c);

        var currentModel = population[currentIndex];

        // Get parameters from each model
        var aParams = InterfaceGuard.Parameterizable(population[a]).GetParameters();
        var bParams = InterfaceGuard.Parameterizable(population[b]).GetParameters();
        var cParams = InterfaceGuard.Parameterizable(population[c]).GetParameters();
        var currentParams = InterfaceGuard.Parameterizable(currentModel).GetParameters();

        // === Partially Vectorized Differential Evolution Mutation using IEngine (Phase B: US-GPU-015) ===
        // Vectorized differential mutation: mutant = a + F * (b - c)
        var bMinusC = (Vector<T>)Engine.Subtract(bParams, cParams);
        var scaledDiff = (Vector<T>)Engine.Multiply(bMinusC, _currentMutationRate);
        var mutant = (Vector<T>)Engine.Add(aParams, scaledDiff);

        var trialParams = new Vector<T>(dimensions);
        int R = Random.Next(dimensions);
        var currentCrossOverRate = Convert.ToDouble(_currentCrossoverRate);

        // Crossover (element-wise due to random per-element decisions)
        for (int i = 0; i < dimensions; i++)
        {
            if (Random.NextDouble() < currentCrossOverRate || i == R)
            {
                trialParams[i] = mutant[i];
            }
            else
            {
                trialParams[i] = currentParams[i];
            }
        }

        // Create a new model with the modified parameters
        return InterfaceGuard.Parameterizable(currentModel).WithParameters(trialParams);
    }

    /// <summary>
    /// Updates the options for the Differential Evolution optimizer.
    /// </summary>
    /// <param name="options">The new options to be set.</param>
    /// <exception cref="ArgumentException">Thrown when the provided options are not of type DifferentialEvolutionOptions.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method allows you to change the settings of the optimizer during runtime.
    /// It ensures that only the correct type of options (specific to Differential Evolution) can be used.
    /// </para>
    /// </remarks>
    protected override void UpdateOptions(OptimizationAlgorithmOptions<T, TInput, TOutput> options)
    {
        if (options is DifferentialEvolutionOptions<T, TInput, TOutput> deOptions)
        {
            _deOptions = deOptions;
        }
        else
        {
            throw new ArgumentException("Options must be of type DifferentialEvolutionOptions", nameof(options));
        }
    }

    /// <summary>
    /// Retrieves the current options of the Differential Evolution optimizer.
    /// </summary>
    /// <returns>The current optimization algorithm options.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method allows you to check the current settings of the optimizer.
    /// It's useful if you need to inspect or copy the current configuration.
    /// </para>
    /// </remarks>
    public override OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions()
    {
        return _deOptions;
    }

    /// <summary>
    /// Creates a differential evolution optimizer for minimizing a plain function, with no model.
    /// </summary>
    /// <param name="options">The optimizer-specific options. If null, defaults are used.</param>
    public static DifferentialEvolutionOptimizer<T, TInput, TOutput> CreateForFunction(
        DifferentialEvolutionOptions<T, TInput, TOutput>? options = null)
        => new(options);

    /// <summary>Backs <see cref="CreateForFunction"/>: the same setup with no model.</summary>
    private DifferentialEvolutionOptimizer(DifferentialEvolutionOptions<T, TInput, TOutput>? options)
        : base(null, options ?? new())
    {
        _deOptions = options ?? new DifferentialEvolutionOptions<T, TInput, TOutput>();
        _currentCrossoverRate = NumOps.Zero;
        _currentMutationRate = NumOps.Zero;

        InitializeAdaptiveParameters();
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// DE/rand/1/bin, the classical scheme of Storn and Price (<i>Journal of Global
    /// Optimization</i> 11, 1997). For each member of the population, three others are drawn at
    /// random and combined into a mutant:
    /// </para>
    /// <code>
    /// mutant = a + F*(b - c)
    /// </code>
    /// <para>
    /// The mutant is then mixed with the original coordinate by coordinate, and replaces it only
    /// if it is better. The step size is never chosen: it comes from the spread of the population
    /// itself, which is why DE contracts automatically as it converges and needs no schedule.
    /// </para>
    /// <para>
    /// <paramref name="tolerance"/> stops the run once the population's spread falls below it,
    /// which is the natural convergence test for a method whose steps ARE that spread.
    /// </para>
    /// </remarks>
    public Vector<T> Minimize(
        Vector<T> initialParameters, Func<Vector<T>, T> objective, int maxIterations, T tolerance)
    {
        ValidateMinimizeArguments(initialParameters, objective, maxIterations);

        var search = new DerivativeFreeSearch(objective, NumOps, initialParameters);
        var random = CreateSearchRandom();

        int dimension = initialParameters.Length;
        int populationSize = Math.Max(4, _deOptions.PopulationSize);

        double differentialWeight = _deOptions.MutationRate;
        double crossoverRate = _deOptions.CrossoverRate;
        double stop = Convert.ToDouble(tolerance);

        var population = new Vector<T>[populationSize];
        var values = new T[populationSize];

        for (int p = 0; p < populationSize; p++)
        {
            population[p] = new Vector<T>(dimension);

            for (int i = 0; i < dimension; i++)
            {
                double offset = p == 0 ? 0.0 : NextGaussian(random);
                population[p][i] = NumOps.Add(initialParameters[i], NumOps.FromDouble(offset));
            }

            values[p] = search.Evaluate(population[p]);
        }

        for (int generation = 0; generation < maxIterations; generation++)
        {
            for (int p = 0; p < populationSize; p++)
            {
                int first = PickOther(random, populationSize, p, -1, -1);
                int second = PickOther(random, populationSize, p, first, -1);
                int third = PickOther(random, populationSize, p, first, second);

                // At least one coordinate always comes from the mutant, so the trial can never be
                // an exact copy of the member it is meant to replace.
                int forced = random.Next(dimension);
                var trial = new Vector<T>(dimension);

                for (int i = 0; i < dimension; i++)
                {
                    if (i == forced || random.NextDouble() < crossoverRate)
                    {
                        double difference = Convert.ToDouble(
                            NumOps.Subtract(population[second][i], population[third][i]));

                        trial[i] = NumOps.Add(
                            population[first][i],
                            NumOps.FromDouble(differentialWeight * difference));
                    }
                    else
                    {
                        trial[i] = population[p][i];
                    }
                }

                T trialValue = search.Evaluate(trial);

                if (NumOps.LessThan(trialValue, values[p]))
                {
                    population[p] = trial;
                    values[p] = trialValue;
                }
            }

            if (PopulationSpread(population, dimension) < stop) break;
        }

        return search.BestPoint;
    }

    /// <summary>Draws an index different from the ones already chosen.</summary>
    private static int PickOther(Random random, int count, int self, int first, int second)
    {
        for (int attempt = 0; attempt < 100; attempt++)
        {
            int candidate = random.Next(count);
            if (candidate != self && candidate != first && candidate != second) return candidate;
        }

        // With a population of four or more the loop above effectively always succeeds; this only
        // exists so a pathological generator cannot hang the search.
        return (self + 1) % count;
    }

    /// <summary>The mean absolute deviation of the population, coordinate by coordinate.</summary>
    private double PopulationSpread(Vector<T>[] population, int dimension)
    {
        double total = 0.0;

        for (int i = 0; i < dimension; i++)
        {
            double mean = 0.0;
            foreach (var member in population) mean += Convert.ToDouble(member[i]);
            mean /= population.Length;

            foreach (var member in population)
            {
                total += Math.Abs(Convert.ToDouble(member[i]) - mean);
            }
        }

        return total / (population.Length * dimension);
    }

}
