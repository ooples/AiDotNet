global using AiDotNet.Genetics;
using Newtonsoft.Json;

using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.Optimizers;

/// <summary>
/// Represents a Genetic Algorithm optimizer for machine learning models.
/// </summary>
/// <remarks>
/// <para>
/// The Genetic Algorithm optimizer is an evolutionary optimization technique inspired by the process of natural selection.
/// It evolves a population of potential solutions over multiple generations to find an optimal or near-optimal solution.
/// </para>
/// <para><b>For Beginners:</b> Think of the Genetic Algorithm optimizer like breeding the best solutions:
/// 
/// - Start with a group of random solutions (like a group of different recipes)
/// - Test how good each solution is (like tasting each recipe)
/// - Choose the best solutions (like picking the tastiest recipes)
/// - Create new solutions by mixing the best ones (like combining ingredients from the best recipes)
/// - Sometimes make small random changes (like accidentally adding a new spice)
/// - Repeat this process many times to find the best solution (or the tastiest recipe!)
/// 
/// This approach is good at finding solutions for complex problems where traditional methods might struggle.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[ComponentType(ComponentType.Optimizer)]
[PipelineStage(PipelineStage.Training)]
public partial class GeneticAlgorithmOptimizer<T, TInput, TOutput> : OptimizerBase<T, TInput, TOutput>, IDerivativeFreeFunctionOptimizer<T>
{
    /// <summary>
    /// The options specific to the Genetic Algorithm.
    /// </summary>
    private GeneticAlgorithmOptimizerOptions<T, TInput, TOutput> _geneticOptions;

    /// <summary>
    /// The current crossover rate, which determines how often solutions are combined.
    /// </summary>
    private T _currentCrossoverRate;

    /// <summary>
    /// The current mutation rate, which determines how often random changes are made to solutions.
    /// </summary>
    private T _currentMutationRate;

    /// <summary>
    /// The genetic algorithm instance used for optimization.
    /// </summary>
    private GeneticBase<T, TInput, TOutput> _geneticAlgorithm;

    /// <summary>
    /// Initializes a new instance of the GeneticAlgorithmOptimizer class.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This sets up the genetic algorithm with its initial settings.
    /// You can customize various aspects of how it works, or use default settings if you're unsure.
    /// </para>
    /// </remarks>
    /// <param name="model">The model to be optimized.</param>
    /// <param name="options">The options for configuring the genetic algorithm.</param>
    public GeneticAlgorithmOptimizer(
        IFullModel<T, TInput, TOutput>? model,
        GeneticAlgorithmOptimizerOptions<T, TInput, TOutput>? options = null,
        GeneticBase<T, TInput, TOutput>? geneticAlgorithm = null,
        IFitnessCalculator<T, TInput, TOutput>? fitnessCalculator = null)
        : base(model, options ?? new())
    {
        _geneticOptions = options ?? new GeneticAlgorithmOptimizerOptions<T, TInput, TOutput>();
        _currentCrossoverRate = NumOps.Zero;
        _currentMutationRate = NumOps.Zero;

        // If no genetic algorithm is provided, create a default StandardGeneticAlgorithm
        if (geneticAlgorithm == null)
        {
            // Use the provided model as a template, falling back to SimpleRegression if not provided
            var templateModel = model;
            IFullModel<T, TInput, TOutput> modelFactory()
            {
                // Clone the template model to get a fresh instance with the same configuration
                if (templateModel != null)
                {
                    return templateModel.Clone();
                }
                return (IFullModel<T, TInput, TOutput>)new SimpleRegression<T>();
            }

            _geneticAlgorithm = new StandardGeneticAlgorithm<T, TInput, TOutput>(
                modelFactory,
                fitnessCalculator ?? new MeanSquaredErrorFitnessCalculator<T, TInput, TOutput>());
        }
        else
        {
            _geneticAlgorithm = geneticAlgorithm;
        }

        InitializeAdaptiveParameters();
    }

    /// <summary>
    /// Updates the adaptive parameters used in the genetic algorithm.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method adjusts how the algorithm behaves based on its recent performance.
    /// It's like a chef adjusting their cooking technique based on how the last few dishes turned out.
    /// </para>
    /// </remarks>
    /// <param name="currentStepData">Data from the current optimization step.</param>
    /// <param name="previousStepData">Data from the previous optimization step.</param>
    protected override void UpdateAdaptiveParameters(OptimizationStepData<T, TInput, TOutput> currentStepData, OptimizationStepData<T, TInput, TOutput> previousStepData)
    {
        base.UpdateAdaptiveParameters(currentStepData, previousStepData);
        AdaptiveParametersHelper<T, TInput, TOutput>.UpdateAdaptiveGeneticParameters(ref _currentCrossoverRate, ref _currentMutationRate,
            currentStepData, previousStepData, _geneticOptions);
    }

    /// <summary>
    /// Initializes the adaptive parameters used in the genetic algorithm.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method sets up the initial rates for crossover (mixing solutions)
    /// and mutation (making small random changes). It's like setting the initial recipe and how often
    /// you'll try new ingredients.
    /// </para>
    /// </remarks>
    private new void InitializeAdaptiveParameters()
    {
        _currentCrossoverRate = NumOps.FromDouble(_geneticOptions.CrossoverRate);
        _currentMutationRate = NumOps.FromDouble(_geneticOptions.MutationRate);
    }

    /// <summary>
    /// Performs the main optimization process using the genetic algorithm.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the heart of the genetic algorithm. It:
    /// 1. Creates an initial group of random solutions
    /// 2. Evaluates how good each solution is
    /// 3. Selects the best solutions
    /// 4. Creates new solutions by mixing the best ones
    /// 5. Sometimes makes small random changes to solutions
    /// 6. Repeats this process for many generations
    /// 
    /// It's like running a cooking competition where each round you keep the best recipes,
    /// combine them to make new recipes, and occasionally add a surprise ingredient.
    /// </para>
    /// </remarks>
    /// <param name="inputData">The input data for the optimization process.</param>
    /// <returns>The result of the optimization process.</returns>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        // Initialize genetic algorithm parameters
        var geneticParams = _geneticAlgorithm.GetGeneticParameters();
        geneticParams.PopulationSize = _geneticOptions.PopulationSize;
        geneticParams.MaxGenerations = Options.MaxIterations;
        geneticParams.CrossoverRate = Convert.ToDouble(_currentCrossoverRate);
        geneticParams.MutationRate = Convert.ToDouble(_currentMutationRate);

        _geneticAlgorithm.ConfigureGeneticParameters(geneticParams);

        // Let the genetic algorithm handle the evolutionary process
        var evolutionStats = _geneticAlgorithm.Evolve(
            Options.MaxIterations,
            inputData.XTrain,
            inputData.YTrain,
            inputData.XValidation,
            inputData.YValidation);

        // Convert the result to optimization result format
        var bestIndividual = _geneticAlgorithm.GetBestIndividual();
        var model = _geneticAlgorithm.IndividualToModel(bestIndividual);

        // Evaluate the best model through the standard pipeline to populate
        // SelectedFeatures, evaluation data, and data subsets
        var bestStepData = EvaluateSolution(model, inputData);
        bestStepData.FitnessScore = bestIndividual.GetFitness();

        // Transfer fitness history and iteration count from the genetic algorithm
        FitnessList.Clear();
        FitnessList.AddRange(evolutionStats.FitnessHistory);
        for (int i = 0; i < evolutionStats.Generation; i++)
        {
            UpdateIterationHistoryAndCheckEarlyStopping(i, bestStepData);
        }

        return CreateOptimizationResult(bestStepData, inputData);
    }

    /// <summary>
    /// Updates the options for the genetic algorithm optimizer.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method allows you to change the settings of the genetic algorithm
    /// while it's running. It's like adjusting the rules of your cooking competition mid-way through.
    /// </para>
    /// </remarks>
    /// <param name="options">The new options to apply to the optimizer.</param>
    /// <exception cref="ArgumentException">Thrown when the provided options are not of the correct type.</exception>
    protected override void UpdateOptions(OptimizationAlgorithmOptions<T, TInput, TOutput> options)
    {
        if (options is GeneticAlgorithmOptimizerOptions<T, TInput, TOutput> geneticOptions)
        {
            _geneticOptions = geneticOptions;
        }
        else
        {
            throw new ArgumentException("Invalid options type. Expected GeneticAlgorithmOptimizerOptions.");
        }
    }

    /// <summary>
    /// Gets the current options for the genetic algorithm optimizer.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method returns the current settings of the genetic algorithm.
    /// It's like checking the current rules of your cooking competition.
    /// </para>
    /// </remarks>
    /// <returns>The current genetic algorithm optimizer options.</returns>
    public override OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions()
    {
        return _geneticOptions;
    }

    /// <summary>
    /// Creates a genetic algorithm optimizer for minimizing a plain function, with no model.
    /// </summary>
    /// <param name="options">The optimizer-specific options. If null, defaults are used.</param>
    public static GeneticAlgorithmOptimizer<T, TInput, TOutput> CreateForFunction(
        GeneticAlgorithmOptimizerOptions<T, TInput, TOutput>? options = null)
        => new(null, options);

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// A real-coded genetic algorithm: tournament selection, blend crossover, Gaussian mutation
    /// and elitism. The encoding is the parameter vector itself rather than a bit string, which is
    /// the usual choice for continuous problems (Herrera, Lozano and Verdegay, <i>Artificial
    /// Intelligence Review</i> 12, 1998).
    /// </para>
    /// <para>
    /// Elitism — carrying the best member forward untouched — is what makes the best value
    /// monotone. Without it a genetic algorithm can and does lose its best solution to the
    /// randomness that is otherwise the point of the method.
    /// </para>
    /// <para>
    /// <paramref name="tolerance"/> stops the run once the population's spread falls below it.
    /// </para>
    /// </remarks>
    public Vector<T> Minimize(
        Vector<T> initialParameters, Func<Vector<T>, T> objective, int maxIterations, T tolerance)
    {
        ValidateMinimizeArguments(initialParameters, objective, maxIterations);

        var search = new DerivativeFreeSearch(objective, NumOps, initialParameters);
        var random = CreateSearchRandom();

        int dimension = initialParameters.Length;
        int populationSize = Math.Max(4, _geneticOptions.PopulationSize);

        double crossoverRate = _geneticOptions.CrossoverRate;
        double mutationRate = _geneticOptions.MutationRate;
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
            var next = new Vector<T>[populationSize];
            var nextValues = new T[populationSize];

            // Elitism: the best member survives unchanged, so the best value cannot go backwards.
            int elite = 0;
            for (int p = 1; p < populationSize; p++)
            {
                if (NumOps.LessThan(values[p], values[elite])) elite = p;
            }

            next[0] = population[elite].Clone();
            nextValues[0] = values[elite];

            for (int p = 1; p < populationSize; p++)
            {
                var firstParent = population[Tournament(random, values, populationSize)];
                var secondParent = population[Tournament(random, values, populationSize)];

                var child = new Vector<T>(dimension);

                for (int i = 0; i < dimension; i++)
                {
                    if (random.NextDouble() < crossoverRate)
                    {
                        // Blend crossover: a point drawn from the interval the parents span,
                        // slightly extended so the population can widen as well as narrow.
                        double low = Convert.ToDouble(firstParent[i]);
                        double high = Convert.ToDouble(secondParent[i]);
                        double reach = 0.5 * (high - low);

                        child[i] = NumOps.FromDouble(
                            low + (1.0 + 2.0 * 0.5) * reach * random.NextDouble()
                                - 0.5 * reach);
                    }
                    else
                    {
                        child[i] = firstParent[i];
                    }

                    if (random.NextDouble() < mutationRate)
                    {
                        child[i] = NumOps.Add(child[i], NumOps.FromDouble(NextGaussian(random)));
                    }
                }

                next[p] = child;
                nextValues[p] = search.Evaluate(child);
            }

            population = next;
            values = nextValues;

            if (Spread(population, dimension) < stop) break;
        }

        return search.BestPoint;
    }

    /// <summary>Picks the better of two randomly drawn members.</summary>
    private int Tournament(Random random, T[] values, int populationSize)
    {
        int first = random.Next(populationSize);
        int second = random.Next(populationSize);

        return NumOps.LessThan(values[first], values[second]) ? first : second;
    }

    /// <summary>The mean absolute deviation of the population, coordinate by coordinate.</summary>
    private double Spread(Vector<T>[] population, int dimension)
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
