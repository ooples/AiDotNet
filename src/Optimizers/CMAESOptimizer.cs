using System.Linq;
using AiDotNet.Helpers;
using AiDotNet.Extensions;
using Newtonsoft.Json;

using AiDotNet.Attributes;
using AiDotNet.Enums;

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
[ComponentType(ComponentType.Optimizer)]
[PipelineStage(PipelineStage.Training)]
public partial class CMAESOptimizer<T, TInput, TOutput> : OptimizerBase<T, TInput, TOutput>, IDerivativeFreeFunctionOptimizer<T>
{
    /// <summary>
    /// The options specific to the CMA-ES optimization algorithm.
    /// </summary>
    /// <summary>Read from the single instance OptimizerBase.Options holds, so there is
    /// no second copy that could disagree with it.</summary>
    private CMAESOptimizerOptions<T, TInput, TOutput> _options => (CMAESOptimizerOptions<T, TInput, TOutput>)Options;

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
        IFullModel<T, TInput, TOutput>? model,
        CMAESOptimizerOptions<T, TInput, TOutput>? options = null,
        IEngine? engine = null)
        : base(model, options ?? new())
    {
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
        // Always use a deep copy of Model to avoid mutating the original during optimization
        var initialSolution = SpawnIndividual(inputData.XTrain);
        _mean = InterfaceGuard.Parameterizable(initialSolution).GetParameters();
        // Use parameter count (coefficients + intercept), not input size (features only)
        // This ensures covariance matrix dimensions match the mean vector length
        int dimensions = _mean.Length;
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
            var currentSolution = InterfaceGuard.Parameterizable(currentBestModel).WithParameters(_mean);
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
            var scaledSample = (Vector<T>)Engine.Multiply(sample, _sigma);
            var individual = (Vector<T>)Engine.Add(_mean, scaledSample);

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
        return Random.NextGaussian();
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
            var solution = InterfaceGuard.Parameterizable(templateModel).WithParameters(population.GetRow(i));
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
        var muPlusHalf = Engine.Fill<T>(mu, NumOps.FromDouble(mu + 0.5));
        var indexPlusMu = (Vector<T>)Engine.Add(indices, muPlusHalf);
        var logValues = (Vector<T>)Engine.Log(indexPlusMu);
        var weights = (Vector<T>)Engine.Subtract(muPlusHalf, logValues);

        // Normalize weights
        T sumWeights = Engine.Sum(weights);
        weights = weights.Divide(sumWeights);

        // Calculate effective mu - vectorized
        // muEff = 1 / sum(weights^2)
        var weightsSquared = new Vector<T>(weights.Length);
        for (int i = 0; i < weights.Length; i++)
        {
            weightsSquared[i] = NumOps.Square(weights[i]);
        }
        T sumSquaredWeights = Engine.Sum(weightsSquared);
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

        // chiN: the expected length of a draw from N(0, I) in this many dimensions.
        //
        // Hansen's step-size rule compares the evolution path against how long a path of PURE
        // CHANCE would be, and adjusts sigma by the ratio. That reference length is E||N(0,I)||,
        // approximated as sqrt(n)(1 - 1/(4n) + 1/(21n^2)); this code used sqrt(n) on its own, which
        // is the leading term without the correction and is always too big.
        //
        // Always too big means always biased the same way. In two dimensions chiN is 1.2543 against
        // a sqrt(n) of 1.4142, so the ratio ||ps||/chiN came out 12.7% low, every generation, and
        // sigma shrank when it should have held. The search contracts early and settles in whatever
        // basin it happens to be in - which on Rastrigin, whose minima sit one lattice step apart,
        // means an answer of about 1.0 instead of 0.
        T chiN = NumOps.FromDouble(
            Math.Sqrt(dimensions) * (1.0 - 1.0 / (4.0 * dimensions) + 1.0 / (21.0 * dimensions * dimensions)));

        // Update evolution paths
        var y = _mean.Subtract(oldMean).Divide(_sigma);
        _ps = _ps.Multiply(NumOps.Subtract(NumOps.One, cs)).Add(
            y.Multiply(NumOps.Sqrt(NumOps.Multiply(cs, NumOps.Subtract(NumOps.FromDouble(2), cs)))).Multiply(NumOps.Sqrt(muEff)));

        // The same reference length belongs here. Hansen's hsig test is
        //   ||ps|| / sqrt(1 - (1-cs)^(2(g+1))) / chiN  <  1.4 + 2/(n+1)
        // and the division by chiN was missing, so the left side was measured in the wrong units
        // and compared against a threshold expressed in the right ones. Without it the test reads
        // high, hsig turns off, and the rank-one update stops accumulating exactly when a long
        // path says it should.
        T hsig = NumOps.LessThan(
            NumOps.Divide(
                NumOps.Divide(
                    _ps.Norm(),
                    NumOps.Sqrt(NumOps.Subtract(NumOps.One, NumOps.Power(NumOps.Subtract(NumOps.One, cs), NumOps.FromDouble(2.0 * _options.MaxGenerations))))
                ),
                chiN
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
                NumOps.Divide(_ps.Norm(), chiN),
                NumOps.One
            )
        )));
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
    /// Creates a CMA-ES optimizer for minimizing a plain function, with no model attached.
    /// </summary>
    /// <param name="options">The optimizer-specific options. If null, defaults are used.</param>
    public static CMAESOptimizer<T, TInput, TOutput> CreateForFunction(
        CMAESOptimizerOptions<T, TInput, TOutput>? options = null)
        => new(options);

    /// <summary>Backs <see cref="CreateForFunction"/>: the same setup with no model.</summary>
    private CMAESOptimizer(CMAESOptimizerOptions<T, TInput, TOutput>? options)
        : base(null, options ?? new())
    {
        _population = Matrix<T>.Empty();
        _mean = Vector<T>.Empty();
        _C = Matrix<T>.Empty();
        _pc = Vector<T>.Empty();
        _ps = Vector<T>.Empty();
        _sigma = NumOps.Zero;

        InitializeAdaptiveParameters();
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// The covariance matrix adaptation evolution strategy of Hansen and Ostermeier
    /// (<i>Evolutionary Computation</i> 9(2), 2001), with the constants from Hansen's 2016
    /// tutorial. Each generation samples a population from a Gaussian, keeps the better half, and
    /// moves the mean towards them - then adapts BOTH the covariance and the step size from the
    /// path the mean has actually travelled.
    /// </para>
    /// <para>
    /// The covariance is what makes this more than a random search: it learns the shape of the
    /// surface, so on a badly conditioned problem it stretches its sampling along the valley
    /// rather than across it. That is the same information a quasi-Newton method extracts from
    /// gradients, obtained here from rankings alone - which is why CMA-ES is the method of choice
    /// when the objective is a simulation with no derivative to be had.
    /// </para>
    /// <para>
    /// The step size is controlled separately, by comparing the length of the conjugate evolution
    /// path against how long a random walk would be. Longer than random means the steps agree with
    /// each other and could afford to be bigger; shorter means they are cancelling.
    /// </para>
    /// <para>
    /// <paramref name="tolerance"/> stops the run when the step size falls below it, the sampling
    /// distribution having collapsed to a point.
    /// </para>
    /// </remarks>
    public Vector<T> Minimize(
        Vector<T> initialParameters, Func<Vector<T>, T> objective, int maxIterations, T tolerance)
    {
        ValidateMinimizeArguments(initialParameters, objective, maxIterations, tolerance);

        var search = new DerivativeFreeSearch(objective, NumOps, initialParameters);
        var random = CreateSearchRandom();
        var state = CreateSearchState(initialParameters);
        double stop = Convert.ToDouble(tolerance);

        while (state.Generation < maxIterations && state.Sigma > stop)
        {
            var generation = Ask(state, random);

            var values = new double[generation.Points.Count];
            for (int k = 0; k < values.Length; k++)
            {
                values[k] = Convert.ToDouble(search.Evaluate(generation.Points[k]));
            }

            TellCore(state, generation, values);

            if (state.IsDegenerate) break;
        }

        return search.BestPoint;
    }

    /// <summary>
    /// Starts an incremental (ask/tell) CMA-ES search centred on <paramref name="initialParameters"/>.
    /// </summary>
    /// <param name="initialParameters">The initial mean of the search distribution.</param>
    /// <returns>A search state at generation zero: identity covariance, zero evolution paths, and the configured
    /// initial step size (0.3 when <c>InitialStepSize</c> is not positive).</returns>
    /// <exception cref="ArgumentNullException"><paramref name="initialParameters"/> is null.</exception>
    /// <exception cref="ArgumentException"><paramref name="initialParameters"/> is empty.</exception>
    /// <remarks>
    /// <para>
    /// The population size is the configured <c>PopulationSize</c> (at least 4), or Hansen's default
    /// 4 + floor(3 ln n) when it is not positive. <see cref="Minimize"/> runs exactly this sequence --
    /// <c>CreateSearchState</c>, then <see cref="Ask"/> and <see cref="Tell"/> once per generation -- so a caller that
    /// spreads the generations over several calls follows the same search.
    /// </para>
    /// <para><b>For Beginners:</b> Use this when you cannot evaluate a whole run at once -- for example, when each
    /// generation is one step of a longer training loop. Keep the returned state and pass it to every Ask and Tell.
    /// </para>
    /// </remarks>
    public CMAESSearchState CreateSearchState(Vector<T> initialParameters)
    {
        Guard.NotNull(initialParameters);
        if (initialParameters.Length == 0)
        {
            throw new ArgumentException(
                "Initial parameters must contain at least one element.",
                nameof(initialParameters));
        }

        int n = initialParameters.Length;
        int lambda = _options.PopulationSize > 0
            ? Math.Max(4, _options.PopulationSize)
            : 4 + (int)(3.0 * Math.Log(n));

        var mean = new double[n];
        for (int i = 0; i < n; i++) mean[i] = Convert.ToDouble(initialParameters[i]);

        return new CMAESSearchState
        {
            Dimension = n,
            PopulationSize = lambda,
            Generation = 0,
            Mean = mean,
            Sigma = _options.InitialStepSize > 0 ? _options.InitialStepSize : 0.3,
            PathSigma = new double[n],
            PathC = new double[n],
            Covariance = CmaIdentityRows(n),
        };
    }

    /// <summary>
    /// Samples the next generation of a search: lambda points m + sigma * y_k with y_k = A z_k ~ N(0, C), where A is
    /// the lower Cholesky factor of C and z_k is standard normal.
    /// </summary>
    /// <param name="state">The search to sample from; it is not modified.</param>
    /// <param name="random">The generator the standard normal draws come from.</param>
    /// <returns>The generation's candidate points, to be evaluated and passed back to <see cref="Tell"/>.</returns>
    /// <exception cref="ArgumentNullException">An argument is null.</exception>
    /// <exception cref="InvalidOperationException">The state's step size is not a positive finite number.</exception>
    public CMAESGeneration<T> Ask(CMAESSearchState state, Random random)
    {
        Guard.NotNull(state);
        Guard.NotNull(random);
        if (state.IsDegenerate)
        {
            throw new InvalidOperationException(
                $"The search's step size ({state.Sigma}) is not a positive finite number, so no generation can be sampled.");
        }

        int n = state.Dimension;
        int lambda = state.PopulationSize;
        var factor = CmaCholesky(state.Covariance, n);

        var draws = new double[lambda][];
        var points = new Vector<T>[lambda];

        for (int k = 0; k < lambda; k++)
        {
            var z = new double[n];
            for (int i = 0; i < n; i++) z[i] = NextGaussian(random);

            var y = new double[n];
            for (int i = 0; i < n; i++)
            {
                double total = 0.0;
                for (int j = 0; j <= i; j++) total += factor[i, j] * z[j];
                y[i] = total;
            }

            var point = new Vector<T>(n);
            for (int i = 0; i < n; i++)
            {
                point[i] = NumOps.FromDouble(state.Mean[i] + state.Sigma * y[i]);
            }

            draws[k] = y;
            points[k] = point;
        }

        return new CMAESGeneration<T>(points, draws, factor, state.Generation);
    }

    /// <summary>
    /// Updates a search from the fitness of the generation <see cref="Ask"/> sampled (lower is better): weighted
    /// recombination of the best mu draws moves the mean, the evolution paths are updated, C receives the rank-one
    /// and rank-mu updates, and sigma follows cumulative step-size adaptation (Hansen 2016, "The CMA Evolution
    /// Strategy: A Tutorial", Section 4 and Table 1).
    /// </summary>
    /// <param name="state">The search the generation was sampled from; it is advanced by one generation.</param>
    /// <param name="generation">The generation returned by <see cref="Ask"/> for this state.</param>
    /// <param name="fitness">The objective value of each point, in the order of <c>generation.Points</c>.</param>
    /// <exception cref="ArgumentNullException">An argument is null.</exception>
    /// <exception cref="ArgumentException">The fitness count does not match the population.</exception>
    /// <exception cref="InvalidOperationException">The generation was not sampled from this state's current
    /// generation, or has already been told.</exception>
    public void Tell(CMAESSearchState state, CMAESGeneration<T> generation, IReadOnlyList<T> fitness)
    {
        Guard.NotNull(state);
        Guard.NotNull(generation);
        Guard.NotNull(fitness);
        if (fitness.Count != generation.Points.Count)
        {
            throw new ArgumentException(
                $"Expected {generation.Points.Count} fitness values, got {fitness.Count}.", nameof(fitness));
        }

        var values = new double[fitness.Count];
        for (int k = 0; k < values.Length; k++) values[k] = Convert.ToDouble(fitness[k]);
        TellCore(state, generation, values);
    }

    private static void TellCore(CMAESSearchState state, CMAESGeneration<T> generation, double[] values)
    {
        if (generation.Told)
            throw new InvalidOperationException("This generation has already been told to the search.");
        if (generation.GenerationIndex != state.Generation || generation.Draws.Length != state.PopulationSize
            || (generation.Draws.Length > 0 && generation.Draws[0].Length != state.Dimension))
        {
            throw new InvalidOperationException(
                "The generation was not sampled from this search state's current generation.");
        }
        generation.Told = true;

        int n = state.Dimension;
        int lambda = state.PopulationSize;
        int mu = Math.Max(1, lambda / 2);

        // Rank-based recombination weights, log-decreasing so the best sample counts for most.
        var weights = new double[mu];
        double weightSum = 0.0;
        for (int i = 0; i < mu; i++)
        {
            weights[i] = Math.Log(mu + 0.5) - Math.Log(i + 1.0);
            weightSum += weights[i];
        }

        double squaredSum = 0.0;
        for (int i = 0; i < mu; i++)
        {
            weights[i] /= weightSum;
            squaredSum += weights[i] * weights[i];
        }

        double muEffective = 1.0 / squaredSum;

        // Hansen's defaults: learning rates for the two evolution paths and the two covariance
        // updates, all derived from the dimension and the effective selection mass.
        double cSigma = (muEffective + 2.0) / (n + muEffective + 5.0);
        double dSigma = 1.0
            + 2.0 * Math.Max(0.0, Math.Sqrt((muEffective - 1.0) / (n + 1.0)) - 1.0) + cSigma;
        double cc = (4.0 + muEffective / n) / (n + 4.0 + 2.0 * muEffective / n);
        double c1 = 2.0 / ((n + 1.3) * (n + 1.3) + muEffective);
        double cmu = Math.Min(
            1.0 - c1,
            2.0 * (muEffective - 2.0 + 1.0 / muEffective)
                / ((n + 2.0) * (n + 2.0) + muEffective));

        double expectedNorm = Math.Sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n));

        var draws = generation.Draws;
        var factor = generation.Factor;
        var mean = state.Mean;
        var pathSigma = state.PathSigma;
        var pathC = state.PathC;
        var covariance = state.Covariance;
        double sigma = state.Sigma;

        var order = Enumerable.Range(0, lambda).OrderBy(k => values[k]).ToArray();

        var weightedDraw = new double[n];
        for (int i = 0; i < n; i++)
        {
            double drawTotal = 0.0;
            for (int r = 0; r < mu; r++) drawTotal += weights[r] * draws[order[r]][i];

            weightedDraw[i] = drawTotal;
            mean[i] += sigma * drawTotal;
        }

        // The conjugate path needs C^(-1/2) times the mean shift; with the Cholesky factor
        // that is a forward substitution against the weighted draw.
        var conjugate = CmaForwardSolve(factor, weightedDraw, n);

        double pathSigmaNorm = 0.0;
        for (int i = 0; i < n; i++)
        {
            pathSigma[i] = (1.0 - cSigma) * pathSigma[i]
                + Math.Sqrt(cSigma * (2.0 - cSigma) * muEffective) * conjugate[i];

            pathSigmaNorm += pathSigma[i] * pathSigma[i];
        }

        pathSigmaNorm = Math.Sqrt(pathSigmaNorm);

        // The Heaviside switch stops a long path inflating the covariance just after the step
        // size has grown, which would count the same evidence twice.
        double heaviside =
            pathSigmaNorm / Math.Sqrt(1.0 - Math.Pow(1.0 - cSigma, 2.0 * (state.Generation + 1)))
                < (1.4 + 2.0 / (n + 1.0)) * expectedNorm ? 1.0 : 0.0;

        for (int i = 0; i < n; i++)
        {
            pathC[i] = (1.0 - cc) * pathC[i]
                + heaviside * Math.Sqrt(cc * (2.0 - cc) * muEffective) * weightedDraw[i];
        }

        double decay = 1.0 - c1 - cmu + (1.0 - heaviside) * c1 * cc * (2.0 - cc);

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double rankMu = 0.0;
                for (int r = 0; r < mu; r++)
                {
                    rankMu += weights[r] * draws[order[r]][i] * draws[order[r]][j];
                }

                covariance[i][j] = decay * covariance[i][j]
                    + c1 * pathC[i] * pathC[j]
                    + cmu * rankMu;
            }
        }

        state.Sigma = sigma * Math.Exp((cSigma / dSigma) * (pathSigmaNorm / expectedNorm - 1.0));
        state.Generation++;
    }

    /// <summary>An identity matrix as plain doubles, for the sampling arithmetic.</summary>
    private static double[,] CmaIdentity(int n)
    {
        var identity = new double[n, n];
        for (int i = 0; i < n; i++) identity[i, i] = 1.0;
        return identity;
    }

    /// <summary>An identity matrix stored as rows, the covariance a search starts from.</summary>
    private static double[][] CmaIdentityRows(int n)
    {
        var identity = new double[n][];
        for (int i = 0; i < n; i++)
        {
            identity[i] = new double[n];
            identity[i][i] = 1.0;
        }
        return identity;
    }

    /// <summary>
    /// The lower Cholesky factor, with a diagonal shift if the covariance has drifted out of
    /// positive definiteness - which rounding will eventually cause on a long run.
    /// </summary>
    private static double[,] CmaCholesky(double[][] matrix, int n)
    {
        for (double shift = 0.0; ; shift = shift == 0.0 ? 1e-12 : shift * 100.0)
        {
            var factor = new double[n, n];
            bool ok = true;

            for (int i = 0; i < n && ok; i++)
            {
                for (int j = 0; j <= i; j++)
                {
                    double total = matrix[i][j] + (i == j ? shift : 0.0);
                    for (int k = 0; k < j; k++) total -= factor[i, k] * factor[j, k];

                    if (i == j)
                    {
                        if (total <= 0.0 || double.IsNaN(total)) { ok = false; break; }
                        factor[i, j] = Math.Sqrt(total);
                    }
                    else
                    {
                        factor[i, j] = total / factor[j, j];
                    }
                }
            }

            if (ok) return factor;
            if (shift > 1.0) return CmaIdentity(n);
        }
    }

    /// <summary>Solves L z = y, which applies C^(-1/2) to y.</summary>
    private static double[] CmaForwardSolve(double[,] factor, double[] target, int n)
    {
        var solution = new double[n];

        for (int i = 0; i < n; i++)
        {
            double total = target[i];
            for (int j = 0; j < i; j++) total -= factor[i, j] * solution[j];
            solution[i] = factor[i, i] == 0.0 ? 0.0 : total / factor[i, i];
        }

        return solution;
    }

}
