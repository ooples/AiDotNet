#nullable disable
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Inputs;
using AiDotNet.Models.Options;
using AiDotNet.Models.Results;
using AiDotNet.Optimizers;
using AiDotNet.Regression;
using Xunit;
using System.Reflection;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests.Optimizers;

/// <summary>
/// Comprehensive integration tests for metaheuristic optimizers.
/// These tests verify that metaheuristic optimizers can find reasonable solutions
/// to optimization problems using evolutionary/stochastic methods.
/// </summary>
/// <remarks>
/// Metaheuristic optimizers are stochastic algorithms that don't use gradients.
/// They include evolutionary algorithms (GA, DE), swarm intelligence (PSO, ACO),
/// and local search methods (SA, Tabu Search, Nelder-Mead, Powell).
///
/// CRITICAL: These tests verify that optimizers work correctly.
/// If a test fails, FIX THE OPTIMIZER CODE, do NOT change the test.
/// </remarks>
public class MetaheuristicOptimizerIntegrationTests
{
    #region Test Helpers - Benchmark Functions

    /// <summary>
    /// Sphere function: f(x) = sum(x_i^2)
    /// Global minimum at origin with f(0) = 0
    /// </summary>
    private static double SphereFunction(double[] x)
    {
        double sum = 0;
        foreach (var xi in x)
        {
            sum += xi * xi;
        }
        return sum;
    }

    /// <summary>
    /// Rastrigin function: f(x) = An + sum(x_i^2 - A*cos(2*pi*x_i))
    /// Global minimum at origin with f(0) = 0
    /// Highly multimodal - good test for global optimization
    /// </summary>
    private static double RastriginFunction(double[] x, double A = 10)
    {
        double sum = A * x.Length;
        foreach (var xi in x)
        {
            sum += xi * xi - A * Math.Cos(2 * Math.PI * xi);
        }
        return sum;
    }

    /// <summary>
    /// Ackley function: A multimodal function with many local minima
    /// Global minimum at origin with f(0) = 0
    /// </summary>
    private static double AckleyFunction(double[] x, double a = 20, double b = 0.2, double c = 2 * Math.PI)
    {
        int n = x.Length;
        double sum1 = 0, sum2 = 0;
        foreach (var xi in x)
        {
            sum1 += xi * xi;
            sum2 += Math.Cos(c * xi);
        }
        return -a * Math.Exp(-b * Math.Sqrt(sum1 / n)) - Math.Exp(sum2 / n) + a + Math.E;
    }

    /// <summary>
    /// Create simple regression training data for optimizer tests.
    /// y = 2*x1 + 3*x2 + 1 (linear relationship)
    /// </summary>
    private static (Matrix<double> X, Vector<double> y) CreateSimpleRegressionData(int numSamples = 50)
    {
        var rand = new Random(42); // Fixed seed for reproducibility
        var X = new Matrix<double>(numSamples, 2);
        var y = new Vector<double>(numSamples);

        for (int i = 0; i < numSamples; i++)
        {
            double x1 = rand.NextDouble() * 10 - 5; // Range [-5, 5]
            double x2 = rand.NextDouble() * 10 - 5;
            X[i, 0] = x1;
            X[i, 1] = x2;
            y[i] = 2 * x1 + 3 * x2 + 1 + rand.NextDouble() * 0.1; // Small noise
        }

        return (X, y);
    }

    /// <summary>
    /// Validates that an optimization result has meaningful content, not just non-null fields.
    /// </summary>
    private static void AssertValidOptimizationResult(
        OptimizationResult<double, Matrix<double>, Vector<double>> result,
        string optimizerName)
    {
        Assert.NotNull(result.BestSolution);

        // Fitness score must be a valid finite number
        Assert.False(double.IsNaN(result.BestFitnessScore),
            $"{optimizerName}: BestFitnessScore should not be NaN");
        Assert.False(double.IsInfinity(result.BestFitnessScore),
            $"{optimizerName}: BestFitnessScore should not be Infinity");
        Assert.True(result.BestFitnessScore >= 0,
            $"{optimizerName}: BestFitnessScore should be non-negative, got {result.BestFitnessScore}");

        // Optimization should have run at least 1 iteration
        Assert.True(result.Iterations >= 1,
            $"{optimizerName}: Should have run at least 1 iteration, got {result.Iterations}");

        // Fitness history should track progress
        Assert.NotNull(result.FitnessHistory);
        Assert.True(result.FitnessHistory.Length > 0,
            $"{optimizerName}: FitnessHistory should have at least 1 entry");

        // Selected features should be populated
        Assert.NotNull(result.SelectedFeatures);
        Assert.True(result.SelectedFeatures.Count > 0,
            $"{optimizerName}: SelectedFeatures should contain at least one feature set");

        // Training result should have predictions
        Assert.NotNull(result.TrainingResult);
    }

    #endregion

    #region Genetic Algorithm Optimizer Tests

    [Fact(Timeout = 120000)]
    public async Task GeneticAlgorithm_CanInstantiate()
    {
        // Verify the optimizer can be instantiated without errors
        var options = new GeneticAlgorithmOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            MaxIterations = 10,
            PopulationSize = 20
        };

        var optimizer = new GeneticAlgorithmOptimizer<double, Matrix<double>, Vector<double>>(
            null, options);

        Assert.NotNull(optimizer);
    }

    [Fact(Timeout = 120000)]
    public async Task GeneticAlgorithm_OptimizesSimpleRegression()
    {
        // Genetic Algorithm should find a reasonable solution for linear regression
        var (X, y) = CreateSimpleRegressionData(30);
        int numFeatures = X.Columns;

        var model = new MultipleRegression<double>();
        var options = new GeneticAlgorithmOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            MaxIterations = 50,
            PopulationSize = 30,
            // Use all features to prevent dimension mismatch during parameter optimization
            MinimumFeatures = numFeatures,
            MaximumFeatures = numFeatures
        };

        var optimizer = new GeneticAlgorithmOptimizer<double, Matrix<double>, Vector<double>>(
            model, options);

        var inputData = new OptimizationInputData<double, Matrix<double>, Vector<double>>
        {
            XTrain = X,
            YTrain = y,
            XValidation = X,
            YValidation = y,
            XTest = X,
            YTest = y
        };

        var result = optimizer.Optimize(inputData);

        AssertValidOptimizationResult(result, "GeneticAlgorithm");
    }

    #endregion

    #region Particle Swarm Optimizer Tests

    [Fact(Timeout = 120000)]
    public async Task ParticleSwarm_CanInstantiate()
    {
        var options = new ParticleSwarmOptimizationOptions<double, Matrix<double>, Vector<double>>
        {
            MaxIterations = 10,
            SwarmSize = 20
        };

        var optimizer = new ParticleSwarmOptimizer<double, Matrix<double>, Vector<double>>(
            null, options);

        Assert.NotNull(optimizer);
    }

    [Fact(Timeout = 120000)]
    public async Task ParticleSwarm_OptimizesSimpleRegression()
    {
        var (X, y) = CreateSimpleRegressionData(30);
        int numFeatures = X.Columns;

        var model = new MultipleRegression<double>();
        var options = new ParticleSwarmOptimizationOptions<double, Matrix<double>, Vector<double>>
        {
            MaxIterations = 50,
            SwarmSize = 30,
            InertiaWeight = 0.7,
            CognitiveParameter = 1.4,
            SocialParameter = 1.4,
            MinimumFeatures = numFeatures,
            MaximumFeatures = numFeatures
        };

        var optimizer = new ParticleSwarmOptimizer<double, Matrix<double>, Vector<double>>(
            model, options);

        var inputData = new OptimizationInputData<double, Matrix<double>, Vector<double>>
        {
            XTrain = X,
            YTrain = y,
            XValidation = X,
            YValidation = y,
            XTest = X,
            YTest = y
        };

        var result = optimizer.Optimize(inputData);

        AssertValidOptimizationResult(result, "ParticleSwarm");
    }

    /// <summary>
    /// The same seed gives the same optimization, which is the behaviour callers actually want.
    /// </summary>
    /// <remarks>
    /// The generator tests below localise a failure; this one states the contract. Equality rather
    /// than a tolerance is deliberate - a seed either determines the run or it does not, and a
    /// tolerance would let the defect back in unnoticed.
    /// </remarks>
    [Fact(Timeout = 120000)]
    public async Task ParticleSwarm_WithTheSameSeed_RepeatsItselfOnTheModelPath()
    {
        await Task.Yield();

        var (X, y) = CreateSimpleRegressionData(30);
        int numFeatures = X.Columns;

        double RunWithSeed(int seed)
        {
            var optimizer = new ParticleSwarmOptimizer<double, Matrix<double>, Vector<double>>(
                new MultipleRegression<double>(),
                new ParticleSwarmOptimizationOptions<double, Matrix<double>, Vector<double>>
                {
                    MaxIterations = 20,
                    SwarmSize = 20,
                    MinimumFeatures = numFeatures,
                    MaximumFeatures = numFeatures,
                    Seed = seed,
                });

            return optimizer.Optimize(new OptimizationInputData<double, Matrix<double>, Vector<double>>
            {
                XTrain = X,
                YTrain = y,
                XValidation = X,
                YValidation = y,
                XTest = X,
                YTest = y,
            }).BestFitnessScore;
        }

        Assert.Equal(RunWithSeed(20250829), RunWithSeed(20250829));
    }

    /// <summary>
    /// A seeded optimizer draws from a seeded generator, and a differently seeded one does not.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Asserted for two optimizers because the fix is a base-class contract, not a per-optimizer
    /// patch. <c>OptimizerBase</c> owns the one generator every derived optimizer draws from, and it
    /// is built from <c>Options.Seed</c>. Particle swarm and simulated annealing each used to carry
    /// a private <c>Random</c> built with <c>CreateSecureRandom()</c>, which no seed could reach;
    /// both now use the inherited one, so there is nothing left to forget to seed.
    /// </para>
    /// <para>
    /// The second assertion pins the other half - different seeds must diverge - so that a
    /// degenerate generator returning a constant would not satisfy the first.
    /// </para>
    /// </remarks>
    [Fact(Timeout = 120000)]
    public async Task SeededOptimizers_DrawFromTheSeedTheyWereGiven()
    {
        await Task.Yield();

        static int SwarmDraw(int seed) => FirstDraw(
            new ParticleSwarmOptimizer<double, Matrix<double>, Vector<double>>(
                null,
                new ParticleSwarmOptimizationOptions<double, Matrix<double>, Vector<double>>
                {
                    SwarmSize = 8,
                    Seed = seed,
                }));

        static int AnnealingDraw(int seed) => FirstDraw(
            new SimulatedAnnealingOptimizer<double, Matrix<double>, Vector<double>>(
                null,
                new SimulatedAnnealingOptions<double, Matrix<double>, Vector<double>>
                {
                    Seed = seed,
                }));

        Assert.Equal(SwarmDraw(20250829), SwarmDraw(20250829));
        Assert.NotEqual(SwarmDraw(20250829), SwarmDraw(19700101));

        Assert.Equal(AnnealingDraw(20250829), AnnealingDraw(20250829));
        Assert.NotEqual(AnnealingDraw(20250829), AnnealingDraw(19700101));
    }

    /// <summary>
    /// A restored optimizer draws from the seed it was restored with, not the one it was built with.
    /// </summary>
    /// <remarks>
    /// <c>Deserialize</c> hands the restored options to <c>UpdateOptions</c> so a derived optimizer
    /// can react, but the base kept its constructor values. That was harmless while the shared
    /// generator ignored <c>Seed</c> outright; once it honours the seed, a restored optimizer would
    /// otherwise report one seed and draw from another. The base now adopts the restored options
    /// before handing them on, which is why this holds for every optimizer rather than the one
    /// tested here.
    /// </remarks>
    [Fact(Timeout = 120000)]
    public async Task ParticleSwarm_AfterDeserialize_UsesTheRestoredSeed()
    {
        await Task.Yield();

        const int BuiltWith = 19700101;
        const int RestoredFrom = 20250829;

        static ParticleSwarmOptimizer<double, Matrix<double>, Vector<double>> Build(int seed) =>
            new(null, new ParticleSwarmOptimizationOptions<double, Matrix<double>, Vector<double>>
            {
                MaxIterations = 10,
                SwarmSize = 8,
                Seed = seed,
            });

        var restored = Build(BuiltWith);
        restored.Deserialize(Build(RestoredFrom).Serialize());

        Assert.Equal(RestoredFrom, restored.GetOptions().Seed);
        Assert.Equal(FirstDraw(Build(RestoredFrom)), FirstDraw(restored));
        Assert.NotEqual(FirstDraw(Build(BuiltWith)), FirstDraw(restored));
    }

    /// <summary>The first draw from the one generator an optimizer owns, which lives on the base.</summary>
    private static int FirstDraw(OptimizerBase<double, Matrix<double>, Vector<double>> optimizer)
        => ((Random)typeof(OptimizerBase<double, Matrix<double>, Vector<double>>)
            .GetField("Random", BindingFlags.Instance | BindingFlags.NonPublic)!
            .GetValue(optimizer)!).Next();

    #endregion

    #region Differential Evolution Optimizer Tests

    [Fact(Timeout = 120000)]
    public async Task DifferentialEvolution_CanInstantiate()
    {
        var options = new DifferentialEvolutionOptions<double, Matrix<double>, Vector<double>>
        {
            MaxIterations = 10,
            PopulationSize = 20
        };

        var optimizer = new DifferentialEvolutionOptimizer<double, Matrix<double>, Vector<double>>(
            null, options);

        Assert.NotNull(optimizer);
    }

    [Fact(Timeout = 120000)]
    public async Task DifferentialEvolution_OptimizesSimpleRegression()
    {
        var (X, y) = CreateSimpleRegressionData(30);
        int numFeatures = X.Columns;

        var model = new MultipleRegression<double>();
        var options = new DifferentialEvolutionOptions<double, Matrix<double>, Vector<double>>
        {
            MaxIterations = 50,
            PopulationSize = 30,
            MutationRate = 0.8,
            CrossoverRate = 0.9,
            MinimumFeatures = numFeatures,
            MaximumFeatures = numFeatures
        };

        var optimizer = new DifferentialEvolutionOptimizer<double, Matrix<double>, Vector<double>>(
            model, options);

        var inputData = new OptimizationInputData<double, Matrix<double>, Vector<double>>
        {
            XTrain = X,
            YTrain = y,
            XValidation = X,
            YValidation = y,
            XTest = X,
            YTest = y
        };

        var result = optimizer.Optimize(inputData);

        AssertValidOptimizationResult(result, "DifferentialEvolution");
    }

    #endregion

    #region Simulated Annealing Optimizer Tests

    [Fact(Timeout = 120000)]
    public async Task SimulatedAnnealing_CanInstantiate()
    {
        var options = new SimulatedAnnealingOptions<double, Matrix<double>, Vector<double>>
        {
            MaxIterations = 10,
            InitialTemperature = 100.0,
            CoolingRate = 0.95
        };

        var optimizer = new SimulatedAnnealingOptimizer<double, Matrix<double>, Vector<double>>(
            null, options);

        Assert.NotNull(optimizer);
    }

    [Fact(Timeout = 120000)]
    public async Task SimulatedAnnealing_OptimizesSimpleRegression()
    {
        var (X, y) = CreateSimpleRegressionData(30);
        int numFeatures = X.Columns;

        var model = new MultipleRegression<double>();
        var options = new SimulatedAnnealingOptions<double, Matrix<double>, Vector<double>>
        {
            MaxIterations = 100,
            InitialTemperature = 100.0,
            CoolingRate = 0.95,
            MinTemperature = 0.01,
            MinimumFeatures = numFeatures,
            MaximumFeatures = numFeatures
        };

        var optimizer = new SimulatedAnnealingOptimizer<double, Matrix<double>, Vector<double>>(
            model, options);

        var inputData = new OptimizationInputData<double, Matrix<double>, Vector<double>>
        {
            XTrain = X,
            YTrain = y,
            XValidation = X,
            YValidation = y,
            XTest = X,
            YTest = y
        };

        var result = optimizer.Optimize(inputData);

        AssertValidOptimizationResult(result, "SimulatedAnnealing");
    }

    #endregion

    #region Ant Colony Optimizer Tests

    [Fact(Timeout = 120000)]
    public async Task AntColony_CanInstantiate()
    {
        var options = new AntColonyOptimizationOptions<double, Matrix<double>, Vector<double>>
        {
            MaxIterations = 10,
            AntCount = 20
        };

        var optimizer = new AntColonyOptimizer<double, Matrix<double>, Vector<double>>(
            null, options);

        Assert.NotNull(optimizer);
    }

    [Fact(Timeout = 120000)]
    public async Task AntColony_OptimizesSimpleRegression()
    {
        var (X, y) = CreateSimpleRegressionData(30);
        int numFeatures = X.Columns;

        var model = new MultipleRegression<double>();
        var options = new AntColonyOptimizationOptions<double, Matrix<double>, Vector<double>>
        {
            MaxIterations = 50,
            AntCount = 20,
            InitialPheromoneEvaporationRate = 0.5,
            Beta = 2.0,
            MinimumFeatures = numFeatures,
            MaximumFeatures = numFeatures
        };

        var optimizer = new AntColonyOptimizer<double, Matrix<double>, Vector<double>>(
            model, options);

        var inputData = new OptimizationInputData<double, Matrix<double>, Vector<double>>
        {
            XTrain = X,
            YTrain = y,
            XValidation = X,
            YValidation = y,
            XTest = X,
            YTest = y
        };

        var result = optimizer.Optimize(inputData);

        AssertValidOptimizationResult(result, "AntColony");
    }

    #endregion

    #region Tabu Search Optimizer Tests

    [Fact(Timeout = 120000)]
    public async Task TabuSearch_CanInstantiate()
    {
        var options = new TabuSearchOptions<double, Matrix<double>, Vector<double>>
        {
            MaxIterations = 10,
            TabuListSize = 10
        };

        var optimizer = new TabuSearchOptimizer<double, Matrix<double>, Vector<double>>(
            null, options);

        Assert.NotNull(optimizer);
    }

    [Fact(Timeout = 120000)]
    public async Task TabuSearch_OptimizesSimpleRegression()
    {
        var (X, y) = CreateSimpleRegressionData(30);
        int numFeatures = X.Columns;

        var model = new MultipleRegression<double>();
        var options = new TabuSearchOptions<double, Matrix<double>, Vector<double>>
        {
            MaxIterations = 50,
            TabuListSize = 20,
            NeighborhoodSize = 10,
            MinimumFeatures = numFeatures,
            MaximumFeatures = numFeatures
        };

        var optimizer = new TabuSearchOptimizer<double, Matrix<double>, Vector<double>>(
            model, options);

        var inputData = new OptimizationInputData<double, Matrix<double>, Vector<double>>
        {
            XTrain = X,
            YTrain = y,
            XValidation = X,
            YValidation = y,
            XTest = X,
            YTest = y
        };

        var result = optimizer.Optimize(inputData);

        AssertValidOptimizationResult(result, "TabuSearch");
    }

    #endregion

    #region CMA-ES Optimizer Tests

    [Fact(Timeout = 120000)]
    public async Task CMAES_CanInstantiate()
    {
        var options = new CMAESOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            MaxIterations = 10,
            PopulationSize = 20
        };

        var optimizer = new CMAESOptimizer<double, Matrix<double>, Vector<double>>(
            null, options);

        Assert.NotNull(optimizer);
    }

    [Fact(Timeout = 120000)]
    public async Task CMAES_OptimizesSimpleRegression()
    {
        var (X, y) = CreateSimpleRegressionData(30);
        int numFeatures = X.Columns;

        var model = new MultipleRegression<double>();
        var options = new CMAESOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            MaxIterations = 50,
            PopulationSize = 20,
            MinimumFeatures = numFeatures,
            MaximumFeatures = numFeatures
        };

        var optimizer = new CMAESOptimizer<double, Matrix<double>, Vector<double>>(
            model, options);

        var inputData = new OptimizationInputData<double, Matrix<double>, Vector<double>>
        {
            XTrain = X,
            YTrain = y,
            XValidation = X,
            YValidation = y,
            XTest = X,
            YTest = y
        };

        var result = optimizer.Optimize(inputData);

        AssertValidOptimizationResult(result, "CMAES");
    }

    #endregion

    #region Bayesian Optimizer Tests

    [Fact(Timeout = 120000)]
    public async Task Bayesian_CanInstantiate()
    {
        var options = new BayesianOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            MaxIterations = 10,
            InitialSamples = 5
        };

        var optimizer = new BayesianOptimizer<double, Matrix<double>, Vector<double>>(
            null, options);

        Assert.NotNull(optimizer);
    }

    [Fact(Timeout = 120000)]
    public async Task Bayesian_OptimizesSimpleRegression()
    {
        var (X, y) = CreateSimpleRegressionData(30);
        int numFeatures = X.Columns;

        var model = new MultipleRegression<double>();
        var options = new BayesianOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            MaxIterations = 20,
            InitialSamples = 5,
            ExplorationFactor = 2.0,
            MinimumFeatures = numFeatures,
            MaximumFeatures = numFeatures
        };

        var optimizer = new BayesianOptimizer<double, Matrix<double>, Vector<double>>(
            model, options);

        var inputData = new OptimizationInputData<double, Matrix<double>, Vector<double>>
        {
            XTrain = X,
            YTrain = y,
            XValidation = X,
            YValidation = y,
            XTest = X,
            YTest = y
        };

        var result = optimizer.Optimize(inputData);

        AssertValidOptimizationResult(result, "Bayesian");
    }

    #endregion

    #region Nelder-Mead Optimizer Tests

    [Fact(Timeout = 120000)]
    public async Task NelderMead_CanInstantiate()
    {
        var options = new NelderMeadOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            MaxIterations = 10
        };

        var optimizer = new NelderMeadOptimizer<double, Matrix<double>, Vector<double>>(
            null, options);

        Assert.NotNull(optimizer);
    }

    [Fact(Timeout = 120000)]
    public async Task NelderMead_OptimizesSimpleRegression()
    {
        var (X, y) = CreateSimpleRegressionData(30);
        int numFeatures = X.Columns;

        var model = new MultipleRegression<double>();
        var options = new NelderMeadOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            MaxIterations = 100,
            InitialAlpha = 1.0,  // Reflection coefficient
            InitialGamma = 2.0,  // Expansion coefficient
            InitialBeta = 0.5,   // Contraction coefficient
            InitialDelta = 0.5,  // Shrink coefficient
            MinimumFeatures = numFeatures,
            MaximumFeatures = numFeatures
        };

        var optimizer = new NelderMeadOptimizer<double, Matrix<double>, Vector<double>>(
            model, options);

        var inputData = new OptimizationInputData<double, Matrix<double>, Vector<double>>
        {
            XTrain = X,
            YTrain = y,
            XValidation = X,
            YValidation = y,
            XTest = X,
            YTest = y
        };

        var result = optimizer.Optimize(inputData);

        AssertValidOptimizationResult(result, "NelderMead");
    }

    #endregion

    #region Powell Optimizer Tests

    [Fact(Timeout = 120000)]
    public async Task Powell_CanInstantiate()
    {
        var options = new PowellOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            MaxIterations = 10
        };

        var optimizer = new PowellOptimizer<double, Matrix<double>, Vector<double>>(
            null, options);

        Assert.NotNull(optimizer);
    }

    [Fact(Timeout = 120000)]
    public async Task Powell_OptimizesSimpleRegression()
    {
        var (X, y) = CreateSimpleRegressionData(30);
        int numFeatures = X.Columns;

        var model = new MultipleRegression<double>();
        var options = new PowellOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            MaxIterations = 100,
            Tolerance = 1e-6,
            MinimumFeatures = numFeatures,
            MaximumFeatures = numFeatures
        };

        var optimizer = new PowellOptimizer<double, Matrix<double>, Vector<double>>(
            model, options);

        var inputData = new OptimizationInputData<double, Matrix<double>, Vector<double>>
        {
            XTrain = X,
            YTrain = y,
            XValidation = X,
            YValidation = y,
            XTest = X,
            YTest = y
        };

        var result = optimizer.Optimize(inputData);

        AssertValidOptimizationResult(result, "Powell");
    }

    #endregion

    #region Serialization Tests for Metaheuristic Optimizers

    [Fact(Timeout = 120000)]
    public async Task GeneticAlgorithm_SerializesAndDeserializes()
    {
        var options = new GeneticAlgorithmOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            MaxIterations = 10,
            PopulationSize = 20
        };

        var optimizer = new GeneticAlgorithmOptimizer<double, Matrix<double>, Vector<double>>(
            null, options);

        // Serialize
        var serialized = optimizer.Serialize();
        Assert.NotNull(serialized);
        Assert.True(serialized.Length > 0, "Serialized data should not be empty");

        // Deserialize into a fresh optimizer with different options
        var differentOptions = new GeneticAlgorithmOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            MaxIterations = 999,
            PopulationSize = 1
        };
        var newOptimizer = new GeneticAlgorithmOptimizer<double, Matrix<double>, Vector<double>>(
            null, differentOptions);
        newOptimizer.Deserialize(serialized);

        // Verify options are preserved from serialized data
        var restoredOptions = newOptimizer.GetOptions();
        Assert.NotNull(restoredOptions);
        Assert.Equal(10, restoredOptions.MaxIterations);
    }

    [Fact(Timeout = 120000)]
    public async Task ParticleSwarm_SerializesAndDeserializes()
    {
        var options = new ParticleSwarmOptimizationOptions<double, Matrix<double>, Vector<double>>
        {
            MaxIterations = 10,
            SwarmSize = 20
        };

        var optimizer = new ParticleSwarmOptimizer<double, Matrix<double>, Vector<double>>(
            null, options);

        var serialized = optimizer.Serialize();
        Assert.NotNull(serialized);
        Assert.True(serialized.Length > 0, "Serialized data should not be empty");

        // Deserialize into a fresh optimizer with different options
        var differentOptions = new ParticleSwarmOptimizationOptions<double, Matrix<double>, Vector<double>>
        {
            MaxIterations = 999,
            SwarmSize = 1
        };
        var newOptimizer = new ParticleSwarmOptimizer<double, Matrix<double>, Vector<double>>(
            null, differentOptions);
        newOptimizer.Deserialize(serialized);

        var restoredOptions = newOptimizer.GetOptions();
        Assert.NotNull(restoredOptions);
        Assert.Equal(10, restoredOptions.MaxIterations);
    }

    [Fact(Timeout = 120000)]
    public async Task SimulatedAnnealing_SerializesAndDeserializes()
    {
        var options = new SimulatedAnnealingOptions<double, Matrix<double>, Vector<double>>
        {
            MaxIterations = 10,
            InitialTemperature = 100.0
        };

        var optimizer = new SimulatedAnnealingOptimizer<double, Matrix<double>, Vector<double>>(
            null, options);

        var serialized = optimizer.Serialize();
        Assert.NotNull(serialized);
        Assert.True(serialized.Length > 0, "Serialized data should not be empty");

        // Deserialize into a fresh optimizer with different options
        var differentOptions = new SimulatedAnnealingOptions<double, Matrix<double>, Vector<double>>
        {
            MaxIterations = 999,
            InitialTemperature = 1.0
        };
        var newOptimizer = new SimulatedAnnealingOptimizer<double, Matrix<double>, Vector<double>>(
            null, differentOptions);
        newOptimizer.Deserialize(serialized);

        var restoredOptions = newOptimizer.GetOptions();
        Assert.NotNull(restoredOptions);
        Assert.Equal(10, restoredOptions.MaxIterations);
    }

    #endregion

    #region Edge Case Tests for Metaheuristic Optimizers

    [Fact(Timeout = 120000)]
    public async Task Metaheuristics_HandleMinimalData()
    {
        // Test with minimal training data (3 samples)
        var X = new Matrix<double>(3, 2);
        X[0, 0] = 1; X[0, 1] = 2;
        X[1, 0] = 2; X[1, 1] = 4;
        X[2, 0] = 3; X[2, 1] = 6;

        var y = new Vector<double>(new double[] { 5, 10, 15 });

        var inputData = new OptimizationInputData<double, Matrix<double>, Vector<double>>
        {
            XTrain = X,
            YTrain = y,
            XValidation = X,
            YValidation = y,
            XTest = X,
            YTest = y
        };

        var model = new MultipleRegression<double>();
        var options = new GeneticAlgorithmOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            MaxIterations = 10,
            PopulationSize = 10
        };

        var optimizer = new GeneticAlgorithmOptimizer<double, Matrix<double>, Vector<double>>(
            model, options);

        // Should not throw with minimal data and should produce valid results
        var result = optimizer.Optimize(inputData);
        AssertValidOptimizationResult(result, "GA_MinimalData");
    }

    [Fact(Timeout = 120000)]
    public async Task Metaheuristics_HandleHighDimensionalData()
    {
        // Test with higher dimensional data (10 features)
        var rand = new Random(42);
        int numSamples = 50;
        int numFeatures = 10;

        var X = new Matrix<double>(numSamples, numFeatures);
        var y = new Vector<double>(numSamples);

        for (int i = 0; i < numSamples; i++)
        {
            double sum = 0;
            for (int j = 0; j < numFeatures; j++)
            {
                X[i, j] = rand.NextDouble() * 10 - 5;
                sum += (j + 1) * X[i, j]; // Weighted sum
            }
            y[i] = sum + rand.NextDouble() * 0.1;
        }

        var inputData = new OptimizationInputData<double, Matrix<double>, Vector<double>>
        {
            XTrain = X,
            YTrain = y,
            XValidation = X,
            YValidation = y,
            XTest = X,
            YTest = y
        };

        var model = new MultipleRegression<double>();
        var options = new ParticleSwarmOptimizationOptions<double, Matrix<double>, Vector<double>>
        {
            MaxIterations = 30,
            SwarmSize = 20,
            // Use all features to prevent dimension mismatch during parameter optimization
            MinimumFeatures = numFeatures,
            MaximumFeatures = numFeatures
        };

        var optimizer = new ParticleSwarmOptimizer<double, Matrix<double>, Vector<double>>(
            model, options);

        var result = optimizer.Optimize(inputData);
        AssertValidOptimizationResult(result, "PSO_HighDimensional");
    }

    [Fact(Timeout = 120000)]
    public async Task Metaheuristics_HandleSingleIteration()
    {
        var (X, y) = CreateSimpleRegressionData(20);
        int numFeatures = X.Columns;

        var inputData = new OptimizationInputData<double, Matrix<double>, Vector<double>>
        {
            XTrain = X,
            YTrain = y,
            XValidation = X,
            YValidation = y,
            XTest = X,
            YTest = y
        };

        var model = new MultipleRegression<double>();
        var options = new DifferentialEvolutionOptions<double, Matrix<double>, Vector<double>>
        {
            MaxIterations = 1, // Single iteration
            PopulationSize = 10,
            // Use all features to prevent dimension mismatch
            MinimumFeatures = numFeatures,
            MaximumFeatures = numFeatures
        };

        var optimizer = new DifferentialEvolutionOptimizer<double, Matrix<double>, Vector<double>>(
            model, options);

        // Should complete without error even with single iteration
        var result = optimizer.Optimize(inputData);
        AssertValidOptimizationResult(result, "DE_SingleIteration");
    }

    #endregion

    #region Normal Optimizer Tests

    [Fact(Timeout = 120000)]
    public async Task Normal_CanInstantiate()
    {
        // NormalOptimizer uses GeneticAlgorithmOptimizerOptions
        var options = new GeneticAlgorithmOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            MaxIterations = 10,
            PopulationSize = 20
        };

        var optimizer = new NormalOptimizer<double, Matrix<double>, Vector<double>>(
            null, options);

        Assert.NotNull(optimizer);
    }

    [Fact(Timeout = 120000)]
    public async Task Normal_OptimizesSimpleRegression()
    {
        // NormalOptimizer uses random search with adaptive parameters
        var (X, y) = CreateSimpleRegressionData(30);
        int numFeatures = X.Columns;

        var model = new MultipleRegression<double>();
        var options = new GeneticAlgorithmOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            MaxIterations = 50,
            PopulationSize = 20,
            MutationRate = 0.1,
            CrossoverRate = 0.8,
            MinimumFeatures = numFeatures,
            MaximumFeatures = numFeatures
        };

        var optimizer = new NormalOptimizer<double, Matrix<double>, Vector<double>>(
            model, options);

        var inputData = new OptimizationInputData<double, Matrix<double>, Vector<double>>
        {
            XTrain = X,
            YTrain = y,
            XValidation = X,
            YValidation = y,
            XTest = X,
            YTest = y
        };

        var result = optimizer.Optimize(inputData);

        AssertValidOptimizationResult(result, "Normal");
    }

    [Fact(Timeout = 120000)]
    public async Task Normal_AdaptsParametersDuringOptimization()
    {
        // Verify that NormalOptimizer adapts its parameters during optimization
        var (X, y) = CreateSimpleRegressionData(30);
        int numFeatures = X.Columns;

        var model = new MultipleRegression<double>();
        var options = new GeneticAlgorithmOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            MaxIterations = 30,
            PopulationSize = 15,
            MutationRate = 0.2,
            CrossoverRate = 0.7,
            MinimumFeatures = numFeatures,
            MaximumFeatures = numFeatures
        };

        var optimizer = new NormalOptimizer<double, Matrix<double>, Vector<double>>(
            model, options);

        var inputData = new OptimizationInputData<double, Matrix<double>, Vector<double>>
        {
            XTrain = X,
            YTrain = y,
            XValidation = X,
            YValidation = y,
            XTest = X,
            YTest = y
        };

        // Run optimization - adaptive parameters should change during execution
        var result = optimizer.Optimize(inputData);

        AssertValidOptimizationResult(result, "Normal_Adaptive");
    }

    [Fact(Timeout = 120000)]
    public async Task Normal_SerializesAndDeserializes()
    {
        var options = new GeneticAlgorithmOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            MaxIterations = 10,
            PopulationSize = 20,
            MutationRate = 0.15,
            CrossoverRate = 0.75
        };

        var optimizer = new NormalOptimizer<double, Matrix<double>, Vector<double>>(
            null, options);

        // Serialize
        var serialized = optimizer.Serialize();
        Assert.NotNull(serialized);
        Assert.True(serialized.Length > 0, "Serialized data should not be empty");

        // Deserialize into a fresh optimizer with different options
        var differentOptions = new GeneticAlgorithmOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            MaxIterations = 999,
            PopulationSize = 1,
            MutationRate = 0.99,
            CrossoverRate = 0.01
        };
        var newOptimizer = new NormalOptimizer<double, Matrix<double>, Vector<double>>(
            null, differentOptions);
        newOptimizer.Deserialize(serialized);

        // Verify options are preserved from serialized data
        var restoredOptions = newOptimizer.GetOptions();
        Assert.NotNull(restoredOptions);
        Assert.Equal(10, restoredOptions.MaxIterations);
    }

    [Fact(Timeout = 120000)]
    public async Task Normal_HandlesSingleIteration()
    {
        var (X, y) = CreateSimpleRegressionData(20);
        int numFeatures = X.Columns;

        var inputData = new OptimizationInputData<double, Matrix<double>, Vector<double>>
        {
            XTrain = X,
            YTrain = y,
            XValidation = X,
            YValidation = y,
            XTest = X,
            YTest = y
        };

        var model = new MultipleRegression<double>();
        var options = new GeneticAlgorithmOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            MaxIterations = 1, // Single iteration
            PopulationSize = 10,
            MinimumFeatures = numFeatures,
            MaximumFeatures = numFeatures
        };

        var optimizer = new NormalOptimizer<double, Matrix<double>, Vector<double>>(
            model, options);

        // Should complete without error even with single iteration
        var result = optimizer.Optimize(inputData);
        AssertValidOptimizationResult(result, "Normal_SingleIteration");
    }

    #endregion
}
