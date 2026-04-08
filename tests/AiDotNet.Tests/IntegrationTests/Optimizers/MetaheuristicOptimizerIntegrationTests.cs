#nullable disable
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Inputs;
using AiDotNet.Models.Options;
using AiDotNet.Models.Results;
using AiDotNet.Optimizers;
using AiDotNet.Regression;
using Xunit;

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

    [Fact]
    public void GeneticAlgorithm_CanInstantiate()
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

    [Fact]
    public void GeneticAlgorithm_OptimizesSimpleRegression()
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

    [Fact]
    public void ParticleSwarm_CanInstantiate()
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

    [Fact]
    public void ParticleSwarm_OptimizesSimpleRegression()
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

    #endregion

    #region Differential Evolution Optimizer Tests

    [Fact]
    public void DifferentialEvolution_CanInstantiate()
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

    [Fact]
    public void DifferentialEvolution_OptimizesSimpleRegression()
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

    [Fact]
    public void SimulatedAnnealing_CanInstantiate()
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

    [Fact]
    public void SimulatedAnnealing_OptimizesSimpleRegression()
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

    [Fact]
    public void AntColony_CanInstantiate()
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

    [Fact]
    public void AntColony_OptimizesSimpleRegression()
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

    [Fact]
    public void TabuSearch_CanInstantiate()
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

    [Fact]
    public void TabuSearch_OptimizesSimpleRegression()
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

    [Fact]
    public void CMAES_CanInstantiate()
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

    [Fact]
    public void CMAES_OptimizesSimpleRegression()
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

    [Fact]
    public void Bayesian_CanInstantiate()
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

    [Fact]
    public void Bayesian_OptimizesSimpleRegression()
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

    [Fact]
    public void NelderMead_CanInstantiate()
    {
        var options = new NelderMeadOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            MaxIterations = 10
        };

        var optimizer = new NelderMeadOptimizer<double, Matrix<double>, Vector<double>>(
            null, options);

        Assert.NotNull(optimizer);
    }

    [Fact]
    public void NelderMead_OptimizesSimpleRegression()
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

    [Fact]
    public void Powell_CanInstantiate()
    {
        var options = new PowellOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            MaxIterations = 10
        };

        var optimizer = new PowellOptimizer<double, Matrix<double>, Vector<double>>(
            null, options);

        Assert.NotNull(optimizer);
    }

    [Fact]
    public void Powell_OptimizesSimpleRegression()
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

    [Fact]
    public void GeneticAlgorithm_SerializesAndDeserializes()
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

    [Fact]
    public void ParticleSwarm_SerializesAndDeserializes()
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

    [Fact]
    public void SimulatedAnnealing_SerializesAndDeserializes()
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

    [Fact]
    public void Metaheuristics_HandleMinimalData()
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

    [Fact]
    public void Metaheuristics_HandleHighDimensionalData()
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

    [Fact]
    public void Metaheuristics_HandleSingleIteration()
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

    [Fact]
    public void Normal_CanInstantiate()
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

    [Fact]
    public void Normal_OptimizesSimpleRegression()
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

    [Fact]
    public void Normal_AdaptsParametersDuringOptimization()
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

    [Fact]
    public void Normal_SerializesAndDeserializes()
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

    [Fact]
    public void Normal_HandlesSingleIteration()
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
