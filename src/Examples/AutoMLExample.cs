using AiDotNet.AutoML;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using System;
using System.Linq;
using System.Threading.Tasks;

namespace AiDotNet.Examples
{
    /// <summary>
    /// Example demonstrating how to use AutoML for automatic model selection and hyperparameter tuning
    /// </summary>
    public class AutoMLExample
    {
        public static async Task RunAsync()
        {
            Console.WriteLine("=== AutoML Example ===\n");

            // Generate synthetic regression data
            var random = new Random(42);
            int numSamples = 200;
            int numFeatures = 5;

            // Generate features
            var features = new double[numSamples][];
            for (int i = 0; i < numSamples; i++)
            {
                features[i] = new double[numFeatures];
                for (int j = 0; j < numFeatures; j++)
                {
                    features[i][j] = random.NextDouble() * 10 - 5; // Random values between -5 and 5
                }
            }

            // Generate targets (linear combination with noise)
            var targets = new double[numSamples];
            var trueCoefficients = new[] { 2.5, -1.3, 0.8, -0.5, 1.2 };
            for (int i = 0; i < numSamples; i++)
            {
                targets[i] = 3.0; // intercept
                for (int j = 0; j < numFeatures; j++)
                {
                    targets[i] += trueCoefficients[j] * features[i][j];
                }
                targets[i] += random.NextDouble() * 2 - 1; // Add noise
            }

            // Split into training and validation sets
            int trainSize = (int)(numSamples * 0.8);
            var trainFeatures = features.Take(trainSize).ToArray();
            var trainTargets = targets.Take(trainSize).ToArray();
            var validFeatures = features.Skip(trainSize).ToArray();
            var validTargets = targets.Skip(trainSize).ToArray();

            // Example 1: Grid Search AutoML
            Console.WriteLine("1. Grid Search AutoML");
            Console.WriteLine("---------------------");
            await RunGridSearchExample(trainFeatures, trainTargets, validFeatures, validTargets);

            // Example 2: Random Search AutoML
            Console.WriteLine("\n2. Random Search AutoML");
            Console.WriteLine("-----------------------");
            await RunRandomSearchExample(trainFeatures, trainTargets, validFeatures, validTargets);

            // Example 3: Bayesian Optimization AutoML
            Console.WriteLine("\n3. Bayesian Optimization AutoML");
            Console.WriteLine("-------------------------------");
            await RunBayesianOptimizationExample(trainFeatures, trainTargets, validFeatures, validTargets);
        }

        private static async Task RunGridSearchExample(
            double[][] trainFeatures, 
            double[] trainTargets,
            double[][] validFeatures,
            double[] validTargets)
        {
            var gridSearch = new GridSearchAutoML<double, Matrix<double>, Vector<double>>(stepsPerDimension: 5, seed: 42);

            // Configure candidate models
            gridSearch.SetCandidateModels(new List<ModelType>
            {
                ModelType.LinearRegression,
                ModelType.DecisionTree,
                ModelType.RandomForest
            });

            // Set custom search space
            var searchSpace = new Dictionary<string, ParameterRange>
            {
                ["learningRate"] = new ParameterRange
                {
                    MinValue = 0.01,
                    MaxValue = 0.1,
                    Type = ParameterType.Continuous
                }
            };
            gridSearch.SetSearchSpace(searchSpace);

            // Set optimization metric
            gridSearch.SetOptimizationMetric(MetricType.RootMeanSquaredError, maximize: false);

            // Enable early stopping
            gridSearch.EnableEarlyStopping(patience: 20, minDelta: 0.0001);

            try
            {
                // Run search
                var bestModel = await gridSearch.SearchAsync(
                    new Matrix<double>(trainFeatures),
                    new Vector<double>(trainTargets),
                    new Matrix<double>(validFeatures),
                    new Vector<double>(validTargets),
                    TimeSpan.FromMinutes(5));

                Console.WriteLine($"Best model found: {gridSearch.BestModel?.GetType().Name ?? "Unknown"}");
                Console.WriteLine($"Best score (RMSE): {gridSearch.BestScore:F4}");
                Console.WriteLine($"Total trials: {gridSearch.GetTrialHistory().Count}");
            }
            catch (NotImplementedException)
            {
                Console.WriteLine("Note: Model creation needs to be integrated with PredictionModelBuilder");
            }
        }

        private static async Task RunRandomSearchExample(
            double[][] trainFeatures, 
            double[] trainTargets,
            double[][] validFeatures,
            double[] validTargets)
        {
            var randomSearch = new RandomSearchAutoML<double, Matrix<double>, Vector<double>>(maxTrials: 50, seed: 42);

            // Configure models and metrics
            randomSearch.SetCandidateModels(new List<ModelType>
            {
                ModelType.LinearRegression,
                ModelType.DecisionTree,
                ModelType.RandomForest,
                ModelType.GradientBoosting
            });

            randomSearch.SetOptimizationMetric(MetricType.RSquared, maximize: true);

            try
            {
                // Run search
                var bestModel = await randomSearch.SearchAsync(
                    new Matrix<double>(trainFeatures),
                    new Vector<double>(trainTargets),
                    new Matrix<double>(validFeatures),
                    new Vector<double>(validTargets),
                    TimeSpan.FromMinutes(3));

                Console.WriteLine($"Best model found: {randomSearch.BestModel?.GetType().Name ?? "Unknown"}");
                Console.WriteLine($"Best score (R²): {randomSearch.BestScore:F4}");
                Console.WriteLine($"Total trials: {randomSearch.GetTrialHistory().Count}");

                // Show top 5 trials
                var topTrials = randomSearch.GetTrialHistory()
                    .OrderByDescending(t => t.Score)
                    .Take(5);

                Console.WriteLine("\nTop 5 trials:");
                foreach (var trial in topTrials)
                {
                    Console.WriteLine($"  Trial {trial.TrialId}: Score={trial.Score:F4}, Model={trial.ModelType}");
                }
            }
            catch (NotImplementedException)
            {
                Console.WriteLine("Note: Model creation needs to be integrated with PredictionModelBuilder");
            }
        }

        private static async Task RunBayesianOptimizationExample(
            double[][] trainFeatures, 
            double[] trainTargets,
            double[][] validFeatures,
            double[] validTargets)
        {
            var bayesianOpt = new BayesianOptimizationAutoML<double, Matrix<double>, Vector<double>>(
                numInitialPoints: 10, 
                explorationWeight: 2.0, 
                seed: 42);

            // Configure models
            bayesianOpt.SetCandidateModels(new List<ModelType>
            {
                ModelType.LinearRegression,
                ModelType.RandomForest,
                ModelType.GradientBoosting,
                ModelType.SupportVectorMachine
            });

            // Set metric
            bayesianOpt.SetOptimizationMetric(MetricType.MeanAbsoluteError, maximize: false);

            // Add constraints
            bayesianOpt.SetConstraints(new List<SearchConstraint>
            {
                new SearchConstraint
                {
                    Name = "MaxTrainingTime",
                    Type = ConstraintType.MaxInferenceTime,
                    Value = TimeSpan.FromSeconds(10)
                }
            });

            try
            {
                // Run search
                var bestModel = await bayesianOpt.SearchAsync(
                    new Matrix<double>(trainFeatures),
                    new Vector<double>(trainTargets),
                    new Matrix<double>(validFeatures),
                    new Vector<double>(validTargets),
                    TimeSpan.FromMinutes(5));

                Console.WriteLine($"Best model found: {bayesianOpt.BestModel?.GetType().Name ?? "Unknown"}");
                Console.WriteLine($"Best score (MAE): {bayesianOpt.BestScore:F4}");
                Console.WriteLine($"Total trials: {bayesianOpt.GetTrialHistory().Count}");

                // Show trial progression
                var history = bayesianOpt.GetTrialHistory();
                if (history.Count > 0)
                {
                    Console.WriteLine("\nScore progression:");
                    for (int i = 0; i < Math.Min(5, history.Count); i++)
                    {
                        var trial = history[i];
                        Console.WriteLine($"  Trial {trial.TrialId}: Score={trial.Score:F4}");
                    }
                    if (history.Count > 5)
                    {
                        Console.WriteLine("  ...");
                        var lastTrial = history.Last();
                        Console.WriteLine($"  Trial {lastTrial.TrialId}: Score={lastTrial.Score:F4}");
                    }
                }
            }
            catch (NotImplementedException)
            {
                Console.WriteLine("Note: Model creation needs to be integrated with PredictionModelBuilder");
            }
        }

        /// <summary>
        /// Demonstrates using the hyperparameter space directly
        /// </summary>
        public static void DemonstrateHyperparameterSpace()
        {
            Console.WriteLine("\n=== Hyperparameter Space Example ===\n");

            var space = new HyperparameterSpace(seed: 42);

            // Define search space
            space.AddContinuous("learningRate", 0.001, 1.0, logScale: true);
            space.AddInteger("numTrees", 10, 100);
            space.AddCategorical("criterion", "gini", "entropy");
            space.AddBoolean("bootstrap");

            // Sample random configurations
            Console.WriteLine("Random samples from search space:");
            for (int i = 0; i < 5; i++)
            {
                var sample = space.Sample();
                Console.WriteLine($"\nSample {i + 1}:");
                foreach (var (param, value) in sample)
                {
                    Console.WriteLine($"  {param}: {value}");
                }
            }

            // Generate grid
            var grid = space.GenerateGrid(stepsPerDimension: 3);
            Console.WriteLine($"\nTotal grid points (3 steps per dimension): {grid.Count}");
            Console.WriteLine($"Expected combinations: {space.GetTotalCombinations(3)}");
        }
    }
}