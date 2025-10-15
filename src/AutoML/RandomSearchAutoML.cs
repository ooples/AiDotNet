using AiDotNet.Enums;
using AiDotNet.Interfaces;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace AiDotNet.AutoML
{
    /// <summary>
    /// Implements AutoML using random search over the hyperparameter space
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations</typeparam>
    /// <typeparam name="TInput">The input data type</typeparam>
    /// <typeparam name="TOutput">The output data type</typeparam>
    public class RandomSearchAutoML<T, TInput, TOutput> : AutoMLModelBase<T, TInput, TOutput>
    {
        private readonly HyperparameterSpace _hyperparameterSpace = default!;
        private readonly int _maxTrials;
        private readonly Random _random = default!;

        /// <summary>
        /// Initializes a new instance of the RandomSearchAutoML class
        /// </summary>
        /// <param name="maxTrials">Maximum number of trials to run</param>
        /// <param name="seed">Random seed for reproducibility</param>
        public RandomSearchAutoML(int maxTrials = 100, int? seed = null)
        {
            _maxTrials = maxTrials;
            _random = seed.HasValue ? new Random(seed.Value) : new Random();
            _hyperparameterSpace = new HyperparameterSpace(seed);
        }

        /// <summary>
        /// Searches for the best model configuration using random search
        /// </summary>
        public override async Task<IFullModel<T, TInput, TOutput>> SearchAsync(
            TInput inputs,
            TOutput targets,
            TInput validationInputs,
            TOutput validationTargets,
            TimeSpan timeLimit,
            CancellationToken cancellationToken = default)
        {
            Status = AutoMLStatus.Running;
            var startTime = DateTime.UtcNow;

            try
            {
                // Initialize search space if not already set
                if (_candidateModels.Count == 0)
                {
                    SetDefaultCandidateModels();
                }

                Console.WriteLine($"Random search: up to {_maxTrials} trials with time limit {timeLimit}");

                while (_trialHistory.Count < _maxTrials)
                {
                    // Check time limit
                    if (DateTime.UtcNow - startTime > timeLimit)
                    {
                        Console.WriteLine("Time limit reached");
                        break;
                    }

                    // Check cancellation
                    if (cancellationToken.IsCancellationRequested)
                    {
                        Status = AutoMLStatus.Cancelled;
                        break;
                    }

                    // Check early stopping
                    if (ShouldStop())
                    {
                        Console.WriteLine($"Early stopping triggered after {_trialsSinceImprovement} trials without improvement");
                        break;
                    }

                    // Sample random configuration
                    var parameters = await SuggestNextTrialAsync();
                    
                    // Select random model type
                    var modelType = _candidateModels[_random.Next(_candidateModels.Count)];
                    
                    var trialStartTime = DateTime.UtcNow;

                    try
                    {
                        // Create and train model
                        var model = await CreateModelAsync(modelType, parameters);
                        model.Train(inputs, targets);

                        // Evaluate model
                        var score = await EvaluateModelAsync(model, validationInputs, validationTargets);
                        var duration = DateTime.UtcNow - trialStartTime;

                        // Report results
                        await ReportTrialResultAsync(parameters, score, duration);

                        // Update best model
                        bool isBetter = _maximize ? score > BestScore : score < BestScore;
                        if (isBetter)
                        {
                            BestModel = model;
                            Console.WriteLine($"New best score: {score:F4} (Trial {_trialHistory.Count})");
                        }

                        // Log progress
                        if (_trialHistory.Count % 10 == 0)
                        {
                            Console.WriteLine($"Progress: {_trialHistory.Count}/{_maxTrials} trials completed, Best score: {BestScore:F4}");
                        }
                    }
                    catch (Exception ex)
                    {
                        // Log failed trial
                        var trial = new TrialResult
                        {
                            TrialId = _trialHistory.Count + 1,
                            Parameters = new Dictionary<string, object>(parameters),
                            Score = _maximize ? double.NegativeInfinity : double.PositiveInfinity,
                            Duration = DateTime.UtcNow - trialStartTime,
                            ModelType = modelType,
                            Timestamp = DateTime.UtcNow,
                            Status = TrialStatus.Failed,
                            ErrorMessage = ex.Message
                        };

                        lock (_lock)
                        {
                            _trialHistory.Add(trial);
                        }

                        Console.WriteLine($"Trial {trial.TrialId} failed: {ex.Message}");
                    }
                }

                Status = BestModel != null ? AutoMLStatus.Completed : AutoMLStatus.Failed;

                if (BestModel == null)
                {
                    throw new InvalidOperationException("No valid model found during search");
                }

                Console.WriteLine($"Random search completed. Total trials: {_trialHistory.Count}, Best score: {BestScore:F4}");
                return BestModel;
            }
            catch (Exception ex)
            {
                Status = AutoMLStatus.Failed;
                throw new InvalidOperationException($"Random search failed: {ex.Message}", ex);
            }
        }

        /// <summary>
        /// Suggests the next hyperparameters to try
        /// </summary>
        public override async Task<Dictionary<string, object>> SuggestNextTrialAsync()
        {
            return await Task.Run((Func<Dictionary<string, object>>)(() =>
            {
                // Get a random model type
                var modelType = _candidateModels[_random.Next(_candidateModels.Count)];

                // Get model-specific search space
                var modelSearchSpace = GetDefaultSearchSpace(modelType);

                // Merge with user-defined search space
                foreach (var kvp in _searchSpace)
                {
                    modelSearchSpace[kvp.Key] = kvp.Value;
                }

                // Create hyperparameter space for sampling
                var space = new HyperparameterSpace(_random.Next());
                foreach (var kvp in modelSearchSpace)
                {
                    var name = kvp.Key;
                    var range = kvp.Value;
                    switch (range.Type)
                    {
                        case ParameterType.Continuous:
                            space.AddContinuous(
                                name,
                                Convert.ToDouble(range.MinValue),
                                Convert.ToDouble(range.MaxValue),
                                range.LogScale);
                            break;

                        case ParameterType.Integer:
                            space.AddInteger(
                                name,
                                Convert.ToInt32(range.MinValue),
                                Convert.ToInt32(range.MaxValue));
                            break;

                        case ParameterType.Categorical:
                            if (range.CategoricalValues != null)
                                space.AddCategorical(name, range.CategoricalValues);
                            break;

                        case ParameterType.Boolean:
                            space.AddBoolean(name);
                            break;
                    }
                }

                // Sample parameters
                var parameters = space.Sample();
                parameters["__modelType__"] = modelType;

                return parameters;
            }));
        }

        /// <summary>
        /// Sets the search space for hyperparameters
        /// </summary>
        public override void SetSearchSpace(Dictionary<string, ParameterRange> searchSpace)
        {
            base.SetSearchSpace(searchSpace);
            
            // Update hyperparameter space
            foreach (var kvp in searchSpace)
            {
                var name = kvp.Key;
                var range = kvp.Value;
                switch (range.Type)
                {
                    case ParameterType.Continuous:
                        _hyperparameterSpace.AddContinuous(
                            name,
                            Convert.ToDouble(range.MinValue),
                            Convert.ToDouble(range.MaxValue),
                            range.LogScale);
                        break;
                    
                    case ParameterType.Integer:
                        _hyperparameterSpace.AddInteger(
                            name,
                            Convert.ToInt32(range.MinValue),
                            Convert.ToInt32(range.MaxValue));
                        break;
                    
                    case ParameterType.Categorical:
                        if (range.CategoricalValues != null)
                            _hyperparameterSpace.AddCategorical(name, range.CategoricalValues);
                        break;
                    
                    case ParameterType.Boolean:
                        _hyperparameterSpace.AddBoolean(name);
                        break;
                }
            }
        }

        /// <summary>
        /// Creates a model instance for the given type and parameters
        /// </summary>
        protected override Task<IFullModel<T, TInput, TOutput>> CreateModelAsync(ModelType modelType, Dictionary<string, object> parameters)
        {
            // This would use PredictionModelBuilder or a factory to create models
            // For now, returning a placeholder
            throw new NotImplementedException("Model creation should be implemented using PredictionModelBuilder");
        }


        /// <summary>
        /// Gets the default search space for a model type
        /// </summary>
        protected override Dictionary<string, ParameterRange> GetDefaultSearchSpace(ModelType modelType)
        {
            var searchSpace = new Dictionary<string, ParameterRange>();

            // Add common hyperparameters based on model type
            switch (modelType)
            {
                case ModelType.LinearRegression:
                    searchSpace["regularization"] = new ParameterRange
                    {
                        MinValue = 0.0001,
                        MaxValue = 10.0,
                        Type = ParameterType.Continuous,
                        LogScale = true
                    };
                    searchSpace["regularizationType"] = new ParameterRange
                    {
                        Type = ParameterType.Categorical,
                        CategoricalValues = new object[] { "L1", "L2", "ElasticNet" }
                    };
                    break;

                case ModelType.DecisionTree:
                    searchSpace["maxDepth"] = new ParameterRange
                    {
                        MinValue = 3,
                        MaxValue = 20,
                        Type = ParameterType.Integer
                    };
                    searchSpace["minSamplesSplit"] = new ParameterRange
                    {
                        MinValue = 2,
                        MaxValue = 100,
                        Type = ParameterType.Integer
                    };
                    searchSpace["minSamplesLeaf"] = new ParameterRange
                    {
                        MinValue = 1,
                        MaxValue = 50,
                        Type = ParameterType.Integer
                    };
                    break;

                case ModelType.RandomForest:
                    searchSpace["numTrees"] = new ParameterRange
                    {
                        MinValue = 10,
                        MaxValue = 200,
                        Type = ParameterType.Integer
                    };
                    searchSpace["maxDepth"] = new ParameterRange
                    {
                        MinValue = 3,
                        MaxValue = 20,
                        Type = ParameterType.Integer
                    };
                    searchSpace["minSamplesSplit"] = new ParameterRange
                    {
                        MinValue = 2,
                        MaxValue = 100,
                        Type = ParameterType.Integer
                    };
                    searchSpace["maxFeatures"] = new ParameterRange
                    {
                        Type = ParameterType.Categorical,
                        CategoricalValues = new object[] { "sqrt", "log2", 0.5, 0.75, 1.0 }
                    };
                    break;

                case ModelType.GradientBoosting:
                    searchSpace["numTrees"] = new ParameterRange
                    {
                        MinValue = 50,
                        MaxValue = 300,
                        Type = ParameterType.Integer
                    };
                    searchSpace["learningRate"] = new ParameterRange
                    {
                        MinValue = 0.01,
                        MaxValue = 0.3,
                        Type = ParameterType.Continuous,
                        LogScale = true
                    };
                    searchSpace["maxDepth"] = new ParameterRange
                    {
                        MinValue = 3,
                        MaxValue = 10,
                        Type = ParameterType.Integer
                    };
                    searchSpace["subsample"] = new ParameterRange
                    {
                        MinValue = 0.5,
                        MaxValue = 1.0,
                        Type = ParameterType.Continuous
                    };
                    break;

                case ModelType.SupportVectorRegression:
                    searchSpace["C"] = new ParameterRange
                    {
                        MinValue = 0.01,
                        MaxValue = 100.0,
                        Type = ParameterType.Continuous,
                        LogScale = true
                    };
                    searchSpace["kernel"] = new ParameterRange
                    {
                        Type = ParameterType.Categorical,
                        CategoricalValues = new object[] { "linear", "rbf", "poly", "sigmoid" }
                    };
                    searchSpace["gamma"] = new ParameterRange
                    {
                        MinValue = 0.0001,
                        MaxValue = 1.0,
                        Type = ParameterType.Continuous,
                        LogScale = true
                    };
                    break;

                // Add more model types as needed
            }

            return searchSpace;
        }
        private void SetDefaultCandidateModels()
        {
            _candidateModels.AddRange(new[]
            {
                ModelType.LinearRegression,
                ModelType.DecisionTree,
                ModelType.RandomForest,
                ModelType.GradientBoosting
            });
        }

        private double CalculateAccuracy(double[] predictions, double[] targets)
        {
            if (predictions.Length != targets.Length)
                throw new ArgumentException("Predictions and targets must have same length");

            int correct = 0;
            for (int i = 0; i < predictions.Length; i++)
            {
                if (Math.Abs(predictions[i] - targets[i]) < 0.5)
                    correct++;
            }

            return (double)correct / predictions.Length;
        }

        private double CalculateMSE(double[] predictions, double[] targets)
        {
            if (predictions.Length != targets.Length)
                throw new ArgumentException("Predictions and targets must have same length");

            double sum = 0;
            for (int i = 0; i < predictions.Length; i++)
            {
                double diff = predictions[i] - targets[i];
                sum += diff * diff;
            }

            return sum / predictions.Length;
        }

        private double CalculateMAE(double[] predictions, double[] targets)
        {
            if (predictions.Length != targets.Length)
                throw new ArgumentException("Predictions and targets must have same length");

            double sum = 0;
            for (int i = 0; i < predictions.Length; i++)
            {
                sum += Math.Abs(predictions[i] - targets[i]);
            }

            return sum / predictions.Length;
        }

        private double CalculateRSquared(double[] predictions, double[] targets)
        {
            if (predictions.Length != targets.Length)
                throw new ArgumentException("Predictions and targets must have same length");

            double mean = targets.Average();
            double ssTotal = 0;
            double ssResidual = 0;

            for (int i = 0; i < targets.Length; i++)
            {
                ssTotal += Math.Pow(targets[i] - mean, 2);
                ssResidual += Math.Pow(targets[i] - predictions[i], 2);
            }

            return 1 - (ssResidual / ssTotal);
        }
    }
}