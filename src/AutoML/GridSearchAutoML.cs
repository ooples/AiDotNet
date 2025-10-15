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
    /// Implements AutoML using exhaustive grid search over the hyperparameter space
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations</typeparam>
    /// <typeparam name="TInput">The input data type</typeparam>
    /// <typeparam name="TOutput">The output data type</typeparam>
    public class GridSearchAutoML<T, TInput, TOutput> : AutoMLModelBase<T, TInput, TOutput>
    {
        private readonly HyperparameterSpace _hyperparameterSpace = default!;
        private List<Dictionary<string, object>>? _gridPoints = default!;
        private int _currentGridIndex = 0;
        private readonly int _stepsPerDimension;

        /// <summary>
        /// Initializes a new instance of the GridSearchAutoML class
        /// </summary>
        /// <param name="stepsPerDimension">Number of steps to divide continuous parameters</param>
        /// <param name="seed">Random seed for reproducibility</param>
        public GridSearchAutoML(int stepsPerDimension = 10, int? seed = null)
        {
            _stepsPerDimension = stepsPerDimension;
            _hyperparameterSpace = new HyperparameterSpace(seed);
        }

        /// <summary>
        /// Searches for the best model configuration using grid search
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

                // Generate grid points
                _gridPoints = GenerateGridPoints();
                _currentGridIndex = 0;

                Console.WriteLine($"Grid search: {_gridPoints.Count} total configurations to evaluate");

                while (_currentGridIndex < _gridPoints.Count)
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

                    // Get next configuration
                    var parameters = await SuggestNextTrialAsync();
                    
                    // Try each candidate model type
                    foreach (var modelType in _candidateModels)
                    {
                        if (cancellationToken.IsCancellationRequested)
                            break;

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
                                Console.WriteLine($"Progress: {_trialHistory.Count} trials completed, Best score: {BestScore:F4}");
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
                        }
                    }
                }

                Status = BestModel != null ? AutoMLStatus.Completed : AutoMLStatus.Failed;

                if (BestModel == null)
                {
                    throw new InvalidOperationException("No valid model found during search");
                }

                return BestModel;
            }
            catch (Exception ex)
            {
                Status = AutoMLStatus.Failed;
                throw new InvalidOperationException($"Grid search failed: {ex.Message}", ex);
            }
        }

        /// <summary>
        /// Suggests the next hyperparameters to try
        /// </summary>
        public override async Task<Dictionary<string, object>> SuggestNextTrialAsync()
        {
            return await Task.Run((Func<Dictionary<string, object>>)(() =>
            {
                if (_gridPoints == null || _currentGridIndex >= _gridPoints.Count)
                {
                    throw new InvalidOperationException("No more grid points to evaluate");
                }

                var parameters = _gridPoints[_currentGridIndex];
                _currentGridIndex++;

                return new Dictionary<string, object>(parameters);
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
                ModelType.RandomForest
            });
        }

        private List<Dictionary<string, object>> GenerateGridPoints()
        {
            // For each model type, generate grid points
            var allGridPoints = new List<Dictionary<string, object>>();

            foreach (var modelType in _candidateModels)
            {
                // Get model-specific search space
                var modelSearchSpace = GetDefaultSearchSpace(modelType);
                
                // Merge with user-defined search space
                foreach (var kvp in _searchSpace)
                {
                    modelSearchSpace[kvp.Key] = kvp.Value;
                }

                // Create hyperparameter space for this model
                var space = new HyperparameterSpace();
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

                // Generate grid for this model
                var modelGrid = space.GenerateGrid(_stepsPerDimension);
                
                // Add model type to each configuration
                foreach (var config in modelGrid)
                {
                    config["__modelType__"] = modelType;
                }

                allGridPoints.AddRange(modelGrid);
            }

            return allGridPoints;
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