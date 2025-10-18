using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Inputs;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace AiDotNet.AutoML
{
    /// <summary>
    /// Implements AutoML using Bayesian optimization for intelligent hyperparameter search
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations</typeparam>
    /// <typeparam name="TInput">The input data type</typeparam>
    /// <typeparam name="TOutput">The output data type</typeparam>
    public class BayesianOptimizationAutoML<T, TInput, TOutput> : AutoMLModelBase<T, TInput, TOutput>
    {
        private readonly HyperparameterSpace _hyperparameterSpace;
        private readonly int _numInitialPoints;
        private readonly double _explorationWeight;
        private readonly Random _random;
        
        // Gaussian Process components for Bayesian optimization
        private readonly List<T[]> _observedPoints = new();
        private readonly List<T> _observedValues = new();
        private T[]? _kernelMatrix;
        private readonly T _noiseVariance;
        private readonly INumericOperations<T> _numOps;

        /// <summary>
        /// Initializes a new instance of the BayesianOptimizationAutoML class
        /// </summary>
        /// <param name="numInitialPoints">Number of random initial points before Bayesian optimization</param>
        /// <param name="explorationWeight">Weight for exploration vs exploitation (higher = more exploration)</param>
        /// <param name="seed">Random seed for reproducibility</param>
        public BayesianOptimizationAutoML(int numInitialPoints = 10, double explorationWeight = 2.0, int? seed = null)
        {
            _numInitialPoints = numInitialPoints;
            _explorationWeight = explorationWeight;
            _random = seed.HasValue ? new Random(seed.Value) : new Random();
            _hyperparameterSpace = new HyperparameterSpace(seed);
            _numOps = Helpers.MathHelper.GetNumericOperations<T>();
            _noiseVariance = _numOps.FromDouble(1e-6);
        }

        /// <summary>
        /// Searches for the best model configuration using Bayesian optimization
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

                Console.WriteLine($"Bayesian optimization: {_numInitialPoints} initial random points");

                // Initial random sampling phase
                while (_trialHistory.Count < _numInitialPoints)
                {
                    if (DateTime.UtcNow - startTime > timeLimit || cancellationToken.IsCancellationRequested)
                        break;

                    await PerformTrialAsync(inputs, targets, validationInputs, validationTargets, 
                        isRandom: true, cancellationToken);
                }

                Console.WriteLine("Starting Bayesian optimization phase");

                // Bayesian optimization phase
                while (true)
                {
                    // Check stopping conditions
                    if (DateTime.UtcNow - startTime > timeLimit)
                    {
                        Console.WriteLine("Time limit reached");
                        break;
                    }

                    if (cancellationToken.IsCancellationRequested)
                    {
                        Status = AutoMLStatus.Cancelled;
                        break;
                    }

                    if (ShouldStop())
                    {
                        Console.WriteLine($"Early stopping triggered after {_trialsSinceImprovement} trials without improvement");
                        break;
                    }

                    // Perform Bayesian-optimized trial
                    await PerformTrialAsync(inputs, targets, validationInputs, validationTargets, 
                        isRandom: false, cancellationToken);
                }

                Status = BestModel != null ? AutoMLStatus.Completed : AutoMLStatus.Failed;

                if (BestModel == null)
                {
                    throw new InvalidOperationException("No valid model found during search");
                }

                Console.WriteLine($"Bayesian optimization completed. Total trials: {_trialHistory.Count}, Best score: {BestScore:F4}");
                return BestModel;
            }
            catch (Exception ex)
            {
                Status = AutoMLStatus.Failed;
                throw new InvalidOperationException($"Bayesian optimization failed: {ex.Message}", ex);
            }
        }

        /// <summary>
        /// Suggests the next hyperparameters to try using acquisition function
        /// </summary>
        public override async Task<Dictionary<string, object>> SuggestNextTrialAsync()
        {
            return await Task.Run((Func<Dictionary<string, object>>)(() =>
            {
                // If we don't have enough observations, sample randomly
                if (_observedPoints.Count < _numInitialPoints)
                {
                    return SampleRandomConfiguration();
                }

                // Update kernel matrix with current observations
                UpdateKernelMatrix();

                // Find the configuration that maximizes the acquisition function
                Dictionary<string, object>? bestConfig = null;
                double bestAcquisition = double.NegativeInfinity;

                // Multi-start optimization of acquisition function
                int numStarts = 25;
                for (int i = 0; i < numStarts; i++)
                {
                    var startPoint = SampleRandomConfiguration();
                    var optimizedConfig = OptimizeAcquisitionFunction(startPoint);
                    double acquisition = CalculateAcquisitionValue(optimizedConfig);

                    if (acquisition > bestAcquisition)
                    {
                        bestAcquisition = acquisition;
                        bestConfig = optimizedConfig;
                    }
                }

                return bestConfig ?? SampleRandomConfiguration();
            }));
        }

        /// <summary>
        /// Reports the result of a trial and updates the Gaussian Process
        /// </summary>
        public override async Task ReportTrialResultAsync(Dictionary<string, object> parameters, double score, TimeSpan duration)
        {
            await base.ReportTrialResultAsync(parameters, score, duration);

            // Add observation to Gaussian Process
            var point = ParametersToVector(parameters);
            lock (_lock)
            {
                _observedPoints.Add(point);
                var scoreT = _numOps.FromDouble(score);
                _observedValues.Add(_maximize ? scoreT : _numOps.Negate(scoreT)); // Normalize to maximization
            }
        }

        private async Task PerformTrialAsync(
            TInput inputs,
            TOutput targets,
            TInput validationInputs,
            TOutput validationTargets,
            bool isRandom,
            CancellationToken cancellationToken)
        {
            var trialStartTime = DateTime.UtcNow;

            try
            {
                // Get configuration
                Dictionary<string, object> parameters;
                if (isRandom)
                {
                    parameters = SampleRandomConfiguration();
                }
                else
                {
                    parameters = await SuggestNextTrialAsync();
                }

                // Extract model type
                var modelType = (ModelType)parameters["__modelType__"];
                parameters.Remove("__modelType__");

                // Create and train model
                var model = await CreateModelAsync(modelType, parameters);
                model.Train(inputs, targets);

                // Evaluate model
                var score = await EvaluateModelAsync(model, validationInputs, validationTargets);
                var duration = DateTime.UtcNow - trialStartTime;

                // Report results
                parameters["__modelType__"] = modelType; // Add back for reporting
                await ReportTrialResultAsync(parameters, score, duration);

                // Update best model
                bool isBetter = _maximize ? score > BestScore : score < BestScore;
                if (isBetter)
                {
                    BestModel = model;
                    Console.WriteLine($"New best score: {score:F4} (Trial {_trialHistory.Count})");
                }

                // Log progress
                if (_trialHistory.Count % 5 == 0)
                {
                    Console.WriteLine($"Progress: {_trialHistory.Count} trials completed, Best score: {BestScore:F4}");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Trial failed: {ex.Message}");
            }
        }

        private Dictionary<string, object> SampleRandomConfiguration()
        {
            var modelType = _candidateModels[_random.Next(_candidateModels.Count)];
            var searchSpace = GetDefaultSearchSpace(modelType);
            
            // Merge with user-defined search space
            foreach (var kvp in _searchSpace)
            {
                searchSpace[kvp.Key] = kvp.Value;
            }

            var space = new HyperparameterSpace(_random.Next());
            ConfigureHyperparameterSpace(space, searchSpace);

            var parameters = space.Sample();
            parameters["__modelType__"] = modelType;
            
            return parameters;
        }

        private void UpdateKernelMatrix()
        {
            int n = _observedPoints.Count;
            _kernelMatrix = new T[n * n];

            // Compute RBF kernel matrix
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    var distance = CalculateDistance(_observedPoints[i], _observedPoints[j]);
                    var distSquared = _numOps.Multiply(distance, distance);
                    var expArg = _numOps.Multiply(_numOps.FromDouble(-0.5), distSquared);
                    _kernelMatrix[i * n + j] = _numOps.Exp(expArg);
                    
                    if (i == j)
                    {
                        _kernelMatrix[i * n + j] = _numOps.Add(_kernelMatrix[i * n + j], _noiseVariance);
                    }
                }
            }
        }

        private Dictionary<string, object> OptimizeAcquisitionFunction(Dictionary<string, object> startPoint)
        {
            // Simple gradient-free optimization of acquisition function
            var currentConfig = new Dictionary<string, object>(startPoint);
            double currentAcquisition = CalculateAcquisitionValue(currentConfig);
            
            int numSteps = 20;
            double stepSize = 0.1;

            for (int step = 0; step < numSteps; step++)
            {
                var improved = false;

                // Try perturbing each parameter
                foreach (var key in currentConfig.Keys.ToList())
                {
                    if (key == "__modelType__") continue;

                    var originalValue = currentConfig[key];
                    var perturbedConfigs = GeneratePerturbations(currentConfig, key, stepSize);

                    foreach (var perturbedConfig in perturbedConfigs)
                    {
                        double acquisition = CalculateAcquisitionValue(perturbedConfig);
                        if (acquisition > currentAcquisition)
                        {
                            currentConfig = new Dictionary<string, object>(perturbedConfig);
                            currentAcquisition = acquisition;
                            improved = true;
                            break;
                        }
                    }

                    if (improved) break;
                }

                if (!improved)
                {
                    stepSize *= 0.5; // Reduce step size
                }
            }

            return currentConfig;
        }

        private List<Dictionary<string, object>> GeneratePerturbations(
            Dictionary<string, object> config, 
            string parameterName, 
            double stepSize)
        {
            var perturbations = new List<Dictionary<string, object>>();
            var value = config[parameterName];

            // Get parameter range
            var modelType = (ModelType)config["__modelType__"];
            var searchSpace = GetDefaultSearchSpace(modelType);
            
            if (!searchSpace.TryGetValue(parameterName, out var range))
                return perturbations;

            switch (range.Type)
            {
                case ParameterType.Continuous:
                    double current = Convert.ToDouble(value);
                    double min = Convert.ToDouble(range.MinValue);
                    double max = Convert.ToDouble(range.MaxValue);
                    double delta = (max - min) * stepSize;

                    // Try increasing and decreasing
                    var increased = new Dictionary<string, object>(config);
                    increased[parameterName] = Math.Min(max, current + delta);
                    perturbations.Add(increased);

                    var decreased = new Dictionary<string, object>(config);
                    decreased[parameterName] = Math.Max(min, current - delta);
                    perturbations.Add(decreased);
                    break;

                case ParameterType.Integer:
                    int intCurrent = Convert.ToInt32(value);
                    int intMin = Convert.ToInt32(range.MinValue);
                    int intMax = Convert.ToInt32(range.MaxValue);

                    if (intCurrent < intMax)
                    {
                        var inc = new Dictionary<string, object>(config);
                        inc[parameterName] = intCurrent + 1;
                        perturbations.Add(inc);
                    }

                    if (intCurrent > intMin)
                    {
                        var dec = new Dictionary<string, object>(config);
                        dec[parameterName] = intCurrent - 1;
                        perturbations.Add(dec);
                    }
                    break;

                case ParameterType.Boolean:
                    var flipped = new Dictionary<string, object>(config);
                    flipped[parameterName] = !(bool)value;
                    perturbations.Add(flipped);
                    break;

                case ParameterType.Categorical:
                    if (range.CategoricalValues != null)
                    {
                        foreach (var catValue in range.CategoricalValues)
                        {
                            if (!catValue.Equals(value))
                            {
                                var changed = new Dictionary<string, object>(config);
                                changed[parameterName] = catValue;
                                perturbations.Add(changed);
                            }
                        }
                    }
                    break;
            }

            return perturbations;
        }

        private double CalculateAcquisitionValue(Dictionary<string, object> config)
        {
            if (_observedPoints.Count == 0)
                return 0.0;

            var point = ParametersToVector(config);
            var (mean, variance) = PredictGaussianProcess(point);
            
            // Upper Confidence Bound (UCB) acquisition function
            var meanDouble = Convert.ToDouble((object)mean!);
            var varianceDouble = Convert.ToDouble((object)variance!);
            double ucb = meanDouble + _explorationWeight * Math.Sqrt(varianceDouble);
            return ucb;
        }

        private (T mean, T variance) PredictGaussianProcess(T[] point)
        {
            int n = _observedPoints.Count;
            if (n == 0 || _kernelMatrix == null)
                return (_numOps.Zero, _numOps.One);

            // Compute kernel vector between test point and observed points
            var k = new T[n];
            for (int i = 0; i < n; i++)
            {
                var distance = CalculateDistance(point, _observedPoints[i]);
                var distSquared = _numOps.Multiply(distance, distance);
                var expArg = _numOps.Multiply(_numOps.FromDouble(-0.5), distSquared);
                k[i] = _numOps.Exp(expArg);
            }

            // Solve for weights: K^-1 * y
            var weights = SolveLinearSystem(_kernelMatrix, _observedValues.ToArray(), n);

            // Compute mean prediction
            var mean = _numOps.Zero;
            for (int i = 0; i < n; i++)
            {
                mean = _numOps.Add(mean, _numOps.Multiply(weights[i], k[i]));
            }

            // Compute variance
            var kInvK = SolveLinearSystem(_kernelMatrix, k, n);
            var variance = _numOps.One; // Prior variance
            for (int i = 0; i < n; i++)
            {
                variance = _numOps.Subtract(variance, _numOps.Multiply(k[i], kInvK[i]));
            }
            // Ensure positive variance
            var minVariance = _numOps.FromDouble(1e-6);
            if (_numOps.LessThan(variance, minVariance))
                variance = minVariance;

            return (mean, variance);
        }

        private T[] SolveLinearSystem(T[] matrix, T[] vector, int n)
        {
            // Simple Gaussian elimination (for demonstration)
            // In practice, use more stable methods like Cholesky decomposition
            var a = new T[n, n];
            var b = (T[])vector.Clone();

            // Copy matrix
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    a[i, j] = matrix[i * n + j];
                }
            }

            // Forward elimination
            for (int k = 0; k < n - 1; k++)
            {
                for (int i = k + 1; i < n; i++)
                {
                    var factor = _numOps.Divide(a[i, k], a[k, k]);
                    for (int j = k + 1; j < n; j++)
                    {
                        a[i, j] = _numOps.Subtract(a[i, j], _numOps.Multiply(factor, a[k, j]));
                    }
                    b[i] = _numOps.Subtract(b[i], _numOps.Multiply(factor, b[k]));
                }
            }

            // Back substitution
            var x = new T[n];
            for (int i = n - 1; i >= 0; i--)
            {
                var sum = b[i];
                for (int j = i + 1; j < n; j++)
                {
                    sum = _numOps.Subtract(sum, _numOps.Multiply(a[i, j], x[j]));
                }
                x[i] = _numOps.Divide(sum, a[i, i]);
            }

            return x;
        }

        private T[] ParametersToVector(Dictionary<string, object> parameters)
        {
            var vector = new List<T>();

            foreach (var kvp in parameters.OrderBy(kvp => kvp.Key))
            {
                var key = kvp.Key;
                var value = kvp.Value;
                
                if (key == "__modelType__")
                {
                    // Encode model type as one-hot
                    var modelType = (ModelType)value;
                    for (int i = 0; i < _candidateModels.Count; i++)
                    {
                        vector.Add(_candidateModels[i] == modelType ? _numOps.One : _numOps.Zero);
                    }
                }
                else if (value is bool b)
                {
                    vector.Add(b ? _numOps.One : _numOps.Zero);
                }
                else if (value is string s)
                {
                    // Simple hash encoding for categorical values
                    vector.Add(_numOps.FromDouble(s.GetHashCode() % 100 / 100.0));
                }
                else
                {
                    vector.Add(_numOps.FromDouble(Convert.ToDouble(value)));
                }
            }

            return vector.ToArray();
        }

        private T CalculateDistance(T[] a, T[] b)
        {
            if (a.Length != b.Length)
                throw new ArgumentException("Vectors must have same length");

            var sum = _numOps.Zero;
            for (int i = 0; i < a.Length; i++)
            {
                var diff = _numOps.Subtract(a[i], b[i]);
                sum = _numOps.Add(sum, _numOps.Multiply(diff, diff));
            }

            return _numOps.Sqrt(sum);
        }

        private void ConfigureHyperparameterSpace(HyperparameterSpace space, Dictionary<string, ParameterRange> searchSpace)
        {
            foreach (var kvp in searchSpace)
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
        }

        /// <summary>
        /// Creates a model instance for the given type and parameters
        /// </summary>
        protected override async Task<IFullModel<T, TInput, TOutput>> CreateModelAsync(ModelType modelType, Dictionary<string, object> parameters)
        {
            // This would use PredictionModelBuilder or a factory to create models
            // For now, throwing an exception to indicate this needs to be implemented
            await Task.CompletedTask;
            throw new NotImplementedException($"Model creation for {modelType} needs to be implemented using PredictionModelBuilder");
        }

        /// <summary>
        /// Evaluates a model on the validation set
        /// </summary>
        protected override async Task<double> EvaluateModelAsync(
            IFullModel<T, TInput, TOutput> model,
            TInput validationInputs,
            TOutput validationTargets)
        {
            // For now, use the base class implementation
            // The base class should handle metric calculation appropriately
            return await base.EvaluateModelAsync(model, validationInputs, validationTargets);
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

    }
}