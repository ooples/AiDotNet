using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.Evaluation;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Inputs;

namespace AiDotNet.AutoML
{
    /// <summary>
    /// Base class for AutoML models that automatically search for optimal model configurations
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations</typeparam>
    /// <typeparam name="TInput">The input data type</typeparam>
    /// <typeparam name="TOutput">The output data type</typeparam>
    public abstract class AutoMLModelBase<T, TInput, TOutput> : IAutoMLModel<T, TInput, TOutput>
    {
        protected readonly List<TrialResult> _trialHistory = new();
        protected readonly Dictionary<string, ParameterRange> _searchSpace = new();
        protected readonly List<ModelType> _candidateModels = new();
        protected readonly List<SearchConstraint> _constraints = new();
        protected readonly object _lock = new();

        protected MetricType _optimizationMetric = MetricType.Accuracy;
        protected bool _maximize = true;
        protected bool _optimizationMetricExplicitlySet;
        protected int? _earlyStoppingPatience;
        protected double _earlyStoppingMinDelta = 0.001;
        protected int _trialsSinceImprovement = 0;
        protected IModelEvaluator<T, TInput, TOutput>? _modelEvaluator;

        /// <summary>
        /// Gets the optimization metric used to rank trials.
        /// </summary>
        /// <remarks>
        /// <para>
        /// This is safe to expose and does not reveal proprietary hyperparameter values or model internals.
        /// </para>
        /// <para><b>For Beginners:</b> This is the score AutoML tries to make better (like Accuracy or RMSE).</para>
        /// </remarks>
        public MetricType OptimizationMetric => _optimizationMetric;

        /// <summary>
        /// Gets a value indicating whether higher metric values are better.
        /// </summary>
        /// <remarks>
        /// <para><b>For Beginners:</b> For some metrics higher is better (Accuracy). For error metrics lower is better (RMSE).</para>
        /// </remarks>
        public bool MaximizeOptimizationMetric => _maximize;

        /// <summary>
        /// Gets the model type
        /// </summary>
        public virtual ModelType Type => ModelType.AutoML;

        /// <summary>
        /// Gets the current optimization status
        /// </summary>
        public AutoMLStatus Status { get; protected set; } = AutoMLStatus.NotStarted;

        /// <summary>
        /// Gets the best model found so far
        /// </summary>
        public IFullModel<T, TInput, TOutput>? BestModel { get; protected set; }

        /// <summary>
        /// Gets the best score achieved
        /// </summary>
        public double BestScore { get; protected set; } = double.NegativeInfinity;

        /// <summary>
        /// Gets or sets the time limit for the AutoML search
        /// </summary>
        public TimeSpan TimeLimit { get; set; } = TimeSpan.FromMinutes(30);

        /// <summary>
        /// Gets or sets the maximum number of trials to run
        /// </summary>
        public int TrialLimit { get; set; } = 100;

        /// <summary>
        /// Searches for the best model configuration
        /// </summary>
        public abstract Task<IFullModel<T, TInput, TOutput>> SearchAsync(
            TInput inputs,
            TOutput targets,
            TInput validationInputs,
            TOutput validationTargets,
            TimeSpan timeLimit,
            CancellationToken cancellationToken = default);

        /// <summary>
        /// Sets the search space for hyperparameters
        /// </summary>
        public virtual void SetSearchSpace(Dictionary<string, ParameterRange> searchSpace)
        {
            lock (_lock)
            {
                _searchSpace.Clear();
                foreach (var kvp in searchSpace)
                {
                    _searchSpace[kvp.Key] = kvp.Value;
                }
            }
        }

        /// <summary>
        /// Sets the models to consider in the search
        /// </summary>
        public virtual void SetCandidateModels(List<ModelType> modelTypes)
        {
            lock (_lock)
            {
                _candidateModels.Clear();
                _candidateModels.AddRange(modelTypes);
            }
        }

        /// <summary>
        /// Sets the optimization metric
        /// </summary>
        public virtual void SetOptimizationMetric(MetricType metric, bool maximize = true)
        {
            _optimizationMetric = metric;
            _maximize = maximize;
            _optimizationMetricExplicitlySet = true;

            // Reset best score when metric changes
            BestScore = maximize ? double.NegativeInfinity : double.PositiveInfinity;
        }

        /// <summary>
        /// Gets the history of all trials
        /// </summary>
        public virtual List<TrialResult> GetTrialHistory()
        {
            lock (_lock)
            {
                return _trialHistory.Select(t => t.CloneRedacted()).ToList();
            }
        }

        /// <summary>
        /// Gets feature importance from the best model
        /// </summary>
        public virtual async Task<Dictionary<int, double>> GetFeatureImportanceAsync()
        {
            if (BestModel == null)
                throw new InvalidOperationException("No best model available. Run search first.");

            // Default implementation returns uniform importance
            return await Task.Run((Func<Dictionary<int, double>>)(() =>
            {
                var importance = new Dictionary<int, double>();
                // This would be overridden by specific implementations
                return importance;
            }));
        }

        /// <summary>
        /// Suggests the next hyperparameters to try
        /// </summary>
        public abstract Task<Dictionary<string, object>> SuggestNextTrialAsync();

        /// <summary>
        /// Reports the result of a trial
        /// </summary>
        public virtual async Task ReportTrialResultAsync(Dictionary<string, object> parameters, double score, TimeSpan duration)
        {
            await Task.Run((Action)(() =>
            {
                lock (_lock)
                {
                    var trial = new TrialResult
                    {
                        TrialId = _trialHistory.Count + 1,
                        CandidateModelType = TryExtractCandidateModelType(parameters),
                        Parameters = new Dictionary<string, object>(parameters),
                        Score = score,
                        Duration = duration,
                        Timestamp = DateTime.UtcNow
                    };

                    _trialHistory.Add(trial);

                    // Update best score and model
                    bool isBetter = _maximize ? score > BestScore : score < BestScore;

                    if (isBetter)
                    {
                        BestScore = score;
                        _trialsSinceImprovement = 0;
                    }
                    else
                    {
                        _trialsSinceImprovement++;
                    }
                }
            }));
        }

        /// <summary>
        /// Enables early stopping
        /// </summary>
        public virtual void EnableEarlyStopping(int patience, double minDelta = 0.001)
        {
            _earlyStoppingPatience = patience;
            _earlyStoppingMinDelta = minDelta;
            _trialsSinceImprovement = 0;
        }

        /// <summary>
        /// Sets constraints for the search
        /// </summary>
        public virtual void SetConstraints(List<SearchConstraint> constraints)
        {
            lock (_lock)
            {
                _constraints.Clear();
                _constraints.AddRange(constraints);
            }
        }

        /// <summary>
        /// Trains the model (legacy method - use SearchAsync instead)
        /// </summary>
        public virtual void Train(double[][] inputs, double[] outputs)
        {
            // AutoML models are trained through SearchAsync
            throw new NotSupportedException("Use SearchAsync to train AutoML models");
        }

        /// <summary>
        /// Makes predictions using the best model (legacy method)
        /// </summary>
        public virtual double[] Predict(double[][] inputs)
        {
            // This is a legacy method - use the generic Predict method instead
            throw new NotSupportedException("Use the generic Predict method instead");
        }

        /// <summary>
        /// Gets model metadata
        /// </summary>
        public virtual ModelMetadata<T> GetModelMetadata()
        {
            var metadata = new ModelMetadata<T>
            {
                Name = "AutoML",
                Description = $"AutoML with {_candidateModels.Count} candidate models",
                Version = "1.0",
                TrainingDate = DateTimeOffset.UtcNow
            };

            metadata.SetProperty("Type", Type.ToString());
            metadata.SetProperty("Status", Status.ToString());
            metadata.SetProperty("BestScore", BestScore);
            metadata.SetProperty("TrialsCompleted", _trialHistory.Count);
            metadata.SetProperty("OptimizationMetric", _optimizationMetric.ToString());
            metadata.SetProperty("Maximize", _maximize);
            metadata.SetProperty("CandidateModels", _candidateModels.Select(m => m.ToString()).ToList());
            metadata.SetProperty("SearchSpaceSize", _searchSpace.Count);
            metadata.SetProperty("Constraints", _constraints.Count);

            return metadata;
        }

        /// <summary>
        /// Checks if early stopping criteria is met
        /// </summary>
        protected bool ShouldStop()
        {
            if (!_earlyStoppingPatience.HasValue)
                return false;

            return _trialsSinceImprovement >= _earlyStoppingPatience.Value;
        }

        /// <summary>
        /// Validates constraints for a given configuration
        /// </summary>
        protected bool ValidateConstraints(Dictionary<string, object> parameters, IFullModel<T, TInput, TOutput>? model = null)
        {
            // This would be implemented by specific AutoML implementations
            // based on the constraint types and model properties
            return true;
        }

        /// <summary>
        /// Creates a model instance for the given type and parameters
        /// </summary>
        protected abstract Task<IFullModel<T, TInput, TOutput>> CreateModelAsync(ModelType modelType, Dictionary<string, object> parameters);

        /// <summary>
        /// Evaluates a model on the validation set
        /// </summary>
        protected virtual async Task<double> EvaluateModelAsync(
            IFullModel<T, TInput, TOutput> model,
            TInput validationInputs,
            TOutput validationTargets)
        {
            return await Task.Run((Func<double>)(() =>
            {
                // Use the model evaluator if available
                if (_modelEvaluator != null)
                {
                    var evaluationInput = new ModelEvaluationInput<T, TInput, TOutput>
                    {
                        Model = model,
                        InputData = new OptimizationInputData<T, TInput, TOutput>
                        {
                            // Provide all three sets to evaluators to avoid failures on empty placeholders.
                            // In AutoML we often only have a validation set for scoring, so we mirror it.
                            XTrain = validationInputs,
                            YTrain = validationTargets,
                            XValidation = validationInputs,
                            YValidation = validationTargets,
                            XTest = validationInputs,
                            YTest = validationTargets
                        }
                    };

                    var evaluationResult = _modelEvaluator.EvaluateModel(evaluationInput);

                    // Extract the appropriate metric based on optimization metric
                    return ExtractMetricFromEvaluation(evaluationResult);
                }
                else
                {
                    // Fallback to simple prediction-based evaluation
                    var predictions = model.Predict(validationInputs);
                    // For now, return a placeholder score
                    // In a real implementation, this would calculate the metric based on the data types
                    return 0.0;
                }
            }));
        }

        /// <summary>
        /// Gets the default search space for a model type
        /// </summary>
        protected abstract Dictionary<string, ParameterRange> GetDefaultSearchSpace(ModelType modelType);

        #region IModel Implementation

        /// <summary>
        /// Trains the AutoML model by searching for the best configuration
        /// </summary>
        public virtual void Train(TInput input, TOutput expectedOutput)
        {
            // AutoML doesn't use traditional training - it searches for the best model
            // This would typically be called internally during the search process
            throw new InvalidOperationException("AutoML models are trained using the SearchAsync method, not the traditional Train method. Please call SearchAsync to initiate the AutoML process.");
        }

        /// <summary>
        /// Makes predictions using the best model found
        /// </summary>
        public virtual TOutput Predict(TInput input)
        {
            if (BestModel == null)
                throw new InvalidOperationException("No best model found. Run SearchAsync first.");

            return BestModel.Predict(input);
        }


        #endregion

        #region IModelSerializer Implementation

        /// <summary>
        /// Saves the model to a file
        /// </summary>
        public virtual void SaveModel(string filePath)
        {
            if (BestModel == null)
                throw new InvalidOperationException("No best model to save.");

            BestModel.SaveModel(filePath);
        }

        /// <summary>
        /// Loads the model from a file
        /// </summary>
        public virtual void LoadModel(string filePath)
        {
            if (BestModel == null)
            {
                // This scenario requires a mechanism to determine the concrete type of BestModel
                // from the serialized data. For now, we'll assume BestModel is already set or can be inferred.
                throw new InvalidOperationException("Cannot load model: BestModel is null. AutoML models should be recreated with SearchAsync or BestModel should be initialized before loading.");
            }
            BestModel.LoadModel(filePath);
        }

        /// <summary>
        /// Serializes the model to bytes
        /// </summary>
        public virtual byte[] Serialize()
        {
            if (BestModel == null)
                throw new InvalidOperationException("No best model to serialize.");

            return BestModel.Serialize();
        }

        /// <summary>
        /// Deserializes the model from bytes
        /// </summary>
        public virtual void Deserialize(byte[] data)
        {
            if (BestModel == null)
            {
                // This scenario requires a mechanism to determine the concrete type of BestModel
                // from the serialized data. For now, we'll assume BestModel is already set or can be inferred.
                throw new InvalidOperationException("Cannot deserialize model: BestModel is null. AutoML models should be recreated with SearchAsync or BestModel should be initialized before deserializing.");
            }
            BestModel.Deserialize(data);
        }

        #endregion

        #region IParameterizable Implementation

        /// <summary>
        /// Gets the model parameters
        /// </summary>
        public virtual Vector<T> GetParameters()
        {
            if (BestModel == null)
                throw new InvalidOperationException("No best model found.");

            return BestModel.GetParameters();
        }

        /// <summary>
        /// Sets the model parameters
        /// </summary>
        public virtual void SetParameters(Vector<T> parameters)
        {
            if (BestModel == null)
                throw new InvalidOperationException("No best model found.");

            BestModel.SetParameters(parameters);
        }

        /// <summary>
        /// Gets the number of parameters
        /// </summary>
        public virtual int ParameterCount => BestModel?.ParameterCount ?? 0;

        /// <summary>
        /// Creates a new instance with the given parameters
        /// </summary>
        public virtual IFullModel<T, TInput, TOutput> WithParameters(Vector<T> parameters)
        {
            if (BestModel == null)
                throw new InvalidOperationException("No best model found. Run SearchAsync, Search, or SearchBestModel first.");

            // Create a deep copy and set the new parameters
            var copy = DeepCopy();
            copy.SetParameters(parameters);
            return copy;
        }

        #endregion

        #region IFeatureAware Implementation

        /// <summary>
        /// Gets the feature names
        /// </summary>
        public virtual string[] FeatureNames { get; set; } = Array.Empty<string>();

        /// <summary>
        /// Gets the feature importance scores
        /// </summary>
        public virtual Dictionary<string, T> GetFeatureImportance()
        {
            if (BestModel == null)
                throw new InvalidOperationException("No best model found.");

            return BestModel.GetFeatureImportance();
        }

        /// <summary>
        /// Gets the indices of active features
        /// </summary>
        public virtual IEnumerable<int> GetActiveFeatureIndices()
        {
            if (BestModel == null)
                throw new InvalidOperationException("No best model found.");

            return BestModel.GetActiveFeatureIndices();
        }

        /// <summary>
        /// Checks if a feature is used
        /// </summary>
        public virtual bool IsFeatureUsed(int featureIndex)
        {
            if (BestModel == null)
                throw new InvalidOperationException("No best model found.");

            return BestModel.IsFeatureUsed(featureIndex);
        }

        /// <summary>
        /// Sets the active feature indices
        /// </summary>
        public virtual void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
        {
            if (BestModel == null)
                throw new InvalidOperationException("No best model found.");

            BestModel.SetActiveFeatureIndices(featureIndices);
        }

        #endregion

        #region ICloneable Implementation

        /// <summary>
        /// Creates a memberwise clone of the AutoML model using MemberwiseClone().
        /// This performs a shallow copy where reference types are shared between the original and clone.
        /// </summary>
        /// <returns>A memberwise clone of the current AutoML model</returns>
        /// <remarks>
        /// For a deep copy with independent collections and state, use DeepCopy() instead.
        /// </remarks>
        public virtual IFullModel<T, TInput, TOutput> Clone()
        {
            return (AutoMLModelBase<T, TInput, TOutput>)MemberwiseClone();
        }

        /// <summary>
        /// Creates a deep copy of the AutoML model
        /// </summary>
        public virtual IFullModel<T, TInput, TOutput> DeepCopy()
        {
            // Create a new instance using the factory method to avoid sharing readonly collections
            var copy = CreateInstanceForCopy();

            // Deep copy collections under lock to ensure thread safety
            lock (_lock)
            {
                // Deep copy trial history
                foreach (var t in _trialHistory)
                {
                    copy._trialHistory.Add(t.Clone());
                }

                // Deep copy search space parameters
                // ParameterRange implements ICloneable, so we always call Clone()
                foreach (var kvp in _searchSpace)
                {
                    copy._searchSpace[kvp.Key] = (ParameterRange)kvp.Value.Clone();
                }

                // Copy candidate models (ModelType is an enum, so no deep copy needed)
                foreach (var model in _candidateModels)
                {
                    copy._candidateModels.Add(model);
                }

                // Deep copy constraints
                // SearchConstraint implements ICloneable, so we always call Clone()
                foreach (var constraint in _constraints)
                {
                    copy._constraints.Add((SearchConstraint)constraint.Clone());
                }
            }

            // Deep copy the best model if it exists
            copy.BestModel = BestModel?.DeepCopy();

            // Copy value types and other properties
            copy._optimizationMetric = _optimizationMetric;
            copy._maximize = _maximize;
            copy._earlyStoppingPatience = _earlyStoppingPatience;
            copy._earlyStoppingMinDelta = _earlyStoppingMinDelta;
            copy._trialsSinceImprovement = _trialsSinceImprovement;
            copy.BestScore = BestScore;
            copy.TimeLimit = TimeLimit;
            copy.TrialLimit = TrialLimit;
            copy.Status = Status;
            copy.FeatureNames = (string[])FeatureNames.Clone();
            copy._modelEvaluator = _modelEvaluator; // Shared reference is acceptable for the evaluator

            return copy;
        }

        /// <summary>
        /// Factory method for creating a new instance for deep copy.
        /// Derived classes must implement this to return a new instance of themselves.
        /// This ensures each copy has its own collections and lock object.
        /// </summary>
        /// <returns>A fresh instance of the derived class with default parameters</returns>
        /// <remarks>
        /// When implementing this method, derived classes should create a fresh instance with default parameters,
        /// and should not attempt to preserve runtime or initialization state from the original instance.
        /// The deep copy logic will transfer relevant state (trial history, search space, etc.) after construction.
        /// </remarks>
        protected abstract AutoMLModelBase<T, TInput, TOutput> CreateInstanceForCopy();


        #endregion

        /// <summary>
        /// Sets the model evaluator to use for evaluating candidate models
        /// </summary>
        public virtual void SetModelEvaluator(IModelEvaluator<T, TInput, TOutput> evaluator)
        {
            _modelEvaluator = evaluator;
        }

        /// <summary>
        /// Extracts the appropriate metric value from the evaluation results
        /// </summary>
        protected virtual double ExtractMetricFromEvaluation(ModelEvaluationData<T, TInput, TOutput> evaluationData)
        {
            var validationStats = evaluationData.ValidationSet;

            var numOps = MathHelper.GetNumericOperations<T>();

            if (evaluationData.ModelStats.HasMetric(_optimizationMetric))
            {
                return numOps.ToDouble(evaluationData.ModelStats.GetMetric(_optimizationMetric));
            }

            if (validationStats.ErrorStats.HasMetric(_optimizationMetric))
            {
                return numOps.ToDouble(validationStats.ErrorStats.GetMetric(_optimizationMetric));
            }

            if (validationStats.PredictionStats.HasMetric(_optimizationMetric))
            {
                return numOps.ToDouble(validationStats.PredictionStats.GetMetric(_optimizationMetric));
            }

            if (validationStats.ActualBasicStats.HasMetric(_optimizationMetric))
            {
                return numOps.ToDouble(validationStats.ActualBasicStats.GetMetric(_optimizationMetric));
            }

            if (validationStats.PredictedBasicStats.HasMetric(_optimizationMetric))
            {
                return numOps.ToDouble(validationStats.PredictedBasicStats.GetMetric(_optimizationMetric));
            }

            return _maximize ? double.NegativeInfinity : double.PositiveInfinity;
        }

        /// <summary>
        /// Reports a failed trial result without terminating the full AutoML run.
        /// </summary>
        /// <param name="parameters">The parameters used in the trial.</param>
        /// <param name="error">The exception that caused the trial to fail.</param>
        /// <param name="duration">The duration of the failed trial.</param>
        protected virtual async Task ReportTrialFailureAsync(Dictionary<string, object> parameters, Exception error, TimeSpan duration)
        {
            await Task.Run((Action)(() =>
            {
                lock (_lock)
                {
                    var trial = new TrialResult
                    {
                        TrialId = _trialHistory.Count + 1,
                        CandidateModelType = TryExtractCandidateModelType(parameters),
                        Parameters = new Dictionary<string, object>(parameters),
                        Score = _maximize ? double.NegativeInfinity : double.PositiveInfinity,
                        Duration = duration,
                        Timestamp = DateTime.UtcNow,
                        Success = false,
                        ErrorMessage = error.Message
                    };

                    _trialHistory.Add(trial);
                    _trialsSinceImprovement++;
                }
            }));
        }

        #region IAutoMLModel Additional Interface Members

        private static ModelType? TryExtractCandidateModelType(IReadOnlyDictionary<string, object> parameters)
        {
            if (parameters is null || !parameters.TryGetValue("ModelType", out var modelTypeObj) || modelTypeObj is null)
            {
                return null;
            }

            if (modelTypeObj is ModelType modelType)
            {
                return modelType;
            }

            if (modelTypeObj is string text && Enum.TryParse<ModelType>(text, ignoreCase: true, out var parsed))
            {
                return parsed;
            }

            return null;
        }

        /// <summary>
        /// Configures the search space for hyperparameter optimization
        /// </summary>
        /// <param name="searchSpace">Dictionary defining parameter ranges to search</param>
        public virtual void ConfigureSearchSpace(Dictionary<string, ParameterRange> searchSpace)
        {
            SetSearchSpace(searchSpace);
        }

        /// <summary>
        /// Sets the time limit for the AutoML search process
        /// </summary>
        /// <param name="timeLimit">Maximum time to spend searching for optimal models</param>
        public virtual void SetTimeLimit(TimeSpan timeLimit)
        {
            TimeLimit = timeLimit;
        }

        /// <summary>
        /// Sets the maximum number of trials to execute during search
        /// </summary>
        /// <param name="maxTrials">Maximum number of model configurations to try</param>
        public virtual void SetTrialLimit(int maxTrials)
        {
            TrialLimit = maxTrials;
        }

        /// <summary>
        /// Enables Neural Architecture Search (NAS) for automatic network design
        /// </summary>
        /// <param name="enabled">Whether to enable NAS</param>
        public virtual void EnableNAS(bool enabled = true)
        {
            // Store NAS flag - derived classes can use this during model creation
            lock (_lock)
            {
                if (!_searchSpace.ContainsKey("EnableNAS"))
                {
                    _searchSpace["EnableNAS"] = new ParameterRange
                    {
                        Type = ParameterType.Boolean,
                        MinValue = enabled,
                        MaxValue = enabled
                    };
                }
            }
        }

        /// <summary>
        /// Searches for the best model configuration (synchronous version)
        /// </summary>
        /// <param name="inputs">Training inputs</param>
        /// <param name="targets">Training targets</param>
        /// <param name="validationInputs">Validation inputs</param>
        /// <param name="validationTargets">Validation targets</param>
        /// <returns>Best model found</returns>
        public virtual IFullModel<T, TInput, TOutput> SearchBestModel(
            TInput inputs,
            TOutput targets,
            TInput validationInputs,
            TOutput validationTargets)
        {
            // Synchronous wrapper around SearchAsync
            return SearchAsync(inputs, targets, validationInputs, validationTargets, TimeLimit, CancellationToken.None)
                .GetAwaiter()
                .GetResult();
        }

        /// <summary>
        /// Performs the AutoML search process (synchronous version)
        /// </summary>
        /// <param name="inputs">Training inputs</param>
        /// <param name="targets">Training targets</param>
        /// <param name="validationInputs">Validation inputs</param>
        /// <param name="validationTargets">Validation targets</param>
        public virtual void Search(
            TInput inputs,
            TOutput targets,
            TInput validationInputs,
            TOutput validationTargets)
        {
            // Synchronous search that updates BestModel
            SearchAsync(inputs, targets, validationInputs, validationTargets, TimeLimit, CancellationToken.None)
                .GetAwaiter()
                .GetResult();
        }

        /// <summary>
        /// Gets the results of all trials performed during search
        /// </summary>
        /// <returns>List of trial results with scores and parameters</returns>
        public virtual List<TrialResult> GetResults()
        {
            return GetTrialHistory();
        }

        /// <summary>
        /// Runs the AutoML optimization process (alternative name for Search)
        /// </summary>
        /// <param name="inputs">Training inputs</param>
        /// <param name="targets">Training targets</param>
        /// <param name="validationInputs">Validation inputs</param>
        /// <param name="validationTargets">Validation targets</param>
        public virtual void Run(
            TInput inputs,
            TOutput targets,
            TInput validationInputs,
            TOutput validationTargets)
        {
            Search(inputs, targets, validationInputs, validationTargets);
        }

        /// <summary>
        /// Sets which model types should be considered during the search
        /// </summary>
        /// <param name="modelTypes">List of model types to evaluate</param>
        public virtual void SetModelsToTry(List<ModelType> modelTypes)
        {
            SetCandidateModels(modelTypes);
        }

        /// <summary>
        /// Gets the default loss function for gradient computation.
        /// </summary>
        /// <remarks>
        /// AutoML delegates to the best model found during search. If no best model exists yet,
        /// returns Mean Squared Error as a sensible default.
        /// </remarks>
        public virtual ILossFunction<T> DefaultLossFunction =>
            BestModel is not null && BestModel != null
                ? BestModel.DefaultLossFunction
                : new MeanSquaredErrorLoss<T>();

        /// <summary>
        /// Computes gradients by delegating to the best model.
        /// </summary>
        public virtual Vector<T> ComputeGradients(TInput input, TOutput target, ILossFunction<T>? lossFunction = null)
        {
            if (BestModel is null || BestModel == null)
                throw new InvalidOperationException(
                    "Cannot compute gradients before AutoML search has found a best model. Call Search() first.");

            return BestModel.ComputeGradients(input, target, lossFunction);
        }

        /// <summary>
        /// Applies gradients by delegating to the best model.
        /// </summary>
        public virtual void ApplyGradients(Vector<T> gradients, T learningRate)
        {
            if (BestModel is null || BestModel == null)
                throw new InvalidOperationException(
                    "Cannot apply gradients before AutoML search has found a best model. Call Search() first.");

            BestModel.ApplyGradients(gradients, learningRate);
        }

        #endregion
        #region IJitCompilable Implementation

        /// <summary>
        /// Gets whether this model currently supports JIT compilation.
        /// </summary>
        /// <value>True if the best model found supports JIT compilation, false otherwise.</value>
        /// <remarks>
        /// <para>
        /// AutoML models delegate JIT compilation support to their best model.
        /// If no best model has been found yet, JIT compilation is not supported.
        /// </para>
        /// <para><b>For Beginners:</b> AutoML models can only be JIT compiled if the best model they found supports it.
        ///
        /// Since AutoML searches across multiple model types, JIT support depends on:
        /// - Whether a best model has been selected
        /// - Whether that specific model supports JIT compilation
        ///
        /// Before running SearchAsync, this will return false.
        /// After finding a best model, it delegates to that model's JIT support.
        /// </para>
        /// </remarks>
        public virtual bool SupportsJitCompilation
        {
            get
            {
                if (BestModel is null || BestModel == null)
                    return false;

                return BestModel.SupportsJitCompilation;
            }
        }

        /// <summary>
        /// Exports the computation graph for JIT compilation by delegating to the best model.
        /// </summary>
        /// <param name="inputNodes">List to populate with input computation nodes.</param>
        /// <returns>The output computation node representing the model's prediction.</returns>
        /// <remarks>
        /// <para>
        /// AutoML models delegate graph export to their best model found during search.
        /// The graph structure and complexity depends on the specific best model type.
        /// </para>
        /// <para><b>For Beginners:</b> This creates a computation graph from the best model found.
        ///
        /// AutoML itself doesn't have a fixed computation structure since it tries multiple model types.
        /// Instead, it delegates to the best model it found:
        /// - If the best model is a neural network, you get a neural network graph
        /// - If it's a regression model, you get a regression graph
        /// - And so on
        ///
        /// This only works after SearchAsync has found and selected a best model.
        /// </para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">
        /// Thrown when no best model exists (SearchAsync not called yet).
        /// </exception>
        /// <exception cref="NotSupportedException">
        /// Thrown when the best model does not support JIT compilation.
        /// </exception>
        public virtual ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
        {
            if (inputNodes == null)
                throw new ArgumentNullException(nameof(inputNodes));

            if (BestModel is null || BestModel == null)
                throw new InvalidOperationException(
                    "Cannot export computation graph: No best model has been found yet. " +
                    "Call SearchAsync to find the best model first.");

            if (!BestModel.SupportsJitCompilation)
                throw new NotSupportedException(
                    $"The best model of type {BestModel.GetType().Name} does not support JIT compilation. " +
                    "JIT compilation availability depends on the specific model type found during AutoML search.");

            return BestModel.ExportComputationGraph(inputNodes);
        }

        #endregion

        /// <summary>
        /// Saves the AutoML model's current state to a stream.
        /// </summary>
        /// <param name="stream">The stream to write the model state to.</param>
        /// <remarks>
        /// <para>
        /// This method serializes the best model found during the AutoML search.
        /// It uses the existing Serialize method and writes the data to the provided stream.
        /// </para>
        /// <para><b>For Beginners:</b> This is like creating a snapshot of your best AutoML model.
        ///
        /// When you call SaveState:
        /// - The best model found during search is written to the stream
        /// - All model parameters and configuration are preserved
        ///
        /// This is particularly useful for:
        /// - Saving the best model after AutoML search
        /// - Checkpointing during long-running searches
        /// - Knowledge distillation from AutoML-optimized models
        /// - Deploying optimized models to production
        ///
        /// You can later use LoadState to restore the model.
        /// </para>
        /// </remarks>
        /// <exception cref="ArgumentNullException">Thrown when stream is null.</exception>
        /// <exception cref="InvalidOperationException">Thrown when no best model exists.</exception>
        /// <exception cref="IOException">Thrown when there's an error writing to the stream.</exception>
        public virtual void SaveState(Stream stream)
        {
            if (stream == null)
                throw new ArgumentNullException(nameof(stream));

            if (!stream.CanWrite)
                throw new ArgumentException("Stream must be writable.", nameof(stream));

            try
            {
                var data = this.Serialize();
                stream.Write(data, 0, data.Length);
                stream.Flush();
            }
            catch (IOException ex)
            {
                throw new IOException($"Failed to save AutoML model state to stream: {ex.Message}", ex);
            }
            catch (InvalidOperationException)
            {
                // Re-throw InvalidOperationException from Serialize (no best model)
                throw;
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Unexpected error while saving AutoML model state: {ex.Message}", ex);
            }
        }

        /// <summary>
        /// Loads the AutoML model's state from a stream.
        /// </summary>
        /// <param name="stream">The stream to read the model state from.</param>
        /// <remarks>
        /// <para>
        /// This method deserializes a best model that was previously saved with SaveState.
        /// It uses the existing Deserialize method after reading data from the stream.
        /// </para>
        /// <para><b>For Beginners:</b> This is like loading a saved snapshot of your best AutoML model.
        ///
        /// When you call LoadState:
        /// - The best model is read from the stream
        /// - All parameters and configuration are restored
        ///
        /// After loading, the model can:
        /// - Make predictions using the restored best model
        /// - Be further optimized if needed
        /// - Be deployed to production
        ///
        /// This is essential for:
        /// - Loading the best model after AutoML search
        /// - Deploying optimized models to production
        /// - Knowledge distillation workflows
        /// </para>
        /// </remarks>
        /// <exception cref="ArgumentNullException">Thrown when stream is null.</exception>
        /// <exception cref="IOException">Thrown when there's an error reading from the stream.</exception>
        /// <exception cref="InvalidOperationException">Thrown when the stream contains invalid or incompatible data, or when BestModel is not initialized.</exception>
        public virtual void LoadState(Stream stream)
        {
            if (stream == null)
                throw new ArgumentNullException(nameof(stream));

            if (!stream.CanRead)
                throw new ArgumentException("Stream must be readable.", nameof(stream));

            try
            {
                using var ms = new MemoryStream();
                stream.CopyTo(ms);
                var data = ms.ToArray();

                if (data.Length == 0)
                    throw new InvalidOperationException("Stream contains no data.");

                this.Deserialize(data);
            }
            catch (IOException ex)
            {
                throw new IOException($"Failed to read AutoML model state from stream: {ex.Message}", ex);
            }
            catch (InvalidOperationException)
            {
                // Re-throw InvalidOperationException from Deserialize
                throw;
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException(
                    $"Failed to deserialize AutoML model state. The stream may contain corrupted or incompatible data: {ex.Message}", ex);
            }
        }
    }
}
