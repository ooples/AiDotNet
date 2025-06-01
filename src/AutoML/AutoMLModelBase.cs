using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Inputs;
using AiDotNet.Evaluation;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

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
        protected int? _earlyStoppingPatience;
        protected double _earlyStoppingMinDelta = 0.001;
        protected int _trialsSinceImprovement = 0;
        protected IModelEvaluator<T, TInput, TOutput>? _modelEvaluator;

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
                return _trialHistory.Select(t => t.Clone()).ToList();
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
            return await Task.Run(() =>
            {
                var importance = new Dictionary<int, double>();
                // This would be overridden by specific implementations
                return importance;
            });
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
            await Task.Run(() =>
            {
                lock (_lock)
                {
                    var trial = new TrialResult
                    {
                        TrialId = _trialHistory.Count + 1,
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
            });
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
        public virtual ModelMetaData<T> GetModelMetaData()
        {
            return new ModelMetaData<T>
            {
                Name = "AutoML",
                Description = $"AutoML with {_candidateModels.Count} candidate models",
                Version = "1.0",
                TrainingDate = DateTime.UtcNow,
                Properties = new Dictionary<string, object>
                {
                    ["Type"] = Type.ToString(),
                    ["Status"] = Status.ToString(),
                    ["BestScore"] = BestScore,
                    ["TrialsCompleted"] = _trialHistory.Count,
                    ["OptimizationMetric"] = _optimizationMetric.ToString(),
                    ["Maximize"] = _maximize,
                    ["CandidateModels"] = _candidateModels.Select(m => m.ToString()).ToList(),
                    ["SearchSpaceSize"] = _searchSpace.Count,
                    ["Constraints"] = _constraints.Count
                }
            };
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
            return await Task.Run(() =>
            {
                // Use the model evaluator if available
                if (_modelEvaluator != null)
                {
                    var evaluationInput = new ModelEvaluationInput<T, TInput, TOutput>
                    {
                        Model = model,
                        InputData = new OptimizationInputData<T, TInput, TOutput>
                        {
                            XValidation = validationInputs,
                            YValidation = validationTargets
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
            });
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
            throw new NotImplementedException("Use SearchAsync method instead for AutoML");
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
            throw new NotImplementedException("AutoML models should be recreated with SearchAsync");
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
            throw new NotImplementedException("AutoML models should be recreated with SearchAsync");
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
        public virtual IParameterizable<T, TInput, TOutput> WithParameters(Vector<T> parameters)
        {
            if (BestModel == null)
                throw new InvalidOperationException("No best model found.");
            
            throw new NotImplementedException("AutoML models should be recreated with SearchAsync");
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
        public virtual double[] GetFeatureImportance()
        {
            if (BestModel == null)
                throw new InvalidOperationException("No best model found.");
            
            return BestModel.GetFeatureImportance();
        }

        /// <summary>
        /// Gets the indices of active features
        /// </summary>
        public virtual int[] GetActiveFeatureIndices()
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
        /// Creates a deep copy of the AutoML model
        /// </summary>
        public virtual IFullModel<T, TInput, TOutput> Clone()
        {
            throw new NotImplementedException("AutoML models should be recreated with SearchAsync");
        }

        /// <summary>
        /// Creates a deep copy of the AutoML model
        /// </summary>
        public virtual IFullModel<T, TInput, TOutput> DeepCopy()
        {
            throw new NotImplementedException("AutoML models should be recreated with SearchAsync");
        }

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
            
            return _optimizationMetric switch
            {
                MetricType.Accuracy => validationStats.ErrorStats?.Accuracy ?? 0.0,
                MetricType.MeanSquaredError => validationStats.ErrorStats?.MeanSquaredError ?? double.MaxValue,
                MetricType.RootMeanSquaredError => validationStats.ErrorStats?.RootMeanSquaredError ?? double.MaxValue,
                MetricType.MeanAbsoluteError => validationStats.ErrorStats?.MeanAbsoluteError ?? double.MaxValue,
                MetricType.RSquared => validationStats.PredictionStats?.RSquared ?? 0.0,
                MetricType.F1Score => validationStats.ErrorStats?.F1Score ?? 0.0,
                MetricType.Precision => validationStats.ErrorStats?.Precision ?? 0.0,
                MetricType.Recall => validationStats.ErrorStats?.Recall ?? 0.0,
                MetricType.AUC => validationStats.ErrorStats?.AUC ?? 0.0,
                _ => 0.0
            };
        }
    }
}