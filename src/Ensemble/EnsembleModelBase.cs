using System.Collections.Concurrent;
using AiDotNet.Models.Options;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Interpretability;
using AiDotNet.LinearAlgebra;
using AiDotNet.Logging;
using AiDotNet.Models;

namespace AiDotNet.Ensemble;

/// <summary>
/// Base class for all ensemble models providing common functionality.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix<double>, Tensor<double>, Vector<double>).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector<double>, Tensor<double>).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This base class provides the foundation for all ensemble models. 
/// It handles common tasks like managing the collection of models, coordinating predictions, 
/// and managing weights. Specific ensemble types (voting, stacking, etc.) build on this base.
/// </para>
/// </remarks>
public abstract class EnsembleModelBase<T, TInput, TOutput> : InterpretableModelBase<T, TInput, TOutput>, IEnsembleModel<T, TInput, TOutput>
{
    protected readonly List<IFullModel<T, TInput, TOutput>> _baseModels;
    protected Vector<T> _modelWeights;
    protected readonly EnsembleOptions<T> _options;
    protected readonly INumericOperations<T> NumOps;
    protected readonly ILogging _logger;
    protected ICombinationStrategy<T, TInput, TOutput> _combinationStrategy;
    
    // Additional fields for advanced ensemble methods
    protected Dictionary<string, object> _modelMetadata;
    protected Vector<T>? _modelPerformanceScores;
    protected Matrix<T>? _modelPredictionHistory;
    protected Dictionary<int, string> _modelIdentifiers;
    protected ConcurrentDictionary<string, double> _performanceCache;
    
    /// <summary>
    /// Gets the base models in the ensemble.
    /// </summary>
    public IReadOnlyList<IFullModel<T, TInput, TOutput>> BaseModels => _baseModels.AsReadOnly();
    
    /// <summary>
    /// Gets the combination strategy.
    /// </summary>
    public ICombinationStrategy<T, TInput, TOutput> CombinationStrategy => _combinationStrategy;
    
    /// <summary>
    /// Gets the model weights.
    /// </summary>
    public Vector<T> ModelWeights => _modelWeights;
    
    /// <summary>
    /// Initializes a new instance of the EnsembleModelBase class.
    /// </summary>
    /// <param name="options">Configuration options for the ensemble.</param>
    protected EnsembleModelBase(EnsembleOptions<T> options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _baseModels = new List<IFullModel<T, TInput, TOutput>>();
        _modelWeights = new Vector<T>(0);
        NumOps = MathHelper.GetNumericOperations<T>();
        _logger = LoggingFactory.GetLogger(GetType());
        _combinationStrategy = CreateCombinationStrategy();
        _modelMetadata = new Dictionary<string, object>();
        _modelIdentifiers = new Dictionary<int, string>();
        _performanceCache = new ConcurrentDictionary<string, double>();
        
        _logger.Information("Initializing {EnsembleType} with strategy {Strategy}", 
            GetType().Name, _options.Strategy);
    }
    
    #region IEnsembleModel Implementation
    
    /// <summary>
    /// Adds a model to the ensemble with validation.
    /// </summary>
    public virtual void AddModel(IFullModel<T, TInput, TOutput> model, T weight)
    {
        if (model == null)
            throw new ArgumentNullException(nameof(model));
            
        if (_baseModels.Count >= _options.MaxModels)
        {
            _logger.Warning("Cannot add model: ensemble already contains {MaxModels} models", _options.MaxModels);
            throw new InvalidOperationException($"Cannot add more than {_options.MaxModels} models to ensemble");
        }
            
        if (!_options.AllowDuplicateModelTypes && 
            _baseModels.Any(m => m.GetType() == model.GetType()))
        {
            _logger.Warning("Cannot add model: {ModelType} already exists in ensemble", model.GetType().Name);
            throw new InvalidOperationException($"Model type {model.GetType().Name} already exists in ensemble");
        }
        
        // Validate model if required
        if (_options.ValidateModelsBeforeAdding)
        {
            _logger.Debug("Validating model before adding to ensemble");
            // Validation will be implemented by derived classes
            ValidateModel(model);
        }
        
        _baseModels.Add(model);
        _modelIdentifiers[_baseModels.Count - 1] = $"{model.GetType().Name}_{Guid.NewGuid():N}";
        
        // Resize weights vector
        var newWeights = new T[_modelWeights.Length + 1];
        for (int i = 0; i < _modelWeights.Length; i++)
        {
            newWeights[i] = _modelWeights[i];
        }
        newWeights[_modelWeights.Length] = weight;
        _modelWeights = new Vector<T>(newWeights);
        
        _logger.Information("Added {ModelType} to ensemble with weight {Weight}. Total models: {Count}", 
            model.GetType().Name, weight?.ToString() ?? "null", _baseModels.Count);
    }
    
    /// <summary>
    /// Removes a model from the ensemble.
    /// </summary>
    public virtual bool RemoveModel(IFullModel<T, TInput, TOutput> model)
    {
        var index = _baseModels.IndexOf(model);
        if (index < 0)
        {
            _logger.Debug("Model not found in ensemble");
            return false;
        }
        
        if (_baseModels.Count <= _options.MinModels)
        {
            _logger.Warning("Cannot remove model: ensemble must contain at least {MinModels} models", _options.MinModels);
            throw new InvalidOperationException($"Ensemble must contain at least {_options.MinModels} models");
        }
        
        _baseModels.RemoveAt(index);
        
        // Remove associated metadata
        if (_modelIdentifiers.ContainsKey(index))
        {
            var modelId = _modelIdentifiers[index];
            _modelIdentifiers.Remove(index);
            
            // Rebuild identifiers dictionary with updated indices
            var newIdentifiers = new Dictionary<int, string>();
            foreach (var kvp in _modelIdentifiers.Where(k => k.Key > index))
            {
                newIdentifiers[kvp.Key - 1] = kvp.Value;
            }
            foreach (var kvp in _modelIdentifiers.Where(k => k.Key < index))
            {
                newIdentifiers[kvp.Key] = kvp.Value;
            }
            _modelIdentifiers = newIdentifiers;
        }
        
        // Update weights
        var newWeights = new T[_modelWeights.Length - 1];
        for (int i = 0, j = 0; i < _modelWeights.Length; i++)
        {
            if (i != index)
            {
                newWeights[j++] = _modelWeights[i];
            }
        }
        _modelWeights = new Vector<T>(newWeights);
        
        _logger.Information("Removed model at index {Index}. Remaining models: {Count}", 
            index, _baseModels.Count);
        
        return true;
    }
    
    /// <summary>
    /// Updates the weights of the base models.
    /// </summary>
    public virtual void UpdateWeights(Vector<T> newWeights)
    {
        if (newWeights.Length != _baseModels.Count)
        {
            throw new ArgumentException(
                $"Weight vector length ({newWeights.Length}) must match number of models ({_baseModels.Count})");
        }
        
        // Apply weight regularization if configured
        if (!NumOps.Equals(_options.WeightRegularization, NumOps.Zero))
        {
            newWeights = ApplyWeightRegularization(newWeights);
        }
        
        // Check for minimum weight threshold
        if (!NumOps.Equals(_options.MinimumModelWeight, NumOps.Zero))
        {
            for (int i = 0; i < newWeights.Length; i++)
            {
                if (NumOps.LessThan(newWeights[i], _options.MinimumModelWeight))
                {
                    _logger.Warning("Model {Index} weight {Weight} is below minimum threshold {Threshold}", 
                        i, newWeights[i]?.ToString() ?? "null", _options.MinimumModelWeight?.ToString() ?? "null");
                }
            }
        }
        
        _modelWeights = newWeights;
        _logger.Debug("Updated model weights");
    }
    
    /// <summary>
    /// Gets individual predictions from each base model.
    /// </summary>
    public virtual List<TOutput> GetIndividualPredictions(TInput input)
    {
        if (_baseModels.Count == 0)
        {
            throw new InvalidOperationException("No models in ensemble to make predictions");
        }
        
        var predictions = new List<TOutput>();
        
        if (_options.PredictInParallel && _baseModels.Count > 1)
        {
            _logger.Debug("Making predictions in parallel with {Count} models", _baseModels.Count);
            
            var parallelOptions = new ParallelOptions
            {
                MaxDegreeOfParallelism = _options.MaxParallelism
            };
            
            var concurrentPredictions = new ConcurrentBag<(int Index, TOutput Prediction)>();
            
            Parallel.ForEach(_baseModels.Select((model, index) => new { model, index }), 
                parallelOptions, 
                item =>
                {
                    try
                    {
                        var prediction = item.model.Predict(input);
                        concurrentPredictions.Add((item.index, prediction));
                    }
                    catch (Exception ex)
                    {
                        _logger.Error(ex, "Error in model {Index} prediction", item.index);
                        throw;
                    }
                });
            
            // Sort predictions by index to maintain order
            predictions = concurrentPredictions
                .OrderBy(p => p.Index)
                .Select(p => p.Prediction)
                .ToList();
        }
        else
        {
            _logger.Debug("Making predictions sequentially with {Count} models", _baseModels.Count);
            
            for (int i = 0; i < _baseModels.Count; i++)
            {
                try
                {
                    predictions.Add(_baseModels[i].Predict(input));
                }
                catch (Exception ex)
                {
                    _logger.Error(ex, "Error in model {Index} prediction", i);
                    throw;
                }
            }
        }
        
        return predictions;
    }
    
    #endregion
    
    #region IFullModel Implementation
    
    /// <summary>
    /// Trains all base models in the ensemble.
    /// </summary>
    public override void Train(TInput input, TOutput output)
    {
        if (_baseModels.Count == 0)
        {
            throw new InvalidOperationException("No models in ensemble to train");
        }
        
        if (_baseModels.Count < _options.MinModels)
        {
            throw new InvalidOperationException(
                $"Ensemble requires at least {_options.MinModels} models, but only {_baseModels.Count} are present");
        }
        
        _logger.Information("Training ensemble with {Count} models using {Strategy} strategy", 
            _baseModels.Count, _options.TrainingStrategy);
        
        var startTime = DateTime.UtcNow;
        
        // Train based on strategy
        switch (_options.TrainingStrategy)
        {
            case EnsembleTrainingStrategy.Parallel:
                TrainParallel(input, output);
                break;
            case EnsembleTrainingStrategy.Sequential:
                TrainSequential(input, output);
                break;
            case EnsembleTrainingStrategy.Bagging:
                TrainWithBagging(input, output);
                break;
            case EnsembleTrainingStrategy.Boosting:
                TrainWithBoosting(input, output);
                break;
            default:
                throw new NotSupportedException($"Training strategy {_options.TrainingStrategy} is not supported");
        }
        
        var trainingTime = DateTime.UtcNow - startTime;
        _logger.Information("Ensemble training completed in {Time:F2} seconds", trainingTime.TotalSeconds);
        
        // Update weights if needed
        if (_options.UpdateWeightsAfterTraining)
        {
            _logger.Debug("Updating model weights after training");
            UpdateModelWeights(input, output);
        }
    }
    
    /// <summary>
    /// Makes predictions using the ensemble.
    /// </summary>
    public override TOutput Predict(TInput input)
    {
        if (_baseModels.Count == 0)
        {
            throw new InvalidOperationException("No models in ensemble to make predictions");
        }
        
        _logger.Debug("Making ensemble prediction with {Count} models", _baseModels.Count);
        
        var predictions = GetIndividualPredictions(input);
        
        if (!_combinationStrategy.CanCombine(predictions))
        {
            throw new InvalidOperationException("Predictions cannot be combined with current strategy");
        }
        
        var result = _combinationStrategy.Combine(predictions, _modelWeights);
        
        _logger.Debug("Ensemble prediction completed");
        
        return result;
    }
    
    /// <summary>
    /// Gets the model parameters as a vector.
    /// </summary>
    public virtual Vector<T> GetParameters()
    {
        var allParameters = new List<T>();
        
        // Add weights as parameters
        allParameters.AddRange(_modelWeights.ToArray());
        
        // Add parameters from each base model
        foreach (var model in _baseModels)
        {
            var modelParams = model.GetParameters();
            allParameters.AddRange(modelParams.ToArray());
        }
        
        return new Vector<T>(allParameters.ToArray());
    }
    
    /// <summary>
    /// Sets the model parameters from a vector.
    /// </summary>
    public virtual void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length < _modelWeights.Length)
        {
            throw new ArgumentException("Parameter vector is too short");
        }
        
        // Extract weights
        var weights = new T[_modelWeights.Length];
        for (int i = 0; i < weights.Length; i++)
        {
            weights[i] = parameters[i];
        }
        UpdateWeights(new Vector<T>(weights));
        
        // Set parameters for each base model
        int offset = _modelWeights.Length;
        foreach (var model in _baseModels)
        {
            var modelParamCount = model.GetParameters().Length;
            if (offset + modelParamCount > parameters.Length)
            {
                throw new ArgumentException("Parameter vector is too short for all models");
            }
            
            var modelParams = new T[modelParamCount];
            for (int i = 0; i < modelParamCount; i++)
            {
                modelParams[i] = parameters[offset + i];
            }
            
            model.SetParameters(new Vector<T>(modelParams));
            offset += modelParamCount;
        }
    }
    
    /// <summary>
    /// Saves the ensemble model to a file.
    /// </summary>
    public override void Save(string path)
    {
        _logger.Information("Saving ensemble model to {Path}", path);
        
        using (var writer = new BinaryWriter(File.Open(path, FileMode.Create)))
        {
            // Write ensemble metadata
            writer.Write("ENSEMBLE_V1"); // Version marker
            writer.Write(_baseModels.Count);
            writer.Write((int)_options.Strategy);
            
            // Write weights
            writer.Write(_modelWeights.Length);
            foreach (var weight in _modelWeights)
            {
                SerializationHelper<T>.WriteValue(writer, weight);
            }
            
            // Write each model
            foreach (var model in _baseModels)
            {
                writer.Write(model.GetType().AssemblyQualifiedName ?? model.GetType().FullName!);
                
                // Serialize model
                var modelBytes = model.Serialize();
                writer.Write(modelBytes.Length);
                writer.Write(modelBytes);
            }
            
            // Write options
            WriteOptions(writer);
        }
        
        _logger.Information("Ensemble model saved successfully");
    }
    
    /// <summary>
    /// Loads the ensemble model from a file.
    /// </summary>
    public override void Load(string path)
    {
        _logger.Information("Loading ensemble model from {Path}", path);
        
        using (var reader = new BinaryReader(File.Open(path, FileMode.Open)))
        {
            // Read and verify version
            var version = reader.ReadString();
            if (version != "ENSEMBLE_V1")
            {
                throw new InvalidOperationException($"Unsupported ensemble version: {version}");
            }
            
            // Read ensemble metadata
            var modelCount = reader.ReadInt32();
            var strategy = (EnsembleStrategy)reader.ReadInt32();
            
            // Clear existing models
            _baseModels.Clear();
            _modelIdentifiers.Clear();
            
            // Read weights
            var weightCount = reader.ReadInt32();
            var weights = new T[weightCount];
            for (int i = 0; i < weightCount; i++)
            {
                weights[i] = SerializationHelper<T>.ReadValue(reader);
            }
            _modelWeights = new Vector<T>(weights);
            
            // Read each model
            for (int i = 0; i < modelCount; i++)
            {
                var typeName = reader.ReadString();
                var modelType = Type.GetType(typeName);
                if (modelType == null)
                {
                    throw new InvalidOperationException($"Cannot find type: {typeName}");
                }
                
                var model = Activator.CreateInstance(modelType) as IFullModel<T, TInput, TOutput>;
                if (model == null)
                {
                    throw new InvalidOperationException($"Cannot create instance of type: {typeName}");
                }
                
                // Read model data
                var modelBytesLength = reader.ReadInt32();
                var modelBytes = reader.ReadBytes(modelBytesLength);
                
                // Deserialize model
                model.Deserialize(modelBytes);
                
                _baseModels.Add(model);
                _modelIdentifiers[i] = $"{model.GetType().Name}_{Guid.NewGuid():N}";
            }
            
            // Read options
            ReadOptions(reader);
            
            // Recreate combination strategy
            _combinationStrategy = CreateCombinationStrategy();
        }
        
        _logger.Information("Ensemble model loaded successfully with {Count} models", _baseModels.Count);
    }
    
    /// <summary>
    /// Creates a deep copy of the ensemble model.
    /// </summary>
    public virtual IFullModel<T, TInput, TOutput> Clone()
    {
        var clone = (EnsembleModelBase<T, TInput, TOutput>)Activator.CreateInstance(GetType(), _options)!;
        
        // Clone each base model
        foreach (var model in _baseModels)
        {
            var clonedModel = model.Clone();
            clone._baseModels.Add(clonedModel);
        }
        
        // Copy weights
        clone._modelWeights = new Vector<T>(_modelWeights.ToArray());
        
        // Copy metadata
        clone._modelMetadata = new Dictionary<string, object>(_modelMetadata);
        clone._modelIdentifiers = new Dictionary<int, string>(_modelIdentifiers);
        
        if (_modelPerformanceScores != null)
        {
            clone._modelPerformanceScores = new Vector<T>(_modelPerformanceScores.ToArray());
        }
        
        return clone;
    }
    
    /// <summary>
    /// Gets metadata about the ensemble model.
    /// </summary>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            ModelType = ModelType.CustomEnsemble,
            Description = $"{GetType().Name} with {_baseModels.Count} models using {_options.Strategy} strategy",
            AdditionalInfo = new Dictionary<string, object>
            {
                ["EnsembleStrategy"] = _options.Strategy.ToString(),
                ["NumberOfModels"] = _baseModels.Count,
                ["ModelTypes"] = string.Join(", ", _baseModels.Select(m => m.GetType().Name)),
                ["TrainingStrategy"] = _options.TrainingStrategy.ToString(),
                ["WeightUpdateMethod"] = _options.WeightUpdateMethod.ToString()
            }
        };
        
        // Add model-specific metadata
        foreach (var kvp in _modelMetadata)
        {
            metadata.AdditionalInfo[kvp.Key] = kvp.Value;
        }
        
        return metadata;
    }
    
    #endregion
    
    #region Abstract Methods
    
    /// <summary>
    /// Creates the appropriate combination strategy based on options.
    /// </summary>
    protected abstract ICombinationStrategy<T, TInput, TOutput> CreateCombinationStrategy();
    
    /// <summary>
    /// Trains models in parallel.
    /// </summary>
    protected abstract void TrainParallel(TInput input, TOutput output);
    
    /// <summary>
    /// Trains models sequentially.
    /// </summary>
    protected abstract void TrainSequential(TInput input, TOutput output);
    
    /// <summary>
    /// Trains models using bagging (bootstrap aggregating).
    /// </summary>
    protected abstract void TrainWithBagging(TInput input, TOutput output);
    
    /// <summary>
    /// Trains models using boosting.
    /// </summary>
    protected abstract void TrainWithBoosting(TInput input, TOutput output);
    
    /// <summary>
    /// Updates model weights based on performance.
    /// </summary>
    protected abstract void UpdateModelWeights(TInput input, TOutput output);
    
    /// <summary>
    /// Validates a model before adding to ensemble.
    /// </summary>
    protected abstract void ValidateModel(IFullModel<T, TInput, TOutput> model);
    
    /// <summary>
    /// Writes additional options to the stream.
    /// </summary>
    protected abstract void WriteOptions(BinaryWriter writer);
    
    /// <summary>
    /// Reads additional options from the stream.
    /// </summary>
    protected abstract void ReadOptions(BinaryReader reader);
    
    #endregion
    
    #region Protected Helper Methods
    
    /// <summary>
    /// Applies weight regularization to prevent any single model from dominating.
    /// </summary>
    protected virtual Vector<T> ApplyWeightRegularization(Vector<T> weights)
    {
        var regularized = new T[weights.Length];
        var sumWeights = NumOps.Zero;
        
        // Calculate sum of weights
        for (int i = 0; i < weights.Length; i++)
        {
            sumWeights = NumOps.Add(sumWeights, weights[i]);
        }
        
        // Apply L2 regularization
        var regFactor = NumOps.Subtract(NumOps.One, _options.WeightRegularization);
        
        for (int i = 0; i < weights.Length; i++)
        {
            var normalized = NumOps.Divide(weights[i], sumWeights);
            var regularizedWeight = NumOps.Multiply(normalized, regFactor);
            var uniform = NumOps.Divide(NumOps.One, NumOps.FromDouble(weights.Length));
            var uniformContribution = NumOps.Multiply(uniform, _options.WeightRegularization);
            regularized[i] = NumOps.Add(regularizedWeight, uniformContribution);
        }
        
        return new Vector<T>(regularized);
    }
    
    /// <summary>
    /// Creates a bootstrap sample of the training data.
    /// </summary>
    /// <remarks>
    /// Must be implemented by derived classes to handle specific data types.
    /// </remarks>
    protected abstract (TInput Input, TOutput Output) CreateBootstrapSample(
        TInput input, TOutput output, Random random);
    
    /// <summary>
    /// Splits data into training and validation sets.
    /// </summary>
    /// <remarks>
    /// Must be implemented by derived classes to handle specific data types.
    /// </remarks>
    protected abstract (TInput TrainInput, TOutput TrainOutput, 
                      TInput ValInput, TOutput ValOutput) 
        SplitData(TInput input, TOutput output, double splitRatio);
    
    #endregion
    
    #region IModelSerializer Implementation
    
    /// <summary>
    /// Serializes the model to a byte array.
    /// </summary>
    public virtual byte[] Serialize()
    {
        // This is a placeholder implementation
        // In practice, you'd serialize all ensemble data
        return new byte[0];
    }
    
    /// <summary>
    /// Deserializes the model from a byte array.
    /// </summary>
    public virtual void Deserialize(byte[] data)
    {
        // This is a placeholder implementation
        // In practice, you'd deserialize all ensemble data
        // For now, do nothing
    }
    
    #endregion
    
    #region IParameterizable Implementation
    
    /// <summary>
    /// Creates a new model with the specified parameters.
    /// </summary>
    public virtual IFullModel<T, TInput, TOutput> WithParameters(Vector<T> parameters)
    {
        var clone = Clone();
        clone.SetParameters(parameters);
        return clone;
    }
    
    #endregion
    
    #region IFeatureAware Implementation
    
    /// <summary>
    /// Gets the indices of active features.
    /// </summary>
    public virtual IEnumerable<int> GetActiveFeatureIndices()
    {
        // Default implementation returns all indices
        // Derived classes can override for specific behavior
        return Enumerable.Range(0, int.MaxValue);
    }
    
    /// <summary>
    /// Checks if a specific feature is used.
    /// </summary>
    public virtual bool IsFeatureUsed(int featureIndex)
    {
        // Default implementation assumes all features are used
        return true;
    }
    
    /// <summary>
    /// Sets the active feature indices.
    /// </summary>
    public virtual void SetActiveFeatureIndices(IEnumerable<int> activeIndices)
    {
        // Default implementation does nothing
        // Derived classes can override for specific behavior
    }
    
    #endregion
    
    #region ICloneable Implementation
    
    /// <summary>
    /// Creates a deep copy of the model.
    /// </summary>
    public virtual IFullModel<T, TInput, TOutput> DeepCopy()
    {
        return Clone();
    }
    
    #endregion

    #region IFullModel Interface Members - Added by Team 23

    /// <summary>
    /// Gets the total number of parameters in the model
    /// </summary>
    public virtual int ParameterCount => GetParameters().Length;

    /// <summary>
    /// Saves the model to a file
    /// </summary>
    public virtual void SaveModel(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
            throw new ArgumentException("File path cannot be null or empty.", nameof(filePath));

        Save(filePath);
    }

    /// <summary>
    /// Gets feature importance scores
    /// </summary>
    public virtual Dictionary<string, T> GetFeatureImportance()
    {
        // Aggregate feature importance from all base models
        var aggregatedImportance = new Dictionary<string, T>();

        for (int i = 0; i < _baseModels.Count; i++)
        {
            var modelImportance = _baseModels[i].GetFeatureImportance();
            var weight = _modelWeights[i];

            foreach (var kvp in modelImportance)
            {
                var weightedImportance = NumOps.Multiply(kvp.Value, weight);

                if (aggregatedImportance.ContainsKey(kvp.Key))
                {
                    aggregatedImportance[kvp.Key] = NumOps.Add(aggregatedImportance[kvp.Key], weightedImportance);
                }
                else
                {
                    aggregatedImportance[kvp.Key] = weightedImportance;
                }
            }
        }

        return aggregatedImportance;
    }

    #endregion
}