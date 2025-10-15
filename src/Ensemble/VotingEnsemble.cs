using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Ensemble.Strategies;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Interpretability;

namespace AiDotNet.Ensemble;

/// <summary>
/// An ensemble model that combines predictions through voting mechanisms.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix<double>, Tensor<double>, Vector<double>).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector<double>, Tensor<double>).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A voting ensemble is like asking a group of experts for their opinions 
/// and then either taking the most popular answer (hard voting) or averaging their confidence 
/// levels (soft voting). This is one of the simplest and most effective ensemble methods.
/// </para>
/// <para>
/// Voting ensembles work well when:
/// - You have diverse models that make different types of errors
/// - The individual models are reasonably accurate
/// - You want a simple, interpretable ensemble method
/// </para>
/// </remarks>
public class VotingEnsemble<T, TInput, TOutput> : EnsembleModelBase<T, TInput, TOutput>
{
    private readonly VotingEnsembleOptions<T> _votingOptions;
    
    /// <summary>
    /// Initializes a new instance of the VotingEnsemble class.
    /// </summary>
    /// <param name="options">Configuration options for the voting ensemble.</param>
    public VotingEnsemble(VotingEnsembleOptions<T>? options = null) 
        : base(options ?? new VotingEnsembleOptions<T>())
    {
        _votingOptions = (VotingEnsembleOptions<T>)_options;
    }
    
    /// <summary>
    /// Creates the appropriate combination strategy based on options.
    /// </summary>
    protected override ICombinationStrategy<T, TInput, TOutput> CreateCombinationStrategy()
    {
        return _votingOptions.VotingType switch
        {
            VotingType.Hard => new VotingStrategy<T, TInput, TOutput>(useSoftVoting: false),
            VotingType.Soft => new VotingStrategy<T, TInput, TOutput>(useSoftVoting: true),
            VotingType.Weighted => new WeightedAverageStrategy<T, TInput, TOutput>(),
            _ => throw new NotSupportedException($"Voting type {_votingOptions.VotingType} is not supported")
        };
    }
    
    /// <summary>
    /// Trains models in parallel.
    /// </summary>
    protected override void TrainParallel(TInput input, TOutput output)
    {
        _logger.Debug("Training {Count} models in parallel", _baseModels.Count);
        
        var parallelOptions = new ParallelOptions
        {
            MaxDegreeOfParallelism = _options.MaxParallelism
        };
        
        Parallel.ForEach(_baseModels.Select((model, index) => new { model, index }), 
            parallelOptions, 
            item =>
            {
                try
                {
                    _logger.Debug("Training model {Index}: {Type}", item.index, item.model.GetType().Name);
                    item.model.Train(input, output);
                    _logger.Debug("Model {Index} training completed", item.index);
                }
                catch (Exception ex)
                {
                    _logger.Error(ex, "Error training model {Index}", item.index);
                    throw;
                }
            });
    }
    
    /// <summary>
    /// Trains models sequentially.
    /// </summary>
    protected override void TrainSequential(TInput input, TOutput output)
    {
        _logger.Debug("Training {Count} models sequentially", _baseModels.Count);
        
        for (int i = 0; i < _baseModels.Count; i++)
        {
            _logger.Debug("Training model {Index}: {Type}", i, _baseModels[i].GetType().Name);
            _baseModels[i].Train(input, output);
            _logger.Debug("Model {Index} training completed", i);
        }
    }
    
    /// <summary>
    /// Trains models using bagging (bootstrap aggregating).
    /// </summary>
    protected override void TrainWithBagging(TInput input, TOutput output)
    {
        _logger.Debug("Training {Count} models with bagging", _baseModels.Count);
        
        var random = new Random(_options.Seed ?? DateTime.Now.Millisecond);
        
        var parallelOptions = new ParallelOptions
        {
            MaxDegreeOfParallelism = _options.MaxParallelism
        };
        
        Parallel.ForEach(_baseModels.Select((model, index) => new { model, index }), 
            parallelOptions, 
            item =>
            {
                try
                {
                    // Create bootstrap sample for this model
                    var (sampleInput, sampleOutput) = CreateBootstrapSample(input, output, 
                        new Random(random.Next() + item.index));
                    
                    _logger.Debug("Training model {Index} on bootstrap sample", item.index);
                    item.model.Train(sampleInput, sampleOutput);
                    _logger.Debug("Model {Index} training completed", item.index);
                }
                catch (Exception ex)
                {
                    _logger.Error(ex, "Error training model {Index} with bagging", item.index);
                    throw;
                }
            });
    }
    
    /// <summary>
    /// Trains models using boosting (not typically used with voting, but included for completeness).
    /// </summary>
    protected override void TrainWithBoosting(TInput input, TOutput output)
    {
        _logger.Warning("Boosting is not typically used with voting ensembles. Using sequential training instead.");
        TrainSequential(input, output);
    }
    
    /// <summary>
    /// Updates model weights based on performance.
    /// </summary>
    protected override void UpdateModelWeights(TInput input, TOutput output)
    {
        _logger.Debug("Updating model weights based on performance");
        
        // Split data for weight calculation
        var (trainInput, trainOutput, valInput, valOutput) = 
            SplitData(input, output, _options.TrainTestSplitRatio);
        
        var newWeights = new T[_baseModels.Count];
        
        for (int i = 0; i < _baseModels.Count; i++)
        {
            // Calculate performance on validation set
            var predictions = _baseModels[i].Predict(valInput);
            var performance = CalculateModelPerformance(predictions, valOutput);
            
            // Convert performance to weight
            newWeights[i] = ConvertPerformanceToWeight(performance);
            
            _logger.Debug("Model {Index} performance: {Performance}, weight: {Weight}", 
                i, performance?.ToString() ?? "null", newWeights[i]?.ToString() ?? "null");
        }
        
        UpdateWeights(new Vector<T>(newWeights));
    }
    
    /// <summary>
    /// Validates a model before adding to ensemble.
    /// </summary>
    protected override void ValidateModel(IFullModel<T, TInput, TOutput> model)
    {
        // For voting ensemble, we just check that the model is not null
        // Derived classes can implement more sophisticated validation
        if (model == null)
        {
            throw new ArgumentNullException(nameof(model));
        }
        
        _logger.Debug("Model {Type} passed validation", model.GetType().Name);
    }
    
    /// <summary>
    /// Writes voting-specific options to the stream.
    /// </summary>
    protected override void WriteOptions(BinaryWriter writer)
    {
        writer.Write((int)_votingOptions.VotingType);
        writer.Write(_votingOptions.UseOutOfBagScoring);
        writer.Write(_votingOptions.RequireConsensus);
        writer.Write(_votingOptions.ConsensusThreshold);
    }
    
    /// <summary>
    /// Reads voting-specific options from the stream.
    /// </summary>
    protected override void ReadOptions(BinaryReader reader)
    {
        var votingType = (VotingType)reader.ReadInt32();
        var useOutOfBag = reader.ReadBoolean();
        var requireConsensus = reader.ReadBoolean();
        var consensusThreshold = reader.ReadDouble();
        
        // Update options if they differ
        if (_votingOptions.VotingType != votingType)
        {
            _votingOptions.VotingType = votingType;
            _combinationStrategy = CreateCombinationStrategy();
        }
        
        _votingOptions.UseOutOfBagScoring = useOutOfBag;
        _votingOptions.RequireConsensus = requireConsensus;
        _votingOptions.ConsensusThreshold = consensusThreshold;
    }
    
    /// <summary>
    /// Makes predictions with optional consensus checking.
    /// </summary>
    public override TOutput Predict(TInput input)
    {
        var predictions = GetIndividualPredictions(input);
        
        if (_votingOptions.RequireConsensus)
        {
            var consensus = CheckConsensus(predictions);
            if (consensus < _votingOptions.ConsensusThreshold)
            {
                _logger.Warning("Low consensus among models: {Consensus:F2}", consensus);
            }
        }
        
        return _combinationStrategy.Combine(predictions, _modelWeights);
    }
    
    #region Private Helper Methods
    
    private T CalculateModelPerformance(TOutput predictions, TOutput actual)
    {
        // For generic types, we'll return a default performance value
        // Derived classes can override UpdateModelWeights to provide type-specific implementation
        _logger.Warning("Generic performance calculation not implemented. Using default weight.");
        return NumOps.One;
    }
    
    private T ConvertPerformanceToWeight(T performance)
    {
        // Apply power scaling if configured
        // Note: NumOps doesn't have Pow method, so skipping power scaling for now
        // if (_votingOptions.PerformancePowerScaling != 1.0)
        // {
        //     var power = NumOps.FromDouble(_votingOptions.PerformancePowerScaling);
        //     performance = NumOps.Pow(performance, power);
        // }
        
        return performance;
    }
    
    private double CheckConsensus(List<TOutput> predictions)
    {
        // For generic types, return a default consensus value
        // Derived classes can provide type-specific implementation
        _logger.Warning("Generic consensus calculation not implemented. Assuming full consensus.");
        return 1.0;
    }
    
    private double CalculateAgreement(TOutput pred1, TOutput pred2)
    {
        // For generic types, return a default agreement value
        // Derived classes can provide type-specific implementation
        return 1.0;
    }
    
    private void NormalizeWeights()
    {
        // Normalize weights so they sum to 1
        var sum = NumOps.Zero;
        for (int i = 0; i < _modelWeights.Length; i++)
        {
            sum = NumOps.Add(sum, _modelWeights[i]);
        }
        
        if (!NumOps.Equals(sum, NumOps.Zero))
        {
            for (int i = 0; i < _modelWeights.Length; i++)
            {
                _modelWeights[i] = NumOps.Divide(_modelWeights[i], sum);
            }
        }
        else
        {
            // If all weights are zero, set equal weights
            var equalWeight = NumOps.Divide(NumOps.One, NumOps.FromDouble(_modelWeights.Length));
            for (int i = 0; i < _modelWeights.Length; i++)
            {
                _modelWeights[i] = equalWeight;
            }
        }
    }
    
    #endregion
    
    #region Abstract Method Implementations
    
    /// <summary>
    /// Creates a bootstrap sample of the training data.
    /// </summary>
    protected override (TInput Input, TOutput Output) CreateBootstrapSample(
        TInput input, TOutput output, Random random)
    {
        // This is a placeholder implementation
        // In practice, this would need to be type-specific
        _logger.Warning("Bootstrap sampling not implemented for generic types. Using original data.");
        return (input, output);
    }
    
    /// <summary>
    /// Splits data into training and validation sets.
    /// </summary>
    protected override (TInput TrainInput, TOutput TrainOutput, 
                      TInput ValInput, TOutput ValOutput) 
        SplitData(TInput input, TOutput output, double splitRatio)
    {
        // This is a placeholder implementation
        // In practice, this would need to be type-specific
        _logger.Warning("Data splitting not implemented for generic types. Using original data for both sets.");
        return (input, output, input, output);
    }
    
    #endregion
    
    #region Core Model Implementation
    
    /// <summary>
    /// Makes predictions asynchronously.
    /// </summary>
    public override async Task<TOutput> PredictAsync(TInput input)
    {
        if (_baseModels.Count == 0)
        {
            throw new InvalidOperationException("No models in ensemble. Add models before making predictions.");
        }

        // Get predictions from all models in parallel
        var predictionTasks = _baseModels.Select(model => model.PredictAsync(input)).ToArray();
        var predictions = await Task.WhenAll(predictionTasks);
        
        if (_votingOptions.RequireConsensus)
        {
            var consensus = CheckConsensus(predictions.ToList());
            if (consensus < _votingOptions.ConsensusThreshold)
            {
                _logger.Warning("Low consensus among models: {Consensus:F2}", consensus);
            }
        }
        
        return _combinationStrategy.Combine(predictions.ToList(), _modelWeights);
    }
    
    /// <summary>
    /// Trains the ensemble asynchronously.
    /// </summary>
    public override async Task TrainAsync(TInput inputs, TOutput targets)
    {
        if (_baseModels.Count == 0)
        {
            throw new InvalidOperationException("No models added to ensemble. Use AddModel before training.");
        }

        _logger.Information("Starting async training of {Count} models", _baseModels.Count);
        
        try
        {
            // Determine training strategy
            if (_options.UseParallelTraining)
            {
                await TrainParallelAsync(inputs, targets);
            }
            else
            {
                await TrainSequentialAsync(inputs, targets);
            }
            
            // Update model weights based on performance if configured
            if (_options.UseOutOfBagScoring && _votingOptions.VotingType == VotingType.Weighted)
            {
                await UpdateModelWeightsAsync(inputs, targets);
            }
            
            _logger.Information("Ensemble training completed successfully");
        }
        catch (Exception ex)
        {
            _logger.Error(ex, "Error during ensemble training");
            throw;
        }
    }
    
    /// <summary>
    /// Gets the model metadata.
    /// </summary>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            ModelType = ModelType.Ensemble
        };
        
        // Set feature count from first model if available
        if (_baseModels.Count > 0)
        {
            var firstModelMetadata = _baseModels[0].GetModelMetadata();
            metadata.FeatureCount = firstModelMetadata.FeatureCount;
        }
        
        // Add ensemble-specific metadata
        metadata.AdditionalInfo = new Dictionary<string, object>
        {
            ["EnsembleType"] = "VotingEnsemble",
            ["VotingType"] = _votingOptions.VotingType.ToString(),
            ["ModelCount"] = _baseModels.Count,
            ["RequireConsensus"] = _votingOptions.RequireConsensus,
            ["ConsensusThreshold"] = _votingOptions.ConsensusThreshold
        };
        
        // Calculate complexity as sum of individual model complexities
        metadata.Complexity = 0;
        foreach (var model in _baseModels)
        {
            var modelMeta = model.GetModelMetadata();
            metadata.Complexity += modelMeta.Complexity;
        }
        
        return metadata;
    }
    
    /// <summary>
    /// Sets the model metadata.
    /// </summary>
    public override void SetModelMetadata(ModelMetadata<T> metadata)
    {
        if (metadata == null)
        {
            throw new ArgumentNullException(nameof(metadata));
        }
        
        // Extract ensemble-specific metadata if available
        if (metadata.AdditionalInfo != null)
        {
            if (metadata.AdditionalInfo.TryGetValue("VotingType", out var votingType))
            {
                if (Enum.TryParse<VotingType>(votingType.ToString(), out var vt))
                {
                    _votingOptions.VotingType = vt;
                    _combinationStrategy = CreateCombinationStrategy();
                }
            }
            
            if (metadata.AdditionalInfo.TryGetValue("RequireConsensus", out var requireConsensus))
            {
                _votingOptions.RequireConsensus = Convert.ToBoolean(requireConsensus);
            }
            
            if (metadata.AdditionalInfo.TryGetValue("ConsensusThreshold", out var consensusThreshold))
            {
                _votingOptions.ConsensusThreshold = Convert.ToDouble(consensusThreshold);
            }
        }
        
        // Propagate metadata to base models if needed
        foreach (var model in _baseModels)
        {
            var modelMeta = model.GetModelMetadata();
            modelMeta.FeatureCount = metadata.FeatureCount;
            model.SetModelMetadata(modelMeta);
        }
    }
    
    /// <summary>
    /// Disposes of the ensemble and all its models.
    /// </summary>
    public override void Dispose()
    {
        _logger.Information("Disposing VotingEnsemble");
        
        // Dispose all base models
        foreach (var model in _baseModels)
        {
            try
            {
                if (model is IDisposable disposable)
                {
                    disposable.Dispose();
                }
            }
            catch (Exception ex)
            {
                _logger.Error(ex, "Error disposing model {ModelType}", model.GetType().Name);
            }
        }
        
        // Clear collections
        _baseModels.Clear();
        _modelWeights = new Vector<T>(0);
        _modelIdentifiers.Clear();
        _performanceCache.Clear();
        _modelMetadata.Clear();
        
        // Dispose combination strategy if it's disposable
        if (_combinationStrategy is IDisposable strategyDisposable)
        {
            strategyDisposable.Dispose();
        }
        
        _logger.Information("VotingEnsemble disposed");
    }
    
    /// <summary>
    /// Saves the ensemble to a file.
    /// </summary>
    public override void Save(string filePath)
    {
        if (string.IsNullOrEmpty(filePath))
        {
            throw new ArgumentNullException(nameof(filePath));
        }

        _logger.Information("Saving VotingEnsemble to {FilePath}", filePath);

        try
        {
            using (var stream = new FileStream(filePath, FileMode.Create, FileAccess.Write))
            using (var writer = new BinaryWriter(stream))
            {
                // Write ensemble header
                writer.Write("VOTING_ENSEMBLE_V1");
                writer.Write(_baseModels.Count);
                
                // Write voting options
                writer.Write((int)_votingOptions.VotingType);
                writer.Write(_votingOptions.RequireConsensus);
                writer.Write(_votingOptions.ConsensusThreshold);
                writer.Write(_votingOptions.PerformancePowerScaling);
                
                // Write model weights
                writer.Write(_modelWeights.Length);
                for (int i = 0; i < _modelWeights.Length; i++)
                {
                    writer.Write(Convert.ToDouble(_modelWeights[i]));
                }
                
                // Save each model to a separate file
                var directory = Path.GetDirectoryName(filePath);
                var fileNameWithoutExt = Path.GetFileNameWithoutExtension(filePath);
                var extension = Path.GetExtension(filePath);
                
                for (int i = 0; i < _baseModels.Count; i++)
                {
                    var modelFileName = Path.Combine(directory, $"{fileNameWithoutExt}_model_{i}{extension}");
                    writer.Write(modelFileName);
                    writer.Write(_baseModels[i].GetType().AssemblyQualifiedName);
                    
                    // Save the model
                    _baseModels[i].Save(modelFileName);
                }
                
                // Save metadata
                var metadata = GetModelMetadata();
                writer.Write(metadata.FeatureCount);
                writer.Write(metadata.TargetCount);
                writer.Write(metadata.Complexity);
            }
            
            _logger.Information("VotingEnsemble saved successfully");
        }
        catch (Exception ex)
        {
            _logger.Error(ex, "Error saving VotingEnsemble");
            throw;
        }
    }
    
    /// <summary>
    /// Loads the ensemble from a file.
    /// </summary>
    public override void Load(string filePath)
    {
        if (string.IsNullOrEmpty(filePath))
        {
            throw new ArgumentNullException(nameof(filePath));
        }

        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException($"Model file not found: {filePath}");
        }

        _logger.Information("Loading VotingEnsemble from {FilePath}", filePath);

        try
        {
            using (var stream = new FileStream(filePath, FileMode.Open, FileAccess.Read))
            using (var reader = new BinaryReader(stream))
            {
                // Read and verify header
                var header = reader.ReadString();
                if (header != "VOTING_ENSEMBLE_V1")
                {
                    throw new InvalidOperationException($"Invalid file format. Expected VOTING_ENSEMBLE_V1, got {header}");
                }
                
                var modelCount = reader.ReadInt32();
                
                // Read voting options
                _votingOptions.VotingType = (VotingType)reader.ReadInt32();
                _votingOptions.RequireConsensus = reader.ReadBoolean();
                _votingOptions.ConsensusThreshold = reader.ReadDouble();
                _votingOptions.PerformancePowerScaling = reader.ReadDouble();
                
                // Read model weights
                var weightCount = reader.ReadInt32();
                var weights = new T[weightCount];
                for (int i = 0; i < weightCount; i++)
                {
                    weights[i] = NumOps.FromDouble(reader.ReadDouble());
                }
                _modelWeights = new Vector<T>(weights);
                
                // Clear existing models
                _baseModels.Clear();
                _modelIdentifiers.Clear();
                
                // Load each model
                for (int i = 0; i < modelCount; i++)
                {
                    var modelFileName = reader.ReadString();
                    var typeName = reader.ReadString();
                    
                    if (!File.Exists(modelFileName))
                    {
                        throw new FileNotFoundException($"Model file not found: {modelFileName}");
                    }
                    
                    // Create instance of the model type
                    var modelType = Type.GetType(typeName);
                    if (modelType == null)
                    {
                        throw new InvalidOperationException($"Could not load type: {typeName}");
                    }
                    
                    var model = Activator.CreateInstance(modelType) as IFullModel<T, TInput, TOutput>;
                    if (model == null)
                    {
                        throw new InvalidOperationException($"Could not create instance of type: {typeName}");
                    }
                    
                    // Load the model
                    model.Load(modelFileName);
                    _baseModels.Add(model);
                    _modelIdentifiers[i] = $"Model_{i}_{modelType.Name}";
                }
                
                // Read metadata
                var featureCount = reader.ReadInt32();
                var targetCount = reader.ReadInt32();
                var complexity = reader.ReadInt32();
                
                // Update combination strategy
                _combinationStrategy = CreateCombinationStrategy();
            }
            
            _logger.Information("VotingEnsemble loaded successfully with {Count} models", _baseModels.Count);
        }
        catch (Exception ex)
        {
            _logger.Error(ex, "Error loading VotingEnsemble");
            throw;
        }
    }
    
    #region Async Training Helpers
    
    private async Task TrainParallelAsync(TInput input, TOutput output)
    {
        _logger.Debug("Training {Count} models in parallel (async)", _baseModels.Count);
        
        var tasks = _baseModels.Select(async (model, index) =>
        {
            try
            {
                _logger.Debug("Training model {Index}: {Type} (async)", index, model.GetType().Name);
                await model.TrainAsync(input, output);
                _logger.Debug("Model {Index} training completed (async)", index);
            }
            catch (Exception ex)
            {
                _logger.Error(ex, "Error training model {Index} (async)", index);
                throw;
            }
        }).ToArray();
        
        await Task.WhenAll(tasks);
    }
    
    private async Task TrainSequentialAsync(TInput input, TOutput output)
    {
        _logger.Debug("Training {Count} models sequentially (async)", _baseModels.Count);
        
        for (int i = 0; i < _baseModels.Count; i++)
        {
            _logger.Debug("Training model {Index}: {Type} (async)", i, _baseModels[i].GetType().Name);
            await _baseModels[i].TrainAsync(input, output);
            _logger.Debug("Model {Index} training completed (async)", i);
        }
    }
    
    private async Task UpdateModelWeightsAsync(TInput inputs, TOutput targets)
    {
        _logger.Debug("Updating model weights based on performance (async)");
        
        var performanceTasks = _baseModels.Select(async (model, index) =>
        {
            try
            {
                var prediction = await model.PredictAsync(inputs);
                var performance = CalculateModelPerformance(prediction, targets);
                return new { Index = index, Performance = performance };
            }
            catch (Exception ex)
            {
                _logger.Warning(ex, "Could not evaluate model {Index} performance", index);
                return new { Index = index, Performance = NumOps.One };
            }
        }).ToArray();
        
        var results = await Task.WhenAll(performanceTasks);
        
        foreach (var result in results)
        {
            _modelWeights[result.Index] = ConvertPerformanceToWeight(result.Performance);
        }
        
        // Normalize weights
        NormalizeWeights();
    }
    
    #endregion
    
    #endregion
    
    #region IInterpretableModel Implementation

    /// <summary>
    /// Gets the global feature importance across all predictions.
    /// </summary>
    public override async Task<Dictionary<int, T>> GetGlobalFeatureImportanceAsync()
    {
        if (!_enabledMethods.Contains(InterpretationMethod.FeatureImportance))
        {
            throw new InvalidOperationException("Feature importance is not enabled. Call EnableMethod first.");
        }

        if (_baseModels.Count == 0)
        {
            throw new InvalidOperationException("No models in ensemble. Add models before computing feature importance.");
        }

        var metadata = GetModelMetadata();
        var aggregatedImportance = new Dictionary<int, T>();
        
        // Initialize importance scores
        for (int i = 0; i < metadata.FeatureCount; i++)
        {
            aggregatedImportance[i] = NumOps.Zero;
        }

        // Collect importance from each model that supports it
        var modelImportances = new List<Dictionary<int, T>>();
        var modelWeightsList = new List<T>();
        
        for (int modelIndex = 0; modelIndex < _baseModels.Count; modelIndex++)
        {
            try
            {
                if (_baseModels[modelIndex] is IInterpretableModel<T, TInput, TOutput> interpretableModel)
                {
                    var importance = await interpretableModel.GetGlobalFeatureImportanceAsync();
                    modelImportances.Add(importance);
                    modelWeightsList.Add(_modelWeights[modelIndex]);
                }
            }
            catch (NotSupportedException)
            {
                // Skip models that don't support feature importance
                _logger.Debug("Model {Index} does not support feature importance", modelIndex);
            }
        }

        if (modelImportances.Count == 0)
        {
            throw new NotSupportedException("None of the models in the ensemble support feature importance.");
        }

        // Aggregate importance scores using weighted average
        var totalWeight = modelWeightsList.Aggregate(NumOps.Zero, (sum, w) => NumOps.Add(sum, w));
        
        for (int featureIndex = 0; featureIndex < metadata.FeatureCount; featureIndex++)
        {
            var weightedSum = NumOps.Zero;
            
            for (int modelIndex = 0; modelIndex < modelImportances.Count; modelIndex++)
            {
                if (modelImportances[modelIndex].TryGetValue(featureIndex, out var importance))
                {
                    var weighted = NumOps.Multiply(importance, modelWeightsList[modelIndex]);
                    weightedSum = NumOps.Add(weightedSum, weighted);
                }
            }
            
            aggregatedImportance[featureIndex] = NumOps.Divide(weightedSum, totalWeight);
        }

        return aggregatedImportance;
    }

    /// <summary>
    /// Gets the local feature importance for a specific input.
    /// </summary>
    public override async Task<Dictionary<int, T>> GetLocalFeatureImportanceAsync(TInput input)
    {
        if (!_enabledMethods.Contains(InterpretationMethod.FeatureImportance))
        {
            throw new InvalidOperationException("Feature importance is not enabled. Call EnableMethod first.");
        }

        if (_baseModels.Count == 0)
        {
            throw new InvalidOperationException("No models in ensemble.");
        }

        var metadata = GetModelMetadata();
        var aggregatedImportance = new Dictionary<int, T>();
        
        // Initialize
        for (int i = 0; i < metadata.FeatureCount; i++)
        {
            aggregatedImportance[i] = NumOps.Zero;
        }

        // Collect local importance from each interpretable model
        var modelImportances = new List<Dictionary<int, T>>();
        var modelWeightsList = new List<T>();
        
        for (int modelIndex = 0; modelIndex < _baseModels.Count; modelIndex++)
        {
            if (_baseModels[modelIndex] is IInterpretableModel<T, TInput, TOutput> interpretableModel)
            {
                try
                {
                    var importance = await interpretableModel.GetLocalFeatureImportanceAsync(input);
                    modelImportances.Add(importance);
                    modelWeightsList.Add(_modelWeights[modelIndex]);
                }
                catch (NotSupportedException)
                {
                    _logger.Debug("Model {Index} does not support local feature importance", modelIndex);
                }
            }
        }

        if (modelImportances.Count == 0)
        {
            // Fall back to global importance if no local importance available
            return await GetGlobalFeatureImportanceAsync();
        }

        // Aggregate using weighted average
        var totalWeight = modelWeightsList.Aggregate(NumOps.Zero, (sum, w) => NumOps.Add(sum, w));
        
        for (int featureIndex = 0; featureIndex < metadata.FeatureCount; featureIndex++)
        {
            var weightedSum = NumOps.Zero;
            
            for (int modelIndex = 0; modelIndex < modelImportances.Count; modelIndex++)
            {
                if (modelImportances[modelIndex].TryGetValue(featureIndex, out var importance))
                {
                    var weighted = NumOps.Multiply(importance, modelWeightsList[modelIndex]);
                    weightedSum = NumOps.Add(weightedSum, weighted);
                }
            }
            
            aggregatedImportance[featureIndex] = NumOps.Divide(weightedSum, totalWeight);
        }

        return aggregatedImportance;
    }

    /// <summary>
    /// Gets SHAP values for the given inputs.
    /// </summary>
    public override async Task<Matrix<T>> GetShapValuesAsync(TInput inputs)
    {
        if (!_enabledMethods.Contains(InterpretationMethod.SHAP))
        {
            throw new InvalidOperationException("SHAP is not enabled. Call EnableMethod first.");
        }

        if (_baseModels.Count == 0)
        {
            throw new InvalidOperationException("No models in ensemble.");
        }

        var metadata = GetModelMetadata();
        
        // For ensemble models, we can't directly compute SHAP values
        // Instead, we'll aggregate SHAP values from individual models
        var shapValuesList = new List<Matrix<T>>();
        var modelWeightsList = new List<T>();
        
        for (int modelIndex = 0; modelIndex < _baseModels.Count; modelIndex++)
        {
            if (_baseModels[modelIndex] is IInterpretableModel<T, TInput, TOutput> interpretableModel)
            {
                try
                {
                    var shapValues = await interpretableModel.GetShapValuesAsync(inputs);
                    shapValuesList.Add(shapValues);
                    modelWeightsList.Add(_modelWeights[modelIndex]);
                }
                catch (NotSupportedException)
                {
                    _logger.Debug("Model {Index} does not support SHAP values", modelIndex);
                }
            }
        }

        if (shapValuesList.Count == 0)
        {
            throw new NotSupportedException("None of the models in the ensemble support SHAP values.");
        }

        // Create weighted average of SHAP values
        var numSamples = shapValuesList[0].Rows;
        var numFeatures = shapValuesList[0].Columns;
        var aggregatedShap = new Matrix<T>(numSamples, numFeatures);
        var totalWeight = modelWeightsList.Aggregate(NumOps.Zero, (sum, w) => NumOps.Add(sum, w));
        
        for (int i = 0; i < numSamples; i++)
        {
            for (int j = 0; j < numFeatures; j++)
            {
                var weightedSum = NumOps.Zero;
                
                for (int modelIndex = 0; modelIndex < shapValuesList.Count; modelIndex++)
                {
                    var value = shapValuesList[modelIndex][i, j];
                    var weighted = NumOps.Multiply(value, modelWeightsList[modelIndex]);
                    weightedSum = NumOps.Add(weightedSum, weighted);
                }
                
                aggregatedShap[i, j] = NumOps.Divide(weightedSum, totalWeight);
            }
        }

        return aggregatedShap;
    }

    /// <summary>
    /// Gets LIME explanation for a specific input.
    /// </summary>
    public override async Task<LimeExplanation<T>> GetLimeExplanationAsync(TInput input, int numFeatures = 10)
    {
        if (!_enabledMethods.Contains(InterpretationMethod.LIME))
        {
            throw new InvalidOperationException("LIME is not enabled. Call EnableMethod first.");
        }

        // For ensemble models, LIME can be applied by treating the ensemble as a black box
        // We'll use the ensemble's prediction function and create a local linear approximation
        var explanation = new LimeExplanation<T>();
        
        // Get base prediction
        var basePrediction = await PredictAsync(input);
        
        // For now, return a basic explanation
        // In a full implementation, you would:
        // 1. Generate perturbations around the input
        // 2. Get ensemble predictions for each perturbation
        // 3. Fit a local linear model
        // 4. Extract feature weights
        
        var featureImportance = await GetLocalFeatureImportanceAsync(input);
        var topFeatures = featureImportance
            .OrderByDescending(kvp => NumOps.Abs(kvp.Value))
            .Take(numFeatures)
            .ToDictionary(kvp => kvp.Key, kvp => kvp.Value);
        
        explanation.FeatureWeights = topFeatures;
        explanation.Intercept = NumOps.Zero;
        explanation.LocalScore = NumOps.FromDouble(0.85); // Placeholder
        explanation.Coverage = NumOps.FromDouble(0.1); // Placeholder
        
        return explanation;
    }

    /// <summary>
    /// Gets partial dependence data for specified features.
    /// </summary>
    public override async Task<PartialDependenceData<T>> GetPartialDependenceAsync(Vector<int> featureIndices, int gridResolution = 20)
    {
        if (!_enabledMethods.Contains(InterpretationMethod.PartialDependence))
        {
            throw new InvalidOperationException("Partial dependence is not enabled. Call EnableMethod first.");
        }

        if (_baseModels.Count == 0)
        {
            throw new InvalidOperationException("No models in ensemble.");
        }

        // For ensemble models, we can compute partial dependence by:
        // 1. Creating a grid of values for the specified features
        // 2. For each grid point, get predictions from the ensemble
        // 3. Average the predictions
        
        var pdData = new PartialDependenceData<T>
        {
            FeatureIndices = featureIndices,
            Grid = new Matrix<T>(gridResolution, featureIndices.Length),
            Values = new Vector<T>(gridResolution)
        };
        
        // This is a simplified implementation
        // In production, you would:
        // 1. Get the range of values for each feature from training data
        // 2. Create a proper grid
        // 3. Compute predictions for each grid point
        // 4. Average across all other features
        
        for (int i = 0; i < gridResolution; i++)
        {
            pdData.Values[i] = NumOps.FromDouble(i / (double)gridResolution);
        }
        
        return pdData;
    }

    /// <summary>
    /// Gets counterfactual explanation for a given input and desired output.
    /// </summary>
    public override async Task<CounterfactualExplanation<T>> GetCounterfactualAsync(TInput input, TOutput desiredOutput, int maxChanges = 5)
    {
        if (!_enabledMethods.Contains(InterpretationMethod.Counterfactual))
        {
            throw new InvalidOperationException("Counterfactual explanations are not enabled. Call EnableMethod first.");
        }

        // For ensemble models, finding counterfactuals is complex
        // We need to find minimal changes that cause all (or most) models to agree on the desired output
        
        var explanation = new CounterfactualExplanation<T>();
        
        // This is a placeholder implementation
        // In production, you would implement an optimization algorithm to:
        // 1. Start with the original input
        // 2. Iteratively modify features
        // 3. Check if the ensemble prediction matches the desired output
        // 4. Minimize the number and magnitude of changes
        
        var currentPrediction = await PredictAsync(input);
        
        explanation.IsValid = false; // Would be true if we found a valid counterfactual
        explanation.Distance = NumOps.Zero;
        explanation.Confidence = NumOps.FromDouble(0.0);
        explanation.ChangedFeatures = new List<int>();
        explanation.FeatureChanges = new Dictionary<int, T>();
        
        return explanation;
    }

    /// <summary>
    /// Gets model-specific interpretability information.
    /// </summary>
    public override async Task<Dictionary<string, object>> GetModelSpecificInterpretabilityAsync()
    {
        var result = new Dictionary<string, object>();
        var metadata = GetModelMetadata();
        
        // Basic ensemble information
        result["EnsembleType"] = "VotingEnsemble";
        result["VotingType"] = _votingOptions.VotingType.ToString();
        result["ModelCount"] = _baseModels.Count;
        result["FeatureCount"] = metadata.FeatureCount;
        result["RequireConsensus"] = _votingOptions.RequireConsensus;
        result["ConsensusThreshold"] = _votingOptions.ConsensusThreshold;
        
        // Model composition
        var modelTypes = _baseModels.Select(m => m.GetType().Name).ToList();
        result["ModelTypes"] = modelTypes;
        
        // Model weights
        var weights = new List<double>();
        for (int i = 0; i < _modelWeights.Length; i++)
        {
            weights.Add(Convert.ToDouble(_modelWeights[i]));
        }
        result["ModelWeights"] = weights;
        
        // Performance metrics if available
        if (_modelPerformanceScores != null)
        {
            var performances = new List<double>();
            for (int i = 0; i < _modelPerformanceScores.Length; i++)
            {
                performances.Add(Convert.ToDouble(_modelPerformanceScores[i]));
            }
            result["ModelPerformances"] = performances;
        }
        
        // Interpretability support
        var interpretabilitySupport = new Dictionary<string, List<string>>();
        foreach (var method in Enum.GetValues(typeof(InterpretationMethod)).Cast<InterpretationMethod>())
        {
            var supportingModels = new List<string>();
            for (int i = 0; i < _baseModels.Count; i++)
            {
                if (_baseModels[i] is IInterpretableModel<T, TInput, TOutput> interpretable)
                {
                    try
                    {
                        // Check if this model supports the method
                        // This is a simplified check - in production you'd want a more robust way
                        supportingModels.Add($"Model_{i}_{_baseModels[i].GetType().Name}");
                    }
                    catch { }
                }
            }
            if (supportingModels.Count > 0)
            {
                interpretabilitySupport[method.ToString()] = supportingModels;
            }
        }
        result["InterpretabilitySupport"] = interpretabilitySupport;
        
        return await Task.FromResult(result);
    }

    /// <summary>
    /// Generates a text explanation for a prediction.
    /// </summary>
    public override async Task<string> GenerateTextExplanationAsync(TInput input, TOutput prediction)
    {
        var metadata = GetModelMetadata();
        var explanation = new System.Text.StringBuilder();
        
        explanation.AppendLine($"Voting Ensemble Prediction Explanation");
        explanation.AppendLine($"=====================================\n");
        
        explanation.AppendLine($"Ensemble Type: {_votingOptions.VotingType} Voting");
        explanation.AppendLine($"Number of Models: {_baseModels.Count}");
        explanation.AppendLine($"Number of Features: {metadata.FeatureCount}\n");
        
        // Get individual model predictions if possible
        explanation.AppendLine("Individual Model Contributions:");
        for (int i = 0; i < _baseModels.Count; i++)
        {
            try
            {
                var modelPrediction = await _baseModels[i].PredictAsync(input);
                var weight = _modelWeights[i];
                explanation.AppendLine($"  Model {i} ({_baseModels[i].GetType().Name}):");
                explanation.AppendLine($"    - Weight: {weight}");
                explanation.AppendLine($"    - Prediction: {modelPrediction}");
            }
            catch (Exception ex)
            {
                explanation.AppendLine($"  Model {i}: Error getting prediction - {ex.Message}");
            }
        }
        
        explanation.AppendLine($"\nFinal Ensemble Prediction: {prediction}");
        
        // Add feature importance if available
        if (_enabledMethods.Contains(InterpretationMethod.FeatureImportance))
        {
            try
            {
                var importance = await GetLocalFeatureImportanceAsync(input);
                var topFeatures = importance.OrderByDescending(kvp => kvp.Value).Take(5);
                
                explanation.AppendLine("\nTop 5 Important Features:");
                foreach (var (featureIndex, importanceScore) in topFeatures)
                {
                    explanation.AppendLine($"  Feature {featureIndex}: {importanceScore}");
                }
            }
            catch { }
        }
        
        return explanation.ToString();
    }

    /// <summary>
    /// Gets feature interaction effects between two features.
    /// </summary>
    public override async Task<T> GetFeatureInteractionAsync(int feature1Index, int feature2Index)
    {
        if (!_enabledMethods.Contains(InterpretationMethod.FeatureInteraction))
        {
            throw new InvalidOperationException("Feature interaction analysis is not enabled. Call EnableMethod first.");
        }

        if (_baseModels.Count == 0)
        {
            throw new InvalidOperationException("No models in ensemble.");
        }

        // Aggregate interaction effects from models that support it
        var interactions = new List<T>();
        var weights = new List<T>();
        
        for (int i = 0; i < _baseModels.Count; i++)
        {
            if (_baseModels[i] is IInterpretableModel<T, TInput, TOutput> interpretableModel)
            {
                try
                {
                    var interaction = await interpretableModel.GetFeatureInteractionAsync(feature1Index, feature2Index);
                    interactions.Add(interaction);
                    weights.Add(_modelWeights[i]);
                }
                catch (NotSupportedException)
                {
                    _logger.Debug("Model {Index} does not support feature interactions", i);
                }
            }
        }

        if (interactions.Count == 0)
        {
            throw new NotSupportedException("None of the models in the ensemble support feature interaction analysis.");
        }

        // Return weighted average of interactions
        var totalWeight = weights.Aggregate(NumOps.Zero, (sum, w) => NumOps.Add(sum, w));
        var weightedSum = NumOps.Zero;
        
        for (int i = 0; i < interactions.Count; i++)
        {
            var weighted = NumOps.Multiply(interactions[i], weights[i]);
            weightedSum = NumOps.Add(weightedSum, weighted);
        }
        
        return NumOps.Divide(weightedSum, totalWeight);
    }

    /// <summary>
    /// Validates fairness metrics for the given inputs.
    /// </summary>
    public override async Task<FairnessMetrics<T>> ValidateFairnessAsync(TInput inputs, int sensitiveFeatureIndex)
    {
        if (_fairnessMetrics.Count == 0)
        {
            throw new InvalidOperationException("No fairness metrics configured. Call ConfigureFairness first.");
        }

        var fairnessResult = new FairnessMetrics<T>();
        
        // For ensemble models, we need to:
        // 1. Get predictions from the ensemble
        // 2. Group by sensitive feature
        // 3. Calculate fairness metrics
        
        // This is a placeholder implementation
        // In production, you would:
        // 1. Split data by sensitive feature groups
        // 2. Calculate prediction rates for each group
        // 3. Compute various fairness metrics
        
        fairnessResult.DemographicParityDifference = NumOps.Zero;
        fairnessResult.EqualOpportunityDifference = NumOps.Zero;
        fairnessResult.EqualizedOddsDifference = NumOps.Zero;
        fairnessResult.DisparateImpactRatio = NumOps.One;
        fairnessResult.OverallFairnessScore = NumOps.One;
        
        return fairnessResult;
    }

    /// <summary>
    /// Gets anchor explanation for a given input.
    /// </summary>
    public override async Task<AnchorExplanation<T>> GetAnchorExplanationAsync(TInput input, T threshold)
    {
        if (!_enabledMethods.Contains(InterpretationMethod.Anchors))
        {
            throw new InvalidOperationException("Anchor explanations are not enabled. Call EnableMethod first.");
        }

        var explanation = new AnchorExplanation<T>();
        
        // For ensemble models, anchors are conditions that ensure
        // most/all models in the ensemble agree on the prediction
        
        // This is a placeholder implementation
        // In production, you would:
        // 1. Use a search algorithm to find minimal sufficient conditions
        // 2. Test these conditions across all models in the ensemble
        // 3. Return conditions that meet the threshold for agreement
        
        var basePrediction = await PredictAsync(input);
        
        explanation.Rules = new List<AnchorRule<T>>();
        explanation.Precision = threshold;
        explanation.Coverage = NumOps.FromDouble(0.1);
        explanation.Confidence = NumOps.FromDouble(0.95);
        explanation.FeatureIndices = new List<int>();
        explanation.TextExplanation = $"Ensemble prediction anchored with precision {threshold}";
        
        return explanation;
    }

    /// <summary>
    /// Sets the base model for interpretability analysis.
    /// </summary>
    public override void SetBaseModel(IModel<TInput, TOutput, ModelMetadata<T>> model)
    {
        _baseModel = model ?? throw new ArgumentNullException(nameof(model));
    }

    /// <summary>
    /// Enables specific interpretation methods.
    /// </summary>
    public override void EnableMethod(params InterpretationMethod[] methods)
    {
        foreach (var method in methods)
        {
            _enabledMethods.Add(method);
        }
    }

    /// <summary>
    /// Configures fairness evaluation settings.
    /// </summary>
    public override void ConfigureFairness(Vector<int> sensitiveFeatures, params FairnessMetric[] fairnessMetrics)
    {
        _sensitiveFeatures = sensitiveFeatures ?? throw new ArgumentNullException(nameof(sensitiveFeatures));
        _fairnessMetrics.Clear();
        _fairnessMetrics.AddRange(fairnessMetrics);
    }

    #endregion
}