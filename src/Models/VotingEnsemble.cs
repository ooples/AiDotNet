using AiDotNet.Models.Options;
using AiDotNet.Ensemble.Strategies;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Models;

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
    private readonly VotingEnsembleOptions<T> _votingOptions = default!;
    
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
}

/// <summary>
/// Configuration options specific to voting ensembles.
/// </summary>
public class VotingEnsembleOptions<T> : EnsembleOptions<T>
{
    /// <summary>
    /// Gets or sets the type of voting to use.
    /// </summary>
    public VotingType VotingType { get; set; } = VotingType.Weighted;
    
    /// <summary>
    /// Gets or sets whether to check for consensus among models.
    /// </summary>
    public bool RequireConsensus { get; set; } = false;
    
    /// <summary>
    /// Gets or sets the minimum consensus threshold (0-1).
    /// </summary>
    public double ConsensusThreshold { get; set; } = 0.7;
    
    /// <summary>
    /// Gets or sets the power scaling for performance-based weights.
    /// </summary>
    /// <remarks>
    /// Values > 1 amplify differences in performance, < 1 reduce differences.
    /// </remarks>
    public double PerformancePowerScaling { get; set; } = 1.0;
}

/// <summary>
/// Defines the types of voting strategies available.
/// </summary>
public enum VotingType
{
    /// <summary>
    /// Each model gets one vote for its predicted class.
    /// </summary>
    Hard,
    
    /// <summary>
    /// Models vote with their prediction probabilities.
    /// </summary>
    Soft,
    
    /// <summary>
    /// Models vote with weights based on their performance.
    /// </summary>
    Weighted
}