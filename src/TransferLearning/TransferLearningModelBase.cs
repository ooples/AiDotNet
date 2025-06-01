using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Helpers;

namespace AiDotNet.TransferLearning;

/// <summary>
/// Base class for models that support transfer learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The type of input data.</typeparam>
/// <typeparam name="TOutput">The type of output data.</typeparam>
public abstract class TransferLearningModelBase<T, TInput, TOutput> : ITransferLearningModel<T, TInput, TOutput>
{
    protected readonly INumericOperations<T> NumOps;
    protected readonly HashSet<int> _frozenLayers = new();
    protected Dictionary<int, T> _layerLearningRates = new();
    protected TransferInfo<T>? _transferInfo;
    protected ILogging? _logger;
    
    /// <summary>
    /// Initializes a new instance of the TransferLearningModelBase class.
    /// </summary>
    protected TransferLearningModelBase(ILogging? logger = null)
    {
        NumOps = MathHelper.GetNumericOperations<T>();
        _logger = logger;
    }
    
    /// <inheritdoc/>
    public virtual void TransferFrom(IFullModel<T, TInput, TOutput> sourceModel, 
                                   TransferLearningStrategy strategy, 
                                   TransferLearningOptions<T>? options = null)
    {
        options ??= new TransferLearningOptions<T>();
        
        _logger?.Information($"Starting transfer learning with strategy: {strategy}");
        
        switch (strategy)
        {
            case TransferLearningStrategy.FeatureExtraction:
                TransferFeatureExtraction(sourceModel, options);
                break;
                
            case TransferLearningStrategy.FineTuning:
                TransferFineTuning(sourceModel, options);
                break;
                
            case TransferLearningStrategy.ProgressiveUnfreezing:
                TransferProgressiveUnfreezing(sourceModel, options);
                break;
                
            case TransferLearningStrategy.DiscriminativeFineTuning:
                TransferDiscriminativeFineTuning(sourceModel, options);
                break;
                
            case TransferLearningStrategy.DomainAdaptation:
                TransferDomainAdaptation(sourceModel, options);
                break;
                
            default:
                TransferCustomStrategy(sourceModel, strategy, options);
                break;
        }
        
        // Record transfer information
        _transferInfo = new TransferInfo<T>
        {
            SourceModelType = sourceModel.GetType().Name,
            TransferStrategy = strategy,
            TransferDate = DateTime.UtcNow,
            LayersTransferred = GetTransferredLayerCount(),
            Options = options
        };
    }
    
    /// <inheritdoc/>
    public abstract void TransferFrom<TSourceInput, TSourceOutput>(
        IFullModel<T, TSourceInput, TSourceOutput> sourceModel,
        IInputAdapter<T, TSourceInput, TInput> inputAdapter,
        IOutputAdapter<T, TSourceOutput, TOutput> outputAdapter,
        TransferLearningStrategy strategy,
        TransferLearningOptions<T>? options = null);
    
    /// <inheritdoc/>
    public virtual void FineTune(TInput[] inputs, TOutput[] outputs, FineTuningOptions<T>? options = null)
    {
        options ??= new FineTuningOptions<T>();
        
        _logger?.Information($"Starting fine-tuning with {inputs.Length} samples");
        
        // Apply fine-tuning specific settings
        if (options.UnfreezeGradually)
        {
            ProgressiveUnfreezing(inputs, outputs, options);
        }
        else
        {
            StandardFineTuning(inputs, outputs, options);
        }
    }
    
    /// <inheritdoc/>
    public virtual void FreezeLayers(IEnumerable<int> layerIndices)
    {
        foreach (var index in layerIndices)
        {
            _frozenLayers.Add(index);
            _logger?.Debug($"Frozen layer {index}");
        }
    }
    
    /// <inheritdoc/>
    public virtual void UnfreezeLayers(IEnumerable<int> layerIndices)
    {
        foreach (var index in layerIndices)
        {
            _frozenLayers.Remove(index);
            _logger?.Debug($"Unfrozen layer {index}");
        }
    }
    
    /// <inheritdoc/>
    public virtual IEnumerable<int> GetFrozenLayers()
    {
        return _frozenLayers.ToList();
    }
    
    /// <inheritdoc/>
    public virtual void SetLayerLearningRates(IDictionary<int, T> layerLearningRates)
    {
        _layerLearningRates = new Dictionary<int, T>(layerLearningRates);
        _logger?.Information($"Set learning rates for {layerLearningRates.Count} layers");
    }
    
    /// <inheritdoc/>
    public virtual TransferInfo<T> GetTransferInfo()
    {
        return _transferInfo ?? new TransferInfo<T>
        {
            SourceModelType = "None",
            TransferStrategy = TransferLearningStrategy.FeatureExtraction,
            TransferDate = DateTime.MinValue,
            LayersTransferred = 0
        };
    }
    
    /// <inheritdoc/>
    public abstract void AdaptDomain(TInput[] sourceData, TInput[] targetData, DomainAdaptationMethod method);
    
    /// <inheritdoc/>
    public abstract T GetTransferabilityScore(TInput[] targetData, TOutput[] targetLabels);
    
    // Protected methods for different transfer strategies
    protected abstract void TransferFeatureExtraction(IFullModel<T, TInput, TOutput> sourceModel, TransferLearningOptions<T> options);
    protected abstract void TransferFineTuning(IFullModel<T, TInput, TOutput> sourceModel, TransferLearningOptions<T> options);
    protected abstract void TransferProgressiveUnfreezing(IFullModel<T, TInput, TOutput> sourceModel, TransferLearningOptions<T> options);
    protected abstract void TransferDiscriminativeFineTuning(IFullModel<T, TInput, TOutput> sourceModel, TransferLearningOptions<T> options);
    protected abstract void TransferDomainAdaptation(IFullModel<T, TInput, TOutput> sourceModel, TransferLearningOptions<T> options);
    protected abstract void TransferCustomStrategy(IFullModel<T, TInput, TOutput> sourceModel, TransferLearningStrategy strategy, TransferLearningOptions<T> options);
    
    protected abstract int GetTransferredLayerCount();
    protected abstract void ProgressiveUnfreezing(TInput[] inputs, TOutput[] outputs, FineTuningOptions<T> options);
    protected abstract void StandardFineTuning(TInput[] inputs, TOutput[] outputs, FineTuningOptions<T> options);
    
    // Abstract methods from IFullModel that must be implemented by derived classes
    public abstract void Train(TInput input, TOutput expectedOutput);
    public abstract TOutput Predict(TInput input);
    public abstract ModelMetaData<T> GetModelMetaData();
    public abstract byte[] Serialize();
    public abstract void Deserialize(byte[] data);
    public abstract IFullModel<T, TInput, TOutput> WithParameters(IDictionary<string, object> parameters);
    public abstract Vector<T> GetParameters();
    public abstract void SetParameters(Vector<T> parameters);
    public abstract IFullModel<T, TInput, TOutput> WithParameters(Vector<T> parameters);
    public abstract int GetInputFeatureCount();
    public abstract int GetOutputFeatureCount();
    public abstract IEnumerable<int> GetActiveFeatureIndices();
    public abstract bool IsFeatureUsed(int featureIndex);
    public abstract void SetActiveFeatureIndices(IEnumerable<int> featureIndices);
    public abstract IFullModel<T, TInput, TOutput> Clone();
    public abstract IFullModel<T, TInput, TOutput> DeepCopy();
}