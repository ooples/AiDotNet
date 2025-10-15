using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Helpers;
using AiDotNet.Interpretability;

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
    protected IFullModel<T, TInput, TOutput>? SourceModel;
    
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
        SourceModel = sourceModel;

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
    public abstract ModelMetadata<T> GetModelMetadata();
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

    #region IInterpretableModel Implementation

    protected readonly HashSet<InterpretationMethod> _enabledMethods = new();
    protected Vector<int> _sensitiveFeatures;
    protected readonly List<FairnessMetric> _fairnessMetrics = new();
    protected IModel<TInput, TOutput, ModelMetadata<T>> _baseModel;

    /// <summary>
    /// Gets the global feature importance across all predictions.
    /// </summary>
    public virtual async Task<Dictionary<int, T>> GetGlobalFeatureImportanceAsync()
    {
        return await InterpretableModelHelper.GetGlobalFeatureImportanceAsync(this, _enabledMethods);
    }

    /// <summary>
    /// Gets the local feature importance for a specific input.
    /// </summary>
    public virtual async Task<Dictionary<int, T>> GetLocalFeatureImportanceAsync(TInput input)
    {
        return await InterpretableModelHelper.GetLocalFeatureImportanceAsync(this, _enabledMethods, input);
    }

    /// <summary>
    /// Gets SHAP values for the given inputs.
    /// </summary>
    public virtual async Task<Matrix<T>> GetShapValuesAsync(TInput inputs)
    {
        return await InterpretableModelHelper.GetShapValuesAsync(this, _enabledMethods);
    }

    /// <summary>
    /// Gets LIME explanation for a specific input.
    /// </summary>
    public virtual async Task<LimeExplanation<T>> GetLimeExplanationAsync(TInput input, int numFeatures = 10)
    {
        return await InterpretableModelHelper.GetLimeExplanationAsync<T>(_enabledMethods, numFeatures);
    }

    /// <summary>
    /// Gets partial dependence data for specified features.
    /// </summary>
    public virtual async Task<PartialDependenceData<T>> GetPartialDependenceAsync(Vector<int> featureIndices, int gridResolution = 20)
    {
        return await InterpretableModelHelper.GetPartialDependenceAsync<T>(_enabledMethods, featureIndices, gridResolution);
    }

    /// <summary>
    /// Gets counterfactual explanation for a given input and desired output.
    /// </summary>
    public virtual async Task<CounterfactualExplanation<T>> GetCounterfactualAsync(TInput input, TOutput desiredOutput, int maxChanges = 5)
    {
        return await InterpretableModelHelper.GetCounterfactualAsync<T>(_enabledMethods, maxChanges);
    }

    /// <summary>
    /// Gets model-specific interpretability information.
    /// </summary>
    public virtual async Task<Dictionary<string, object>> GetModelSpecificInterpretabilityAsync()
    {
        return await InterpretableModelHelper.GetModelSpecificInterpretabilityAsync(this);
    }

    /// <summary>
    /// Generates a text explanation for a prediction.
    /// </summary>
    public virtual async Task<string> GenerateTextExplanationAsync(TInput input, TOutput prediction)
    {
        return await InterpretableModelHelper.GenerateTextExplanationAsync(this, input, prediction);
    }

    /// <summary>
    /// Gets feature interaction effects between two features.
    /// </summary>
    public virtual async Task<T> GetFeatureInteractionAsync(int feature1Index, int feature2Index)
    {
        return await InterpretableModelHelper.GetFeatureInteractionAsync<T>(_enabledMethods, feature1Index, feature2Index);
    }

    /// <summary>
    /// Validates fairness metrics for the given inputs.
    /// </summary>
    public virtual async Task<FairnessMetrics<T>> ValidateFairnessAsync(TInput inputs, int sensitiveFeatureIndex)
    {
        return await InterpretableModelHelper.ValidateFairnessAsync<T>(_fairnessMetrics);
    }

    /// <summary>
    /// Gets anchor explanation for a given input.
    /// </summary>
    public virtual async Task<AnchorExplanation<T>> GetAnchorExplanationAsync(TInput input, T threshold)
    {
        return await InterpretableModelHelper.GetAnchorExplanationAsync(_enabledMethods, threshold);
    }

    /// <summary>
    /// Sets the base model for interpretability analysis.
    /// </summary>
    public virtual void SetBaseModel(IModel<TInput, TOutput, ModelMetadata<T>> model)
    {
        _baseModel = model ?? throw new ArgumentNullException(nameof(model));
    }

    /// <summary>
    /// Enables specific interpretation methods.
    /// </summary>
    public virtual void EnableMethod(params InterpretationMethod[] methods)
    {
        foreach (var method in methods)
        {
            _enabledMethods.Add(method);
        }
    }

    /// <summary>
    /// Configures fairness evaluation settings.
    /// </summary>
    public virtual void ConfigureFairness(Vector<int> sensitiveFeatures, params FairnessMetric[] fairnessMetrics)
    {
        _sensitiveFeatures = sensitiveFeatures ?? throw new ArgumentNullException(nameof(sensitiveFeatures));
        _fairnessMetrics.Clear();
        _fairnessMetrics.AddRange(fairnessMetrics);
    }

    #endregion
}