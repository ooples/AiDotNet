using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Logging;
using AiDotNet.Factories;
using AiDotNet.Helpers;
using AiDotNet.Statistics;

using AiDotNet.Interpretability;

namespace AiDotNet.OnlineLearning;

/// <summary>
/// Base class for online learning models.
/// </summary>
public abstract class OnlineModelBase<T, TInput, TOutput> : IOnlineModel<T, TInput, TOutput>
{
    protected readonly INumericOperations<T> NumOps;
    protected readonly ILogging _logger;
    protected T _learningRate;
    protected bool _adaptiveLearningRate;
    protected long _samplesSeen;
    protected readonly object _lockObject = new();
    
    /// <summary>
    /// Gets or sets the model parameters as a vector.
    /// </summary>
    protected Vector<T> ModelParameters { get; set; } = Vector<T>.Empty();
    
    /// <summary>
    /// Initializes a new instance of the OnlineModelBase class.
    /// </summary>
    protected OnlineModelBase(T initialLearningRate, ILogging? logger = null)
    {
        NumOps = MathHelper.GetNumericOperations<T>();
        _logger = logger ?? LoggingFactory.GetLogger<OnlineModelBase<T, TInput, TOutput>>();
        _learningRate = initialLearningRate;
        _adaptiveLearningRate = false;
        _samplesSeen = 0;
    }
    
    /// <inheritdoc/>
    public T LearningRate => _learningRate;
    
    /// <inheritdoc/>
    public bool AdaptiveLearningRate
    {
        get => _adaptiveLearningRate;
        set => _adaptiveLearningRate = value;
    }
    
    /// <inheritdoc/>
    public long SamplesSeen => _samplesSeen;
    
    /// <inheritdoc/>
    public virtual void PartialFit(TInput input, TOutput expectedOutput)
    {
        lock (_lockObject)
        {
            var effectiveLearningRate = GetAdaptiveLearningRate();
            
            // Perform the actual update
            PerformUpdate(input, expectedOutput, effectiveLearningRate);
            
            _samplesSeen++;
            
            // Update learning rate if adaptive
            if (_adaptiveLearningRate)
            {
                UpdateLearningRate();
            }
        }
    }
    
    /// <inheritdoc/>
    public virtual void PartialFit(TInput input, TOutput expectedOutput, T learningRate)
    {
        lock (_lockObject)
        {
            // Perform the actual update with custom learning rate
            PerformUpdate(input, expectedOutput, learningRate);
            
            _samplesSeen++;
        }
    }
    
    /// <inheritdoc/>
    public virtual void PartialFitBatch(TInput[] inputs, TOutput[] expectedOutputs)
    {
        if (inputs.Length != expectedOutputs.Length)
        {
            throw new ArgumentException("Input and output arrays must have the same length.");
        }
        
        lock (_lockObject)
        {
            var effectiveLearningRate = GetAdaptiveLearningRate();
            
            // Process batch
            for (int i = 0; i < inputs.Length; i++)
            {
                PerformUpdate(inputs[i], expectedOutputs[i], effectiveLearningRate);
                _samplesSeen++;
            }
            
            // Update learning rate once after batch if adaptive
            if (_adaptiveLearningRate)
            {
                UpdateLearningRate();
            }
        }
    }
    
    /// <inheritdoc/>
    public virtual void PartialFitBatch(TInput[] inputs, TOutput[] expectedOutputs, T learningRate)
    {
        if (inputs.Length != expectedOutputs.Length)
        {
            throw new ArgumentException("Input and output arrays must have the same length.");
        }
        
        lock (_lockObject)
        {
            // Process batch with custom learning rate
            for (int i = 0; i < inputs.Length; i++)
            {
                PerformUpdate(inputs[i], expectedOutputs[i], learningRate);
                _samplesSeen++;
            }
        }
    }
    
    /// <inheritdoc/>
    public virtual void ResetStatistics()
    {
        lock (_lockObject)
        {
            _samplesSeen = 0;
            OnStatisticsReset();
        }
    }
    
    /// <summary>
    /// Performs the actual parameter update for a single sample.
    /// </summary>
    protected abstract void PerformUpdate(TInput input, TOutput expectedOutput, T learningRate);
    
    /// <summary>
    /// Gets the current adaptive learning rate.
    /// </summary>
    protected virtual T GetAdaptiveLearningRate()
    {
        if (!_adaptiveLearningRate)
        {
            return _learningRate;
        }
        
        // Default decay schedule: lr = lr_0 / (1 + decay * t)
        var decay = NumOps.FromDouble(0.001);
        var t = NumOps.FromDouble((double)_samplesSeen);
        var denominator = NumOps.Add(NumOps.One, NumOps.Multiply(decay, t));
        return NumOps.Divide(_learningRate, denominator);
    }
    
    /// <summary>
    /// Updates the learning rate based on performance.
    /// </summary>
    protected virtual void UpdateLearningRate()
    {
        // Subclasses can override for custom learning rate schedules
        _learningRate = GetAdaptiveLearningRate();
    }
    
    /// <summary>
    /// Called when statistics are reset.
    /// </summary>
    protected virtual void OnStatisticsReset()
    {
        // Subclasses can override to reset their own statistics
    }
    
    // Abstract methods from IModel that must be implemented by derived classes
    public abstract void Train(TInput input, TOutput expectedOutput);
    public abstract TOutput Predict(TInput input);
    public abstract ModelMetadata<T> GetModelMetadata();
    
    // Abstract methods from IModelSerializer
    public abstract byte[] Serialize();
    public abstract void Deserialize(byte[] data);
    
    // Abstract methods from IParameterizable - must be implemented by derived classes
    public abstract Vector<T> GetParameters();
    public abstract void SetParameters(Vector<T> parameters);
    public abstract IFullModel<T, TInput, TOutput> WithParameters(Vector<T> parameters);
    
    // Additional abstract method for dictionary-based parameters
    public abstract IFullModel<T, TInput, TOutput> WithParameters(IDictionary<string, object> parameters);
    
    // Abstract methods from IFeatureAware
    public abstract int GetInputFeatureCount();
    public abstract int GetOutputFeatureCount();
    public abstract IEnumerable<int> GetActiveFeatureIndices();
    public abstract bool IsFeatureUsed(int featureIndex);
    public abstract void SetActiveFeatureIndices(IEnumerable<int> featureIndices);
    
    // Abstract methods from ICloneable
    public abstract IFullModel<T, TInput, TOutput> Clone();
    public abstract IFullModel<T, TInput, TOutput> DeepCopy();
    
    // Abstract properties from IFullModel
    public abstract int InputDimensions { get; }
    public abstract int OutputDimensions { get; }
    public abstract bool IsTrained { get; }
    
    // Abstract methods from IFullModel
    public abstract TOutput[] PredictBatch(TInput[] inputBatch);
    public abstract Dictionary<string, double> Evaluate(TInput testData, TOutput testLabels);
    public abstract void SaveModel(string filePath);
    public abstract double GetTrainingLoss();
    public abstract double GetValidationLoss();
    public abstract Vector<T> GetModelParameters();
    public abstract ModelStats<T> GetStats();
    public abstract void Save();
    public abstract void Load();
    public abstract void Dispose();

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