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
    public abstract ModelMetaData<T> GetModelMetaData();
    
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
}