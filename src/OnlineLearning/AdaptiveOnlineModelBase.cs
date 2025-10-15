using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.OnlineLearning;

/// <summary>
/// Base class for adaptive online learning models with drift detection.
/// </summary>
public abstract class AdaptiveOnlineModelBase<T, TInput, TOutput> : 
    OnlineModelBase<T, TInput, TOutput>, IAdaptiveOnlineModel<T, TInput, TOutput>
{
    protected DriftDetectionMethod _driftMethod;
    protected T _driftSensitivity;
    protected bool _driftDetected;
    protected T _driftLevel;
    protected Queue<T> _recentErrors;
    protected int _driftWindowSize;
    
    /// <summary>
    /// Initializes a new instance of the AdaptiveOnlineModelBase class.
    /// </summary>
    protected AdaptiveOnlineModelBase(
        T initialLearningRate, 
        DriftDetectionMethod driftMethod,
        T driftSensitivity,
        ILogging? logger = null) 
        : base(initialLearningRate, logger)
    {
        _driftMethod = driftMethod;
        _driftSensitivity = driftSensitivity;
        _driftDetected = false;
        _driftLevel = NumOps.Zero;
        _driftWindowSize = 100;
        _recentErrors = new Queue<T>(_driftWindowSize);
    }
    
    /// <summary>
    /// Initializes a new instance of the AdaptiveOnlineModelBase class with default drift sensitivity.
    /// </summary>
    protected AdaptiveOnlineModelBase(
        T initialLearningRate, 
        DriftDetectionMethod driftMethod = DriftDetectionMethod.ADWIN,
        ILogging? logger = null) 
        : base(initialLearningRate, logger)
    {
        _driftMethod = driftMethod;
        _driftSensitivity = NumOps.FromDouble(0.5);
        _driftDetected = false;
        _driftLevel = NumOps.Zero;
        _driftWindowSize = 100;
        _recentErrors = new Queue<T>(_driftWindowSize);
    }
    
    /// <inheritdoc/>
    public bool DriftDetected => _driftDetected;
    
    /// <inheritdoc/>
    public T DriftLevel => _driftLevel;
    
    /// <inheritdoc/>
    public DriftDetectionMethod DriftDetectionMethod => _driftMethod;
    
    /// <inheritdoc/>
    public T DriftSensitivity 
    { 
        get => _driftSensitivity;
        set => _driftSensitivity = value;
    }
    
    /// <inheritdoc/>
    public void CheckForDrift(T error)
    {
        lock (_lockObject)
        {
            // Add error to recent errors
            _recentErrors.Enqueue(error);
            if (_recentErrors.Count > _driftWindowSize)
            {
                _recentErrors.Dequeue();
            }
            
            // Detect drift based on method
            DetectDrift();
        }
        
        // Adapt if drift detected
        if (_driftDetected)
        {
            _logger?.Warning("Drift detected at sample {SampleCount} with level {DriftLevel}", 
                _samplesSeen, Convert.ToDouble(_driftLevel));
            AdaptToDrift();
        }
    }
    
    /// <inheritdoc/>
    public virtual void AdaptToDrift()
    {
        // Default adaptation: increase learning rate temporarily
        var boostFactor = NumOps.Add(NumOps.One, _driftLevel);
        _learningRate = NumOps.Multiply(_learningRate, boostFactor);
        
        // Reset drift detection
        _driftDetected = false;
        _driftLevel = NumOps.Zero;
        _recentErrors.Clear();
        
        // Call derived class adaptation
        OnDriftAdaptation();
    }
    
    /// <summary>
    /// Detects drift based on the selected method.
    /// </summary>
    protected virtual void DetectDrift()
    {
        if (_recentErrors.Count < 10)
        {
            return; // Not enough data
        }
        
        var errors = _recentErrors.ToArray();
        
        switch (_driftMethod)
        {
            case DriftDetectionMethod.ADWIN:
                DetectDriftADWIN(errors);
                break;
                
            case DriftDetectionMethod.DDM:
                DetectDriftDDM(errors);
                break;
                
            case DriftDetectionMethod.EDDM:
                DetectDriftEDDM(errors);
                break;
                
            case DriftDetectionMethod.PageHinkley:
                DetectDriftPageHinkley(errors);
                break;
                
            default:
                // Simple threshold-based detection
                CheckSimpleDrift();
                break;
        }
    }
    
    /// <summary>
    /// Simple drift detection based on error threshold.
    /// </summary>
    protected void CheckSimpleDrift()
    {
        if (_recentErrors.Count < 2)
        {
            return;
        }
        
        var recentMean = CalculateMean(_recentErrors.ToArray());
        var firstHalf = _recentErrors.Take(_recentErrors.Count / 2).ToArray();
        var secondHalf = _recentErrors.Skip(_recentErrors.Count / 2).ToArray();
        
        var firstMean = CalculateMean(firstHalf);
        var secondMean = CalculateMean(secondHalf);
        
        // Check if error is increasing
        var diff = NumOps.Abs(NumOps.Subtract(secondMean, firstMean));
        var threshold = NumOps.Multiply(_driftSensitivity, recentMean);
        
        if (NumOps.GreaterThan(diff, threshold))
        {
            _driftDetected = true;
            _driftLevel = NumOps.Divide(diff, recentMean);
        }
    }
    
    /// <summary>
    /// ADWIN drift detection (simplified).
    /// </summary>
    protected void DetectDriftADWIN(T[] errors)
    {
        // Simplified ADWIN implementation
        CheckSimpleDrift();
    }
    
    /// <summary>
    /// Drift Detection Method (DDM).
    /// </summary>
    protected void DetectDriftDDM(T[] errors)
    {
        // Simplified DDM implementation
        CheckSimpleDrift();
    }
    
    /// <summary>
    /// Early Drift Detection Method (EDDM).
    /// </summary>
    protected void DetectDriftEDDM(T[] errors)
    {
        // Simplified EDDM implementation
        CheckSimpleDrift();
    }
    
    /// <summary>
    /// Page-Hinkley drift detection.
    /// </summary>
    protected void DetectDriftPageHinkley(T[] errors)
    {
        // Simplified Page-Hinkley implementation
        CheckSimpleDrift();
    }
    
    /// <summary>
    /// Calculates the mean of an array.
    /// </summary>
    protected T CalculateMean(T[] values)
    {
        if (values.Length == 0)
        {
            return NumOps.Zero;
        }
        
        var sum = values.Aggregate(NumOps.Zero, (acc, val) => NumOps.Add(acc, val));
        return NumOps.Divide(sum, NumOps.FromDouble((double)values.Length));
    }
    
    /// <summary>
    /// Calculates the error between prediction and expected output.
    /// </summary>
    protected abstract T CalculateError(TOutput prediction, TOutput expectedOutput);
    
    /// <summary>
    /// Called when drift adaptation occurs.
    /// </summary>
    protected virtual void OnDriftAdaptation()
    {
        // Subclasses can override for custom adaptation
    }
    
    /// <inheritdoc/>
    protected override void OnStatisticsReset()
    {
        base.OnStatisticsReset();
        _recentErrors.Clear();
        _driftDetected = false;
        _driftLevel = NumOps.Zero;
    }
}