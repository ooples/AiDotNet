using System;
using AiDotNet.Enums;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for online learning models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class OnlineModelOptions<T> : ModelOptions
{
    /// <summary>
    /// Gets or sets the initial learning rate.
    /// </summary>
    public T InitialLearningRate { get; set; } = default!;
    
    /// <summary>
    /// Gets or sets whether to use adaptive learning rate.
    /// </summary>
    public bool UseAdaptiveLearningRate { get; set; } = true;
    
    /// <summary>
    /// Gets or sets the learning rate decay factor.
    /// </summary>
    public T LearningRateDecay { get; set; } = default!;
    
    /// <summary>
    /// Gets or sets the mini-batch size for batch updates.
    /// </summary>
    public int MiniBatchSize { get; set; } = 1;
    
    /// <summary>
    /// Gets or sets the regularization parameter.
    /// </summary>
    public T RegularizationParameter { get; set; } = default!;
    
    /// <summary>
    /// Gets or sets whether to use momentum.
    /// </summary>
    public bool UseMomentum { get; set; } = false;
    
    /// <summary>
    /// Gets or sets the momentum factor.
    /// </summary>
    public T MomentumFactor { get; set; } = default!;
    
    /// <summary>
    /// Gets or sets the maximum number of features to keep (for sparse models).
    /// </summary>
    public int? MaxFeatures { get; set; }
    
    /// <summary>
    /// Gets or sets whether to normalize inputs.
    /// </summary>
    public bool NormalizeInputs { get; set; } = true;
    
    /// <summary>
    /// Gets or sets the buffer size for streaming data.
    /// </summary>
    public int StreamBufferSize { get; set; } = 100;
    
    /// <summary>
    /// Gets or sets the timeout for stream buffer processing.
    /// </summary>
    public TimeSpan StreamBufferTimeout { get; set; } = TimeSpan.FromSeconds(1);
    
    /// <summary>
    /// Gets or sets the aggressiveness parameter (for Passive-Aggressive algorithms).
    /// </summary>
    public T AggressivenessParameter { get; set; } = default!;
    
    /// <summary>
    /// Gets or sets the epsilon parameter (for loss-insensitive algorithms).
    /// </summary>
    public T Epsilon { get; set; } = default!;
}

/// <summary>
/// Configuration options for adaptive online learning models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class AdaptiveOnlineModelOptions<T> : OnlineModelOptions<T>
{
    /// <summary>
    /// Gets or sets the drift detection method.
    /// </summary>
    public DriftDetectionMethod DriftDetectionMethod { get; set; } = DriftDetectionMethod.ADWIN;
    
    /// <summary>
    /// Gets or sets the drift sensitivity (0 = least sensitive, 1 = most sensitive).
    /// </summary>
    public T DriftSensitivity { get; set; } = default!;
    
    /// <summary>
    /// Gets or sets the window size for drift detection.
    /// </summary>
    public int DriftWindowSize { get; set; } = 100;
    
    /// <summary>
    /// Gets or sets whether to reset the model when drift is detected.
    /// </summary>
    public bool ResetOnDrift { get; set; } = false;
    
    /// <summary>
    /// Gets or sets the learning rate boost factor when drift is detected.
    /// </summary>
    public T DriftLearningRateBoost { get; set; } = default!;
}

/// <summary>
/// Configuration options for forgetful models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class ForgetfulModelOptions<T> : AdaptiveOnlineModelOptions<T>
{
    /// <summary>
    /// Gets or sets the forgetting factor (0 = no forgetting, 1 = complete forgetting).
    /// </summary>
    public T ForgettingFactor { get; set; } = default!;
    
    /// <summary>
    /// Gets or sets the sliding window size.
    /// </summary>
    public int? WindowSize { get; set; }
    
    /// <summary>
    /// Gets or sets whether to use exponential forgetting.
    /// </summary>
    public bool UseExponentialForgetting { get; set; } = true;
}