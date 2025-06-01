namespace AiDotNet.Interfaces;

/// <summary>
/// Represents an online model with concept drift detection capabilities.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The type of input data.</typeparam>
/// <typeparam name="TOutput">The type of output data.</typeparam>
public interface IAdaptiveOnlineModel<T, TInput, TOutput> : IOnlineModel<T, TInput, TOutput>
{
    /// <summary>
    /// Gets whether concept drift has been detected.
    /// </summary>
    bool DriftDetected { get; }
    
    /// <summary>
    /// Gets the current drift level (0 = no drift, 1 = maximum drift).
    /// </summary>
    T DriftLevel { get; }
    
    /// <summary>
    /// Adapts the model to handle detected concept drift.
    /// </summary>
    void AdaptToDrift();
    
    /// <summary>
    /// Gets or sets the sensitivity of drift detection (0 = least sensitive, 1 = most sensitive).
    /// </summary>
    T DriftSensitivity { get; set; }
}