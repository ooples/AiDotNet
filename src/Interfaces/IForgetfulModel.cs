namespace AiDotNet.Interfaces;

/// <summary>
/// Represents a model that can forget old patterns (useful for non-stationary data).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public interface IForgetfulModel<T>
{
    /// <summary>
    /// Gets or sets the forgetting factor (0 = no forgetting, 1 = complete forgetting).
    /// </summary>
    T ForgettingFactor { get; set; }
    
    /// <summary>
    /// Gets or sets the window size for sliding window approaches.
    /// </summary>
    int? WindowSize { get; set; }
}