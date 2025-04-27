namespace AiDotNet.Enums;

/// <summary>
/// Defines the mode of optimization to use.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> This determines what aspects of the model the optimizer will try to improve:
/// just the features used, just the parameter values, or both at once.
/// </remarks>
public enum OptimizationMode
{
    /// <summary>
    /// Optimize by selecting the best subset of features.
    /// </summary>
    FeatureSelectionOnly,

    /// <summary>
    /// Optimize by adjusting model parameters.
    /// </summary>
    ParametersOnly,

    /// <summary>
    /// Optimize both feature selection and model parameters.
    /// </summary>
    Both
}