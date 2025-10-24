namespace AiDotNet.Enums;

/// <summary>
/// Specifies the optimization mode for machine learning algorithms.
/// </summary>
/// <remarks>
/// <para>
/// This enum determines what aspects of the model the optimizer will focus on improving:
/// feature selection, parameter adjustment, or both.
/// </para>
/// <para><b>For Beginners:</b> Think of this as choosing what the optimizer should improve:
/// - FeatureSelectionOnly: Only choose which inputs to use
/// - ParametersOnly: Only adjust model weights
/// - Both: Optimize both features and weights together
/// </para>
/// </remarks>
public enum OptimizationMode
{
    /// <summary>
    /// Optimize only feature selection (which inputs/features to use).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The optimizer will only try different combinations of input features,
    /// without changing model weights. This is useful when you want to find the best subset of features.</para>
    /// </remarks>
    FeatureSelectionOnly,

    /// <summary>
    /// Optimize only model parameters (weights and coefficients).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The optimizer will only adjust model weights, using all available features.
    /// This is the traditional approach where you tune the model's internal parameters.</para>
    /// </remarks>
    ParametersOnly,

    /// <summary>
    /// Optimize both feature selection and model parameters simultaneously.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The optimizer will try different feature combinations AND adjust weights.
    /// This is the most flexible but can take longer since it's optimizing more things at once.</para>
    /// </remarks>
    Both
}
