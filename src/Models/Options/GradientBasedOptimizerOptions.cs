namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for gradient-based optimization algorithms.
/// </summary>
/// <remarks>
/// <para>
/// Gradient-based optimizers are algorithms that find the minimum or maximum of a function
/// by following the direction of steepest descent or ascent (the gradient).
/// </para>
/// <para><b>For Beginners:</b> Imagine you're in a hilly landscape and want to find the lowest point.
/// Gradient-based optimization is like always walking downhill in the steepest direction until you can't go any lower.
/// The "gradient" is simply the direction of the steepest slope at your current position.
/// </para>
/// <para>
/// These algorithms are fundamental to training many machine learning models, including neural networks,
/// linear regression, and logistic regression.
/// </para>
/// <para>
/// This class inherits from <see cref="OptimizationAlgorithmOptions"/>, which means it includes all the
/// base configuration options for optimization algorithms plus any additional options specific to
/// gradient-based methods.
/// </para>
/// </remarks>
public class GradientBasedOptimizerOptions : OptimizationAlgorithmOptions
{
    // Currently, this class doesn't add any additional properties beyond what's in the base class.
    // It serves as a base class for more specific gradient-based optimizers like SGD, Adam, etc.
}