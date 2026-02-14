namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the functionality for non-linear regression models in the AiDotNet library.
/// </summary>
/// <remarks>
/// This interface extends IFullModel to provide specialized capabilities for non-linear regression,
/// which is used to model relationships that don't follow a straight line.
/// 
/// <b>For Beginners:</b> Non-linear regression helps you find patterns in data that follow curves instead of straight lines.
/// 
/// What is non-linear regression?
/// - Linear regression finds relationships that follow straight lines (like y = mx + b)
/// - Non-linear regression finds relationships that follow curves or more complex patterns
/// - These could be exponential curves, bell curves, sine waves, or other non-straight patterns
/// 
/// Real-world examples where non-linear patterns appear:
/// - Population growth (often follows exponential curves)
/// - Learning curves (progress is fast at first, then slows down)
/// - Seasonal sales data (follows cyclical patterns)
/// - Chemical reactions (may follow logarithmic or exponential patterns)
/// - Disease spread (often follows S-shaped logistic curves)
/// 
/// When to use non-linear regression:
/// - When plotting your data shows a clear curve rather than a straight line
/// - When you know from theory that the relationship should be non-linear
/// - When linear models give poor results or don't make sense for your problem
/// 
/// This interface provides methods to create, train, and use non-linear regression models
/// through the functionality inherited from IFullModel.
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
[AiDotNet.Configuration.YamlConfigurable("NonLinearRegression")]
public interface INonLinearRegression<T> : IRegression<T>
{
    // This interface inherits all methods and properties from IFullModel<T>
    // No additional methods or properties are defined at this level
}
