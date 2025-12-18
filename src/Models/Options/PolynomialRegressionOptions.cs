namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Polynomial Regression, an extension of linear regression that models
/// the relationship between variables using polynomial functions to capture non-linear relationships in data.
/// </summary>
/// <remarks>
/// <para>
/// Polynomial Regression transforms the original features by adding polynomial terms (squared, cubed, etc.)
/// to the regression equation. This allows the model to fit curved or more complex relationships that cannot
/// be adequately represented by a straight line. While more flexible than linear regression, polynomial models
/// with higher degrees can lead to overfitting if not properly regularized or validated. Polynomial regression
/// is widely used in various fields including economics, social sciences, biology, and engineering when relationships
/// between variables are suspected to be non-linear. The implementation typically involves creating new features
/// by raising the original features to various powers, then applying standard linear regression techniques.
/// </para>
/// <para><b>For Beginners:</b> Polynomial Regression is like an upgraded version of regular linear regression that can handle curved relationships.
/// 
/// Imagine you're trying to model this relationship:
/// - Regular (linear) regression can only draw straight lines
/// - But many real-world relationships follow curves, not straight lines
/// - Examples: plant growth over time, diminishing returns on investment, learning curves
/// 
/// What polynomial regression does:
/// - It adds "power terms" to your equation (squared, cubed, etc.)
/// - This lets your model create curves instead of just straight lines
/// - A degree 2 polynomial can make parabolas (U-shapes)
/// - A degree 3 polynomial can make S-curves
/// - Higher degrees can create more complex shapes
/// 
/// Think of it like drawing tools:
/// - Linear regression gives you only a ruler (straight lines)
/// - Polynomial regression gives you flexible curve tools
/// - The degree setting controls how flexible those curves can be
/// 
/// This class lets you configure how curved or complex your model can be by setting the polynomial degree.
/// </para>
/// </remarks>
public class PolynomialRegressionOptions<T> : RegressionOptions<T>
{
    /// <summary>
    /// Gets or sets the degree of the polynomial used in the regression model.
    /// </summary>
    /// <value>The polynomial degree, defaulting to 2.</value>
    /// <remarks>
    /// <para>
    /// This parameter determines the highest power to which the independent variables will be raised in the polynomial
    /// equation. A value of 1 corresponds to linear regression, 2 introduces quadratic terms, 3 adds cubic terms, and
    /// so on. The choice of degree significantly impacts model complexity and flexibility. Higher degree polynomials
    /// can capture more complex non-linear relationships but require more training data to avoid overfitting. The optimal
    /// degree depends on the underlying data generation process and is often determined through cross-validation or by
    /// examining validation metrics. Note that computational complexity increases with the degree, as does the risk of
    /// numerical instability with very high degrees.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how "bendy" your regression line can be.
    /// 
    /// The default value of 2 means:
    /// - Your model can create quadratic curves (parabolas or U-shapes)
    /// - It can model relationships that go up, then down (or vice versa)
    /// - It's more flexible than a straight line but still fairly constrained
    /// 
    /// Think of polynomial degree like this:
    /// - Degree 1: Straight line (can only go up or down consistently)
    /// - Degree 2: Parabola (can curve once, making U or upside-down U shapes)
    /// - Degree 3: Cubic curve (can have S-shapes with two changes in direction)
    /// - Degree 4+: Increasingly wiggly lines with more changes in direction
    /// 
    /// You might want a higher degree (like 3 or 4):
    /// - When your data clearly shows multiple changes in direction
    /// - For complex physical or financial processes with known non-linear behavior
    /// - When you have plenty of data points to support a more complex model
    /// 
    /// You might want a lower degree (just 1):
    /// - When the relationship appears linear in scatter plots
    /// - When you have limited data and want to avoid overfitting
    /// - When simpler models are preferred for interpretability or domain knowledge
    /// 
    /// Warning: Be careful with high degree values!
    /// - Too high a degree can cause "overfitting" where your model learns the noise in your data
    /// - This creates wild fluctuations that look good on training data but fail on new data
    /// - It's like memorizing the training examples instead of learning the underlying pattern
    /// - As a rule of thumb, be very cautious about using degrees above 5
    /// </para>
    /// </remarks>
    public int Degree { get; set; } = 2;
}
