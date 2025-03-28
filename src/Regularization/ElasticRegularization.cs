namespace AiDotNet.Regularization;

/// <summary>
/// Implements Elastic Net regularization, a hybrid approach that combines L1 (Lasso) and L2 (Ridge) regularization techniques.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Elastic Net regularization provides the benefits of both L1 and L2 regularization methods. It helps prevent overfitting
/// by penalizing large coefficient values, while also encouraging sparsity (more coefficients set to zero) when appropriate.
/// The L1Ratio parameter controls the balance between L1 and L2 regularization.
/// </para>
/// <para><b>For Beginners:</b> Elastic Net is like having two different tools to prevent your model from becoming too complex.
/// 
/// Think of it like this:
/// - L1 (Lasso) regularization tends to completely eliminate less important features (setting them to zero)
/// - L2 (Ridge) regularization keeps all features but makes them smaller overall
/// - Elastic Net lets you blend these two approaches for the best of both worlds
/// 
/// For example:
/// - With L1 ratio = 1.0: You get pure L1 regularization (feature elimination)
/// - With L1 ratio = 0.0: You get pure L2 regularization (feature shrinkage)
/// - With L1 ratio = 0.5: You get a 50/50 blend of both techniques
/// 
/// This helps your model find a good balance between simplicity and accuracy.
/// </para>
/// </remarks>
public class ElasticNetRegularization<T> : RegularizationBase<T>
{
    /// <summary>
    /// Initializes a new instance of the ElasticNetRegularization class with optional custom options.
    /// </summary>
    /// <param name="options">
    /// Configuration options for regularization including strength and L1 ratio. If not provided,
    /// default values will be used (strength = 0.1, L1 ratio = 0.5).
    /// </param>
    /// <remarks>
    /// <para>
    /// This constructor creates an Elastic Net regularization instance with the specified options or default values.
    /// The L1 ratio determines the balance between L1 and L2 regularization, with values ranging from 0 (pure L2) to 1 (pure L1).
    /// </para>
    /// <para><b>For Beginners:</b> This sets up your Elastic Net regularization with your chosen settings.
    /// 
    /// When creating Elastic Net regularization:
    /// - The strength parameter determines how strongly to penalize large coefficients (higher = stronger penalty)
    /// - The L1 ratio parameter (between 0 and 1) controls the balance between the two types of regularization
    /// 
    /// If you don't specify any options, it uses sensible defaults:
    /// - Strength of 0.1 (moderate regularization)
    /// - L1 ratio of 0.5 (equal blend of L1 and L2)
    /// 
    /// These defaults work well for many problems, but you might want to adjust them based on your specific needs.
    /// </para>
    /// </remarks>
    public ElasticNetRegularization(RegularizationOptions? options = null) : base(options ?? new RegularizationOptions
        {
            Type = RegularizationType.ElasticNet,
            Strength = 0.1, // Default Elastic Net regularization strength
            L1Ratio = 0.5  // Default balance between L1 and L2
        })
    {
    }

    /// <summary>
    /// Applies Elastic Net regularization to a matrix.
    /// </summary>
    /// <param name="matrix">The input matrix to regularize.</param>
    /// <returns>The regularized matrix, unchanged in this implementation.</returns>
    /// <remarks>
    /// <para>
    /// For Elastic Net regularization, this method typically returns the input matrix unchanged, as the regularization
    /// is applied directly to the coefficients rather than to the input data matrix. This is consistent with how
    /// Elastic Net regularization is traditionally implemented in machine learning.
    /// </para>
    /// <para><b>For Beginners:</b> This method doesn't change the input data.
    /// 
    /// Unlike some other techniques that might transform your raw data:
    /// - Elastic Net regularization applies its effects during the coefficient calculation phase
    /// - It doesn't need to modify your input matrix
    /// - It returns the same matrix that was provided as input
    /// 
    /// Think of this as "let the data stay as it is, we'll handle the regularization elsewhere."
    /// </para>
    /// </remarks>
    public override Matrix<T> RegularizeMatrix(Matrix<T> matrix)
    {
        // Elastic Net regularization typically doesn't modify the input matrix
        return matrix;
    }

    /// <summary>
    /// Applies Elastic Net regularization to model coefficients.
    /// </summary>
    /// <param name="coefficients">The coefficient vector to regularize.</param>
    /// <returns>The regularized coefficient vector.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the core of Elastic Net regularization by applying a combination of L1 and L2 penalties
    /// to the model coefficients. The L1 component promotes sparsity (more zeros in the coefficients), while the L2
    /// component promotes smaller coefficient values overall. The L1Ratio parameter controls the balance between these
    /// two effects.
    /// </para>
    /// <para><b>For Beginners:</b> This method adjusts the model's learned patterns to prevent overfitting.
    /// 
    /// When regularizing coefficients:
    /// - The L1 part tends to eliminate less important features entirely (set to zero)
    /// - The L2 part reduces the size of all coefficients proportionally
    /// - The L1 ratio controls how much of each effect is applied
    /// 
    /// This helps create a model that:
    /// - Focuses on the most important patterns in your data
    /// - Ignores noise and random fluctuations
    /// - Generalizes better to new, unseen data
    /// 
    /// It's like carefully trimming a tree - removing some branches entirely while just shortening others.
    /// </para>
    /// </remarks>
    public override Vector<T> RegularizeCoefficients(Vector<T> coefficients)
    {
        var regularizationStrength = NumOps.FromDouble(Options.Strength);
        var l1Ratio = NumOps.FromDouble(Options.L1Ratio);
        var l2Ratio = NumOps.FromDouble(1 - Options.L1Ratio);
        return coefficients.Transform(c =>
        {
            var subPart = NumOps.Subtract(NumOps.Abs(c), NumOps.Multiply(regularizationStrength, l1Ratio));
            var l1Part = NumOps.Multiply(
                NumOps.SignOrZero(c),
                NumOps.GreaterThan(subPart, NumOps.Zero) ? subPart : NumOps.Zero
            );
            var l2Part = NumOps.Multiply(
                c,
                NumOps.Subtract(NumOps.One, NumOps.Multiply(regularizationStrength, l2Ratio))
            );

            return NumOps.Add(l1Part, l2Part);
        });
    }

    /// <summary>
    /// Adjusts the gradient vector to account for Elastic Net regularization during optimization.
    /// </summary>
    /// <param name="gradient">The original gradient vector from the loss function.</param>
    /// <param name="coefficients">The current coefficient vector.</param>
    /// <returns>The regularized gradient vector.</returns>
    /// <remarks>
    /// <para>
    /// This method modifies the gradient vector during optimization to account for the Elastic Net regularization
    /// penalty. It adds the derivative of the Elastic Net penalty term with respect to the coefficients to the
    /// original gradient, steering the optimization toward solutions with appropriate regularization.
    /// </para>
    /// <para><b>For Beginners:</b> This method helps guide the model's learning process.
    /// 
    /// During model training:
    /// - The gradient tells the model which direction to adjust coefficients to improve
    /// - This method modifies that gradient to include the regularization effects
    /// - It pushes the model toward simpler solutions with smaller or zero coefficients
    /// 
    /// Think of it like adding another voice to the conversation when deciding how to update the model:
    /// - The original gradient says "change to better fit the data"
    /// - The regularization part says "but also stay simple and focused"
    /// - The combined gradient balances these two goals
    /// </para>
    /// </remarks>
    public override Vector<T> RegularizeGradient(Vector<T> gradient, Vector<T> coefficients)
    {
        var regularizationStrength = NumOps.FromDouble(Options.Strength);
        var l1Ratio = NumOps.FromDouble(Options.L1Ratio);
        var l2Ratio = NumOps.FromDouble(1 - Options.L1Ratio);

        return gradient.Add(coefficients.Transform(c =>
        {
            var l1Part = NumOps.Multiply(regularizationStrength, NumOps.Multiply(l1Ratio, NumOps.SignOrZero(c)));
            var l2Part = NumOps.Multiply(regularizationStrength, NumOps.Multiply(l2Ratio, c));
            return NumOps.Add(l1Part, l2Part);
        }));
    }
}