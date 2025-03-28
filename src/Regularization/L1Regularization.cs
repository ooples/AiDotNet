namespace AiDotNet.Regularization;

/// <summary>
/// Implements L1 regularization (also known as Lasso), a technique that adds a penalty equal to the
/// absolute value of the magnitude of coefficients.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// L1 regularization adds a penalty term to the loss function equal to the sum of the absolute values
/// of the model coefficients, multiplied by a regularization strength parameter. This encourages sparse
/// models by driving some coefficients to exactly zero, effectively performing feature selection.
/// </para>
/// <para><b>For Beginners:</b> L1 regularization helps create simpler models by completely removing less important features.
/// 
/// Think of it like a strict budget committee:
/// - It forces the model to focus only on the most important features
/// - Less important features get their coefficients reduced to exactly zero
/// - This means some features are completely eliminated from the model
/// 
/// For example, if you're predicting house prices with 50 features:
/// - Without regularization, the model might use all 50 features
/// - With L1 regularization, it might eliminate 30 features and only use the 20 most important ones
/// - This makes the model simpler, faster, and often more accurate on new data
/// 
/// L1 regularization is particularly useful when you suspect many of your features aren't relevant
/// or when you want to identify which features matter most.
/// </para>
/// </remarks>
public class L1Regularization<T> : RegularizationBase<T>
{
    /// <summary>
    /// Initializes a new instance of the L1Regularization class with optional custom options.
    /// </summary>
    /// <param name="options">
    /// Configuration options for regularization including strength. If not provided,
    /// default values will be used (strength = 0.1).
    /// </param>
    /// <remarks>
    /// <para>
    /// This constructor creates an L1 regularization instance with the specified options or default values.
    /// For L1 regularization, the L1Ratio is always set to 1.0, as it represents pure L1 regularization.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up your L1 regularization with your chosen settings.
    /// 
    /// When creating L1 regularization:
    /// - The strength parameter determines how aggressively to eliminate less important features
    /// - Higher strength values lead to more features being completely eliminated (set to zero)
    /// - Lower strength values are more permissive, allowing more features to remain active
    /// 
    /// If you don't specify any options, it uses a default strength of 0.1, which provides
    /// moderate feature selection without being too aggressive.
    /// </para>
    /// </remarks>
    public L1Regularization(RegularizationOptions? options = null)
        : base(options ?? new RegularizationOptions
        {
            Type = RegularizationType.L1,
            Strength = 0.1, // Default L1 regularization strength
            L1Ratio = 1.0  // For L1, this should always be 1.0
        })
    {
    }

    /// <summary>
    /// Applies L1 regularization to a matrix.
    /// </summary>
    /// <param name="matrix">The input matrix to regularize.</param>
    /// <returns>The regularized matrix, unchanged in this implementation.</returns>
    /// <remarks>
    /// <para>
    /// For L1 regularization, this method typically returns the input matrix unchanged, as the regularization
    /// is applied directly to the coefficients rather than to the input data matrix. This is consistent with how
    /// L1 regularization is traditionally implemented in machine learning.
    /// </para>
    /// <para><b>For Beginners:</b> This method doesn't change the input data.
    /// 
    /// L1 regularization works by:
    /// - Leaving your original data (the matrix) completely unchanged
    /// - Applying its effects later during the coefficient calculation phase
    /// 
    /// Think of it like keeping your raw ingredients the same, but changing the recipe for
    /// how they're combined to create the final dish.
    /// </para>
    /// </remarks>
    public override Matrix<T> RegularizeMatrix(Matrix<T> matrix)
    {
        // L1 regularization typically doesn't modify the input matrix
        return matrix;
    }

    /// <summary>
    /// Applies L1 regularization to model coefficients.
    /// </summary>
    /// <param name="coefficients">The coefficient vector to regularize.</param>
    /// <returns>The regularized coefficient vector.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the core of L1 regularization by applying a soft thresholding operation to
    /// the coefficients. Coefficients with absolute values less than the regularization strength are set to zero,
    /// and those above the threshold are shrunk toward zero by the regularization strength amount.
    /// This operation is also known as "soft thresholding."
    /// </para>
    /// <para><b>For Beginners:</b> This method reduces or eliminates the impact of less important features.
    /// 
    /// When regularizing coefficients:
    /// - If a coefficient's absolute value is less than the regularization strength, it becomes zero
    /// - If it's larger, it gets reduced by the regularization strength, but remains non-zero
    /// - This is called "soft thresholding" - small effects are eliminated, large ones are just reduced
    /// 
    /// For example, with a regularization strength of 0.1:
    /// - A coefficient of 0.05 would become 0 (eliminated completely)
    /// - A coefficient of 0.3 would become 0.2 (reduced by 0.1)
    /// - A coefficient of -0.4 would become -0.3 (reduced by 0.1, keeping its sign)
    /// 
    /// This creates a model that focuses only on the strongest, most important patterns in your data.
    /// </para>
    /// </remarks>
    public override Vector<T> RegularizeCoefficients(Vector<T> coefficients)
    {
        var regularizationStrength = NumOps.FromDouble(Options.Strength);
        return coefficients.Transform(c =>
        {
            var sub = NumOps.Subtract(NumOps.Abs(c), regularizationStrength);
            return NumOps.Multiply(
                NumOps.SignOrZero(c),
                NumOps.GreaterThan(sub, NumOps.Zero) ? sub : NumOps.Zero
            );
        });
    }

    /// <summary>
    /// Adjusts the gradient vector to account for L1 regularization during optimization.
    /// </summary>
    /// <param name="gradient">The original gradient vector from the loss function.</param>
    /// <param name="coefficients">The current coefficient vector.</param>
    /// <returns>The regularized gradient vector.</returns>
    /// <remarks>
    /// <para>
    /// This method modifies the gradient vector during optimization to account for the L1 regularization
    /// penalty. It adds the subdifferential of the L1 norm (the sign of each coefficient multiplied by
    /// the regularization strength) to the original gradient, steering the optimization toward sparse solutions.
    /// </para>
    /// <para><b>For Beginners:</b> This method guides the model's learning toward simpler solutions.
    /// 
    /// During model training:
    /// - The model adjusts its coefficients based on the gradient (direction of steepest improvement)
    /// - This method modifies that gradient to include the effect of L1 regularization
    /// - It pushes coefficients toward zero, with more pressure on smaller coefficients
    /// 
    /// Think of it like adding a constant force pulling all coefficients toward zero:
    /// - For positive coefficients, it adds a positive value to the gradient (pushing them down)
    /// - For negative coefficients, it adds a negative value to the gradient (pushing them up)
    /// - For zero coefficients, it doesn't add anything (leaving them at zero)
    /// 
    /// This helps the model converge to a solution where many coefficients are exactly zero,
    /// effectively selecting only the most important features.
    /// </para>
    /// </remarks>
    public override Vector<T> RegularizeGradient(Vector<T> gradient, Vector<T> coefficients)
    {
        var regularizationStrength = NumOps.FromDouble(Options.Strength);
        return gradient.Add(coefficients.Transform(c => 
            NumOps.Multiply(regularizationStrength, NumOps.SignOrZero(c))
        ));
    }
}