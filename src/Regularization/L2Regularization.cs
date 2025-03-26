namespace AiDotNet.Regularization;

/// <summary>
/// Implements L2 regularization (also known as Ridge), a technique that adds a penalty equal to the
/// square of the magnitude of coefficients.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// L2 regularization adds a penalty term to the loss function equal to the sum of the squared values
/// of the model coefficients, multiplied by a regularization strength parameter. This encourages smaller,
/// more evenly distributed coefficient values, which helps prevent overfitting and improves model stability.
/// </para>
/// <para><b>For Beginners:</b> L2 regularization helps create smoother models by making all coefficients smaller.
/// 
/// Think of it like a gentle pull that shrinks all coefficients proportionally:
/// - It doesn't eliminate features entirely (unlike L1 regularization)
/// - It reduces the impact of all features by making their coefficients smaller
/// - It particularly penalizes large coefficient values
/// 
/// For example, if you're predicting house prices:
/// - Without regularization, some features might get very large coefficients
/// - With L2 regularization, all coefficients become smaller, but stay non-zero
/// - This prevents any single feature from dominating the prediction
/// 
/// L2 regularization is particularly useful when you have many correlated features
/// or when you want to prevent the model from becoming too sensitive to any single feature.
/// </para>
/// </remarks>
public class L2Regularization<T> : RegularizationBase<T>
{
    /// <summary>
    /// Initializes a new instance of the L2Regularization class with optional custom options.
    /// </summary>
    /// <param name="options">
    /// Configuration options for regularization including strength. If not provided,
    /// default values will be used (strength = 0.01).
    /// </param>
    /// <remarks>
    /// <para>
    /// This constructor creates an L2 regularization instance with the specified options or default values.
    /// For L2 regularization, the L1Ratio is always set to 0.0, as it represents pure L2 regularization.
    /// Note that the default strength for L2 regularization (0.01) is typically smaller than for L1
    /// because it's applied to squared values.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up your L2 regularization with your chosen settings.
    /// 
    /// When creating L2 regularization:
    /// - The strength parameter determines how aggressively to shrink coefficients
    /// - Higher strength values lead to smaller coefficients and a simpler model
    /// - Lower strength values allow coefficients to remain larger
    /// 
    /// If you don't specify any options, it uses a default strength of 0.01, which provides
    /// moderate regularization without being too aggressive. This default is smaller than
    /// L1 regularization (which often uses 0.1) because L2 squares the values, making the
    /// penalties larger for the same strength value.
    /// </para>
    /// </remarks>
    public L2Regularization(RegularizationOptions? options = null) : base(options ?? new RegularizationOptions
        {
            Type = RegularizationType.L2,
            Strength = 0.01, // Default L2 regularization strength
            L1Ratio = 0.0  // For L2, this should always be 0.0
        })
    {
    }

    /// <summary>
    /// Applies L2 regularization to a matrix.
    /// </summary>
    /// <param name="matrix">The input matrix to regularize.</param>
    /// <returns>The regularized matrix, unchanged in this implementation.</returns>
    /// <remarks>
    /// <para>
    /// For L2 regularization, this method typically returns the input matrix unchanged, as the regularization
    /// is applied directly to the coefficients rather than to the input data matrix. This is consistent with how
    /// L2 regularization is traditionally implemented in machine learning.
    /// </para>
    /// <para><b>For Beginners:</b> This method doesn't change the input data.
    /// 
    /// L2 regularization works by:
    /// - Leaving your original data (the matrix) completely unchanged
    /// - Applying its effects later during the coefficient calculation phase
    /// 
    /// Think of it like keeping your raw measurements the same, but changing how you
    /// weight and interpret them when building your model.
    /// </para>
    /// </remarks>
    public override Matrix<T> RegularizeMatrix(Matrix<T> matrix)
    {
        // L2 regularization typically doesn't modify the input matrix
        return matrix;
    }

    /// <summary>
    /// Applies L2 regularization to model coefficients.
    /// </summary>
    /// <param name="coefficients">The coefficient vector to regularize.</param>
    /// <returns>The regularized coefficient vector.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the core of L2 regularization by uniformly shrinking all coefficients
    /// toward zero. Each coefficient is multiplied by a factor slightly less than 1, determined by the 
    /// regularization strength. This causes all coefficients to become smaller but none to become exactly zero
    /// (unlike L1 regularization).
    /// </para>
    /// <para><b>For Beginners:</b> This method reduces the size of all coefficients proportionally.
    /// 
    /// When regularizing coefficients:
    /// - All coefficients are multiplied by a value slightly less than 1
    /// - Higher regularization strength means a smaller multiplier
    /// - This shrinks all coefficients toward zero, but none become exactly zero
    /// 
    /// For example, with a regularization strength of 0.01:
    /// - All coefficients are multiplied by 0.99
    /// - A coefficient of 2.0 would become 1.98
    /// - A coefficient of -0.5 would become -0.495
    /// 
    /// This creates a model that gives more balanced importance to all features,
    /// preventing any single feature from having too much influence.
    /// </para>
    /// </remarks>
    public override Vector<T> RegularizeCoefficients(Vector<T> coefficients)
    {
        var regularizationStrength = NumOps.FromDouble(Options.Strength);
        return coefficients.Multiply(NumOps.Subtract(NumOps.One, regularizationStrength));
    }

    /// <summary>
    /// Adjusts the gradient vector to account for L2 regularization during optimization.
    /// </summary>
    /// <param name="gradient">The original gradient vector from the loss function.</param>
    /// <param name="coefficients">The current coefficient vector.</param>
    /// <returns>The regularized gradient vector.</returns>
    /// <remarks>
    /// <para>
    /// This method modifies the gradient vector during optimization to account for the L2 regularization
    /// penalty. It adds the derivative of the L2 penalty term (the coefficients multiplied by the regularization
    /// strength) to the original gradient, steering the optimization toward solutions with smaller coefficient values.
    /// </para>
    /// <para><b>For Beginners:</b> This method guides the model's learning toward simpler solutions.
    /// 
    /// During model training:
    /// - The model adjusts its coefficients based on the gradient (direction of steepest improvement)
    /// - This method modifies that gradient to include the effect of L2 regularization
    /// - It pushes all coefficients toward zero, with more pressure on larger coefficients
    /// 
    /// Think of it like adding a force that pulls each coefficient toward zero with a strength
    /// proportional to the coefficient's current value:
    /// - Larger coefficients feel a stronger pull toward zero
    /// - Smaller coefficients feel a weaker pull
    /// - This creates a natural balancing effect among all features
    /// 
    /// This helps the model converge to a solution where all coefficients are appropriately sized,
    /// improving its ability to generalize to new data.
    /// </para>
    /// </remarks>
    public override Vector<T> RegularizeGradient(Vector<T> gradient, Vector<T> coefficients)
    {
        var regularizationStrength = NumOps.FromDouble(Options.Strength);
        return gradient.Add(coefficients.Multiply(regularizationStrength));
    }
}