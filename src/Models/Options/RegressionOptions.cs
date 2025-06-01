namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for regression models, which are statistical methods used to estimate 
/// relationships between variables and make predictions.
/// </summary>
/// <typeparam name="T">The data type used in matrix operations for the regression model.</typeparam>
/// <remarks>
/// <para>
/// Regression analysis is a fundamental statistical technique used to model the relationship between a 
/// dependent variable and one or more independent variables. This class provides configuration options 
/// for regression models, allowing customization of how the regression algorithm operates. Regression 
/// models are widely used in machine learning and statistics for prediction, forecasting, and understanding 
/// variable relationships. Common regression types include linear regression, polynomial regression, 
/// and regularized regression methods like ridge and lasso regression.
/// </para>
/// <para><b>For Beginners:</b> Regression is a way to find patterns in your data that help predict numbers.
/// 
/// Think about predicting house prices:
/// - You have information like square footage, number of bedrooms, and neighborhood
/// - Regression helps you create a formula that uses these features to predict the price
/// - The formula might look like: Price = (Square Footage × Factor1) + (Bedrooms × Factor2) + BaseValue
/// 
/// What regression does:
/// - It analyzes your existing data (houses with known prices)
/// - It finds the best values for each factor in the formula
/// - It creates a model that can predict prices for new houses
/// 
/// This is useful when:
/// - You need to predict a numerical value (like price, temperature, or sales)
/// - You have data with features that might influence that value
/// - You want to understand how much each feature contributes to the prediction
/// 
/// For example, a weather app might use regression to predict tomorrow's temperature based on 
/// today's readings, humidity levels, and seasonal patterns.
///
/// This class lets you configure how the regression model is constructed and calculated.
/// </para>
/// </remarks>
public class RegressionOptions<T> : ModelOptions
{
    /// <summary>
    /// Gets or sets the matrix decomposition method used in the regression calculations.
    /// </summary>
    /// <value>An implementation of IMatrixDecomposition&lt;T&gt; or null to use the default method.</value>
    /// <remarks>
    /// <para>
    /// Matrix<double> decomposition is a crucial mathematical technique used in solving regression problems, 
    /// especially when dealing with systems of linear equations. Different decomposition methods have 
    /// varying characteristics in terms of numerical stability, computational efficiency, and applicability 
    /// to specific problem types. Common decomposition methods include Singular Value Decomposition (SVD), 
    /// QR decomposition, Cholesky decomposition, and LU decomposition. The choice of decomposition method 
    /// can significantly impact the accuracy and performance of the regression algorithm, particularly for 
    /// ill-conditioned problems or large datasets.
    /// </para>
    /// <para><b>For Beginners:</b> Matrix<double> decomposition is a mathematical technique that helps solve complex 
    /// equations more efficiently.
    /// 
    /// Think of it like breaking down a difficult math problem into simpler steps:
    /// - Regression often involves solving systems of equations with many variables
    /// - These equations can be represented as matrices (tables of numbers)
    /// - Matrix<double> decomposition breaks these complex matrices into simpler components
    /// - This makes calculations faster, more accurate, and more stable
    /// 
    /// Different decomposition methods have different strengths:
    /// - Some are faster but less accurate for certain problems
    /// - Some handle special cases better (like when variables are highly correlated)
    /// - Some are more numerically stable (less likely to have rounding errors)
    /// 
    /// You can leave this setting as null (the default) to let the algorithm choose an appropriate 
    /// method, or specify a particular decomposition method if you have specific requirements or 
    /// knowledge about your data's characteristics.
    /// 
    /// Unless you have a specific reason to change this, it's usually best to leave it as the default.
    /// </para>
    /// </remarks>
    public IMatrixDecomposition<T>? DecompositionMethod { get; set; }
    
    /// <summary>
    /// Gets or sets whether the regression model should include an intercept term (also known as bias term).
    /// </summary>
    /// <value>True to include an intercept term; false to force the regression line through the origin. Default is true.</value>
    /// <remarks>
    /// <para>
    /// The intercept term (also called the bias or constant term) represents the expected value of the 
    /// dependent variable when all independent variables are zero. Including an intercept allows the 
    /// regression line to "float" and not be constrained to pass through the origin. This is appropriate 
    /// for most real-world problems where a baseline value exists even when predictor variables are zero. 
    /// In some specific scientific or engineering contexts where a zero input must theoretically produce 
    /// a zero output, setting UseIntercept to false may be more appropriate. The decision to include or 
    /// exclude an intercept should be based on domain knowledge and the specific requirements of the 
    /// modeling task.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines whether your prediction formula includes a 
    /// "starting value" or base amount.
    /// 
    /// Using our house price example:
    /// - With UseIntercept = true (default): Price = (Square Footage × Factor1) + (Bedrooms × Factor2) + BaseValue
    /// - With UseIntercept = false: Price = (Square Footage × Factor1) + (Bedrooms × Factor2)
    /// 
    /// The difference is that "BaseValue" (the intercept):
    /// - Represents the predicted price when all other factors are zero
    /// - Allows the model to have a starting point that isn't zero
    /// - Makes the model more flexible for most real-world problems
    /// 
    /// When to use true (include intercept):
    /// - Most real-world scenarios (the default and recommended for beginners)
    /// - When you expect there to be some base value even when all features are zero
    /// - For example, even a 0 square foot house with 0 bedrooms might still have some land value
    /// 
    /// When to use false (no intercept):
    /// - When you know theoretically that zero input should produce zero output
    /// - In scientific models where this constraint is physically meaningful
    /// - For example, if measuring how far a spring stretches based on weight, zero weight means zero stretch
    /// 
    /// For most applications, keep this set to true unless you have a specific reason to change it.
    /// </para>
    /// </remarks>
    public bool UseIntercept { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to allow fallback to alternative decomposition methods when the primary method fails.
    /// </summary>
    /// <value>True to allow fallbacks; otherwise, false. Defaults to true.</value>
    /// <remarks>
    /// <para>
    /// When set to true, if the specified decomposition method fails, the system will automatically try
    /// alternative methods in order of robustness. When set to false, only the specified method (or the default
    /// Normal decomposition if none is specified) will be attempted.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls whether the system can try different mathematical 
    /// approaches if the first one fails.
    /// 
    /// - True (default): The system will try multiple methods to solve your problem, starting with faster
    ///   methods and moving to more robust ones as needed. This is like having multiple tools available
    ///   and using whichever one works.
    ///   
    /// - False: The system will only use the exact method you specified. If that method fails, the system
    ///   will report an error rather than trying alternative approaches.
    ///   
    /// Set to False when you specifically need a particular decomposition method for consistency or
    /// mathematical reasons. Otherwise, leaving this as True gives the best chance of successfully
    /// solving your problem.
    /// </para>
    /// </remarks>
    public bool AllowDecompositionFallbacks { get; set; } = true;
}