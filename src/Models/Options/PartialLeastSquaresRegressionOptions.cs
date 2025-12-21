namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Partial Least Squares Regression (PLS), a technique that combines
/// features of principal component analysis and multiple regression to handle multicollinearity
/// and high-dimensional data.
/// </summary>
/// <remarks>
/// <para>
/// Partial Least Squares Regression is a powerful statistical method that finds a linear regression
/// model by projecting both the predictor variables (X) and the response variables (Y) to a new space.
/// Unlike ordinary least squares regression, which can struggle with multicollinearity (highly correlated
/// predictors) and high-dimensional data, PLS creates latent variables (components) that maximize the
/// covariance between X and Y. This approach is particularly valuable in situations where the number of
/// predictor variables exceeds the number of observations, or when predictors are highly correlated.
/// PLS has found wide application in fields such as chemometrics, spectroscopy, and bioinformatics.
/// </para>
/// <para><b>For Beginners:</b> Partial Least Squares Regression is a special technique for finding relationships in complex data.
/// 
/// Imagine you're trying to predict house prices based on 50 different measurements:
/// - Many of these measurements might be related (like square footage and number of rooms)
/// - With so many variables, traditional regression methods might struggle
/// 
/// What PLS does:
/// - Instead of using all 50 original measurements directly
/// - It creates a smaller set of "super measurements" (called components)
/// - Each component combines multiple original measurements in a smart way
/// - These components capture the most important patterns relevant to your prediction
/// 
/// Think of it like cooking:
/// - Traditional regression uses each ingredient separately
/// - PLS creates a few key flavor profiles (combining multiple ingredients)
/// - Then uses these flavor profiles to create the final dish
/// 
/// This approach is especially useful when:
/// - You have more variables than data points
/// - Your variables are highly correlated with each other
/// - You need to reduce noise and focus on the most important patterns
/// 
/// This class lets you configure how many of these "super measurements" (components)
/// to use in your analysis.
/// </para>
/// </remarks>
public class PartialLeastSquaresRegressionOptions<T> : RegressionOptions<T>
{
    /// <summary>
    /// Gets or sets the number of latent components to extract in the PLS model.
    /// </summary>
    /// <value>The number of components, defaulting to 2.</value>
    /// <remarks>
    /// <para>
    /// This parameter determines how many latent components (also called factors or latent variables)
    /// the PLS algorithm will extract. Each component is a linear combination of the original predictor
    /// variables, constructed to maximize the covariance with the response variable(s) while maintaining
    /// orthogonality with previous components. The optimal number of components depends on the complexity
    /// of the relationship between predictors and response, as well as the amount of noise in the data.
    /// Too few components may result in underfitting, while too many may lead to overfitting. Cross-validation
    /// is often used to determine the optimal number.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how many "super measurements" (components)
    /// the model will create from your original variables.
    /// 
    /// The default value of 2 means:
    /// - The algorithm will create 2 components from your original variables
    /// - These 2 components will be used to make predictions
    /// 
    /// Think of it like data compression:
    /// - Each component captures a certain amount of the important information
    /// - The first component captures the most important patterns
    /// - Each additional component captures less and less additional information
    /// 
    /// You might want more components (like 5 or 10) if:
    /// - Your data has complex relationships that can't be captured by just a few components
    /// - You have many original variables with distinct information
    /// - You have sufficient data to support a more complex model
    /// 
    /// You might want fewer components (even just 1) if:
    /// - The relationship is relatively simple
    /// - You want to prevent overfitting, especially with limited data
    /// - Your variables are so highly correlated that one component is sufficient
    /// 
    /// Finding the right number of components is crucial:
    /// - Too few: The model misses important patterns (underfitting)
    /// - Too many: The model learns noise in the data (overfitting)
    /// 
    /// In practice, techniques like cross-validation are often used to find the optimal
    /// number of components that gives the best predictive performance.
    /// </para>
    /// </remarks>
    public int NumComponents { get; set; } = 2;
}
