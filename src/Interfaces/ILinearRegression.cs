namespace AiDotNet.Interfaces;

/// <summary>
/// Defines an interface for linear regression in machine learning, which predict outputs as a weighted sum of inputs plus an optional constant.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> This interface represents the simplest and most fundamental type of machine learning model.
/// 
/// Imagine you're trying to predict house prices based on features like:
/// - Square footage
/// - Number of bedrooms
/// - Age of the house
/// 
/// A linear model works like this:
/// - Each feature gets assigned a "weight" (coefficient) that represents its importance
/// - For example: Each square foot might add $100 to the price
/// - Each bedroom might add $15,000 to the price
/// - Each year of age might subtract $500 from the price
/// - There might also be a "starting price" (intercept) of $50,000
/// 
/// To make a prediction, the model:
/// 1. Multiplies each feature by its weight
/// 2. Adds all these values together
/// 3. Adds the intercept (if there is one)
/// 
/// So for a 2,000 sq ft, 3-bedroom, 10-year-old house:
/// Price = $50,000 + (2,000 × $100) + (3 × $15,000) + (10 × -$500)
/// Price = $50,000 + $200,000 + $45,000 - $5,000
/// Price = $290,000
/// 
/// Linear models are popular because they're:
/// - Simple to understand
/// - Fast to train
/// - Easy to interpret (you can see exactly how each feature affects the prediction)
/// - Often surprisingly effective despite their simplicity
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("LinearRegression")]
public interface ILinearRegression<T> : IRegression<T>
{
    /// <summary>
    /// Gets the weights (coefficients) assigned to each input feature in the linear model.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> These are the numbers that determine how important each input feature is to your prediction.
    /// 
    /// In our house price example:
    /// - The coefficient for square footage might be 100 (each square foot adds $100)
    /// - The coefficient for bedrooms might be 15000 (each bedroom adds $15,000)
    /// - The coefficient for house age might be -500 (each year subtracts $500)
    /// 
    /// Positive coefficients mean that as the feature increases, the prediction increases.
    /// Negative coefficients mean that as the feature increases, the prediction decreases.
    /// 
    /// The size of the coefficient (whether it's a large or small number) tells you how strongly
    /// that feature influences the prediction.
    /// 
    /// This property returns all coefficients as a vector (essentially a list of numbers),
    /// where each position corresponds to a specific input feature.
    /// </remarks>
    Vector<T> Coefficients { get; }

    /// <summary>
    /// Gets the constant term (bias) added to the weighted sum of features in the linear model.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is the "starting value" or "base value" for your prediction before 
    /// considering any of the input features.
    /// 
    /// In our house price example:
    /// - The intercept might be $50,000, meaning that's the base price before we consider
    ///   square footage, bedrooms, or age
    /// 
    /// The intercept allows the model to make reasonable predictions even when some feature values are zero.
    /// For example, a house with zero square footage doesn't make sense, but the model still needs
    /// to produce reasonable values.
    /// 
    /// Some models don't use an intercept (when HasIntercept is false), which forces the prediction
    /// to be zero when all features are zero.
    /// </remarks>
    T Intercept { get; }

    /// <summary>
    /// Gets a value indicating whether this linear model includes an intercept term.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This tells you whether the model includes a "starting value" (intercept) or not.
    /// 
    /// When this is true:
    /// - The model includes an intercept term (the base value)
    /// - Predictions are calculated as: Intercept + (Feature1 × Weight1) + (Feature2 × Weight2) + ...
    /// 
    /// When this is false:
    /// - The model does not include an intercept term
    /// - Predictions are calculated as: (Feature1 × Weight1) + (Feature2 × Weight2) + ...
    /// - When all features are zero, the prediction will be exactly zero
    /// 
    /// Most linear models include an intercept (HasIntercept = true) because it gives the model
    /// more flexibility to fit the data. However, in some special cases, you might want to force
    /// the model to pass through the origin (0,0), which requires HasIntercept = false.
    /// </remarks>
    bool HasIntercept { get; }
}
