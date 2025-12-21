namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the common interface for all regression algorithms in the AiDotNet library.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> Regression is a type of machine learning algorithm used to predict continuous values.
/// 
/// For example, regression can be used to:
/// - Predict house prices based on features like size, location, and age
/// - Forecast sales numbers based on historical data
/// - Estimate a person's income based on education, experience, and other factors
/// 
/// This interface inherits from ILinearModel, which means all regression models share common
/// functionality for training on data and making predictions.
/// 
/// Unlike classification algorithms (which predict categories), regression algorithms predict
/// numeric values on a continuous scale.
/// </remarks>
public interface IRegression<T> : IFullModel<T, Matrix<T>, Vector<T>>
{
}
