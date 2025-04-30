namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the common interface for all classification algorithms in the AiDotNet library.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> Classification is a type of machine learning algorithm used to predict categorical values.
/// 
/// For example, classification can be used to:
/// - Predict whether an email is spam or not spam
/// - Categorize images into different classes (e.g., cat, dog, bird)
/// - Determine if a loan application should be approved or rejected
/// 
/// This interface inherits from IFullModel, which means all classification models share common
/// functionality for training on data and making predictions.
/// 
/// Unlike regression algorithms (which predict continuous numeric values), classification algorithms predict
/// discrete categories or classes.
/// </remarks>
public interface IClassificationModel<T> : IFullModel<T, Matrix<T>, Vector<T>>
{
}