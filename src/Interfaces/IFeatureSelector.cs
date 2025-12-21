namespace AiDotNet.Interfaces;

/// <summary>
/// Defines an interface for selecting the most relevant features from a dataset.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> Feature selection is like choosing which ingredients actually matter for your recipe.
/// 
/// In machine learning, "features" are the individual measurable properties of what you're observing.
/// For example, if you're predicting house prices, features might include:
/// - Square footage
/// - Number of bedrooms
/// - Location
/// - Year built
/// - etc.
/// 
/// However, not all features are equally useful. Some might:
/// - Be redundant (e.g., house area in square feet and square meters)
/// - Have little impact on what you're predicting
/// - Add "noise" that makes predictions less accurate
/// 
/// Feature selection helps identify which features are most important, which:
/// - Makes your model simpler and faster
/// - Often improves prediction accuracy
/// - Makes results easier to interpret
/// - Reduces the risk of "overfitting" (when a model learns noise instead of true patterns)
/// 
/// This interface provides a standard way to implement different feature selection techniques.
/// </remarks>
public interface IFeatureSelector<T, TInput>
{
    /// <summary>
    /// Selects the most relevant features from the provided feature matrix.
    /// </summary>
    /// <param name="allFeaturesMatrix">The complete matrix of features where each row represents a sample and each column represents a feature.</param>
    /// <returns>A matrix containing only the selected features, with the same number of rows but potentially fewer columns.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method analyzes your data and keeps only the columns (features) that matter most.
    /// 
    /// Imagine you have a spreadsheet with 100 columns of data about houses, but only 10 of those
    /// columns actually help predict the house price. This method would identify and keep just those
    /// 10 important columns, discarding the rest.
    /// 
    /// The input parameter:
    /// - allFeaturesMatrix: Your complete dataset organized as a matrix (like a table or spreadsheet)
    ///   - Each row is one example (like one house)
    ///   - Each column is one feature (like square footage, number of bedrooms, etc.)
    /// 
    /// The output:
    /// - A new matrix with the same number of rows (same examples)
    /// - But fewer columns (only the important features)
    /// 
    /// Different implementations of this interface might use different techniques to decide which
    /// features are important, such as:
    /// - Statistical tests
    /// - Correlation with the target variable
    /// - Machine learning algorithms that can rank feature importance
    /// 
    /// By removing less important features, your models can run faster and often make better predictions.
    /// </remarks>
    TInput SelectFeatures(TInput allFeaturesMatrix);
}
