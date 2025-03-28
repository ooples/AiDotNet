namespace AiDotNet.Interfaces;

/// <summary>
/// Defines an interface for data preprocessing operations commonly used in machine learning workflows.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> Data preprocessing is like preparing ingredients before cooking.
/// 
/// Raw data often isn't ready for machine learning algorithms. It might have missing values,
/// different scales (like mixing inches and centimeters), or other issues. Data preprocessing
/// cleans and transforms this raw data into a format that machine learning models can use effectively.
/// 
/// Common preprocessing steps include:
/// - Normalization: Adjusting values to a common scale (like 0-1)
/// - Handling missing values: Filling in or removing data gaps
/// - Feature selection: Choosing which data columns are most useful
/// - Data splitting: Dividing data into training, validation, and test sets
/// 
/// This interface provides standard methods to perform these essential preprocessing tasks.
/// </remarks>
public interface IDataPreprocessor<T>
{
    /// <summary>
    /// Preprocesses the input data by applying normalization and other transformations.
    /// </summary>
    /// <param name="X">The matrix of input features where each row represents a sample and each column represents a feature.</param>
    /// <param name="y">The vector of target values corresponding to each sample in the input matrix.</param>
    /// <returns>
    /// A tuple containing:
    /// - The preprocessed feature matrix
    /// - The preprocessed target vector
    /// - Normalization information that can be used to transform new data consistently
    /// </returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method cleans and transforms your raw data to make it suitable for machine learning.
    /// 
    /// Parameters explained:
    /// - X: Your input data organized as a matrix (think of it as a table or spreadsheet)
    ///   - Each row is one example or data point
    ///   - Each column is one feature or characteristic
    /// - y: The target values you want to predict (like prices, categories, etc.)
    /// 
    /// The method returns three things:
    /// 1. Your transformed input data (X)
    /// 2. Your transformed target values (y)
    /// 3. Information about how the transformation was done (normInfo)
    /// 
    /// The third item (normInfo) is important because when you get new data later,
    /// you need to transform it in exactly the same way as your training data.
    /// 
    /// For example, if you're predicting house prices:
    /// - If during training you divided all prices by $1,000,000 to normalize them
    /// - Then for new predictions, you need to apply the same division
    /// - The normInfo stores these details so you can apply consistent transformations
    /// </remarks>
    (Matrix<T> X, Vector<T> y, NormalizationInfo<T> normInfo) PreprocessData(Matrix<T> X, Vector<T> y);
    
    /// <summary>
    /// Splits the dataset into training, validation, and test sets.
    /// </summary>
    /// <param name="X">The matrix of input features.</param>
    /// <param name="y">The vector of target values.</param>
    /// <returns>
    /// A tuple containing six elements:
    /// - XTrain: Feature matrix for training
    /// - yTrain: Target vector for training
    /// - XValidation: Feature matrix for validation
    /// - yValidation: Target vector for validation
    /// - XTest: Feature matrix for testing
    /// - yTest: Target vector for testing
    /// </returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method divides your data into three separate sets, each with a specific purpose.
    /// 
    /// Imagine you're learning to cook a new recipe:
    /// - Training set: This is where you practice and learn the recipe (70-80% of your data)
    /// - Validation set: This is where you taste and adjust your cooking (10-15% of your data)
    /// - Test set: This is the final taste test with fresh ingredients (10-15% of your data)
    /// 
    /// Why split the data?
    /// - Training: The model learns patterns from this data
    /// - Validation: You use this to tune your model settings without overfitting
    /// - Testing: You use this to get an honest estimate of how well your model will perform on new data
    /// 
    /// "Overfitting" is like memorizing test answers instead of understanding the subject.
    /// The model performs well on data it has seen but fails on new data.
    /// 
    /// Each set contains both features (X) and targets (y), keeping the relationship between
    /// input data and expected outputs intact for each portion of the data.
    /// </remarks>
    (Matrix<T> XTrain, Vector<T> yTrain, Matrix<T> XValidation, Vector<T> yValidation, Matrix<T> XTest, Vector<T> yTest) 
        SplitData(Matrix<T> X, Vector<T> y);
}