namespace AiDotNet.Interfaces;

/// <summary>
/// Defines an interface for asynchronous tree-based machine learning models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> This interface extends the regular tree-based model interface to add
/// asynchronous (async) capabilities.
/// 
/// Tree-based models are machine learning algorithms that make decisions using a tree-like
/// structure of questions - similar to a flowchart. Popular examples include Decision Trees,
/// Random Forests, and Gradient Boosting Trees.
/// 
/// "Asynchronous" (or "async") means the model can run in the background without blocking
/// other operations. This is especially useful for:
/// - Training large models that take a long time
/// - Working with web applications where you don't want to freeze the user interface
/// - Processing large datasets efficiently
/// 
/// Think of it like ordering food at a restaurant - instead of standing at the counter
/// waiting for your order (synchronous), you get a buzzer and can do other things until
/// your food is ready (asynchronous).
/// 
/// This interface inherits all the regular methods from ITreeBasedModel but adds async
/// versions of the training and prediction methods.
/// </remarks>
public interface IAsyncTreeBasedModel<T> : ITreeBasedRegression<T>
{
    /// <summary>
    /// Trains the tree-based model asynchronously using the provided input features and target values.
    /// </summary>
    /// <param name="x">The matrix of input features where each row represents a sample and each column represents a feature.</param>
    /// <param name="y">The vector of target values corresponding to each sample in the input matrix.</param>
    /// <returns>A Task representing the asynchronous training operation.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method teaches the model to make predictions based on your data,
    /// but does it asynchronously (in the background).
    /// 
    /// Parameters explained:
    /// - x: Your input data organized as a matrix (think of it as a table or spreadsheet)
    ///   - Each row is one example or data point
    ///   - Each column is one feature or characteristic
    /// - y: The correct answers or target values you want the model to learn to predict
    ///   - One target value for each row in your input matrix
    /// 
    /// For example, if you're predicting house prices:
    /// - x would contain features like square footage, number of bedrooms, location, etc.
    /// - y would contain the actual sale prices of those houses
    /// 
    /// The "Async" suffix and Task return type mean this method can run in the background
    /// while your application continues doing other work.
    /// </remarks>
    Task TrainAsync(Matrix<T> x, Vector<T> y);

    /// <summary>
    /// Makes predictions asynchronously using the trained model for the given input data.
    /// </summary>
    /// <param name="input">The matrix of input features for which to make predictions.</param>
    /// <returns>A Task containing a vector of predicted values, one for each row in the input matrix.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> After your model is trained, this method uses it to make predictions
    /// on new data asynchronously (in the background).
    /// 
    /// The input parameter is a matrix (table) of features, similar to what you used during training,
    /// but these are new examples that the model hasn't seen before.
    /// 
    /// For example, continuing with the house price prediction:
    /// - input would contain features of houses you want to predict prices for
    /// - The returned vector would contain the predicted prices for those houses
    /// 
    /// The "Async" suffix and Task return type mean this method can run in the background.
    /// When the Task completes, you'll get back a Vector containing all the predictions.
    /// </remarks>
    Task<Vector<T>> PredictAsync(Matrix<T> input);
}
