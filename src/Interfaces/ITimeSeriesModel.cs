namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the core functionality for time series prediction models.
/// </summary>
/// <remarks>
/// Time series models analyze sequential data points collected over time to identify patterns
/// and make predictions about future values.
/// 
/// <b>For Beginners:</b> A time series model helps you predict future values based on past data that
/// was collected in sequence over time. For example:
/// 
/// - Predicting tomorrow's temperature based on weather patterns from the past week
/// - Forecasting next month's sales based on previous months' sales data
/// - Estimating website traffic for next week based on historical visitor counts
/// 
/// These models look for patterns in your historical data (like trends, seasonal effects, and cycles)
/// to make educated guesses about what will happen next.
/// 
/// This interface inherits from IModelSerializer, which means these models can be saved to disk
/// and loaded back later - useful for when you've trained a good model and want to use it again
/// without retraining.
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
public interface ITimeSeriesModel<T> : IFullModel<T, Matrix<T>, Vector<T>>
{
    /// <summary>
    /// Evaluates the model's performance using test data.
    /// </summary>
    /// <remarks>
    /// This method measures how well the model's predictions match actual values on data it hasn't seen during training.
    /// 
    /// <b>For Beginners:</b> This method helps you check how accurate your model is. You provide:
    /// 
    /// - xTest: Input data that the model hasn't seen during training
    /// - yTest: The actual correct values for those inputs
    /// 
    /// The method will:
    /// 1. Use your model to make predictions for xTest
    /// 2. Compare those predictions to the actual values in yTest
    /// 3. Calculate various error metrics to tell you how close the predictions were
    /// 
    /// Common metrics include:
    /// - Mean Absolute Error (MAE): The average size of the errors
    /// - Mean Squared Error (MSE): Similar to MAE but penalizes large errors more
    /// - R-squared: How much of the variation in the data is explained by the model
    /// 
    /// Lower error values and higher R-squared values generally indicate a better model.
    /// </remarks>
    /// <param name="xTest">The matrix of input features for testing.</param>
    /// <param name="yTest">The vector of actual target values corresponding to xTest.</param>
    /// <returns>A dictionary of evaluation metrics with metric names as keys and metric values as values.</returns>
    Dictionary<string, T> EvaluateModel(Matrix<T> xTest, Vector<T> yTest);

    /// <summary>
    /// Predicts a single value based on the provided input vector.
    /// </summary>
    /// <param name="input">The input vector containing features for prediction.</param>
    /// <returns>The predicted value for the given input.</returns>
    /// <remarks>
    /// <para>
    /// This method generates a single prediction based on the input vector, providing a 
    /// convenient way to get individual predictions without creating a matrix.
    /// </para>
    /// <para><b>For Beginners:</b> This lets you get a prediction for a single point in time
    /// rather than a whole series of predictions at once. It's like asking "What's the forecast
    /// for tomorrow?" instead of "What's the forecast for the next week?"
    /// </para>
    /// </remarks>
    T PredictSingle(Vector<T> input);
}
