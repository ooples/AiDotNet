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
public interface ITimeSeriesModel<T> : IModelSerializer
{
    /// <summary>
    /// Trains the time series model using historical data.
    /// </summary>
    /// <remarks>
    /// This method fits the model to the provided input-output pairs to learn patterns in the data.
    /// 
    /// <b>For Beginners:</b> Training is how the model "learns" from your historical data. You provide:
    /// 
    /// - x: Your input features organized in rows and columns. For time series, these are often
    ///   previous values and possibly other related variables. For example, to predict tomorrow's
    ///   temperature, your inputs might include today's temperature, humidity, and pressure.
    ///   
    /// - y: The actual values you want to predict (the "correct answers" for your historical data).
    ///   Following our example, these would be the temperatures that actually occurred.
    /// 
    /// During training, the model analyzes these input-output pairs to discover patterns and
    /// relationships it can use to make future predictions.
    /// </remarks>
    /// <param name="x">The matrix of input features where each row represents a time point and each column represents a feature.</param>
    /// <param name="y">The vector of target values corresponding to each row in x.</param>
    void Train(Matrix<T> x, Vector<T> y);

    /// <summary>
    /// Generates predictions for new input data based on the trained model.
    /// </summary>
    /// <remarks>
    /// This method applies the patterns learned during training to make predictions on new data.
    /// 
    /// <b>For Beginners:</b> After your model has learned from historical data, you can use this method
    /// to make predictions for new situations. You provide:
    /// 
    /// - input: New data points that you want predictions for, formatted the same way as your
    ///   training data. For example, if you want to predict tomorrow's temperature, you would
    ///   provide today's temperature, humidity, and pressure.
    /// 
    /// The model will return its predictions based on the patterns it learned during training.
    /// 
    /// Note: You must train the model before making predictions, otherwise the results won't be meaningful.
    /// </remarks>
    /// <param name="input">The matrix of input features to generate predictions for.</param>
    /// <returns>A vector containing the predicted values for each row of input data.</returns>
    Vector<T> Predict(Matrix<T> input);

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
}