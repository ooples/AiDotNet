namespace AiDotNet.Models;

/// <summary>
/// Represents a comprehensive collection of statistical measures and data for evaluating model performance on a dataset.
/// </summary>
/// <remarks>
/// <para>
/// This class encapsulates various statistical measures and the actual data used to evaluate a model's performance on a 
/// specific dataset. It includes error statistics that quantify prediction errors, basic statistics for both actual and 
/// predicted values, prediction quality statistics, and the raw data including features, actual values, and predicted values. 
/// This comprehensive collection of information allows for thorough analysis of model performance on the dataset.
/// </para>
/// <para><b>For Beginners:</b> This class stores all the statistics and data needed to evaluate how well a model performs.
/// 
/// When evaluating a model's performance:
/// - You need to measure different aspects of accuracy and error
/// - You want to compare actual values with predicted values
/// - You need to keep track of the input data that produced the predictions
/// 
/// This class stores all that information, including:
/// - Various error measurements (how far predictions are from actual values)
/// - Basic statistics about both actual values and predictions
/// - Statistics about prediction quality (how well the model captures patterns)
/// - The actual input features, target values, and model predictions
/// 
/// Having all this information in one place makes it easier to analyze model performance,
/// create visualizations, and compare different models.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class DataSetStats<T>
{
    /// <summary>
    /// Gets or sets the error statistics for the model's predictions.
    /// </summary>
    /// <value>An ErrorStats&lt;T&gt; object containing various error metrics, initialized as an empty object.</value>
    /// <remarks>
    /// <para>
    /// This property contains various error statistics that quantify the differences between the model's predictions and the 
    /// actual target values. These statistics include measures such as mean squared error (MSE), root mean squared error (RMSE), 
    /// mean absolute error (MAE), and others. Error statistics focus on the magnitude of prediction errors and provide different 
    /// perspectives on the model's accuracy. Lower values for these metrics generally indicate better model performance.
    /// </para>
    /// <para><b>For Beginners:</b> This contains measurements of how far off your predictions are from the actual values.
    /// 
    /// The error statistics:
    /// - Measure the size of prediction errors in different ways
    /// - Lower values indicate better performance
    /// - Different metrics emphasize different aspects of error
    /// 
    /// Common error metrics include:
    /// - MSE (Mean Squared Error): Average of squared differences between predictions and actual values
    /// - RMSE (Root Mean Squared Error): Square root of MSE, in the same units as the target variable
    /// - MAE (Mean Absolute Error): Average of absolute differences between predictions and actual values
    /// 
    /// These metrics help you understand how accurate your model is and
    /// can guide you in improving it.
    /// </para>
    /// </remarks>
    public ErrorStats<T> ErrorStats { get; set; } = ErrorStats<T>.Empty();
    
    /// <summary>
    /// Gets or sets the basic descriptive statistics for the actual target values.
    /// </summary>
    /// <value>A BasicStats&lt;T&gt; object containing descriptive statistics for the actual values, initialized as an empty object.</value>
    /// <remarks>
    /// <para>
    /// This property contains basic descriptive statistics for the actual target values in the dataset. These statistics 
    /// include measures such as mean, median, standard deviation, minimum, maximum, and others. They provide a summary of the 
    /// distribution of the target variable and can be useful for understanding the data and interpreting the model's performance.
    /// </para>
    /// <para><b>For Beginners:</b> This contains summary statistics about the actual values in your dataset.
    /// 
    /// The actual basic statistics:
    /// - Summarize the distribution of the actual target values
    /// - Help you understand the data you're working with
    /// - Provide context for interpreting model performance
    /// 
    /// Common statistics include:
    /// - Mean: The average value
    /// - Median: The middle value when sorted
    /// - Standard Deviation: How spread out the values are
    /// - Min/Max: The smallest and largest values
    /// 
    /// These statistics help you understand what you're trying to predict
    /// and provide context for evaluating your model's predictions.
    /// </para>
    /// </remarks>
    public BasicStats<T> ActualBasicStats { get; set; } = BasicStats<T>.Empty();
    
    /// <summary>
    /// Gets or sets the basic descriptive statistics for the predicted values.
    /// </summary>
    /// <value>A BasicStats&lt;T&gt; object containing descriptive statistics for the predicted values, initialized as an empty object.</value>
    /// <remarks>
    /// <para>
    /// This property contains basic descriptive statistics for the model's predictions. These statistics include measures such 
    /// as mean, median, standard deviation, minimum, maximum, and others. They provide a summary of the distribution of the 
    /// predicted values and can be compared with the statistics for the actual values to assess how well the model captures 
    /// the overall distribution of the target variable.
    /// </para>
    /// <para><b>For Beginners:</b> This contains summary statistics about your model's predictions.
    /// 
    /// The predicted basic statistics:
    /// - Summarize the distribution of the values your model predicted
    /// - Can be compared with the actual statistics to see if your model captures the overall distribution
    /// - Help identify systematic biases in your predictions
    /// 
    /// Common statistics include:
    /// - Mean: The average predicted value
    /// - Median: The middle predicted value when sorted
    /// - Standard Deviation: How spread out the predictions are
    /// - Min/Max: The smallest and largest predictions
    /// 
    /// Comparing these with the actual statistics can reveal issues like:
    /// - Predictions that are systematically too high or too low (comparing means)
    /// - Predictions that don't capture the full range of variation (comparing standard deviations)
    /// - Predictions that don't reach the extremes of the actual data (comparing min/max)
    /// </para>
    /// </remarks>
    public BasicStats<T> PredictedBasicStats { get; set; } = BasicStats<T>.Empty();
    
    /// <summary>
    /// Gets or sets the prediction quality statistics for the model.
    /// </summary>
    /// <value>A PredictionStats&lt;T&gt; object containing various prediction quality metrics, initialized as an empty object.</value>
    /// <remarks>
    /// <para>
    /// This property contains various statistics that assess the quality of the model's predictions beyond simple error 
    /// measurements. These statistics include measures such as R-squared (coefficient of determination), adjusted R-squared, 
    /// correlation between predictions and actual values, and others. Prediction quality statistics focus on how well the 
    /// model captures the patterns in the data and explains the variance in the target variable. Higher values for these 
    /// metrics generally indicate better model performance.
    /// </para>
    /// <para><b>For Beginners:</b> This contains measurements of how well your model captures patterns in the data.
    /// 
    /// The prediction statistics:
    /// - Assess how well your model explains the patterns in the data
    /// - Higher values typically indicate better performance
    /// - Focus on the relationship between predictions and actual values
    /// 
    /// Common prediction metrics include:
    /// - R² (R-squared): Proportion of variance explained by the model (0-1, higher is better)
    /// - Adjusted R²: R-squared adjusted for the number of predictors
    /// - Correlation: How strongly predictions and actual values are related
    /// 
    /// These metrics help you understand how well your model captures
    /// the underlying patterns rather than just measuring error size.
    /// </para>
    /// </remarks>
    public PredictionStats<T> PredictionStats { get; set; } = PredictionStats<T>.Empty();
    
    /// <summary>
    /// Gets or sets the vector of predicted values.
    /// </summary>
    /// <value>A vector of predicted values, initialized as an empty vector.</value>
    /// <remarks>
    /// <para>
    /// This property contains the model's predictions for the dataset. Each element in the vector corresponds to an observation 
    /// in the dataset and represents the model's prediction for that observation. These predictions are compared with the actual 
    /// target values to calculate various performance metrics and assess the model's accuracy.
    /// </para>
    /// <para><b>For Beginners:</b> This contains what your model predicted for each data point.
    /// 
    /// The predicted values:
    /// - Are the values your model output for each data point
    /// - Are stored as a vector (array) of values
    /// - Are compared to the actual values to measure performance
    /// 
    /// This vector is important because:
    /// - It contains the raw predictions from your model
    /// - It allows for detailed analysis beyond summary statistics
    /// - It can be used for visualizations like scatter plots of predicted vs. actual values
    /// 
    /// For example, if predicting house prices, this would contain your model's
    /// estimated price for each house in the dataset.
    /// </para>
    /// </remarks>
    public Vector<T> Predicted { get; set; } = Vector<T>.Empty();
    
    /// <summary>
    /// Gets or sets the vector of actual target values.
    /// </summary>
    /// <value>A vector of actual target values, initialized as an empty vector.</value>
    /// <remarks>
    /// <para>
    /// This property contains the actual target values for the dataset. Each element in the vector corresponds to an observation 
    /// in the dataset and represents the true value that the model attempts to predict. These values are compared with the 
    /// model's predictions to calculate various performance metrics and assess the model's accuracy.
    /// </para>
    /// <para><b>For Beginners:</b> This contains the actual values you're trying to predict.
    /// 
    /// The actual values:
    /// - Are what your model is trying to predict
    /// - Are stored as a vector (array) of values
    /// - Are used to calculate how accurate your predictions are
    /// 
    /// This vector is important because:
    /// - It contains the ground truth for your prediction task
    /// - It's used as the reference point for all performance metrics
    /// - It can be used for visualizations like scatter plots of predicted vs. actual values
    /// 
    /// For example, if predicting house prices, this would contain the
    /// actual sale price for each house in your dataset.
    /// </para>
    /// </remarks>
    public Vector<T> Actual { get; set; } = Vector<T>.Empty();
    
    /// <summary>
    /// Gets or sets the matrix of input features.
    /// </summary>
    /// <value>A matrix of input features, initialized as an empty matrix.</value>
    /// <remarks>
    /// <para>
    /// This property contains the input feature matrix for the dataset, where each row represents an observation (data point) 
    /// and each column represents a feature (input variable). This matrix contains the independent variables used to make 
    /// predictions. It is stored to allow for further analysis, visualization, or reuse of the model with the same data.
    /// </para>
    /// <para><b>For Beginners:</b> This contains the input data used for making predictions.
    /// 
    /// The feature matrix:
    /// - Contains all the input variables (features) for each data point
    /// - Is organized with each row representing one observation
    /// - Each column represents a different input variable
    /// 
    /// This matrix is important because:
    /// - It contains the raw input data that produced the predictions
    /// - It allows for analysis of how specific features affect predictions
    /// - It can be used for feature importance analysis or partial dependence plots
    /// 
    /// For example, if predicting house prices, this might contain features like
    /// square footage, number of bedrooms, location, etc. for each house.
    /// </para>
    /// </remarks>
    public Matrix<T> Features { get; set; } = Matrix<T>.Empty();
}