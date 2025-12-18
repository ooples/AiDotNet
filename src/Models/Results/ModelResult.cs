namespace AiDotNet.Models.Results;

/// <summary>
/// Represents the complete results of a model-building process, including the model solution, fitness metrics,
/// fit detection results, evaluation data, and selected features.
/// </summary>
/// <remarks>
/// <para>
/// This struct encapsulates all the important information produced during the model-building and evaluation process. 
/// It includes the symbolic model itself, a fitness score indicating how well the model performs, detailed fit 
/// detection results that analyze potential issues like overfitting or underfitting, comprehensive evaluation data 
/// with various performance metrics, and information about which features were selected for the model. This 
/// comprehensive package of information allows for thorough analysis and comparison of different models.
/// </para>
/// <para><b>For Beginners:</b> This struct is like a container that holds everything about a model's performance.
/// 
/// When building machine learning or statistical models:
/// - You need to track many different aspects of model performance
/// - You want to compare different models to choose the best one
/// - You need to understand not just how well a model performs, but why
/// 
/// This struct stores:
/// - The actual model solution (the equation or algorithm)
/// - How well the model fits the data (fitness score)
/// - Analysis of potential issues like overfitting or underfitting
/// - Detailed performance metrics on different datasets
/// - Which input features were used in the model
/// 
/// Having all this information in one place makes it easier to evaluate, compare,
/// and document your models.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public struct ModelResult<T, TInput, TOutput>
{
    public IFullModel<T, TInput, TOutput> Solution { get; set; }

    /// <summary>
    /// Gets or sets the fitness score of the model.
    /// </summary>
    /// <value>A numeric value representing the model's fitness or performance.</value>
    /// <remarks>
    /// <para>
    /// This property represents the fitness or performance score of the model. The fitness score is a single numeric 
    /// value that quantifies how well the model performs, typically on the training data. Higher values usually 
    /// indicate better performance, though the exact interpretation depends on the specific fitness function used. 
    /// Common fitness metrics include R-squared (coefficient of determination), negative mean squared error, or 
    /// accuracy. The fitness score is often used as the primary criterion for comparing and selecting models during 
    /// the model-building process, especially in evolutionary algorithms or other optimization approaches.
    /// </para>
    /// <para><b>For Beginners:</b> This value tells you how well the model performs overall.
    /// 
    /// The fitness score:
    /// - Measures how well the model fits the data
    /// - Is typically a single number that summarizes performance
    /// - Higher values usually indicate better performance
    /// 
    /// Common fitness metrics include:
    /// - R² (R-squared): Measures the proportion of variance explained (higher is better)
    /// - Negative MSE (Mean Squared Error): Measures prediction error (closer to zero is better)
    /// - Accuracy: For classification problems, the percentage of correct predictions
    /// 
    /// This value is important because:
    /// - It allows you to quickly compare different models
    /// - It's often used to guide the model selection process
    /// - It provides a simple summary of model performance
    /// </para>
    /// </remarks>
    public T Fitness { get; set; }

    /// <summary>
    /// Gets or sets the results of fit detection analysis.
    /// </summary>
    /// <value>A FitDetectorResult&lt;T&gt; object containing detailed fit analysis.</value>
    /// <remarks>
    /// <para>
    /// This property contains the results of fit detection analysis, which evaluates how well the model fits the data 
    /// and identifies potential issues such as underfitting or overfitting. Underfitting occurs when the model is too 
    /// simple to capture the underlying patterns in the data, resulting in poor performance on both training and test 
    /// data. Overfitting occurs when the model is too complex and captures noise in the training data, resulting in 
    /// good performance on training data but poor performance on test data. The FitDetectorResult includes the type 
    /// of fit detected, a confidence level for that assessment, and recommendations for improving the model.
    /// </para>
    /// <para><b>For Beginners:</b> This contains an analysis of how well your model fits the data.
    /// 
    /// The fit detection result:
    /// - Identifies if your model is underfitting, overfitting, or has a good fit
    /// - Provides a confidence level for this assessment
    /// - Offers specific recommendations to improve your model
    /// 
    /// This analysis helps you understand:
    /// - If your model is too simple (underfitting)
    /// - If your model is too complex (overfitting)
    /// - What steps you should take to improve it
    /// 
    /// For example, if your model is overfitting, the fit detection might recommend
    /// adding regularization or reducing model complexity.
    /// </para>
    /// </remarks>
    public FitDetectorResult<T> FitDetectionResult { get; set; }

    /// <summary>
    /// Gets or sets the detailed evaluation data for the model.
    /// </summary>
    /// <value>A ModelEvaluationData&lt;T&gt; object containing comprehensive performance metrics.</value>
    /// <remarks>
    /// <para>
    /// This property contains detailed evaluation data for the model, including various performance metrics calculated 
    /// on different datasets (typically training, validation, and test sets). The evaluation data provides a more 
    /// comprehensive view of model performance than the single fitness score. It may include metrics such as R-squared, 
    /// mean squared error, mean absolute error, and others, calculated for each dataset. This detailed information 
    /// helps in understanding how well the model generalizes to unseen data and in identifying potential issues with 
    /// the model.
    /// </para>
    /// <para><b>For Beginners:</b> This contains detailed performance metrics for your model.
    /// 
    /// The evaluation data:
    /// - Provides multiple performance metrics (not just the fitness score)
    /// - Shows how the model performs on different datasets (training, validation, test)
    /// - Gives a more complete picture of model performance
    /// 
    /// Common metrics included might be:
    /// - R² (R-squared): How much variance is explained
    /// - MSE (Mean Squared Error): Average squared difference between predictions and actual values
    /// - MAE (Mean Absolute Error): Average absolute difference between predictions and actual values
    /// - RMSE (Root Mean Squared Error): Square root of MSE, in the same units as the target variable
    /// 
    /// This detailed information helps you:
    /// - Better understand model strengths and weaknesses
    /// - Compare models using different metrics
    /// - Assess how well the model will generalize to new data
    /// </para>
    /// </remarks>
    public ModelEvaluationData<T, TInput, TOutput> EvaluationData { get; set; }

    /// <summary>
    /// Gets or sets the list of feature vectors selected for the model.
    /// </summary>
    /// <value>A list of Vector&lt;T&gt; objects representing the selected features.</value>
    /// <remarks>
    /// <para>
    /// This property contains the list of feature vectors that were selected for use in the model. Feature selection 
    /// is the process of identifying and selecting the most relevant features (input variables) for the model, which 
    /// can improve model performance, reduce overfitting, and enhance interpretability. The selected features are 
    /// represented as vectors, where each vector corresponds to a feature and contains the values of that feature 
    /// across all observations in the dataset. This information is useful for understanding which input variables 
    /// the model considers important and for reproducing the model with new data.
    /// </para>
    /// <para><b>For Beginners:</b> This list shows which input variables were used in the model.
    /// 
    /// The selected features:
    /// - Represent the input variables that were chosen for the model
    /// - Are stored as vectors (arrays) of values
    /// - May be a subset of all available features if feature selection was performed
    /// 
    /// Feature selection is important because:
    /// - Using too many features can lead to overfitting
    /// - Some features may be irrelevant or redundant
    /// - Models with fewer features are often more interpretable
    /// 
    /// For example, if you started with 20 potential input variables but the model
    /// only uses 5 of them, this list would contain those 5 selected features.
    /// </para>
    /// </remarks>
    public List<Vector<T>> SelectedFeatures { get; set; }
}
