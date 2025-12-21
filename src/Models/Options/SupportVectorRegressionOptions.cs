namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Support Vector Regression (SVR), a powerful regression technique
/// that uses support vector machines to predict continuous values.
/// </summary>
/// <remarks>
/// <para>
/// Support Vector Regression (SVR) extends the principles of Support Vector Machines (SVM) to regression 
/// problems. SVR works by finding a function that deviates from the observed target values by a value no 
/// greater than a specified margin (epsilon) for each training point, while also remaining as flat as 
/// possible. This approach creates an epsilon-insensitive tube around the function, ignoring errors within 
/// this tube. Points outside the tube become support vectors that determine the function. SVR is particularly 
/// effective for non-linear regression problems when combined with kernel functions, handling complex 
/// relationships in the data while maintaining good generalization properties. This class inherits from 
/// NonLinearRegressionOptions and adds parameters specific to SVR, such as the margin width (epsilon) and 
/// the regularization parameter (C).
/// </para>
/// <para><b>For Beginners:</b> Support Vector Regression is a technique for predicting continuous values that works well with complex data.
/// 
/// When performing regression (predicting continuous values):
/// - Traditional methods like linear regression work well for simple relationships
/// - But real-world data often contains complex, non-linear patterns
/// 
/// Support Vector Regression solves this by:
/// - Creating a "tube" around the regression line/curve
/// - Ignoring small errors that fall within this tube
/// - Focusing on preventing large errors outside the tube
/// - Using "kernel functions" to handle non-linear relationships
/// 
/// This approach offers several benefits:
/// - Handles non-linear relationships well
/// - Less sensitive to outliers than many other methods
/// - Often generalizes well to new data
/// - Works effectively even with limited training data
/// 
/// This class lets you configure the SVR algorithm's behavior.
/// </para>
/// </remarks>
public class SupportVectorRegressionOptions : NonLinearRegressionOptions
{
    /// <summary>
    /// Gets or sets the width of the epsilon-insensitive tube.
    /// </summary>
    /// <value>A positive double value, defaulting to 0.1.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the width of the epsilon-insensitive tube used in SVR. The algorithm ignores errors 
    /// smaller than epsilon, creating a tube around the regression function where no penalty is associated with 
    /// predicted values. Only points outside this tube contribute to the loss function and affect the model. A 
    /// larger epsilon value creates a wider tube, potentially using fewer support vectors and creating a simpler 
    /// model, but possibly at the cost of accuracy. A smaller epsilon creates a narrower tube, potentially using 
    /// more support vectors and fitting the training data more closely, but possibly at the risk of overfitting. 
    /// The default value of 0.1 provides a moderate tube width suitable for many applications, but the optimal 
    /// value depends on the noise level in the data and the desired trade-off between model complexity and accuracy.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how much error is tolerated before it affects the model.
    /// 
    /// The epsilon parameter:
    /// - Defines the width of the "tube" around the regression function
    /// - Errors smaller than epsilon are completely ignored
    /// - Only points outside this tube influence the model
    /// 
    /// The default value of 0.1 means:
    /// - Predictions within ±0.1 of the actual value are considered "good enough"
    /// - The model focuses on reducing errors larger than this threshold
    /// 
    /// Think of it like this:
    /// - Larger values (e.g., 0.5): More tolerant of errors, simpler model, potentially less accurate
    /// - Smaller values (e.g., 0.01): Less tolerant of errors, more complex model, potentially more accurate
    /// 
    /// When to adjust this value:
    /// - Increase it when your data is noisy and you want to avoid overfitting
    /// - Decrease it when you need more precise predictions and have clean data
    /// - Scale it according to the range of your target variable
    /// 
    /// For example, if predicting house prices in thousands of dollars, epsilon=0.1 means
    /// errors up to $100 are ignored, which might be too small. You might increase it to
    /// 5.0 to ignore errors up to $5,000.
    /// </para>
    /// </remarks>
    public double Epsilon { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the regularization parameter.
    /// </summary>
    /// <value>A positive double value, defaulting to 1.0.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the regularization parameter C, which controls the trade-off between achieving a 
    /// low training error and a low testing error. It determines the penalty for errors outside the epsilon tube. 
    /// A larger C value places more emphasis on minimizing errors, potentially leading to a more complex model 
    /// that fits the training data more closely but might not generalize well. A smaller C value places more 
    /// emphasis on model simplicity, potentially leading to a smoother function that might not fit the training 
    /// data as closely but could generalize better. The default value of 1.0 provides a balanced trade-off 
    /// suitable for many applications, but the optimal value depends on the specific dataset and the desired 
    /// balance between fitting the training data and generalization.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how much the model tries to avoid errors versus keeping the model simple.
    /// 
    /// The C parameter:
    /// - Controls the trade-off between model simplicity and accuracy
    /// - Determines how much to penalize errors outside the epsilon tube
    /// 
    /// The default value of 1.0 means:
    /// - A balanced approach between fitting the data and keeping the model simple
    /// 
    /// Think of it like this:
    /// - Larger values (e.g., 10.0): Focus more on reducing errors, potentially more complex model
    /// - Smaller values (e.g., 0.1): Focus more on model simplicity, potentially more errors
    /// 
    /// When to adjust this value:
    /// - Increase it when accuracy on training data is more important than generalization
    /// - Decrease it when you suspect overfitting or want a smoother function
    /// - Often requires experimentation to find the optimal value
    /// 
    /// For example, if your model is underfitting (high error on both training and test data),
    /// you might increase C to 10.0 to allow a more complex model that can better capture
    /// the patterns in your data.
    /// </para>
    /// </remarks>
    public double C { get; set; } = 1.0;
}
