namespace AiDotNet.Enums;

/// <summary>
/// Specifies different loss functions and fitness calculators for evaluating model performance.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Loss functions and fitness calculators measure how well your AI model's predictions match the actual data.
/// 
/// Think of these metrics like grades on a test:
/// - They tell you how well your model is performing
/// - Different metrics focus on different aspects of performance
/// - For most loss functions, lower values mean better performance
/// - For some metrics (like R-squared), higher values mean better performance
/// 
/// When building AI models, you need ways to:
/// - Compare different models to choose the best one
/// - Know when to stop training your model
/// - Understand if your model is actually learning useful patterns
/// - Detect if your model is overfitting (memorizing data instead of learning)
/// 
/// Different metrics are better for different situations, so it's common to look at multiple metrics
/// when evaluating a model.
/// </para>
/// </remarks>
public enum FitnessCalculatorType
{
    /// <summary>
    /// Calculates the average of the squared differences between predicted and actual values.
    /// </summary>
    MeanSquaredError,

    /// <summary>
    /// Calculates the average of the absolute differences between predicted and actual values.
    /// </summary>
    MeanAbsoluteError,

    /// <summary>
    /// Measures the proportion of variance in the dependent variable explained by the independent variables.
    /// </summary>
    RSquared,

    /// <summary>
    /// A modified version of R-squared that adjusts for the number of predictors in the model.
    /// </summary>
    AdjustedRSquared,

    /// <summary>
    /// Calculates the logarithm of the hyperbolic cosine of the prediction error.
    /// </summary>
    LogCosh,

    /// <summary>
    /// Measures the cross-entropy loss for binary classification problems.
    /// </summary>
    BinaryCrossEntropy,

    /// <summary>
    /// Measures the cross-entropy loss for multi-class classification problems.
    /// </summary>
    CategoricalCrossEntropy,

    /// <summary>
    /// A loss function specifically designed for ordinal regression problems.
    /// </summary>
    OrdinalRegressionLoss,

    /// <summary>
    /// Calculates the Huber loss, which combines properties of MSE and MAE.
    /// </summary>
    HuberLoss,

    /// <summary>
    /// Measures the maximum deviation between predicted and actual values.
    /// </summary>
    MaxError,

    /// <summary>
    /// Calculates the mean squared logarithmic error.
    /// </summary>
    MeanSquaredLogError,

    /// <summary>
    /// Measures the median absolute error between predicted and actual values.
    /// </summary>
    MedianAbsoluteError,

    /// <summary>
    /// Calculates the root mean squared error.
    /// </summary>
    RootMeanSquaredError,

    /// <summary>
    /// Measures the mean absolute percentage error.
    /// </summary>
    MeanAbsolutePercentageError,

    /// <summary>
    /// Calculates the exponential loss, which heavily penalizes large errors.
    /// </summary>
    ExponentialLoss,

    /// <summary>
    /// A custom loss function or fitness calculator defined by the user.
    /// </summary>
    Custom
}
