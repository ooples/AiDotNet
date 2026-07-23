namespace AiDotNet.Enums;

/// <summary>
/// Represents different types of model fit quality and common issues in machine learning models.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Model fit describes how well your AI model matches the data it's trying to learn from.
/// 
/// Think of model fit like trying on clothes:
/// - A good fit means the model captures the true patterns in your data
/// - A poor fit means the model doesn't match the data well
/// - Different types of poor fits have different causes and solutions
/// 
/// Common fit problems include:
/// - Overfitting: The model memorizes the training data instead of learning general patterns
/// - Underfitting: The model is too simple to capture important patterns in the data
/// - Bias and variance issues: Different types of errors that affect how your model performs
/// - Multicollinearity: When input variables are too closely related to each other
/// - Autocorrelation: When data points are related to previous data points in a sequence
/// 
/// Understanding the type of fit helps you diagnose problems with your model and make improvements.
/// </para>
/// </remarks>
public enum FitType
{
    /// <summary>
    /// Indicates that the model fits the data well, capturing the underlying patterns without memorizing noise.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A good fit means your model has found the right balance - it captures the important patterns
    /// in your data without being influenced too much by random noise or outliers.
    /// 
    /// Characteristics:
    /// - Performs well on both training and test data
    /// - Captures the true underlying relationship in the data
    /// - Makes reasonable predictions on new, unseen data
    /// - Has appropriate complexity for the problem
    /// 
    /// This is the ideal outcome for any machine learning model.
    /// </para>
    /// </remarks>
    GoodFit,

    /// <summary>
    /// Indicates that the model has memorized the training data too closely, including its noise and outliers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Overfitting happens when your model learns the training data too well, 
    /// memorizing even the random noise instead of just the important patterns.
    /// 
    /// Think of it like a student who memorizes test answers without understanding the concepts:
    /// - Does extremely well on practice questions (training data)
    /// - Performs poorly on new questions (test data)
    /// - Has "memorized" rather than "learned"
    /// 
    /// Characteristics:
    /// - Very high accuracy on training data
    /// - Much lower accuracy on test data
    /// - Model is unnecessarily complex
    /// - Makes unreliable predictions on new data
    /// 
    /// Common causes:
    /// - Model is too complex for the amount of data
    /// - Training for too many iterations
    /// - Not enough regularization
    /// - Too many features compared to data points
    /// 
    /// Solutions:
    /// - Simplify your model
    /// - Get more training data
    /// - Use regularization techniques
    /// - Implement early stopping
    /// - Use cross-validation
    /// </para>
    /// </remarks>
    Overfit,

    /// <summary>
    /// Indicates that the model is too simple to capture the important patterns in the data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Underfitting happens when your model is too simple to capture the important 
    /// patterns in your data.
    /// 
    /// Think of it like using a straight line to describe a curved relationship:
    /// - The model misses important patterns
    /// - It's too simplistic to represent the true relationship
    /// - It performs poorly on both training and test data
    /// 
    /// Characteristics:
    /// - Poor performance on training data
    /// - Similarly poor performance on test data
    /// - Model is too simple
    /// - High error rates across all datasets
    /// 
    /// Common causes:
    /// - Model is too simple (not enough parameters)
    /// - Important features are missing
    /// - Too much regularization
    /// - Not training long enough
    /// 
    /// Solutions:
    /// - Use a more complex model
    /// - Add more relevant features
    /// - Reduce regularization
    /// - Train for more iterations
    /// - Feature engineering to better represent the data
    /// </para>
    /// </remarks>
    Underfit,

    /// <summary>
    /// Indicates that the model consistently misses the true relationship in the data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> High bias means your model makes systematic errors because it's missing 
    /// important patterns in the data.
    /// 
    /// Think of bias like a consistent error in measurement:
    /// - The model consistently underestimates or overestimates values
    /// - It's too simplified to capture the true relationship
    /// - It makes the same kinds of mistakes repeatedly
    /// 
    /// Characteristics:
    /// - Consistently wrong in the same direction
    /// - Underfits the training data
    /// - Similar (poor) performance on training and test data
    /// - Model predictions are far from actual values
    /// 
    /// Common causes:
    /// - Model is too simple
    /// - Important features or interactions are missing
    /// - Incorrect assumptions about the data
    /// 
    /// Solutions:
    /// - Use a more complex model
    /// - Add more features or feature interactions
    /// - Reduce regularization
    /// - Try different model architectures
    /// 
    /// High bias is related to underfitting but specifically refers to the systematic error component.
    /// </para>
    /// </remarks>
    HighBias,

    /// <summary>
    /// Indicates that the model is too sensitive to small fluctuations in the training data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> High variance means your model changes dramatically with small changes in the training data.
    /// 
    /// Think of variance like inconsistency:
    /// - The model is very sensitive to which specific data points it sees during training
    /// - It learns random noise along with the true patterns
    /// - It performs very differently on different subsets of data
    /// 
    /// Characteristics:
    /// - Great performance on training data
    /// - Much worse performance on test data
    /// - Model predictions vary widely with small changes to training data
    /// - Complex model with many parameters
    /// 
    /// Common causes:
    /// - Model is too complex for the amount of data
    /// - Not enough training examples
    /// - Too many features relative to data points
    /// - Insufficient regularization
    /// 
    /// Solutions:
    /// - Get more training data
    /// - Simplify the model
    /// - Use regularization techniques
    /// - Feature selection to reduce dimensionality
    /// - Ensemble methods to average out variance
    /// 
    /// High variance is related to overfitting but specifically refers to the model's sensitivity to changes in training data.
    /// </para>
    /// </remarks>
    HighVariance,

    /// <summary>
    /// Indicates that small changes in the input data cause large, unpredictable changes in the model's predictions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> An unstable model produces wildly different predictions with small changes to input data.
    /// 
    /// Think of instability like a wobbly table:
    /// - Small changes cause big, unpredictable movements
    /// - The model is unreliable because similar inputs produce very different outputs
    /// - Results aren't consistent or trustworthy
    /// 
    /// Characteristics:
    /// - Predictions change dramatically with small input changes
    /// - Different training runs produce very different models
    /// - Performance varies widely across different data subsets
    /// - Often has numerical issues during training
    /// 
    /// Common causes:
    /// - Poor feature scaling
    /// - Multicollinearity (highly correlated features)
    /// - Numerical precision issues
    /// - Too high learning rate
    /// - Complex model with insufficient data
    /// 
    /// Solutions:
    /// - Feature scaling (normalize or standardize inputs)
    /// - Address multicollinearity
    /// - Use more stable algorithms
    /// - Ensemble methods to average out instability
    /// - Regularization techniques
    /// </para>
    /// </remarks>
    Unstable,

    /// <summary>
    /// Indicates that input variables are highly correlated, causing unreliable coefficient estimates.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Severe multicollinearity means some of your input features are so closely related 
    /// that the model can't tell them apart.
    /// 
    /// Think of it like trying to determine the individual contributions of two chefs who always cook together:
    /// - You can't tell which chef is responsible for which aspects of the meal
    /// - The model can't determine which feature is truly causing the effect
    /// - Small changes in data can cause large changes in feature importance
    /// 
    /// Characteristics:
    /// - Very high correlation between two or more input features
    /// - Coefficient estimates are unstable and can flip signs
    /// - Standard errors of coefficients are very large
    /// - Individual feature importance is unreliable
    /// - Overall predictions may still be accurate
    /// 
    /// Common causes:
    /// - Redundant features (e.g., age and birth year)
    /// - Derived features that are closely related
    /// - Features that measure the same underlying factor
    /// 
    /// Solutions:
    /// - Remove one of the correlated features
    /// - Combine correlated features (e.g., using PCA)
    /// - Use regularization techniques (Ridge regression)
    /// - Create interaction terms instead of using separate features
    /// </para>
    /// </remarks>
    SevereMulticollinearity,

    /// <summary>
    /// Indicates that input variables have some correlation, potentially affecting coefficient stability.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Moderate multicollinearity means some of your input features are somewhat related, 
    /// which can make your model less reliable but not completely unstable.
    /// 
    /// Think of it like having two explanatory variables that overlap partially:
    /// - They share some information but also have unique contributions
    /// - The model can still function but coefficient interpretation becomes tricky
    /// - Feature importance may be somewhat misleading
    /// 
    /// Characteristics:
    /// - Moderate correlation between two or more input features
    /// - Coefficient estimates are somewhat unstable
    /// - Standard errors are larger than ideal
    /// - Individual feature importance is somewhat unreliable
    /// - Overall predictions are usually still accurate
    /// 
    /// Common causes:
    /// - Natural correlations in real-world data
    /// - Features that partially measure the same underlying factor
    /// - Trend variables that move together over time
    /// 
    /// Solutions:
    /// - Consider whether all features are necessary
    /// - Use regularization techniques
    /// - Be cautious when interpreting individual coefficients
    /// - Monitor variance inflation factors (VIFs)
    /// </para>
    /// </remarks>
    ModerateMulticollinearity,

    /// <summary>
    /// Indicates that the model does not fit the data well but is not completely useless.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A poor fit means your model captures some patterns in the data but misses many important relationships.
    /// 
    /// Characteristics:
    /// - Below-average performance metrics
    /// - Captures only the strongest patterns in the data
    /// - Makes frequent errors in predictions
    /// - May have issues with both bias and variance
    /// 
    /// Common causes:
    /// - Missing important features
    /// - Wrong type of model for the problem
    /// - Insufficient data preprocessing
    /// - Data quality issues
    /// 
    /// Solutions:
    /// - Feature engineering to create better inputs
    /// - Try different model architectures
    /// - Improve data quality and preprocessing
    /// - Gather more or better data
    /// </para>
    /// </remarks>
    PoorFit,

    /// <summary>
    /// Indicates that the model performs extremely poorly and fails to capture meaningful patterns in the data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A very poor fit means your model is almost completely failing to learn from your data.
    /// 
    /// Think of it like trying to predict the weather by flipping a coin:
    /// - The model's predictions have little to no relationship with the actual outcomes
    /// - It's barely better than random guessing
    /// - Almost no useful patterns are being captured
    /// 
    /// Characteristics:
    /// - Very low performance metrics (close to random)
    /// - Large errors across all predictions
    /// - No meaningful relationship between predictions and actual values
    /// - Model fails on both training and test data
    /// 
    /// Common causes:
    /// - Completely wrong model for the problem
    /// - Major data quality issues
    /// - Missing critical features
    /// - Serious implementation errors
    /// - Data that has no predictable pattern
    /// 
    /// Solutions:
    /// - Reconsider your entire approach
    /// - Check for implementation errors
    /// - Verify data quality and relevance
    /// - Consider if the problem is actually predictable
    /// - Start with a simpler model and build up gradually
    /// </para>
    /// </remarks>
    VeryPoorFit,

    /// <summary>
    /// Indicates that data points are strongly correlated with previous data points in a positive direction.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Strong positive autocorrelation means your data points are strongly related to previous data points.
    /// 
    /// Think of it like weather patterns:
    /// - If today is hot, tomorrow is very likely to be hot too
    /// - Values tend to stay high for a while, then low for a while
    /// - You see clear patterns or "runs" in your data over time
    /// 
    /// Characteristics:
    /// - Data points strongly depend on previous data points
    /// - Errors in your model tend to be similar across consecutive predictions
    /// - If one prediction is too high, the next one is also likely too high
    /// - Data shows clear trends or cycles
    /// 
    /// Common causes:
    /// - Time series data with strong trends
    /// - Seasonal patterns
    /// - Missing important time-dependent variables
    /// - Data collected at intervals shorter than the natural cycle of the phenomenon
    /// 
    /// Solutions:
    /// - Use time series specific models (ARIMA, etc.)
    /// - Include lagged variables as features
    /// - Difference the data to remove trends
    /// - Add features that capture seasonality
    /// - Use specialized error terms that account for autocorrelation
    /// </para>
    /// </remarks>
    StrongPositiveAutocorrelation,

    /// <summary>
    /// Indicates that data points are strongly correlated with previous data points in a negative direction.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Strong negative autocorrelation means your data tends to swing back and forth, with high values 
    /// typically followed by low values and vice versa.
    /// 
    /// Think of it like a pendulum:
    /// - If today's value is high, tomorrow's is likely to be low
    /// - Values tend to alternate between high and low
    /// - The data appears to zigzag when plotted over time
    /// 
    /// Characteristics:
    /// - Data points strongly depend on previous points, but in the opposite direction
    /// - If one value is above average, the next is likely below average
    /// - Errors in your model tend to alternate between positive and negative
    /// - Data shows oscillating patterns
    /// 
    /// Common causes:
    /// - Overcorrection in controlled systems
    /// - Inventory or supply chain oscillations
    /// - Measurement errors or calibration issues
    /// - Alternating data collection methods
    /// 
    /// Solutions:
    /// - Use time series specific models
    /// - Include lagged variables as features
    /// - Consider models that capture oscillating behavior
    /// - Check for measurement or recording issues
    /// - Analyze if the alternating pattern is a real phenomenon or an artifact
    /// </para>
    /// </remarks>
    StrongNegativeAutocorrelation,

    /// <summary>
    /// Indicates that data points have some correlation with previous data points, but the relationship is not strong.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Weak autocorrelation means your data shows some relationship to previous values, 
    /// but the connection isn't very strong.
    /// 
    /// Think of it like the relationship between today's and next week's weather:
    /// - There's some connection, but it's not reliable for prediction
    /// - You can see hints of patterns, but with many exceptions
    /// - The relationship is present but not dominant
    /// 
    /// Characteristics:
    /// - Data points have some dependence on previous points
    /// - Patterns exist but with considerable noise
    /// - Autocorrelation tests show statistically significant but small effects
    /// - Some clustering of similar values, but not consistent
    /// 
    /// Common causes:
    /// - Mild time dependencies in the data
    /// - Distant seasonal effects
    /// - Weak system memory or inertia
    /// - Multiple competing factors affecting the data
    /// 
    /// Solutions:
    /// - Consider whether time series methods would help
    /// - Test if adding lagged variables improves your model
    /// - May be acceptable to ignore if the effect is very small
    /// - Use robust standard errors in statistical testing
    /// </para>
    /// </remarks>
    WeakAutocorrelation,

    /// <summary>
    /// Indicates that data points are not correlated with previous data points.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> No autocorrelation means each data point is independent of previous data points.
    /// 
    /// Think of it like flipping a coin:
    /// - Previous flips don't influence the next flip
    /// - Each data point stands on its own
    /// - There are no time-based patterns to exploit
    /// 
    /// Characteristics:
    /// - Data points show no dependence on previous points
    /// - Errors in your model are randomly distributed over time
    /// - No visible patterns when data is plotted in sequence
    /// - Autocorrelation tests show no significant effects
    /// 
    /// This is often the ideal situation for many statistical models, as it means the 
    /// independence assumption is satisfied. Standard regression and classification 
    /// methods work best when there's no autocorrelation.
    /// 
    /// If your data shows no autocorrelation:
    /// - You can use standard statistical methods with confidence
    /// - You don't need specialized time series approaches
    /// - Your confidence intervals and p-values are more reliable
    /// - You can treat observations as independent samples
    /// </para>
    /// </remarks>
    NoAutocorrelation,

    /// <summary>
    /// Indicates a moderate level of effect or relationship in the data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Moderate indicates a middle-ground situation - not strong enough 
    /// to be concerning, but not weak enough to ignore completely.
    /// 
    /// Think of it like a partly cloudy day:
    /// - Neither completely sunny nor completely overcast
    /// - Has elements of both conditions
    /// - Requires some attention but not immediate action
    /// 
    /// This is a general-purpose value that can apply to different aspects of model fit,
    /// depending on context. It might refer to:
    /// - Moderate correlation between variables
    /// - Moderate fit quality
    /// - Moderate level of any statistical effect
    /// 
    /// When you see a "Moderate" classification:
    /// - The effect is real and worth noting
    /// - It may warrant some attention but isn't critical
    /// - You might want to monitor it in case it becomes stronger
    /// - It represents a middle ground between extremes
    /// </para>
    /// </remarks>
    Moderate
}
