namespace AiDotNet.Models;

/// <summary>
/// Configuration options for the Residual Analysis Fit Detector, which evaluates model fit quality
/// by analyzing prediction residuals against various statistical thresholds.
/// </summary>
/// <remarks>
/// <para>
/// Residual analysis is a critical technique in regression modeling that examines the differences between 
/// observed values and predicted values (residuals) to assess model fit quality. This class provides 
/// configuration options for threshold values used to determine whether a model's residuals indicate 
/// a good fit. The detector evaluates several statistical measures including the mean of residuals, 
/// standard deviation of residuals, Mean Absolute Percentage Error (MAPE), and the coefficient of 
/// determination (R²). By adjusting these thresholds, users can control how strictly the detector 
/// evaluates model fit according to their specific requirements and domain knowledge.
/// </para>
/// <para><b>For Beginners:</b> This class helps you decide if your prediction model is doing a good job.
/// 
/// When a model makes predictions, it's rarely perfect. The differences between what your model 
/// predicted and the actual values are called "residuals." Analyzing these residuals helps determine 
/// if your model is working well.
/// 
/// Think of it like this:
/// - You have a weather app that predicts temperatures
/// - Some days it predicts 75°F when the actual temperature is 73°F (residual of -2°F)
/// - Other days it predicts 68°F when the actual temperature is 72°F (residual of +4°F)
/// - By analyzing all these differences, you can tell if your model is reliable
/// 
/// This class lets you set thresholds for different statistical measures:
/// - How close the average residual should be to zero
/// - How consistent the residuals should be (not too scattered)
/// - How small the percentage errors should be
/// - How much of the data variation your model explains
/// 
/// If your model's residuals stay within these thresholds, it passes the "fit test" and is 
/// considered reliable for making predictions.
/// </para>
/// </remarks>
public class ResidualAnalysisFitDetectorOptions
{
    /// <summary>
    /// Gets or sets the threshold for the mean (average) of residuals.
    /// </summary>
    /// <value>A double value between 0 and 1, defaulting to 0.1.</value>
    /// <remarks>
    /// <para>
    /// The mean of residuals measures the average difference between predicted and actual values. In an 
    /// ideal model, this value should be close to zero, indicating that the model does not systematically 
    /// overpredict or underpredict. This threshold determines how close to zero the mean residual must be 
    /// for the model to be considered well-fitted. A smaller threshold enforces a stricter requirement for 
    /// the model to have balanced residuals, while a larger threshold allows more systematic bias in the 
    /// predictions. The default value of 0.1 provides a moderate constraint that is suitable for many 
    /// applications but may need adjustment based on the specific domain and data characteristics.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how close the average error should be to zero.
    /// 
    /// In an ideal prediction model:
    /// - Some predictions are too high (positive residuals)
    /// - Some predictions are too low (negative residuals)
    /// - These errors should balance out, with an average close to zero
    /// 
    /// The MeanThreshold value (default 0.1) determines how close to zero this average must be:
    /// - Lower values (like 0.05): Stricter requirement, the model must have very balanced errors
    /// - Higher values (like 0.2): More lenient, allowing some systematic bias in predictions
    /// 
    /// For example, if your model consistently predicts temperatures that are 2 degrees too high,
    /// it would have a mean residual of 2. With the default threshold of 0.1, this might be considered
    /// too biased, depending on the scale of your data.
    /// 
    /// Adjust this threshold based on how important it is that your model doesn't consistently
    /// over-predict or under-predict values.
    /// </para>
    /// </remarks>
    public double MeanThreshold { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the threshold for the standard deviation of residuals.
    /// </summary>
    /// <value>A double value between 0 and 1, defaulting to 0.2.</value>
    /// <remarks>
    /// <para>
    /// The standard deviation of residuals measures how widely the residuals are dispersed from their mean. 
    /// A lower standard deviation indicates that the residuals are clustered more tightly around the mean, 
    /// suggesting more consistent prediction errors. This threshold determines the maximum acceptable 
    /// standard deviation for the model to be considered well-fitted. A smaller threshold enforces a stricter 
    /// requirement for consistent prediction errors, while a larger threshold allows more variability in the 
    /// residuals. The default value of 0.2 provides a moderate constraint that is suitable for many applications 
    /// but may need adjustment based on the specific domain and data characteristics.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how consistent your model's errors should be.
    /// 
    /// Standard deviation measures how scattered or spread out your errors are:
    /// - Low standard deviation: Errors are consistently similar in size
    /// - High standard deviation: Some errors are very small while others are very large
    /// 
    /// The StdThreshold value (default 0.2) determines how consistent these errors must be:
    /// - Lower values (like 0.1): Stricter requirement, errors must be very consistent
    /// - Higher values (like 0.3): More lenient, allowing some predictions to be much further off
    /// 
    /// For example, if your weather model is usually within 1-2 degrees but occasionally off by 10 degrees,
    /// it would have a high standard deviation of residuals. With the default threshold of 0.2, this might
    /// be considered too inconsistent.
    /// 
    /// Adjust this threshold based on how important it is that your model makes consistently reliable
    /// predictions versus occasionally having larger errors.
    /// </para>
    /// </remarks>
    public double StdThreshold { get; set; } = 0.2;

    /// <summary>
    /// Gets or sets the threshold for the Mean Absolute Percentage Error (MAPE).
    /// </summary>
    /// <value>A double value between 0 and 1, defaulting to 0.1 (representing 10%).</value>
    /// <remarks>
    /// <para>
    /// The Mean Absolute Percentage Error (MAPE) measures the average of the absolute percentage errors 
    /// between predicted and actual values. It expresses accuracy as a percentage, making it scale-independent 
    /// and thus useful for comparing model performance across different datasets. This threshold determines 
    /// the maximum acceptable MAPE for the model to be considered well-fitted. A smaller threshold enforces 
    /// a stricter requirement for percentage accuracy, while a larger threshold allows larger percentage errors. 
    /// The default value of 0.1 (representing 10%) provides a moderate constraint that is suitable for many 
    /// applications but may need adjustment based on the specific domain and accuracy requirements.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how large your model's percentage errors can be.
    /// 
    /// MAPE (Mean Absolute Percentage Error) measures errors as percentages rather than absolute values:
    /// - It calculates: |actual - predicted| / |actual| for each prediction
    /// - Then takes the average of these percentage errors
    /// - This makes it easier to understand errors across different scales
    /// 
    /// The MapeThreshold value (default 0.1) means:
    /// - The average percentage error should be no more than 10%
    /// - Lower values (like 0.05): Stricter requirement, predictions must be within 5% on average
    /// - Higher values (like 0.2): More lenient, allowing predictions to be off by 20% on average
    /// 
    /// For example, if predicting house prices:
    /// - A $200,000 house predicted as $220,000 has a 10% error
    /// - A $500,000 house predicted as $550,000 has a 10% error
    /// - With the default threshold of 0.1, these would be right at the acceptable limit
    /// 
    /// MAPE is especially useful when your data spans different scales, as it normalizes errors
    /// to percentages rather than absolute values.
    /// </para>
    /// </remarks>
    public double MapeThreshold { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the threshold for the coefficient of determination (R²).
    /// </summary>
    /// <value>A double value between 0 and 1, defaulting to 0.1.</value>
    /// <remarks>
    /// <para>
    /// The coefficient of determination (R²) measures the proportion of variance in the dependent variable 
    /// that is predictable from the independent variables. It ranges from 0 to 1, where 1 indicates that 
    /// the model explains all the variability in the data, and 0 indicates that the model explains none 
    /// of the variability. This threshold determines how close to 1 the R² value must be for the model to 
    /// be considered well-fitted. A larger threshold enforces a stricter requirement for the model to explain 
    /// more of the data variance, while a smaller threshold allows the model to explain less of the variance. 
    /// The default value of 0.1 is relatively lenient and may need adjustment based on the specific domain 
    /// and expectations for model performance.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how much of the data variation your model should explain.
    /// 
    /// R² (R-squared) is a measure between 0 and 1 that indicates how well your model explains the patterns in your data:
    /// - R² = 1.0: Perfect model that explains 100% of the variation in your data
    /// - R² = 0.0: Model that explains 0% of the variation (no better than guessing the average)
    /// 
    /// The R2Threshold value (default 0.1) has a special meaning:
    /// - It's not the minimum R² value your model should have
    /// - Instead, it's how close to 1.0 your R² value should be
    /// - With the default of 0.1, your model's R² should be at least 0.9 (1.0 - 0.1)
    /// - Lower values (like 0.05): Stricter requirement, R² should be at least 0.95
    /// - Higher values (like 0.3): More lenient, R² can be as low as 0.7
    /// 
    /// For example, if your model has an R² of 0.85:
    /// - This means it explains 85% of the variation in your data
    /// - With the default threshold of 0.1, this wouldn't be good enough (as 0.85 < 0.9)
    /// - You'd need to either improve your model or increase the threshold
    /// 
    /// R² is one of the most common ways to evaluate regression models, as it directly measures
    /// how much of the data pattern your model captures.
    /// </para>
    /// </remarks>
    public double R2Threshold { get; set; } = 0.1;
}
