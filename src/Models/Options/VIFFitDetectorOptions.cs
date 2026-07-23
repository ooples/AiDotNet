namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for detecting multicollinearity in regression models using Variance Inflation Factor (VIF) analysis.
/// </summary>
/// <remarks>
/// <para>
/// Variance Inflation Factor (VIF) is a statistical measure used to detect the severity of multicollinearity in 
/// regression analysis. Multicollinearity occurs when independent variables in a regression model are highly 
/// correlated with each other, which can lead to unstable and unreliable coefficient estimates. VIF quantifies 
/// how much the variance of an estimated regression coefficient is increased due to collinearity with other 
/// predictors. This class provides configuration options for thresholds used to interpret VIF values and detect 
/// problematic levels of multicollinearity in regression models. These thresholds help automate the process of 
/// model evaluation and variable selection.
/// </para>
/// <para><b>For Beginners:</b> This class helps you detect when predictor variables in your model are too closely related to each other.
/// 
/// When building regression models:
/// - Multicollinearity occurs when predictor variables are highly correlated with each other
/// - This can make your model unstable and difficult to interpret
/// - Coefficients may change dramatically with small changes in the data
/// - Standard errors of coefficients become inflated
/// 
/// VIF (Variance Inflation Factor):
/// - Measures how much the variance of a coefficient is increased due to multicollinearity
/// - Higher VIF values indicate more severe multicollinearity
/// - VIF = 1 means no multicollinearity
/// - VIF > 1 indicates some degree of multicollinearity
/// 
/// This class provides thresholds to automatically detect problematic levels of
/// multicollinearity in your models, helping you identify when you should consider
/// removing or combining variables.
/// </para>
/// </remarks>
public class VIFFitDetectorOptions : ModelOptions
{
    /// <summary>
    /// Gets or sets the threshold for detecting severe multicollinearity.
    /// </summary>
    /// <value>A positive double value, defaulting to 10.0.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the VIF threshold above which multicollinearity is considered severe. Severe 
    /// multicollinearity indicates that the predictor variables are so highly correlated that the regression 
    /// coefficients are likely to be very unstable and the model may be unreliable. When a predictor has a VIF 
    /// value exceeding this threshold, it is often recommended to remove it from the model or to combine it with 
    /// other correlated predictors. The default value of 10.0 is a commonly used threshold in statistical practice, 
    /// though some fields may use more or less conservative values depending on the specific application. A lower 
    /// threshold is more strict, flagging more variables as having severe multicollinearity, while a higher 
    /// threshold is more lenient.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines when multicollinearity is considered a serious problem.
    /// 
    /// Severe multicollinearity:
    /// - Indicates predictor variables are so highly correlated that the model is likely unreliable
    /// - Often requires intervention such as removing variables or using regularization techniques
    /// - Can lead to coefficients with incorrect signs or implausible magnitudes
    /// 
    /// The default value of 10.0 means:
    /// - VIF values above 10 indicate severe multicollinearity
    /// - This is a widely accepted threshold in statistical practice
    /// 
    /// Think of it like this:
    /// - Lower values (e.g., 5.0): More strict, flags more variables as severely multicollinear
    /// - Higher values (e.g., 20.0): More lenient, only flags the most extreme cases
    /// 
    /// When to adjust this value:
    /// - Decrease it in fields where precise coefficient estimates are critical
    /// - Increase it when working with naturally correlated predictors where some multicollinearity is expected
    /// - Consider domain-specific standards in your field
    /// 
    /// For example, in medical research where precise effect estimates are crucial,
    /// you might decrease this to 5.0 to be more conservative about multicollinearity.
    /// </para>
    /// </remarks>
    public double SevereMulticollinearityThreshold { get; set; } = 10.0;

    /// <summary>
    /// Gets or sets the threshold for detecting moderate multicollinearity.
    /// </summary>
    /// <value>A positive double value, defaulting to 5.0.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the VIF threshold above which multicollinearity is considered moderate. Moderate 
    /// multicollinearity indicates that there is some correlation among predictor variables that may affect the 
    /// stability of the regression coefficients, but not necessarily to a degree that requires immediate action. 
    /// When a predictor has a VIF value between this threshold and the SevereMulticollinearityThreshold, it might 
    /// be worth monitoring or investigating further, but it may not require removal from the model. The default 
    /// value of 5.0 is a commonly used threshold in statistical practice, representing a middle ground between 
    /// no multicollinearity and severe multicollinearity. A lower threshold is more strict, flagging more variables 
    /// as having moderate multicollinearity, while a higher threshold is more lenient.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines when multicollinearity should raise some concern.
    /// 
    /// Moderate multicollinearity:
    /// - Indicates predictor variables have enough correlation to potentially affect your model
    /// - May warrant further investigation but doesn't necessarily require immediate action
    /// - Can make it harder to determine the individual importance of correlated predictors
    /// 
    /// The default value of 5.0 means:
    /// - VIF values between 5 and 10 indicate moderate multicollinearity
    /// - This represents a level where you should be aware of potential issues
    /// 
    /// Think of it like this:
    /// - Lower values (e.g., 2.5): More strict, flags more variables as moderately multicollinear
    /// - Higher values (e.g., 7.5): More lenient, fewer variables will be flagged
    /// 
    /// When to adjust this value:
    /// - Decrease it when you want to be more cautious about potential multicollinearity
    /// - Increase it when you're more concerned about severe cases than moderate ones
    /// - Consider using different thresholds for exploratory versus confirmatory analyses
    /// 
    /// For example, in an exploratory data analysis where you want to be alerted to potential issues,
    /// you might decrease this to 2.5 to catch more potential multicollinearity problems early.
    /// </para>
    /// </remarks>
    public double ModerateMulticollinearityThreshold { get; set; } = 5.0;

    /// <summary>
    /// Gets or sets the threshold for determining a good fit in terms of the primary metric.
    /// </summary>
    /// <value>A double value between 0 and 1, defaulting to 0.7.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the minimum value of the primary metric (typically R² or adjusted R²) required for 
    /// the model to be considered a good fit. The primary metric is specified by the PrimaryMetric property. For 
    /// R² and similar metrics, higher values indicate better fit, with 1.0 representing a perfect fit and 0.0 
    /// representing no fit. The default value of 0.7 indicates that the model should explain at least 70% of the 
    /// variance in the dependent variable to be considered a good fit. A higher threshold is more strict, requiring 
    /// better model performance, while a lower threshold is more lenient. The appropriate value depends on the 
    /// specific application and the typical range of the primary metric in the field of study.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines how well your model must perform to be considered good.
    /// 
    /// The good fit threshold:
    /// - Defines the minimum acceptable value for your primary performance metric
    /// - For R², it represents how much variance your model explains
    /// - Helps you automatically evaluate if your model performs well enough
    /// 
    /// The default value of 0.7 means:
    /// - For R², the model should explain at least 70% of the variance
    /// - This is a moderate threshold suitable for many applications
    /// 
    /// Think of it like this:
    /// - Higher values (e.g., 0.8): More strict, requires better model performance
    /// - Lower values (e.g., 0.5): More lenient, accepts models with more unexplained variance
    /// 
    /// When to adjust this value:
    /// - Increase it in fields where high predictive accuracy is expected
    /// - Decrease it for problems where even modest predictive power is valuable
    /// - Adjust based on the typical R² values in your specific field
    /// 
    /// For example, in physical sciences where relationships are often well-defined,
    /// you might increase this to 0.8 or 0.9, while in social sciences or complex
    /// behavioral predictions, you might decrease it to 0.5 or 0.6.
    /// </para>
    /// </remarks>
    public double GoodFitThreshold { get; set; } = 0.7;

    /// <summary>
    /// Gets or sets the primary metric used to evaluate model fit.
    /// </summary>
    /// <value>A value from the MetricType enumeration, defaulting to MetricType.R2.</value>
    /// <remarks>
    /// <para>
    /// This property specifies which metric is used as the primary criterion for evaluating model fit. The most 
    /// common metric is R² (coefficient of determination), which measures the proportion of variance in the 
    /// dependent variable that is predictable from the independent variables. Other possible metrics might include 
    /// adjusted R² (which adjusts for the number of predictors), mean squared error (MSE), or information criteria 
    /// such as AIC or BIC. The default value of MetricType.R2 specifies R² as the primary metric, which is 
    /// appropriate for many applications. The optimal choice depends on the specific goals of the analysis and 
    /// the characteristics of the data.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines which statistical measure is used to evaluate your model's performance.
    /// 
    /// The primary metric:
    /// - Specifies which measure is used to evaluate how well your model fits the data
    /// - Different metrics emphasize different aspects of model performance
    /// - Works with GoodFitThreshold to determine if your model performs well enough
    /// 
    /// The default value of R2 means:
    /// - The coefficient of determination (R²) is used as the primary metric
    /// - R² measures the proportion of variance explained by your model
    /// - Values range from 0 (no explanation) to 1 (perfect explanation)
    /// 
    /// Common alternatives include:
    /// - AdjustedR2: Similar to R² but penalizes adding unnecessary predictors
    /// - MSE: Mean Squared Error, measures the average squared difference between predictions and actual values
    /// - AIC/BIC: Information criteria that balance fit and complexity
    /// 
    /// When to adjust this value:
    /// - Change to AdjustedR2 when comparing models with different numbers of predictors
    /// - Change to error-based metrics (like MSE) when prediction accuracy is more important than explanation
    /// - Consider your specific goals (explanation vs. prediction) when choosing
    /// 
    /// For example, if you're comparing models with different numbers of variables,
    /// you might change this to MetricType.AdjustedR2 to account for model complexity.
    /// </para>
    /// </remarks>
    public MetricType PrimaryMetric { get; set; } = MetricType.R2;
}
