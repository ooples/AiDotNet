namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the Heteroscedasticity Fit Detector, which analyzes whether a model's
/// prediction errors have constant variance across all prediction values.
/// </summary>
/// <remarks>
/// <para>
/// Heteroscedasticity refers to the situation where the variance of errors (residuals) varies across
/// the range of predicted values. In regression analysis, one of the key assumptions is homoscedasticity,
/// which means the error variance should be constant across all levels of the predicted values. When this
/// assumption is violated (heteroscedasticity), it can lead to inefficient parameter estimates and
/// unreliable confidence intervals.
/// </para>
/// <para><b>For Beginners:</b> This detector checks if your model's errors (the differences between
/// predicted and actual values) are consistent across all predictions or if they vary in a systematic way.
/// 
/// Imagine you have a model predicting house prices. If your model tends to make small errors for
/// low-priced houses but much larger errors for expensive houses, that's heteroscedasticity. It means
/// your model's accuracy depends on what it's predicting, which is usually not ideal.
/// 
/// Think of it like a weather forecaster who's very accurate when predicting mild temperatures but
/// wildly inaccurate when predicting extreme temperatures. You'd want to know about this inconsistency
/// because it affects how much you can trust different predictions.
/// 
/// When heteroscedasticity is detected, it often suggests that:
/// - Your model might be missing important features
/// - You might need to transform your target variable (e.g., use log of price instead of price)
/// - You might need a different type of model altogether
/// - You should be more cautious about the model's predictions in certain ranges</para>
/// </remarks>
public class HeteroscedasticityFitDetectorOptions : ModelOptions
{
    /// <summary>
    /// Gets or sets the p-value threshold for detecting heteroscedasticity in model residuals.
    /// </summary>
    /// <value>The heteroscedasticity threshold, defaulting to 0.05 (5%).</value>
    /// <remarks>
    /// <para>
    /// This threshold is used in statistical tests for heteroscedasticity (like the Breusch-Pagan test).
    /// If the p-value from the test is below this threshold, the null hypothesis of homoscedasticity is
    /// rejected, and the model is considered to have heteroscedastic errors. Lower values make the test
    /// more conservative, requiring stronger evidence to declare heteroscedasticity.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines how strong the evidence needs to be before
    /// the detector concludes that your model has inconsistent error patterns (heteroscedasticity).
    /// With the default value of 0.05, if the statistical test finds that there's less than a 5% chance
    /// that the varying error patterns happened by random chance, it will flag heteroscedasticity.
    /// 
    /// Think of it like a jury's standard of evidence - with 0.05, you're saying "I want to be at least
    /// 95% confident before declaring there's a problem with inconsistent errors." If you want to be more
    /// certain before raising an alarm, you could lower this threshold (e.g., to 0.01, requiring 99%
    /// confidence). If you want to be more sensitive to potential issues, you could raise it (e.g., to 0.1).
    /// 
    /// In statistical terms, this is a p-value threshold. The p-value represents the probability that you
    /// would see the observed pattern of errors if the errors were actually consistent (homoscedastic).
    /// A small p-value suggests that the inconsistent pattern is unlikely to be due to random chance.</para>
    /// </remarks>
    public double HeteroscedasticityThreshold { get; set; } = 0.05; // p-value threshold for heteroscedasticity

    /// <summary>
    /// Gets or sets the p-value threshold for confirming homoscedasticity in model residuals.
    /// </summary>
    /// <value>The homoscedasticity threshold, defaulting to 0.1 (10%).</value>
    /// <remarks>
    /// <para>
    /// This threshold is used to positively confirm homoscedasticity (constant error variance). If the
    /// p-value from the heteroscedasticity test is above this threshold, the model is considered to have
    /// homoscedastic errors. Higher values make the test more conservative, requiring stronger evidence
    /// to declare homoscedasticity.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines how confident the detector needs to be before
    /// declaring that your model has consistent error patterns (homoscedasticity). With the default value
    /// of 0.1, if the statistical test finds that there's more than a 10% chance that any varying error
    /// patterns happened by random chance, it will confirm homoscedasticity.
    /// 
    /// Think of it like a doctor giving you a clean bill of health - with 0.1, the doctor is saying "I'm
    /// reasonably confident you don't have this condition, but I can't be absolutely certain." This is
    /// intentionally set higher than the heteroscedasticity threshold to create a "buffer zone" between
    /// definitely heteroscedastic and definitely homoscedastic.
    /// 
    /// If the p-value falls between the heteroscedasticity threshold (0.05) and the homoscedasticity
    /// threshold (0.1), the detector will indicate that the evidence is inconclusive - there might be
    /// some inconsistency in errors, but it's not strong enough to be certain.
    /// 
    /// In practice, having consistent errors (homoscedasticity) is generally preferred because it means
    /// your model's accuracy is similar across all predictions, making it more reliable and easier to
    /// interpret.</para>
    /// </remarks>
    public double HomoscedasticityThreshold { get; set; } = 0.1; // p-value threshold for homoscedasticity
}
