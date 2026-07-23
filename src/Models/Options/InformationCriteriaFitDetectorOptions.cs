namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the Information Criteria Fit Detector, which uses statistical information
/// criteria like AIC and BIC to evaluate model quality and complexity trade-offs.
/// </summary>
/// <remarks>
/// <para>
/// Information criteria are statistical measures that balance model fit against complexity to help select
/// the most appropriate model. The two most common criteria are AIC (Akaike Information Criterion) and
/// BIC (Bayesian Information Criterion), which penalize models based on the number of parameters they use.
/// This helps prevent overfitting by favoring simpler models unless more complex ones provide significantly
/// better fit.
/// </para>
/// <para><b>For Beginners:</b> This detector helps you choose the best model by balancing two competing
/// goals: how well the model fits your data and how simple the model is.
/// 
/// Think of it like shopping for a car. You want good performance (model fit), but you also care about
/// fuel efficiency (model simplicity). Information criteria like AIC and BIC give you a single score that
/// considers both aspects, helping you make better decisions.
/// 
/// Why is this important? Because a very complex model might fit your training data perfectly but perform
/// poorly on new data (like a gas-guzzling sports car that's impractical for daily use). On the other hand,
/// a model that's too simple might miss important patterns (like an underpowered car that can't handle hills).
/// 
/// The Information Criteria Fit Detector uses these scores to help you find the "sweet spot" - a model that's
/// just complex enough to capture the important patterns in your data, but no more complex than necessary.</para>
/// </remarks>
public class InformationCriteriaFitDetectorOptions : ModelOptions
{
    /// <summary>
    /// Gets or sets the threshold for significant differences in AIC (Akaike Information Criterion) values
    /// when comparing models.
    /// </summary>
    /// <value>The AIC threshold, defaulting to 2.0.</value>
    /// <remarks>
    /// <para>
    /// This threshold determines when the difference between AIC values of two models is considered significant.
    /// If the difference in AIC exceeds this threshold, the model with the lower AIC is considered substantially
    /// better. The AIC balances model fit against complexity with a relatively lighter penalty for complexity
    /// compared to BIC.
    /// </para>
    /// <para><b>For Beginners:</b> AIC (Akaike Information Criterion) is a score that helps compare different
    /// models - lower scores are better. This threshold setting determines how much better one model's AIC
    /// score needs to be before we consider it meaningfully superior to another model.
    /// 
    /// With the default value of 2.0, if Model A has an AIC that's at least 2.0 points lower than Model B,
    /// we would consider Model A to be significantly better than Model B. If the difference is smaller,
    /// we might consider the models roughly equivalent.
    /// 
    /// AIC tends to be more lenient toward complex models than BIC, making it useful when you have a lot of
    /// data and want to capture subtle patterns. It's like a judge who's a bit more forgiving about fuel
    /// efficiency as long as the car performs well.
    /// 
    /// The value of 2.0 comes from statistical theory - differences of about 2 or more suggest "substantial"
    /// evidence favoring the model with the lower AIC. You might adjust this threshold based on how
    /// conservative you want to be in model selection.</para>
    /// </remarks>
    public double AicThreshold { get; set; } = 2.0;

    /// <summary>
    /// Gets or sets the threshold for significant differences in BIC (Bayesian Information Criterion) values
    /// when comparing models.
    /// </summary>
    /// <value>The BIC threshold, defaulting to 2.0.</value>
    /// <remarks>
    /// <para>
    /// This threshold determines when the difference between BIC values of two models is considered significant.
    /// If the difference in BIC exceeds this threshold, the model with the lower BIC is considered substantially
    /// better. The BIC balances model fit against complexity with a stronger penalty for complexity compared to AIC,
    /// especially as sample size increases.
    /// </para>
    /// <para><b>For Beginners:</b> BIC (Bayesian Information Criterion) is another score for comparing models,
    /// similar to AIC but with a stronger preference for simpler models. Like AIC, lower scores are better.
    /// This threshold setting determines how much better one model's BIC score needs to be before we consider
    /// it meaningfully superior to another model.
    /// 
    /// With the default value of 2.0, if Model A has a BIC that's at least 2.0 points lower than Model B,
    /// we would consider Model A to be significantly better than Model B. If the difference is smaller,
    /// we might consider the models roughly equivalent.
    /// 
    /// BIC is more strict about model complexity than AIC, especially when you have a lot of data. It's like
    /// a judge who really values fuel efficiency and will only accept gas-guzzling sports cars if they offer
    /// dramatically better performance.
    /// 
    /// BIC is often preferred when you want to avoid overfitting and are willing to potentially miss some
    /// subtle patterns in exchange for a more robust, generalizable model. The threshold of 2.0 is based on
    /// statistical theory about what constitutes "substantial" evidence favoring one model over another.</para>
    /// </remarks>
    public double BicThreshold { get; set; } = 2.0;

    /// <summary>
    /// Gets or sets the threshold for detecting overfitting based on the relative difference between
    /// information criteria of nested models.
    /// </summary>
    /// <value>The overfit threshold, defaulting to 0.1 (10%).</value>
    /// <remarks>
    /// <para>
    /// This threshold determines when a more complex model is considered to be overfitting compared to a simpler
    /// nested model. If the relative improvement in fit (adjusted for the information criterion penalty) is less
    /// than this threshold when moving to a more complex model, the more complex model is likely overfitting.
    /// </para>
    /// <para><b>For Beginners:</b> This setting helps identify when adding complexity to your model isn't
    /// providing enough benefit to justify the added complexity. With the default value of 0.1, if adding
    /// more features or parameters to your model improves the fit by less than 10% (after accounting for
    /// the complexity penalty in the information criteria), the more complex model is flagged as potentially
    /// overfitting.
    /// 
    /// For example, if you have a simple model with 3 features and a more complex model with 10 features,
    /// but the complex model only improves the adjusted fit by 8%, it would be flagged as overfitting. This
    /// suggests that the additional 7 features aren't providing enough value to justify their inclusion.
    /// 
    /// Think of it like upgrading to a more expensive car that only drives slightly better - probably not
    /// worth the extra cost. When overfitting is detected, you're usually better off sticking with the
    /// simpler model, which is likely to perform better on new data even if it's slightly less accurate
    /// on the training data.</para>
    /// </remarks>
    public double OverfitThreshold { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the threshold for detecting underfitting based on the relative difference between
    /// information criteria of nested models.
    /// </summary>
    /// <value>The underfit threshold, defaulting to 0.1 (10%).</value>
    /// <remarks>
    /// <para>
    /// This threshold determines when a simpler model is considered to be underfitting compared to a more complex
    /// nested model. If the relative improvement in fit (adjusted for the information criterion penalty) exceeds
    /// this threshold when moving to a more complex model, the simpler model is likely underfitting.
    /// </para>
    /// <para><b>For Beginners:</b> This setting helps identify when your model is too simple and would benefit
    /// significantly from added complexity. With the default value of 0.1, if adding more features or parameters
    /// to your model improves the fit by more than 10% (after accounting for the complexity penalty in the
    /// information criteria), the simpler model is flagged as potentially underfitting.
    /// 
    /// For example, if you have a simple model with 3 features and a more complex model with 5 features,
    /// and the complex model improves the adjusted fit by 15%, the simpler model would be flagged as underfitting.
    /// This suggests that the additional 2 features are capturing important patterns that the simpler model misses.
    /// 
    /// Think of it like upgrading from a basic car to a mid-range model that drives significantly better -
    /// the upgrade is probably worth the extra cost. When underfitting is detected, you're usually better off
    /// using the more complex model, as the simpler one is missing important patterns in the data.</para>
    /// </remarks>
    public double UnderfitThreshold { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the threshold for detecting high variance based on the relative difference between
    /// information criteria across different data samples.
    /// </summary>
    /// <value>The high variance threshold, defaulting to 0.2 (20%).</value>
    /// <remarks>
    /// <para>
    /// This threshold determines when a model is considered to have high variance based on how much its
    /// information criteria scores vary across different data samples. If the relative standard deviation
    /// of the information criteria exceeds this threshold when evaluated on different subsets of the data,
    /// the model likely has high variance and may be sensitive to the specific data used for training.
    /// </para>
    /// <para><b>For Beginners:</b> This setting helps identify when your model's quality is too inconsistent
    /// across different subsets of your data. With the default value of 0.2, if your model's information
    /// criteria scores vary by more than 20% when evaluated on different random samples of your data, it's
    /// flagged as having high variance.
    /// 
    /// For example, if you calculate AIC scores for your model using 5 different random samples of your data
    /// and get values that vary widely (like 120, 90, 150, 105, 135), the relative standard deviation might
    /// exceed 20%, indicating high variance. This suggests your model's performance depends too much on which
    /// specific data points it sees.
    /// 
    /// Think of it like a car that performs great on dry roads but terribly in rain - it's inconsistent and
    /// not reliable across different conditions. High variance often indicates that your model is too complex
    /// for the amount of data you have, or that there's too much noise in your data.
    /// 
    /// When high variance is detected, you might want to:
    /// - Simplify your model
    /// - Gather more training data
    /// - Use regularization techniques
    /// - Apply ensemble methods to stabilize predictions</para>
    /// </remarks>
    public double HighVarianceThreshold { get; set; } = 0.2;
}
