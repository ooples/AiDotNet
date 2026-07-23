namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Stepwise Regression, an automated feature selection approach
/// that iteratively adds or removes predictors based on their statistical significance.
/// </summary>
/// <typeparam name="T">The data type used in matrix operations for the regression model.</typeparam>
/// <remarks>
/// <para>
/// Stepwise Regression is an automated approach to building regression models by iteratively adding or removing 
/// predictor variables based on their statistical significance. This technique helps identify the most important 
/// features while excluding those that contribute little to the model's predictive power, resulting in more 
/// parsimonious and potentially more interpretable models. There are several variants of stepwise regression, 
/// including forward selection (starting with no predictors and adding them one by one), backward elimination 
/// (starting with all predictors and removing them one by one), and bidirectional elimination (a combination of 
/// both approaches). This class provides configuration options for controlling the stepwise regression process, 
/// including the selection method, constraints on the number of features, and criteria for determining when to 
/// stop adding or removing features.
/// </para>
/// <para><b>For Beginners:</b> Stepwise Regression helps automatically select the most important variables for your model.
/// 
/// When building a regression model:
/// - You often have many potential predictor variables
/// - Not all variables are equally useful
/// - Including too many variables can lead to overfitting
/// - Including too few might miss important relationships
/// 
/// Stepwise regression solves this by:
/// - Systematically testing different combinations of variables
/// - Adding or removing variables one at a time
/// - Keeping only those that significantly improve the model
/// - Stopping when further changes don't help much
/// 
/// This approach helps you:
/// - Identify which variables actually matter
/// - Create simpler, more interpretable models
/// - Avoid the computational cost of unnecessary variables
/// - Potentially improve prediction accuracy
/// 
/// This class lets you configure exactly how the stepwise selection process works.
/// </para>
/// </remarks>
public class StepwiseRegressionOptions<T> : RegressionOptions<T>
{
    /// <summary>
    /// Gets or sets the stepwise selection method to use.
    /// </summary>
    /// <value>A value from the StepwiseMethod enumeration, defaulting to StepwiseMethod.Forward.</value>
    /// <remarks>
    /// <para>
    /// This property specifies which stepwise selection method to use for building the regression model. Different 
    /// methods have different approaches to feature selection, with trade-offs in terms of computational efficiency 
    /// and the quality of the resulting model. Forward selection (the default) starts with no predictors and adds 
    /// them one by one, selecting at each step the variable that provides the most significant improvement to the 
    /// model. Backward elimination starts with all predictors and removes them one by one, eliminating at each step 
    /// the variable that contributes least to the model. Bidirectional elimination (also known as stepwise selection) 
    /// combines both approaches, allowing variables to be added or removed at each step based on their significance. 
    /// The choice of method can affect both the computational efficiency of the selection process and the final set 
    /// of selected features.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines the strategy used to select variables for your model.
    /// 
    /// The Method property controls the approach to variable selection:
    /// - Forward: Starts with no variables and adds them one by one
    /// - Backward: Starts with all variables and removes them one by one
    /// - Bidirectional: Can both add and remove variables at each step
    /// 
    /// The default Forward method:
    /// - Begins with an empty model
    /// - Adds the most significant variable first
    /// - Continues adding variables as long as they improve the model
    /// - Stops when no remaining variable would significantly improve the model
    /// 
    /// Think of it like this:
    /// - Forward: Building a team by adding the best available player at each step
    /// - Backward: Starting with everyone and cutting the least valuable player at each step
    /// - Bidirectional: Both adding and removing players to optimize the team
    /// 
    /// When to adjust this value:
    /// - Use Forward (default) when you have many variables and want to build a minimal model
    /// - Use Backward when you suspect most variables are relevant
    /// - Use Bidirectional for the most thorough (but computationally intensive) selection
    /// 
    /// For example, with 100 potential predictors, Forward selection is usually more efficient,
    /// while with 10 predictors that are all potentially important, Backward might be better.
    /// </para>
    /// </remarks>
    public StepwiseMethod Method { get; set; } = StepwiseMethod.Forward;

    /// <summary>
    /// Gets or sets the maximum number of features to include in the final model.
    /// </summary>
    /// <value>A positive integer, defaulting to int.MaxValue (no limit).</value>
    /// <remarks>
    /// <para>
    /// This property specifies the maximum number of predictor variables that can be included in the final regression 
    /// model. It serves as a constraint on the model complexity, preventing the stepwise procedure from adding too 
    /// many features even if they appear to be statistically significant. The default value of int.MaxValue effectively 
    /// means there is no upper limit on the number of features, allowing the stepwise procedure to include as many 
    /// features as meet the statistical criteria. Setting a lower value can help prevent overfitting, especially when 
    /// the number of observations is limited relative to the number of potential predictors. The appropriate value 
    /// depends on the specific application, the number of available observations, and the desired balance between 
    /// model complexity and predictive power.
    /// </para>
    /// <para><b>For Beginners:</b> This setting limits how many variables can be included in your final model.
    /// 
    /// The maximum features constraint:
    /// - Sets an upper limit on model complexity
    /// - Prevents the model from using too many variables
    /// - Helps avoid overfitting (when a model learns noise instead of patterns)
    /// 
    /// The default value of int.MaxValue means:
    /// - No artificial limit is imposed
    /// - The stepwise procedure will include as many variables as meet its statistical criteria
    /// 
    /// Think of it like this:
    /// - Setting MaxFeatures=5: The model will use at most 5 variables, even if more seem useful
    /// - Setting MaxFeatures=10: Allows up to 10 variables in the final model
    /// - Default (int.MaxValue): No limit - uses as many variables as the statistical criteria suggest
    /// 
    /// When to adjust this value:
    /// - Set a specific limit when you need a simpler, more interpretable model
    /// - Set a limit when you have limited data compared to the number of potential predictors
    /// - Leave at default when you want the statistical criteria to determine feature count
    /// 
    /// For example, in a medical study with 100 patients and 50 potential predictors,
    /// you might set MaxFeatures=10 to ensure the model doesn't become too complex for the available data.
    /// </para>
    /// </remarks>
    public int MaxFeatures { get; set; } = int.MaxValue;

    /// <summary>
    /// Gets or sets the minimum number of features to include in the final model.
    /// </summary>
    /// <value>A non-negative integer, defaulting to 1.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the minimum number of predictor variables that must be included in the final regression 
    /// model. It ensures that the model retains a certain level of complexity, even if some features do not meet the 
    /// statistical significance criteria. The default value of 1 ensures that at least one predictor is included in 
    /// the model, preventing it from reducing to just an intercept term. This is particularly relevant for backward 
    /// elimination, where features are progressively removed. Setting a higher value can be useful when certain 
    /// predictors are known to be theoretically important or when a certain level of model complexity is desired for 
    /// other reasons. The appropriate value depends on the specific application and the prior knowledge about the 
    /// importance of the predictors.
    /// </para>
    /// <para><b>For Beginners:</b> This setting ensures your model includes at least a certain number of variables.
    /// 
    /// The minimum features constraint:
    /// - Sets a lower limit on model complexity
    /// - Ensures the model doesn't become too simplistic
    /// - Is particularly important for backward selection methods
    /// 
    /// The default value of 1 means:
    /// - The model must include at least one predictor variable
    /// - This prevents having a model with only an intercept term
    /// 
    /// Think of it like this:
    /// - Setting MinFeatures=3: The model must use at least 3 variables
    /// - Setting MinFeatures=0: Allows the possibility of a model with no predictors (intercept only)
    /// - Default (1): Ensures at least one predictor is included
    /// 
    /// When to adjust this value:
    /// - Increase it when you know certain variables must be included based on domain knowledge
    /// - Set to 0 if you want to allow the possibility of an intercept-only model
    /// - Set higher when using backward selection to prevent removing too many variables
    /// 
    /// For example, if you're modeling house prices and know that square footage, location, and
    /// number of bedrooms are always relevant, you might set MinFeatures=3 to ensure these
    /// fundamental predictors remain in the model.
    /// </para>
    /// </remarks>
    public int MinFeatures { get; set; } = 1;

    /// <summary>
    /// Gets or sets the minimum improvement in the model's fit statistic required to add or remove a feature.
    /// </summary>
    /// <value>A positive double value, defaulting to 0.001.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the minimum improvement in the model's fit statistic (such as R-squared, adjusted 
    /// R-squared, or information criteria like AIC or BIC) required to add or remove a feature during the stepwise 
    /// selection process. It serves as a stopping criterion, determining when the improvement from adding or removing 
    /// features becomes too small to justify further changes to the model. The default value of 0.001 requires that 
    /// each step improves the fit statistic by at least 0.1%, which is a moderate threshold suitable for many 
    /// applications. A smaller value would allow more features to be included, potentially capturing more subtle 
    /// relationships but increasing the risk of overfitting. A larger value would be more selective, including only 
    /// features that provide substantial improvements, resulting in a more parsimonious model. The appropriate value 
    /// depends on the specific application and the desired balance between model complexity and fit.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines how much a variable must improve the model to be included.
    /// 
    /// The minimum improvement threshold:
    /// - Controls how selective the algorithm is about adding or removing variables
    /// - Determines when to stop the stepwise process
    /// - Helps balance model complexity against goodness of fit
    /// 
    /// The default value of 0.001 means:
    /// - Each variable must improve the model's fit statistic by at least 0.001 (0.1%)
    /// - This is a moderate threshold that works well for many applications
    /// 
    /// Think of it like this:
    /// - Higher values (e.g., 0.01): More selective, only includes variables with substantial impact
    /// - Lower values (e.g., 0.0001): Less selective, might include variables with subtle effects
    /// 
    /// When to adjust this value:
    /// - Increase it when you want a simpler model with only the strongest predictors
    /// - Decrease it when you want to capture more subtle relationships
    /// - Adjust based on the scale of your fit statistic (R-squared, AIC, BIC, etc.)
    /// 
    /// For example, in a marketing model where you want only the strongest predictors of
    /// customer behavior, you might increase this to 0.01 to include only variables that
    /// improve the model by at least 1%.
    /// </para>
    /// </remarks>
    public double MinImprovement { get; set; } = 0.001;
}
