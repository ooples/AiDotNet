namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Principal Component Regression (PCR), which combines principal component analysis
/// with linear regression to address multicollinearity and dimensionality issues in regression problems.
/// </summary>
/// <remarks>
/// <para>
/// Principal Component Regression (PCR) is a two-step technique that first uses Principal Component Analysis (PCA)
/// to reduce the dimensionality of the feature space, and then performs linear regression on the resulting principal
/// components. This approach is particularly valuable when dealing with datasets where the predictor variables are
/// highly correlated (multicollinearity) or when the number of predictors is large relative to the number of
/// observations. By transforming the original features into uncorrelated principal components, PCR mitigates issues
/// such as model instability and overfitting that can arise in standard regression. The reduced dimensionality
/// also improves computational efficiency and interpretability. PCR is widely used in fields such as spectroscopy,
/// chemometrics, bioinformatics, and econometrics, where high-dimensional, correlated data is common.
/// </para>
/// <para><b>For Beginners:</b> Principal Component Regression helps solve problems with complex, highly related data.
/// 
/// Imagine you're trying to predict house prices with 50 different variables:
/// - Many of these variables are strongly related (like number of rooms, square footage, number of bathrooms)
/// - Using all these related variables directly can confuse your model
/// - The model might become unstable or "overfit" to your training data
/// 
/// What Principal Component Regression does:
/// 
/// Step 1: Principal Component Analysis (PCA)
/// - It combines your original variables into new "super variables" called principal components
/// - Each component captures a different pattern in your data
/// - The first component captures the strongest pattern, the second component the next strongest, and so on
/// - These components are completely unrelated to each other (uncorrelated)
/// 
/// Step 2: Regression
/// - Instead of using your original 50 variables, it uses the top principal components
/// - This makes your model more stable and often more accurate
/// 
/// Think of it like cooking:
/// - Your original variables are like individual spices
/// - PCA combines these into a few special spice mixes (components)
/// - Your recipe now uses these few special mixes instead of dozens of individual spices
/// - This makes cooking (modeling) simpler and often gives better results
/// 
/// This class lets you configure how many components to use and how much information to retain.
/// </para>
/// </remarks>
public class PrincipalComponentRegressionOptions<T> : RegressionOptions<T>
{
    /// <summary>
    /// Gets or sets the number of principal components to use in the regression model.
    /// </summary>
    /// <value>The number of components, defaulting to 0 (auto-selection based on explained variance).</value>
    /// <remarks>
    /// <para>
    /// This parameter specifies the exact number of principal components to retain for the regression step.
    /// When set to a positive integer, the algorithm will use exactly that many components, regardless of how
    /// much variance they explain. A value of 0 (the default) indicates that the number of components should
    /// be automatically determined based on the ExplainedVarianceRatio parameter. Setting a specific number
    /// of components gives precise control over model complexity but requires domain knowledge or cross-validation
    /// to determine the optimal value. Using too few components may lead to underfitting, while using too many
    /// may reintroduce the issues of multicollinearity and overfitting that PCR aims to address.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls exactly how many principal components to use in your regression model.
    /// 
    /// The default value of 0 means:
    /// - The system will automatically choose the number of components
    /// - It will select enough components to explain the percentage of variance specified in ExplainedVarianceRatio
    /// - This automatic selection is usually a good starting point
    /// 
    /// Think of principal components like summarizing a long book:
    /// - The first component captures the main plot
    /// - The second component adds important subplots
    /// - Additional components add more and more details
    /// - At some point, additional components add very little meaningful information
    /// 
    /// You might want to specify a certain number (like 5 or 10):
    /// - When you have expert knowledge about how many underlying factors influence your data
    /// - When you've used cross-validation to determine the optimal number
    /// - When you want to ensure consistent model structure across different datasets
    /// - When you need to control model complexity precisely
    /// 
    /// You might want to keep it at 0 (automatic):
    /// - When you're not sure how many components you need
    /// - When you want the model to adapt to different datasets
    /// - When you prefer to specify how much variance to capture (via ExplainedVarianceRatio)
    /// 
    /// Note: When NumComponents is 0, the ExplainedVarianceRatio parameter determines how many components are used.
    /// When NumComponents is positive, ExplainedVarianceRatio is ignored.
    /// </para>
    /// </remarks>
    public int NumComponents { get; set; } = 0;

    /// <summary>
    /// Gets or sets the minimum ratio of variance to be explained by the selected principal components.
    /// </summary>
    /// <value>The explained variance ratio threshold, defaulting to 0.95 (95%).</value>
    /// <remarks>
    /// <para>
    /// This parameter determines the cumulative proportion of variance that should be explained by the
    /// selected principal components when automatic component selection is used (NumComponents = 0).
    /// The algorithm will select the minimum number of components needed to reach this threshold.
    /// For example, a value of 0.95 means that enough components will be included to explain at least
    /// 95% of the total variance in the original features. This approach provides a data-driven method
    /// for dimensionality reduction that adapts to the characteristics of each dataset. Higher values
    /// preserve more information but reduce the dimensionality benefit, while lower values provide more
    /// aggressive dimensionality reduction but may lose important signal.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines how much of the original information should be retained when automatically selecting components.
    /// 
    /// The default value of 0.95 means:
    /// - The automatically selected components should capture at least 95% of the information in your original data
    /// - The remaining 5% is considered less important and can be discarded
    /// - This provides a good balance between simplification and information preservation
    /// 
    /// Think of it like compressing a photo:
    /// - A value of 1.0 would keep 100% of the details (no real compression)
    /// - A value of 0.95 might remove subtle details but keep the image looking very good
    /// - A value of 0.80 would compress more aggressively, losing some visible details
    /// - A value of 0.50 would lose significant details, leaving only the main elements
    /// 
    /// You might want a higher value (like 0.99):
    /// - When preserving maximum information is critical
    /// - When you want to ensure subtle patterns aren't lost
    /// - When you have plenty of computational resources
    /// - For exploratory analysis where you want to retain most of the data's complexity
    /// 
    /// You might want a lower value (like 0.90 or 0.80):
    /// - When you need more aggressive dimensionality reduction
    /// - When you suspect some of the variance in your data is just noise
    /// - When you're dealing with very high-dimensional data
    /// - When simpler models are preferred for interpretability or computational efficiency
    /// 
    /// Note: This parameter is only used when NumComponents = 0. If NumComponents is set to a positive value,
    /// ExplainedVarianceRatio is ignored.
    /// </para>
    /// </remarks>
    public double ExplainedVarianceRatio { get; set; } = 0.95;
}
