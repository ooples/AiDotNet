namespace AiDotNet.Models;

/// <summary>
/// Configuration options for regularization techniques used to prevent overfitting in machine learning models.
/// </summary>
/// <remarks>
/// <para>
/// Regularization is a technique used in machine learning to prevent overfitting by adding a penalty term to the 
/// loss function. Overfitting occurs when a model learns the training data too well, including its noise and 
/// outliers, resulting in poor performance on new, unseen data. By adding regularization, the model is 
/// encouraged to learn simpler patterns, improving its generalization capabilities. This class provides 
/// configuration options for different types of regularization methods, including L1 (Lasso), L2 (Ridge), 
/// and Elastic Net regularization, allowing users to control the strength and behavior of the regularization 
/// applied to their models.
/// </para>
/// <para><b>For Beginners:</b> Regularization helps prevent your model from "memorizing" the training data.
/// 
/// Think of regularization like training wheels for your machine learning model:
/// - Without regularization, your model might become too complex and "overfit" the training data
/// - Overfitting means your model performs well on training data but poorly on new data
/// - Regularization adds constraints that keep your model simpler and more general
/// 
/// There are three main types of regularization you can choose:
/// - L1 (Lasso): Tends to create sparse models by setting some weights exactly to zero
/// - L2 (Ridge): Keeps all weights small but non-zero
/// - Elastic Net: A mix of both L1 and L2 approaches
/// 
/// For example, if you're predicting house prices:
/// - Without regularization: Your model might put too much importance on rare features like "has a wine cellar"
/// - With regularization: Your model focuses more on common, reliable patterns like square footage and location
/// 
/// This class lets you configure what type of regularization to use and how strongly to apply it.
/// </para>
/// </remarks>
public class RegularizationOptions
{
    /// <summary>
    /// Gets or sets the type of regularization to apply to the model.
    /// </summary>
    /// <value>The regularization type, defaulting to None.</value>
    /// <remarks>
    /// <para>
    /// This property determines which regularization technique will be applied to the model. Different 
    /// regularization types have different characteristics and are suitable for different types of problems. 
    /// L1 regularization (Lasso) adds a penalty equal to the absolute value of the magnitude of coefficients, 
    /// which can lead to sparse models by forcing some coefficients to be exactly zero. L2 regularization (Ridge) 
    /// adds a penalty equal to the square of the magnitude of coefficients, which results in all coefficients 
    /// being small but non-zero. Elastic Net combines both L1 and L2 regularization, offering a balance between 
    /// the two approaches. Setting this property to None disables regularization entirely.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls what kind of regularization technique to use.
    /// 
    /// The options are:
    /// - None: No regularization applied (default)
    /// - L1 (Lasso): Good when you suspect many features are irrelevant
    /// - L2 (Ridge): Good for most problems, keeps all features but reduces their impact
    /// - ElasticNet: A combination of L1 and L2, offering benefits of both
    /// 
    /// When to use each type:
    /// - None: When you have very little data or when overfitting isn't a concern
    /// - L1: When you want feature selection (automatically identifying important features)
    /// - L2: When you want to keep all features but prevent any from having too much influence
    /// - ElasticNet: When you want some feature selection but also want to keep related features together
    /// 
    /// For beginners, L2 (Ridge) regularization is often a good starting point as it's less aggressive
    /// and easier to tune than L1.
    /// </para>
    /// </remarks>
    public RegularizationType Type { get; set; } = RegularizationType.None;

    /// <summary>
    /// Gets or sets the strength of the regularization penalty.
    /// </summary>
    /// <value>A double value representing regularization strength, defaulting to 0.0 (no regularization).</value>
    /// <remarks>
    /// <para>
    /// This property controls how strongly the regularization penalty is applied to the model. Higher values 
    /// result in stronger regularization, which means the model will be simpler but might have higher bias. 
    /// Lower values result in weaker regularization, allowing the model to be more complex and potentially 
    /// fit the training data better. The optimal regularization strength depends on the specific dataset and 
    /// problem, and is often determined through techniques like cross-validation. A value of 0.0 effectively 
    /// disables regularization, regardless of the regularization type selected.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how strongly to apply the regularization.
    /// 
    /// Think of it like adjusting the tightness of training wheels:
    /// - 0.0 (default): No regularization effect at all
    /// - Small values (0.001 - 0.01): Gentle regularization that slightly discourages complexity
    /// - Medium values (0.01 - 0.1): Moderate regularization that significantly reduces overfitting
    /// - Large values (0.1 - 1.0): Strong regularization that greatly simplifies the model
    /// - Very large values (>1.0): Extreme regularization that might oversimplify the model
    /// 
    /// Finding the right value often requires experimentation:
    /// - Too low: Your model might still overfit
    /// - Too high: Your model might be too simple and underfit (not capture important patterns)
    /// - Just right: Your model generalizes well to new data
    /// 
    /// A common approach is to try several values (like 0.001, 0.01, 0.1, 1.0) and choose the one
    /// that performs best on validation data.
    /// </para>
    /// </remarks>
    public double Strength { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets the mixing ratio between L1 and L2 regularization when using Elastic Net.
    /// </summary>
    /// <value>A value between 0 and 1 representing the ratio, defaulting to 0.5 (equal mix).</value>
    /// <remarks>
    /// <para>
    /// This property is only relevant when the regularization Type is set to ElasticNet. It determines the 
    /// balance between L1 and L2 regularization in the Elastic Net approach. A value of 0 corresponds to 
    /// pure L2 regularization, while a value of 1 corresponds to pure L1 regularization. Values between 0 
    /// and 1 represent a mix of both approaches. The default value of 0.5 gives equal weight to both L1 and 
    /// L2 penalties. Adjusting this ratio allows fine-tuning of the regularization behavior to best suit the 
    /// specific characteristics of the dataset and problem.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls the mix between L1 and L2 regularization when using Elastic Net.
    /// 
    /// This only matters when Type is set to ElasticNet. The value represents:
    /// - 0.0: Pure L2 regularization (Ridge)
    /// - 0.5 (default): Equal mix of L1 and L2
    /// - 1.0: Pure L1 regularization (Lasso)
    /// - Any value between: A proportional mix
    /// 
    /// When to adjust this:
    /// - Move toward 0 (more L2) when you want to keep most features but reduce their impact
    /// - Move toward 1 (more L1) when you want more aggressive feature selection
    /// - Keep at 0.5 when you're not sure which approach is better
    /// 
    /// For example, if you have many related features (like different measurements of the same thing),
    /// a value closer to 0 might work better. If you have many potentially irrelevant features,
    /// a value closer to 1 might work better.
    /// </para>
    /// </remarks>
    public double L1Ratio { get; set; } = 0.5;
}
