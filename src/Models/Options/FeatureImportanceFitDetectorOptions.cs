namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the Feature Importance Fit Detector, which analyzes how different input features
/// contribute to a model's predictions and evaluates potential issues with model fit.
/// </summary>
/// <remarks>
/// <para>
/// Feature importance analysis helps identify which input variables have the strongest influence on model predictions.
/// This detector uses permutation importance (randomly shuffling feature values and measuring the impact on predictions)
/// to assess feature relevance and detect potential issues like overfitting, underfitting, or redundant features.
/// </para>
/// <para><b>For Beginners:</b> Think of this as a tool that helps you understand which of your input data points
/// actually matter for making predictions. For example, if you're predicting house prices, this would tell you whether
/// square footage, number of bedrooms, or neighborhood has the biggest impact on price predictions. It also helps
/// identify potential problems with your model, like whether it's focusing too much on unimportant details or not
/// capturing important patterns. The options below let you adjust how sensitive this analysis should be.</para>
/// </remarks>
public class FeatureImportanceFitDetectorOptions
{
    /// <summary>
    /// Gets or sets the threshold for considering feature importance as high.
    /// </summary>
    /// <value>The high importance threshold, defaulting to 0.1 (10%).</value>
    /// <remarks>
    /// <para>
    /// Features with importance scores above this threshold are considered highly influential on the model's predictions.
    /// The importance score represents how much the model's performance decreases when the feature values are randomly shuffled.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines when a feature is considered "very important" to your model.
    /// With the default value of 0.1, any feature that, when randomized, causes your model's accuracy to drop by 10% or more
    /// is considered highly important. Think of it like identifying the key players on a sports team - these are the features
    /// that your model relies on heavily to make good predictions. If you want to be more selective about what counts as
    /// important, you could increase this value (e.g., to 0.15 or 0.2).</para>
    /// </remarks>
    public double HighImportanceThreshold { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the threshold for considering feature importance as low.
    /// </summary>
    /// <value>The low importance threshold, defaulting to 0.01 (1%).</value>
    /// <remarks>
    /// <para>
    /// Features with importance scores below this threshold are considered to have minimal influence on the model's predictions.
    /// These features might be candidates for removal to simplify the model without significant loss of performance.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines when a feature is considered "not important" to your model.
    /// With the default value of 0.01, any feature that, when randomized, causes your model's accuracy to drop by less than 1%
    /// is considered unimportant. These are like bench players who rarely affect the outcome of a game. Identifying these
    /// features can help you simplify your model by removing inputs that don't contribute much. If you want to be more
    /// aggressive about removing features, you could increase this threshold (e.g., to 0.02 or 0.03).</para>
    /// </remarks>
    public double LowImportanceThreshold { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the threshold for considering importance variance as low.
    /// </summary>
    /// <value>The low variance threshold, defaulting to 0.05 (5%).</value>
    /// <remarks>
    /// <para>
    /// This threshold determines when the variance (inconsistency) in a feature's importance scores across multiple
    /// permutations is considered low. Low variance suggests that the feature's importance is stable and reliable.
    /// </para>
    /// <para><b>For Beginners:</b> This setting helps determine when a feature's importance is consistent and reliable.
    /// The system calculates importance multiple times (see NumPermutations), and this threshold checks how much the
    /// results vary. With the default value of 0.05, if the importance scores vary by less than 5% across calculations,
    /// the feature's importance is considered stable. Think of it like measuring a runner's race times - if they always
    /// finish within a few seconds of the same time, their performance is consistent. Features with low variance are
    /// ones you can confidently say are either important or unimportant to your model.</para>
    /// </remarks>
    public double LowVarianceThreshold { get; set; } = 0.05;

    /// <summary>
    /// Gets or sets the threshold for considering importance variance as high.
    /// </summary>
    /// <value>The high variance threshold, defaulting to 0.2 (20%).</value>
    /// <remarks>
    /// <para>
    /// This threshold determines when the variance in a feature's importance scores across multiple permutations
    /// is considered high. High variance suggests that the feature's importance is unstable and might indicate
    /// complex interactions or potential overfitting.
    /// </para>
    /// <para><b>For Beginners:</b> This setting helps identify features whose importance is inconsistent or unreliable.
    /// With the default value of 0.2, if a feature's importance scores vary by more than 20% across calculations,
    /// its importance is considered unstable. Continuing the runner analogy, this would be like a runner whose race
    /// times are all over the place - sometimes fast, sometimes slow. Features with high variance might indicate
    /// complex relationships in your data or potential problems with your model. They warrant closer investigation
    /// since the model's reliance on them is unpredictable.</para>
    /// </remarks>
    public double HighVarianceThreshold { get; set; } = 0.2;

    /// <summary>
    /// Gets or sets the threshold for considering features as correlated.
    /// </summary>
    /// <value>The correlation threshold, defaulting to 0.7 (70%).</value>
    /// <remarks>
    /// <para>
    /// Features with correlation coefficients above this threshold (in absolute value) are considered strongly correlated.
    /// Highly correlated features often provide redundant information, and removing some of them might simplify the model
    /// without significant loss of performance.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines when two features are considered to contain similar information.
    /// With the default value of 0.7, if two features have a correlation of 70% or higher, they're considered strongly related.
    /// For example, in housing data, square footage and number of rooms might be highly correlated because larger houses
    /// tend to have more rooms. Including both features might not add much value compared to just using one of them.
    /// Identifying correlated features can help you simplify your model by removing redundant inputs. A higher threshold
    /// (like 0.8 or 0.9) would only flag the most extremely correlated features.</para>
    /// </remarks>
    public double CorrelationThreshold { get; set; } = 0.7;

    /// <summary>
    /// Gets or sets the threshold for the ratio of uncorrelated feature pairs to consider features as mostly uncorrelated.
    /// </summary>
    /// <value>The uncorrelated ratio threshold, defaulting to 0.8 (80%).</value>
    /// <remarks>
    /// <para>
    /// This threshold determines when the overall feature set is considered to have low correlation. If the proportion of
    /// feature pairs with correlation below the CorrelationThreshold exceeds this value, the feature set is considered
    /// mostly uncorrelated, which is generally desirable.
    /// </para>
    /// <para><b>For Beginners:</b> This setting helps evaluate whether your overall set of features is diverse or redundant.
    /// With the default value of 0.8, if at least 80% of all possible feature pairs have low correlation (below the
    /// CorrelationThreshold), then your feature set is considered diverse with minimal redundancy. This is generally good
    /// because it means each feature is contributing unique information. If this threshold isn't met, it suggests many of
    /// your features contain overlapping information, and you might benefit from feature selection or dimensionality
    /// reduction techniques to simplify your model.</para>
    /// </remarks>
    public double UncorrelatedRatioThreshold { get; set; } = 0.8;

    /// <summary>
    /// Gets or sets the random seed for feature permutation.
    /// </summary>
    /// <value>The random seed, defaulting to 42.</value>
    /// <remarks>
    /// <para>
    /// The random seed ensures reproducibility when shuffling feature values during permutation importance calculation.
    /// Using the same seed value will produce the same random shuffles each time the analysis is run.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls the randomization process used when calculating feature importance.
    /// The specific value (42) doesn't matter much, but keeping it constant ensures you get the same results each time you
    /// run the analysis. This is important for reproducibility - like setting a starting point for a random number generator.
    /// You generally don't need to change this unless you want to verify that your results are stable across different
    /// randomizations, in which case you might run the analysis multiple times with different seed values.</para>
    /// </remarks>
    public int RandomSeed { get; set; } = 42;

    /// <summary>
    /// Gets or sets the number of permutations to perform for each feature when calculating importance.
    /// </summary>
    /// <value>The number of permutations, defaulting to 5.</value>
    /// <remarks>
    /// <para>
    /// This parameter determines how many times each feature is randomly shuffled to calculate its importance.
    /// More permutations provide more stable importance estimates but increase computation time.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines how many times the system calculates each feature's importance.
    /// With the default value of 5, the system will shuffle each feature's values 5 different times and measure the impact
    /// on predictions each time, then average the results. More permutations (like 10 or 20) give more reliable importance
    /// scores but take longer to calculate. Think of it like taking multiple measurements to get a more accurate average -
    /// more measurements generally mean more confidence in the result, but at the cost of more time and effort.</para>
    /// </remarks>
    public int NumPermutations { get; set; } = 5;
}
