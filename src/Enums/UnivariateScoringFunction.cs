namespace AiDotNet.Enums;

/// <summary>
/// Defines the scoring functions available for univariate feature selection.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> These are different statistical tests that measure how much a single feature
/// is related to the target variable you're trying to predict.
/// </para>
/// <para>
/// Each scoring function is best suited for different types of data:
/// - Chi-Squared: Best for categorical features and categorical targets
/// - ANOVA F-Value: Best for continuous features and categorical targets
/// - Mutual Information: Works well for both categorical and continuous features with any target type
/// </para>
/// </remarks>
public enum UnivariateScoringFunction
{
    /// <summary>
    /// Uses the Chi-Squared test to measure the dependence between categorical features and the target.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Chi-Squared test compares the observed frequencies in your data
    /// with the frequencies you would expect if there was no relationship between the feature and target.
    /// </para>
    /// <para>
    /// Higher scores indicate stronger relationships. This test works best with categorical data
    /// (data that falls into distinct categories like "red", "green", "blue").
    /// </para>
    /// </remarks>
    ChiSquared,

    /// <summary>
    /// Uses ANOVA F-value to measure how well continuous features can distinguish between different classes.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> ANOVA (Analysis of Variance) F-value measures whether the means of different
    /// groups are significantly different from each other.
    /// </para>
    /// <para>
    /// For example, if you're predicting house prices in different neighborhoods, ANOVA can tell you
    /// if features like square footage vary significantly across neighborhoods. Higher F-values indicate
    /// that the feature is more useful for distinguishing between classes.
    /// </para>
    /// </remarks>
    FValue,

    /// <summary>
    /// Uses Mutual Information to measure how much knowing one feature tells you about the target.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Mutual Information measures how much information a feature provides about
    /// the target variable. It answers the question: "If I know the value of this feature, how much does
    /// that help me predict the target?"
    /// </para>
    /// <para>
    /// Higher scores mean the feature provides more information. Mutual Information works well with
    /// both categorical and continuous features, making it very versatile.
    /// </para>
    /// </remarks>
    MutualInformation
}
