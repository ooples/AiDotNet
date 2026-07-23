namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for ordinal classification (ordinal regression) models.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Ordinal classification handles categories with a natural order,
/// like ratings (1-5 stars), education levels, or pain severity. Unlike regular classification,
/// the order matters - predicting 5 stars when true is 4 stars is better than predicting 1 star.</para>
///
/// <para><b>Key differences from regular classification:</b>
/// <list type="bullet">
/// <item>Classes have a natural ordering (1 &lt; 2 &lt; 3 &lt; ...)</item>
/// <item>Errors near the true class are less severe than distant errors</item>
/// <item>Models often use cumulative probabilities P(Y ≤ k)</item>
/// </list>
/// </para>
///
/// <para><b>Common approaches:</b>
/// <list type="bullet">
/// <item><b>Threshold models:</b> Learn thresholds that separate ordered classes</item>
/// <item><b>Proportional odds:</b> Ordinal logistic regression</item>
/// <item><b>Regression-based:</b> Treat ordinal as continuous, round predictions</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[AiDotNet.Configuration.YamlConfigurable("OrdinalClassifier")]
public interface IOrdinalClassifier<T> : IClassifier<T>
{
    /// <summary>
    /// Gets the ordered class labels from lowest to highest.
    /// </summary>
    Vector<T>? OrderedClasses { get; }

    /// <summary>
    /// Predicts cumulative probabilities P(Y ≤ k) for each class threshold.
    /// </summary>
    /// <param name="features">Feature matrix [n_samples, n_features].</param>
    /// <returns>Cumulative probability matrix [n_samples, n_classes-1].</returns>
    Matrix<T> PredictCumulativeProbabilities(Matrix<T> features);

    /// <summary>
    /// Predicts ordinal class labels.
    /// </summary>
    /// <param name="features">Feature matrix.</param>
    /// <returns>Predicted ordinal class labels.</returns>
    new Vector<T> Predict(Matrix<T> features);
}
