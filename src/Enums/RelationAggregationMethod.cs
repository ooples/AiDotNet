namespace AiDotNet.Enums;

/// <summary>
/// Methods for aggregating multiple relation scores in Relation Networks.
/// </summary>
/// <remarks>
/// <para>
/// When there are multiple support examples per class, we need a way to combine
/// the relation scores from comparing a query with each support example.
/// </para>
/// <para><b>For Beginners:</b> In few-shot learning, each class has several example
/// images. When classifying a new query image, we compare it to ALL examples of each
/// class. This enum controls how those multiple similarity scores are combined into
/// a single score for each class.
///
/// For example, if we have 5 dog examples and compare a query to each:
/// - Mean: Average all 5 scores
/// - Max: Take the highest score (most similar dog example)
/// - Attention: Weight scores by relevance
/// - LearnedWeighting: Let the network learn optimal weights
/// </para>
/// </remarks>
public enum RelationAggregationMethod
{
    /// <summary>
    /// Compute mean of all scores.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Simply averages all relation scores for a class.
    /// </para>
    /// <para><b>For Beginners:</b> This is the simplest approach - just average
    /// all the similarity scores. If a query is similar to most examples of a class,
    /// it will get a high average score for that class.
    /// </para>
    /// </remarks>
    Mean,

    /// <summary>
    /// Use maximum score.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Takes the highest relation score among all support examples.
    /// </para>
    /// <para><b>For Beginners:</b> This picks the best match. If a query looks
    /// very similar to even ONE example of a class, that class gets a high score.
    /// Useful when class examples are diverse.
    /// </para>
    /// </remarks>
    Max,

    /// <summary>
    /// Use attention-weighted average.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Computes attention weights to weight each support example's contribution.
    /// </para>
    /// <para><b>For Beginners:</b> This gives more weight to more relevant examples.
    /// If some support examples are more similar to the query, their scores count more.
    /// </para>
    /// </remarks>
    Attention,

    /// <summary>
    /// Use learned weighting.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Learns a neural network to weight the contribution of each support example.
    /// </para>
    /// <para><b>For Beginners:</b> Instead of using fixed rules (mean, max), the
    /// network learns the best way to combine scores during training. This is the
    /// most flexible but requires more data to learn the weighting.
    /// </para>
    /// </remarks>
    LearnedWeighting
}
