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
    /// Weights each support example's relation by <c>softmax_s(h_s' U q)</c> within its class, a learned bilinear
    /// attention between the support embedding <c>h_s</c> and the query <c>q</c>. <c>U</c> starts at zero, where the
    /// weights are uniform - the mean. It used to fall back to the mean silently.
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
    /// Weights the k-th support example of every class by <c>exp(w_k)</c>, with <c>w</c> learned per shot position
    /// and starting at zero - the mean. It used to fall back to the mean silently.
    /// </para>
    /// <para><b>For Beginners:</b> Instead of using fixed rules (mean, max), the
    /// network learns the best way to combine scores during training. This is the
    /// most flexible but requires more data to learn the weighting.
    /// </para>
    /// </remarks>
    LearnedWeighting,

    /// <summary>
    /// Sum each class's support embeddings and compute one relation per class.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The paper's K-shot rule (Sung et al. 2018, section 3.2): "we element-wise sum over the embedding module
    /// outputs of all samples from each training class to form this class' feature map". One relation score per
    /// class, however many shots.
    /// </para>
    /// <para><b>For Beginners:</b> All of a class's examples are added together into one description of the class,
    /// and the query is compared with that.
    /// </para>
    /// </remarks>
    EmbeddingSum
}
