namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration for handling missing features in vertical federated learning.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> In real-world VFL deployments, not all parties have data for
/// all entities. For example, a bank might have records for 100,000 customers, but the
/// partnering hospital only has records for 30,000 of those patients. The other 70,000
/// customers have "missing" hospital features.</para>
///
/// <para>This class controls how those missing features are handled during training
/// and inference.</para>
///
/// <para>Example:
/// <code>
/// var missingOptions = new MissingFeatureOptions
/// {
///     Strategy = MissingFeatureStrategy.Mean,
///     MinimumOverlapRatio = 0.3,
///     AlignmentThreshold = 0.8
/// };
/// </code>
/// </para>
/// </remarks>
public class MissingFeatureOptions
{
    /// <summary>
    /// Gets or sets the strategy for imputing missing features.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> "Skip" is safest (only uses fully-aligned data),
    /// "Mean" is a good default (fills in averages), "Learned" is most accurate but complex.</para>
    /// </remarks>
    public MissingFeatureStrategy Strategy { get; set; } = MissingFeatureStrategy.Skip;

    /// <summary>
    /// Gets or sets the minimum required overlap ratio between parties before training
    /// can proceed. A ratio of 0.3 means at least 30% of the smaller party's entities
    /// must be present in all parties.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> If the overlap is too small, VFL training is unlikely to
    /// produce a good model because there aren't enough fully-aligned samples to learn from.
    /// This threshold prevents wasting resources on insufficient data.</para>
    /// </remarks>
    public double MinimumOverlapRatio { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the alignment confidence threshold when using fuzzy entity matching.
    /// Only fuzzy matches with confidence above this threshold are used for training.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When entity IDs don't match exactly (typos, formatting),
    /// fuzzy matching assigns a confidence score (0.0 to 1.0). This threshold filters out
    /// low-confidence matches. Set higher (e.g., 0.9) for critical applications.</para>
    /// </remarks>
    public double AlignmentThreshold { get; set; } = 0.8;

    /// <summary>
    /// Gets or sets whether to include partially-aligned entities in training.
    /// If true, entities that exist in some (but not all) parties can be used,
    /// with missing features handled by the imputation strategy.
    /// </summary>
    public bool AllowPartialAlignment { get; set; }

    /// <summary>
    /// Gets or sets whether to create a binary indicator feature marking which
    /// features are imputed vs. observed. This helps the model learn to weight
    /// observed features more heavily.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Adding an indicator tells the model "this value was
    /// filled in, not actually observed" so it can learn to trust observed values more.</para>
    /// </remarks>
    public bool AddMissingnessIndicator { get; set; } = true;
}
