namespace AiDotNet.Evaluation.Enums;

/// <summary>
/// Specifies the normalization method for confusion matrices and metrics.
/// </summary>
/// <remarks>
/// <para>
/// Confusion matrix normalization helps interpret results, especially with imbalanced classes.
/// Different normalizations answer different questions.
/// </para>
/// <para>
/// <b>For Beginners:</b> A raw confusion matrix shows counts (e.g., "50 true positives").
/// Normalized versions show proportions, which are easier to interpret:
/// <list type="bullet">
/// <item><b>None:</b> Raw counts</item>
/// <item><b>ByTrue:</b> "What percentage of actual positives were correctly identified?"</item>
/// <item><b>ByPredicted:</b> "What percentage of predicted positives were correct?"</item>
/// <item><b>All:</b> "What percentage of all samples fall into each cell?"</item>
/// </list>
/// </para>
/// </remarks>
public enum NormalizationType
{
    /// <summary>
    /// No normalization: Raw counts in confusion matrix.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Shows the actual number of samples in each cell.
    /// Useful for understanding the absolute scale but hard to compare across datasets.</para>
    /// <para><b>When to use:</b> When absolute counts matter, debugging.</para>
    /// </remarks>
    None = 0,

    /// <summary>
    /// Normalize by true labels (rows): Each row sums to 1.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Shows what fraction of each actual class was predicted
    /// as each label. Row values show where true class samples ended up.</para>
    /// <para><b>Interpretation:</b> "Of all actual class X, what % were predicted as Y?"</para>
    /// <para><b>When to use:</b> To see recall/sensitivity for each class.</para>
    /// <para><b>Example:</b> If row "Cat" shows [0.8, 0.2], then 80% of actual cats
    /// were correctly identified, 20% were misclassified.</para>
    /// </remarks>
    ByTrue = 1,

    /// <summary>
    /// Normalize by predicted labels (columns): Each column sums to 1.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Shows what fraction of each predicted class actually
    /// belongs to each true label. Column values show where predictions came from.</para>
    /// <para><b>Interpretation:</b> "Of all predicted as X, what % were actually Y?"</para>
    /// <para><b>When to use:</b> To see precision/PPV for each class.</para>
    /// <para><b>Example:</b> If column "Cat" shows [0.9, 0.1], then 90% of cat predictions
    /// were correct, 10% were actually dogs.</para>
    /// </remarks>
    ByPredicted = 2,

    /// <summary>
    /// Normalize by total: All cells sum to 1.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Shows the proportion of all samples that fall into
    /// each cell. The entire matrix sums to 1.0 (100%).</para>
    /// <para><b>Interpretation:</b> "What % of all samples are in each cell?"</para>
    /// <para><b>When to use:</b> To see overall distribution of predictions and errors.</para>
    /// </remarks>
    All = 3,

    /// <summary>
    /// Min-max normalization: Scale values to [0, 1] range.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Transform values so the minimum becomes 0 and
    /// maximum becomes 1. Useful for comparing across different scales.</para>
    /// <para><b>Formula:</b> (x - min) / (max - min)</para>
    /// <para><b>When to use:</b> When comparing metrics with different ranges.</para>
    /// </remarks>
    MinMax = 4,

    /// <summary>
    /// Z-score (standard) normalization: Transform to mean=0, std=1.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Transform values so they have zero mean and unit
    /// standard deviation. Shows how many standard deviations from the mean.</para>
    /// <para><b>Formula:</b> (x - mean) / std</para>
    /// <para><b>When to use:</b> Statistical analysis, comparing distributions.</para>
    /// </remarks>
    ZScore = 5,

    /// <summary>
    /// Robust normalization: Use median and IQR instead of mean and std.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Like z-score but uses median and interquartile range,
    /// which are less affected by outliers.</para>
    /// <para><b>Formula:</b> (x - median) / IQR</para>
    /// <para><b>When to use:</b> When data has outliers.</para>
    /// </remarks>
    Robust = 6,

    /// <summary>
    /// Log normalization: Apply log transformation.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Apply logarithm to compress large ranges and make
    /// multiplicative relationships additive.</para>
    /// <para><b>When to use:</b> Highly skewed data, multiplicative effects.</para>
    /// </remarks>
    Log = 7
}
