namespace AiDotNet.Evaluation.Enums;

/// <summary>
/// Specifies the averaging method for multi-class/multi-label classification metrics.
/// </summary>
/// <remarks>
/// <para>
/// When computing metrics like precision, recall, or F1-score for multi-class problems,
/// there are different ways to aggregate per-class values into a single number.
/// </para>
/// <para>
/// <b>For Beginners:</b> Imagine you have a model that classifies images into 3 categories:
/// cats, dogs, and birds. Each category has its own precision/recall. The averaging method
/// determines how to combine these into a single score:
/// <list type="bullet">
/// <item><b>Micro:</b> Treat all samples equally (good when classes are balanced)</item>
/// <item><b>Macro:</b> Treat all classes equally (good for imbalanced data)</item>
/// <item><b>Weighted:</b> Weight by class frequency (compromise between micro/macro)</item>
/// </list>
/// </para>
/// </remarks>
public enum AveragingMethod
{
    /// <summary>
    /// Micro-averaging: Aggregate contributions of all classes to compute the metric.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Count total true positives, false positives, and false negatives
    /// across ALL classes, then compute the metric. This treats each sample equally.</para>
    /// <para><b>When to use:</b> When all samples are equally important, regardless of class.</para>
    /// <para><b>Effect:</b> Classes with more samples have more influence on the final score.</para>
    /// <para><b>Example:</b> If class A has 90 samples and class B has 10 samples, micro-averaging
    /// will be dominated by class A's performance.</para>
    /// </remarks>
    Micro = 0,

    /// <summary>
    /// Macro-averaging: Compute metric for each class, then take unweighted mean.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Calculate precision/recall/F1 separately for each class,
    /// then average them. This treats each class equally, regardless of size.</para>
    /// <para><b>When to use:</b> When all classes are equally important, even rare ones.</para>
    /// <para><b>Effect:</b> Each class contributes equally to the final score.</para>
    /// <para><b>Example:</b> If you have 3 classes with F1 scores of 0.9, 0.8, and 0.6,
    /// macro F1 = (0.9 + 0.8 + 0.6) / 3 = 0.767</para>
    /// </remarks>
    Macro = 1,

    /// <summary>
    /// Weighted-averaging: Compute metric for each class, then take weighted mean by support.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Like macro-averaging, but weight each class by its number
    /// of samples. A compromise between micro and macro averaging.</para>
    /// <para><b>When to use:</b> When you want to account for class imbalance but still
    /// see per-class contributions.</para>
    /// <para><b>Effect:</b> Larger classes have more influence, but it's more interpretable
    /// than micro-averaging since you compute per-class metrics first.</para>
    /// </remarks>
    Weighted = 2,

    /// <summary>
    /// Samples-averaging: For multi-label classification, compute metric for each sample
    /// and average across samples.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> In multi-label problems (where each sample can have
    /// multiple labels), compute the metric for each sample individually, then average.
    /// This is specific to multi-label scenarios.</para>
    /// <para><b>When to use:</b> Multi-label classification only.</para>
    /// <para><b>Example:</b> Document tagging where each document can have tags like
    /// "sports", "politics", "technology" simultaneously.</para>
    /// </remarks>
    Samples = 3,

    /// <summary>
    /// No averaging: Return a score for each class individually.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Don't combine the scores at all. Get a separate metric
    /// value for each class so you can analyze performance per-class.</para>
    /// <para><b>When to use:</b> When you need detailed per-class analysis.</para>
    /// </remarks>
    None = 4,

    /// <summary>
    /// Binary: Only report results for the positive class (class 1).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> For binary classification, only compute the metric
    /// for the positive class (e.g., "has disease" vs "healthy").</para>
    /// <para><b>When to use:</b> Binary classification when you care about the positive class.</para>
    /// </remarks>
    Binary = 5
}
