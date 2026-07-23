namespace AiDotNet.Evaluation.Enums;

/// <summary>
/// Specifies the method for selecting classification thresholds.
/// </summary>
/// <remarks>
/// <para>
/// Classification models often output probabilities (0.0 to 1.0). A threshold determines
/// when to classify as positive vs negative. The default 0.5 isn't always optimal.
/// </para>
/// <para>
/// <b>For Beginners:</b> If your model says "70% chance of fraud", should you flag it as fraud?
/// That depends on your threshold. At 0.5 threshold, yes (70% > 50%). At 0.8 threshold, no (70% &lt; 80%).
/// Different methods help you pick the best threshold for your needs.
/// </para>
/// </remarks>
public enum ThresholdSelectionMethod
{
    /// <summary>
    /// Youden's J statistic: Maximizes sensitivity + specificity - 1.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Finds the threshold where the model best balances
    /// finding true positives and avoiding false positives. Visualized as the point
    /// on the ROC curve furthest from the diagonal line.</para>
    /// <para><b>When to use:</b> General-purpose threshold selection when costs are equal.</para>
    /// <para><b>Formula:</b> J = Sensitivity + Specificity - 1</para>
    /// </remarks>
    Youden = 0,

    /// <summary>
    /// F1-score maximization: Selects threshold that maximizes F1 score.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Finds the threshold that gives the best balance between
    /// precision (how many positives are correct) and recall (how many positives are found).</para>
    /// <para><b>When to use:</b> When you care about both precision and recall equally.</para>
    /// </remarks>
    F1Max = 1,

    /// <summary>
    /// F-beta maximization: Maximizes F-beta score with configurable beta.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Like F1, but you can weight recall more (beta > 1) or
    /// precision more (beta &lt; 1). F2 emphasizes recall, F0.5 emphasizes precision.</para>
    /// <para><b>When to use:</b> When recall and precision have different importance.</para>
    /// </remarks>
    FBetaMax = 2,

    /// <summary>
    /// Cost-sensitive: Minimizes expected cost based on false positive/negative costs.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Specify the cost of false positives vs false negatives,
    /// and find the threshold that minimizes total cost.</para>
    /// <para><b>When to use:</b> When you know the business cost of different errors.</para>
    /// <para><b>Example:</b> In fraud detection, missing fraud (FN) might cost $1000,
    /// while blocking legitimate transaction (FP) might cost $10.</para>
    /// </remarks>
    CostSensitive = 3,

    /// <summary>
    /// Fixed sensitivity: Choose threshold to achieve a target sensitivity (recall).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Set a minimum recall you need (e.g., "we must catch
    /// at least 95% of fraud cases") and find the threshold that achieves it.</para>
    /// <para><b>When to use:</b> When missing positives is critical (medical screening).</para>
    /// </remarks>
    FixedSensitivity = 4,

    /// <summary>
    /// Fixed specificity: Choose threshold to achieve a target specificity.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Set a minimum specificity you need (e.g., "we must
    /// avoid false alarms at least 99% of the time") and find the threshold.</para>
    /// <para><b>When to use:</b> When false positives are costly (spam filtering).</para>
    /// </remarks>
    FixedSpecificity = 5,

    /// <summary>
    /// Fixed precision: Choose threshold to achieve a target precision (PPV).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Set a minimum precision you need (e.g., "when we say
    /// positive, we should be right at least 90% of the time").</para>
    /// <para><b>When to use:</b> When acting on positives is expensive.</para>
    /// </remarks>
    FixedPrecision = 6,

    /// <summary>
    /// Fixed NPV: Choose threshold to achieve a target negative predictive value.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Set a minimum NPV (e.g., "when we say negative,
    /// we should be right at least 99% of the time").</para>
    /// <para><b>When to use:</b> When negative predictions must be reliable.</para>
    /// </remarks>
    FixedNPV = 7,

    /// <summary>
    /// Precision-recall breakeven: Where precision equals recall.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Find the threshold where precision and recall are equal.
    /// A simple way to balance the two metrics.</para>
    /// <para><b>When to use:</b> Quick balanced threshold without parameter tuning.</para>
    /// </remarks>
    PrecisionRecallBreakeven = 8,

    /// <summary>
    /// Geometric mean maximization: Maximizes sqrt(sensitivity * specificity).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Similar to Youden but uses geometric mean instead
    /// of arithmetic sum. Better for imbalanced datasets.</para>
    /// <para><b>When to use:</b> Imbalanced classification problems.</para>
    /// </remarks>
    GeometricMean = 9,

    /// <summary>
    /// Matthews Correlation Coefficient maximization: Maximizes MCC.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> MCC considers all four quadrants of the confusion matrix
    /// and works well even with imbalanced classes. Ranges from -1 to +1.</para>
    /// <para><b>When to use:</b> Imbalanced datasets, when you want a balanced single metric.</para>
    /// </remarks>
    MCCMax = 10,

    /// <summary>
    /// Closest to (0,1) on ROC: Point on ROC curve closest to perfect classification.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Find the threshold where the ROC point is closest to
    /// the perfect corner (100% sensitivity, 100% specificity).</para>
    /// <para><b>When to use:</b> Alternative to Youden for balanced optimization.</para>
    /// </remarks>
    ClosestToTopLeft = 11,

    /// <summary>
    /// Prevalence-adjusted: Adjusts threshold based on class prevalence.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> If positives are rare (e.g., 1% disease prevalence),
    /// this adjusts the threshold to account for the base rate.</para>
    /// <para><b>When to use:</b> When test set prevalence differs from deployment.</para>
    /// </remarks>
    PrevalenceAdjusted = 12,

    /// <summary>
    /// Informedness maximization: Maximizes informedness (same as Youden's J).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Informedness = Sensitivity + Specificity - 1.
    /// Same as Youden's J but under a different name in some literature.</para>
    /// <para><b>When to use:</b> Same as Youden.</para>
    /// </remarks>
    Informedness = 13,

    /// <summary>
    /// Markedness maximization: Maximizes PPV + NPV - 1.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Focuses on how reliable your predictions are
    /// (both positive and negative predictions).</para>
    /// <para><b>When to use:</b> When prediction reliability is the primary concern.</para>
    /// </remarks>
    Markedness = 14,

    /// <summary>
    /// Default threshold of 0.5.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The standard threshold - classify as positive if
    /// probability > 50%. Simple but often not optimal.</para>
    /// <para><b>When to use:</b> When you have no specific requirements.</para>
    /// </remarks>
    Default = 15
}
