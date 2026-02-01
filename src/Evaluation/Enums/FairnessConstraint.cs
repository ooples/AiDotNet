namespace AiDotNet.Evaluation.Enums;

/// <summary>
/// Specifies the fairness constraint or metric for model evaluation.
/// </summary>
/// <remarks>
/// <para>
/// Fairness metrics measure whether a model treats different groups equitably.
/// Different definitions of "fair" may conflict - choose based on your context and values.
/// </para>
/// <para>
/// <b>For Beginners:</b> Imagine a loan approval model. "Fair" could mean:
/// <list type="bullet">
/// <item>Same approval rate across groups (demographic parity)</item>
/// <item>Same accuracy across groups (equalized odds)</item>
/// <item>Same outcomes for similar people (individual fairness)</item>
/// </list>
/// These definitions often conflict - you can't optimize for all simultaneously.
/// Choose the one that aligns with your ethical and legal requirements.
/// </para>
/// </remarks>
public enum FairnessConstraint
{
    /// <summary>
    /// Demographic parity: Equal positive prediction rates across groups.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each group should receive positive predictions at the same rate.
    /// If 30% of group A gets approved, 30% of group B should too.</para>
    /// <para><b>Formula:</b> P(Ŷ=1|A=a) = P(Ŷ=1|A=b) for all groups a, b</para>
    /// <para><b>When to use:</b> When equal treatment is the primary concern.</para>
    /// <para><b>Limitation:</b> May require different accuracy across groups.</para>
    /// </remarks>
    DemographicParity = 0,

    /// <summary>
    /// Equalized odds: Equal TPR and FPR across groups.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The model should be equally accurate for each group.
    /// Both the true positive rate and false positive rate should be the same across groups.</para>
    /// <para><b>Formula:</b> P(Ŷ=1|Y=y,A=a) = P(Ŷ=1|Y=y,A=b) for y ∈ {0,1}</para>
    /// <para><b>When to use:</b> When accuracy should be equal regardless of group.</para>
    /// </remarks>
    EqualizedOdds = 1,

    /// <summary>
    /// Equal opportunity: Equal TPR (true positive rate) across groups.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Among people who deserve a positive outcome,
    /// each group should have an equal chance of getting it. A relaxation of equalized odds.</para>
    /// <para><b>Formula:</b> P(Ŷ=1|Y=1,A=a) = P(Ŷ=1|Y=1,A=b)</para>
    /// <para><b>When to use:</b> When equal benefit to qualified individuals matters most.</para>
    /// </remarks>
    EqualOpportunity = 2,

    /// <summary>
    /// Predictive parity: Equal PPV (precision) across groups.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When the model predicts positive, it should be equally
    /// reliable for all groups. Same precision across groups.</para>
    /// <para><b>Formula:</b> P(Y=1|Ŷ=1,A=a) = P(Y=1|Ŷ=1,A=b)</para>
    /// <para><b>When to use:</b> When you want predictions to mean the same thing for everyone.</para>
    /// </remarks>
    PredictiveParity = 3,

    /// <summary>
    /// Calibration: Equal probability calibration across groups.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When the model says "80% probability", it should be
    /// right 80% of the time for all groups, not just overall.</para>
    /// <para><b>When to use:</b> When predicted probabilities are used for decisions.</para>
    /// </remarks>
    Calibration = 4,

    /// <summary>
    /// Treatment equality: Equal ratio of FN to FP across groups.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The ratio of false negatives to false positives should
    /// be the same across groups. Balances the two types of errors equally.</para>
    /// <para><b>When to use:</b> When both error types have equal importance.</para>
    /// </remarks>
    TreatmentEquality = 5,

    /// <summary>
    /// Balance for positive class: Equal TPR across groups (same as equal opportunity).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Among actual positive cases, prediction rates should
    /// be equal across groups. Same as equal opportunity under a different name.</para>
    /// </remarks>
    BalancePositive = 6,

    /// <summary>
    /// Balance for negative class: Equal TNR across groups.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Among actual negative cases, correct rejection rates
    /// should be equal across groups.</para>
    /// </remarks>
    BalanceNegative = 7,

    /// <summary>
    /// Conditional demographic parity: Equal prediction rates within strata.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Like demographic parity, but controlling for
    /// legitimate factors. Equal rates among people with similar qualifications.</para>
    /// <para><b>When to use:</b> When you have legitimate factors that explain differences.</para>
    /// </remarks>
    ConditionalDemographicParity = 8,

    /// <summary>
    /// Counterfactual fairness: Same prediction if group membership changed.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Would the prediction be the same if this person
    /// had been in a different group? Uses causal reasoning.</para>
    /// <para><b>When to use:</b> When you have causal models of the domain.</para>
    /// <para><b>Research reference:</b> Kusner et al. (2017)</para>
    /// </remarks>
    CounterfactualFairness = 9,

    /// <summary>
    /// Individual fairness: Similar individuals receive similar predictions.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> People who are similar (except for group membership)
    /// should get similar predictions. Requires defining "similar".</para>
    /// <para><b>When to use:</b> When you can define a meaningful similarity metric.</para>
    /// <para><b>Research reference:</b> Dwork et al. (2012)</para>
    /// </remarks>
    IndividualFairness = 10,

    /// <summary>
    /// Disparate impact ratio: Ratio of positive rates between groups.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The ratio of approval rates between groups.
    /// The "80% rule" in US law says this should be at least 0.8.</para>
    /// <para><b>Formula:</b> P(Ŷ=1|A=minority) / P(Ŷ=1|A=majority)</para>
    /// <para><b>When to use:</b> Legal compliance, simple fairness auditing.</para>
    /// </remarks>
    DisparateImpact = 11,

    /// <summary>
    /// Average odds difference: Average of TPR and FPR differences.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> A single number summarizing the equalized odds violation.
    /// Average of the TPR difference and FPR difference between groups.</para>
    /// </remarks>
    AverageOddsDifference = 12,

    /// <summary>
    /// Statistical parity difference: Difference in positive prediction rates.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Simply the difference in approval rates between groups.
    /// Zero means perfect demographic parity.</para>
    /// </remarks>
    StatisticalParityDifference = 13,

    /// <summary>
    /// Theil index: Inequality measure from economics applied to predictions.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> An entropy-based measure of inequality borrowed from
    /// economics. Can be decomposed into between-group and within-group inequality.</para>
    /// </remarks>
    TheilIndex = 14,

    /// <summary>
    /// Between-group generalized entropy: Measures fairness across groups.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Measures how much of the total inequality in predictions
    /// is due to differences between groups vs within groups.</para>
    /// </remarks>
    BetweenGroupEntropy = 15,

    /// <summary>
    /// No fairness constraint applied.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Optimize purely for accuracy without fairness constraints.
    /// Use this as a baseline, but be aware of potential disparities.</para>
    /// </remarks>
    None = 16
}
