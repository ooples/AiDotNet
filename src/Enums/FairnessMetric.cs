namespace AiDotNet.Enums;

/// <summary>
/// Defines metrics for measuring and ensuring model fairness across different groups.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Fairness metrics help ensure AI models treat different groups of people 
/// equitably. These metrics measure various aspects of fairness to prevent discrimination in 
/// model predictions.
/// </para>
/// </remarks>
public enum FairnessMetric
{
    /// <summary>
    /// No fairness metric applied.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> No fairness checking - the model operates without considering whether 
    /// it treats different groups fairly.
    /// </remarks>
    None,

    /// <summary>
    /// Demographic parity (statistical parity).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Ensures equal acceptance rates across groups - like making sure a 
    /// loan approval model approves the same percentage from each demographic group.
    /// </remarks>
    DemographicParity,

    /// <summary>
    /// Equal opportunity for positive outcomes.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Ensures qualified individuals from all groups have equal chance of 
    /// positive outcomes - focuses on fairness among those who deserve the positive outcome.
    /// </remarks>
    EqualOpportunity,

    /// <summary>
    /// Equalized odds across groups.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Ensures both true positive and false positive rates are equal across 
    /// groups - balances fairness for both positive and negative outcomes.
    /// </remarks>
    EqualizedOdds,

    /// <summary>
    /// Disparate impact ratio.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Measures the ratio of positive outcomes between groups - commonly 
    /// used in legal contexts (often requiring at least 80% ratio).
    /// </remarks>
    DisparateImpact,

    /// <summary>
    /// Individual fairness (similar treatment).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Similar individuals should receive similar predictions - like ensuring 
    /// two people with identical qualifications get the same result.
    /// </remarks>
    IndividualFairness,

    /// <summary>
    /// Counterfactual fairness.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Predictions shouldn't change if only sensitive attributes (like race 
    /// or gender) were different - tests "what if" scenarios.
    /// </remarks>
    CounterfactualFairness,

    /// <summary>
    /// Calibration across groups.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Ensures prediction probabilities mean the same thing for all groups - 
    /// if model says 70% chance, it should be accurate for everyone.
    /// </remarks>
    Calibration,

    /// <summary>
    /// Treatment equality.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Ensures the ratio of false positives to false negatives is equal 
    /// across groups - balances different types of errors fairly.
    /// </remarks>
    TreatmentEquality,

    /// <summary>
    /// Conditional statistical parity.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Like demographic parity but accounts for legitimate factors - ensures 
    /// fairness among people with similar relevant characteristics.
    /// </remarks>
    ConditionalStatisticalParity,

    /// <summary>
    /// Balance for positive/negative class.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Ensures average scores for positive and negative outcomes are consistent 
    /// across groups - prevents systematic score differences.
    /// </remarks>
    Balance,

    /// <summary>
    /// Fairness through awareness.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Uses sensitive attributes explicitly to ensure fairness - sometimes 
    /// knowing group membership helps create fairer outcomes.
    /// </remarks>
    FairnessThroughAwareness,

    /// <summary>
    /// Fairness through unawareness.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Removes sensitive attributes from model inputs - simple approach but 
    /// may not prevent indirect discrimination.
    /// </remarks>
    FairnessThroughUnawareness,

    /// <summary>
    /// Rawlsian fairness (maximin).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Maximizes outcomes for the worst-off group - based on philosopher 
    /// John Rawls' theory of justice.
    /// </remarks>
    RawlsianFairness,

    /// <summary>
    /// Intersectional fairness.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Considers fairness across combinations of groups - recognizes that 
    /// people belong to multiple groups simultaneously (e.g., race AND gender).
    /// </remarks>
    IntersectionalFairness,

    /// <summary>
    /// Custom fairness metric.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Allows you to define your own fairness criteria specific to your 
    /// application's ethical requirements.
    /// </remarks>
    Custom
}