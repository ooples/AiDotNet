using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the interface for causal inference models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Causal inference models estimate the causal effect of a treatment on an outcome,
/// distinguishing between correlation and causation in observational data.
/// </para>
/// <para>
/// <b>For Beginners:</b> Causal inference is about answering "what if" questions:
/// "What would happen to Y if we changed X?" This is different from prediction,
/// which asks "What is Y likely to be given X?"
///
/// Key concepts:
/// - Treatment (T): The intervention (e.g., taking a medication, receiving training)
/// - Outcome (Y): What we measure (e.g., health improvement, salary increase)
/// - Confounders (X): Variables affecting both treatment and outcome
/// - Counterfactual: What would have happened without treatment
///
/// The fundamental problem of causal inference is that we can never observe both
/// outcomes (with and without treatment) for the same individual. Causal models
/// use statistical methods to estimate what the counterfactual would have been.
///
/// Common applications:
/// - Medicine: Does a drug cause improvement? (not just correlation)
/// - Economics: Does education cause higher wages?
/// - Marketing: Does an ad campaign cause more sales?
/// - Policy: Does a new law cause behavior change?
///
/// References:
/// - Rubin, D. B. (1974). "Estimating causal effects of treatments"
/// - Pearl, J. (2009). "Causality: Models, Reasoning, and Inference"
/// </para>
/// </remarks>
public interface ICausalModel<T> : IFullModel<T, Matrix<T>, Vector<T>>
{
    /// <summary>
    /// Estimates the Average Treatment Effect (ATE) from the data.
    /// </summary>
    /// <param name="x">The covariate/confounder matrix.</param>
    /// <param name="treatment">Binary treatment indicator (1 = treated, 0 = control).</param>
    /// <param name="outcome">The observed outcomes.</param>
    /// <returns>The estimated ATE with standard error.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Average Treatment Effect is the average difference in
    /// outcomes between treated and untreated groups, adjusted for confounders.
    ///
    /// ATE = E[Y(1) - Y(0)] = Average outcome if everyone treated - Average outcome if no one treated
    ///
    /// This tells you: "On average, how much does the treatment change the outcome?"
    ///
    /// Example: If ATE = 5 for a job training program on salary:
    /// "On average, the training increases salary by $5,000"
    /// </para>
    /// </remarks>
    (T estimate, T standardError) EstimateATE(Matrix<T> x, Vector<int> treatment, Vector<T> outcome);

    /// <summary>
    /// Estimates the Average Treatment Effect on the Treated (ATT).
    /// </summary>
    /// <param name="x">The covariate/confounder matrix.</param>
    /// <param name="treatment">Binary treatment indicator (1 = treated, 0 = control).</param>
    /// <param name="outcome">The observed outcomes.</param>
    /// <returns>The estimated ATT with standard error.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> ATT focuses only on those who received treatment:
    /// "What was the effect for people who actually got treated?"
    ///
    /// ATT = E[Y(1) - Y(0) | T=1]
    ///
    /// This is useful when:
    /// - Treatment is selective (people choose to be treated)
    /// - You want to evaluate the effect for the actual treated population
    ///
    /// Example: For a voluntary job training program:
    /// "Among people who chose to take the training, how much did it help?"
    /// </para>
    /// </remarks>
    (T estimate, T standardError) EstimateATT(Matrix<T> x, Vector<int> treatment, Vector<T> outcome);

    /// <summary>
    /// Estimates the Conditional Average Treatment Effect (CATE) for each individual.
    /// </summary>
    /// <param name="x">The covariate/confounder matrix.</param>
    /// <param name="treatment">Binary treatment indicator (1 = treated, 0 = control).</param>
    /// <param name="outcome">The observed outcomes.</param>
    /// <returns>Vector of individual treatment effects.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> CATE tells you how the treatment effect varies across individuals
    /// based on their characteristics. Not everyone responds the same way to treatment!
    ///
    /// CATE(x) = E[Y(1) - Y(0) | X = x]
    ///
    /// This is useful for:
    /// - Personalized medicine: "Which patients will benefit most from this drug?"
    /// - Targeted marketing: "Which customers respond best to this promotion?"
    /// - Policy targeting: "Who benefits most from this program?"
    ///
    /// Example: A medication might have:
    /// - CATE = +10 for younger patients (large benefit)
    /// - CATE = +2 for older patients (small benefit)
    /// </para>
    /// </remarks>
    Vector<T> EstimateCATEPerIndividual(Matrix<T> x, Vector<int> treatment, Vector<T> outcome);

    /// <summary>
    /// Predicts the treatment effect for new individuals based on their covariates.
    /// </summary>
    /// <param name="x">The covariate matrix for new individuals.</param>
    /// <returns>Predicted treatment effects for each individual.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> After fitting the model, you can predict how a new person
    /// would respond to treatment based on their characteristics.
    ///
    /// This is the key to treatment personalization:
    /// "Based on this patient's age, health, etc., how much would they benefit?"
    /// </para>
    /// </remarks>
    Vector<T> PredictTreatmentEffect(Matrix<T> x);

    /// <summary>
    /// Estimates the propensity score (probability of receiving treatment) for each individual.
    /// </summary>
    /// <param name="x">The covariate matrix.</param>
    /// <returns>Propensity scores (probabilities) for each individual.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The propensity score is the probability that someone
    /// receives treatment, given their characteristics.
    ///
    /// e(x) = P(T=1 | X=x)
    ///
    /// Why it matters:
    /// - If treatment is not random, some people are more likely to be treated
    /// - Propensity scores help adjust for this selection bias
    /// - People with similar propensity scores are comparable regardless of actual treatment
    ///
    /// Example: In a job training study:
    /// - Young, unemployed → high propensity (likely to enroll)
    /// - Old, employed → low propensity (unlikely to enroll)
    /// </para>
    /// </remarks>
    Vector<T> EstimatePropensityScores(Matrix<T> x);

    /// <summary>
    /// Checks the overlap/positivity assumption by examining propensity score distributions.
    /// </summary>
    /// <param name="x">The covariate matrix.</param>
    /// <param name="treatment">Binary treatment indicator.</param>
    /// <returns>Overlap statistics (min/max propensity scores for each group).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> For causal inference to work, both treated and control groups
    /// must have some individuals at each level of covariates ("overlap").
    ///
    /// If propensity = 0 or 1 for some individuals, we can't estimate their counterfactual
    /// because we'd never observe them in the other treatment state.
    ///
    /// Good overlap: Both groups have propensity scores in range [0.1, 0.9]
    /// Bad overlap: Treated group has scores [0.8, 1.0], control has [0.0, 0.2]
    /// </para>
    /// </remarks>
    (T treatmentMin, T treatmentMax, T controlMin, T controlMax) CheckOverlap(
        Matrix<T> x, Vector<int> treatment);
}
