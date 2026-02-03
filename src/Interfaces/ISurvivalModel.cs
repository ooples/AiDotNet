using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the interface for survival analysis models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Survival analysis models predict the time until an event occurs (e.g., death, failure,
/// churn) while handling censored data (subjects who haven't experienced the event yet).
/// </para>
/// <para>
/// <b>For Beginners:</b> Survival analysis is used when you want to predict "how long until
/// something happens." Common applications include:
///
/// - Medical: How long until a disease recurs?
/// - Engineering: How long until a machine fails?
/// - Business: How long until a customer cancels their subscription?
///
/// The tricky part is "censoring" - some subjects in your study haven't experienced the event
/// yet when you analyze the data. For example, a patient might still be alive at the end of
/// a study. We know they survived AT LEAST this long, but not their actual survival time.
///
/// Survival models handle this uncertainty properly, unlike regular regression which would
/// either exclude these cases or treat them incorrectly.
///
/// Key concepts:
/// - Survival function S(t): Probability of surviving beyond time t
/// - Hazard function h(t): Instantaneous risk of the event at time t
/// - Censoring: When we don't observe the actual event time
/// </para>
/// </remarks>
public interface ISurvivalModel<T> : IFullModel<T, Matrix<T>, Vector<T>>
{
    /// <summary>
    /// Fits the survival model to time-to-event data.
    /// </summary>
    /// <param name="x">The feature matrix (covariates).</param>
    /// <param name="times">The observed times (event time or censoring time).</param>
    /// <param name="events">Event indicators (1 = event occurred, 0 = censored).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This trains the model on your survival data. You need:
    /// - x: Features about each subject (e.g., age, treatment type)
    /// - times: How long each subject was observed
    /// - events: Did the event happen (1) or was it censored (0)?
    ///
    /// Example:
    /// Subject A: time=5 years, event=1 → Patient died at year 5
    /// Subject B: time=3 years, event=0 → Patient was still alive at year 3 (study ended)
    /// </para>
    /// </remarks>
    void FitSurvival(Matrix<T> x, Vector<T> times, Vector<int> events);

    /// <summary>
    /// Predicts survival probabilities at specified time points.
    /// </summary>
    /// <param name="x">The feature matrix for prediction.</param>
    /// <param name="times">Time points at which to predict survival.</param>
    /// <returns>Matrix of survival probabilities (rows = subjects, columns = time points).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Returns the probability that each subject survives beyond each time point.
    /// Values range from 0 to 1, with 1 meaning certain survival and 0 meaning certain event.
    ///
    /// Example output for 2 patients at times [1, 3, 5] years:
    /// Patient 1: [0.95, 0.80, 0.60] → 95% survive 1yr, 80% survive 3yr, 60% survive 5yr
    /// Patient 2: [0.99, 0.95, 0.85] → Better prognosis (higher survival probabilities)
    /// </para>
    /// </remarks>
    Matrix<T> PredictSurvivalProbability(Matrix<T> x, Vector<T> times);

    /// <summary>
    /// Predicts hazard ratios relative to a baseline.
    /// </summary>
    /// <param name="x">The feature matrix for prediction.</param>
    /// <returns>Vector of hazard ratios for each subject.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The hazard ratio tells you how much more (or less) likely
    /// a subject is to experience the event compared to a baseline.
    ///
    /// - Hazard ratio = 1.0: Same risk as baseline
    /// - Hazard ratio = 2.0: Twice the risk (event happens twice as fast)
    /// - Hazard ratio = 0.5: Half the risk (event takes twice as long)
    ///
    /// This is commonly used in medical research to compare treatments.
    /// </para>
    /// </remarks>
    Vector<T> PredictHazardRatio(Matrix<T> x);

    /// <summary>
    /// Predicts median survival time for each subject.
    /// </summary>
    /// <param name="x">The feature matrix for prediction.</param>
    /// <returns>Vector of median survival times.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The median survival time is when the survival probability drops to 50%.
    /// In other words, half the subjects with these characteristics would have experienced
    /// the event by this time.
    ///
    /// Note: This may be undefined (return infinity) if the survival never drops below 50%.
    /// </para>
    /// </remarks>
    Vector<T> PredictMedianSurvivalTime(Matrix<T> x);

    /// <summary>
    /// Gets the baseline survival function (when all covariates are at reference level).
    /// </summary>
    /// <param name="times">Time points at which to evaluate.</param>
    /// <returns>Vector of baseline survival probabilities.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The baseline survival is the survival curve for a "reference" subject
    /// (typically with all covariates at zero or their mean). Individual survival curves are
    /// derived from this baseline adjusted by the subject's covariates.
    /// </para>
    /// </remarks>
    Vector<T> GetBaselineSurvival(Vector<T> times);

    /// <summary>
    /// Calculates the concordance index (C-index) for model evaluation.
    /// </summary>
    /// <param name="x">The feature matrix.</param>
    /// <param name="times">The observed times.</param>
    /// <param name="events">Event indicators.</param>
    /// <returns>The concordance index between 0 and 1.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The C-index measures how well the model ranks subjects by risk.
    /// It's the probability that, for a random pair of subjects, the one who had the event
    /// first was correctly predicted to have higher risk.
    ///
    /// - C-index = 0.5: Random guessing (no predictive ability)
    /// - C-index = 1.0: Perfect prediction
    /// - C-index &gt; 0.7: Generally considered good
    ///
    /// It's similar to AUC-ROC for classification but adapted for survival data.
    /// </para>
    /// </remarks>
    T CalculateConcordanceIndex(Matrix<T> x, Vector<T> times, Vector<int> events);
}
