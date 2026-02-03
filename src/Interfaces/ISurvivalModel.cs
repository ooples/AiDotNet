namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for survival analysis models.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Survival analysis models time-to-event data, where we're interested
/// in predicting when an event will occur (e.g., customer churn, equipment failure, patient survival).
/// A key challenge is "censoring" - when we don't observe the event for some subjects.</para>
///
/// <para><b>Key concepts:</b>
/// <list type="bullet">
/// <item><b>Survival function S(t):</b> Probability of surviving beyond time t</item>
/// <item><b>Hazard function h(t):</b> Instantaneous risk of event at time t</item>
/// <item><b>Cumulative hazard H(t):</b> Accumulated risk up to time t</item>
/// <item><b>Censoring:</b> When event is not observed (e.g., study ends before event)</item>
/// </list>
/// </para>
///
/// <para><b>Common models:</b>
/// <list type="bullet">
/// <item><b>Kaplan-Meier:</b> Non-parametric survival curve estimation</item>
/// <item><b>Nelson-Aalen:</b> Non-parametric cumulative hazard estimation</item>
/// <item><b>Cox PH:</b> Semi-parametric proportional hazards model</item>
/// <item><b>AFT:</b> Accelerated failure time models (Weibull, Log-Normal)</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public interface ISurvivalModel<T> : IFullModel<T, Matrix<T>, Vector<T>>
{
    /// <summary>
    /// Fits the survival model to time-to-event data.
    /// </summary>
    /// <param name="times">Observed times (event or censoring times).</param>
    /// <param name="events">Event indicators (1 = event occurred, 0 = censored).</param>
    /// <param name="features">Optional feature matrix for regression models.</param>
    void Fit(Vector<T> times, Vector<T> events, Matrix<T>? features = null);

    /// <summary>
    /// Predicts survival probability at specified times.
    /// </summary>
    /// <param name="times">Times at which to predict survival.</param>
    /// <param name="features">Features for new subjects (for regression models).</param>
    /// <returns>Survival probabilities S(t) for each time point.</returns>
    Matrix<T> PredictSurvival(Vector<T> times, Matrix<T>? features = null);

    /// <summary>
    /// Predicts cumulative hazard at specified times.
    /// </summary>
    /// <param name="times">Times at which to predict cumulative hazard.</param>
    /// <param name="features">Features for new subjects (for regression models).</param>
    /// <returns>Cumulative hazard H(t) for each time point.</returns>
    Matrix<T> PredictCumulativeHazard(Vector<T> times, Matrix<T>? features = null);

    /// <summary>
    /// Predicts risk scores for subjects (higher = higher risk).
    /// </summary>
    /// <param name="features">Feature matrix for subjects.</param>
    /// <returns>Risk scores for each subject.</returns>
    Vector<T> PredictRisk(Matrix<T> features);

    /// <summary>
    /// Gets the estimated median survival time.
    /// </summary>
    /// <param name="features">Features for subjects (for regression models).</param>
    /// <returns>Median survival times.</returns>
    Vector<T> PredictMedianSurvivalTime(Matrix<T>? features = null);

    /// <summary>
    /// Gets the unique event times from the training data.
    /// </summary>
    Vector<T>? EventTimes { get; }

    /// <summary>
    /// Gets the baseline survival function values at event times.
    /// </summary>
    Vector<T>? BaselineSurvival { get; }
}
