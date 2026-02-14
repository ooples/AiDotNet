namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for causal inference models (meta-learners).
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Causal inference models estimate the causal effect of a treatment
/// on an outcome. Unlike prediction, we want to know "what would happen if we applied treatment X?"
/// This is called the treatment effect.</para>
///
/// <para><b>Key concepts:</b>
/// <list type="bullet">
/// <item><b>Treatment:</b> The intervention we're studying (e.g., a drug, a marketing campaign)</item>
/// <item><b>Outcome:</b> The result we measure (e.g., health, sales)</item>
/// <item><b>CATE:</b> Conditional Average Treatment Effect - effect for specific subgroups</item>
/// <item><b>ATE:</b> Average Treatment Effect - overall average effect</item>
/// </list>
/// </para>
///
/// <para><b>Meta-learners:</b>
/// <list type="bullet">
/// <item><b>S-Learner:</b> Single model with treatment as feature</item>
/// <item><b>T-Learner:</b> Two separate models for treatment/control</item>
/// <item><b>X-Learner:</b> Cross-fitting approach for heterogeneous effects</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[AiDotNet.Configuration.YamlConfigurable("CausalModel")]
public interface ICausalModel<T> : IFullModel<T, Matrix<T>, Vector<T>>
{
    /// <summary>
    /// Fits the causal model to observational data.
    /// </summary>
    /// <param name="features">Feature matrix [n_samples, n_features].</param>
    /// <param name="treatment">Binary treatment indicator (1 = treated, 0 = control).</param>
    /// <param name="outcome">Observed outcomes.</param>
    void Fit(Matrix<T> features, Vector<T> treatment, Vector<T> outcome);

    /// <summary>
    /// Estimates the Conditional Average Treatment Effect (CATE) for subjects.
    /// </summary>
    /// <param name="features">Feature matrix for subjects.</param>
    /// <returns>Estimated treatment effect for each subject.</returns>
    Vector<T> EstimateTreatmentEffect(Matrix<T> features);

    /// <summary>
    /// Estimates the Average Treatment Effect (ATE) across the population.
    /// </summary>
    /// <param name="features">Feature matrix for the population.</param>
    /// <returns>The average treatment effect.</returns>
    T EstimateAverageTreatmentEffect(Matrix<T> features);

    /// <summary>
    /// Predicts the outcome under treatment.
    /// </summary>
    /// <param name="features">Feature matrix for subjects.</param>
    /// <returns>Predicted outcomes if treated.</returns>
    Vector<T> PredictTreated(Matrix<T> features);

    /// <summary>
    /// Predicts the outcome under control.
    /// </summary>
    /// <param name="features">Feature matrix for subjects.</param>
    /// <returns>Predicted outcomes if not treated.</returns>
    Vector<T> PredictControl(Matrix<T> features);
}
