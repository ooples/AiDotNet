using AiDotNet.Distributions;
using AiDotNet.Helpers;

namespace AiDotNet.Scoring;

/// <summary>
/// Defines a proper scoring rule for evaluating probabilistic predictions.
/// </summary>
/// <remarks>
/// <para>
/// A scoring rule measures how well a predicted probability distribution matches
/// the observed outcomes. Proper scoring rules are maximized (or minimized) when
/// the predicted distribution equals the true distribution, incentivizing honest forecasts.
/// </para>
/// <para>
/// <b>For Beginners:</b> When you make a probabilistic prediction (like "70% chance of rain"),
/// scoring rules tell you how good your prediction was after you see what actually happened.
/// A proper scoring rule rewards you most for predicting probabilities that match reality -
/// you can't game the system by being overconfident or underconfident.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public interface IScoringRule<T>
{
    /// <summary>
    /// Gets the name of this scoring rule.
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Gets whether this scoring rule should be minimized (true) or maximized (false).
    /// </summary>
    /// <remarks>
    /// For example, log score is typically maximized (higher is better),
    /// while CRPS is typically minimized (lower is better).
    /// </remarks>
    bool IsMinimized { get; }

    /// <summary>
    /// Computes the score for a single prediction-observation pair.
    /// </summary>
    /// <param name="distribution">The predicted probability distribution.</param>
    /// <param name="observation">The observed value.</param>
    /// <returns>The score for this prediction.</returns>
    T Score(IParametricDistribution<T> distribution, T observation);

    /// <summary>
    /// Computes the gradient of the score with respect to distribution parameters.
    /// </summary>
    /// <param name="distribution">The predicted probability distribution.</param>
    /// <param name="observation">The observed value.</param>
    /// <returns>Array of gradients, one per distribution parameter.</returns>
    /// <remarks>
    /// These gradients are used in gradient-based optimization like NGBoost.
    /// </remarks>
    Vector<T> ScoreGradient(IParametricDistribution<T> distribution, T observation);

    /// <summary>
    /// Computes the mean score over multiple prediction-observation pairs.
    /// </summary>
    /// <param name="distributions">Array of predicted distributions.</param>
    /// <param name="observations">Array of observed values.</param>
    /// <returns>The mean score.</returns>
    T MeanScore(IParametricDistribution<T>[] distributions, Vector<T> observations);
}

/// <summary>
/// Base class for scoring rules providing common functionality.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract class ScoringRuleBase<T> : IScoringRule<T>
{
    /// <summary>
    /// Numeric operations for type T.
    /// </summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <inheritdoc/>
    public abstract string Name { get; }

    /// <inheritdoc/>
    public abstract bool IsMinimized { get; }

    /// <inheritdoc/>
    public abstract T Score(IParametricDistribution<T> distribution, T observation);

    /// <inheritdoc/>
    public abstract Vector<T> ScoreGradient(IParametricDistribution<T> distribution, T observation);

    /// <inheritdoc/>
    public virtual T MeanScore(IParametricDistribution<T>[] distributions, Vector<T> observations)
    {
        if (distributions.Length != observations.Length)
            throw new ArgumentException("Distributions and observations must have the same length.");

        if (distributions.Length == 0)
            throw new ArgumentException("At least one observation is required.");

        T sum = NumOps.Zero;
        for (int i = 0; i < distributions.Length; i++)
        {
            sum = NumOps.Add(sum, Score(distributions[i], observations[i]));
        }

        return NumOps.Divide(sum, NumOps.FromDouble(distributions.Length));
    }
}
