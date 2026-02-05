using AiDotNet.Distributions;

namespace AiDotNet.Scoring;

/// <summary>
/// Logarithmic scoring rule (negative log likelihood).
/// </summary>
/// <remarks>
/// <para>
/// The logarithmic score (also called log loss or cross-entropy) is the negative
/// log probability density/mass assigned to the observed value. It's the most
/// widely used proper scoring rule due to its connection to maximum likelihood.
/// </para>
/// <para>
/// <b>For Beginners:</b> The log score asks "how surprised was the model by what
/// actually happened?" If the model assigned high probability to the true outcome,
/// it gets a low (good) score. If it assigned low probability to what happened,
/// it gets a high (bad) penalty.
///
/// For example, if you predict 90% chance of rain and it rains, that's a good prediction.
/// But if you predict 1% chance of rain and it rains, that's a terrible prediction and
/// you get heavily penalized.
/// </para>
/// <para>
/// Score = -log(p(y)) where p(y) is the density at observation y.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class LogScore<T> : ScoringRuleBase<T>
{
    /// <inheritdoc/>
    public override string Name => "LogScore";

    /// <inheritdoc/>
    public override bool IsMinimized => true;  // Lower is better (negative log likelihood)

    /// <inheritdoc/>
    public override T Score(IParametricDistribution<T> distribution, T observation)
    {
        // Score = -log(pdf(y))
        T logPdf = distribution.LogPdf(observation);
        return NumOps.Negate(logPdf);
    }

    /// <inheritdoc/>
    public override T[] ScoreGradient(IParametricDistribution<T> distribution, T observation)
    {
        // Gradient of -log(pdf) = -gradient of log(pdf)
        T[] gradLogPdf = distribution.GradLogPdf(observation);
        T[] gradients = new T[gradLogPdf.Length];

        for (int i = 0; i < gradients.Length; i++)
        {
            gradients[i] = NumOps.Negate(gradLogPdf[i]);
        }

        return gradients;
    }
}
