using System;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Finance.Portfolio;

/// <summary>
/// SIT's decision-focused training objective: the Conditional Value-at-Risk of the realized
/// portfolio loss (Hwang and Zohren, arXiv:2510.03129).
/// </summary>
/// <remarks>
/// <para>
/// This is the paper's central argument made concrete. Predict-then-optimize pipelines minimize a
/// forecasting error and let a downstream optimizer turn small inaccuracies into fragile
/// allocations; SIT instead trains on the portfolio's own tail risk, so the training signal and the
/// financial goal are the same quantity.
/// </para>
/// <para>
/// <b>CVaR is the sole training signal.</b> The paper is explicit that no auxiliary prediction
/// losses are used. Adding an MSE, expected-return or turnover term would reintroduce exactly the
/// objective mismatch the method exists to remove, so this class has no such terms and none should
/// be added.
/// </para>
/// <para>
/// Transaction costs appear in the paper only as a post-hoc sensitivity analysis, not in the training
/// objective (the reference implementation defaults to zero basis points). <see cref="Turnover"/> and
/// <see cref="TransactionCost"/> are provided for that evaluation, deliberately separate from
/// <see cref="ConditionalValueAtRisk"/>.
/// </para>
/// <para><b>For Beginners:</b> Value-at-Risk is a loss level you only exceed rarely. CVaR is the
/// AVERAGE loss in those rare bad cases, so it cares about how bad the bad days are rather than just
/// how often they happen.</para>
/// </remarks>
/// <typeparam name="T">The numeric type.</typeparam>
public class CVaRPortfolioObjective<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Conventional CVaR confidence level, used as the default.
    /// </summary>
    /// <remarks>
    /// The paper writes the level symbolically as alpha and does not state a number, and the
    /// reference implementation does not document one either, so this 0.95 is the standard
    /// risk-management convention (average of the worst 5%) rather than a value quoted from the
    /// paper. Configure it explicitly if a specific tail is required.
    /// </remarks>
    public const double ConventionalAlpha = 0.95;

    private readonly double _alpha;

    /// <summary>Gets the confidence level alpha.</summary>
    public double Alpha => _alpha;

    /// <summary>Creates the objective.</summary>
    /// <param name="alpha">Confidence level in (0, 1). See <see cref="ConventionalAlpha"/>.</param>
    public CVaRPortfolioObjective(double alpha = ConventionalAlpha)
    {
        if (alpha is <= 0.0 or >= 1.0 || double.IsNaN(alpha))
            throw new ArgumentOutOfRangeException(nameof(alpha), alpha, "alpha must be in (0, 1).");
        _alpha = alpha;
    }

    /// <summary>
    /// Per-step portfolio loss <c>L = -(w^T r)</c>: the NEGATIVE realized return.
    /// </summary>
    /// <param name="weights">Portfolio weights over assets.</param>
    /// <param name="returns">Realized asset returns for the same step.</param>
    /// <remarks>
    /// Negated so that "loss" grows as the portfolio does worse, which is what the CVaR tail must be
    /// taken over. Feeding returns directly would put CVaR on the profitable tail and train the model
    /// to avoid gains.
    /// </remarks>
    public double Loss(Vector<T> weights, Vector<T> returns)
    {
        if (weights == null) throw new ArgumentNullException(nameof(weights));
        if (returns == null) throw new ArgumentNullException(nameof(returns));
        if (weights.Length != returns.Length)
            throw new ArgumentException(
                $"Weights and returns must align; got {weights.Length} and {returns.Length}.",
                nameof(returns));

        double portfolioReturn = 0.0;
        for (int i = 0; i < weights.Length; i++)
            portfolioReturn += NumOps.ToDouble(weights[i]) * NumOps.ToDouble(returns[i]);

        return -portfolioReturn;
    }

    /// <summary>
    /// Value-at-Risk: the alpha-quantile of the loss sample.
    /// </summary>
    /// <remarks>
    /// Uses the lower empirical quantile (index <c>ceil(alpha * n) - 1</c> of the sorted losses),
    /// which is the convention the Rockafellar-Uryasev dual is stated against, so
    /// <see cref="ConditionalValueAtRisk"/> and this function agree on where the tail starts.
    /// </remarks>
    public double ValueAtRisk(double[] losses)
    {
        if (losses == null) throw new ArgumentNullException(nameof(losses));
        if (losses.Length == 0)
            throw new ArgumentException("At least one loss sample is required.", nameof(losses));

        var sorted = (double[])losses.Clone();
        Array.Sort(sorted);

        int index = (int)Math.Ceiling(_alpha * sorted.Length) - 1;
        if (index < 0) index = 0;
        if (index >= sorted.Length) index = sorted.Length - 1;
        return sorted[index];
    }

    /// <summary>
    /// Conditional Value-at-Risk: the mean loss in the worst <c>(1 - alpha)</c> tail.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The tail is the worst <c>ceil((1 - alpha) * n)</c> observations. Sizing it from the tail
    /// FRACTION is what makes this agree with the Rockafellar-Uryasev dual
    /// <c>CVaR_alpha(Z) = min_nu ( nu + 1/(1-alpha) * E[(Z - nu)+] )</c>; an earlier revision instead
    /// averaged everything from the VaR index onwards, which includes the threshold observation itself
    /// and so averaged a 40% tail at alpha = 0.8 — reporting 0.16 where the dual gives 0.30.
    /// <see cref="DualObjective"/> and <see cref="OptimalNu"/> are kept public precisely so the two
    /// formulations can be cross-checked rather than trusted.
    /// </para>
    /// <para>
    /// CVaR is at least VaR by construction: it averages over the tail rather than reporting its
    /// boundary.
    /// </para>
    /// </remarks>
    public double ConditionalValueAtRisk(double[] losses)
    {
        if (losses == null) throw new ArgumentNullException(nameof(losses));
        if (losses.Length == 0)
            throw new ArgumentException("At least one loss sample is required.", nameof(losses));

        var sorted = (double[])losses.Clone();
        Array.Sort(sorted);

        // At least one observation, so the worst loss is always in the tail; a tail averaging nothing
        // would silently report zero risk.
        int tailCount = (int)Math.Ceiling((1.0 - _alpha) * sorted.Length);
        if (tailCount < 1) tailCount = 1;
        if (tailCount > sorted.Length) tailCount = sorted.Length;

        double sum = 0.0;
        for (int i = sorted.Length - tailCount; i < sorted.Length; i++) sum += sorted[i];
        return sum / tailCount;
    }

    /// <summary>
    /// The Rockafellar-Uryasev dual objective at a given threshold nu:
    /// <c>nu + 1/(1 - alpha) * mean((Z - nu)+)</c>.
    /// </summary>
    /// <remarks>
    /// This is the form that makes CVaR trainable by gradient descent: it is convex in nu and its
    /// hinge is differentiable almost everywhere, whereas sorting to find a quantile is not.
    /// </remarks>
    public double DualObjective(double[] losses, double nu)
    {
        if (losses == null) throw new ArgumentNullException(nameof(losses));
        if (losses.Length == 0)
            throw new ArgumentException("At least one loss sample is required.", nameof(losses));

        double excess = 0.0;
        for (int i = 0; i < losses.Length; i++)
        {
            double d = losses[i] - nu;
            if (d > 0.0) excess += d;
        }

        return nu + (excess / (losses.Length * (1.0 - _alpha)));
    }

    /// <summary>
    /// The nu minimizing <see cref="DualObjective"/>, which is the VaR of the sample.
    /// </summary>
    public double OptimalNu(double[] losses) => ValueAtRisk(losses);

    /// <summary>
    /// Long-only, fully invested weights: <c>softmax(scores / tau)</c>.
    /// </summary>
    /// <param name="scores">Per-asset scores from the network.</param>
    /// <param name="temperature">
    /// tau. Lower concentrates the portfolio, higher spreads it. The paper reports an interior
    /// optimum near 1.3.
    /// </param>
    /// <remarks>
    /// <para>
    /// The softmax is what ENFORCES the constraints rather than merely encouraging them: outputs are
    /// positive (long-only) and sum to one (fully invested) by construction, so no projection step or
    /// penalty term is needed and the constraints cannot be violated mid-training.
    /// </para>
    /// <para>
    /// The maximum score is subtracted before exponentiating. That is a numerical necessity, not a
    /// style choice: with a small tau the scaled scores grow large and a direct exp overflows to
    /// infinity, producing NaN weights.
    /// </para>
    /// </remarks>
    public Vector<T> Weights(Vector<T> scores, double temperature)
    {
        if (scores == null) throw new ArgumentNullException(nameof(scores));
        if (scores.Length == 0)
            throw new ArgumentException("At least one asset is required.", nameof(scores));
        if (temperature <= 0.0 || double.IsNaN(temperature) || double.IsInfinity(temperature))
            throw new ArgumentOutOfRangeException(nameof(temperature), temperature,
                "temperature must be finite and positive.");

        double max = double.NegativeInfinity;
        for (int i = 0; i < scores.Length; i++)
            max = Math.Max(max, NumOps.ToDouble(scores[i]) / temperature);

        double sum = 0.0;
        var exps = new double[scores.Length];
        for (int i = 0; i < scores.Length; i++)
        {
            exps[i] = Math.Exp((NumOps.ToDouble(scores[i]) / temperature) - max);
            sum += exps[i];
        }

        var result = new Vector<T>(scores.Length);
        for (int i = 0; i < scores.Length; i++) result[i] = NumOps.FromDouble(exps[i] / sum);
        return result;
    }

    /// <summary>
    /// One-way turnover between consecutive allocations: <c>0.5 * sum |w_new - w_old|</c>.
    /// </summary>
    /// <remarks>
    /// For evaluation only — see the class remarks on why this is not part of the training objective.
    /// Halved because a rebalance both sells and buys, so summing the absolute differences would
    /// count each trade twice.
    /// </remarks>
    public double Turnover(Vector<T> previousWeights, Vector<T> newWeights)
    {
        if (previousWeights == null) throw new ArgumentNullException(nameof(previousWeights));
        if (newWeights == null) throw new ArgumentNullException(nameof(newWeights));
        if (previousWeights.Length != newWeights.Length)
            throw new ArgumentException(
                $"Allocations must align; got {previousWeights.Length} and {newWeights.Length}.",
                nameof(newWeights));

        double sum = 0.0;
        for (int i = 0; i < newWeights.Length; i++)
            sum += Math.Abs(NumOps.ToDouble(newWeights[i]) - NumOps.ToDouble(previousWeights[i]));

        return 0.5 * sum;
    }

    /// <summary>
    /// Transaction cost of a rebalance, in return units, at the given cost in basis points.
    /// For evaluation only; the reference implementation trains with 0 bps.
    /// </summary>
    public double TransactionCost(Vector<T> previousWeights, Vector<T> newWeights, double basisPoints)
    {
        // Non-finite is rejected, not just NaN (net471-safe: no double.IsFinite). PositiveInfinity
        // passed the old guard and propagated infinity into the cost for any non-zero turnover, and
        // produced NaN (infinity times zero) when turnover was zero -- a cost of NaN reported as if
        // it were a real number.
        if (double.IsNaN(basisPoints) || double.IsInfinity(basisPoints) || basisPoints < 0.0)
            throw new ArgumentOutOfRangeException(nameof(basisPoints), basisPoints,
                "basisPoints must be a finite, non-negative value.");

        return Turnover(previousWeights, newWeights) * (basisPoints / 10000.0);
    }
}
