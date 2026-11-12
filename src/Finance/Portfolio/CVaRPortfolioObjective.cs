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

    private static IEngine Engine => AiDotNetEngine.Current;

    /// <summary>The loss samples sorted ascending.</summary>
    private static Tensor<T> SortedLosses(Vector<T> losses)
        => Engine.TensorSort(Tensor<T>.FromVector(losses), 0, false).Values;

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
    public T Loss(Vector<T> weights, Vector<T> returns)
    {
        if (weights == null) throw new ArgumentNullException(nameof(weights));
        if (returns == null) throw new ArgumentNullException(nameof(returns));
        if (weights.Length != returns.Length)
            throw new ArgumentException(
                $"Weights and returns must align; got {weights.Length} and {returns.Length}.",
                nameof(returns));

        if (weights.Length == 0) return NumOps.Negate(NumOps.Zero);
        return NumOps.FromDouble(-Engine.DotProduct(Tensor<T>.FromVector(weights).Cast<double>().ToVector(), Tensor<T>.FromVector(returns).Cast<double>().ToVector()));
    }

    /// <summary>
    /// Value-at-Risk: the alpha-quantile of the loss sample.
    /// </summary>
    /// <remarks>
    /// Uses the lower empirical quantile (index <c>ceil(alpha * n) - 1</c> of the sorted losses),
    /// which is the convention the Rockafellar-Uryasev dual is stated against, so
    /// <see cref="ConditionalValueAtRisk"/> and this function agree on where the tail starts.
    /// </remarks>
    public T ValueAtRisk(Vector<T> losses)
    {
        if (losses == null) throw new ArgumentNullException(nameof(losses));
        if (losses.Length == 0)
            throw new ArgumentException("At least one loss sample is required.", nameof(losses));

        var sorted = SortedLosses(losses).AsSpan();

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
    /// The tail is the worst <c>(1 - alpha) * n</c> observations, INTERPOLATED: whole observations
    /// beyond the boundary count fully, and the boundary observation contributes its fractional share.
    /// That is what makes this agree with the Rockafellar-Uryasev dual
    /// <c>CVaR_alpha(Z) = min_nu ( nu + 1/(1-alpha) * E[(Z - nu)+] )</c> for EVERY sample size, not
    /// only when <c>(1 - alpha) * n</c> lands on an integer. Rounding the count up instead
    /// over-weighted the boundary loss: with alpha = 0.75 and losses {0, 1, 2, 3, 4}, a whole-
    /// observation tail of ceil(1.25) = 2 gives (3 + 4) / 2 = 3.5, while the dual at its optimal
    /// nu = 3 gives 3 + 1 / (5 * 0.25) = 3.8. An earlier revision was further out still, averaging
    /// everything from the VaR index onwards -- a 40% tail at alpha = 0.8, reporting 0.16 where the
    /// dual gives 0.30.
    /// <see cref="DualObjective"/> and <see cref="OptimalNu"/> are kept public precisely so the two
    /// formulations can be cross-checked rather than trusted.
    /// </para>
    /// <para>
    /// CVaR is at least VaR by construction: it averages over the tail rather than reporting its
    /// boundary.
    /// </para>
    /// </remarks>
    public T ConditionalValueAtRisk(Vector<T> losses)
    {
        if (losses == null) throw new ArgumentNullException(nameof(losses));
        if (losses.Length == 0)
            throw new ArgumentException("At least one loss sample is required.", nameof(losses));

        var sortedTensor = SortedLosses(losses);
        int count = sortedTensor.Length;

        // The tail size is fractional in general. Averaging over ceil(tailSize) whole observations
        // dilutes the boundary loss by counting it as a full observation, which is exactly where this
        // used to disagree with DualObjective. Weighting it by its fractional share reproduces the
        // dual at every sample size.
        double tailSize = (1.0 - _alpha) * count;
        if (tailSize < 1.0) tailSize = 1.0;
        if (tailSize > count) tailSize = count;

        int wholeCount = (int)Math.Floor(tailSize);
        double boundaryWeight = tailSize - wholeCount;

        double sum = wholeCount > 0
            ? Engine.TensorSum(Engine.TensorNarrow(sortedTensor.Cast<double>(), 0, count - wholeCount, wholeCount))
            : 0.0;

        // The observation just inside the tail boundary, counted for the fraction of it that the tail
        // actually covers.
        int boundaryIndex = count - wholeCount - 1;
        if (boundaryWeight > 0.0 && boundaryIndex >= 0)
            sum += boundaryWeight * NumOps.ToDouble(sortedTensor.AsSpan()[boundaryIndex]);

        return NumOps.FromDouble(sum / tailSize);
    }

    /// <summary>
    /// The Rockafellar-Uryasev dual objective at a given threshold nu:
    /// <c>nu + 1/(1 - alpha) * mean((Z - nu)+)</c>.
    /// </summary>
    /// <remarks>
    /// This is the form that makes CVaR trainable by gradient descent: it is convex in nu and its
    /// hinge is differentiable almost everywhere, whereas sorting to find a quantile is not.
    /// </remarks>
    public T DualObjective(Vector<T> losses, T nu)
    {
        if (losses == null) throw new ArgumentNullException(nameof(losses));
        if (losses.Length == 0)
            throw new ArgumentException("At least one loss sample is required.", nameof(losses));

        // Only the positive part of each excess counts (a NaN excess counts as nothing, as before).
        double threshold = NumOps.ToDouble(nu);
        var d = Engine.TensorSubtractScalar(Tensor<T>.FromVector(losses).Cast<double>(), threshold);
        double excess = Engine.TensorSum(Engine.TensorWhere(
            Engine.TensorGreaterThan(d, 0.0), d, Engine.TensorZerosLike(d)));
        return NumOps.FromDouble(threshold + excess / (losses.Length * (1.0 - _alpha)));
    }

    /// <summary>
    /// The nu minimizing <see cref="DualObjective"/>, which is the VaR of the sample.
    /// </summary>
    public T OptimalNu(Vector<T> losses) => ValueAtRisk(losses);

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

        // Max-subtracted tempered softmax, so a tiny temperature cannot overflow exp.
        var scoreTensor = Tensor<T>.FromVector(scores).Cast<double>();
        if (Engine.TensorCountNonzero(Engine.TensorIsNan(scoreTensor)) > 0
            || Engine.TensorCountNonzero(Engine.TensorIsInf(scoreTensor)) > 0)
            throw new ArgumentException("scores must be finite.", nameof(scores));
        var scaled = Engine.TensorDivideScalar(scoreTensor, temperature);
        Tensor<double> exps;
        if (Engine.TensorCountNonzero(Engine.TensorIsFinite(scaled)) == scaled.Length)
            exps = Engine.TensorExp(Engine.TensorSubtractScalar(scaled, Engine.TensorMaxValue(scaled)));
        else
        {
            // A finite temperature may still make s/tau overflow. Shift first in that case.
            var shifted = Engine.TensorSubtractScalar(scoreTensor, Engine.TensorMaxValue(scoreTensor));
            exps = Engine.TensorWhere(Engine.TensorEquals(shifted, 0.0), Engine.TensorOnesLike(shifted),
                Engine.TensorExp(Engine.TensorDivideScalar(shifted, temperature)));
        }
        return Engine.TensorDivideScalar(exps, Engine.TensorSum(exps)).Cast<T>().ToVector();
    }

    /// <summary>
    /// One-way turnover between consecutive allocations: <c>0.5 * sum |w_new - w_old|</c>.
    /// </summary>
    /// <remarks>
    /// For evaluation only — see the class remarks on why this is not part of the training objective.
    /// Halved because a rebalance both sells and buys, so summing the absolute differences would
    /// count each trade twice.
    /// </remarks>
    public T Turnover(Vector<T> previousWeights, Vector<T> newWeights)
    {
        if (previousWeights == null) throw new ArgumentNullException(nameof(previousWeights));
        if (newWeights == null) throw new ArgumentNullException(nameof(newWeights));
        if (previousWeights.Length != newWeights.Length)
            throw new ArgumentException(
                $"Allocations must align; got {previousWeights.Length} and {newWeights.Length}.",
                nameof(newWeights));

        if (newWeights.Length == 0) return NumOps.Zero;
        double sum = Engine.TensorSum(Engine.TensorAbs(
            Engine.TensorSubtract(Tensor<T>.FromVector(newWeights).Cast<double>(), Tensor<T>.FromVector(previousWeights).Cast<double>())));
        return NumOps.FromDouble(0.5 * sum);
    }

    /// <summary>
    /// Transaction cost of a rebalance, in return units, at the given cost in basis points.
    /// For evaluation only; the reference implementation trains with 0 bps.
    /// </summary>
    public T TransactionCost(Vector<T> previousWeights, Vector<T> newWeights, double basisPoints)
    {
        // Non-finite is rejected, not just NaN (net471-safe: no double.IsFinite). PositiveInfinity
        // passed the old guard and propagated infinity into the cost for any non-zero turnover, and
        // produced NaN (infinity times zero) when turnover was zero -- a cost of NaN reported as if
        // it were a real number.
        if (double.IsNaN(basisPoints) || double.IsInfinity(basisPoints) || basisPoints < 0.0)
            throw new ArgumentOutOfRangeException(nameof(basisPoints), basisPoints,
                "basisPoints must be a finite, non-negative value.");

        return NumOps.FromDouble(NumOps.ToDouble(Turnover(previousWeights, newWeights)) * (basisPoints / 10000.0));
    }
}
