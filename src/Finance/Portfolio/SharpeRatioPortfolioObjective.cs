using System;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Finance.Portfolio;

/// <summary>
/// The Sharpe-ratio training objective and the sum-normalizing allocation layer for
/// <see cref="GraphAttentionPortfolio{T}"/> (Korangi, Mues and Bravo, arXiv:2407.15532).
/// </summary>
/// <remarks>
/// <para>
/// The loss is <c>LF = -ln(mu_p) + ln(sigma_p)</c>, which is <c>-ln(mu_p / sigma_p)</c> — the negative
/// log Sharpe ratio. Maximizing a ratio is awkward for a gradient optimizer; taking the negative log
/// turns it into a plain minimization and splits the ratio into two additive terms.
/// </para>
/// <para>
/// <b>Allocation does NOT use a softmax</b>, and that is a deliberate finding of the paper rather than
/// an omission: with an investment universe of thousands of firms a softmax "would lead to many tiny
/// holdings of firms, which is impractical and can bring high transaction costs". Softmax cannot emit
/// an exact zero, so every asset always receives some capital. Instead the weights are
/// <c>w_u = s_u / sum_v s_v</c> over non-negative scores, so any score driven to zero — which the L1
/// regularization on the preceding feed-forward layer is there to do — produces a weight of EXACTLY
/// zero and the firm drops out of the portfolio entirely. The paper also considered sparsemax and
/// reported this mechanism as more stable, with a smoother loss curve and more reproducible weights
/// across runs.
/// </para>
/// <para><b>For Beginners:</b> The Sharpe ratio is return divided by volatility — reward per unit of
/// risk. This trains the model to maximize it directly, and allocates capital in a way that can leave
/// an asset out completely rather than giving everything a sliver.</para>
/// </remarks>
/// <typeparam name="T">The numeric type.</typeparam>
public class SharpeRatioPortfolioObjective<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Floor applied to the mean return before taking its logarithm.
    /// </summary>
    /// <remarks>
    /// The paper's <c>-ln(mu_p)</c> is undefined for a non-positive mean return, and it does not
    /// discuss the case — but a portfolio losing money on average is entirely reachable during
    /// training. Flooring <c>mu_p</c> here makes that situation the WORST finite loss (as
    /// <c>mu_p -&gt; 0</c>, <c>-ln(mu_p) -&gt; +inf</c>) instead of producing NaN and poisoning every
    /// subsequent gradient. This is a documented deviation, chosen because it preserves the objective's
    /// ordering: a losing portfolio scores worse than any profitable one.
    /// </remarks>
    public const double MeanReturnFloor = 1e-12;

    /// <summary>
    /// The loss floor assigned to a portfolio whose mean return is not positive.
    /// </summary>
    /// <remarks>
    /// Chosen to exceed any loss a profitable portfolio can reach. The profitable branch returns
    /// <c>-ln(mu) + ln(sigma)</c>; with <c>mu</c> above zero and <c>sigma</c> floored at
    /// <see cref="MeanReturnFloor"/>, that stays well inside +/-100 for every input a return series
    /// can produce. Adding <c>-mean</c> on top keeps losing portfolios ordered among themselves by
    /// how far under water they are, so an optimizer still has a gradient to follow between two bad
    /// candidates.
    /// </remarks>
    public const double LosingPortfolioPenalty = 1e3;

    /// <summary>
    /// Mean and (population) standard deviation of a portfolio return series.
    /// </summary>
    public (double Mean, double Volatility) Moments(double[] portfolioReturns)
    {
        if (portfolioReturns == null) throw new ArgumentNullException(nameof(portfolioReturns));
        if (portfolioReturns.Length == 0)
            throw new ArgumentException("At least one return is required.", nameof(portfolioReturns));

        double mean = 0.0;
        for (int i = 0; i < portfolioReturns.Length; i++) mean += portfolioReturns[i];
        mean /= portfolioReturns.Length;

        double variance = 0.0;
        for (int i = 0; i < portfolioReturns.Length; i++)
        {
            double d = portfolioReturns[i] - mean;
            variance += d * d;
        }

        return (mean, Math.Sqrt(variance / portfolioReturns.Length));
    }

    /// <summary>
    /// The Sharpe ratio <c>mu_p / sigma_p</c>, assuming a zero risk-free rate as the paper does.
    /// </summary>
    /// <remarks>
    /// A constant return series has zero volatility and an undefined ratio; reported as 0 rather than
    /// infinity, so a degenerate series cannot masquerade as an infinitely good portfolio.
    /// </remarks>
    public double SharpeRatio(double[] portfolioReturns)
    {
        var (mean, volatility) = Moments(portfolioReturns);
        return volatility <= 0.0 ? 0.0 : mean / volatility;
    }

    /// <summary>
    /// The training loss <c>LF = -ln(mu_p) + ln(sigma_p)</c>.
    /// </summary>
    /// <remarks>
    /// Equivalent to <c>-ln(Sharpe)</c>, so MINIMIZING it maximizes the Sharpe ratio. Getting the sign
    /// backwards would train the model to seek the worst risk-adjusted return, which is why the
    /// relationship to <see cref="SharpeRatio"/> is asserted in the tests rather than assumed.
    /// </remarks>
    public double Loss(double[] portfolioReturns)
    {
        var (mean, volatility) = Moments(portfolioReturns);

        // A NON-POSITIVE mean is its own case, not a floored one. Flooring both terms with the same
        // constant destroyed the ordering this objective documents: a degenerate losing series with
        // mean = -0.01 and volatility = 1e-15 collapsed to -ln(1e-12) + ln(1e-12) = 0, while an
        // honest profitable series with mean = 0.01 and volatility = 0.02 scored about 0.69. Minimizing
        // the objective therefore PREFERRED the losing portfolio.
        //
        // A losing portfolio now returns a large finite penalty that no volatility term can offset,
        // which is what "a losing portfolio scores worse than any profitable one" requires. Finite
        // rather than infinity so an optimizer can still compare two losing candidates by how far
        // under water they are.
        if (mean <= 0.0)
        {
            return LosingPortfolioPenalty - mean;
        }

        // Zero volatility would send ln(sigma) to -infinity. Floor it so a degenerate series cannot
        // produce a non-finite loss. Only reachable now for a PROFITABLE portfolio, where the mean
        // term is already bounded.
        double safeVolatility = Math.Max(volatility, MeanReturnFloor);

        return -Math.Log(mean) + Math.Log(safeVolatility);
    }


    /// <summary>
    /// Portfolio return series for a fixed allocation: <c>r_pt = sum_i w_i * r_it</c>.
    /// </summary>
    /// <param name="weights">Allocation over assets.</param>
    /// <param name="assetReturns">Returns shaped <c>[steps, assets]</c>.</param>
    public double[] PortfolioReturns(Vector<T> weights, Tensor<T> assetReturns)
    {
        if (weights == null) throw new ArgumentNullException(nameof(weights));
        if (assetReturns == null) throw new ArgumentNullException(nameof(assetReturns));
        if (assetReturns.Shape.Length != 2)
            throw new ArgumentException(
                $"assetReturns must be [steps, assets]; got rank {assetReturns.Shape.Length}.",
                nameof(assetReturns));

        int steps = assetReturns.Shape[0];
        int assets = assetReturns.Shape[1];
        if (assets != weights.Length)
            throw new ArgumentException(
                $"assetReturns has {assets} assets but weights has {weights.Length}.", nameof(assetReturns));

        var result = new double[steps];
        for (int s = 0; s < steps; s++)
        {
            double sum = 0.0;
            for (int a = 0; a < assets; a++)
                sum += NumOps.ToDouble(weights[a]) * NumOps.ToDouble(assetReturns[(s * assets) + a]);
            result[s] = sum;
        }

        return result;
    }

    /// <summary>
    /// The paper's "Importance Layer": <c>w_u = s_u / sum_v s_v</c> over non-negative scores.
    /// </summary>
    /// <param name="scores">
    /// Scores from the final feed-forward layer. Negative entries are clamped to zero.
    /// </param>
    /// <returns>Weights in [0, 1] summing to 1, preserving any exact zeros in the scores.</returns>
    /// <remarks>
    /// <para>
    /// Clamping rather than shifting or taking absolute values: a negative score means "do not hold
    /// this", and the portfolio is long-only, so zero is the faithful reading. Shifting the whole
    /// vector to be non-negative would destroy the sparsity by giving every previously-zero score a
    /// positive share, which is exactly the softmax behaviour the paper set out to avoid.
    /// </para>
    /// <para>
    /// When every score is non-positive there is no information to allocate on, so this falls back to
    /// an equal weighting — a defined, fully-invested portfolio — rather than dividing by zero.
    /// </para>
    /// </remarks>
    public Vector<T> Allocate(Vector<T> scores)
    {
        if (scores == null) throw new ArgumentNullException(nameof(scores));
        if (scores.Length == 0)
            throw new ArgumentException("At least one asset is required.", nameof(scores));

        var clamped = new double[scores.Length];
        double sum = 0.0;
        for (int i = 0; i < scores.Length; i++)
        {
            double v = NumOps.ToDouble(scores[i]);
            clamped[i] = v > 0.0 ? v : 0.0;
            sum += clamped[i];
        }

        var result = new Vector<T>(scores.Length);

        if (sum <= 0.0)
        {
            double equal = 1.0 / scores.Length;
            for (int i = 0; i < scores.Length; i++) result[i] = NumOps.FromDouble(equal);
            return result;
        }

        for (int i = 0; i < scores.Length; i++) result[i] = NumOps.FromDouble(clamped[i] / sum);
        return result;
    }

    /// <summary>
    /// Number of assets receiving a strictly positive allocation.
    /// </summary>
    /// <remarks>
    /// The quantity the paper's allocation mechanism exists to keep small, since each held position
    /// carries transaction cost. Worth measuring directly: a softmax allocation would report every
    /// asset as held.
    /// </remarks>
    public int ActivePositions(Vector<T> weights)
    {
        if (weights == null) throw new ArgumentNullException(nameof(weights));

        int count = 0;
        for (int i = 0; i < weights.Length; i++)
            if (NumOps.ToDouble(weights[i]) > 0.0) count++;
        return count;
    }

    /// <summary>
    /// L1 penalty <c>sum |s_i|</c> over the final-layer scores.
    /// </summary>
    /// <remarks>
    /// The paper applies LASSO regularization to the second feed-forward network "to further shrink and
    /// eliminate unnecessary weights". This is the sparsity driver: without it the scores stay dense,
    /// the sum-normalization spreads capital across every asset, and the allocation layer loses the
    /// property that motivated choosing it over softmax.
    /// </remarks>
    public double L1Penalty(Vector<T> scores)
    {
        if (scores == null) throw new ArgumentNullException(nameof(scores));

        double sum = 0.0;
        for (int i = 0; i < scores.Length; i++) sum += Math.Abs(NumOps.ToDouble(scores[i]));
        return sum;
    }
}
