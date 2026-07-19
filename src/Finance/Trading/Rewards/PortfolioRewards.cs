using System;

namespace AiDotNet.Finance.Trading.Rewards;

/// <summary>
/// The information a portfolio reward function needs at each environment step. All values are in double
/// precision (reward shaping is scale-sensitive and does not need the environment's generic numeric type):
/// the realized one-step portfolio return (ALREADY net of trading frictions, which the environment deducts
/// from cash), plus exposure/turnover/drawdown context so a reward can penalize churn, leverage, or losses.
/// </summary>
public readonly struct PortfolioRewardContext
{
    /// <summary>One-step portfolio return, net of all frictions the environment applied this step.</summary>
    public double PortfolioReturn { get; }

    /// <summary>Fraction of portfolio value traded this step (sum of |Δnotional| / value) — turnover.</summary>
    public double Turnover { get; }

    /// <summary>Gross exposure after the step: sum of |position notional| / portfolio value (1 = fully invested).</summary>
    public double GrossExposure { get; }

    /// <summary>Short exposure after the step: sum of short position notional / portfolio value (>= 0).</summary>
    public double ShortExposure { get; }

    /// <summary>Current drawdown from the running peak portfolio value, in [0, 1].</summary>
    public double Drawdown { get; }

    public PortfolioRewardContext(double portfolioReturn, double turnover, double grossExposure, double shortExposure, double drawdown)
    {
        PortfolioReturn = portfolioReturn;
        Turnover = turnover;
        GrossExposure = grossExposure;
        ShortExposure = shortExposure;
        Drawdown = drawdown;
    }
}

/// <summary>
/// A pluggable portfolio reward function for reinforcement-learning trading agents. Making the objective
/// swappable is what lets the same environment run the reward as an EXPERIMENT — total return vs a
/// risk-adjusted (Sharpe) objective vs a drawdown-penalized one — and rank them on an untouched holdout.
/// Stateful rewards (e.g. the online Sharpe) must reset their running statistics in <see cref="Reset"/>.
/// </summary>
public interface IPortfolioReward
{
    /// <summary>Reward for one environment step given its <paramref name="context"/>.</summary>
    double Reward(in PortfolioRewardContext context);

    /// <summary>Clears any running statistics at the start of an episode.</summary>
    void Reset();
}

/// <summary>
/// Plain total-return reward with optional turnover and drawdown penalties: <c>return − turnoverPenalty·turnover
/// − drawdownPenalty·drawdown</c>. The simplest baseline objective; use it as the control an
/// <see cref="IPortfolioReward"/> experiment compares against.
/// </summary>
public sealed class TotalReturnReward : IPortfolioReward
{
    private readonly double _turnoverPenalty;
    private readonly double _drawdownPenalty;

    public TotalReturnReward(double turnoverPenalty = 0.0, double drawdownPenalty = 0.0)
    {
        if ((double.IsNaN(turnoverPenalty) || double.IsInfinity(turnoverPenalty)) || turnoverPenalty < 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(turnoverPenalty), "Penalty must be finite and non-negative.");
        }

        if ((double.IsNaN(drawdownPenalty) || double.IsInfinity(drawdownPenalty)) || drawdownPenalty < 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(drawdownPenalty), "Penalty must be finite and non-negative.");
        }

        _turnoverPenalty = turnoverPenalty;
        _drawdownPenalty = drawdownPenalty;
    }

    public double Reward(in PortfolioRewardContext context)
    {
        double r = context.PortfolioReturn - (_turnoverPenalty * context.Turnover) - (_drawdownPenalty * context.Drawdown);
        return double.IsNaN(r) || double.IsInfinity(r) ? 0.0 : r;
    }

    public void Reset()
    {
    }
}

/// <summary>
/// Differential Sharpe Ratio reward (Moody &amp; Saffell, 1998) — the canonical risk-adjusted objective for RL
/// trading. It rewards the MARGINAL contribution of each step's return to the online Sharpe ratio, so the agent
/// is pushed toward smooth, consistent gains rather than volatile ones (a big gain that spikes variance can
/// score LOWER than a smaller steady gain). Maintains exponentially-weighted first (A) and second (B) moments of
/// the return with decay <c>eta</c>; the differential is
/// <c>D_t = (B·ΔA − 0.5·A·ΔB) / (B − A²)^{3/2}</c>, which is the derivative of the Sharpe ratio w.r.t. the EWMA
/// weight — a per-step, online, backprop-free shaping signal. An optional turnover penalty is subtracted.
/// </summary>
public sealed class DifferentialSharpeReward : IPortfolioReward
{
    private readonly double _eta;
    private readonly double _turnoverPenalty;
    private double _a; // EWMA of returns
    private double _b; // EWMA of squared returns
    private bool _initialized;

    /// <param name="eta">EWMA decay for the moment estimates (smaller = longer memory). Default 0.04.</param>
    /// <param name="turnoverPenalty">Optional penalty per unit turnover, subtracted from the differential.</param>
    public DifferentialSharpeReward(double eta = 0.04, double turnoverPenalty = 0.0)
    {
        if ((double.IsNaN(eta) || double.IsInfinity(eta)) || eta <= 0.0 || eta >= 1.0)
        {
            throw new ArgumentOutOfRangeException(nameof(eta), "eta must be finite and strictly between 0 and 1.");
        }

        if ((double.IsNaN(turnoverPenalty) || double.IsInfinity(turnoverPenalty)) || turnoverPenalty < 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(turnoverPenalty), "Penalty must be finite and non-negative.");
        }

        _eta = eta;
        _turnoverPenalty = turnoverPenalty;
    }

    public double Reward(in PortfolioRewardContext context)
    {
        double r = context.PortfolioReturn;
        if (double.IsNaN(r) || double.IsInfinity(r))
        {
            r = 0.0;
        }

        double reward;
        if (!_initialized)
        {
            // Bootstrap: no variance estimate yet, so the marginal Sharpe is undefined — fall back to the raw
            // return so the very first step still gives a sensible signal.
            _a = r;
            _b = r * r;
            _initialized = true;
            reward = r;
        }
        else
        {
            double deltaA = r - _a;
            double deltaB = (r * r) - _b;
            double variance = _b - (_a * _a);
            double denom = variance > 1e-12 ? Math.Pow(variance, 1.5) : 0.0;
            reward = denom > 0.0 ? ((_b * deltaA) - (0.5 * _a * deltaB)) / denom : r;

            _a += _eta * deltaA;
            _b += _eta * deltaB;
        }

        reward -= _turnoverPenalty * context.Turnover;
        return double.IsNaN(reward) || double.IsInfinity(reward) ? 0.0 : reward;
    }

    public void Reset()
    {
        _a = 0.0;
        _b = 0.0;
        _initialized = false;
    }
}
