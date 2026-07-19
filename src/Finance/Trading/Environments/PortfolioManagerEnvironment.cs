using System;
using System.Collections.Generic;
using System.Globalization;
using AiDotNet.Finance.Trading.Rewards;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Finance.Trading.Environments;

/// <summary>
/// A SOTA cross-sectional portfolio-manager environment: the agent manages a whole book at once. Each step it
/// emits a target-weight VECTOR over the tradable universe (one weight per asset, in [-1, 1], subject to a
/// gross-leverage budget), and the environment reconciles every position toward its target — so entry, exit,
/// sizing, and rotation across names are a single learned decision. This is the cross-sectional counterpart to
/// the single-asset <see cref="TradingEnvironment{T}"/>, and it goes beyond the basic
/// <see cref="PortfolioTradingEnvironment{T}"/> (which forces weights to sum to 1 — always fully invested, no
/// cash, no leverage budget, every column tradable, default return reward, no realistic frictions).
/// <para>
/// It differs in four SOTA-relevant ways: (1) a gross-leverage BUDGET (<c>sum(|w|) &lt;= maxLeverage</c>) so the
/// agent can hold cash and lever deliberately; (2) observation-only FEATURE columns (per-name forecasts,
/// uncertainty, "analyst" recommendations for a hierarchical setup) that ride in the observation but are never
/// traded; (3) a pluggable <see cref="IPortfolioReward"/> objective (total return, differential Sharpe,
/// drawdown-penalized) so the reward is an experiment knob; (4) realistic FRICTIONS — transaction cost,
/// turnover-proportional slippage, short-borrow financing, and gross carry — all deducted from cash so portfolio
/// value reflects true costs. Each asset's position and cash are already in the base observation, so the policy
/// conditions its exits on what it holds.
/// </para>
/// </summary>
/// <typeparam name="T">Element type — <c>float</c> (GPU) or <c>double</c> (CPU/accuracy).</typeparam>
public sealed class PortfolioManagerEnvironment<T> : TradingEnvironment<T>
{
    private readonly int _tradableCount;
    private readonly IPortfolioReward _reward;
    private readonly double _maxLeverage;
    private readonly double _slippageCoefficient;
    private readonly double _borrowCostPerStep;
    private readonly double _holdingCostPerStep;

    private double _lastTurnover;
    private double _grossExposure;
    private double _shortExposure;
    private double _peakValue;

    /// <param name="assetPrices">Price series per TRADABLE asset (all equal length). Column i is tradable asset i.</param>
    /// <param name="featureColumns">Observation-only columns (forecasts/signals/etc.), same length — never traded.</param>
    /// <param name="reward">The objective. Swap it to run reward experiments on the same environment.</param>
    /// <param name="maxLeverage">Gross-leverage budget: target weights are scaled so sum(|w_i|) &lt;= this.</param>
    /// <param name="transactionCost">Proportional per-trade cost (fraction of trade notional).</param>
    /// <param name="slippageCoefficient">Extra cost = coefficient x turnover notional (market-impact proxy).</param>
    /// <param name="annualBorrowCost">Annualized short-borrow financing rate (applied per step / 252).</param>
    /// <param name="annualHoldingCost">Annualized carry/financing on gross exposure (applied per step / 252).</param>
    public PortfolioManagerEnvironment(
        IReadOnlyList<double[]> assetPrices,
        IReadOnlyList<double[]>? featureColumns,
        int windowSize,
        double initialCapital,
        IPortfolioReward reward,
        double maxLeverage = 1.0,
        double transactionCost = 0.001,
        double slippageCoefficient = 0.0005,
        double annualBorrowCost = 0.03,
        double annualHoldingCost = 0.0,
        bool allowShortSelling = true,
        int? seed = null)
        : base(BuildTensor(assetPrices, featureColumns), windowSize, FromDouble(initialCapital), transactionCost,
               allowShortSelling, randomStart: false, maxEpisodeLength: 0, seed: seed)
    {
        // assetPrices is already validated (non-null, non-empty, equal-length series) by BuildTensor, which
        // runs in the base(...) initializer above before this body.
        _tradableCount = assetPrices.Count;
        _reward = reward ?? throw new ArgumentNullException(nameof(reward));
        _maxLeverage = maxLeverage > 0 ? maxLeverage : throw new ArgumentOutOfRangeException(nameof(maxLeverage));
        _slippageCoefficient = Math.Max(0.0, slippageCoefficient);
        _borrowCostPerStep = Math.Max(0.0, annualBorrowCost) / 252.0;
        _holdingCostPerStep = Math.Max(0.0, annualHoldingCost) / 252.0;
        _peakValue = initialCapital;
    }

    /// <summary>One target weight per tradable asset (feature columns are never traded).</summary>
    public override int ActionSpaceSize => _tradableCount;

    /// <summary>Continuous target-weight vector.</summary>
    public override bool IsContinuousActionSpace => true;

    private static double ToDouble(T value) => Convert.ToDouble(value, CultureInfo.InvariantCulture);

    private static T FromDouble(double value) => (T)Convert.ChangeType(value, typeof(T), CultureInfo.InvariantCulture);

    private static Tensor<T> BuildTensor(IReadOnlyList<double[]> assetPrices, IReadOnlyList<double[]>? featureColumns)
    {
        ArgumentNullException.ThrowIfNull(assetPrices);
        if (assetPrices.Count == 0)
        {
            throw new ArgumentException("At least one tradable asset series is required.", nameof(assetPrices));
        }

        if (assetPrices[0] is null || assetPrices[0].Length == 0)
        {
            throw new ArgumentException("Asset series must be non-empty.", nameof(assetPrices));
        }

        int n = assetPrices.Count;
        int m = featureColumns?.Count ?? 0;
        int time = assetPrices[0].Length;

        // Every asset and feature series must be present and share the exact time length. Zero-padding a short
        // or missing series would manufacture prices/signals the model would then trade on, so reject instead.
        for (int i = 0; i < n; i++)
        {
            if (assetPrices[i] is null || assetPrices[i].Length != time)
            {
                throw new ArgumentException(
                    $"All asset series must be non-null and of equal length ({time}); series {i} does not match.",
                    nameof(assetPrices));
            }
        }

        for (int j = 0; j < m; j++)
        {
            if (featureColumns![j] is null || featureColumns[j].Length != time)
            {
                throw new ArgumentException(
                    $"All feature series must be non-null and match the asset-series length ({time}); feature {j} does not match.",
                    nameof(featureColumns));
            }
        }

        int assets = n + m;
        var data = new T[time * assets];
        for (int t = 0; t < time; t++)
        {
            for (int i = 0; i < n; i++)
            {
                data[(t * assets) + i] = FromDouble(assetPrices[i][t]);
            }

            for (int j = 0; j < m; j++)
            {
                data[(t * assets) + n + j] = FromDouble(featureColumns![j][t]);
            }
        }

        return new Tensor<T>([time, assets], new Vector<T>(data));
    }

    /// <summary>
    /// Reconciles every tradable position toward the action's target weights, then applies financing frictions.
    /// Target weights are clamped to [-1, 1] and scaled down so gross leverage sum(|w|) does not exceed the
    /// budget. Only assets [0, tradableCount) are traded; feature columns keep a zero position and never affect
    /// portfolio value.
    /// </summary>
    protected override void ApplyAction(Vector<T> action, Vector<T> prices)
    {
        if (action.Length != _tradableCount)
        {
            throw new ArgumentException(
                $"Action length {action.Length} must equal the tradable-asset count {_tradableCount}.", nameof(action));
        }

        double portfolioValue = ToDouble(_portfolioValue);
        if (!(portfolioValue > 0))
        {
            _lastTurnover = 0;
            return;
        }

        // Decode + clamp target weights, then enforce the gross-leverage budget.
        var weights = new double[_tradableCount];
        double gross = 0;
        for (int i = 0; i < _tradableCount; i++)
        {
            double w = ToDouble(action[i]);
            if (double.IsNaN(w) || double.IsInfinity(w))
            {
                w = 0.0;
            }

            w = Math.Clamp(w, -1.0, 1.0);
            weights[i] = w;
            gross += Math.Abs(w);
        }

        if (gross > _maxLeverage && gross > 0)
        {
            double scale = _maxLeverage / gross;
            for (int i = 0; i < _tradableCount; i++)
            {
                weights[i] *= scale;
            }
        }

        // Reconcile each position toward its exact target weight. Sells run first (they free cash), then buys —
        // so buy sizing sees the cash the sells released. Turnover accumulates the quantity ACTUALLY filled
        // (ExecuteTrade caps a buy to available cash and clamps a no-short sell), not the requested delta.
        double turnoverNotional = 0;
        for (int pass = 0; pass < 2; pass++)
        {
            for (int i = 0; i < _tradableCount; i++)
            {
                double price = ToDouble(prices[i]);
                if (!(price > 0))
                {
                    continue;
                }

                double desiredUnits = weights[i] * (portfolioValue / price);
                if (double.IsNaN(desiredUnits) || double.IsInfinity(desiredUnits))
                {
                    continue;
                }

                double delta = desiredUnits - ToDouble(_positions[i]);
                bool isSell = delta < 0;
                if ((pass == 0) != isSell)
                {
                    continue; // pass 0 = sells, pass 1 = buys
                }

                if (!isSell)
                {
                    // Cost-aware fill: a buy can only draw the cash on hand after the transaction and slippage
                    // charges it will incur, so cap the increase to what cash affords rather than reserving a
                    // fixed headroom fraction. This also keeps ExecuteTrade (which checks cash before filling)
                    // from silently rejecting the whole order.
                    double unitCost = price * (1.0 + TransactionCost + _slippageCoefficient);
                    double affordable = unitCost > 0 ? ToDouble(_cash) / unitCost : 0.0;
                    if (delta > affordable)
                    {
                        delta = affordable;
                    }

                    if (delta <= 0)
                    {
                        continue;
                    }
                }

                double before = ToDouble(_positions[i]);
                ExecuteTrade(i, FromDouble(delta), prices[i]);
                double filled = ToDouble(_positions[i]) - before;
                turnoverNotional += Math.Abs(filled) * price;
            }
        }

        _lastTurnover = turnoverNotional / portfolioValue;

        // Post-trade positions marked at current prices.
        double grossNotional = 0, shortNotional = 0, positionValue = 0;
        for (int i = 0; i < _tradableCount; i++)
        {
            double notional = ToDouble(_positions[i]) * ToDouble(prices[i]);
            positionValue += notional;
            grossNotional += Math.Abs(notional);
            if (notional < 0)
            {
                shortNotional += -notional;
            }
        }

        // Financing frictions are deducted from cash FIRST — turnover-proportional slippage + short-borrow +
        // gross carry (the per-trade transaction cost is already applied inside ExecuteTrade) — THEN exposures
        // are taken against the resulting post-cost NAV, so the cached ratios and the reward context reflect
        // every cost rather than the pre-cost book.
        double financing = (_slippageCoefficient * turnoverNotional)
                           + (_borrowCostPerStep * shortNotional)
                           + (_holdingCostPerStep * grossNotional);
        if (financing > 0 && double.IsFinite(financing))
        {
            _cash = FromDouble(ToDouble(_cash) - financing);
        }

        double nav = ToDouble(_cash) + positionValue;
        if (nav > 0 && double.IsFinite(nav))
        {
            _grossExposure = grossNotional / nav;
            _shortExposure = shortNotional / nav;
        }
        else
        {
            _grossExposure = 0;
            _shortExposure = 0;
        }
    }

    /// <summary>
    /// Reward = the pluggable objective applied to this step's net portfolio return, with turnover, exposure, and
    /// drawdown context. Frictions are already reflected in the return (deducted from cash in
    /// <see cref="ApplyAction"/>), so the objective sees the true, cost-net outcome.
    /// </summary>
    protected override T ComputeReward(T previousValue, T currentValue)
    {
        double previous = ToDouble(previousValue);
        double current = ToDouble(currentValue);
        double portfolioReturn = previous > 0 && double.IsFinite(previous) && double.IsFinite(current)
            ? (current - previous) / previous
            : 0.0;

        if (current > _peakValue)
        {
            _peakValue = current;
        }
        double drawdown = _peakValue > 0 ? Math.Max(0.0, (_peakValue - current) / _peakValue) : 0.0;

        var context = new PortfolioRewardContext(portfolioReturn, _lastTurnover, _grossExposure, _shortExposure, drawdown);
        double reward = _reward.Reward(in context);
        return FromDouble(double.IsNaN(reward) || double.IsInfinity(reward) ? 0.0 : reward);
    }

    /// <summary>Turnover (fraction of portfolio value traded) on the most recent step — for logging/tests.</summary>
    public double LastTurnover => _lastTurnover;

    /// <summary>Gross exposure after the most recent step (sum |position notional| / value).</summary>
    public double GrossExposure => _grossExposure;

    /// <summary>Current portfolio value (cash + marked positions). Equals the initial capital right after
    /// <see cref="TradingEnvironment{T}.Reset"/>, before any step — the pre-trade baseline a backtest seeds with.</summary>
    public double CurrentValue => ToDouble(_portfolioValue);

    /// <summary>
    /// Resets the per-episode reward statistics and drawdown peak. Call after <see cref="TradingEnvironment{T}.Reset"/>
    /// when reusing one environment instance across independent episodes (e.g. evaluation). Not needed for the
    /// default single-pass training configuration (one episode over the full series).
    /// </summary>
    public void ResetEpisodeState()
    {
        _reward.Reset();
        _peakValue = ToDouble(InitialCapital);
        _lastTurnover = 0;
        _grossExposure = 0;
        _shortExposure = 0;
    }
}
