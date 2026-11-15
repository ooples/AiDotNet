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
public class PortfolioManagerEnvironment<T> : TradingEnvironment<T>
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
        if (assetPrices is null) throw new ArgumentNullException(nameof(assetPrices));
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
            if ((featureColumns ?? throw new InvalidOperationException("featureColumns is null where it is required."))[j] is null || featureColumns[j].Length != time)
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
                data[(t * assets) + n + j] = FromDouble((featureColumns ?? throw new InvalidOperationException("featureColumns is null where it is required."))[j][t]);
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

        T portfolioValue = _portfolioValue;
        if (!NumOps.GreaterThan(portfolioValue, NumOps.Zero))
        {
            _lastTurnover = 0;
            return;
        }

        // Decode + clamp target weights, then enforce the gross-leverage budget.
        var weights = ResolveTargetWeights(action);
        if (weights is null || weights.Length != _tradableCount)
        {
            throw new InvalidOperationException(
                $"{nameof(ResolveTargetWeights)} must return one weight per tradable asset "
                + $"({_tradableCount}); got {(weights is null ? "null" : weights.Length.ToString(CultureInfo.InvariantCulture))}.");
        }

        var engine = Engine;
        var tradablePrices = prices.GetSubVector(0, _tradableCount);
        var positions = _positions.GetSubVector(0, _tradableCount);
        var zeros = new Tensor<T>(new[] { _tradableCount });

        // An asset without a finite positive price (a gap in the data) is not traded this step, as before. Its price
        // is replaced by zero in every cash reduction below, so a NaN price never reaches the cash (0 * NaN is NaN);
        // the post-trade marking still uses the real prices.
        var priceTensor = tradablePrices.AsTensor();
        var tradable = engine.TensorWhere(
            engine.TensorIsFinite(priceTensor), engine.TensorGreaterThan(priceTensor, NumOps.Zero), zeros);
        var tradePrices = engine.TensorWhere(tradable, priceTensor, zeros).ToVector();

        // Reconcile each position toward its exact target weight: delta = w * V / p - position. An untradable asset,
        // or one whose target is not finite, is not traded.
        var desiredUnits = engine.Divide(engine.Multiply(weights, portfolioValue), tradePrices);
        var finiteDelta = engine.TensorWhere(
            engine.TensorIsFinite(desiredUnits.AsTensor()),
            engine.Subtract(desiredUnits, positions).AsTensor(),
            zeros);
        var delta = engine.TensorWhere(tradable, finiteDelta, zeros).ToVector();

        // Sells run first (they free cash). With short selling off a sell never takes a position below zero.
        var buys = engine.ReLU(delta);
        var sells = engine.Subtract(delta, buys);
        if (!AllowShortSelling)
        {
            sells = engine.Max(sells, engine.Negate(engine.ReLU(positions)));
        }

        T soldNotional = NumOps.Negate(engine.DotProduct(sells, tradePrices));
        _cash = NumOps.Add(_cash, NumOps.Multiply(soldNotional, NumOps.FromDouble(1.0 - TransactionCost)));
        positions = engine.Add(positions, sells);

        // Buys see the cash the sells released. Each unit costs p * (1 + c + slippage); when the whole buy book costs
        // more than the cash on hand every buy is scaled by the same fraction, so the fill does not depend on the
        // order the assets are listed in (the old per-asset loop filled low-index assets first and starved the rest).
        T unitCostFactor = NumOps.FromDouble(1.0 + TransactionCost + _slippageCoefficient);
        T required = NumOps.Multiply(engine.DotProduct(buys, tradePrices), unitCostFactor);
        // The financing deduction below also charges slippage on the sells, so the buys may spend only the cash left
        // after that charge; otherwise funding buys from sells could end the step with negative cash.
        T buyCash = NumOps.Subtract(_cash, NumOps.Multiply(soldNotional, NumOps.FromDouble(_slippageCoefficient)));
        if (NumOps.GreaterThan(required, buyCash))
        {
            T fraction = NumOps.GreaterThan(buyCash, NumOps.Zero) ? NumOps.Divide(buyCash, required) : NumOps.Zero;
            buys = engine.Multiply(buys, fraction);
        }

        T boughtNotional = engine.DotProduct(buys, tradePrices);
        _cash = NumOps.Subtract(_cash, NumOps.Multiply(boughtNotional, NumOps.FromDouble(1.0 + TransactionCost)));
        positions = engine.Add(positions, buys);
        positions.AsSpan().CopyTo(_positions.AsWritableSpan());

        // Turnover is the quantity actually filled, not the requested delta.
        T turnoverNotional = NumOps.Add(soldNotional, boughtNotional);
        _lastTurnover = NumOps.ToDouble(NumOps.Divide(turnoverNotional, portfolioValue));

        // Post-trade positions marked at current prices.
        var notional = engine.Multiply(positions, tradablePrices);
        T positionValue = engine.Sum(notional);
        T grossNotional = engine.Sum(engine.Abs(notional));
        T shortNotional = engine.Sum(engine.ReLU(engine.Negate(notional)));

        // Financing frictions are deducted from cash FIRST (turnover-proportional slippage + short-borrow +
        // gross carry; the per-trade transaction cost is already applied above), THEN exposures are taken
        // against the resulting post-cost NAV, so the cached ratios and the reward context reflect every cost
        // rather than the pre-cost book.
        T financing = NumOps.Add(
            NumOps.Add(
                NumOps.Multiply(NumOps.FromDouble(_slippageCoefficient), turnoverNotional),
                NumOps.Multiply(NumOps.FromDouble(_borrowCostPerStep), shortNotional)),
            NumOps.Multiply(NumOps.FromDouble(_holdingCostPerStep), grossNotional));
        if (NumOps.GreaterThan(financing, NumOps.Zero) && IsFinite(financing))
        {
            _cash = NumOps.Subtract(_cash, financing);
        }

        T nav = NumOps.Add(_cash, positionValue);
        if (NumOps.GreaterThan(nav, NumOps.Zero) && IsFinite(nav))
        {
            _grossExposure = NumOps.ToDouble(NumOps.Divide(grossNotional, nav));
            _shortExposure = NumOps.ToDouble(NumOps.Divide(shortNotional, nav));
        }
        else
        {
            _grossExposure = 0;
            _shortExposure = 0;
        }
    }

    private bool IsFinite(T value) => !NumOps.IsNaN(value) && !NumOps.IsInfinity(value);

    /// <summary>
    /// Turns the agent's raw action into the target weight per tradable asset, before any position is
    /// reconciled toward it.
    /// </summary>
    /// <param name="action">The agent's raw output, one element per tradable asset.</param>
    /// <returns>One target weight per tradable asset, in the same order.</returns>
    /// <remarks>
    /// <para><b>The default is the environment's own rule, unchanged:</b> a non-finite element becomes zero,
    /// each weight is clamped to <c>[-1, 1]</c>, and the whole vector is scaled down only if gross leverage
    /// <c>sum(|w|)</c> exceeds the budget. Overriding is opt-in and the base behaviour is what it always was.</para>
    ///
    /// <para><b>Why this is a seam.</b> A production system frequently applies a richer allocation rule at
    /// SERVING time — inverse-volatility weighting, a per-asset concentration cap, net-exposure limits,
    /// portfolio-volatility targeting — that this environment does not model. When training applies one rule
    /// and serving applies another, the two can disagree about more than scale: they can ORDER the book
    /// differently, so the name a policy ranked second is served last. A policy trained against a rule it will
    /// never be served under has learned a different problem. Overriding this method lets a caller train under
    /// the same allocation rule it deploys, which is the only way those two agree by construction rather than
    /// by coincidence.</para>
    ///
    /// <para>Implementations must return exactly <c>tradableCount</c> weights; the caller verifies this and
    /// throws rather than silently trading a mis-sized book.</para>
    /// </remarks>
    protected virtual Vector<T> ResolveTargetWeights(Vector<T> action)
    {
        var engine = Engine;
        var raw = action.GetSubVector(0, _tradableCount).AsTensor();
        var finite = engine.TensorWhere(engine.TensorIsFinite(raw), raw, new Tensor<T>(new[] { _tradableCount }));
        var weights = engine.Clamp(finite.ToVector(), NumOps.Negate(NumOps.One), NumOps.One);

        T gross = engine.Sum(engine.Abs(weights));
        T maxLeverage = NumOps.FromDouble(_maxLeverage);
        if (NumOps.GreaterThan(gross, maxLeverage) && NumOps.GreaterThan(gross, NumOps.Zero))
        {
            weights = engine.Multiply(weights, NumOps.Divide(maxLeverage, gross));
        }

        return weights;
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
        double portfolioReturn = previous > 0 && (!double.IsNaN(previous) && !double.IsInfinity(previous)) && (!double.IsNaN(current) && !double.IsInfinity(current))
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
    /// Resets the per-episode reward statistics, drawdown peak, turnover and exposure.
    /// <see cref="TradingEnvironment{T}.Reset"/> now calls this automatically (through
    /// <see cref="OnReset"/>), so every episode starts from the same baseline; calling it explicitly as well is
    /// harmless (it is idempotent).
    /// </summary>
    public void ResetEpisodeState()
    {
        _reward.Reset();
        _peakValue = ToDouble(InitialCapital);
        _lastTurnover = 0;
        _grossExposure = 0;
        _shortExposure = 0;
    }

    /// <summary>
    /// Clears this environment's per-episode state whenever <see cref="TradingEnvironment{T}.Reset"/> starts a
    /// new episode — without it, reward statistics and the drawdown peak leaked from one episode into the next.
    /// </summary>
    protected override void OnReset() => ResetEpisodeState();
}
