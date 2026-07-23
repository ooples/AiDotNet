using System.Collections.Generic;

namespace AiDotNet.Finance.Interfaces;

/// <summary>
/// A bet/position-sizing rule: given an edge, returns the fraction of capital to allocate.
/// </summary>
/// <remarks>
/// <para>
/// This is a customization point, not a trainable model. Trading agents and portfolio models depend on
/// this interface to decide <i>how much</i> to commit to a signal and default to the
/// <see cref="AiDotNet.Finance.Portfolio.KellyCriterion{T}"/> implementation, but callers can substitute
/// a fixed-fraction, volatility-target, or risk-budget sizer without changing the consuming model.
/// </para>
/// <para><b>For Beginners:</b> A "position sizer" answers "how big should this bet be?" The default is the
/// Kelly criterion (the growth-optimal fraction); swap in your own rule if you prefer.</para>
/// </remarks>
/// <typeparam name="T">Numeric type (float/double).</typeparam>
public interface IPositionSizer<T>
{
    /// <summary>Sizing fraction from a discrete win probability and win/loss payoff ratio.</summary>
    T Discrete(T winProbability, T winLossRatio);

    /// <summary>Sizing fraction from the (Gaussian) mean and variance of returns.</summary>
    T Continuous(T expectedReturn, T variance);

    /// <summary>Scales a base sizing fraction (e.g. half-Kelly) by <paramref name="fraction"/>.</summary>
    T Fractional(T baseFraction, double fraction);

    /// <summary>Sizing fraction estimated from a realized return series.</summary>
    T FromReturns(IEnumerable<T> returns, double fraction = 1.0);
}
