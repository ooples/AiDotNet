using System;
using System.Linq;
using AiDotNet.Finance.Trading.Environments;
using AiDotNet.Finance.Trading.Rewards;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Finance.Portfolio;

/// <summary>
/// <see cref="PortfolioManagerEnvironment{T}.ResolveTargetWeights"/> — the seam that lets a caller train under
/// the same allocation rule it will be served under.
///
/// <para><b>Why the seam exists.</b> A production system often applies a richer rule at serving time
/// (inverse-volatility weighting, a per-asset concentration cap, net-exposure limits, volatility targeting)
/// than this environment models. Measured on one such system, the two rules did not merely disagree about
/// scale — they ORDERED the book differently on 32 to 48 of 48 held-out steps, so the name the policy ranked
/// second was served last. A policy trained against a rule it is never served under has learned a different
/// problem.</para>
///
/// <para>These tests pin both halves of the contract: the DEFAULT is byte-for-byte the rule the environment
/// always applied, and an override genuinely reaches the position reconciliation.</para>
/// </summary>
public class PortfolioWeightResolutionSeamTests
{
    private const int WindowSize = 4;
    private const int Bars = 40;
    private const double InitialCapital = 100_000;
    private const double MaxLeverage = 2.0;

    /// <summary>
    /// The default must be exactly the previous behaviour: clamp to [-1, 1], scale only when gross exceeds the
    /// budget. If this changes, every existing caller's training distribution changes with it.
    /// </summary>
    [Theory]
    [Trait("category", "unit")]
    // Inside the budget: passes through untouched.
    [InlineData(new[] { 0.5, -0.4, 0.3 }, new[] { 0.5, -0.4, 0.3 })]
    // Over the budget (gross 2.4 > 2.0): scaled by 2.0/2.4.
    [InlineData(new[] { 1.0, -1.0, 0.4 }, new[] { 0.8333333333, -0.8333333333, 0.3333333333 })]
    // Beyond the per-weight range: clamped BEFORE the gross check.
    [InlineData(new[] { 3.0, 0.0, 0.0 }, new[] { 1.0, 0.0, 0.0 })]
    public void The_default_rule_is_unchanged(double[] raw, double[] expected)
    {
        var environment = new ProbeEnvironment(MaxLeverage);

        var resolved = environment.Resolve(Action(raw));

        for (var i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], resolved[i], 6);
        }
    }

    /// <summary>A non-finite element becomes zero rather than poisoning the whole book.</summary>
    [Fact]
    [Trait("category", "unit")]
    public void The_default_rule_neutralises_a_non_finite_weight()
    {
        var environment = new ProbeEnvironment(MaxLeverage);

        var resolved = environment.Resolve(Action([double.NaN, 0.5, double.PositiveInfinity]));

        Assert.Equal(0.0, resolved[0], 6);
        Assert.Equal(0.5, resolved[1], 6);
        Assert.Equal(0.0, resolved[2], 6);
    }

    /// <summary>
    /// THE POINT OF THE SEAM. An override must actually drive the book — not be computed and discarded.
    /// Asserted through <see cref="PortfolioManagerEnvironment{T}.GrossExposure"/>, which is derived from the
    /// positions the environment actually took, so it cannot pass unless the override reached reconciliation.
    /// </summary>
    [Fact]
    [Trait("category", "unit")]
    public void An_override_reaches_the_position_reconciliation()
    {
        var overridden = new FixedWeightEnvironment([0.2, 0.2, 0.2]);
        overridden.Reset();

        // The agent asks for a full-leverage book; the override insists on 0.6 gross.
        overridden.Step(Action([1.0, 1.0, 1.0]));

        Assert.Equal(0.6, overridden.GrossExposure, 2);
    }

    /// <summary>
    /// And the base class is what it was: the same action through the DEFAULT rule produces a different,
    /// larger book. Without this, the test above could pass against an environment that ignored actions
    /// entirely.
    /// </summary>
    [Fact]
    [Trait("category", "unit")]
    public void The_same_action_under_the_default_rule_produces_a_different_book()
    {
        var baseline = new PortfolioManagerEnvironment<double>(
            Series(), null, WindowSize, InitialCapital, new TotalReturnReward(),
            maxLeverage: MaxLeverage, transactionCost: 0, slippageCoefficient: 0,
            annualBorrowCost: 0, annualHoldingCost: 0, allowShortSelling: true,
            seed: 17);
        baseline.Reset();

        baseline.Step(Action([1.0, 1.0, 1.0]));

        // NOT an exact figure. Gross 3.0 scales to the 2.0 budget, but a long-only book has no short proceeds
        // to fund it and ExecuteTrade caps each buy at the cash on hand — so the realised ratio settles below
        // the budget rather than on it. What the control actually needs is that the default rule produces a
        // MATERIALLY LARGER book than the override's 0.6; the exact number was never the point, and pinning
        // one would be asserting the cash cap rather than the seam.
        Assert.True(
            baseline.GrossExposure > 0.6 + 0.1,
            $"the default rule should build a materially larger book than the override's 0.6; "
            + $"observed {baseline.GrossExposure:F4}");
        Assert.True(
            baseline.GrossExposure <= MaxLeverage + 1e-6,
            $"and it must still respect the leverage budget; observed {baseline.GrossExposure:F4}");
    }

    /// <summary>
    /// A mis-sized override is refused rather than silently trading a book of the wrong width. A short vector
    /// would otherwise leave trailing assets at whatever they happened to hold.
    /// </summary>
    [Fact]
    [Trait("category", "unit")]
    public void An_override_returning_the_wrong_count_is_refused()
    {
        var wrong = new FixedWeightEnvironment([0.2, 0.2]);   // two weights, three assets
        wrong.Reset();

        var ex = Assert.Throws<InvalidOperationException>(() => wrong.Step(Action([1.0, 1.0, 1.0])));

        Assert.Contains("ResolveTargetWeights", ex.Message, StringComparison.Ordinal);
    }

    private static Vector<double> Action(double[] weights)
    {
        var action = new Vector<double>(weights.Length);
        for (var i = 0; i < weights.Length; i++)
        {
            action[i] = weights[i];
        }

        return action;
    }

    private static double[][] Series() =>
        [.. Enumerable.Range(0, 3).Select(asset =>
            Enumerable.Range(0, Bars)
                .Select(i => 100.0 + asset * 10.0 + Math.Sin((i + asset) * 0.3) * 2.0 + i * 0.05)
                .ToArray())];

    /// <summary>
    /// Exposes the protected default so it can be asserted directly, without stepping the book.
    /// </summary>
    /// <remarks>
    /// The constants are qualified by the OUTER class deliberately. Unqualified, <c>WindowSize</c> and
    /// <c>InitialCapital</c> bind to the inherited instance members on <see cref="TradingEnvironment{T}"/>,
    /// which cannot be read in a base-constructor argument list (CS0120).
    /// </remarks>
    private sealed class ProbeEnvironment(double maxLeverage) : PortfolioManagerEnvironment<double>(
        Series(), null,
        PortfolioWeightResolutionSeamTests.WindowSize,
        PortfolioWeightResolutionSeamTests.InitialCapital,
        new TotalReturnReward(),
        maxLeverage: maxLeverage, transactionCost: 0, slippageCoefficient: 0,
        annualBorrowCost: 0, annualHoldingCost: 0, allowShortSelling: true, seed: 17)
    {
        public double[] Resolve(Vector<double> action) => ResolveTargetWeights(action);
    }

    /// <summary>An override that ignores the action entirely — the clearest possible evidence it is in force.</summary>
    private sealed class FixedWeightEnvironment(double[] fixedWeights) : PortfolioManagerEnvironment<double>(
        Series(), null,
        PortfolioWeightResolutionSeamTests.WindowSize,
        PortfolioWeightResolutionSeamTests.InitialCapital,
        new TotalReturnReward(),
        maxLeverage: PortfolioWeightResolutionSeamTests.MaxLeverage,
        transactionCost: 0, slippageCoefficient: 0,
        annualBorrowCost: 0, annualHoldingCost: 0, allowShortSelling: true, seed: 17)
    {
        protected override double[] ResolveTargetWeights(Vector<double> action) => fixedWeights;
    }
}
