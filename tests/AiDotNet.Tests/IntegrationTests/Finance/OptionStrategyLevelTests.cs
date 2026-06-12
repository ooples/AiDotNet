using AiDotNet.Finance.Options;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Finance;

/// <summary>
/// The options risk-classification + broker-level model (#71). Strategies classify into broker-INDEPENDENT
/// archetypes (covered/secured, long premium, defined-risk spread, naked equity, naked index); a
/// configurable <see cref="BrokerOptionsProfile"/> maps those to a broker's 1–5 levels and enforces the
/// authorized tier. These pin the classification and the fail-safe rejection of uncovered risk.
/// </summary>
public class OptionStrategyLevelTests
{
    [Fact]
    public void CoveredCall_CSP_ProtectivePut_classify_as_CoveredOrSecured()
    {
        Assert.Equal(OptionStrategyClass.CoveredOrSecured, OptionStrategy.CoveredCall(105, 0.25).Class);
        Assert.Equal(OptionStrategyClass.CoveredOrSecured, OptionStrategy.CashSecuredPut(95, 0.25).Class);
        Assert.Equal(OptionStrategyClass.CoveredOrSecured, OptionStrategy.ProtectivePut(95, 0.25).Class);
    }

    [Fact]
    public void LongOptions_and_Straddle_classify_as_LongPremium()
    {
        Assert.Equal(OptionStrategyClass.LongPremium, OptionStrategy.LongCall(100, 0.25).Class);
        Assert.Equal(OptionStrategyClass.LongPremium, OptionStrategy.LongPut(100, 0.25).Class);
        Assert.Equal(OptionStrategyClass.LongPremium, OptionStrategy.LongStraddle(100, 0.25).Class);
    }

    [Fact]
    public void Spreads_and_IronCondor_are_DefinedRiskSpread()
    {
        var bull = OptionStrategy.BullCallSpread(100, 110, 0.25);
        var condor = OptionStrategy.IronCondor(90, 95, 105, 110, 0.25);
        Assert.Equal(OptionStrategyClass.DefinedRiskSpread, bull.Class);
        Assert.Equal(OptionStrategyClass.DefinedRiskSpread, condor.Class);
        Assert.True(bull.IsDefinedRisk);
        Assert.True(condor.IsDefinedRisk);
    }

    [Fact]
    public void NakedCall_is_NakedEquity_or_NakedIndex_and_NOT_DefinedRisk()
    {
        var nakedEq = OptionStrategy.NakedCall(100, 0.25);
        var nakedIdx = OptionStrategy.NakedCall(100, 0.25, isIndex: true);
        Assert.Equal(OptionStrategyClass.NakedEquity, nakedEq.Class);
        Assert.Equal(OptionStrategyClass.NakedIndex, nakedIdx.Class);
        Assert.False(nakedEq.IsDefinedRisk);
        Assert.False(nakedIdx.IsDefinedRisk);
    }

    [Fact]
    public void DefaultBrokerProfile_maps_archetypes_to_levels_1_through_5()
    {
        var p = BrokerOptionsProfile.Default; // authorized to Level5, default scheme
        Assert.Equal(OptionsApprovalLevel.Level1, p.LevelFor(OptionStrategyClass.CoveredOrSecured));
        Assert.Equal(OptionsApprovalLevel.Level2, p.LevelFor(OptionStrategyClass.LongPremium));
        Assert.Equal(OptionsApprovalLevel.Level3, p.LevelFor(OptionStrategyClass.DefinedRiskSpread));
        Assert.Equal(OptionsApprovalLevel.Level4, p.LevelFor(OptionStrategyClass.NakedEquity));
        Assert.Equal(OptionsApprovalLevel.Level5, p.LevelFor(OptionStrategyClass.NakedIndex));
    }

    [Fact]
    public void Profile_permits_at_or_below_authorized_and_rejects_above()
    {
        var level2 = new BrokerOptionsProfile(OptionsApprovalLevel.Level2);
        Assert.True(level2.IsPermitted(OptionStrategy.LongCall(100, 0.25)));        // L2 <= L2
        Assert.True(level2.IsPermitted(OptionStrategy.CoveredCall(105, 0.25)));     // L1 <= L2
        Assert.False(level2.IsPermitted(OptionStrategy.BullCallSpread(100, 110, 0.25))); // L3 > L2
        Assert.False(level2.IsPermitted(OptionStrategy.NakedCall(100, 0.25)));      // L4 > L2
        Assert.Contains("Level3", level2.Deny(OptionStrategy.BullCallSpread(100, 110, 0.25)) ?? "");
    }

    [Fact]
    public void NoAuthorization_rejects_everything()
    {
        var none = new BrokerOptionsProfile(OptionsApprovalLevel.None);
        Assert.False(none.IsPermitted(OptionStrategy.LongCall(100, 0.25)));
        Assert.Contains("not authorized", none.Deny(OptionStrategy.LongCall(100, 0.25)) ?? "");
    }

    [Fact]
    public void Configurable_profile_handles_brokers_that_number_levels_differently()
    {
        // A broker that gates cash-secured puts at Level 2 (not Level 1) — the map is configurable.
        var custom = new BrokerOptionsProfile(OptionsApprovalLevel.Level1, new System.Collections.Generic.Dictionary<OptionStrategyClass, OptionsApprovalLevel>
        {
            [OptionStrategyClass.None] = OptionsApprovalLevel.None,
            [OptionStrategyClass.CoveredOrSecured] = OptionsApprovalLevel.Level2,
            [OptionStrategyClass.LongPremium] = OptionsApprovalLevel.Level2,
            [OptionStrategyClass.DefinedRiskSpread] = OptionsApprovalLevel.Level3,
            [OptionStrategyClass.NakedEquity] = OptionsApprovalLevel.Level4,
            [OptionStrategyClass.NakedIndex] = OptionsApprovalLevel.Level5,
        });
        // CSP now requires L2, so a L1-authorized account is denied it under this broker's scheme.
        Assert.False(custom.IsPermitted(OptionStrategy.CashSecuredPut(95, 0.25)));
    }

    [Fact]
    public void BullCallSpread_payoff_is_bounded_both_sides()
    {
        var (maxLoss, maxGain, breakevens) = OptionStrategy.BullCallSpread(100, 110, 0.25).PayoffProfile(netPremium: -4.0);
        Assert.True(maxLoss > -1e6 && maxLoss < 0, "loss bounded and negative");
        Assert.True(maxGain > 0 && maxGain < 1e6, "gain bounded and positive");
        Assert.NotEmpty(breakevens);
    }
}
