namespace AiDotNet.Tests.FederatedLearning;

using AiDotNet.FederatedLearning.Privacy.Accounting;
using Xunit;

public class RdpPrivacyAccountantTests
{
    [Theory]
    [InlineData(0.0)]
    [InlineData(-1.0)]
    public void Constructor_WithNonPositiveClipNorm_Throws(double clipNorm)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new RdpPrivacyAccountant(clipNorm));
    }

    [Fact]
    public void Constructor_WithInvalidOrders_Throws()
    {
        Assert.Throws<ArgumentException>(() => new RdpPrivacyAccountant(clipNorm: 1.0, orders: Array.Empty<double>()));
        Assert.Throws<ArgumentException>(() => new RdpPrivacyAccountant(clipNorm: 1.0, orders: new[] { 1.0, 2.0 }));
    }

    [Theory]
    [InlineData(0.0, 1e-5, 1.0)]
    [InlineData(1.0, 0.0, 1.0)]
    [InlineData(1.0, 1.0, 1.0)]
    [InlineData(1.0, 1e-5, 0.0)]
    [InlineData(1.0, 1e-5, 1.1)]
    public void AddRound_Throws_WhenParametersInvalid(double epsilon, double delta, double samplingRate)
    {
        var accountant = new RdpPrivacyAccountant();

        Assert.Throws<ArgumentOutOfRangeException>(() => accountant.AddRound(epsilon, delta, samplingRate));
    }

    [Fact]
    public void AddRound_AccumulatesDeltaAndIncreasesEpsilon()
    {
        var accountant = new RdpPrivacyAccountant(clipNorm: 1.0);

        accountant.AddRound(epsilon: 1.0, delta: 1e-5, samplingRate: 1.0);
        var epsilon1 = accountant.GetEpsilonAtDelta(1e-5);

        accountant.AddRound(epsilon: 1.0, delta: 1e-5, samplingRate: 1.0);
        var epsilon2 = accountant.GetEpsilonAtDelta(1e-5);

        Assert.Equal(2e-5, accountant.GetTotalDeltaConsumed(), precision: 10);
        Assert.True(epsilon2 > epsilon1, $"Expected epsilon to increase after additional rounds: {epsilon1} -> {epsilon2}");
        Assert.Equal("RDP", accountant.GetAccountantName());
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(1.0)]
    public void GetEpsilonAtDelta_Throws_WhenTargetDeltaInvalid(double targetDelta)
    {
        var accountant = new RdpPrivacyAccountant();

        Assert.Throws<ArgumentOutOfRangeException>(() => accountant.GetEpsilonAtDelta(targetDelta));
    }
}

