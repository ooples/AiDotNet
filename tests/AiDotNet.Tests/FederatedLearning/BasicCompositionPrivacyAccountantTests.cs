namespace AiDotNet.Tests.FederatedLearning;

using AiDotNet.FederatedLearning.Privacy.Accounting;
using Xunit;

public class BasicCompositionPrivacyAccountantTests
{
    [Theory]
    [InlineData(0.0, 1e-5, 1.0)]
    [InlineData(1.0, 0.0, 1.0)]
    [InlineData(1.0, 1.0, 1.0)]
    [InlineData(1.0, 1e-5, 0.0)]
    [InlineData(1.0, 1e-5, 1.1)]
    public void AddRound_Throws_WhenParametersInvalid(double epsilon, double delta, double samplingRate)
    {
        var accountant = new BasicCompositionPrivacyAccountant();

        Assert.Throws<ArgumentOutOfRangeException>(() => accountant.AddRound(epsilon, delta, samplingRate));
    }

    [Fact]
    public void AddRound_AccumulatesEpsilonAndDelta()
    {
        var accountant = new BasicCompositionPrivacyAccountant();

        accountant.AddRound(epsilon: 0.5, delta: 1e-5, samplingRate: 1.0);
        accountant.AddRound(epsilon: 0.25, delta: 2e-5, samplingRate: 0.5);

        Assert.Equal(0.75, accountant.GetTotalEpsilonConsumed(), precision: 10);
        Assert.Equal(3e-5, accountant.GetTotalDeltaConsumed(), precision: 10);
        Assert.Equal("Basic", accountant.GetAccountantName());
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(1.0)]
    public void GetEpsilonAtDelta_Throws_WhenTargetDeltaInvalid(double targetDelta)
    {
        var accountant = new BasicCompositionPrivacyAccountant();

        Assert.Throws<ArgumentOutOfRangeException>(() => accountant.GetEpsilonAtDelta(targetDelta));
    }

    [Fact]
    public void GetEpsilonAtDelta_ReturnsTotalEpsilon()
    {
        var accountant = new BasicCompositionPrivacyAccountant();
        accountant.AddRound(epsilon: 0.5, delta: 1e-5, samplingRate: 1.0);

        Assert.Equal(0.5, accountant.GetEpsilonAtDelta(1e-5), precision: 10);
    }
}

