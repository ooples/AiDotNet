namespace AiDotNet.Tests.FederatedLearning;

using AiDotNet.FederatedLearning.Privacy;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

public class GaussianDifferentialPrivacyVectorTests
{
    [Theory]
    [InlineData(0.0)]
    [InlineData(-1.0)]
    public void Constructor_WithNonPositiveClipNorm_Throws(double clipNorm)
    {
        Assert.Throws<ArgumentException>(() => new GaussianDifferentialPrivacyVector<double>(clipNorm));
    }

    [Fact]
    public void ApplyPrivacy_Throws_WhenModelNull()
    {
        var dp = new GaussianDifferentialPrivacyVector<double>(clipNorm: 1.0, randomSeed: 1);

        Assert.Throws<ArgumentNullException>(() => dp.ApplyPrivacy(model: null!, epsilon: 1.0, delta: 1e-5));
    }

    [Theory]
    [InlineData(0.0, 1e-5)]
    [InlineData(-1.0, 1e-5)]
    [InlineData(1.0, 0.0)]
    [InlineData(1.0, 1.0)]
    public void ApplyPrivacy_Throws_WhenEpsilonOrDeltaInvalid(double epsilon, double delta)
    {
        var dp = new GaussianDifferentialPrivacyVector<double>(clipNorm: 1.0, randomSeed: 1);
        var model = new Vector<double>(new[] { 1.0, 2.0 });

        Assert.Throws<ArgumentException>(() => dp.ApplyPrivacy(model, epsilon: epsilon, delta: delta));
    }

    [Fact]
    public void ApplyPrivacy_IncrementsAndResetsPrivacyBudget()
    {
        var dp = new GaussianDifferentialPrivacyVector<double>(clipNorm: 1.0, randomSeed: 123);
        var model = new Vector<double>(new[] { 1.0, 2.0 });

        dp.ApplyPrivacy(model, epsilon: 0.5, delta: 1e-5);
        dp.ApplyPrivacy(model, epsilon: 0.25, delta: 1e-5);

        Assert.Equal(0.75, dp.GetPrivacyBudgetConsumed(), precision: 10);

        dp.ResetPrivacyBudget();
        Assert.Equal(0.0, dp.GetPrivacyBudgetConsumed(), precision: 10);
    }

    [Fact]
    public void ApplyPrivacy_WithSameSeed_ProducesSameNoise()
    {
        var dp1 = new GaussianDifferentialPrivacyVector<double>(clipNorm: 1.0, randomSeed: 42);
        var dp2 = new GaussianDifferentialPrivacyVector<double>(clipNorm: 1.0, randomSeed: 42);
        var model = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

        var noisy1 = dp1.ApplyPrivacy(model, epsilon: 1.0, delta: 1e-5);
        var noisy2 = dp2.ApplyPrivacy(model, epsilon: 1.0, delta: 1e-5);

        Assert.Equal(noisy1.ToArray(), noisy2.ToArray());
    }

    [Fact]
    public void ApplyPrivacy_ClipsByL2Norm_BeforeAddingNoise()
    {
        var dp = new GaussianDifferentialPrivacyVector<double>(clipNorm: 1.0, randomSeed: 7);
        var model = new Vector<double>(new[] { 10.0, 10.0, 10.0 });

        var noisy = dp.ApplyPrivacy(model, epsilon: 1000.0, delta: 0.99);

        var values = noisy.ToArray();
        var l2Norm = Math.Sqrt(values.Sum(v => v * v));

        Assert.True(l2Norm <= 1.05, $"Expected clipped L2 norm close to 1.0, got {l2Norm}.");
        Assert.Equal(1.0, dp.GetClipNorm(), precision: 10);
        Assert.Equal("Gaussian Mechanism (Vector)", dp.GetMechanismName());
    }
}

