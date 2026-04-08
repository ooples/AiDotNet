using AiDotNet.FederatedLearning.Privacy;
using Xunit;

namespace AiDotNet.Tests.FederatedLearning;

public class SecureAggregationVectorTests
{
    [Fact]
    public void Constructor_ThrowsForNonPositiveParameterCount()
    {
        Assert.Throws<ArgumentException>(() => new SecureAggregationVector<double>(parameterCount: 0));
    }

    [Fact]
    public void GeneratePairwiseSecrets_FiltersInvalidClientIds_AndMasksCancelInSum()
    {
        var secureAggregation = new SecureAggregationVector<double>(parameterCount: 2, randomSeed: 123);
        secureAggregation.GeneratePairwiseSecrets(new List<int> { -1, 2, 2, 0 });

        var update0 = new Vector<double>(new[] { 1.0, 2.0 });
        var update2 = new Vector<double>(new[] { 3.0, 4.0 });

        var maskedUpdates = new Dictionary<int, Vector<double>>
        {
            [0] = secureAggregation.MaskUpdate(0, update0),
            [2] = secureAggregation.MaskUpdate(2, update2)
        };

        var sum = secureAggregation.AggregateSumSecurely(maskedUpdates);

        Assert.Equal(4.0, sum[0], precision: 12);
        Assert.Equal(6.0, sum[1], precision: 12);
    }

    [Fact]
    public void MaskUpdate_ThrowsForInvalidInputs()
    {
        var secureAggregation = new SecureAggregationVector<double>(parameterCount: 2, randomSeed: 1);
        secureAggregation.GeneratePairwiseSecrets(new List<int> { 0, 1 });

        Assert.Throws<ArgumentNullException>(() => secureAggregation.MaskUpdate(0, null!));
        Assert.Throws<ArgumentException>(() => secureAggregation.MaskUpdate(0, new Vector<double>(3)));
        Assert.Throws<ArgumentOutOfRangeException>(() => secureAggregation.MaskUpdate(0, new Vector<double>(2), clientWeight: 0.0));
    }

    [Fact]
    public void AggregateSecurely_ThrowsWhenWeightsMissingOrInvalid()
    {
        var secureAggregation = new SecureAggregationVector<double>(parameterCount: 1, randomSeed: 5);
        secureAggregation.GeneratePairwiseSecrets(new List<int> { 0, 1 });

        var maskedUpdates = new Dictionary<int, Vector<double>>
        {
            [0] = secureAggregation.MaskUpdate(0, new Vector<double>(new[] { 1.0 }), clientWeight: 1.0),
            [1] = secureAggregation.MaskUpdate(1, new Vector<double>(new[] { 2.0 }), clientWeight: 1.0)
        };

        Assert.Throws<ArgumentException>(() => secureAggregation.AggregateSecurely(maskedUpdates, clientWeights: new Dictionary<int, double>()));
        Assert.Throws<ArgumentException>(() => secureAggregation.AggregateSecurely(maskedUpdates, clientWeights: new Dictionary<int, double> { [0] = 1.0 }));
        Assert.Throws<ArgumentException>(() => secureAggregation.AggregateSecurely(maskedUpdates, clientWeights: new Dictionary<int, double> { [0] = -1.0, [1] = 0.0 }));
    }

    [Fact]
    public void GeneratePairwiseSecrets_UsesSecureRandom_WhenSeedNotProvided()
    {
        using var secureAggregation = new SecureAggregationVector<double>(parameterCount: 1);
        secureAggregation.GeneratePairwiseSecrets(new List<int> { 0, 1 });
    }

    [Fact]
    public void Dispose_MakesInstanceUnusable()
    {
        var secureAggregation = new SecureAggregationVector<double>(parameterCount: 1, randomSeed: 7);
        secureAggregation.Dispose();

        Assert.Throws<ObjectDisposedException>(() => secureAggregation.ClearSecrets());
    }
}
