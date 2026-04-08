using AiDotNet.FederatedLearning.Privacy;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.FederatedLearning;

public class ThresholdSecureAggregationVectorTests
{
    [Fact(Timeout = 60000)]
    public async Task Constructor_ThrowsForNonPositiveParameterCount()
    {
        Assert.Throws<ArgumentException>(() => new ThresholdSecureAggregationVector<double>(parameterCount: 0));
    }

    [Fact(Timeout = 60000)]
    public async Task InitializeRound_FiltersInvalidClientIds_AndComputesDefaults()
    {
        var secureAggregation = new ThresholdSecureAggregationVector<double>(parameterCount: 2, randomSeed: 123);
        secureAggregation.InitializeRound(new List<int> { -1, 2, 2, 0, 1 });

        Assert.Equal(2, secureAggregation.MinimumUploaderCount);
        Assert.Equal(2, secureAggregation.ReconstructionThreshold);
    }

    [Fact(Timeout = 60000)]
    public async Task MasksCancel_WithFullParticipation()
    {
        var secureAggregation = new ThresholdSecureAggregationVector<double>(parameterCount: 2, randomSeed: 123);
        secureAggregation.InitializeRound(new List<int> { 0, 1, 2 });

        var update0 = new Vector<double>(new[] { 1.0, 2.0 });
        var update1 = new Vector<double>(new[] { 3.0, 4.0 });
        var update2 = new Vector<double>(new[] { 5.0, 6.0 });

        var maskedUpdates = new Dictionary<int, Vector<double>>
        {
            [0] = secureAggregation.MaskUpdate(0, update0),
            [1] = secureAggregation.MaskUpdate(1, update1),
            [2] = secureAggregation.MaskUpdate(2, update2)
        };

        var sum = secureAggregation.AggregateSumSecurely(maskedUpdates);

        Assert.Equal(9.0, sum[0], precision: 12);
        Assert.Equal(12.0, sum[1], precision: 12);
    }

    [Fact(Timeout = 60000)]
    public async Task AggregateSumSecurely_Succeeds_WithUploadDropout()
    {
        var secureAggregation = new ThresholdSecureAggregationVector<double>(parameterCount: 1, randomSeed: 5);
        secureAggregation.InitializeRound(new List<int> { 0, 1, 2, 3 });

        var maskedUpdates = new Dictionary<int, Vector<double>>
        {
            [0] = secureAggregation.MaskUpdate(0, new Vector<double>(new[] { 1.0 })),
            [2] = secureAggregation.MaskUpdate(2, new Vector<double>(new[] { 2.0 })),
            [3] = secureAggregation.MaskUpdate(3, new Vector<double>(new[] { 3.0 }))
        };

        var sum = secureAggregation.AggregateSumSecurely(maskedUpdates);

        Assert.Equal(6.0, sum[0], precision: 12);
    }

    [Fact(Timeout = 60000)]
    public async Task AggregateSumSecurely_Succeeds_WithUnmaskingDropout_ByReconstructingSelfMask()
    {
        var secureAggregation = new ThresholdSecureAggregationVector<double>(parameterCount: 1, randomSeed: 7);
        secureAggregation.InitializeRound(new List<int> { 0, 1, 2 });

        var maskedUpdates = new Dictionary<int, Vector<double>>
        {
            [0] = secureAggregation.MaskUpdate(0, new Vector<double>(new[] { 1.0 })),
            [1] = secureAggregation.MaskUpdate(1, new Vector<double>(new[] { 2.0 })),
            [2] = secureAggregation.MaskUpdate(2, new Vector<double>(new[] { 3.0 }))
        };

        var sum = secureAggregation.AggregateSumSecurely(maskedUpdates, unmaskingClientIds: new[] { 0, 1 });

        Assert.Equal(6.0, sum[0], precision: 12);
    }

    [Fact(Timeout = 60000)]
    public async Task AggregateSumSecurely_Throws_WhenUnmaskersBelowThreshold()
    {
        var secureAggregation = new ThresholdSecureAggregationVector<double>(parameterCount: 1, randomSeed: 11);
        secureAggregation.InitializeRound(new List<int> { 0, 1, 2 }, minimumUploaderCount: 3, reconstructionThreshold: 3);

        var maskedUpdates = new Dictionary<int, Vector<double>>
        {
            [0] = secureAggregation.MaskUpdate(0, new Vector<double>(new[] { 1.0 })),
            [1] = secureAggregation.MaskUpdate(1, new Vector<double>(new[] { 2.0 })),
            [2] = secureAggregation.MaskUpdate(2, new Vector<double>(new[] { 3.0 }))
        };

        var ex = Assert.Throws<InvalidOperationException>(() =>
            secureAggregation.AggregateSumSecurely(maskedUpdates, unmaskingClientIds: new[] { 0, 1 }));

        Assert.Contains("unmasking", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact(Timeout = 60000)]
    public async Task AggregateSecurely_ReturnsWeightedAverage()
    {
        var secureAggregation = new ThresholdSecureAggregationVector<double>(parameterCount: 2, randomSeed: 1);
        secureAggregation.InitializeRound(new List<int> { 0, 1 });

        var update0 = new Vector<double>(new[] { 1.0, 3.0 });
        var update1 = new Vector<double>(new[] { 5.0, 7.0 });

        var maskedUpdates = new Dictionary<int, Vector<double>>
        {
            [0] = secureAggregation.MaskUpdate(0, update0, clientWeight: 2.0),
            [1] = secureAggregation.MaskUpdate(1, update1, clientWeight: 1.0)
        };

        var weights = new Dictionary<int, double> { [0] = 2.0, [1] = 1.0 };
        var averaged = secureAggregation.AggregateSecurely(maskedUpdates, weights);

        Assert.Equal(7.0 / 3.0, averaged[0], precision: 12);
        Assert.Equal(13.0 / 3.0, averaged[1], precision: 12);
    }

    [Fact(Timeout = 60000)]
    public async Task Dispose_MakesInstanceUnusable()
    {
        var secureAggregation = new ThresholdSecureAggregationVector<double>(parameterCount: 1, randomSeed: 7);
        secureAggregation.Dispose();

        Assert.Throws<ObjectDisposedException>(() => secureAggregation.ClearSecrets());
    }
}

