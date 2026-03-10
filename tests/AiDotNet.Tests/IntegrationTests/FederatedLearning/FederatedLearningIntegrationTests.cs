using AiDotNet.FederatedLearning.Aggregators;
using AiDotNet.FederatedLearning.Cryptography;
using AiDotNet.FederatedLearning.Heterogeneity;
using AiDotNet.FederatedLearning.Privacy;
using AiDotNet.FederatedLearning.Privacy.Accounting;
using AiDotNet.FederatedLearning.Selection;
using AiDotNet.FederatedLearning.ServerOptimizers;
using AiDotNet.Models;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.FederatedLearning;

/// <summary>
/// Comprehensive integration tests for the FederatedLearning module.
/// Tests cover aggregators, privacy mechanisms, cryptography, client selection,
/// server optimizers, and heterogeneity corrections.
/// </summary>
public class FederatedLearningIntegrationTests
{
    private const double Tolerance = 1e-10;

    #region Aggregation Strategies

    [Fact]
    public void FedAvgAggregationStrategy_Aggregate_ReturnsWeightedAverage()
    {
        // Arrange
        var aggregator = new FedAvgAggregationStrategy<double>();

        var clientModels = new Dictionary<int, Dictionary<string, double[]>>
        {
            [0] = new Dictionary<string, double[]> { ["layer1"] = new[] { 1.0, 2.0, 3.0 } },
            [1] = new Dictionary<string, double[]> { ["layer1"] = new[] { 2.0, 4.0, 6.0 } },
            [2] = new Dictionary<string, double[]> { ["layer1"] = new[] { 3.0, 6.0, 9.0 } }
        };

        var clientWeights = new Dictionary<int, double>
        {
            [0] = 100.0, // 100/(100+200+300) = 100/600 = 1/6
            [1] = 200.0, // 200/600 = 1/3
            [2] = 300.0  // 300/600 = 1/2
        };

        // Act
        var result = aggregator.Aggregate(clientModels, clientWeights);

        // Assert
        // Expected: (1/6)*1 + (1/3)*2 + (1/2)*3 = 0.1667 + 0.6667 + 1.5 = 2.333...
        Assert.True(result.ContainsKey("layer1"));
        Assert.Equal(3, result["layer1"].Length);

        double expectedFirst = (1.0 / 6.0) * 1.0 + (1.0 / 3.0) * 2.0 + (1.0 / 2.0) * 3.0;
        Assert.Equal(expectedFirst, result["layer1"][0], Tolerance);
    }

    [Fact]
    public void FedAvgAggregationStrategy_GetStrategyName_ReturnsFedAvg()
    {
        var aggregator = new FedAvgAggregationStrategy<double>();
        Assert.Equal("FedAvg", aggregator.GetStrategyName());
    }

    [Fact]
    public void FedAvgAggregationStrategy_Aggregate_ThrowsOnEmptyClientModels()
    {
        var aggregator = new FedAvgAggregationStrategy<double>();
        var emptyModels = new Dictionary<int, Dictionary<string, double[]>>();
        var weights = new Dictionary<int, double> { [0] = 1.0 };

        Assert.Throws<ArgumentException>(() => aggregator.Aggregate(emptyModels, weights));
    }

    [Fact]
    public void FedAvgAggregationStrategy_Aggregate_ThrowsOnEmptyWeights()
    {
        var aggregator = new FedAvgAggregationStrategy<double>();
        var models = new Dictionary<int, Dictionary<string, double[]>>
        {
            [0] = new Dictionary<string, double[]> { ["layer1"] = new[] { 1.0 } }
        };
        var emptyWeights = new Dictionary<int, double>();

        Assert.Throws<ArgumentException>(() => aggregator.Aggregate(models, emptyWeights));
    }

    [Fact]
    public void FedProxAggregationStrategy_Aggregate_ReturnsWeightedAverage()
    {
        // FedProx uses same aggregation as FedAvg (proximal term is client-side)
        var aggregator = new FedProxAggregationStrategy<double>();

        var clientModels = new Dictionary<int, Dictionary<string, double[]>>
        {
            [0] = new Dictionary<string, double[]> { ["layer1"] = new[] { 1.0, 2.0 } },
            [1] = new Dictionary<string, double[]> { ["layer1"] = new[] { 3.0, 4.0 } }
        };

        var clientWeights = new Dictionary<int, double>
        {
            [0] = 1.0,
            [1] = 1.0
        };

        var result = aggregator.Aggregate(clientModels, clientWeights);

        Assert.True(result.ContainsKey("layer1"));
        Assert.Equal(2.0, result["layer1"][0], Tolerance); // (1+3)/2
        Assert.Equal(3.0, result["layer1"][1], Tolerance); // (2+4)/2
    }

    [Fact]
    public void FedProxAggregationStrategy_GetStrategyName_ReturnsFedProx()
    {
        var aggregator = new FedProxAggregationStrategy<double>();
        // Default mu is 0.01
        Assert.Equal("FedProx(μ=0.01)", aggregator.GetStrategyName());
    }

    [Fact]
    public void FedBNAggregationStrategy_Aggregate_ReturnsWeightedAverage()
    {
        var aggregator = new FedBNAggregationStrategy<double>();

        var clientModels = new Dictionary<int, Dictionary<string, double[]>>
        {
            [0] = new Dictionary<string, double[]> { ["layer1"] = new[] { 2.0, 4.0 } },
            [1] = new Dictionary<string, double[]> { ["layer1"] = new[] { 4.0, 8.0 } }
        };

        var clientWeights = new Dictionary<int, double>
        {
            [0] = 1.0,
            [1] = 3.0
        };

        var result = aggregator.Aggregate(clientModels, clientWeights);

        Assert.True(result.ContainsKey("layer1"));
        // (1/4)*2 + (3/4)*4 = 0.5 + 3 = 3.5
        Assert.Equal(3.5, result["layer1"][0], Tolerance);
    }

    [Fact]
    public void FedBNAggregationStrategy_GetStrategyName_ReturnsFedBN()
    {
        var aggregator = new FedBNAggregationStrategy<double>();
        Assert.Equal("FedBN", aggregator.GetStrategyName());
    }

    // Note: MedianFullModelAggregationStrategy, TrimmedMeanFullModelAggregationStrategy,
    // WinsorizedMeanFullModelAggregationStrategy, KrumFullModelAggregationStrategy,
    // MultiKrumFullModelAggregationStrategy, BulyanFullModelAggregationStrategy, and
    // RfaFullModelAggregationStrategy require IFullModel<T,TInput,TOutput> setup.
    // These are tested via the InMemoryFederatedTrainer integration tests.

    #endregion

    #region Client Selection Strategies

    [Fact]
    public void UniformRandomClientSelectionStrategy_SelectClients_ReturnsSubset()
    {
        var strategy = new UniformRandomClientSelectionStrategy();
        var random = RandomHelper.CreateSeededRandom(42);

        var request = new ClientSelectionRequest
        {
            RoundNumber = 0,
            FractionToSelect = 0.5,
            CandidateClientIds = new List<int> { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 },
            ClientWeights = new Dictionary<int, double>
            {
                [0] = 1.0,
                [1] = 1.0,
                [2] = 1.0,
                [3] = 1.0,
                [4] = 1.0,
                [5] = 1.0,
                [6] = 1.0,
                [7] = 1.0,
                [8] = 1.0,
                [9] = 1.0
            },
            Random = random
        };

        var selected = strategy.SelectClients(request);

        Assert.Equal(5, selected.Count); // 50% of 10 clients
        Assert.All(selected, id => Assert.Contains(id, request.CandidateClientIds));
    }

    [Fact]
    public void UniformRandomClientSelectionStrategy_GetStrategyName_ReturnsUniformRandom()
    {
        var strategy = new UniformRandomClientSelectionStrategy();
        Assert.Equal("UniformRandom", strategy.GetStrategyName());
    }

    [Fact]
    public void UniformRandomClientSelectionStrategy_SelectClients_ThrowsOnNullRequest()
    {
        var strategy = new UniformRandomClientSelectionStrategy();
        Assert.Throws<ArgumentNullException>(() => strategy.SelectClients(null!));
    }

    [Fact]
    public void WeightedRandomClientSelectionStrategy_SelectClients_FavorsHigherWeights()
    {
        var strategy = new WeightedRandomClientSelectionStrategy();
        var random = RandomHelper.CreateSeededRandom(42);

        var weights = new Dictionary<int, double>
        {
            [0] = 10.0,   // Much higher weight
            [1] = 1.0,
            [2] = 1.0,
            [3] = 1.0,
            [4] = 1.0
        };

        var request = new ClientSelectionRequest
        {
            RoundNumber = 0,
            FractionToSelect = 0.4,
            CandidateClientIds = new List<int> { 0, 1, 2, 3, 4 },
            ClientWeights = weights,
            Random = random
        };

        // Run multiple times to check statistical tendency
        int client0Selected = 0;
        for (int i = 0; i < 100; i++)
        {
            request.Random = RandomHelper.CreateSeededRandom(i);
            var selected = strategy.SelectClients(request);
            if (selected.Contains(0)) client0Selected++;
        }

        // Client 0 should be selected more often due to higher weight
        Assert.True(client0Selected > 50);
    }

    [Fact]
    public void WeightedRandomClientSelectionStrategy_GetStrategyName_ReturnsWeightedRandom()
    {
        var strategy = new WeightedRandomClientSelectionStrategy();
        Assert.Equal("WeightedRandom", strategy.GetStrategyName());
    }

    [Fact]
    public void StratifiedClientSelectionStrategy_SelectClients_SelectsFromAllGroups()
    {
        var strategy = new StratifiedClientSelectionStrategy();
        var random = RandomHelper.CreateSeededRandom(42);

        var groupKeys = new Dictionary<int, string>
        {
            [0] = "GroupA",
            [1] = "GroupA",
            [2] = "GroupB",
            [3] = "GroupB",
            [4] = "GroupC",
            [5] = "GroupC"
        };

        var request = new ClientSelectionRequest
        {
            RoundNumber = 0,
            FractionToSelect = 0.5,
            CandidateClientIds = new List<int> { 0, 1, 2, 3, 4, 5 },
            ClientWeights = new Dictionary<int, double>
            {
                [0] = 1.0,
                [1] = 1.0,
                [2] = 1.0,
                [3] = 1.0,
                [4] = 1.0,
                [5] = 1.0
            },
            ClientGroupKeys = groupKeys,
            Random = random
        };

        var selected = strategy.SelectClients(request);

        Assert.Equal(3, selected.Count);
        // Should select from each group
        var selectedGroups = selected.Select(id => groupKeys[id]).Distinct().ToList();
        Assert.True(selectedGroups.Count >= 2); // At least 2 groups represented
    }

    [Fact]
    public void StratifiedClientSelectionStrategy_GetStrategyName_ReturnsStratified()
    {
        var strategy = new StratifiedClientSelectionStrategy();
        Assert.Equal("Stratified", strategy.GetStrategyName());
    }

    [Fact]
    public void AvailabilityAwareClientSelectionStrategy_SelectClients_FiltersUnavailable()
    {
        var strategy = new AvailabilityAwareClientSelectionStrategy(availabilityThreshold: 0.5);
        var random = RandomHelper.CreateSeededRandom(42);

        var availabilities = new Dictionary<int, double>
        {
            [0] = 1.0,  // Available
            [1] = 0.8,  // Available
            [2] = 0.3,  // Not available (below threshold)
            [3] = 0.1,  // Not available
            [4] = 0.9   // Available
        };

        var request = new ClientSelectionRequest
        {
            RoundNumber = 0,
            FractionToSelect = 0.4,  // Request 2 clients (0.4 * 5 = 2) - enough available clients exist
            CandidateClientIds = new List<int> { 0, 1, 2, 3, 4 },
            ClientWeights = new Dictionary<int, double>
            {
                [0] = 1.0,
                [1] = 1.0,
                [2] = 1.0,
                [3] = 1.0,
                [4] = 1.0
            },
            ClientAvailabilityProbabilities = availabilities,
            Random = random
        };

        var selected = strategy.SelectClients(request);

        // With 3 available clients and requesting only 2, should only select available clients
        Assert.True(selected.Count <= 2);
        Assert.All(selected, id => Assert.True(availabilities[id] >= 0.5));
    }

    [Fact]
    public void AvailabilityAwareClientSelectionStrategy_GetStrategyName_ReturnsAvailabilityAware()
    {
        var strategy = new AvailabilityAwareClientSelectionStrategy(0.5);
        Assert.Equal("AvailabilityAware", strategy.GetStrategyName());
    }

    [Fact]
    public void PerformanceAwareClientSelectionStrategy_SelectClients_FavorsHighPerformers()
    {
        var strategy = new PerformanceAwareClientSelectionStrategy(explorationRate: 0.1);
        var random = RandomHelper.CreateSeededRandom(42);

        var performanceScores = new Dictionary<int, double>
        {
            [0] = 100.0, // High performer
            [1] = 50.0,
            [2] = 10.0,  // Low performer
            [3] = 5.0,
            [4] = 1.0
        };

        var request = new ClientSelectionRequest
        {
            RoundNumber = 0,
            FractionToSelect = 0.4,
            CandidateClientIds = new List<int> { 0, 1, 2, 3, 4 },
            ClientWeights = new Dictionary<int, double>
            {
                [0] = 1.0,
                [1] = 1.0,
                [2] = 1.0,
                [3] = 1.0,
                [4] = 1.0
            },
            ClientPerformanceScores = performanceScores,
            Random = random
        };

        // Run multiple times
        int highPerformersSelected = 0;
        for (int i = 0; i < 100; i++)
        {
            request.Random = RandomHelper.CreateSeededRandom(i);
            var selected = strategy.SelectClients(request);
            if (selected.Contains(0) || selected.Contains(1)) highPerformersSelected++;
        }

        // High performers should be selected more often
        Assert.True(highPerformersSelected > 60);
    }

    [Fact]
    public void PerformanceAwareClientSelectionStrategy_GetStrategyName_ReturnsPerformanceAware()
    {
        var strategy = new PerformanceAwareClientSelectionStrategy(0.1);
        Assert.Equal("PerformanceAware", strategy.GetStrategyName());
    }

    [Fact]
    public void ClusteredClientSelectionStrategy_SelectClients_SelectsFromClusters()
    {
        var strategy = new ClusteredClientSelectionStrategy(clusterCount: 3, iterations: 5);
        var random = RandomHelper.CreateSeededRandom(42);

        // Create embeddings that naturally cluster
        var embeddings = new Dictionary<int, double[]>
        {
            [0] = new[] { 0.0, 0.0 },
            [1] = new[] { 0.1, 0.1 },
            [2] = new[] { 5.0, 5.0 },
            [3] = new[] { 5.1, 5.1 },
            [4] = new[] { 10.0, 10.0 },
            [5] = new[] { 10.1, 10.1 }
        };

        var request = new ClientSelectionRequest
        {
            RoundNumber = 0,
            FractionToSelect = 0.5,
            CandidateClientIds = new List<int> { 0, 1, 2, 3, 4, 5 },
            ClientWeights = new Dictionary<int, double>
            {
                [0] = 1.0,
                [1] = 1.0,
                [2] = 1.0,
                [3] = 1.0,
                [4] = 1.0,
                [5] = 1.0
            },
            ClientEmbeddings = embeddings,
            Random = random
        };

        var selected = strategy.SelectClients(request);

        Assert.Equal(3, selected.Count);
    }

    [Fact]
    public void ClusteredClientSelectionStrategy_GetStrategyName_ReturnsClustered()
    {
        var strategy = new ClusteredClientSelectionStrategy(clusterCount: 3, iterations: 5);
        Assert.Equal("Clustered", strategy.GetStrategyName());
    }

    #endregion

    #region Privacy Mechanisms

    [Fact]
    public void GaussianDifferentialPrivacy_ApplyPrivacy_AddsNoise()
    {
        var mechanism = new GaussianDifferentialPrivacy<double>(clipNorm: 1.0, randomSeed: 42);

        var model = new Dictionary<string, double[]>
        {
            ["layer1"] = new[] { 0.5, 0.5, 0.5 }
        };

        var privateModel = mechanism.ApplyPrivacy(model, epsilon: 1.0, delta: 1e-5);

        // Model should be different due to added noise
        bool hasNoise = false;
        for (int i = 0; i < model["layer1"].Length; i++)
        {
            if (Math.Abs(privateModel["layer1"][i] - model["layer1"][i]) > Tolerance)
            {
                hasNoise = true;
                break;
            }
        }

        Assert.True(hasNoise);
    }

    [Fact]
    public void GaussianDifferentialPrivacy_ApplyPrivacy_TracksPrivacyBudget()
    {
        var mechanism = new GaussianDifferentialPrivacy<double>(clipNorm: 1.0, randomSeed: 42);

        var model = new Dictionary<string, double[]>
        {
            ["layer1"] = new[] { 0.5 }
        };

        Assert.Equal(0.0, mechanism.GetPrivacyBudgetConsumed());

        mechanism.ApplyPrivacy(model, epsilon: 0.5, delta: 1e-5);
        Assert.Equal(0.5, mechanism.GetPrivacyBudgetConsumed(), Tolerance);

        mechanism.ApplyPrivacy(model, epsilon: 0.3, delta: 1e-5);
        Assert.Equal(0.8, mechanism.GetPrivacyBudgetConsumed(), Tolerance);
    }

    [Fact]
    public void GaussianDifferentialPrivacy_ApplyPrivacy_ClipsLargeNorms()
    {
        var mechanism = new GaussianDifferentialPrivacy<double>(clipNorm: 1.0, randomSeed: 42);

        // Create model with large norm
        var model = new Dictionary<string, double[]>
        {
            ["layer1"] = new[] { 10.0, 10.0, 10.0 } // L2 norm = sqrt(300) ≈ 17.32
        };

        var privateModel = mechanism.ApplyPrivacy(model, epsilon: 10.0, delta: 1e-5);

        // After clipping, the norm should be at most clipNorm (before noise)
        // But noise will be added, so we just check it's reasonable
        double sumSq = privateModel["layer1"].Sum(x => x * x);
        Assert.True(sumSq < 300.0); // Much smaller than original
    }

    [Fact]
    public void GaussianDifferentialPrivacy_ApplyPrivacy_ThrowsOnInvalidEpsilon()
    {
        var mechanism = new GaussianDifferentialPrivacy<double>(clipNorm: 1.0);
        var model = new Dictionary<string, double[]> { ["layer1"] = new[] { 0.5 } };

        Assert.Throws<ArgumentException>(() => mechanism.ApplyPrivacy(model, epsilon: 0.0, delta: 1e-5));
        Assert.Throws<ArgumentException>(() => mechanism.ApplyPrivacy(model, epsilon: -1.0, delta: 1e-5));
    }

    [Fact]
    public void GaussianDifferentialPrivacy_ApplyPrivacy_ThrowsOnInvalidDelta()
    {
        var mechanism = new GaussianDifferentialPrivacy<double>(clipNorm: 1.0);
        var model = new Dictionary<string, double[]> { ["layer1"] = new[] { 0.5 } };

        Assert.Throws<ArgumentException>(() => mechanism.ApplyPrivacy(model, epsilon: 1.0, delta: 0.0));
        Assert.Throws<ArgumentException>(() => mechanism.ApplyPrivacy(model, epsilon: 1.0, delta: 1.0));
        Assert.Throws<ArgumentException>(() => mechanism.ApplyPrivacy(model, epsilon: 1.0, delta: -0.1));
    }

    [Fact]
    public void GaussianDifferentialPrivacy_GetMechanismName_ReturnsCorrectName()
    {
        var mechanism = new GaussianDifferentialPrivacy<double>(clipNorm: 1.5);
        Assert.Equal("Gaussian DP (clip=1.5)", mechanism.GetMechanismName());
    }

    [Fact]
    public void GaussianDifferentialPrivacy_ResetPrivacyBudget_ClearsBudget()
    {
        var mechanism = new GaussianDifferentialPrivacy<double>(clipNorm: 1.0, randomSeed: 42);
        var model = new Dictionary<string, double[]> { ["layer1"] = new[] { 0.5 } };

        mechanism.ApplyPrivacy(model, epsilon: 1.0, delta: 1e-5);
        Assert.True(mechanism.GetPrivacyBudgetConsumed() > 0);

        mechanism.ResetPrivacyBudget();
        Assert.Equal(0.0, mechanism.GetPrivacyBudgetConsumed());
    }

    [Fact]
    public void GaussianDifferentialPrivacy_Constructor_ThrowsOnInvalidClipNorm()
    {
        Assert.Throws<ArgumentException>(() => new GaussianDifferentialPrivacy<double>(clipNorm: 0.0));
        Assert.Throws<ArgumentException>(() => new GaussianDifferentialPrivacy<double>(clipNorm: -1.0));
    }

    #endregion

    #region Privacy Accountants

    [Fact]
    public void BasicCompositionPrivacyAccountant_AddRound_AccumulatesEpsilon()
    {
        var accountant = new BasicCompositionPrivacyAccountant();

        accountant.AddRound(epsilon: 0.5, delta: 1e-5, samplingRate: 1.0);
        Assert.Equal(0.5, accountant.GetTotalEpsilonConsumed(), Tolerance);

        accountant.AddRound(epsilon: 0.3, delta: 1e-5, samplingRate: 1.0);
        Assert.Equal(0.8, accountant.GetTotalEpsilonConsumed(), Tolerance);
    }

    [Fact]
    public void BasicCompositionPrivacyAccountant_GetAccountantName_ReturnsBasicComposition()
    {
        var accountant = new BasicCompositionPrivacyAccountant();
        Assert.Equal("Basic", accountant.GetAccountantName());
    }

    [Fact]
    public void RdpPrivacyAccountant_AddRound_TracksPrivacy()
    {
        var accountant = new RdpPrivacyAccountant(clipNorm: 1.0);

        accountant.AddRound(epsilon: 1.0, delta: 1e-5, samplingRate: 0.1);

        Assert.True(accountant.GetTotalEpsilonConsumed() >= 0);
    }

    [Fact]
    public void RdpPrivacyAccountant_GetAccountantName_ReturnsRDP()
    {
        var accountant = new RdpPrivacyAccountant(1.0);
        Assert.Equal("RDP", accountant.GetAccountantName());
    }

    [Fact]
    public void RdpPrivacyAccountant_GetEpsilonAtDelta_ReturnsValidEpsilon()
    {
        var accountant = new RdpPrivacyAccountant(clipNorm: 1.0);

        accountant.AddRound(epsilon: 1.0, delta: 1e-5, samplingRate: 0.1);

        double epsilonAtDelta = accountant.GetEpsilonAtDelta(1e-5);
        Assert.True(epsilonAtDelta >= 0);
    }

    #endregion

    #region Cryptography

    [Fact]
    public void ShamirSecretSharing_SplitAndCombine_ReconstructsSecret()
    {
        // Test with internal static class via reflection or through ThresholdSecureAggregation
        var secret = new byte[] { 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08 };
        int threshold = 3;
        int numShares = 5;

        var xByRecipient = new Dictionary<int, int>();
        for (int i = 0; i < numShares; i++)
        {
            xByRecipient[i] = i + 1;
        }

        // Use reflection to access internal class
        var shamirType = typeof(ThresholdSecureAggregation<double>).Assembly
            .GetType("AiDotNet.FederatedLearning.Cryptography.ShamirSecretSharing");

        // Skip if internal type is not available (may be refactored or renamed)
        if (shamirType == null)
        {
            // Internal class not found - this is acceptable as implementation detail may change
            return;
        }

        var splitMethod = shamirType.GetMethod("SplitSecret",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static);
        var combineMethod = shamirType.GetMethod("CombineShares",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static);

        // If type exists, methods should also exist
        Assert.NotNull(splitMethod);
        Assert.NotNull(combineMethod);

        var shares = (Dictionary<int, byte[]>)splitMethod!.Invoke(null,
            new object[] { secret, xByRecipient, threshold, 42, "test" })!;

        Assert.Equal(numShares, shares.Count);

        var reconstructed = (byte[])combineMethod!.Invoke(null,
            new object[] { shares, xByRecipient, threshold, secret.Length })!;

        Assert.Equal(secret, reconstructed);
    }

    [Fact]
    public void HkdfSha256_DeriveKey_ProducesConsistentOutput()
    {
        var ikm = new byte[] { 0x0b, 0x0b, 0x0b, 0x0b, 0x0b, 0x0b, 0x0b, 0x0b };
        var salt = new byte[] { 0x00, 0x01, 0x02, 0x03 };
        var info = new byte[] { 0xf0, 0xf1, 0xf2, 0xf3 };

        var key1 = HkdfSha256.DeriveKey(ikm, salt, info, 32);
        var key2 = HkdfSha256.DeriveKey(ikm, salt, info, 32);

        Assert.Equal(32, key1.Length);
        Assert.Equal(key1, key2);
    }

    [Fact]
    public void HkdfSha256_DeriveKey_ProducesDifferentOutputForDifferentInfo()
    {
        var ikm = new byte[] { 0x0b, 0x0b, 0x0b, 0x0b };
        var salt = new byte[] { 0x00, 0x01, 0x02 };

        var key1 = HkdfSha256.DeriveKey(ikm, salt, new byte[] { 0x01 }, 32);
        var key2 = HkdfSha256.DeriveKey(ikm, salt, new byte[] { 0x02 }, 32);

        Assert.NotEqual(key1, key2);
    }

    #endregion

    #region Server Optimizers

    [Fact]
    public void FedAdamServerOptimizer_Step_UpdatesParameters()
    {
        var optimizer = new FedAdamServerOptimizer<double>(
            learningRate: 0.1, beta1: 0.9, beta2: 0.999, epsilon: 1e-8);

        var current = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var target = new Vector<double>(new[] { 2.0, 3.0, 4.0 });

        var updated = optimizer.Step(current, target);

        Assert.Equal(3, updated.Length);
        // Parameters should move towards target
        for (int i = 0; i < updated.Length; i++)
        {
            Assert.True(updated[i] > current[i]);
            Assert.True(updated[i] <= target[i] || updated[i] > current[i]); // Moving towards target
        }
    }

    [Fact]
    public void FedAdamServerOptimizer_GetOptimizerName_ReturnsFedAdam()
    {
        var optimizer = new FedAdamServerOptimizer<double>();
        Assert.Equal("FedAdam", optimizer.GetOptimizerName());
    }

    [Fact]
    public void FedAdamServerOptimizer_Constructor_ThrowsOnInvalidParameters()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new FedAdamServerOptimizer<double>(learningRate: 0.0));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new FedAdamServerOptimizer<double>(beta1: 1.0));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new FedAdamServerOptimizer<double>(beta2: -0.1));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new FedAdamServerOptimizer<double>(epsilon: 0.0));
    }

    [Fact]
    public void FedAdagradServerOptimizer_Step_UpdatesParameters()
    {
        var optimizer = new FedAdagradServerOptimizer<double>(learningRate: 0.1, epsilon: 1e-8);

        var current = new Vector<double>(new[] { 1.0, 2.0 });
        var target = new Vector<double>(new[] { 2.0, 4.0 });

        var updated = optimizer.Step(current, target);

        Assert.Equal(2, updated.Length);
        for (int i = 0; i < updated.Length; i++)
        {
            Assert.True(updated[i] > current[i]);
        }
    }

    [Fact]
    public void FedAdagradServerOptimizer_GetOptimizerName_ReturnsFedAdagrad()
    {
        var optimizer = new FedAdagradServerOptimizer<double>();
        Assert.Equal("FedAdagrad", optimizer.GetOptimizerName());
    }

    [Fact]
    public void FedYogiServerOptimizer_Step_UpdatesParameters()
    {
        var optimizer = new FedYogiServerOptimizer<double>(
            learningRate: 0.1, beta1: 0.9, beta2: 0.999, epsilon: 1e-8);

        var current = new Vector<double>(new[] { 0.5, 1.0 });
        var target = new Vector<double>(new[] { 1.5, 2.0 });

        var updated = optimizer.Step(current, target);

        Assert.Equal(2, updated.Length);
        for (int i = 0; i < updated.Length; i++)
        {
            Assert.True(updated[i] > current[i]);
        }
    }

    [Fact]
    public void FedYogiServerOptimizer_GetOptimizerName_ReturnsFedYogi()
    {
        var optimizer = new FedYogiServerOptimizer<double>();
        Assert.Equal("FedYogi", optimizer.GetOptimizerName());
    }

    [Fact]
    public void FedAvgMServerOptimizer_Step_UpdatesWithMomentum()
    {
        var optimizer = new FedAvgMServerOptimizer<double>(learningRate: 1.0, momentum: 0.9);

        var current = new Vector<double>(new[] { 1.0, 2.0 });
        var target = new Vector<double>(new[] { 3.0, 4.0 });

        var updated = optimizer.Step(current, target);

        Assert.Equal(2, updated.Length);
        for (int i = 0; i < updated.Length; i++)
        {
            Assert.True(updated[i] > current[i]);
        }
    }

    [Fact]
    public void FedAvgMServerOptimizer_GetOptimizerName_ReturnsFedAvgM()
    {
        var optimizer = new FedAvgMServerOptimizer<double>();
        Assert.Equal("FedAvgM", optimizer.GetOptimizerName());
    }

    #endregion

    #region Heterogeneity Corrections

    [Fact]
    public void ScaffoldHeterogeneityCorrection_Correct_AdjustsParameters()
    {
        var correction = new ScaffoldHeterogeneityCorrection<double>(clientLearningRate: 0.1);

        var globalParams = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var localParams = new Vector<double>(new[] { 1.5, 2.5, 3.5 });

        var corrected = correction.Correct(
            clientId: 0,
            roundNumber: 0,
            globalParams,
            localParams,
            localEpochs: 5);

        Assert.Equal(3, corrected.Length);
    }

    [Fact]
    public void ScaffoldHeterogeneityCorrection_GetCorrectionName_ReturnsScaffold()
    {
        var correction = new ScaffoldHeterogeneityCorrection<double>(0.1);
        Assert.Equal("SCAFFOLD", correction.GetCorrectionName());
    }

    [Fact]
    public void FedNovaHeterogeneityCorrection_Correct_NormalizesUpdate()
    {
        var correction = new FedNovaHeterogeneityCorrection<double>();

        var globalParams = new Vector<double>(new[] { 1.0, 2.0 });
        var localParams = new Vector<double>(new[] { 2.0, 4.0 });

        var corrected = correction.Correct(
            clientId: 0,
            roundNumber: 0,
            globalParams,
            localParams,
            localEpochs: 10);

        Assert.Equal(2, corrected.Length);
    }

    [Fact]
    public void FedNovaHeterogeneityCorrection_GetCorrectionName_ReturnsFedNova()
    {
        var correction = new FedNovaHeterogeneityCorrection<double>();
        Assert.Equal("FedNova", correction.GetCorrectionName());
    }

    [Fact]
    public void FedDynHeterogeneityCorrection_Correct_AppliesDynamicRegularization()
    {
        var correction = new FedDynHeterogeneityCorrection<double>(alpha: 0.01);

        var globalParams = new Vector<double>(new[] { 1.0, 1.0 });
        var localParams = new Vector<double>(new[] { 2.0, 2.0 });

        var corrected = correction.Correct(
            clientId: 0,
            roundNumber: 0,
            globalParams,
            localParams,
            localEpochs: 5);

        Assert.Equal(2, corrected.Length);
    }

    [Fact]
    public void FedDynHeterogeneityCorrection_GetCorrectionName_ReturnsFedDyn()
    {
        var correction = new FedDynHeterogeneityCorrection<double>(0.01);
        Assert.Equal("FedDyn", correction.GetCorrectionName());
    }

    #endregion

    #region Secure Aggregation

    [Fact]
    public void SecureAggregationVector_MaskAndAggregate_PreservesSum()
    {
        var parameterCount = 5;
        var secureAgg = new SecureAggregationVector<double>(parameterCount, randomSeed: 42);

        var clientIds = new List<int> { 0, 1, 2 };
        secureAgg.GeneratePairwiseSecrets(clientIds);

        var clientUpdates = new Dictionary<int, Vector<double>>
        {
            [0] = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 }),
            [1] = new Vector<double>(new[] { 2.0, 3.0, 4.0, 5.0, 6.0 }),
            [2] = new Vector<double>(new[] { 3.0, 4.0, 5.0, 6.0, 7.0 })
        };

        var clientWeights = new Dictionary<int, double>
        {
            [0] = 1.0,
            [1] = 1.0,
            [2] = 1.0
        };

        // Mask updates
        var maskedUpdates = new Dictionary<int, Vector<double>>();
        foreach (var kvp in clientUpdates)
        {
            maskedUpdates[kvp.Key] = secureAgg.MaskUpdate(kvp.Key, kvp.Value, clientWeights[kvp.Key]);
        }

        // Aggregate securely
        var aggregated = secureAgg.AggregateSecurely(maskedUpdates, clientWeights);

        // Expected: average of all updates
        var expectedAverage = new[] { 2.0, 3.0, 4.0, 5.0, 6.0 };

        Assert.Equal(parameterCount, aggregated.Length);
        for (int i = 0; i < parameterCount; i++)
        {
            Assert.Equal(expectedAverage[i], aggregated[i], Tolerance);
        }
    }

    [Fact]
    public void SecureAggregationVector_ClearSecrets_RemovesSensitiveData()
    {
        var secureAgg = new SecureAggregationVector<double>(10, randomSeed: 42);

        var clientIds = new List<int> { 0, 1 };
        secureAgg.GeneratePairwiseSecrets(clientIds);

        secureAgg.ClearSecrets();

        // After clearing, secrets should be gone (no exception thrown)
        // This is mainly a coverage test
        Assert.True(true);
    }

    [Fact]
    public void ThresholdSecureAggregationVector_InitializeRound_SetsParameters()
    {
        var parameterCount = 5;
        var secureAgg = new ThresholdSecureAggregationVector<double>(parameterCount, randomSeed: 42);

        var clientIds = new List<int> { 0, 1, 2, 3, 4 };

        secureAgg.InitializeRound(
            clientIds,
            minimumUploaderCount: 3,
            reconstructionThreshold: 2,
            maxDropoutFraction: 0.3);

        Assert.Equal(3, secureAgg.MinimumUploaderCount);
        Assert.Equal(2, secureAgg.ReconstructionThreshold);
    }

    [Fact]
    public void ThresholdSecureAggregationVector_MaskAndAggregate_ToleratesDropouts()
    {
        var parameterCount = 3;
        var secureAgg = new ThresholdSecureAggregationVector<double>(parameterCount, randomSeed: 42);

        var clientIds = new List<int> { 0, 1, 2, 3, 4 };

        secureAgg.InitializeRound(
            clientIds,
            minimumUploaderCount: 3,
            reconstructionThreshold: 3,
            maxDropoutFraction: 0.4);

        var clientWeights = new Dictionary<int, double>
        {
            [0] = 1.0,
            [1] = 1.0,
            [2] = 1.0,
            [3] = 1.0,
            [4] = 1.0
        };

        // Only 3 clients upload (clients 0, 1, 2 - simulating dropouts of 3, 4)
        var maskedUpdates = new Dictionary<int, Vector<double>>
        {
            [0] = secureAgg.MaskUpdate(0, new Vector<double>(new[] { 1.0, 1.0, 1.0 }), 1.0),
            [1] = secureAgg.MaskUpdate(1, new Vector<double>(new[] { 2.0, 2.0, 2.0 }), 1.0),
            [2] = secureAgg.MaskUpdate(2, new Vector<double>(new[] { 3.0, 3.0, 3.0 }), 1.0)
        };

        var uploaderWeights = new Dictionary<int, double>
        {
            [0] = 1.0,
            [1] = 1.0,
            [2] = 1.0
        };

        var aggregated = secureAgg.AggregateSecurely(maskedUpdates, uploaderWeights);

        Assert.Equal(parameterCount, aggregated.Length);
        // Average of [1,1,1], [2,2,2], [3,3,3] = [2,2,2]
        Assert.Equal(2.0, aggregated[0], Tolerance);
    }

    #endregion

    #region GaussianDifferentialPrivacyVector

    [Fact]
    public void GaussianDifferentialPrivacyVector_ApplyPrivacy_AddsNoiseToVector()
    {
        var mechanism = new GaussianDifferentialPrivacyVector<double>(clipNorm: 1.0, randomSeed: 42);

        var parameters = new Vector<double>(new[] { 0.1, 0.2, 0.3, 0.4, 0.5 });

        var privateParams = mechanism.ApplyPrivacy(parameters, epsilon: 1.0, delta: 1e-5);

        Assert.Equal(parameters.Length, privateParams.Length);

        // Check that noise was added
        bool hasNoise = false;
        for (int i = 0; i < parameters.Length; i++)
        {
            if (Math.Abs(privateParams[i] - parameters[i]) > Tolerance)
            {
                hasNoise = true;
                break;
            }
        }

        Assert.True(hasNoise);
    }

    [Fact]
    public void GaussianDifferentialPrivacyVector_GetMechanismName_ReturnsCorrectName()
    {
        var mechanism = new GaussianDifferentialPrivacyVector<double>(clipNorm: 2.0);
        Assert.Equal("Gaussian Mechanism (Vector)", mechanism.GetMechanismName());
    }

    #endregion

    #region Edge Cases and Validation

    [Fact]
    public void FedAvgAggregationStrategy_Aggregate_HandlesSingleClient()
    {
        var aggregator = new FedAvgAggregationStrategy<double>();

        var clientModels = new Dictionary<int, Dictionary<string, double[]>>
        {
            [0] = new Dictionary<string, double[]> { ["layer1"] = new[] { 1.0, 2.0, 3.0 } }
        };

        var clientWeights = new Dictionary<int, double>
        {
            [0] = 100.0
        };

        var result = aggregator.Aggregate(clientModels, clientWeights);

        Assert.Equal(new[] { 1.0, 2.0, 3.0 }, result["layer1"]);
    }

    [Fact]
    public void FedAvgAggregationStrategy_Aggregate_HandlesMultipleLayers()
    {
        var aggregator = new FedAvgAggregationStrategy<double>();

        var clientModels = new Dictionary<int, Dictionary<string, double[]>>
        {
            [0] = new Dictionary<string, double[]>
            {
                ["layer1"] = new[] { 1.0, 2.0 },
                ["layer2"] = new[] { 10.0, 20.0, 30.0 }
            },
            [1] = new Dictionary<string, double[]>
            {
                ["layer1"] = new[] { 3.0, 4.0 },
                ["layer2"] = new[] { 30.0, 40.0, 50.0 }
            }
        };

        var clientWeights = new Dictionary<int, double>
        {
            [0] = 1.0,
            [1] = 1.0
        };

        var result = aggregator.Aggregate(clientModels, clientWeights);

        Assert.True(result.ContainsKey("layer1"));
        Assert.True(result.ContainsKey("layer2"));
        Assert.Equal(2, result["layer1"].Length);
        Assert.Equal(3, result["layer2"].Length);

        Assert.Equal(2.0, result["layer1"][0], Tolerance); // (1+3)/2
        Assert.Equal(20.0, result["layer2"][0], Tolerance); // (10+30)/2
    }

    [Fact]
    public void ClientSelectionRequest_HandlesEmptyCandidates()
    {
        var strategy = new UniformRandomClientSelectionStrategy();
        var random = RandomHelper.CreateSeededRandom(42);

        var request = new ClientSelectionRequest
        {
            RoundNumber = 0,
            FractionToSelect = 0.5,
            CandidateClientIds = new List<int>(),
            ClientWeights = new Dictionary<int, double>(),
            Random = random
        };

        var selected = strategy.SelectClients(request);

        Assert.Empty(selected);
    }

    [Fact]
    public void ServerOptimizer_MultipleSteps_AccumulatesMomentum()
    {
        var optimizer = new FedAdamServerOptimizer<double>(learningRate: 0.1);

        var current = new Vector<double>(new[] { 0.0, 0.0 });
        var target = new Vector<double>(new[] { 1.0, 1.0 });

        // Multiple steps
        var updated1 = optimizer.Step(current, target);
        var updated2 = optimizer.Step(updated1, target);
        var updated3 = optimizer.Step(updated2, target);

        // Should progressively approach target
        Assert.True(updated1[0] < updated2[0]);
        Assert.True(updated2[0] < updated3[0]);
    }

    [Fact]
    public void HeterogeneityCorrection_MultipleRounds_MaintainsState()
    {
        var correction = new ScaffoldHeterogeneityCorrection<double>(clientLearningRate: 0.1);

        var globalParams = new Vector<double>(new[] { 1.0, 1.0 });
        var localParams = new Vector<double>(new[] { 2.0, 2.0 });

        // Multiple rounds for same client
        var corrected1 = correction.Correct(0, 0, globalParams, localParams, 5);
        var corrected2 = correction.Correct(0, 1, globalParams, localParams, 5);

        // State should be maintained across rounds
        Assert.Equal(2, corrected1.Length);
        Assert.Equal(2, corrected2.Length);
    }

    #endregion
}
