using AiDotNet.FederatedLearning.Fairness;
using AiDotNet.Models.Options;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.FederatedLearning;

/// <summary>
/// Tests for fairness constraints, contribution evaluation, and incentives (#850).
/// </summary>
public class FederatedFairnessAndContributionTests
{
    private static Tensor<double> CreateTensor(params double[] values)
    {
        var tensor = new Tensor<double>(new[] { values.Length });
        for (int i = 0; i < values.Length; i++)
        {
            tensor[i] = values[i];
        }

        return tensor;
    }

    private static Dictionary<int, Tensor<double>> CreateClientModels(int clientCount, int modelSize, double divergenceScale = 0.1)
    {
        var models = new Dictionary<int, Tensor<double>>();
        var rng = new Random(42);

        for (int c = 0; c < clientCount; c++)
        {
            var values = new double[modelSize];
            for (int i = 0; i < modelSize; i++)
            {
                values[i] = 0.5 + (c + 1) * divergenceScale * (rng.NextDouble() - 0.5);
            }

            models[c] = CreateTensor(values);
        }

        return models;
    }

    private static Tensor<double> CreateGlobalModel(int size, double value = 0.5)
    {
        var values = new double[size];
        for (int i = 0; i < size; i++)
        {
            values[i] = value + i * 0.01;
        }

        return CreateTensor(values);
    }

    private static Dictionary<int, List<Tensor<double>>> CreateClientHistories(int clientCount, int rounds, int modelSize)
    {
        var histories = new Dictionary<int, List<Tensor<double>>>();
        var rng = new Random(42);

        for (int c = 0; c < clientCount; c++)
        {
            var roundList = new List<Tensor<double>>();
            for (int r = 0; r < rounds; r++)
            {
                var values = new double[modelSize];
                for (int i = 0; i < modelSize; i++)
                {
                    values[i] = rng.NextDouble();
                }

                roundList.Add(CreateTensor(values));
            }

            histories[c] = roundList;
        }

        return histories;
    }

    // ========== ShapleyValueEvaluator Tests ==========

    [Fact]
    public void ShapleyValue_EvaluatesAllClients()
    {
        var options = new ContributionEvaluationOptions
        {
            Method = ContributionMethod.ShapleyValue,
            SamplingRounds = 50
        };
        var evaluator = new ShapleyValueEvaluator<double>(options);
        var clientModels = CreateClientModels(4, 10);
        var globalModel = CreateGlobalModel(10);
        var histories = CreateClientHistories(4, 3, 10);

        var scores = evaluator.EvaluateContributions(clientModels, globalModel, histories);

        Assert.Equal(4, scores.Count);
        foreach (var kvp in scores)
        {
            Assert.InRange(kvp.Value, 0.0, 1.0);
        }
    }

    [Fact]
    public void ShapleyValue_IdentifiesFreeRiders()
    {
        var options = new ContributionEvaluationOptions
        {
            Method = ContributionMethod.ShapleyValue,
            FreeRiderThreshold = 0.5
        };
        var evaluator = new ShapleyValueEvaluator<double>(options);

        var scores = new Dictionary<int, double>
        {
            { 0, 0.9 },
            { 1, 0.01 },
            { 2, 0.8 },
            { 3, 0.02 }
        };

        var freeRiders = evaluator.IdentifyFreeRiders(scores);

        Assert.Contains(1, freeRiders);
        Assert.Contains(3, freeRiders);
        Assert.DoesNotContain(0, freeRiders);
        Assert.DoesNotContain(2, freeRiders);
    }

    [Fact]
    public void ShapleyValue_MethodName_ReturnsCorrectName()
    {
        var options = new ContributionEvaluationOptions { Method = ContributionMethod.ShapleyValue };
        var evaluator = new ShapleyValueEvaluator<double>(options);

        Assert.Equal("ShapleyValue", evaluator.MethodName);
    }

    [Fact]
    public void ShapleyValue_NullOptions_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => new ShapleyValueEvaluator<double>(null));
    }

    // ========== DataShapleyEvaluator Tests ==========

    [Fact]
    public void DataShapley_EvaluatesAllClients()
    {
        var options = new ContributionEvaluationOptions
        {
            Method = ContributionMethod.DataShapley,
            SamplingRounds = 30
        };
        var evaluator = new DataShapleyEvaluator<double>(options);
        var clientModels = CreateClientModels(5, 8);
        var globalModel = CreateGlobalModel(8);
        var histories = CreateClientHistories(5, 3, 8);

        var scores = evaluator.EvaluateContributions(clientModels, globalModel, histories);

        Assert.Equal(5, scores.Count);
        foreach (var kvp in scores)
        {
            Assert.InRange(kvp.Value, 0.0, 1.0);
        }
    }

    [Fact]
    public void DataShapley_WithPerformanceCache_ScoresConsistent()
    {
        var options = new ContributionEvaluationOptions
        {
            Method = ContributionMethod.DataShapley,
            SamplingRounds = 20,
            UsePerformanceCache = true
        };
        var evaluator = new DataShapleyEvaluator<double>(options);
        var clientModels = CreateClientModels(3, 6);
        var globalModel = CreateGlobalModel(6);
        var histories = CreateClientHistories(3, 3, 6);

        var scores = evaluator.EvaluateContributions(clientModels, globalModel, histories);

        // All scores should be non-negative after normalization
        foreach (var kvp in scores)
        {
            Assert.True(kvp.Value >= 0.0, $"Client {kvp.Key} has negative score: {kvp.Value}");
        }
    }

    [Fact]
    public void DataShapley_NullOptions_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => new DataShapleyEvaluator<double>(null));
    }

    // ========== PrototypicalContributionEvaluator Tests ==========

    [Fact]
    public void Prototypical_EvaluatesAllClients()
    {
        var options = new ContributionEvaluationOptions
        {
            Method = ContributionMethod.Prototypical
        };
        var evaluator = new PrototypicalContributionEvaluator<double>(options);
        var clientModels = CreateClientModels(4, 10, 0.5);
        var globalModel = CreateGlobalModel(10);
        var histories = CreateClientHistories(4, 3, 10);

        var scores = evaluator.EvaluateContributions(clientModels, globalModel, histories);

        Assert.Equal(4, scores.Count);
        foreach (var kvp in scores)
        {
            Assert.InRange(kvp.Value, 0.0, 1.0);
        }
    }

    [Fact]
    public void Prototypical_SingleClient_ReturnsScore()
    {
        var options = new ContributionEvaluationOptions { Method = ContributionMethod.Prototypical };
        var evaluator = new PrototypicalContributionEvaluator<double>(options);
        var clientModels = new Dictionary<int, Tensor<double>>
        {
            { 0, CreateTensor(1.0, 2.0, 3.0) }
        };
        var globalModel = CreateTensor(1.0, 2.0, 3.0);
        var histories = CreateClientHistories(1, 2, 3);

        var scores = evaluator.EvaluateContributions(clientModels, globalModel, histories);

        Assert.Single(scores);
        Assert.True(scores.ContainsKey(0));
    }

    [Fact]
    public void Prototypical_NullOptions_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => new PrototypicalContributionEvaluator<double>(null));
    }

    // ========== GroupFairnessConstraint Tests ==========

    [Fact]
    public void GroupFairness_EvaluatesFairnessScore()
    {
        var options = new FederatedFairnessOptions
        {
            Enabled = true,
            ConstraintType = FairnessConstraintType.DemographicParity,
            NumberOfGroups = 2,
            FairnessThreshold = 0.05
        };
        var constraint = new GroupFairnessConstraint<double>(options);
        var clientModels = CreateClientModels(4, 8);
        var globalModel = CreateGlobalModel(8);
        var clientGroups = new Dictionary<int, int>
        {
            { 0, 0 }, { 1, 0 }, { 2, 1 }, { 3, 1 }
        };

        double fairnessScore = constraint.EvaluateFairness(clientModels, globalModel, clientGroups);

        Assert.InRange(fairnessScore, 0.0, 1.0);
    }

    [Fact]
    public void GroupFairness_AdjustsWeights()
    {
        var options = new FederatedFairnessOptions
        {
            Enabled = true,
            ConstraintType = FairnessConstraintType.MinimaxFairness,
            NumberOfGroups = 2,
            FairnessLambda = 0.1
        };
        var constraint = new GroupFairnessConstraint<double>(options);
        var clientModels = CreateClientModels(4, 8);
        var globalModel = CreateGlobalModel(8);
        var originalWeights = new Dictionary<int, double>
        {
            { 0, 0.25 }, { 1, 0.25 }, { 2, 0.25 }, { 3, 0.25 }
        };
        var clientGroups = new Dictionary<int, int>
        {
            { 0, 0 }, { 1, 0 }, { 2, 1 }, { 3, 1 }
        };

        var adjustedWeights = constraint.AdjustWeights(originalWeights, clientModels, globalModel, clientGroups);

        Assert.Equal(4, adjustedWeights.Count);

        // Weights should sum to approximately 1.0
        double sum = 0;
        foreach (var w in adjustedWeights.Values) sum += w;
        Assert.InRange(sum, 0.99, 1.01);
    }

    [Fact]
    public void GroupFairness_ConstraintName_MatchesType()
    {
        var options = new FederatedFairnessOptions
        {
            ConstraintType = FairnessConstraintType.EqualizedOdds
        };
        var constraint = new GroupFairnessConstraint<double>(options);

        Assert.Equal("EqualizedOdds", constraint.ConstraintName);
    }

    [Fact]
    public void GroupFairness_NullOptions_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => new GroupFairnessConstraint<double>(null));
    }

    // ========== ContributionBasedIncentive Tests ==========

    [Fact]
    public void ContributionIncentive_ComputesRewards()
    {
        var options = new ContributionEvaluationOptions { Method = ContributionMethod.DataShapley };
        var incentive = new ContributionBasedIncentive<double>(options);
        var scores = new Dictionary<int, double>
        {
            { 0, 0.8 }, { 1, 0.5 }, { 2, 0.2 }, { 3, 0.9 }
        };

        var rewards = incentive.ComputeRewards(scores, 100.0);

        Assert.Equal(4, rewards.Count);

        // All rewards should be positive
        foreach (var r in rewards.Values)
        {
            Assert.True(r > 0, "All clients should receive some reward");
        }

        // Total rewards should equal the budget
        double totalReward = 0;
        foreach (var r in rewards.Values) totalReward += r;
        Assert.InRange(totalReward, 99.99, 100.01);

        // Higher contribution should get higher reward
        Assert.True(rewards[3] > rewards[2], "Client with higher contribution should get more reward");
    }

    [Fact]
    public void ContributionIncentive_ComputesTrustScores()
    {
        var options = new ContributionEvaluationOptions { Method = ContributionMethod.DataShapley };
        var incentive = new ContributionBasedIncentive<double>(options);
        var history = new Dictionary<int, List<double>>
        {
            { 0, new List<double> { 0.8, 0.85, 0.82, 0.87 } },
            { 1, new List<double> { 0.2, 0.9, 0.1, 0.95 } },
            { 2, new List<double> { 0.5, 0.5, 0.5, 0.5 } }
        };

        var trustScores = incentive.ComputeTrustScores(history);

        Assert.Equal(3, trustScores.Count);

        // All trust scores should be between 0 and 1
        foreach (var t in trustScores.Values)
        {
            Assert.InRange(t, 0.0, 1.0);
        }

        // Client 2 (consistent) should have higher trust than client 1 (erratic)
        Assert.True(trustScores[2] > trustScores[1],
            "Consistent client should have higher trust than erratic one");
    }

    [Fact]
    public void ContributionIncentive_MechanismName()
    {
        var options = new ContributionEvaluationOptions { Method = ContributionMethod.DataShapley };
        var incentive = new ContributionBasedIncentive<double>(options);
        Assert.Equal("ContributionBased", incentive.MechanismName);
    }

    // ========== Options Tests ==========

    [Fact]
    public void FederatedFairnessOptions_DefaultValues()
    {
        var options = new FederatedFairnessOptions();

        Assert.False(options.Enabled);
        Assert.Equal(FairnessConstraintType.None, options.ConstraintType);
        Assert.Equal(0.05, options.FairnessThreshold);
        Assert.Equal(0.1, options.FairnessLambda);
        Assert.Equal(2, options.NumberOfGroups);
    }

    [Fact]
    public void ContributionEvaluationOptions_DefaultValues()
    {
        var options = new ContributionEvaluationOptions();

        Assert.Equal(ContributionMethod.DataShapley, options.Method);
        Assert.Equal(100, options.SamplingRounds);
        Assert.Equal(5, options.EvaluationFrequency);
        Assert.True(options.UsePerformanceCache);
        Assert.Equal(0.01, options.ConvergenceTolerance);
        Assert.Equal(0.01, options.FreeRiderThreshold);
    }

    [Fact]
    public void FairnessConstraintType_HasAllExpectedValues()
    {
        Assert.True(Enum.IsDefined(typeof(FairnessConstraintType), FairnessConstraintType.None));
        Assert.True(Enum.IsDefined(typeof(FairnessConstraintType), FairnessConstraintType.DemographicParity));
        Assert.True(Enum.IsDefined(typeof(FairnessConstraintType), FairnessConstraintType.EqualizedOdds));
        Assert.True(Enum.IsDefined(typeof(FairnessConstraintType), FairnessConstraintType.EqualOpportunity));
        Assert.True(Enum.IsDefined(typeof(FairnessConstraintType), FairnessConstraintType.MinimaxFairness));
    }

    [Fact]
    public void ContributionMethod_HasAllExpectedValues()
    {
        Assert.True(Enum.IsDefined(typeof(ContributionMethod), ContributionMethod.ShapleyValue));
        Assert.True(Enum.IsDefined(typeof(ContributionMethod), ContributionMethod.DataShapley));
        Assert.True(Enum.IsDefined(typeof(ContributionMethod), ContributionMethod.Prototypical));
    }

    // ========== Integration with FederatedLearningOptions ==========

    [Fact]
    public void FederatedLearningOptions_CanSetFairnessAndContribution()
    {
        var flOptions = new FederatedLearningOptions
        {
            Fairness = new FederatedFairnessOptions
            {
                Enabled = true,
                ConstraintType = FairnessConstraintType.EqualizedOdds
            },
            ContributionEvaluation = new ContributionEvaluationOptions
            {
                Method = ContributionMethod.DataShapley,
                SamplingRounds = 200
            }
        };

        Assert.NotNull(flOptions.Fairness);
        Assert.True(flOptions.Fairness.Enabled);
        Assert.NotNull(flOptions.ContributionEvaluation);
        Assert.Equal(200, flOptions.ContributionEvaluation.SamplingRounds);
    }

    // ========== Metadata Tests ==========

    [Fact]
    public void RoundMetadata_HasContributionScoreFields()
    {
        var metadata = new AiDotNet.Models.RoundMetadata();

        Assert.NotNull(metadata.ClientContributionScores);
        Assert.Empty(metadata.ClientContributionScores);
        Assert.Equal("None", metadata.ContributionMethodUsed);
        Assert.False(metadata.DriftDetected);
        Assert.NotNull(metadata.DriftingClientIds);
        Assert.Empty(metadata.DriftingClientIds);
    }

    [Fact]
    public void FederatedLearningMetadata_HasContributionFields()
    {
        var metadata = new AiDotNet.Models.FederatedLearningMetadata();

        Assert.NotNull(metadata.CumulativeContributionScores);
        Assert.Empty(metadata.CumulativeContributionScores);
        Assert.NotNull(metadata.IdentifiedFreeRiders);
        Assert.Empty(metadata.IdentifiedFreeRiders);
        Assert.False(metadata.FairnessConstraintsEnabled);
        Assert.Equal("None", metadata.FairnessConstraintUsed);
        Assert.Equal(0, metadata.DriftDetectedRounds);
        Assert.Equal("None", metadata.DriftDetectionMethodUsed);
        Assert.False(metadata.UnlearningPerformed);
        Assert.Equal(0, metadata.UnlearningRequestsProcessed);
    }
}
