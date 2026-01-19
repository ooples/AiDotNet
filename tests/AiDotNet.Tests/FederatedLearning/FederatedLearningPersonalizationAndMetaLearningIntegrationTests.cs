using AiDotNet.Data.Loaders;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Tests.Helpers;
using Xunit;

namespace AiDotNet.Tests.FederatedLearning;

public class FederatedLearningPersonalizationAndMetaLearningIntegrationTests
{
    [Fact]
    public async Task BuildAsync_WithFedPerPersonalization_MasksPersonalizedParametersFromGlobalAggregation()
    {
        var (x, y) = CreateToyData();
        var loader = DataLoaders.FromMatrixVector(x, y);

        var model = new MockFullModel(_ => new Vector<double>(10), parameterCount: 10);
        var initial = model.GetParameters();
        var optimizer = new FederatedDeterministicDeltaOptimizer(model);

        var flOptions = new FederatedLearningOptions
        {
            NumberOfClients = 2,
            MaxRounds = 1,
            ClientSelectionFraction = 1.0,
            LocalEpochs = 1,
            AggregationStrategy = FederatedAggregationStrategy.FedAvg,
            RandomSeed = 7,
            MinRoundsBeforeConvergence = 100,
            ConvergenceThreshold = 0.0,
            Personalization = new FederatedPersonalizationOptions
            {
                Enabled = true,
                Strategy = "FedPer",
                PersonalizedParameterFraction = 0.5,
                LocalAdaptationEpochs = 0
            }
        };

        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(model)
            .ConfigureOptimizer(optimizer)
            .ConfigureFederatedLearning(flOptions)
            .BuildAsync();

        var metadata = result.GetFederatedLearningMetadata();
        Assert.NotNull(metadata);
        Assert.True(metadata!.PersonalizationEnabled);
        Assert.Equal("FedPer", metadata.PersonalizationStrategyUsed);

        var parameters = result.GetParameters();
        Assert.Equal(10, parameters.Length);

        // First half (shared) should update by +0.01, second half (personalized head) should remain at baseline.
        for (int i = 0; i < 5; i++)
        {
            Assert.Equal(initial[i] + 0.01, parameters[i], precision: 6);
        }

        for (int i = 5; i < 10; i++)
        {
            Assert.Equal(initial[i], parameters[i], precision: 6);
        }
    }

    [Fact]
    public async Task BuildAsync_WithReptileMetaLearning_AppliesServerStepSize()
    {
        var (x, y) = CreateToyData();
        var loader = DataLoaders.FromMatrixVector(x, y);

        var model = new MockFullModel(_ => new Vector<double>(10), parameterCount: 10);
        var initial = model.GetParameters();
        var optimizer = new FederatedDeterministicDeltaOptimizer(model);

        var flOptions = new FederatedLearningOptions
        {
            NumberOfClients = 2,
            MaxRounds = 1,
            ClientSelectionFraction = 1.0,
            LocalEpochs = 1,
            AggregationStrategy = FederatedAggregationStrategy.FedAvg,
            RandomSeed = 11,
            MinRoundsBeforeConvergence = 100,
            ConvergenceThreshold = 0.0,
            MetaLearning = new FederatedMetaLearningOptions
            {
                Enabled = true,
                Strategy = "Reptile",
                MetaLearningRate = 0.5,
                InnerEpochs = 1
            }
        };

        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(model)
            .ConfigureOptimizer(optimizer)
            .ConfigureFederatedLearning(flOptions)
            .BuildAsync();

        var metadata = result.GetFederatedLearningMetadata();
        Assert.NotNull(metadata);
        Assert.True(metadata!.MetaLearningEnabled);
        Assert.Equal("Reptile", metadata.MetaLearningStrategyUsed);
        Assert.Equal(0.5, metadata.MetaLearningRateUsed, precision: 6);

        // Clients adapt by +0.01; meta-learning rate 0.5 should move the global by +0.005.
        var parameters = result.GetParameters();
        Assert.Equal(initial[0] + 0.005, parameters[0], precision: 6);
        Assert.Equal(initial[9] + 0.005, parameters[9], precision: 6);
    }

    private static (Matrix<double> x, Vector<double> y) CreateToyData()
    {
        var x = new Matrix<double>(20, 2);
        var y = new Vector<double>(20);
        for (int i = 0; i < 20; i++)
        {
            x[i, 0] = i;
            x[i, 1] = i * 2;
            y[i] = i;
        }

        return (x, y);
    }
}
