using AiDotNet.Data.Loaders;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Inputs;
using AiDotNet.Models.Options;
using AiDotNet.Models.Results;
using AiDotNet.Regression;
using Xunit;

namespace AiDotNet.Tests.FederatedLearning;

public class FederatedLearningClientSelectionIntegrationTests
{
    [Theory]
    [InlineData(FederatedClientSelectionStrategy.UniformRandom)]
    [InlineData(FederatedClientSelectionStrategy.WeightedRandom)]
    [InlineData(FederatedClientSelectionStrategy.Stratified)]
    [InlineData(FederatedClientSelectionStrategy.AvailabilityAware)]
    [InlineData(FederatedClientSelectionStrategy.PerformanceAware)]
    [InlineData(FederatedClientSelectionStrategy.Clustered)]
    public async Task BuildAsync_WithClientSelectionStrategy_CompletesAndReportsClients(FederatedClientSelectionStrategy strategy)
    {
        var (x, y) = CreateToyData();
        var loader = DataLoaders.FromMatrixVector(x, y);

        var model = new MultipleRegression<double>();
        model.SetParameters(new Vector<double>(new[] { 1.0 }));

        var optimizer = new FederatedNoOpOptimizer(model);

        var selectionOptions = new ClientSelectionOptions
        {
            Strategy = strategy,
            ExplorationRate = 0.2,
            ClusterCount = 2,
            KMeansIterations = 3,
            AvailabilityThreshold = 0.0,
            ClientGroupKeys = CreateGroupKeysForSixClients(),
            ClientAvailabilityProbabilities = CreateAvailabilityForSixClients()
        };

        var flOptions = new FederatedLearningOptions
        {
            NumberOfClients = 6,
            MaxRounds = 2,
            ClientSelectionFraction = 0.5,
            ClientSelection = selectionOptions,
            LocalEpochs = 1,
            AggregationStrategy = FederatedAggregationStrategy.FedAvg,
            RandomSeed = 123,
            MinRoundsBeforeConvergence = 100,
            ConvergenceThreshold = 0.0
        };

        var result = await new PredictionModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(model)
            .ConfigureOptimizer(optimizer)
            .ConfigureFederatedLearning(flOptions)
            .BuildAsync();

        var metadata = result.GetFederatedLearningMetadata();
        Assert.NotNull(metadata);
        Assert.Equal(2, metadata!.RoundsCompleted);
        Assert.All(metadata.RoundMetrics, r =>
        {
            Assert.True(r.SelectedClientIds.Count >= 1);
            Assert.True(r.SelectedClientIds.SequenceEqual(r.SelectedClientIds.OrderBy(id => id)));
        });
        Assert.True(metadata.TotalCommunicationMB > 0.0);
    }

    private static Dictionary<int, string> CreateGroupKeysForSixClients()
    {
        return new Dictionary<int, string>
        {
            [0] = "A",
            [1] = "A",
            [2] = "A",
            [3] = "B",
            [4] = "B",
            [5] = "B"
        };
    }

    private static Dictionary<int, double> CreateAvailabilityForSixClients()
    {
        return new Dictionary<int, double>
        {
            [0] = 1.0,
            [1] = 0.9,
            [2] = 0.8,
            [3] = 0.7,
            [4] = 0.6,
            [5] = 0.5
        };
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
