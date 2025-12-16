using AiDotNet.Data.Loaders;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Inputs;
using AiDotNet.Models.Options;
using AiDotNet.Models.Results;
using AiDotNet.Regression;
using Xunit;

namespace AiDotNet.Tests.FederatedLearning;

public class FederatedLearningRobustAggregationIntegrationTests
{
    [Theory]
    [InlineData("Median", 5)]
    [InlineData("TrimmedMean", 5)]
    [InlineData("WinsorizedMean", 5)]
    [InlineData("RFA", 5)]
    [InlineData("Krum", 6)]
    [InlineData("MultiKrum", 7)]
    [InlineData("Bulyan", 9)]
    public async Task BuildAsync_WithRobustAggregationStrategy_CompletesAndReportsStrategy(string aggregationStrategy, int numberOfClients)
    {
        var (x, y) = CreateToyData();
        var loader = DataLoaders.FromMatrixVector(x, y);

        var model = new MultipleRegression<double>();
        model.SetParameters(new Vector<double>(new[] { 1.0 }));

        var optimizer = new FederatedNoOpOptimizer(model);

        var flOptions = new FederatedLearningOptions
        {
            NumberOfClients = numberOfClients,
            MaxRounds = 2,
            ClientSelectionFraction = 1.0,
            LocalEpochs = 1,
            AggregationStrategy = aggregationStrategy,
            RobustAggregation = new RobustAggregationOptions
            {
                TrimFraction = 0.2,
                ByzantineClientCount = 1,
                MultiKrumSelectionCount = 0,
                UseClientWeightsWhenAveragingSelectedUpdates = false,
                GeometricMedianMaxIterations = 5
            },
            RandomSeed = 42,
            MinRoundsBeforeConvergence = 1,
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
        Assert.Equal(aggregationStrategy, metadata!.AggregationStrategyUsed);
        Assert.True(metadata.RoundsCompleted >= 1);
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
