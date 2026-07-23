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
    [InlineData(FederatedAggregationStrategy.Median, 5)]
    [InlineData(FederatedAggregationStrategy.TrimmedMean, 5)]
    [InlineData(FederatedAggregationStrategy.WinsorizedMean, 5)]
    [InlineData(FederatedAggregationStrategy.Rfa, 5)]
    [InlineData(FederatedAggregationStrategy.Krum, 6)]
    [InlineData(FederatedAggregationStrategy.MultiKrum, 7)]
    [InlineData(FederatedAggregationStrategy.Bulyan, 9)]
    public async Task BuildAsync_WithRobustAggregationStrategy_CompletesAndReportsStrategy(FederatedAggregationStrategy aggregationStrategy, int numberOfClients)
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

        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(model)
            .ConfigureOptimizer(optimizer)
            .ConfigureFederatedLearning(flOptions)
            .BuildAsync();

        var metadata = result.GetFederatedLearningMetadata();
        Assert.NotNull(metadata);
        Assert.Equal(aggregationStrategy == FederatedAggregationStrategy.Rfa ? "RFA" : aggregationStrategy.ToString(), metadata!.AggregationStrategyUsed);
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
