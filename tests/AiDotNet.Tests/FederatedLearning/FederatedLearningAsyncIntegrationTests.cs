using AiDotNet.Data.Loaders;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Regression;
using Xunit;

namespace AiDotNet.Tests.FederatedLearning;

public class FederatedLearningAsyncIntegrationTests
{
    [Theory]
    [InlineData(FederatedAsyncMode.FedAsync)]
    [InlineData(FederatedAsyncMode.FedBuff)]
    public async Task BuildAsync_WithAsyncFederatedLearning_CompletesAndReportsAsyncMode(FederatedAsyncMode mode)
    {
        var (x, y) = CreateToyData();
        var loader = DataLoaders.FromMatrixVector(x, y);

        var model = new MultipleRegression<double>();
        model.SetParameters(new Vector<double>(new[] { 1.0 }));

        var optimizer = new FederatedNoOpOptimizer(model);

        var flOptions = new FederatedLearningOptions
        {
            NumberOfClients = 6,
            MaxRounds = 5,
            ClientSelectionFraction = 0.7,
            LocalEpochs = 1,
            AggregationStrategy = FederatedAggregationStrategy.FedAvg,
            RandomSeed = 123,
            MinRoundsBeforeConvergence = 100,
            ConvergenceThreshold = 0.0,
            AsyncFederatedLearning = new AsyncFederatedLearningOptions
            {
                Mode = mode,
                SimulatedMaxClientDelaySteps = 2,
                FedBuffBufferSize = 3,
                FedAsyncMixingRate = 0.5,
                StalenessWeighting = FederatedStalenessWeighting.Inverse,
                StalenessDecayRate = 1.0
            }
        };

        var result = await new PredictionModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(model)
            .ConfigureOptimizer(optimizer)
            .ConfigureFederatedLearning(flOptions)
            .BuildAsync();

        var metadata = result.GetFederatedLearningMetadata();
        Assert.NotNull(metadata);
        Assert.Equal(mode.ToString(), metadata!.AsyncModeUsed);
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
