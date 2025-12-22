using AiDotNet.Data.Loaders;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Tests.Helpers;
using Xunit;

namespace AiDotNet.Tests.FederatedLearning;

public class FederatedLearningHeterogeneityCorrectionIntegrationTests
{
    [Theory]
    [InlineData(FederatedHeterogeneityCorrection.Scaffold)]
    [InlineData(FederatedHeterogeneityCorrection.FedNova)]
    [InlineData(FederatedHeterogeneityCorrection.FedDyn)]
    public async Task BuildAsync_WithHeterogeneityCorrection_CompletesAndReportsCorrection(FederatedHeterogeneityCorrection algorithm)
    {
        var (x, y) = CreateToyData();
        var loader = DataLoaders.FromMatrixVector(x, y);

        var model = new MockFullModel(_ => new Vector<double>(10), parameterCount: 10);
        var optimizer = new FederatedNoOpOptimizer(model);

        var flOptions = new FederatedLearningOptions
        {
            NumberOfClients = 6,
            MaxRounds = 2,
            ClientSelectionFraction = 1.0,
            LocalEpochs = 2,
            AggregationStrategy = FederatedAggregationStrategy.FedAvg,
            RandomSeed = 7,
            MinRoundsBeforeConvergence = 100,
            ConvergenceThreshold = 0.0,
            HeterogeneityCorrection = new FederatedHeterogeneityCorrectionOptions
            {
                Algorithm = algorithm,
                ClientLearningRate = 1.0,
                FedDynAlpha = 0.01
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
        Assert.Equal(algorithm == FederatedHeterogeneityCorrection.Scaffold ? "SCAFFOLD" : algorithm.ToString(), metadata!.HeterogeneityCorrectionUsed);
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

