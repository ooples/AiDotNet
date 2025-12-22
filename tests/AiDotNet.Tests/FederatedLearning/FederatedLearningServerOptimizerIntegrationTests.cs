using AiDotNet.Data.Loaders;
using AiDotNet.Interfaces;
using AiDotNet.Models.Inputs;
using AiDotNet.Models.Options;
using AiDotNet.Models.Results;
using AiDotNet.Regression;
using Xunit;

namespace AiDotNet.Tests.FederatedLearning;

public class FederatedLearningServerOptimizerIntegrationTests
{
    [Theory]
    [InlineData(FederatedServerOptimizer.FedAvgM)]
    [InlineData(FederatedServerOptimizer.FedAdam)]
    [InlineData(FederatedServerOptimizer.FedAdagrad)]
    [InlineData(FederatedServerOptimizer.FedYogi)]
    public async Task BuildAsync_WithServerOptimizer_CompletesAndReportsOptimizer(FederatedServerOptimizer optimizerName)
    {
        var (x, y) = CreateToyData();
        var loader = DataLoaders.FromMatrixVector(x, y);

        var model = new MultipleRegression<double>();
        model.SetParameters(new Vector<double>(new[] { 1.0 }));

        var optimizer = new FederatedNoOpOptimizer(model);

        var flOptions = new FederatedLearningOptions
        {
            NumberOfClients = 4,
            MaxRounds = 2,
            ClientSelectionFraction = 1.0,
            LocalEpochs = 1,
            AggregationStrategy = FederatedAggregationStrategy.FedAvg,
            ServerOptimizer = new FederatedServerOptimizerOptions
            {
                Optimizer = optimizerName,
                LearningRate = 1.0,
                Momentum = 0.9,
                Beta1 = 0.9,
                Beta2 = 0.999,
                Epsilon = 1e-8
            },
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
        Assert.Equal(optimizerName.ToString(), metadata!.ServerOptimizerUsed);
        Assert.Equal(2, metadata.RoundsCompleted);
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
