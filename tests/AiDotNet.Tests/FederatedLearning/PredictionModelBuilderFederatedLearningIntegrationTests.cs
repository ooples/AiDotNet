using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Inputs;
using AiDotNet.Models.Options;
using AiDotNet.Models.Results;
using AiDotNet.Regression;
using AiDotNet.Data.Loaders;
using Xunit;

namespace AiDotNet.Tests.FederatedLearning;

public class PredictionModelBuilderFederatedLearningIntegrationTests
{
    [Fact]
    public async Task BuildAsync_WithFederatedLearning_AddsFederatedLearningMetadata()
    {
        var x = new Matrix<double>(20, 2);
        var y = new Vector<double>(20);
        for (int i = 0; i < 20; i++)
        {
            x[i, 0] = i;
            x[i, 1] = i * 2;
            y[i] = i;
        }

        var model = new MultipleRegression<double>();
        model.SetParameters(new Vector<double>(new[] { 1.0 })); // intercept only (coefficients start empty by default)

        var optimizer = new FederatedNoOpOptimizer(model);

        var flOptions = new FederatedLearningOptions
        {
            NumberOfClients = 4,
            MaxRounds = 3,
            ClientSelectionFraction = 1.0,
            LocalEpochs = 1,
            AggregationStrategy = FederatedAggregationStrategy.FedAvg,
            RandomSeed = 123,
            MinRoundsBeforeConvergence = 1,
            ConvergenceThreshold = 0.0
        };

        var loader = DataLoaders.FromMatrixVector(x, y);

        var result = await new PredictionModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(model)
            .ConfigureOptimizer(optimizer)
            .ConfigureFederatedLearning(flOptions)
            .BuildAsync();

        var metadata = result.GetFederatedLearningMetadata();
        Assert.NotNull(metadata);
        Assert.Equal("FedAvg", metadata!.AggregationStrategyUsed);
        Assert.True(metadata.RoundsCompleted >= 1);
        Assert.True(metadata.Converged);
    }

}
