using AiDotNet.Data.Loaders;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Tests.Helpers;
using Xunit;

namespace AiDotNet.Tests.FederatedLearning;

public class FederatedLearningCompressionIntegrationTests
{
    [Theory]
    [InlineData("TopK")]
    [InlineData("RandomK")]
    [InlineData("Threshold")]
    [InlineData("UniformQuantization")]
    [InlineData("StochasticQuantization")]
    public async Task BuildAsync_WithCompressionStrategy_ReportsCompressionMetadata(string strategy)
    {
        var (x, y) = CreateToyData();
        var loader = DataLoaders.FromMatrixVector(x, y);

        var model = new MockFullModel(_ => new Vector<double>(50), parameterCount: 50);
        var initialParameters = new double[50];
        initialParameters[0] = 1.0;
        model.SetParameters(new Vector<double>(initialParameters));

        var optimizer = new FederatedDeterministicDeltaOptimizer(model);

        var flOptions = new FederatedLearningOptions
        {
            NumberOfClients = 6,
            MaxRounds = 2,
            ClientSelectionFraction = 1.0,
            LocalEpochs = 1,
            AggregationStrategy = FederatedAggregationStrategy.FedAvg,
            RandomSeed = 123,
            MinRoundsBeforeConvergence = 100,
            ConvergenceThreshold = 0.0,
            Compression = new FederatedCompressionOptions
            {
                Strategy = strategy,
                Ratio = 0.1,
                UseErrorFeedback = true,
                Threshold = strategy == "Threshold" ? 0.02 : 0.0,
                QuantizationBits = 4
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
        Assert.True(metadata!.CompressionEnabled);
        Assert.Equal(strategy, metadata.CompressionStrategyUsed);
        Assert.All(metadata.RoundMetrics, r => Assert.True(r.UploadCompressionRatio <= 1.0));
    }

    [Fact]
    public async Task BuildAsync_WithUniformQuantizationAndZeroDelta_Completes()
    {
        var (x, y) = CreateToyData();
        var loader = DataLoaders.FromMatrixVector(x, y);

        var model = new MockFullModel(_ => new Vector<double>(10), parameterCount: 10);
        var optimizer = new FederatedNoOpOptimizer(model);

        var flOptions = new FederatedLearningOptions
        {
            NumberOfClients = 3,
            MaxRounds = 1,
            ClientSelectionFraction = 1.0,
            LocalEpochs = 1,
            AggregationStrategy = FederatedAggregationStrategy.FedAvg,
            RandomSeed = 123,
            MinRoundsBeforeConvergence = 100,
            ConvergenceThreshold = 0.0,
            Compression = new FederatedCompressionOptions
            {
                Strategy = "UniformQuantization",
                QuantizationBits = 4,
                UseErrorFeedback = false
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
        Assert.True(metadata!.CompressionEnabled);
        Assert.Equal("UniformQuantization", metadata.CompressionStrategyUsed);
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
