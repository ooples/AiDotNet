using AiDotNet.Data.Loaders;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Tests.Helpers;
using Xunit;

namespace AiDotNet.Tests.FederatedLearning;

public class FederatedLearningHomomorphicEncryptionIntegrationTests
{
    [Theory]
    [InlineData(HomomorphicEncryptionScheme.Ckks)]
    [InlineData(HomomorphicEncryptionScheme.Bfv)]
    public async Task BuildAsync_WithHomomorphicEncryption_HEOnly_CompletesAndReportsScheme(HomomorphicEncryptionScheme scheme)
    {
        var (x, y) = CreateToyData();
        var loader = DataLoaders.FromMatrixVector(x, y);

        var model = new MockFullModel(_ => new Vector<double>(8), parameterCount: 8);
        var optimizer = new FederatedDeterministicDeltaOptimizer(model);

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
            HomomorphicEncryption = new HomomorphicEncryptionOptions
            {
                Enabled = true,
                Scheme = scheme,
                Mode = HomomorphicEncryptionMode.HeOnly,
                PolyModulusDegree = 4096
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
        Assert.True(metadata!.HomomorphicEncryptionEnabled);
        Assert.Equal(scheme == HomomorphicEncryptionScheme.Ckks ? "CKKS" : "BFV", metadata.HomomorphicEncryptionSchemeUsed);
        Assert.Equal("HEOnly", metadata.HomomorphicEncryptionModeUsed);
        Assert.Equal("SEAL", metadata.HomomorphicEncryptionProviderUsed);
    }

    [Fact]
    public async Task BuildAsync_WithHomomorphicEncryption_HybridWithSecureAgg_CompletesAndReportsMode()
    {
        var (x, y) = CreateToyData();
        var loader = DataLoaders.FromMatrixVector(x, y);

        var model = new MockFullModel(_ => new Vector<double>(8), parameterCount: 8);
        var optimizer = new FederatedDeterministicDeltaOptimizer(model);

        var flOptions = new FederatedLearningOptions
        {
            NumberOfClients = 3,
            MaxRounds = 1,
            ClientSelectionFraction = 1.0,
            LocalEpochs = 1,
            AggregationStrategy = FederatedAggregationStrategy.FedAvg,
            RandomSeed = 123,
            UseSecureAggregation = true,
            MinRoundsBeforeConvergence = 100,
            ConvergenceThreshold = 0.0,
            HomomorphicEncryption = new HomomorphicEncryptionOptions
            {
                Enabled = true,
                Scheme = HomomorphicEncryptionScheme.Ckks,
                Mode = HomomorphicEncryptionMode.Hybrid,
                PolyModulusDegree = 4096,
                EncryptedRanges = new List<ParameterIndexRange>
                {
                    new ParameterIndexRange { Start = 0, Length = 4 }
                }
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
        Assert.True(metadata!.HomomorphicEncryptionEnabled);
        Assert.Equal("Hybrid", metadata.HomomorphicEncryptionModeUsed);
        Assert.True(metadata.SecureAggregationEnabled);
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
