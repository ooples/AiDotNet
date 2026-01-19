using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Regression;
using Xunit;

namespace AiDotNet.Tests.FederatedLearning;

public class AiModelBuilderFederatedLearningLeafIntegrationTests
{
    private const string TinyLeafJson = @"
{
  ""users"": [""u1"", ""u2""],
  ""num_samples"": [2, 1],
  ""user_data"": {
    ""u1"": {
      ""x"": [
        [[1, 2], [3]],
        [[4, 5], [6]]
      ],
      ""y"": [0, 1]
    },
    ""u2"": {
      ""x"": [
        [7, 8, 9]
      ],
      ""y"": [1]
    }
  }
}";

    [Fact]
    public async Task BuildAsync_WithLeafFederatedDataLoader_UsesEffectiveClientCountForDpSampling()
    {
        string trainPath = Path.Combine(Path.GetTempPath(), $"leaf_train_{Guid.NewGuid():N}.json");
        File.WriteAllText(trainPath, TinyLeafJson);

        try
        {
            var loader = DataLoaders.FromLeafFederatedJsonFiles<double>(trainPath);

            var model = new MultipleRegression<double>();
            model.SetParameters(new Vector<double>(new[] { 1.0 }));

            var optimizer = new FederatedNoOpOptimizer(model);

            var flOptions = new FederatedLearningOptions
            {
                // Intentionally incorrect: should not be used when the data loader provides an explicit client partition.
                // If the trainer used 1 here while selecting 2 clients, DP samplingRate could become > 1.0 and BasicComposition would throw.
                NumberOfClients = 1,
                MaxRounds = 1,
                ClientSelectionFraction = 1.0,
                LocalEpochs = 1,
                AggregationStrategy = FederatedAggregationStrategy.FedAvg,
                RandomSeed = 123,
                MinRoundsBeforeConvergence = 0,
                ConvergenceThreshold = 0.0,
                UseDifferentialPrivacy = true,
                DifferentialPrivacyMode = DifferentialPrivacyMode.Central,
                PrivacyAccountant = FederatedPrivacyAccountant.Basic,
                DifferentialPrivacyClipNorm = 1.0,
                PrivacyEpsilon = 1.0,
                PrivacyDelta = 1e-5
            };

            var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
                .ConfigureDataLoader(loader)
                .ConfigureModel(model)
                .ConfigureOptimizer(optimizer)
                .ConfigureFederatedLearning(flOptions)
                .BuildAsync();

            var metadata = result.GetFederatedLearningMetadata();
            Assert.NotNull(metadata);
            Assert.True(metadata!.DifferentialPrivacyEnabled);
            Assert.Equal("Basic", metadata.PrivacyAccountantUsed);
            Assert.True(metadata.TotalPrivacyBudgetConsumed > 0.0);
            Assert.True(metadata.RoundsCompleted >= 1);
            Assert.Equal(2, metadata.TotalClientsParticipated);
        }
        finally
        {
            if (File.Exists(trainPath))
            {
                File.Delete(trainPath);
            }
        }
    }
}
