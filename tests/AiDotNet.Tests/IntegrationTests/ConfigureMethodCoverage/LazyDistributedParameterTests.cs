using AiDotNet.DistributedTraining;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.ConfigureMethodCoverage;

[Collection("ConfigureMethodCoverage")]
public class LazyDistributedParameterTests : ConfigureMethodTestBase
{
    [Theory]
    [InlineData(DistributedStrategy.DDP)]
    [InlineData(DistributedStrategy.ZeRO1)]
    [InlineData(DistributedStrategy.ZeRO2)]
    [InlineData(DistributedStrategy.ZeRO3)]
    [InlineData(DistributedStrategy.FSDP)]
    public async Task FirstBackwardResolvesParametersBeforeDistributedUpdate(DistributedStrategy strategy)
    {
        using var model = MakeCanaryModel();
        var initialCount = model.GetParameters().Length;
        var (features, labels) = MakeMemorizationSet();
        using var featureOwner = features;
        using var labelOwner = labels;
        var backend = new InMemoryCommunicationBackend<float>(0, 1, Guid.NewGuid().ToString("N"));
        try
        {
            var result = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
                .ConfigureModel(model)
                .ConfigureDataLoader(MakeCanaryLoader(features, labels))
                .ConfigureDistributedTraining(backend, strategy)
                .BuildAsync();
            var sharded = Assert.IsAssignableFrom<IShardedModel<float, Tensor<float>, Tensor<float>>>(result.Model);
            Assert.True(sharded.ParameterCount > initialCount, "The fixture must exercise lazy parameter materialization.");
            using var output = result.Predict(features);
            Assert.Equal(features.Shape[0] * CanaryVocab, output.Length);
            Assert.All(output.ToArray(), value => Assert.True(!float.IsNaN(value) && !float.IsInfinity(value)));
        }
        finally
        {
            backend.Shutdown();
        }
    }
}
