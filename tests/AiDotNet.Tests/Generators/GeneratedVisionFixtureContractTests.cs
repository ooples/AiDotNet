using AiDotNet.Generators;
using Xunit;

namespace AiDotNet.Tests.Generators;

public sealed class GeneratedVisionFixtureContractTests
{
    [Fact]
    public void BoundedMedClipArchitecture_DrivesItsFixtureDimensions()
    {
        const string expression = """
            new AiDotNet.VisionLanguage.Encoders.MedCLIP<double>(
                new AiDotNet.NeuralNetworks.NeuralNetworkArchitecture<double>(
                    inputType: AiDotNet.Enums.InputType.ThreeDimensional,
                    taskType: AiDotNet.Enums.NeuralNetworkTaskType.Embedding,
                    inputHeight: 32,
                    inputWidth: 32,
                    inputDepth: 3,
                    outputSize: 16),
                new AiDotNet.VisionLanguage.Encoders.MedCLIPOptions
                {
                    ImageSize = 32,
                    VisionEmbeddingDim = 32,
                    ProjectionDim = 16
                })
            """;

        bool found = GeneratedVisionFixtureContract.TryGetArchitectureSpatialSize(
            expression,
            out int height,
            out int width);

        Assert.True(found);
        Assert.Equal(32, height);
        Assert.Equal(32, width);
    }

    [Fact]
    public void RectangularArchitecture_PreservesNamedDimensionOrder()
    {
        const string expression =
            "new Model(new NeuralNetworkArchitecture<double>(inputWidth: 96, inputHeight: 48))";

        bool found = GeneratedVisionFixtureContract.TryGetArchitectureSpatialSize(
            expression,
            out int height,
            out int width);

        Assert.True(found);
        Assert.Equal(48, height);
        Assert.Equal(96, width);
    }

    [Fact]
    public void UnrelatedNamedArguments_DoNotOverrideTheArchitectureContract()
    {
        const string expression = """
            new Wrapper(
                new OtherOptions(inputHeight: 999, inputWidth: 999),
                new NeuralNetworkArchitecture<double>(inputHeight: 24, inputWidth: 40))
            """;

        bool found = GeneratedVisionFixtureContract.TryGetArchitectureSpatialSize(
            expression,
            out int height,
            out int width);

        Assert.True(found);
        Assert.Equal(24, height);
        Assert.Equal(40, width);
    }

    [Theory]
    [InlineData("")]
    [InlineData("new Model(inputHeight: 32, inputWidth: 32)")]
    [InlineData("new Model(new NeuralNetworkArchitecture<double>(inputHeight: 32))")]
    [InlineData("new Model(new NeuralNetworkArchitecture<double>(inputHeight: size, inputWidth: size))")]
    [InlineData("new Model(new NeuralNetworkArchitecture<double>(inputHeight: 0, inputWidth: 32))")]
    public void MissingOrNonLiteralArchitectureSize_UsesTheFamilyFallback(string expression)
    {
        bool found = GeneratedVisionFixtureContract.TryGetArchitectureSpatialSize(
            expression,
            out _,
            out _);

        Assert.False(found);
    }
}
